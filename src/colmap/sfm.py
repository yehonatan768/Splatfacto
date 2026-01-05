from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.shell import run
from src.utils.log import stage


def _run_capture(cmd: List[str]) -> str:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return p.stdout or ""


def _has(help_text: str, flag: str) -> bool:
    return flag in help_text


def _b01(x: Any) -> str:
    return "1" if bool(x) else "0"


def _add(args: List[str], help_text: str, flag: str, value: str) -> None:
    """Append flag/value if flag appears in --help output."""
    if _has(help_text, flag):
        args.extend([flag, value])


def run_sfm(
    images_dir: Path,
    colmap_dir: Path,
    cfg: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Advanced COLMAP SfM for drone/aerial video with maximum pair connectivity.

    Supports:
      - matching_method: sequential | vocab_tree_matcher | exhaustive
      - sequential + loop_detection uses vocab tree internally (recommended default)
      - robust flag wiring based on COLMAP --help (avoids unrecognized options)
    """
    if logger is None:
        logger = logging.getLogger("colmap")

    # Paths
    colmap_dir.mkdir(parents=True, exist_ok=True)
    db = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    sparse0 = sparse / "0"
    sparse.mkdir(parents=True, exist_ok=True)

    # Ensure images_dir contains only images (COLMAP tries to read everything)
    for bad in ["meta.json", "extract_meta.json"]:
        p = images_dir / bad
        if p.exists():
            raise RuntimeError(f"Non-image file found in images_dir (remove/move it): {p}")

    # Reset DB
    if db.exists():
        db.unlink()

    # Config sections
    log_level = str(int(cfg.get("log_level", 2)))
    c = cfg.get("colmap", {})
    sift = cfg.get("sift", {})
    siftm = cfg.get("sift_matching", {})
    twoview = cfg.get("two_view_geometry", {})
    seq = cfg.get("sequential_matcher", {})
    vtm = cfg.get("vocab_tree_matcher", {})
    mapper = cfg.get("mapper", {})

    matching_method = str(cfg.get("matching_method", "sequential")).strip().lower()

    def base_flags() -> List[str]:
        return ["--log_level", log_level]

    # Probe help to ensure option compatibility with your installed COLMAP
    fe_help = _run_capture(["colmap", "feature_extractor", "--help"])
    sm_help = _run_capture(["colmap", "sequential_matcher", "--help"])
    vm_help = _run_capture(["colmap", "vocab_tree_matcher", "--help"])
    ex_help = _run_capture(["colmap", "exhaustive_matcher", "--help"])
    mp_help = _run_capture(["colmap", "mapper", "--help"])

    # ---------------------------------------------------------------------
    # 1) Feature extraction
    # ---------------------------------------------------------------------
    with stage(logger, "COLMAP: feature_extractor"):
        cmd: List[str] = [
            "colmap", "feature_extractor",
            *base_flags(),
            "--database_path", str(db),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", str(c.get("camera_model", "SIMPLE_RADIAL")),
            "--ImageReader.single_camera", _b01(c.get("single_camera", False)),
        ]

        # GPU extraction flags in your COLMAP are FeatureExtraction.*
        _add(cmd, fe_help, "--FeatureExtraction.use_gpu", _b01(sift.get("extraction_use_gpu", True)))
        _add(cmd, fe_help, "--FeatureExtraction.gpu_index", str(int(c.get("gpu_index", 0))))

        # Aerial-friendly SIFT richness knobs (these are SiftExtraction.* in your build)
        _add(cmd, fe_help, "--SiftExtraction.max_num_features", str(int(sift.get("max_num_features", 8192))))
        _add(cmd, fe_help, "--SiftExtraction.peak_threshold", str(float(sift.get("peak_threshold", 0.006))))
        _add(cmd, fe_help, "--SiftExtraction.edge_threshold", str(int(sift.get("edge_threshold", 10))))

        # Optional: limit image size for extraction (leave None for max quality)
        max_img = sift.get("max_image_size", None)
        if max_img is not None:
            _add(cmd, fe_help, "--SiftExtraction.max_image_size", str(int(max_img)))

        run(cmd, logger=logger, tool="colmap")

    # ---------------------------------------------------------------------
    # 2) Matching (maximize pairs)
    # ---------------------------------------------------------------------
    if matching_method == "sequential":
        # Sequential is best for videos; loop_detection adds vocab-tree based loop closure candidates.
        with stage(logger, "COLMAP: sequential_matcher"):
            cmd = [
                "colmap", "sequential_matcher",
                *base_flags(),
                "--database_path", str(db),
                "--SequentialMatching.overlap", str(int(seq.get("overlap", 60))),
                "--SequentialMatching.loop_detection", _b01(seq.get("loop_detection", True)),
            ]

            # Use vocab tree for loop detection (COLMAP can download/cache from the provided URL spec)
            vocab_tree_path = str(seq.get("vocab_tree_path", "")).strip()
            if vocab_tree_path:
                _add(cmd, sm_help, "--SequentialMatching.vocab_tree_path", vocab_tree_path)

            # Increase loop closure aggressiveness (more candidate pairs)
            _add(cmd, sm_help, "--SequentialMatching.loop_detection_period", str(int(seq.get("loop_detection_period", 10))))
            _add(cmd, sm_help, "--SequentialMatching.loop_detection_num_images", str(int(seq.get("loop_detection_num_images", 80))))
            _add(cmd, sm_help, "--SequentialMatching.loop_detection_num_nearest_neighbors", str(int(seq.get("loop_detection_num_nearest_neighbors", 3))))
            _add(cmd, sm_help, "--SequentialMatching.loop_detection_num_checks", str(int(seq.get("loop_detection_num_checks", 128))))

            # GPU matching flags in your COLMAP are FeatureMatching.*
            _add(cmd, sm_help, "--FeatureMatching.use_gpu", _b01(sift.get("matching_use_gpu", True)))
            _add(cmd, sm_help, "--FeatureMatching.gpu_index", str(int(c.get("gpu_index", 0))))
            _add(cmd, sm_help, "--FeatureMatching.guided_matching", _b01(siftm.get("guided_matching", True)))
            # If you want to cap matches per pair:
            _add(cmd, sm_help, "--FeatureMatching.max_num_matches", str(int(siftm.get("max_num_matches", 32768))))

            # SIFT matching heuristics
            _add(cmd, sm_help, "--SiftMatching.max_ratio", str(float(siftm.get("max_ratio", 0.8))))
            _add(cmd, sm_help, "--SiftMatching.max_distance", str(float(siftm.get("max_distance", 0.7))))
            _add(cmd, sm_help, "--SiftMatching.cross_check", _b01(siftm.get("cross_check", True)))

            # Two-view geometry robustness (loosen slightly for aerial)
            _add(cmd, sm_help, "--TwoViewGeometry.min_num_inliers", str(int(twoview.get("min_num_inliers", 15))))
            _add(cmd, sm_help, "--TwoViewGeometry.max_error", str(float(twoview.get("max_error", 6.0))))
            _add(cmd, sm_help, "--TwoViewGeometry.confidence", str(float(twoview.get("confidence", 0.999))))
            _add(cmd, sm_help, "--TwoViewGeometry.max_num_trials", str(int(twoview.get("max_num_trials", 20000))))

            run(cmd, logger=logger, tool="colmap")

    elif matching_method == "vocab_tree_matcher":
        # Global retrieval-based matching. Good when sequential adjacency is weak.
        with stage(logger, "COLMAP: vocab_tree_matcher"):
            cmd = [
                "colmap", "vocab_tree_matcher",
                *base_flags(),
                "--database_path", str(db),
            ]

            vocab_tree_path = str(vtm.get("vocab_tree_path", "")).strip()
            if not vocab_tree_path:
                raise RuntimeError(
                    "matching_method=vocab_tree_matcher requires vocab_tree_matcher.vocab_tree_path.\n"
                    "Tip: you can reuse the SequentialMatching.vocab_tree_path URL-spec from COLMAP help."
                )
            _add(cmd, vm_help, "--VocabTreeMatching.vocab_tree_path", vocab_tree_path)

            # How many candidates retrieved per image (more => more pairs)
            _add(cmd, vm_help, "--VocabTreeMatching.num_images", str(int(vtm.get("num_images", 80))))

            # GPU matching flags may exist as FeatureMatching.* for this command too
            _add(cmd, vm_help, "--FeatureMatching.use_gpu", _b01(sift.get("matching_use_gpu", True)))
            _add(cmd, vm_help, "--FeatureMatching.gpu_index", str(int(c.get("gpu_index", 0))))
            _add(cmd, vm_help, "--FeatureMatching.guided_matching", _b01(siftm.get("guided_matching", True)))
            _add(cmd, vm_help, "--FeatureMatching.max_num_matches", str(int(siftm.get("max_num_matches", 32768))))

            # SiftMatching knobs (if supported by this command)
            _add(cmd, vm_help, "--SiftMatching.max_ratio", str(float(siftm.get("max_ratio", 0.8))))
            _add(cmd, vm_help, "--SiftMatching.max_distance", str(float(siftm.get("max_distance", 0.7))))
            _add(cmd, vm_help, "--SiftMatching.cross_check", _b01(siftm.get("cross_check", True)))

            run(cmd, logger=logger, tool="colmap")

    elif matching_method == "exhaustive":
        # O(N^2) max pairs. Only use for small N.
        with stage(logger, "COLMAP: exhaustive_matcher"):
            cmd = [
                "colmap", "exhaustive_matcher",
                *base_flags(),
                "--database_path", str(db),
            ]

            # exhaustive matcher in recent COLMAP often shares FeatureMatching/SiftMatching options;
            # add them if present.
            _add(cmd, ex_help, "--FeatureMatching.use_gpu", _b01(sift.get("matching_use_gpu", True)))
            _add(cmd, ex_help, "--FeatureMatching.gpu_index", str(int(c.get("gpu_index", 0))))
            _add(cmd, ex_help, "--FeatureMatching.guided_matching", _b01(siftm.get("guided_matching", True)))
            _add(cmd, ex_help, "--SiftMatching.max_ratio", str(float(siftm.get("max_ratio", 0.8))))
            _add(cmd, ex_help, "--SiftMatching.cross_check", _b01(siftm.get("cross_check", True)))

            run(cmd, logger=logger, tool="colmap")

    else:
        raise RuntimeError(
            f"Unsupported matching_method={matching_method!r}. "
            "Use: sequential | vocab_tree_matcher | exhaustive"
        )

    # ---------------------------------------------------------------------
    # 3) Mapper (register as many images as possible)
    # ---------------------------------------------------------------------
    with stage(logger, "COLMAP: mapper"):
        cmd = [
            "colmap", "mapper",
            *base_flags(),
            "--database_path", str(db),
            "--image_path", str(images_dir),
            "--output_path", str(sparse),
        ]

        # Your mapper --help wasn’t pasted, so we add only if supported.
        _add(cmd, mp_help, "--Mapper.min_num_matches", str(int(mapper.get("min_num_matches", 12))))
        _add(cmd, mp_help, "--Mapper.init_min_num_inliers", str(int(mapper.get("init_min_num_inliers", 50))))

        _add(cmd, mp_help, "--Mapper.ba_refine_focal_length", _b01(mapper.get("ba_refine_focal_length", True)))
        _add(cmd, mp_help, "--Mapper.ba_refine_principal_point", _b01(mapper.get("ba_refine_principal_point", False)))
        _add(cmd, mp_help, "--Mapper.ba_refine_extra_params", _b01(mapper.get("ba_refine_extra_params", False)))

        _add(cmd, mp_help, "--Mapper.multiple_models", _b01(mapper.get("multiple_models", True)))
        _add(cmd, mp_help, "--Mapper.ba_global_function_tolerance", str(float(mapper.get("ba_global_function_tolerance", 1e-6))))

        # GPU index for mapper (if supported)
        if c.get("use_gpu", True):
            _add(cmd, mp_help, "--Mapper.gpu_index", str(int(c.get("gpu_index", 0))))

        run(cmd, logger=logger, tool="colmap")

    if not sparse0.exists():
        raise RuntimeError(f"Missing sparse model at {sparse0}")

    # ---------------------------------------------------------------------
    # 4) Export sparse to PLY
    # ---------------------------------------------------------------------
    with stage(logger, "COLMAP: model_converter"):
        run(
            [
                "colmap", "model_converter",
                *base_flags(),
                "--input_path", str(sparse0),
                "--output_path", str(colmap_dir / "sparse0.ply"),
                "--output_type", "PLY",
            ],
            logger=logger,
            tool="colmap",
        )

    return sparse0
