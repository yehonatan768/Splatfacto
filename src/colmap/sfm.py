from __future__ import annotations

import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.shell import run
from src.utils.log import stage


def _run_capture(cmd: List[str]) -> str:
    """Run a command and capture stdout+stderr as text (never raises)."""
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return p.stdout or ""


def _has_flag(help_text: str, flag: str) -> bool:
    return flag in help_text


def _b01(x: Any) -> str:
    return "1" if bool(x) else "0"


def _add_if_supported(args: List[str], help_text: str, flag: str, value: str) -> None:
    if _has_flag(help_text, flag):
        args.extend([flag, value])


def run_sfm(
    images_dir: Path,
    colmap_dir: Path,
    cfg: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Advanced COLMAP SfM pipeline tuned for drone/aerial video and maximum connectivity.

    Key design goals:
      - Maximize matching graph connectivity (high overlap + loop detection, optionally vocab-tree).
      - Conservative intrinsics refinement to avoid instability (SIMPLE_RADIAL + limited refinement).
      - Robust to different COLMAP builds by probing --help and passing only supported flags.
    """
    if logger is None:
        logger = logging.getLogger("colmap")

    colmap_dir.mkdir(parents=True, exist_ok=True)
    db = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    sparse0 = sparse / "0"
    sparse.mkdir(parents=True, exist_ok=True)

    # Safety: COLMAP will attempt to read everything in the image dir
    for bad in ["meta.json", "extract_meta.json"]:
        p = images_dir / bad
        if p.exists():
            raise RuntimeError(f"Non-image file found in images_dir (remove/move it): {p}")

    # Reset DB
    if db.exists():
        db.unlink()

    # Sections
    log_level = str(int(cfg.get("log_level", 2)))
    c = cfg.get("colmap", {})
    sift = cfg.get("sift", {})
    siftm = cfg.get("sift_matching", {})
    seq = cfg.get("sequential_matcher", {})
    vtm = cfg.get("vocab_tree_matcher", {})
    mapper = cfg.get("mapper", {})

    matching_method = str(cfg.get("matching_method", "sequential")).strip().lower()

    def base_flags() -> List[str]:
        return ["--log_level", log_level]

    # Probe supported flags for THIS COLMAP binary
    fe_help = _run_capture(["colmap", "feature_extractor", "--help"])
    sm_help = _run_capture(["colmap", "sequential_matcher", "--help"])
    vm_help = _run_capture(["colmap", "vocab_tree_matcher", "--help"])
    mp_help = _run_capture(["colmap", "mapper", "--help"])

    # Your COLMAP build rejected SiftExtraction.*.
    # Support both:
    #   - SiftExtraction.* (newer/typical)
    #   - FeatureExtraction.* (your observed working build)
    if _has_flag(fe_help, "--SiftExtraction.use_gpu"):
        extract_prefix = "SiftExtraction"
    elif _has_flag(fe_help, "--FeatureExtraction.use_gpu"):
        extract_prefix = "FeatureExtraction"
    else:
        raise RuntimeError(
            "Could not find a supported extraction GPU flag. Expected one of:\n"
            "  --SiftExtraction.use_gpu\n"
            "  --FeatureExtraction.use_gpu\n"
            "Run: colmap feature_extractor --help"
        )

    # 1) Feature extraction
    with stage(logger, "COLMAP: feature_extractor"):
        extract_args: List[str] = [
            f"--{extract_prefix}.use_gpu", _b01(sift.get("extraction_use_gpu", True)),
        ]
        _add_if_supported(
            extract_args, fe_help,
            f"--{extract_prefix}.gpu_index",
            str(int(c.get("gpu_index", 0))),
        )

        # Rich keypoints for aerial scenes (small texture far away)
        _add_if_supported(
            extract_args, fe_help,
            f"--{extract_prefix}.max_num_features",
            str(int(sift.get("max_num_features", 8192))),
        )
        _add_if_supported(
            extract_args, fe_help,
            f"--{extract_prefix}.peak_threshold",
            str(float(sift.get("peak_threshold", 0.006))),
        )
        _add_if_supported(
            extract_args, fe_help,
            f"--{extract_prefix}.edge_threshold",
            str(int(sift.get("edge_threshold", 10))),
        )

        cmd = [
            "colmap", "feature_extractor",
            *base_flags(),
            "--database_path", str(db),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", str(c.get("camera_model", "SIMPLE_RADIAL")),
            "--ImageReader.single_camera", _b01(c.get("single_camera", False)),
            *extract_args,
        ]
        run(cmd, logger=logger, tool="colmap")

    # 2) Matching (maximize pairs)
    if matching_method == "sequential":
        with stage(logger, "COLMAP: sequential_matcher"):
            cmd: List[str] = [
                "colmap", "sequential_matcher",
                *base_flags(),
                "--database_path", str(db),
                "--SequentialMatching.overlap", str(int(seq.get("overlap", 60))),
                "--SequentialMatching.loop_detection", _b01(seq.get("loop_detection", True)),
            ]

            # These are usually supported (but still probe)
            _add_if_supported(cmd, sm_help, "--SiftMatching.max_ratio", str(float(siftm.get("max_ratio", 0.8))))
            _add_if_supported(cmd, sm_help, "--SiftMatching.cross_check", _b01(siftm.get("cross_check", True)))
            _add_if_supported(cmd, sm_help, "--SiftMatching.guided_matching", _b01(siftm.get("guided_matching", True)))

            # GPU matching is build-dependent; only pass if supported
            if _has_flag(sm_help, "--SiftMatching.use_gpu"):
                cmd += ["--SiftMatching.use_gpu", _b01(sift.get("matching_use_gpu", True))]

            run(cmd, logger=logger, tool="colmap")

    elif matching_method == "vocab_tree":
        # For maximum pair coverage across loops/trajectory revisits.
        # Requires a vocab tree file (download/build once).
        vocab_tree_path = str(vtm.get("vocab_tree_path", "")).strip()
        if not vocab_tree_path:
            raise RuntimeError(
                "matching_method=vocab_tree requires:\n"
                "  vocab_tree_matcher.vocab_tree_path: <path/to/vocab_tree.bin>\n"
                "You can use COLMAP's provided tree or build one with vocab_tree_builder."
            )

        with stage(logger, "COLMAP: vocab_tree_matcher"):
            cmd: List[str] = [
                "colmap", "vocab_tree_matcher",
                *base_flags(),
                "--database_path", str(db),
                "--VocabTreeMatching.vocab_tree_path", vocab_tree_path,
            ]

            # How many candidate images to retrieve per query image (more = more pairs)
            _add_if_supported(
                cmd, vm_help,
                "--VocabTreeMatching.num_images",
                str(int(vtm.get("num_images", 50))),
            )

            # Matching knobs
            _add_if_supported(cmd, vm_help, "--SiftMatching.max_ratio", str(float(siftm.get("max_ratio", 0.8))))
            _add_if_supported(cmd, vm_help, "--SiftMatching.cross_check", _b01(siftm.get("cross_check", True)))
            _add_if_supported(cmd, vm_help, "--SiftMatching.guided_matching", _b01(siftm.get("guided_matching", True)))
            if _has_flag(vm_help, "--SiftMatching.use_gpu"):
                cmd += ["--SiftMatching.use_gpu", _b01(sift.get("matching_use_gpu", True))]

            run(cmd, logger=logger, tool="colmap")

    elif matching_method == "exhaustive":
        # Maximum pairs but O(N^2) — only for small N (e.g., <= 250).
        with stage(logger, "COLMAP: exhaustive_matcher"):
            cmd = [
                "colmap", "exhaustive_matcher",
                *base_flags(),
                "--database_path", str(db),
            ]
            run(cmd, logger=logger, tool="colmap")

    else:
        raise RuntimeError(f"Unsupported matching_method: {matching_method!r}. Use sequential|vocab_tree|exhaustive.")

    # 3) Mapper (register as many images as possible)
    with stage(logger, "COLMAP: mapper"):
        mapper_cmd: List[str] = [
            "colmap", "mapper",
            *base_flags(),
            "--database_path", str(db),
            "--image_path", str(images_dir),
            "--output_path", str(sparse),
        ]

        # Higher registration rate (looser acceptance)
        _add_if_supported(mapper_cmd, mp_help, "--Mapper.min_num_matches", str(int(mapper.get("min_num_matches", 12))))
        _add_if_supported(mapper_cmd, mp_help, "--Mapper.init_min_num_inliers", str(int(mapper.get("init_min_num_inliers", 50))))

        # Conservative intrinsics refinement (avoid unstable fit on aerial)
        _add_if_supported(mapper_cmd, mp_help, "--Mapper.ba_refine_focal_length", _b01(mapper.get("ba_refine_focal_length", True)))
        _add_if_supported(mapper_cmd, mp_help, "--Mapper.ba_refine_principal_point", _b01(mapper.get("ba_refine_principal_point", False)))
        _add_if_supported(mapper_cmd, mp_help, "--Mapper.ba_refine_extra_params", _b01(mapper.get("ba_refine_extra_params", False)))

        # Allow multiple partial models (often increases total registered images)
        _add_if_supported(mapper_cmd, mp_help, "--Mapper.multiple_models", _b01(mapper.get("multiple_models", True)))

        # BA tolerance
        _add_if_supported(
            mapper_cmd, mp_help,
            "--Mapper.ba_global_function_tolerance",
            str(float(mapper.get("ba_global_function_tolerance", 1.0e-6))),
        )

        # GPU index for mapper if supported and requested
        if c.get("use_gpu", True):
            gi = int(c.get("gpu_index", 0))
            if gi >= 0 and _has_flag(mp_help, "--Mapper.gpu_index"):
                mapper_cmd += ["--Mapper.gpu_index", str(gi)]

        run(mapper_cmd, logger=logger, tool="colmap")

    if not sparse0.exists():
        raise RuntimeError(f"Missing sparse model at {sparse0}")

    # 4) Export sparse to PLY (for quick inspection)
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
