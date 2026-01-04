from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import logging

from src.utils.shell import run
from src.utils.log import stage


def run_sfm(
    images_dir: Path,
    colmap_dir: Path,
    cfg: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Path:
    colmap_dir.mkdir(parents=True, exist_ok=True)
    db = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    sparse0 = sparse / "0"
    sparse.mkdir(parents=True, exist_ok=True)

    # Ensure images_dir contains only images
    for bad in ["meta.json", "extract_meta.json"]:
        p = images_dir / bad
        if p.exists():
            raise RuntimeError(f"Non-image file found in images_dir (remove/move it): {p}")

    if db.exists():
        db.unlink()

    c = cfg["colmap"]
    sift = cfg.get("sift", {})
    siftm = cfg.get("sift_matching", {})
    mapper = cfg.get("mapper", {})
    seq = cfg.get("sequential_matcher", {})

    if logger is None:
        logger = logging.getLogger("colmap")

    def b01(x: bool) -> str:
        return "1" if bool(x) else "0"

    # Allow overriding log level from YAML if you want (default stays 1)
    log_level = str(int(cfg.get("log_level", 1)))

    def base_flags() -> list[str]:
        return ["--log_level", log_level]

    # 1) Feature extraction
    with stage(logger, "COLMAP: feature_extractor"):
        cmd = [
            "colmap", "feature_extractor",
            *base_flags(),
            "--database_path", str(db),
            "--image_path", str(images_dir),

            "--ImageReader.camera_model", str(c["camera_model"]),
            "--ImageReader.single_camera", b01(c.get("single_camera", False)),

            # IMPORTANT: use SiftExtraction.* (standard COLMAP flags)
            "--SiftExtraction.use_gpu", b01(sift.get("extraction_use_gpu", True)),
            "--SiftExtraction.gpu_index", str(int(c.get("gpu_index", 0))),

            "--SiftExtraction.max_num_features", str(int(sift.get("max_num_features", 8192))),
            "--SiftExtraction.peak_threshold", str(float(sift.get("peak_threshold", 0.006))),
            "--SiftExtraction.edge_threshold", str(int(sift.get("edge_threshold", 10))),
        ]
        run(cmd, logger=logger, tool="colmap")

    # 2) Sequential matcher
    with stage(logger, "COLMAP: sequential_matcher"):
        cmd = [
            "colmap", "sequential_matcher",
            *base_flags(),
            "--database_path", str(db),

            "--SequentialMatching.overlap", str(int(seq.get("overlap", 40))),
            "--SequentialMatching.loop_detection", b01(seq.get("loop_detection", True)),

            # Matching quality knobs (these are typically supported)
            "--SiftMatching.max_ratio", str(float(siftm.get("max_ratio", 0.8))),
            "--SiftMatching.cross_check", b01(siftm.get("cross_check", True)),
            "--SiftMatching.guided_matching", b01(siftm.get("guided_matching", True)),
        ]

        # GPU matching flag is build-dependent. If your build fails on it, keep it disabled.
        # If you want to try it, uncomment:
        # cmd += ["--SiftMatching.use_gpu", b01(sift.get("matching_use_gpu", True))]

        run(cmd, logger=logger, tool="colmap")

    # 3) Mapper
    with stage(logger, "COLMAP: mapper"):
        mapper_cmd = [
            "colmap", "mapper",
            *base_flags(),
            "--database_path", str(db),
            "--image_path", str(images_dir),
            "--output_path", str(sparse),

            "--Mapper.ba_global_function_tolerance",
            str(float(mapper.get("ba_global_function_tolerance", 1.0e-6))),

            "--Mapper.min_num_matches",
            str(int(mapper.get("min_num_matches", 12))),

            "--Mapper.init_min_num_inliers",
            str(int(mapper.get("init_min_num_inliers", 50))),

            "--Mapper.ba_refine_focal_length",
            b01(mapper.get("ba_refine_focal_length", True)),

            "--Mapper.ba_refine_principal_point",
            b01(mapper.get("ba_refine_principal_point", False)),

            "--Mapper.ba_refine_extra_params",
            b01(mapper.get("ba_refine_extra_params", False)),

            "--Mapper.multiple_models",
            b01(mapper.get("multiple_models", True)),
        ]

        # COLMAP has gpu_index for some steps; safe to pass when >=0
        if c.get("use_gpu", True) and int(c.get("gpu_index", 0)) >= 0:
            mapper_cmd += ["--Mapper.gpu_index", str(int(c.get("gpu_index", 0)))]

        run(mapper_cmd, logger=logger, tool="colmap")

    if not sparse0.exists():
        raise RuntimeError(f"Missing sparse model at {sparse0}")

    # 4) Export sparse to PLY
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
