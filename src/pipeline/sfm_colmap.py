from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.shell import run_cmd


@dataclass(frozen=True)
class ColmapConfig:
    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = True
    use_gpu: bool = True
    gpu_index: int = 0
    log_level: int = 1


@dataclass(frozen=True)
class SiftConfig:
    max_image_size: int = 3200
    max_num_features: int = 8192
    peak_threshold: float = 0.006
    edge_threshold: int = 10
    extraction_use_gpu: bool = True


@dataclass(frozen=True)
class MatchingConfig:
    method: str = "sequential"
    use_gpu: bool = True
    gpu_index: int = 0
    guided_matching: bool = True
    max_num_matches: int = 32768
    max_ratio: float = 0.8
    cross_check: bool = True
    overlap: int = 60
    loop_detection: bool = True
    vocab_tree_path: str = ""  # empty => COLMAP default


@dataclass(frozen=True)
class MapperConfig:
    min_num_matches: int = 12
    init_min_num_inliers: int = 50
    multiple_models: bool = True
    ba_refine_focal_length: bool = True
    ba_refine_principal_point: bool = False
    ba_refine_extra_params: bool = False
    ba_global_function_tolerance: float = 1e-6


def _base_flags(log_level: int) -> list[str]:
    return ["--log_level", str(int(log_level))]


def run_sfm(images_dir: Path, colmap_dir: Path, cfg: Dict[str, Any], *, logger=None) -> Path:
    """
    Runs COLMAP SfM (feature_extractor + sequential_matcher + mapper) into:
      colmap_dir/database.db
      colmap_dir/sparse/0
    Returns sparse/0 path.
    """
    colmap_dir.mkdir(parents=True, exist_ok=True)
    db = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    sparse0 = sparse / "0"
    sparse.mkdir(parents=True, exist_ok=True)

    if db.exists():
        db.unlink()

    c = cfg["colmap"]
    sift = cfg["sift"]
    m = cfg["matching"]
    mp = cfg["mapper"]

    # 1) feature_extractor (COLMAP 3.13 flags)
    cmd = [
        "colmap", "feature_extractor",
        *_base_flags(int(c.get("log_level", 1))),
        "--database_path", str(db),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", str(c.get("camera_model", "SIMPLE_RADIAL")),
        "--ImageReader.single_camera", "1" if bool(c.get("single_camera", True)) else "0",
        "--FeatureExtraction.use_gpu", "1" if bool(sift.get("extraction_use_gpu", True)) else "0",
        "--FeatureExtraction.gpu_index", str(int(c.get("gpu_index", -1))),
        "--SiftExtraction.max_image_size", str(int(sift.get("max_image_size", 3200))),
        "--SiftExtraction.max_num_features", str(int(sift.get("max_num_features", 8192))),
        "--SiftExtraction.peak_threshold", str(float(sift.get("peak_threshold", 0.00667))),
        "--SiftExtraction.edge_threshold", str(int(sift.get("edge_threshold", 10))),
    ]
    run_cmd(cmd, logger=logger)

    # 2) sequential_matcher (video)
    seq_cmd = [
        "colmap", "sequential_matcher",
        *_base_flags(int(c.get("log_level", 1))),
        "--database_path", str(db),
        "--FeatureMatching.use_gpu", "1" if bool(m.get("use_gpu", True)) else "0",
        "--FeatureMatching.gpu_index", str(int(m.get("gpu_index", -1))),
        "--FeatureMatching.guided_matching", "1" if bool(m.get("guided_matching", True)) else "0",
        "--FeatureMatching.max_num_matches", str(int(m.get("max_num_matches", 32768))),
        "--SiftMatching.max_ratio", str(float(m.get("max_ratio", 0.8))),
        "--SiftMatching.cross_check", "1" if bool(m.get("cross_check", True)) else "0",
        "--SequentialMatching.overlap", str(int(m.get("overlap", 10))),
        "--SequentialMatching.loop_detection", "1" if bool(m.get("loop_detection", False)) else "0",
    ]
    # Optional: vocab tree path (FAISS-compatible; can be URL per COLMAP 3.13 defaults)
    vt = str(m.get("vocab_tree_path", "")).strip()
    if vt:
        seq_cmd += ["--SequentialMatching.vocab_tree_path", vt]

    run_cmd(seq_cmd, logger=logger)

    # 3) mapper
    mapper_cmd = [
        "colmap", "mapper",
        *_base_flags(int(c.get("log_level", 1))),
        "--database_path", str(db),
        "--image_path", str(images_dir),
        "--output_path", str(sparse),
        "--Mapper.min_num_matches", str(int(mp.get("min_num_matches", 12))),
        "--Mapper.init_min_num_inliers", str(int(mp.get("init_min_num_inliers", 50))),
        "--Mapper.multiple_models", "1" if bool(mp.get("multiple_models", True)) else "0",
        "--Mapper.ba_refine_focal_length", "1" if bool(mp.get("ba_refine_focal_length", True)) else "0",
        "--Mapper.ba_refine_principal_point", "1" if bool(mp.get("ba_refine_principal_point", False)) else "0",
        "--Mapper.ba_refine_extra_params", "1" if bool(mp.get("ba_refine_extra_params", False)) else "0",
        "--Mapper.ba_global_function_tolerance", str(float(mp.get("ba_global_function_tolerance", 1e-6))),
    ]
    if bool(c.get("use_gpu", True)) and int(c.get("gpu_index", -1)) >= 0:
        mapper_cmd += ["--Mapper.gpu_index", str(int(c.get("gpu_index", 0)))]

    run_cmd(mapper_cmd, logger=logger)

    if not sparse0.exists():
        raise RuntimeError(f"COLMAP did not produce sparse model: {sparse0}")

    return sparse0
