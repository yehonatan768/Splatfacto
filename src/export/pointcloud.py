from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from src.utils.shell import run


def export_colmap_pointcloud(images_dir: Path, colmap_dir: Path, cfg: Dict[str, Any]) -> None:
    sparse0 = colmap_dir / "sparse" / "0"
    if not sparse0.exists():
        raise RuntimeError(f"Missing sparse model: {sparse0}")

    run([
        "colmap", "model_converter",
        "--input_path", str(sparse0),
        "--output_path", str(colmap_dir / "sparse0.ply"),
        "--output_type", "PLY",
    ])

    if not cfg["pointcloud"]["dense"]:
        return

    dense_dir = colmap_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)

    run([
        "colmap", "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(sparse0),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(cfg["pointcloud"]["max_image_size"]),
    ])

    run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.use_gpu", "1" if cfg["pointcloud"]["use_gpu"] else "0",
    ])

    run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_dir / "dense.ply"),
    ])
