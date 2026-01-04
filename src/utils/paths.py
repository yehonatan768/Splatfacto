from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    videos: Path
    experiments: Path
    exp_data: Path
    exp_recon: Path
    exp_splats: Path


def resolve_paths(cfg_paths: Dict[str, Any]) -> ProjectPaths:
    root = Path(__file__).resolve().parents[2]  # .../project
    videos = root / cfg_paths["project"]["videos_dir"]
    exps = root / cfg_paths["project"]["experiments_dir"]

    data_dir = exps / cfg_paths["experiments"]["data_dir"]
    recon_dir = exps / cfg_paths["experiments"]["recon_dir"]
    splats_dir = exps / cfg_paths["experiments"]["splats_dir"]

    return ProjectPaths(
        root=root,
        videos=videos,
        experiments=exps,
        exp_data=data_dir,
        exp_recon=recon_dir,
        exp_splats=splats_dir,
    )


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
