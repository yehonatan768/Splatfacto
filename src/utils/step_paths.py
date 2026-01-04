from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StepPaths:
    frames_dir: Path
    colmap_dir: Path
    ns_data_dir: Path
    splats_dir: Path


def build_step_paths(exp_data: Path, exp_recon: Path, exp_splats: Path, exp: str, ver: str) -> StepPaths:
    return StepPaths(
        frames_dir=exp_data / exp / ver / "frames",
        colmap_dir=exp_recon / exp / ver / "colmap",
        ns_data_dir=exp_recon / exp / ver / "ns_data",
        splats_dir=exp_splats / exp / ver,
    )


def ensure_only(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
