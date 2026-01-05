from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from src.utils.shell import run_cmd


@dataclass(frozen=True)
class ExportConfig:
    gaussian_splat: bool = True
    point_cloud: bool = True
    splat_extra_args: List[str] = None
    pointcloud_extra_args: List[str] = None
    run_dir: str = ""  # if empty, auto-detect latest run


def _latest_run_dir(outputs_dir: Path) -> Path:
    # Nerfstudio typically creates: outputs/<method>/<timestamp>/
    # But with --output-dir we control root; still, find the newest config.yml.
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Missing outputs dir: {outputs_dir}")
    candidates = list(outputs_dir.rglob("config.yml")) + list(outputs_dir.rglob("config.yaml"))
    if not candidates:
        raise RuntimeError(f"No Nerfstudio run config found under: {outputs_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].parent


def export_outputs(outputs_dir: Path, exports_dir: Path, cfg: ExportConfig, *, logger=None) -> None:
    exports_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(cfg.run_dir).expanduser() if cfg.run_dir else _latest_run_dir(outputs_dir)
    extra_splat = cfg.splat_extra_args or []
    extra_pc = cfg.pointcloud_extra_args or []

    if cfg.gaussian_splat:
        out = exports_dir / "splat"
        out.mkdir(parents=True, exist_ok=True)
        cmd = ["ns-export", "gaussian-splat", "--load-config", str(run_dir / "config.yml"), "--output-dir", str(out)] + list(extra_splat)
        # Some versions use config.yaml; fallback
        if not (run_dir / "config.yml").exists() and (run_dir / "config.yaml").exists():
            cmd = ["ns-export", "gaussian-splat", "--load-config", str(run_dir / "config.yaml"), "--output-dir", str(out)] + list(extra_splat)
        run_cmd(cmd, logger=logger)

    if cfg.point_cloud:
        out = exports_dir / "pointcloud"
        out.mkdir(parents=True, exist_ok=True)
        cmd = ["ns-export", "pointcloud", "--load-config", str(run_dir / "config.yml"), "--output-dir", str(out)] + list(extra_pc)
        if not (run_dir / "config.yml").exists() and (run_dir / "config.yaml").exists():
            cmd = ["ns-export", "pointcloud", "--load-config", str(run_dir / "config.yaml"), "--output-dir", str(out)] + list(extra_pc)
        run_cmd(cmd, logger=logger)
