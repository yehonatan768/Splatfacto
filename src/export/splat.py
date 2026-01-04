from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from src.utils.shell import run


def export_splat(config_yml: Path, out_dir: Path, cfg: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = cfg["export"]["splat_filename"]

    run([
        "ns-export", "gaussian-splat",
        "--load-config", str(config_yml),
        "--output-dir", str(out_dir),
        "--output-filename", f,
        "--ply-color-mode", str(cfg["export"]["ply_color_mode"]),
    ])
    return out_dir / f
