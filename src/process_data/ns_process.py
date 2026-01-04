from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.shell import run


def _ns_images_help() -> str:
    p = subprocess.run(
        ["ns-process-data", "images", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return p.stdout or ""


def _ns_has(flag: str) -> bool:
    return flag in _ns_images_help()


def process_images(
    images_dir: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
    logger=None,
) -> None:
    """
    Nerfstudio ns-process-data wrapper tuned for aerial/drone troubleshooting:
      - Default num_downscales=0 (do not throw away small texture)
      - Uses matching_method if the installed CLI supports it
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ns_cfg = cfg.get("ns_process", cfg)
    num_downscales = int(ns_cfg.get("num_downscales", 0))  # IMPORTANT: default 0
    matching_method = str(ns_cfg.get("matching_method", "sequential")).strip().lower()

    cmd: List[str] = [
        "ns-process-data", "images",
        "--data", str(images_dir),
        "--output-dir", str(out_dir),
        "--num-downscales", str(num_downscales),
    ]

    # matching-method is supported in many versions
    if _ns_has("--matching-method"):
        cmd += ["--matching-method", matching_method]

    # Windows-safe: force UTF-8 + disable Rich (prevents emoji/encoding issues)
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["RICH_DISABLE"] = "1"

    run(cmd, logger=logger, tool="generic", env=env)
