from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.shell import run


def _ns_images_help_has(flag: str) -> bool:
    p = subprocess.run(
        ["ns-process-data", "images", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return flag in (p.stdout or "")


def process_images(
    images_dir: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
    logger=None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ns_cfg = cfg.get("ns_process", cfg)
    num_downscales = int(ns_cfg.get("num_downscales", 3))
    matching_method = str(ns_cfg.get("matching_method", "sequential")).strip().lower()

    cmd: List[str] = [
        "ns-process-data", "images",
        "--data", str(images_dir),
        "--output-dir", str(out_dir),
        "--num-downscales", str(num_downscales),
    ]

    # Avoid vocab_tree (FAISS/FLANN mismatch)
    if _ns_images_help_has("--matching-method"):
        cmd += ["--matching-method", matching_method]

    # Windows-safe: force UTF-8 + disable Rich (prevents emoji crashes)
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["RICH_DISABLE"] = "1"

    run(cmd, logger=logger, tool="generic", env=env)
