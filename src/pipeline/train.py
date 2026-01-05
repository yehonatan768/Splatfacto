from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.utils.shell import run_cmd


@dataclass(frozen=True)
class TrainConfig:
    method: str = "splatfacto"
    extra_args: List[str] = None  # appended to ns-train


def train_splatfacto(ns_data_dir: Path, outputs_dir: Path, cfg: TrainConfig, *, logger=None) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    extra = cfg.extra_args or []
    cmd = [
        "ns-train",
        str(cfg.method),
        "--data", str(ns_data_dir),
        "--output-dir", str(outputs_dir),
    ] + list(extra)
    run_cmd(cmd, logger=logger)
