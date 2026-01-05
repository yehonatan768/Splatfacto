from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LogConfig:
    level: str = "INFO"  # DEBUG|INFO|WARNING|ERROR
    to_file: bool = True
    file_name: str = "run.log"  # inside run_dir


def _level(name: str) -> int:
    name = (name or "INFO").upper().strip()
    return getattr(logging, name, logging.INFO)


def build_logger(name: str, run_dir: Optional[Path] = None, cfg: Optional[LogConfig] = None) -> logging.Logger:
    """
    Advanced-but-clean logger:
      - one line format (timestamp | level | context | message)
      - console + optional file handler
      - no duplicate handlers if called multiple times
    """
    cfg = cfg or LogConfig()
    logger = logging.getLogger(name)
    logger.setLevel(_level(cfg.level))
    logger.propagate = False

    # Avoid duplicate handlers
    if getattr(logger, "_configured", False):
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(_level(cfg.level))
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if cfg.to_file and run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        fh_path = run_dir / cfg.file_name
        fh = logging.FileHandler(fh_path, encoding="utf-8")
        fh.setLevel(_level(cfg.level))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger._configured = True  # type: ignore[attr-defined]
    return logger
