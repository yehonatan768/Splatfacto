from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console: concise
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File: full detail
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


@contextmanager
def stage(logger: logging.Logger, title: str) -> Iterator[None]:
    logger.info(f"[START] {title}")
    try:
        yield
        logger.info(f"[DONE ] {title}")
    except Exception:
        logger.exception(f"[FAIL ] {title}")
        raise
