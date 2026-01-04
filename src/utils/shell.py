from __future__ import annotations

import logging
import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional


COLMAP_KEYS = [
    "E", "ERROR", "WARNING",
    "Failed", "Could not", "FATAL",
]


FFMPEG_KEYS = [
    "Input #0", "Output #0", "Stream #",
    "frame=", "fps=", "time=", "speed=",
    "Error", "error",
]


def _compile_patterns(keys: Iterable[str]) -> list[re.Pattern]:
    return [re.compile(k, re.IGNORECASE) for k in keys]


def run(
    cmd: List[str],
    *,
    logger: Optional[logging.Logger] = None,
    tool: str = "generic",
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    show_progress: bool = True,
    tail_lines_on_error: int = 200,
) -> None:
    """
    Runs a subprocess while streaming output.

    - Full output -> logger.debug (file)
    - Key output  -> logger.info (console)
    - On failure  -> logger.error tail of output (console)
    """
    if logger is None:
        print("\n>> " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)
        return

    logger.info(f">> {' '.join(cmd)}")

    if tool.lower() == "colmap":
        patterns = _compile_patterns(COLMAP_KEYS)
    elif tool.lower() == "ffmpeg":
        patterns = _compile_patterns(FFMPEG_KEYS)
    else:
        patterns = []

    tail = deque(maxlen=max(50, int(tail_lines_on_error)))

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=str(cwd) if cwd else None,
        env=env,
    )

    last_progress = ""
    assert p.stdout is not None

    for line in p.stdout:
        line = line.rstrip("\n")
        tail.append(line)

        # Full output always goes to file logs
        logger.debug(line)

        # Console filtering
        if patterns:
            if any(pt.search(line) for pt in patterns):
                # Throttle FFmpeg progress spam
                if tool.lower() == "ffmpeg" and ("frame=" in line or "time=" in line):
                    if not show_progress:
                        continue
                    if line == last_progress:
                        continue
                    last_progress = line
                logger.info(line)
        else:
            logger.info(line)

    rc = p.wait()
    if rc != 0:
        logger.error(f"[{tool}] Command failed with exit code {rc}")
        logger.error("---- COLLECTED OUTPUT TAIL ----")
        for l in list(tail)[-tail_lines_on_error:]:
            logger.error(l)
        logger.error("---- END OUTPUT TAIL ----")
        raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")
