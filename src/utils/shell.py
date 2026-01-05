from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str


def run_cmd(cmd: List[str], *, logger=None, env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None) -> RunResult:
    """
    Robust command runner:
      - captures output
      - logs only the command and tail on failure (to avoid log spam)
    """
    if logger is not None:
        logger.info(">> %s", " ".join(cmd))

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=(os.environ | env) if env else None,
        cwd=cwd,
        shell=False,
    )
    out, err = p.stdout or "", p.stderr or ""

    if p.returncode != 0:
        tail = ("\n".join((out + "\n" + err).splitlines()[-60:])).strip()
        if logger is not None:
            logger.error("[cmd] failed rc=%s", p.returncode)
            if tail:
                logger.error("---- tail ----\n%s\n---- end ----", tail)
        raise RuntimeError(f"Command failed (rc={p.returncode}): {' '.join(cmd)}")

    # Keep stdout/stderr available for debugging but do not spam logs
    return RunResult(p.returncode, out, err)
