from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

from src.utils.shell import run_cmd


@dataclass(frozen=True)
class FfmpegConfig:
    fps: float = 3.0
    scale_max_width: int = 2048
    quality_qv: int = 2
    loglevel: str = "info"


@dataclass(frozen=True)
class SmartSelectConfig:
    enabled: bool = True
    max_frames: int = 600
    min_sharpness: float = 10.0
    keep_every_n_after_filter: int = 1
    min_hist_delta: float = 0.012


def _sharpness_proxy(gray: np.ndarray) -> float:
    # Fast proxy for sharpness without OpenCV:
    # variance of finite differences
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    v = float(gx.var() + gy.var())
    return v


def _hist(gray: np.ndarray, bins: int = 64) -> np.ndarray:
    h, _ = np.histogram(gray, bins=bins, range=(0, 255), density=True)
    return h.astype(np.float32)


def _hist_delta(h1: np.ndarray, h2: np.ndarray) -> float:
    # L1 distance in [0..2] roughly; normalize to [0..1]
    return float(np.abs(h1 - h2).sum() / 2.0)


def extract_frames(video_path: Path, frames_dir: Path, candidates_dir: Path, ff: FfmpegConfig, *, logger=None) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    out_pattern = candidates_dir / "c%06d.jpg"
    vf = f"fps={ff.fps},scale='min(iw,{int(ff.scale_max_width)})':-2"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", ff.loglevel,
        "-i", str(video_path),
        "-vf", vf,
        "-q:v", str(int(ff.quality_qv)),
        "-start_number", "0",
        str(out_pattern),
    ]
    run_cmd(cmd, logger=logger)


def smart_select(candidates_dir: Path, frames_dir: Path, cfg: SmartSelectConfig, *, logger=None) -> None:
    # Collect candidate images
    cands = sorted([p for p in candidates_dir.glob("*.jpg") if p.is_file()])
    if not cands:
        raise RuntimeError(f"No candidate frames found in: {candidates_dir}")

    kept: List[Path] = []
    prev_hist: Optional[np.ndarray] = None
    kept_count = 0

    # First pass: filter by sharpness + near-duplicates
    for p in cands:
        try:
            with Image.open(p) as im:
                im = im.convert("L")
                arr = np.asarray(im, dtype=np.uint8)
        except Exception:
            continue

        sharp = _sharpness_proxy(arr)
        if sharp < cfg.min_sharpness:
            continue

        h = _hist(arr)
        if prev_hist is not None:
            d = _hist_delta(prev_hist, h)
            if d < cfg.min_hist_delta:
                continue

        prev_hist = h
        kept.append(p)

    if not kept:
        raise RuntimeError("Smart selection removed all frames. Lower min_sharpness or min_hist_delta.")

    # Downsample if too many
    if cfg.keep_every_n_after_filter > 1:
        kept = kept[:: int(cfg.keep_every_n_after_filter)]

    if len(kept) > cfg.max_frames:
        # Uniform sampling
        idx = np.linspace(0, len(kept) - 1, cfg.max_frames).round().astype(int)
        kept = [kept[i] for i in idx]

    # Emit final frames as frame_000000.jpg ...
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(kept):
        dst = frames_dir / f"frame_{i:06d}.jpg"
        # Copy is fine; candidates are temporary anyway
        dst.write_bytes(p.read_bytes())

    if logger is not None:
        logger.info("[extract] candidates=%d kept=%d -> %s", len(cands), len(kept), frames_dir)
