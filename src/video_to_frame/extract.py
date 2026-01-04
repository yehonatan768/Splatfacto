from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import logging

from src.utils.imgq import metrics_bgr, dct_phash, hamming
from src.utils.shell import run as run_cmd


def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")
    return p.stdout.strip()


def ffprobe_duration(video: Path) -> float:
    out = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ])
    return float(out)


def _resize_maxw(img: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    s = max_width / float(w)
    return cv2.resize(img, (max_width, int(round(h * s))), interpolation=cv2.INTER_AREA)


def _ffmpeg_extract_candidates(video: Path, out_dir: Path, fps: float, max_width: int, jpg_q: int, logger: logging.Logger) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps},scale='min(iw,{max_width})':-2"

    run_cmd(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "info",
            "-i", str(video),
            "-vf", vf,
            "-q:v", "2",
            "-start_number", "0",
            str(out_dir / "c%06d.jpg"),
        ],
        logger=logger,
        tool="ffmpeg",
        show_progress=True,
    )


def extract_frames(video: Path, out_dir: Path, cfg: Dict[str, Any], logger=None) -> None:
    """
    Advanced drone-friendly extractor for COLMAP + splatfacto:
      - ffmpeg candidate extraction
      - blur filtering (laplacian + tenengrad)
      - exposure consistency (robust + smoothness)
      - near-duplicate removal via DCT pHash
      - viewpoint diversity via motion score (simple pixel diff proxy)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_cfg = cfg["frames"]
    q_cfg = cfg["quality"]
    dedup_cfg = cfg.get("dedup", {})
    motion_cfg = cfg.get("motion", {})
    ff_cfg = cfg.get("ffmpeg", {})
    save_cfg = cfg["save"]

    target = int(frames_cfg["target"])
    max_width = int(frames_cfg["max_width"])
    min_dt = float(frames_cfg["min_dt_sec"])
    cand_mult = float(frames_cfg["candidate_mult"])

    # Blur thresholds
    blur_thr = float(q_cfg["blur_laplacian_thresh"])
    ten_thr = float(q_cfg.get("tenengrad_thresh", 0.0))  # optional; set in yaml
    vmin = float(q_cfg["v_mean_min"])
    vmax = float(q_cfg["v_mean_max"])
    zthr = float(q_cfg["exposure_robust_z"])
    max_exposure_step = float(q_cfg.get("max_exposure_step", 18.0))  # smoothness in V-mean

    # Dedup parameters
    phash_hamming = int(dedup_cfg.get("phash_hamming_max", 10))
    hash_size = int(dedup_cfg.get("hash_size", 8))
    highfreq_factor = int(dedup_cfg.get("highfreq_factor", 4))

    # Motion diversity
    motion_min = float(motion_cfg.get("min_frame_diff", 4.0))  # in mean abs diff (0..255)

    # Saving
    jpg_q = int(save_cfg["jpg_quality"])
    keep_candidates = bool(ff_cfg.get("keep_candidates", False))

    duration = ffprobe_duration(video)

    # Candidate FPS: target*cand_mult spread across duration
    cand_target = int(math.ceil(target * cand_mult))
    # If min_dt is set, it implies an upper bound on candidate FPS:
    fps_from_min_dt = 1.0 / max(min_dt, 1e-6)
    fps_from_target = cand_target / max(duration, 1e-6)
    cand_fps = min(fps_from_min_dt, fps_from_target)
    cand_fps = max(cand_fps, float(ff_cfg.get("min_candidate_fps", 0.5)))
    cand_fps = min(cand_fps, float(ff_cfg.get("max_candidate_fps", 12.0)))

    cand_dir = out_dir / "_candidates"
    if cand_dir.exists():
        shutil.rmtree(cand_dir)
    cand_dir.mkdir(parents=True, exist_ok=True)

    _ffmpeg_extract_candidates(video, cand_dir, cand_fps, max_width, jpg_q, logger)

    cand_files = sorted(cand_dir.glob("c*.jpg"))
    if not cand_files:
        raise RuntimeError("ffmpeg produced no candidate frames. Check ffmpeg installation and input video.")

    # Load candidates and compute metrics
    items = []
    for p in cand_files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = _resize_maxw(img, max_width)
        m = metrics_bgr(img)

        # Absolute exposure gates
        if not (vmin <= m.v_mean <= vmax):
            continue

        # Blur gates
        if m.blur_lap_var < blur_thr:
            continue
        if ten_thr > 0.0 and m.sharp_tenengrad < ten_thr:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ph = dct_phash(gray, hash_size=hash_size, highfreq_factor=highfreq_factor)
        items.append({"path": p, "img": img, "metrics": m, "phash": ph})

    if not items:
        raise RuntimeError("All candidates rejected. Lower blur thresholds or widen exposure limits.")

    # Robust exposure consistency filter (on v_mean)
    vmeans = np.array([it["metrics"].v_mean for it in items], dtype=np.float32)
    med = float(np.median(vmeans))
    mad = float(np.median(np.abs(vmeans - med))) + 1e-6
    sigma = 1.4826 * mad
    items2 = [it for it in items if abs((it["metrics"].v_mean - med) / sigma) <= zthr]
    if not items2:
        items2 = items  # fallback

    # Sort by time/order (candidate index) for smoothness + motion gating
    items2.sort(key=lambda it: it["path"].name)

    selected = []
    last_hash = None
    last_img_gray = None
    last_vmean = None

    for it in items2:
        if len(selected) >= target:
            break

        # Exposure smoothness: avoid jumps (helps COLMAP consistency)
        if last_vmean is not None:
            if abs(it["metrics"].v_mean - last_vmean) > max_exposure_step:
                continue

        # Dedup via pHash
        if last_hash is not None:
            if hamming(it["phash"], last_hash) <= phash_hamming:
                continue

        # Motion diversity: cheap proxy using mean absolute diff in grayscale
        g = cv2.cvtColor(it["img"], cv2.COLOR_BGR2GRAY)
        if last_img_gray is not None:
            diff = float(np.mean(np.abs(g.astype(np.float32) - last_img_gray.astype(np.float32))))
            if diff < motion_min:
                continue

        selected.append(it)
        last_hash = it["phash"]
        last_img_gray = g
        last_vmean = it["metrics"].v_mean

    # Fallback fill: if we under-selected, relax dedup+motion gradually
    if len(selected) < target:
        # Prioritize sharp frames among remaining
        remaining = [it for it in items2 if it not in selected]
        remaining.sort(key=lambda it: (-it["metrics"].blur_lap_var, -it["metrics"].sharp_tenengrad))
        for it in remaining:
            if len(selected) >= target:
                break
            selected.append(it)

    # Save final frames with stable naming for COLMAP/Nerfstudio
    meta = {
        "video": str(video),
        "duration_sec": duration,
        "candidate_fps": cand_fps,
        "candidates_total": len(cand_files),
        "candidates_after_basic": len(items),
        "candidates_after_exposure": len(items2),
        "selected": len(selected),
        "exposure": {"median_vmean": med, "mad": mad, "robust_sigma": sigma},
        "cfg": cfg,
    }
    meta_dir = out_dir / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "extract_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for i, it in enumerate(selected[:target]):
        out_path = out_dir / f"f{i:05d}.jpg"
        cv2.imwrite(str(out_path), it["img"], [int(cv2.IMWRITE_JPEG_QUALITY), jpg_q])

    if not keep_candidates:
        shutil.rmtree(cand_dir, ignore_errors=True)

    if logger is None:
        logger = logging.getLogger("extract")
