from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.shell import run_cmd


# -----------------------------
# COLMAP binary reader (minimal)
# -----------------------------

def _read_next_bytes(fid, num_bytes: int, fmt: str):
    import struct
    data = fid.read(num_bytes)
    return struct.unpack(fmt, data)

def _read_string(fid) -> str:
    chars = []
    while True:
        c = fid.read(1)
        if c in (b"\x00", b""):
            break
        chars.append(c)
    return b"".join(chars).decode("utf-8", errors="replace")

@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

@dataclass
class ColmapImage:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

def read_cameras_bin(path: Path) -> Dict[int, ColmapCamera]:
    import struct
    model_map = {
        0: ("SIMPLE_PINHOLE", 3),
        1: ("PINHOLE", 4),
        2: ("SIMPLE_RADIAL", 4),
        3: ("RADIAL", 5),
        4: ("OPENCV", 8),
        5: ("OPENCV_FISHEYE", 8),
        6: ("FULL_OPENCV", 12),
        7: ("FOV", 5),
        8: ("SIMPLE_RADIAL_FISHEYE", 4),
        9: ("RADIAL_FISHEYE", 5),
        10: ("THIN_PRISM_FISHEYE", 12),
    }
    cams: Dict[int, ColmapCamera] = {}
    with path.open("rb") as fid:
        n = _read_next_bytes(fid, 8, "<Q")[0]
        for _ in range(n):
            cam_id = _read_next_bytes(fid, 4, "<I")[0]
            model_id = _read_next_bytes(fid, 4, "<i")[0]
            width = _read_next_bytes(fid, 8, "<Q")[0]
            height = _read_next_bytes(fid, 8, "<Q")[0]
            model, np_ = model_map.get(model_id, ("UNKNOWN", 0))
            params = np.array(_read_next_bytes(fid, 8*np_, "<" + "d"*np_), dtype=np.float64)
            cams[cam_id] = ColmapCamera(int(cam_id), model, int(width), int(height), params)
    return cams

def read_images_bin(path: Path) -> Dict[int, ColmapImage]:
    imgs: Dict[int, ColmapImage] = {}
    with path.open("rb") as fid:
        n = _read_next_bytes(fid, 8, "<Q")[0]
        for _ in range(n):
            image_id = _read_next_bytes(fid, 4, "<I")[0]
            qvec = np.array(_read_next_bytes(fid, 8*4, "<dddd"), dtype=np.float64)
            tvec = np.array(_read_next_bytes(fid, 8*3, "<ddd"), dtype=np.float64)
            cam_id = _read_next_bytes(fid, 4, "<I")[0]
            name = _read_string(fid)
            # skip points2D
            num_pts2d = _read_next_bytes(fid, 8, "<Q")[0]
            fid.seek(num_pts2d * (8*2 + 8), 1)
            imgs[image_id] = ColmapImage(int(image_id), qvec, tvec, int(cam_id), name)
    return imgs

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)

def colmap_w2c_to_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R = qvec_to_rotmat(qvec)
    t = tvec.reshape(3, 1)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3:4] = -R.T @ t
    return c2w

def intrinsics(cam: ColmapCamera) -> Tuple[float, float, float, float]:
    p = cam.params
    if cam.model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        return float(f), float(f), float(cx), float(cy)
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = p
        return float(fx), float(fy), float(cx), float(cy)
    if cam.model == "SIMPLE_RADIAL":
        f, cx, cy, _k = p
        return float(f), float(f), float(cx), float(cy)
    if cam.model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = p[:4]
        return float(fx), float(fy), float(cx), float(cy)
    raise ValueError(f"Unsupported camera model: {cam.model}")

def opencv_to_opengl(c2w: np.ndarray) -> np.ndarray:
    # Convert from OpenCV camera coords (x right, y down, z forward)
    # to OpenGL (x right, y up, z back)
    # Equivalent: flip Y and Z axes.
    T = np.eye(4, dtype=np.float64)
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    return c2w @ T

def auto_center_scale(frames: List[Dict[str, Any]], target_radius: float) -> None:
    centers = np.array([np.array(f["transform_matrix"], dtype=np.float64)[:3, 3] for f in frames], dtype=np.float64)
    center = centers.mean(axis=0)
    centers0 = centers - center
    rad = float(np.percentile(np.linalg.norm(centers0, axis=1), 90))
    if rad <= 1e-9:
        rad = 1.0
    s = float(target_radius / rad)
    for f in frames:
        M = np.array(f["transform_matrix"], dtype=np.float64)
        M[:3, 3] = (M[:3, 3] - center) * s
        f["transform_matrix"] = M.tolist()

def auto_orient_up(frames: List[Dict[str, Any]]) -> None:
    # Heuristic: align average camera "up" vector to +Z to stabilize aerial datasets.
    ups = []
    for f in frames:
        M = np.array(f["transform_matrix"], dtype=np.float64)
        # camera up in world for OpenGL-ish convention is +Y axis of camera -> column 1
        up = M[:3, 1]
        ups.append(up / (np.linalg.norm(up) + 1e-12))
    v = np.mean(np.stack(ups, axis=0), axis=0)
    v = v / (np.linalg.norm(v) + 1e-12)
    tgt = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Rodrigues rotation from v to tgt
    axis = np.cross(v, tgt)
    s = np.linalg.norm(axis)
    c = float(np.dot(v, tgt))
    if s < 1e-8:
        return
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64)
    R = np.eye(3) + K + K @ K * ((1 - c) / (s**2))

    for f in frames:
        M = np.array(f["transform_matrix"], dtype=np.float64)
        M[:3, :3] = R @ M[:3, :3]
        M[:3, 3] = R @ M[:3, 3]
        f["transform_matrix"] = M.tolist()


@dataclass(frozen=True)
class NsDatasetConfig:
    hardlink_images: bool = True
    auto_orient_up: bool = True
    auto_center: bool = True
    auto_scale: bool = True
    target_radius: float = 1.0
    sort_by_name: bool = True
    opencv_to_opengl: bool = True


def _hardlink_or_copy(src: Path, dst: Path, hardlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    dst.write_bytes(src.read_bytes())


def build_ns_dataset(frames_dir: Path, colmap_sparse0: Path, ns_data_dir: Path, cfg: NsDatasetConfig, *, logger=None) -> Path:
    """
    Converts your COLMAP sparse model into Nerfstudio-compatible dataset:
      ns_data_dir/
        images/
        transforms.json
    """
    cams = read_cameras_bin(colmap_sparse0 / "cameras.bin")
    imgs = read_images_bin(colmap_sparse0 / "images.bin")
    if not imgs:
        raise RuntimeError("COLMAP sparse model contains 0 images.")

    # Copy/link images to ns_data/images
    out_images = ns_data_dir / "images"
    ns_data_dir.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    # Choose order
    items = list(imgs.values())
    if cfg.sort_by_name:
        items.sort(key=lambda im: im.name)

    for im in items:
        src = frames_dir / im.name
        if not src.exists():
            # If your frames are named frame_*.jpg but COLMAP kept those names, this matches.
            raise FileNotFoundError(f"Missing image referenced by COLMAP: {src}")
        _hardlink_or_copy(src, out_images / im.name, hardlink=cfg.hardlink_images)

    # Build frames list
    frames: List[Dict[str, Any]] = []
    for im in items:
        cam = cams[im.camera_id]
        fx, fy, cx, cy = intrinsics(cam)
        c2w = colmap_w2c_to_c2w(im.qvec, im.tvec)
        if cfg.opencv_to_opengl:
            c2w = opencv_to_opengl(c2w)
        frames.append({
            "file_path": str(Path("images") / im.name).replace("\\", "/"),
            "transform_matrix": c2w.tolist(),
            # If multi-camera, we include per-frame intrinsics to be safe.
            "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": cam.width, "h": cam.height,
        })

    # Apply optional normalization/orientation (aerial robustness)
    if cfg.auto_orient_up:
        auto_orient_up(frames)
    if cfg.auto_center or cfg.auto_scale:
        auto_center_scale(frames, target_radius=float(cfg.target_radius))

    # If single camera, promote intrinsics to top-level too (cleaner)
    cam_ids = {im.camera_id for im in items}
    top: Dict[str, Any] = {"frames": frames}
    if len(cam_ids) == 1:
        cam = cams[next(iter(cam_ids))]
        fx, fy, cx, cy = intrinsics(cam)
        top.update({"w": cam.width, "h": cam.height, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy})

        # remove per-frame duplicates
        for f in frames:
            for k in ("fl_x","fl_y","cx","cy","w","h"):
                f.pop(k, None)

    out_path = ns_data_dir / "transforms.json"
    out_path.write_text(json.dumps(top, indent=2), encoding="utf-8")

    if logger is not None:
        logger.info("[ns_dataset] wrote %s (frames=%d)", out_path, len(frames))
    return out_path
