from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np


# -----------------------------
# Minimal COLMAP binary reader
# -----------------------------
# This is adapted from COLMAP's "read_write_model.py" logic (common utility script).
# It supports cameras.bin + images.bin (points3D not required for transforms.json).

def _read_next_bytes(fid, num_bytes: int, fmt: str):
    import struct
    data = fid.read(num_bytes)
    return struct.unpack(fmt, data)

def _read_string(fid) -> str:
    chars = []
    while True:
        c = fid.read(1)
        if c == b"\x00" or c == b"":
            break
        chars.append(c)
    return b"".join(chars).decode("utf-8", errors="replace")

@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # model-specific

@dataclass
class ColmapImage:
    id: int
    qvec: np.ndarray      # (qw,qx,qy,qz)
    tvec: np.ndarray      # (tx,ty,tz)
    camera_id: int
    name: str

def read_cameras_binary(path: Path) -> Dict[int, ColmapCamera]:
    import struct
    cameras: Dict[int, ColmapCamera] = {}
    with path.open("rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "<Q")[0]
        for _ in range(num_cameras):
            cam_id = _read_next_bytes(fid, 4, "<I")[0]
            model_id = _read_next_bytes(fid, 4, "<i")[0]
            width = _read_next_bytes(fid, 8, "<Q")[0]
            height = _read_next_bytes(fid, 8, "<Q")[0]
            # model id -> name + num params
            # COLMAP models (common): SIMPLE_PINHOLE(3), PINHOLE(4), SIMPLE_RADIAL(4), OPENCV(8), OPENCV_FISHEYE(8) ...
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
            model_name, num_params = model_map.get(model_id, ("UNKNOWN", 0))
            params = np.array(_read_next_bytes(fid, 8 * num_params, "<" + "d" * num_params), dtype=np.float64)
            cameras[cam_id] = ColmapCamera(cam_id, model_name, int(width), int(height), params)
    return cameras

def read_images_binary(path: Path) -> Dict[int, ColmapImage]:
    images: Dict[int, ColmapImage] = {}
    with path.open("rb") as fid:
        num_images = _read_next_bytes(fid, 8, "<Q")[0]
        for _ in range(num_images):
            image_id = _read_next_bytes(fid, 4, "<I")[0]
            qvec = np.array(_read_next_bytes(fid, 8 * 4, "<dddd"), dtype=np.float64)
            tvec = np.array(_read_next_bytes(fid, 8 * 3, "<ddd"), dtype=np.float64)
            camera_id = _read_next_bytes(fid, 4, "<I")[0]
            name = _read_string(fid)
            # skip 2D points
            num_points2d = _read_next_bytes(fid, 8, "<Q")[0]
            fid.seek(num_points2d * (8 * 2 + 8), 1)  # x,y(double,double) + point3D_id(long long)
            images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name)
    return images

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    # COLMAP qvec = [qw,qx,qy,qz]
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)

def colmap_w2c_to_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    # COLMAP stores world-to-camera: X_cam = R * X_world + t
    R = qvec_to_rotmat(qvec)
    t = tvec.reshape(3, 1)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3:4] = -R.T @ t
    return c2w

def intrinsics_from_camera(cam: ColmapCamera) -> Tuple[float, float, float, float]:
    # Return fx, fy, cx, cy in pixel coords
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
    raise ValueError(f"Unsupported camera model for intrinsics export: {cam.model}")

def write_transforms_json(dataset_dir: Path, images_dir: Path, sparse0_dir: Path) -> Path:
    cams = read_cameras_binary(sparse0_dir / "cameras.bin")
    imgs = read_images_binary(sparse0_dir / "images.bin")

    # Nerfstudio expects one set of intrinsics in transforms.json for typical COLMAP pipelines.
    # If multiple cameras exist, this uses the first camera found.
    first_cam = cams[next(iter(cams.keys()))]
    fx, fy, cx, cy = intrinsics_from_camera(first_cam)

    frames: List[Dict[str, Any]] = []
    for _, im in sorted(imgs.items(), key=lambda kv: kv[0]):
        c2w = colmap_w2c_to_c2w(im.qvec, im.tvec)
        frames.append({
            "file_path": str(Path("images") / im.name).replace("\\", "/"),
            "transform_matrix": c2w.tolist(),
        })

    out = {
        "w": first_cam.width,
        "h": first_cam.height,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "frames": frames,
    }

    out_path = dataset_dir / "transforms.json"
    out_path.write_text(json.dumps(out, indent=2))
    return out_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True, help="Folder containing images/ and colmap/sparse/0/")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir
    images_dir = dataset_dir / "images"
    sparse0_dir = dataset_dir / "colmap" / "sparse" / "0"

    if not images_dir.exists():
        raise SystemExit(f"Missing: {images_dir}")
    if not sparse0_dir.exists():
        raise SystemExit(f"Missing: {sparse0_dir}")

    out = write_transforms_json(dataset_dir, images_dir, sparse0_dir)
    print(f"Wrote {out}")
