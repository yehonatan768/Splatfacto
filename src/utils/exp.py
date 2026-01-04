from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re


@dataclass(frozen=True)
class ExpRef:
    exp_name: str          # video-based slug, e.g., dji_0004_barn
    version: str           # e.g., v0002


@dataclass(frozen=True)
class ExpDirs:
    frames_dir: Path
    colmap_dir: Path
    ns_data_dir: Path
    splats_dir: Path


def _fmt(prefix: str, num: int, digits: int) -> str:
    return f"{prefix}{num:0{digits}d}"


def slugify_video_name(video_path_or_name: str) -> str:
    """
    Convert a video filename/path into a stable experiment name.

    Examples:
      'DJI_0004.MP4' -> 'dji_0004'
      'Barn Flight (Take 2).mp4' -> 'barn_flight_take_2'
      'C:\\videos\\My Shot.MOV' -> 'my_shot'
    """
    name = Path(video_path_or_name).name
    stem = Path(name).stem
    s = stem.lower().strip()

    # Replace whitespace and separators with underscore
    s = re.sub(r"[ \t\-]+", "_", s)

    # Remove any remaining unsafe characters
    s = re.sub(r"[^a-z0-9_]+", "", s)

    # Collapse multiple underscores and trim
    s = re.sub(r"_+", "_", s).strip("_")

    if not s:
        s = "video"
    return s


def list_existing(experiments_root: Path, ver_prefix: str) -> List[ExpRef]:
    """
    Canonical index: experiments/splats/<exp_name>/<version>/
    exp_name is now video_slug (free-form but sanitized), version is v####.
    """
    out: List[ExpRef] = []
    if not experiments_root.exists():
        return out

    ver_re = re.compile(rf"^{re.escape(ver_prefix)}\d+$")

    for exp_dir in sorted(experiments_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        # exp_dir.name can be any slug; accept it as long as it has version children.
        for ver_dir in sorted(exp_dir.iterdir()):
            if ver_dir.is_dir() and ver_re.match(ver_dir.name):
                out.append(ExpRef(exp_dir.name, ver_dir.name))

    return out


def next_version_name(existing: List[ExpRef], exp_name: str, ver_prefix: str, digits: int) -> str:
    mx = 0
    for e in existing:
        if e.exp_name == exp_name and e.version.startswith(ver_prefix):
            try:
                n = int(e.version.replace(ver_prefix, ""))
                mx = max(mx, n)
            except ValueError:
                continue
    return _fmt(ver_prefix, mx + 1, digits)


def make_dirs(exp_data: Path, exp_recon: Path, exp_splats: Path, exp_name: str, version: str) -> ExpDirs:
    frames_dir = exp_data / exp_name / version / "frames"
    colmap_dir = exp_recon / exp_name / version / "colmap"
    ns_data_dir = exp_recon / exp_name / version / "ns_data"
    splats_dir = exp_splats / exp_name / version

    for p in (frames_dir, colmap_dir, ns_data_dir, splats_dir):
        p.mkdir(parents=True, exist_ok=True)

    return ExpDirs(frames_dir=frames_dir, colmap_dir=colmap_dir, ns_data_dir=ns_data_dir, splats_dir=splats_dir)


def print_menu(existing: List[ExpRef]) -> None:
    if not existing:
        print("No existing experiments found.")
        return

    print("\nExisting experiments (by video):")
    grouped: dict[str, list[str]] = {}
    for e in existing:
        grouped.setdefault(e.exp_name, []).append(e.version)

    for i, exp in enumerate(sorted(grouped.keys()), start=1):
        vers = ", ".join(sorted(grouped[exp]))
        print(f"  [{i}] {exp}: {vers}")


def choose_existing(existing: List[ExpRef]) -> Optional[ExpRef]:
    if not existing:
        return None

    grouped: dict[str, list[str]] = {}
    for e in existing:
        grouped.setdefault(e.exp_name, []).append(e.version)

    exps = sorted(grouped.keys())
    while True:
        idx = input("\nChoose video experiment number (or empty to cancel): ").strip()
        if idx == "":
            return None
        if idx.isdigit() and 1 <= int(idx) <= len(exps):
            exp_name = exps[int(idx) - 1]
            versions = sorted(grouped[exp_name])
            print(f"Versions for {exp_name}: {', '.join(versions)}")
            ver = input("Choose version (e.g., v0001): ").strip()
            if ver in versions:
                return ExpRef(exp_name=exp_name, version=ver)
        print("Invalid selection, try again.")
