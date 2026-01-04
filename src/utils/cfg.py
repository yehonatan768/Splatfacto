from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "paths": read_yaml(config_dir / "paths.yaml"),
        "frames": read_yaml(config_dir / "frames.yaml"),
        "colmap": read_yaml(config_dir / "colmap.yaml"),
        "ns_process": read_yaml(config_dir / "ns_process.yaml"),
        "train": read_yaml(config_dir / "train.yaml"),
        "export": read_yaml(config_dir / "export.yaml"),
    }
