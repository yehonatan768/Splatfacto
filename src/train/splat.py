from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from src.utils.shell import run


def train_splat(data_dir: Path, out_dir: Path, cfg: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    t = cfg["train"]
    tweaks = cfg["model_tweaks"]

    cmd = [
        "ns-train", t["method"],
        "--data", str(data_dir),
        "--output-dir", str(out_dir),
        "--max-num-iterations", str(t["max_num_iterations"]),
        "--pipeline.model.use-scale-regularization", "True" if tweaks["use_scale_regularization"] else "False",
    ]
    if t.get("quit_on_completion", True):
        cmd += ["--viewer.quit-on-train-completion", "True"]
    run(cmd)
