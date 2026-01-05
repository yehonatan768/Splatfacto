from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import build_logger, LogConfig
from src.utils.paths import RunPaths
from src.utils.yamlio import read_yaml, deep_merge

from src.pipeline.extract_frames import FfmpegConfig, SmartSelectConfig, extract_frames, smart_select
from src.pipeline.sfm_colmap import run_sfm
from src.pipeline.ns_dataset import NsDatasetConfig, build_ns_dataset
from src.pipeline.train import TrainConfig, train_splatfacto
from src.pipeline.export import ExportConfig, export_outputs


def _project_root() -> Path:
    # Assume running from repo root; fall back to cwd.
    return Path.cwd()


def _load_cfgs() -> Dict[str, Dict[str, Any]]:
    cfg_dir = Path(__file__).parent / "config"
    return {
        "extract": read_yaml(cfg_dir / "extract.yaml"),
        "colmap": read_yaml(cfg_dir / "colmap.yaml"),
        "ns_dataset": read_yaml(cfg_dir / "ns_dataset.yaml"),
        "train": read_yaml(cfg_dir / "train.yaml"),
        "export": read_yaml(cfg_dir / "export.yaml"),
    }


def cmd_extract(P: RunPaths, cfgs: Dict[str, Dict[str, Any]], video: Path, *, logger):
    ex = cfgs["extract"]
    ff = ex.get("ffmpeg", {})
    ss = ex.get("smart_select", {})

    # 1) ffmpeg extraction into _candidates
    extract_frames(
        video_path=video,
        frames_dir=P.frames_dir,
        candidates_dir=P.frames_candidates_dir,
        ff=FfmpegConfig(
            fps=float(ff.get("fps", 3.0)),
            scale_max_width=int(ff.get("scale_max_width", 2048)),
            quality_qv=int(ff.get("quality_qv", 2)),
            loglevel=str(ff.get("loglevel", "info")),
        ),
        logger=logger,
    )

    # 2) smart selection into frames/
    if bool(ss.get("enabled", True)):
        smart_select(
            candidates_dir=P.frames_candidates_dir,
            frames_dir=P.frames_dir,
            cfg=SmartSelectConfig(
                enabled=True,
                max_frames=int(ss.get("max_frames", 600)),
                min_sharpness=float(ss.get("min_sharpness", 10.0)),
                keep_every_n_after_filter=int(ss.get("keep_every_n_after_filter", 1)),
                min_hist_delta=float(ss.get("min_hist_delta", 0.012)),
            ),
            logger=logger,
        )


def cmd_process(P: RunPaths, cfgs: Dict[str, Dict[str, Any]], *, logger):
    # 1) Run your COLMAP SfM
    sparse0 = run_sfm(P.frames_dir, P.colmap_dir, cfgs["colmap"], logger=logger)

    # 2) Convert COLMAP -> Nerfstudio dataset (no ns-process-data)
    ns_cfg = cfgs["ns_dataset"]
    conv = ns_cfg.get("convert", {})
    coords = ns_cfg.get("coords", {})

    build_ns_dataset(
        frames_dir=P.frames_dir,
        colmap_sparse0=sparse0,
        ns_data_dir=P.ns_data_dir,
        cfg=NsDatasetConfig(
            hardlink_images=bool(conv.get("hardlink_images", True)),
            auto_orient_up=bool(conv.get("auto_orient_up", True)),
            auto_center=bool(conv.get("auto_center", True)),
            auto_scale=bool(conv.get("auto_scale", True)),
            target_radius=float(conv.get("target_radius", 1.0)),
            sort_by_name=bool(conv.get("sort_by_name", True)),
            opencv_to_opengl=bool(coords.get("opencv_to_opengl", True)),
        ),
        logger=logger,
    )


def cmd_train(P: RunPaths, cfgs: Dict[str, Dict[str, Any]], *, logger):
    tr = cfgs["train"].get("train", {})
    extra = tr.get("extra_args", []) or []
    train_splatfacto(
        ns_data_dir=P.ns_data_dir,
        outputs_dir=P.outputs_dir,
        cfg=TrainConfig(method=str(tr.get("method", "splatfacto")), extra_args=list(extra)),
        logger=logger,
    )


def cmd_export(P: RunPaths, cfgs: Dict[str, Dict[str, Any]], *, logger):
    ex = cfgs["export"].get("export", {})
    export_outputs(
        outputs_dir=P.outputs_dir,
        exports_dir=P.exports_dir,
        cfg=ExportConfig(
            gaussian_splat=bool(ex.get("gaussian_splat", True)),
            point_cloud=bool(ex.get("point_cloud", True)),
            splat_extra_args=list(ex.get("splat_extra_args", []) or []),
            pointcloud_extra_args=list(ex.get("pointcloud_extra_args", []) or []),
            run_dir=str(ex.get("run_dir", "") or ""),
        ),
        logger=logger,
    )


def main():
    ap = argparse.ArgumentParser(description="Splatfacto aerial pipeline (FFmpeg + COLMAP + NS dataset + train + export)")
    ap.add_argument("--exp", required=True, help="experiment name (e.g., dji0004)")
    ap.add_argument("--ver", required=True, help="version (e.g., v0001)")
    ap.add_argument("--project-root", default="", help="optional project root (default: cwd)")
    ap.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="extract smart frames from video")
    p_ex.add_argument("--video", required=True, help="video filename under videos/ or full path")

    sub.add_parser("process", help="run COLMAP SfM + convert to Nerfstudio dataset")
    sub.add_parser("train", help="train splatfacto on ns dataset")
    sub.add_parser("export", help="export gaussian splat + pointcloud from latest run")

    p_all = sub.add_parser("all", help="run extract -> process -> train -> export")
    p_all.add_argument("--video", required=True, help="video filename under videos/ or full path")

    args = ap.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else _project_root()
    P = RunPaths(project_root=project_root, exp=args.exp, ver=args.ver)

    logger = build_logger(f"{args.exp}/{args.ver}", run_dir=P.run_logs_dir, cfg=LogConfig(level=args.log_level, to_file=True))
    cfgs = _load_cfgs()

    def resolve_video(v: str) -> Path:
        p = Path(v)
        if p.exists():
            return p.resolve()
        cand = project_root / "videos" / v
        if cand.exists():
            return cand.resolve()
        raise FileNotFoundError(f"Video not found: {v} (or {cand})")

    if args.cmd == "extract":
        cmd_extract(P, cfgs, resolve_video(args.video), logger=logger)
    elif args.cmd == "process":
        cmd_process(P, cfgs, logger=logger)
    elif args.cmd == "train":
        cmd_train(P, cfgs, logger=logger)
    elif args.cmd == "export":
        cmd_export(P, cfgs, logger=logger)
    elif args.cmd == "all":
        cmd_extract(P, cfgs, resolve_video(args.video), logger=logger)
        cmd_process(P, cfgs, logger=logger)
        cmd_train(P, cfgs, logger=logger)
        cmd_export(P, cfgs, logger=logger)
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
