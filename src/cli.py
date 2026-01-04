from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.cfg import load_all_configs
from src.utils.paths import resolve_paths, ensure_dirs
from src.utils.exp import slugify_video_name
from src.utils.log import get_logger, stage

from src.utils.step_paths import build_step_paths, ensure_only

from src.video_to_frame.extract import extract_frames
from src.colmap.sfm import run_sfm
from src.process_data.ns_process import process_images
from src.train.splat import train_splat
from src.export.splat import export_splat
from src.export.pointcloud import export_colmap_pointcloud


def _base(cfg_dir: Path):
    cfgs = load_all_configs(cfg_dir)
    P = resolve_paths(cfgs["paths"])
    ensure_dirs(P.videos, P.experiments, P.exp_data, P.exp_recon, P.exp_splats)
    return cfgs, P


def _logger_for(P, exp: str, ver: str):
    # one log file per exp/version regardless of step
    log_file = P.exp_splats / exp / ver / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return get_logger(f"{exp}/{ver}", log_file=log_file)


def main():
    parser = argparse.ArgumentParser(prog="splat-pipeline", description="Run steps individually")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # extract
    p = sub.add_parser("extract", help="Video -> frames")
    p.add_argument("--video", required=True, help="Video filename inside videos/ OR full path")
    p.add_argument("--exp", default="", help="Experiment name (default: slug from video)")
    p.add_argument("--ver", required=True, help="Version like v0001")

    # sfm
    p = sub.add_parser("sfm", help="Frames -> COLMAP sparse")
    p.add_argument("--exp", required=True)
    p.add_argument("--ver", required=True)

    # ns-process
    p = sub.add_parser("ns", help="Frames -> Nerfstudio dataset (transforms.json)")
    p.add_argument("--exp", required=True)
    p.add_argument("--ver", required=True)

    # train
    p = sub.add_parser("train", help="Train splatfacto")
    p.add_argument("--exp", required=True)
    p.add_argument("--ver", required=True)

    # export splat
    p = sub.add_parser("export-splat", help="Export splat PLY from trained run")
    p.add_argument("--exp", required=True)
    p.add_argument("--ver", required=True)

    # export point cloud
    p = sub.add_parser("export-pc", help="Export COLMAP point cloud")
    p.add_argument("--exp", required=True)
    p.add_argument("--ver", required=True)

    args = parser.parse_args()
    cfg_dir = Path(__file__).resolve().parent / "config"
    cfgs, P = _base(cfg_dir)

    if args.cmd == "extract":
        # Resolve video path
        vp = Path(args.video)
        if not vp.is_file():
            vp = P.videos / args.video
        if not vp.exists():
            raise FileNotFoundError(f"Video not found: {vp}")

        exp = args.exp.strip() or slugify_video_name(vp.name)
        ver = args.ver.strip()

        D = build_step_paths(P.exp_data, P.exp_recon, P.exp_splats, exp, ver)
        logger = _logger_for(P, exp, ver)

        # Only create frames dir for this step
        ensure_only(D.frames_dir)

        with stage(logger, "STEP extract"):
            extract_frames(vp, D.frames_dir, cfgs["frames"], logger=logger)
        return

    # All other steps: require exp/ver
    exp = args.exp.strip()
    ver = args.ver.strip()
    D = build_step_paths(P.exp_data, P.exp_recon, P.exp_splats, exp, ver)
    logger = _logger_for(P, exp, ver)

    if args.cmd == "sfm":
        ensure_only(D.colmap_dir)  # only output folder for sfm
        with stage(logger, "STEP sfm"):
            run_sfm(D.frames_dir, D.colmap_dir, cfgs["colmap"], logger=logger)
        return

    if args.cmd == "ns":
        ensure_only(D.ns_data_dir)
        with stage(logger, "STEP ns-process"):
            process_images(D.frames_dir, D.ns_data_dir, cfgs["ns_process"], logger=logger)
        return

    if args.cmd == "train":
        ensure_only(D.splats_dir)
        with stage(logger, "STEP train"):
            train_splat(D.ns_data_dir, D.splats_dir, cfgs["train"])
        return

    if args.cmd == "export-splat":
        exports_dir = D.splats_dir / "exports"
        ensure_only(exports_dir)
        with stage(logger, "STEP export-splat"):
            configs = list(D.splats_dir.rglob("config.yml"))
            if not configs:
                raise FileNotFoundError(f"No config.yml found under: {D.splats_dir} (train first)")
            export_splat(configs[0], exports_dir, cfgs["export"])
        return

    if args.cmd == "export-pc":
        ensure_only(D.colmap_dir)
        with stage(logger, "STEP export-pointcloud"):
            export_colmap_pointcloud(D.frames_dir, D.colmap_dir, cfgs["export"])
        return


if __name__ == "__main__":
    main()
