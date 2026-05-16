from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn import tasks as yolo_nn_tasks

from custom_modules import ProjectCBAM, ResidualCBFuse

# Baseline example:
# python train.py --data datasets/DOTA-split-lite/data.yaml --model yolo11s-obb.pt --name yolo11s_obb_baseline
# P2 experiment example:
# python train.py --data datasets/DOTA-split-lite/data.yaml --cfg models/yolo11s-obb-p2.yaml --weights yolo11s-obb.pt --name yolo11s_obb_p2 --batch 4
# Generic P2 config example with explicit scale:
# python train.py --data datasets/DOTA-split-lite/data.yaml --cfg models/yolo11-obb-p2.yaml --scale s --weights yolo11s-obb.pt --name yolo11s_obb_p2 --batch 4

DEFAULTS = {
    "model": "yolo11s-obb.pt",
    "weights": "yolo11s-obb.pt",
    "imgsz": 1024,
    "epochs": 100,
    "batch": 8,
    "device": "auto",
    "workers": 8,
    "optimizer": "AdamW",
    "lr0": 1e-3,
    "weight_decay": 5e-4,
    "project": "dota_runs",
    "name": None,
    "exist_ok": True,
    "plots": True,
    "verbose": True,
    "mosaic": 1.0,
    "mixup": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "close_mosaic": 10,
    "patience": 50,
    "save_period": -1,
    "amp": True,
    "cache": "False",
    "seed": 0,
    "deterministic": True,
    "wandb_project": "rsdd_yolo11_obb",
    "wandb_entity": "",
    "wandb_mode": "online",
}
SCALE_PATTERN = re.compile(r"yolo(?:v)?\d+([nslmx])")




def register_custom_modules() -> None:
    """Register project-local custom modules for Ultralytics YAML parsing."""
    yolo_nn_tasks.CBAM = ProjectCBAM
    yolo_nn_tasks.CBFuse = ResidualCBFuse
    yolo_nn_tasks.ResidualCBFuse = ResidualCBFuse


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="YOLO11 OBB training launcher with optional P2 config support")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data", type=Path, default=None, help="Optional explicit data.yaml path")
    parser.add_argument("--model", type=str, default=DEFAULTS["model"], help="Direct model source for baseline training or resume")
    parser.add_argument("--cfg", type=str, default=None, help="Model YAML path for custom architectures, e.g. models/yolo11s-obb-p2.yaml")
    parser.add_argument("--weights", type=str, default=DEFAULTS["weights"], help="Optional pretrained weights to partially load when --cfg is used")
    parser.add_argument("--scale", type=str, default=None, choices=["n", "s", "m", "l", "x"], help="Explicit model scale for generic cfg names like yolo11-obb-p2.yaml")
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    parser.add_argument("--device", type=str, default=DEFAULTS["device"], help="auto, cpu, 0, 0,1")
    parser.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    parser.add_argument("--project", type=str, default=DEFAULTS["project"])
    parser.add_argument("--name", type=str, default=DEFAULTS["name"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None, help="Explicit checkpoint path for resume mode")
    parser.add_argument("--cache", type=str, default=DEFAULTS["cache"], help="False, ram, or disk")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--close-mosaic", type=int, default=DEFAULTS["close_mosaic"])
    parser.add_argument("--save-period", type=int, default=DEFAULTS["save_period"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--wandb-project", type=str, default=DEFAULTS["wandb_project"])
    parser.add_argument("--wandb-entity", type=str, default=DEFAULTS["wandb_entity"])
    parser.add_argument("--wandb-mode", type=str, default=DEFAULTS["wandb_mode"], choices=["online", "offline", "disabled"])
    parser.add_argument("--use-nwd", action="store_true", help="Enable NWD loss term in RotatedBboxLoss. See docs/paper/nwd.md.")
    parser.add_argument("--nwd-c", type=float, default=64.0, help="NWD normalization constant in pixels (default 64 for DOTA-split-lite).")
    parser.add_argument("--nwd-weight", type=float, default=0.5, help="Weight alpha of NWD loss in alpha*L_NWD + (1-alpha)*L_ProbIoU.")
    return parser.parse_args()


def resolve_data_yaml(project_root: Path, explicit_data: Path | None) -> Path:
    if explicit_data is not None:
        explicit_path = explicit_data if explicit_data.is_absolute() else (project_root / explicit_data)
        explicit_path = explicit_path.resolve()
        if not explicit_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {explicit_path}")
        return explicit_path

    candidates = [
        project_root / "datasets" / "DOTA-split-lite" / "data.yaml",
        project_root / "datasets" / "DOTA-split-nobg" / "data.yaml",
        project_root / "datasets" / "DOTA" / "data.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError("No dataset yaml found. Checked: " + ", ".join(str(p) for p in candidates))


def resolve_local_path(project_root: Path, value: str | Path | None, description: str, *, allow_missing: bool = False) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    candidate = path if path.is_absolute() else (project_root / path)
    candidate = candidate.resolve()
    if candidate.exists() or allow_missing:
        return candidate
    raise FileNotFoundError(f"{description} not found: {candidate}")


def configure_wandb(args: argparse.Namespace) -> None:
    os.environ["WANDB_MODE"] = args.wandb_mode
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity


def save_resolved_config(run_dir: Path, config: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def resolve_run_name(args: argparse.Namespace, cfg_path: Path | None, weight_path: Path | None, model_source: Path | None) -> str:
    if args.name:
        return args.name
    if cfg_path is not None:
        weight_tag = weight_path.stem if weight_path is not None else "scratch"
        return f"{cfg_path.stem}_{weight_tag}"
    if model_source is not None:
        return model_source.stem
    return "yolo11_obb_run"


def guess_model_scale(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    match = SCALE_PATTERN.search(Path(str(value)).stem)
    return match.group(1) if match else None


def resolve_cfg_scale(project_root: Path, cfg_path: Path, explicit_scale: str | None, weight_path: Path | None, model_path: Path | None) -> tuple[Path, str | None]:
    cfg_scale = guess_model_scale(cfg_path)
    target_scale = explicit_scale or cfg_scale or guess_model_scale(weight_path) or guess_model_scale(model_path)
    if target_scale is None or cfg_scale is not None:
        return cfg_path, target_scale

    stem = cfg_path.stem
    match = re.match(r"(yolo(?:v)?\d+)(-.+)", stem)
    if not match:
        return cfg_path, target_scale

    scaled_path = cfg_path.with_name(f"{match.group(1)}{target_scale}{match.group(2)}{cfg_path.suffix}")
    if not scaled_path.exists():
        shutil.copyfile(cfg_path, scaled_path)
    return scaled_path.resolve(), target_scale


def resolve_device(requested_device: str) -> str:
    try:
        import torch
    except Exception:
        return "cpu" if requested_device in {"", "auto"} else requested_device

    if requested_device in {"", "auto"}:
        return "0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    if requested_device != "cpu" and not torch.cuda.is_available():
        print(f"[train] warning: requested device={requested_device} but torch.cuda.is_available() is False. Falling back to CPU.")
        return "cpu"
    return requested_device


def build_model(project_root: Path, args: argparse.Namespace) -> tuple[YOLO, dict]:
    if args.resume and args.cfg:
        raise ValueError("--resume and --cfg should not be used together.")

    if args.resume:
        resume_path = resolve_local_path(project_root, args.resume_from or args.model, "resume checkpoint")
        assert resume_path is not None
        model = YOLO(str(resume_path))
        return model, {
            "build_mode": "resume",
            "resume_from": str(resume_path),
            "model_source": str(resume_path),
            "cfg": None,
            "weights": None,
            "scale": guess_model_scale(resume_path),
        }

    if args.cfg:
        weight_path = resolve_local_path(project_root, args.weights, "pretrained weights") if args.weights else None
        model_path = resolve_local_path(project_root, args.model, "model source") if args.model else None
        cfg_path = resolve_local_path(project_root, args.cfg, "model cfg")
        assert cfg_path is not None
        resolved_cfg_path, resolved_scale = resolve_cfg_scale(project_root, cfg_path, args.scale, weight_path, model_path)
        model = YOLO(str(resolved_cfg_path))
        if weight_path is not None:
            model = model.load(str(weight_path))
        return model, {
            "build_mode": "cfg+weights",
            "resume_from": None,
            "model_source": str(resolved_cfg_path),
            "cfg": str(resolved_cfg_path),
            "weights": str(weight_path) if weight_path is not None else None,
            "scale": resolved_scale,
        }

    model_source = resolve_local_path(project_root, args.model, "model source")
    assert model_source is not None
    model = YOLO(str(model_source))
    return model, {
        "build_mode": "direct",
        "resume_from": None,
        "model_source": str(model_source),
        "cfg": None,
        "weights": None,
        "scale": guess_model_scale(model_source),
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    register_custom_modules()
    data_yaml = resolve_data_yaml(project_root, args.data)
    model, build_info = build_model(project_root, args)

    if args.use_nwd:
        from nwd_loss import enable_nwd_loss
        enable_nwd_loss(nwd_c=args.nwd_c, nwd_weight=args.nwd_weight)
        print(f"[nwd] enabled: C={args.nwd_c}, alpha={args.nwd_weight}")

    cfg_path = Path(build_info["cfg"]) if build_info.get("cfg") else None
    weight_path = Path(build_info["weights"]) if build_info.get("weights") else None
    model_source = Path(build_info["model_source"]) if build_info.get("model_source") else None
    run_name = resolve_run_name(args, cfg_path, weight_path, model_source)
    resolved_device = resolve_device(args.device)

    configure_wandb(args)

    train_kwargs = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": resolved_device,
        "workers": args.workers,
        "optimizer": DEFAULTS["optimizer"],
        "lr0": DEFAULTS["lr0"],
        "weight_decay": DEFAULTS["weight_decay"],
        "project": args.project,
        "name": run_name,
        "exist_ok": DEFAULTS["exist_ok"],
        "plots": DEFAULTS["plots"],
        "verbose": DEFAULTS["verbose"],
        "mosaic": DEFAULTS["mosaic"],
        "mixup": DEFAULTS["mixup"],
        "degrees": DEFAULTS["degrees"],
        "translate": DEFAULTS["translate"],
        "scale": DEFAULTS["scale"],
        "shear": DEFAULTS["shear"],
        "perspective": DEFAULTS["perspective"],
        "close_mosaic": args.close_mosaic,
        "patience": args.patience,
        "save_period": args.save_period,
        "amp": DEFAULTS["amp"],
        "cache": args.cache,
        "seed": args.seed,
        "deterministic": DEFAULTS["deterministic"],
    }
    if args.resume:
        train_kwargs["resume"] = True

    if cfg_path is not None and "p2" in cfg_path.stem.lower() and args.batch >= 8:
        print("[train] warning: P2 head increases memory usage. If you see OOM, retry with --batch 4 or --batch 2.")

    print(f"[train] build_mode={build_info['build_mode']}")
    print(f"[train] model_source={build_info['model_source']}")
    if build_info.get("weights"):
        print(f"[train] pretrained_weights={build_info['weights']}")
    if build_info.get("scale"):
        print(f"[train] model_scale={build_info['scale']}")
    print(f"[train] data={data_yaml}")
    print(f"[train] project={args.project}")
    print(f"[train] name={run_name}")
    print(f"[train] imgsz={args.imgsz} batch={args.batch} epochs={args.epochs}")
    print(f"[train] device={resolved_device}")

    run_dir = project_root / args.project / run_name
    save_resolved_config(
        run_dir,
        {
            "project_root": str(project_root),
            "data": str(data_yaml),
            "run_name": run_name,
            "requested_device": args.device,
            "resolved_device": resolved_device,
            "use_nwd": bool(args.use_nwd),
            "nwd_c": float(args.nwd_c) if args.use_nwd else None,
            "nwd_weight": float(args.nwd_weight) if args.use_nwd else None,
            **build_info,
            **train_kwargs,
        },
    )

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
