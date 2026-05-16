"""Standard (full-image) YOLO11 OBB inference, output-compatible with sahi_inference.py.

Used as the apples-to-apples baseline against ``sahi_inference.py`` when both
are run on the same image subset and evaluated with ``analyze_errors.py --predictions-only``.

Output: one YOLO OBB txt per image, format
``class_id x1 y1 x2 y2 x3 y3 x4 y4 conf`` in original-image pixel coordinates.

The ``--from-sahi-dir`` option restricts processing to the images that already
have prediction files in the provided SAHI output directory, so the standard
and SAHI runs cover identical image sets.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard YOLO11 OBB inference (no slicing).")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="Path to data.yaml.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit images for debugging (independent of --from-sahi-dir).")
    parser.add_argument("--from-sahi-dir", type=Path, default=None,
                        help="Only process images whose stem matches a .txt file in this directory. "
                             "Useful for apples-to-apples comparison against a prior SAHI run.")
    return parser.parse_args()


def load_image_paths(dataset_yaml: Path, split: str) -> list[Path]:
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    base = Path(cfg.get("path", dataset_yaml.parent))
    if not base.is_absolute():
        base = (dataset_yaml.parent / base).resolve()
    split_rel = cfg.get(split)
    if split_rel is None:
        raise ValueError(f"split '{split}' not found in {dataset_yaml}")
    img_dir = (base / split_rel).resolve()
    if not img_dir.is_dir():
        raise FileNotFoundError(f"image dir {img_dir} does not exist")
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


def filter_by_sahi_dir(images: list[Path], sahi_dir: Path) -> list[Path]:
    if not sahi_dir.is_dir():
        raise FileNotFoundError(f"--from-sahi-dir {sahi_dir} not found")
    allowed_stems = {p.stem for p in sahi_dir.iterdir() if p.suffix == ".txt"}
    kept = [img for img in images if img.stem in allowed_stems]
    if not kept:
        raise RuntimeError(
            f"No images matched any .txt in {sahi_dir}. "
            f"Sample image stems: {[p.stem for p in images[:3]]}; "
            f"sample txt stems: {sorted(allowed_stems)[:3]}"
        )
    return kept


def obb_corners(cx: float, cy: float, w: float, h: float, angle: float) -> list[float]:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    w2, h2 = w / 2.0, h / 2.0
    corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
    out = []
    for x, y in corners:
        out.append(cx + x * cos_a - y * sin_a)
        out.append(cy + x * sin_a + y * cos_a)
    return out


def write_predictions_txt(out_path: Path, preds: torch.Tensor) -> None:
    """preds: (N, 7) = (cx, cy, w, h, angle, conf, class)."""
    lines = []
    for row in preds.tolist():
        cx, cy, w, h, angle, conf, cls = row
        cls_id = int(cls)
        corners = obb_corners(cx, cy, w, h, angle)
        coords_str = " ".join(f"{c:.2f}" for c in corners)
        lines.append(f"{cls_id} {coords_str} {conf:.6f}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_single_image(
    model: YOLO,
    img_path: Path,
    imgsz: int,
    conf_thres: float,
    nms_iou: float,
    device: str,
    max_det: int,
) -> torch.Tensor:
    results = model.predict(
        str(img_path),
        imgsz=imgsz,
        conf=conf_thres,
        iou=nms_iou,
        device=device,
        verbose=False,
        max_det=max_det,
    )
    obb = results[0].obb
    if obb is None or len(obb) == 0:
        return torch.empty((0, 7))
    xywhr = obb.xywhr.cpu()
    conf = obb.conf.cpu().unsqueeze(-1)
    cls = obb.cls.cpu().unsqueeze(-1)
    return torch.cat([xywhr, conf, cls], dim=-1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = load_image_paths(args.dataset, args.split)
    if args.from_sahi_dir is not None:
        images = filter_by_sahi_dir(images, args.from_sahi_dir)
        print(f"[standard] filtered to {len(images)} images that match {args.from_sahi_dir}")
    if args.max_images is not None:
        images = images[: args.max_images]

    print(f"[standard] loading {args.weights}")
    model = YOLO(str(args.weights))

    total_t = 0.0
    total_preds = 0
    print(f"[standard] inferring {len(images)} images at imgsz={args.imgsz}")

    for i, img_path in enumerate(images, 1):
        t0 = time.time()
        preds = run_single_image(
            model=model,
            img_path=img_path,
            imgsz=args.imgsz,
            conf_thres=args.conf_thres,
            nms_iou=args.nms_iou,
            device=args.device,
            max_det=args.max_det,
        )
        dt = time.time() - t0
        total_t += dt
        total_preds += len(preds)

        out_path = args.output_dir / f"{img_path.stem}.txt"
        write_predictions_txt(out_path, preds)

        if i % 50 == 0 or i == len(images):
            print(f"[standard] {i}/{len(images)}  avg {total_t / i:.3f}s/img  total_preds={total_preds}")

    meta = {
        "weights": str(args.weights),
        "dataset": str(args.dataset),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf_thres": args.conf_thres,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "from_sahi_dir": str(args.from_sahi_dir) if args.from_sahi_dir else None,
        "num_images": len(images),
        "total_inference_seconds": round(total_t, 2),
        "average_seconds_per_image": round(total_t / max(1, len(images)), 4),
        "total_predictions": total_preds,
        "average_predictions_per_image": round(total_preds / max(1, len(images)), 2),
    }
    (args.output_dir / "standard_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[standard] done. meta saved to {args.output_dir / 'standard_meta.json'}")


if __name__ == "__main__":
    main()
