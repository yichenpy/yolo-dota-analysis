"""SAHI-style sliced inference for YOLO11 OBB on DOTA-style datasets.

This is a self-contained reimplementation that does not depend on the upstream
``sahi`` library. It uses Ultralytics for per-slice inference and Ultralytics'
own rotated NMS for cross-slice merging, which avoids known OBB compatibility
caveats in sahi < 0.12.

Output: one YOLO OBB txt file per validation image, format
``class_id x1 y1 x2 y2 x3 y3 x4 y4 conf`` in original-image pixel coordinates.
Compatible with ``analyze_errors.py --predictions <dir> --prediction-layout class_xyxyxyxy_conf``.

See docs/paper/sahi.md and docs/design/sahi-inference.md for theory and design.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def rotated_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Cross-version OBB NMS.

    Tries import paths in order: 8.3.x direct ``nms_rotated``, 8.4.x ``TorchNMS.fast_nms``
    with ``batch_probiou``, and finally a self-contained greedy NMS using ``batch_probiou``.
    Returns indices into the input that survive suppression, sorted by descending score.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    try:
        from ultralytics.utils.ops import nms_rotated as _nms_rotated
        return _nms_rotated(boxes, scores, threshold=iou_threshold)
    except (ImportError, AttributeError):
        pass

    try:
        from ultralytics.utils.nms import TorchNMS
        from ultralytics.utils.metrics import batch_probiou
        return TorchNMS.fast_nms(boxes, scores, iou_threshold, iou_func=batch_probiou)
    except (ImportError, AttributeError):
        pass

    from ultralytics.utils.metrics import batch_probiou
    order = scores.argsort(descending=True)
    keep: list[int] = []
    remaining = order.tolist()
    while remaining:
        i = remaining[0]
        keep.append(i)
        if len(remaining) == 1:
            break
        rest = torch.as_tensor(remaining[1:], dtype=torch.long, device=boxes.device)
        ious = batch_probiou(boxes[i:i + 1], boxes[rest]).squeeze(0)
        survive = (ious <= iou_threshold).nonzero(as_tuple=True)[0]
        remaining = rest[survive].tolist()
    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAHI-style sliced inference for YOLO11 OBB.")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="Path to data.yaml.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--slice-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--no-standard-pred", action="store_true",
                        help="Skip the full-image inference pass; only run on slices.")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="imgsz for the full-image pass when --no-standard-pred is not set.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-det-per-slice", type=int, default=300)
    return parser.parse_args()


def load_image_paths(dataset_yaml: Path, split: str) -> tuple[list[Path], dict]:
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
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    return images, cfg


def compute_slice_starts(total: int, slice_size: int, overlap: float) -> list[int]:
    """Slide-window start positions covering [0, total] with the requested overlap."""
    if total <= slice_size:
        return [0]
    step = max(1, int(round(slice_size * (1.0 - overlap))))
    starts = list(range(0, total - slice_size + 1, step))
    last = total - slice_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def slice_image(img: np.ndarray, slice_size: int, overlap: float) -> list[tuple[np.ndarray, int, int]]:
    """Return list of (patch, offset_x, offset_y)."""
    h, w = img.shape[:2]
    xs = compute_slice_starts(w, slice_size, overlap)
    ys = compute_slice_starts(h, slice_size, overlap)
    patches = []
    for y0 in ys:
        for x0 in xs:
            patch = img[y0:y0 + slice_size, x0:x0 + slice_size]
            patches.append((patch, x0, y0))
    return patches


def obb_corners(cx: float, cy: float, w: float, h: float, angle: float) -> list[float]:
    """Convert OBB (cx, cy, w, h, angle) to 8 corner coordinates."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    w2, h2 = w / 2.0, h / 2.0
    corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
    out = []
    for x, y in corners:
        out.append(cx + x * cos_a - y * sin_a)
        out.append(cy + x * sin_a + y * cos_a)
    return out


def run_inference_on_image(
    model: YOLO,
    img: np.ndarray,
    slice_size: int,
    overlap: float,
    perform_standard_pred: bool,
    full_imgsz: int,
    conf_thres: float,
    nms_iou: float,
    max_det_per_slice: int,
    device: str,
) -> torch.Tensor:
    """Return tensor of shape (N, 7) = (cx, cy, w, h, angle, conf, class) in original-image coords."""
    all_boxes = []  # list of (5,) tensors
    all_scores = []
    all_classes = []

    h_img, w_img = img.shape[:2]

    # Per-slice inference
    patches = slice_image(img, slice_size, overlap)
    for patch, x0, y0 in patches:
        if patch.size == 0:
            continue
        results = model.predict(
            patch,
            imgsz=slice_size,
            conf=conf_thres,
            iou=nms_iou,
            device=device,
            verbose=False,
            max_det=max_det_per_slice,
        )
        obb = results[0].obb
        if obb is None or len(obb) == 0:
            continue
        xywhr = obb.xywhr.cpu()      # (n, 5)
        confs = obb.conf.cpu()        # (n,)
        clses = obb.cls.cpu().long()  # (n,)
        # Translate centers back to original image
        xywhr[:, 0] += x0
        xywhr[:, 1] += y0
        all_boxes.append(xywhr)
        all_scores.append(confs)
        all_classes.append(clses)

    # Full-image inference (catch large objects)
    if perform_standard_pred:
        results = model.predict(
            img,
            imgsz=full_imgsz,
            conf=conf_thres,
            iou=nms_iou,
            device=device,
            verbose=False,
            max_det=max_det_per_slice * 3,
        )
        obb = results[0].obb
        if obb is not None and len(obb) > 0:
            all_boxes.append(obb.xywhr.cpu())
            all_scores.append(obb.conf.cpu())
            all_classes.append(obb.cls.cpu().long())

    if not all_boxes:
        return torch.empty((0, 7))

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    classes = torch.cat(all_classes, dim=0)

    # Global cross-slice NMS (per-class via offset trick)
    if boxes.numel():
        max_dim = float(max(h_img, w_img)) * 2.0
        offset = classes.to(boxes.dtype) * max_dim
        offset_boxes = boxes.clone()
        offset_boxes[:, 0] += offset
        offset_boxes[:, 1] += offset
        keep = rotated_nms(offset_boxes, scores, iou_threshold=nms_iou)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

    out = torch.cat([
        boxes,
        scores.unsqueeze(-1),
        classes.to(boxes.dtype).unsqueeze(-1),
    ], dim=-1)
    return out


def write_predictions_txt(out_path: Path, preds: torch.Tensor) -> None:
    lines = []
    for row in preds.tolist():
        cx, cy, w, h, angle, conf, cls = row
        cls_id = int(cls)
        corners = obb_corners(cx, cy, w, h, angle)
        coords_str = " ".join(f"{c:.2f}" for c in corners)
        lines.append(f"{cls_id} {coords_str} {conf:.6f}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    images, cfg = load_image_paths(args.dataset, args.split)
    if args.max_images is not None:
        images = images[: args.max_images]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sahi] loading {args.weights}")
    model = YOLO(str(args.weights))

    total_t = 0.0
    total_preds = 0
    print(f"[sahi] inferring {len(images)} images, slice={args.slice_size}, overlap={args.overlap}, "
          f"standard_pred={not args.no_standard_pred}")

    for i, img_path in enumerate(images, 1):
        img = np.array(Image.open(img_path).convert("RGB"))
        t0 = time.time()
        preds = run_inference_on_image(
            model=model,
            img=img,
            slice_size=args.slice_size,
            overlap=args.overlap,
            perform_standard_pred=not args.no_standard_pred,
            full_imgsz=args.imgsz,
            conf_thres=args.conf_thres,
            nms_iou=args.nms_iou,
            max_det_per_slice=args.max_det_per_slice,
            device=args.device,
        )
        dt = time.time() - t0
        total_t += dt
        total_preds += len(preds)

        out_path = args.output_dir / f"{img_path.stem}.txt"
        write_predictions_txt(out_path, preds)

        if i % 50 == 0 or i == len(images):
            avg = total_t / i
            print(f"[sahi] {i}/{len(images)}  avg {avg:.2f}s/img  total_preds={total_preds}")

    meta = {
        "weights": str(args.weights),
        "dataset": str(args.dataset),
        "split": args.split,
        "slice_size": args.slice_size,
        "overlap": args.overlap,
        "conf_thres": args.conf_thres,
        "nms_iou": args.nms_iou,
        "perform_standard_pred": not args.no_standard_pred,
        "full_imgsz": args.imgsz,
        "num_images": len(images),
        "total_inference_seconds": round(total_t, 2),
        "average_seconds_per_image": round(total_t / max(1, len(images)), 3),
        "total_predictions": total_preds,
        "average_predictions_per_image": round(total_preds / max(1, len(images)), 2),
    }
    (args.output_dir / "sahi_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[sahi] done. meta saved to {args.output_dir / 'sahi_meta.json'}")


if __name__ == "__main__":
    main()
