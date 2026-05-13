from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from .geometry import box_metric_dimensions
from .io import image_to_label_path, read_image_size
from .labels import parse_yolo_label_file
from .matching import box_iou, match_boxes
from .preprocess import assign_size_bucket, box_metric
from .schemas import BoxRecord, DatasetContext
from .utils import ProgressCallback, emit_progress, safe_div


def _confidence_bin(confidence: float | None) -> str:
    if confidence is None:
        return "n/a"
    if confidence < 0.10:
        return "[0.00, 0.10)"
    if confidence < 0.25:
        return "[0.10, 0.25)"
    if confidence < 0.50:
        return "[0.25, 0.50)"
    if confidence < 0.75:
        return "[0.50, 0.75)"
    return "[0.75, 1.00]"


def _metric_from_box(box: BoxRecord, size_metric: str) -> tuple[float, str]:
    dims = box_metric_dimensions(box)
    if size_metric == "area":
        return dims["area"], assign_size_bucket(dims["area"], 0, float("inf"))
    return box_metric(dims["width"], dims["height"], size_metric), dims["aspect_ratio"]


def _box_metric_value(box: BoxRecord, size_metric: str) -> float:
    dims = box_metric_dimensions(box)
    if size_metric == "area":
        return dims["area"]
    if size_metric == "sqrt_area":
        return dims["area"] ** 0.5
    return box_metric(dims["width"], dims["height"], size_metric)


def _best_match_to_box(target_box: BoxRecord, candidate_boxes: list[BoxRecord]) -> tuple[BoxRecord | None, float]:
    best_box = None
    best_iou = 0.0
    for candidate in candidate_boxes:
        iou_value = box_iou(target_box, candidate)
        if iou_value > best_iou:
            best_iou = iou_value
            best_box = candidate
    return best_box, best_iou


def _fn_reason(gt_box: BoxRecord, pred_boxes: list[BoxRecord], iou_threshold: float) -> tuple[str, float, float, str | None]:
    same_class_preds = [pred for pred in pred_boxes if pred.class_id == gt_box.class_id]
    best_same_pred, best_same_iou = _best_match_to_box(gt_box, same_class_preds)
    best_any_pred, best_any_iou = _best_match_to_box(gt_box, pred_boxes)

    if best_any_pred is not None and best_any_iou >= iou_threshold and best_any_pred.class_id != gt_box.class_id:
        return "类别混淆", best_same_iou, best_any_iou, best_any_pred.class_name
    if best_same_iou > 0:
        return "定位不足", best_same_iou, best_any_iou, best_same_pred.class_name if best_same_pred is not None else None
    if best_any_iou > 0:
        return "候选偏移", best_same_iou, best_any_iou, best_any_pred.class_name if best_any_pred is not None else None
    return "未检出候选", best_same_iou, best_any_iou, None


def _fp_reason(pred_box: BoxRecord, gt_boxes: list[BoxRecord], iou_threshold: float) -> tuple[str, float, float, str | None]:
    same_class_gts = [gt for gt in gt_boxes if gt.class_id == pred_box.class_id]
    best_same_gt, best_same_iou = _best_match_to_box(pred_box, same_class_gts)
    best_any_gt, best_any_iou = _best_match_to_box(pred_box, gt_boxes)

    if best_any_gt is not None and best_any_iou >= iou_threshold and best_any_gt.class_id != pred_box.class_id:
        return "类别混淆", best_same_iou, best_any_iou, best_any_gt.class_name
    if best_same_iou > 0:
        return "定位偏差", best_same_iou, best_any_iou, best_same_gt.class_name if best_same_gt is not None else None
    if best_any_iou > 0:
        return "邻近背景误检", best_same_iou, best_any_iou, best_any_gt.class_name if best_any_gt is not None else None
    return "纯背景误检", best_same_iou, best_any_iou, None


def load_ground_truth(context: DatasetContext, progress_callback: ProgressCallback = None, start: float = 0.0, end: float = 1.0) -> dict[str, list[BoxRecord]]:
    gt_by_image: dict[str, list[BoxRecord]] = {}
    total_images = len(context.image_paths)
    span = max(0.0, end - start)
    for index, image_path in enumerate(context.image_paths, start=1):
        if total_images > 0:
            fraction = start + span * ((index - 1) / total_images)
            emit_progress(progress_callback, fraction, f"读取 GT 标注: {index - 1}/{total_images}")
        size_xy = read_image_size(image_path)
        label_path = image_to_label_path(image_path, context.image_dir, context.label_dir)
        gt_by_image[image_path] = parse_yolo_label_file(label_path, image_path, size_xy, context.class_names)
    emit_progress(progress_callback, end, f"读取 GT 标注完成: {total_images}/{total_images}")
    return gt_by_image


def analyze_errors(
    context: DatasetContext,
    predictions: dict[str, list[BoxRecord]],
    iou_threshold: float,
    size_metric: str,
    small_thr: float,
    medium_thr: float,
    progress_callback: ProgressCallback = None,
) -> dict:
    gt_by_image = load_ground_truth(context, progress_callback=progress_callback, start=0.0, end=0.25)
    image_rows: list[dict] = []
    detail_rows: list[dict] = []
    per_class_counter: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    image_cases: dict[str, dict] = {}
    total_images = len(context.image_paths)
    uses_rotated_iou = False

    for index, image_path in enumerate(context.image_paths, start=1):
        fraction = 0.25 + 0.75 * ((index - 1) / max(total_images, 1))
        emit_progress(progress_callback, fraction, f"匹配 TP/FP/FN: {index - 1}/{total_images}")
        gt_boxes = gt_by_image.get(image_path, [])
        pred_boxes = predictions.get(image_path, [])
        if not uses_rotated_iou:
            uses_rotated_iou = any(box.meta.get("polygon") for box in gt_boxes + pred_boxes if isinstance(box.meta, dict))
        matches, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes, iou_threshold, class_aware=True)

        tp_count = len(matches)
        fp_count = len(unmatched_pred)
        fn_count = len(unmatched_gt)
        image_rows.append(
            {
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "tp": tp_count,
                "fp": fp_count,
                "fn": fn_count,
                "error_total": fp_count + fn_count,
                "has_fn": fn_count > 0,
                "has_fp": fp_count > 0,
            }
        )

        tp_boxes: list[BoxRecord] = []
        fn_boxes: list[BoxRecord] = []
        fp_boxes: list[BoxRecord] = []

        for match in matches:
            gt_box = gt_boxes[match.gt_index]
            pred_box = pred_boxes[match.pred_index]
            metric_value = _box_metric_value(gt_box, size_metric)
            size_bucket = assign_size_bucket(metric_value, small_thr, medium_thr)
            tp_boxes.append(pred_box)
            per_class_counter[gt_box.class_name]["tp"] += 1
            detail_rows.append(
                {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "error_type": "TP",
                    "class_name": gt_box.class_name,
                    "class_id": gt_box.class_id,
                    "confidence": pred_box.confidence,
                    "confidence_bin": _confidence_bin(pred_box.confidence),
                    "iou": match.iou,
                    "metric_value": metric_value,
                    "size_bucket": size_bucket,
                    "reason": "命中",
                    "best_same_iou": match.iou,
                    "best_any_iou": match.iou,
                    "best_match_class": gt_box.class_name,
                    "gt_xyxy": gt_box.xyxy,
                    "pred_xyxy": pred_box.xyxy,
                    "gt_polygon": gt_box.meta.get("polygon") if isinstance(gt_box.meta, dict) else None,
                    "pred_polygon": pred_box.meta.get("polygon") if isinstance(pred_box.meta, dict) else None,
                }
            )

        for gt_index in unmatched_gt:
            gt_box = gt_boxes[gt_index]
            metric_value = _box_metric_value(gt_box, size_metric)
            size_bucket = assign_size_bucket(metric_value, small_thr, medium_thr)
            reason, best_same_iou, best_any_iou, best_match_class = _fn_reason(gt_box, pred_boxes, iou_threshold)
            fn_boxes.append(gt_box)
            per_class_counter[gt_box.class_name]["fn"] += 1
            detail_rows.append(
                {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "error_type": "FN",
                    "class_name": gt_box.class_name,
                    "class_id": gt_box.class_id,
                    "confidence": None,
                    "confidence_bin": "n/a",
                    "iou": 0.0,
                    "metric_value": metric_value,
                    "size_bucket": size_bucket,
                    "reason": reason,
                    "best_same_iou": best_same_iou,
                    "best_any_iou": best_any_iou,
                    "best_match_class": best_match_class,
                    "gt_xyxy": gt_box.xyxy,
                    "pred_xyxy": None,
                    "gt_polygon": gt_box.meta.get("polygon") if isinstance(gt_box.meta, dict) else None,
                    "pred_polygon": None,
                }
            )

        for pred_index in unmatched_pred:
            pred_box = pred_boxes[pred_index]
            metric_value = _box_metric_value(pred_box, size_metric)
            size_bucket = assign_size_bucket(metric_value, small_thr, medium_thr)
            reason, best_same_iou, best_any_iou, best_match_class = _fp_reason(pred_box, gt_boxes, iou_threshold)
            fp_boxes.append(pred_box)
            per_class_counter[pred_box.class_name]["fp"] += 1
            detail_rows.append(
                {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "error_type": "FP",
                    "class_name": pred_box.class_name,
                    "class_id": pred_box.class_id,
                    "confidence": pred_box.confidence,
                    "confidence_bin": _confidence_bin(pred_box.confidence),
                    "iou": 0.0,
                    "metric_value": metric_value,
                    "size_bucket": size_bucket,
                    "reason": reason,
                    "best_same_iou": best_same_iou,
                    "best_any_iou": best_any_iou,
                    "best_match_class": best_match_class,
                    "gt_xyxy": None,
                    "pred_xyxy": pred_box.xyxy,
                    "gt_polygon": None,
                    "pred_polygon": pred_box.meta.get("polygon") if isinstance(pred_box.meta, dict) else None,
                }
            )

        image_cases[image_path] = {
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes,
            "tp_boxes": tp_boxes,
            "fn_boxes": fn_boxes,
            "fp_boxes": fp_boxes,
            "matches": matches,
        }

    emit_progress(progress_callback, 1.0, f"漏检/虚检分析完成: {total_images}/{total_images}")
    per_class_rows: list[dict] = []
    for class_name, counts in sorted(per_class_counter.items()):
        tp_value = counts["tp"]
        fp_value = counts["fp"]
        fn_value = counts["fn"]
        per_class_rows.append(
            {
                "class_name": class_name,
                "tp": tp_value,
                "fp": fp_value,
                "fn": fn_value,
                "miss_rate": safe_div(fn_value, tp_value + fn_value),
                "false_detection_rate": safe_div(fp_value, tp_value + fp_value),
            }
        )

    image_df = pd.DataFrame(image_rows)
    detail_df = pd.DataFrame(detail_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    fn_df = detail_df[detail_df["error_type"] == "FN"].copy() if not detail_df.empty else pd.DataFrame()
    fp_df = detail_df[detail_df["error_type"] == "FP"].copy() if not detail_df.empty else pd.DataFrame()

    summary = {
        "total_tp": int(image_df["tp"].sum()) if not image_df.empty else 0,
        "total_fp": int(image_df["fp"].sum()) if not image_df.empty else 0,
        "total_fn": int(image_df["fn"].sum()) if not image_df.empty else 0,
        "num_images": len(image_df),
        "images_with_fn": int((image_df["fn"] > 0).sum()) if not image_df.empty else 0,
        "images_with_fp": int((image_df["fp"] > 0).sum()) if not image_df.empty else 0,
        "iou_mode": "rotated_polygon" if uses_rotated_iou else "axis_aligned_bbox",
    }

    fn_reason_df = fn_df.groupby("reason").size().reset_index(name="count") if not fn_df.empty else pd.DataFrame(columns=["reason", "count"])
    fn_size_df = fn_df.groupby("size_bucket").size().reset_index(name="count") if not fn_df.empty else pd.DataFrame(columns=["size_bucket", "count"])
    fn_class_df = per_class_df[["class_name", "tp", "fn", "miss_rate"]].sort_values("fn", ascending=False) if not per_class_df.empty else pd.DataFrame()

    fp_reason_df = fp_df.groupby("reason").size().reset_index(name="count") if not fp_df.empty else pd.DataFrame(columns=["reason", "count"])
    fp_conf_df = fp_df.groupby("confidence_bin").size().reset_index(name="count") if not fp_df.empty else pd.DataFrame(columns=["confidence_bin", "count"])
    fp_class_df = per_class_df[["class_name", "tp", "fp", "false_detection_rate"]].sort_values("fp", ascending=False) if not per_class_df.empty else pd.DataFrame()

    hard_images_df = image_df.sort_values(["error_total", "fn", "fp"], ascending=[False, False, False]) if not image_df.empty else pd.DataFrame()

    return {
        "gt_by_image": gt_by_image,
        "image_df": image_df,
        "detail_df": detail_df,
        "per_class_df": per_class_df,
        "image_cases": image_cases,
        "summary": summary,
        "fn_df": fn_df,
        "fp_df": fp_df,
        "fn_reason_df": fn_reason_df,
        "fn_size_df": fn_size_df,
        "fn_class_df": fn_class_df,
        "fp_reason_df": fp_reason_df,
        "fp_conf_df": fp_conf_df,
        "fp_class_df": fp_class_df,
        "hard_images_df": hard_images_df,
    }
