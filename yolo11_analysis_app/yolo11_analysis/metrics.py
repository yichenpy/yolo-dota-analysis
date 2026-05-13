from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .error_analysis import load_ground_truth
from .matching import box_iou, match_boxes
from .schemas import BoxRecord, DatasetContext
from .utils import ProgressCallback, emit_progress, safe_div

IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


def _collect_class_boxes(
    gt_by_image: dict[str, list[BoxRecord]],
    predictions: dict[str, list[BoxRecord]],
    class_id: int,
) -> tuple[dict[str, list[BoxRecord]], list[BoxRecord], int]:
    gt_class: dict[str, list[BoxRecord]] = {}
    pred_class: list[BoxRecord] = []
    total_gt = 0

    all_image_keys = set(gt_by_image) | set(predictions)
    for image_path in all_image_keys:
        gt_boxes = [box for box in gt_by_image.get(image_path, []) if box.class_id == class_id]
        pred_boxes = [box for box in predictions.get(image_path, []) if box.class_id == class_id]
        gt_class[image_path] = gt_boxes
        pred_class.extend(pred_boxes)
        total_gt += len(gt_boxes)

    pred_class.sort(key=lambda box: box.confidence if box.confidence is not None else 0.0, reverse=True)
    return gt_class, pred_class, total_gt


def compute_ap_curve(
    gt_by_image: dict[str, list[BoxRecord]],
    predictions: dict[str, list[BoxRecord]],
    class_id: int,
    iou_threshold: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    gt_class, pred_class, total_gt = _collect_class_boxes(gt_by_image, predictions, class_id)
    if total_gt == 0:
        return 0.0, np.asarray([]), np.asarray([])
    if not pred_class:
        return 0.0, np.asarray([0.0]), np.asarray([0.0])

    matched = {image_path: np.zeros(len(boxes), dtype=bool) for image_path, boxes in gt_class.items()}
    tp = np.zeros(len(pred_class), dtype=np.float64)
    fp = np.zeros(len(pred_class), dtype=np.float64)

    for index, pred_box in enumerate(pred_class):
        gt_boxes = gt_class.get(pred_box.image_path, [])
        best_gt_index = -1
        best_iou = 0.0
        for gt_index, gt_box in enumerate(gt_boxes):
            if matched[pred_box.image_path][gt_index]:
                continue
            iou_value = box_iou(gt_box, pred_box)
            if iou_value > best_iou:
                best_iou = iou_value
                best_gt_index = gt_index

        if best_gt_index >= 0 and best_iou >= iou_threshold:
            matched[pred_box.image_path][best_gt_index] = True
            tp[index] = 1.0
        else:
            fp[index] = 1.0

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / max(total_gt, 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    recall_points = np.linspace(0.0, 1.0, 101)
    precision_interp = np.interp(recall_points, mrec, mpre)
    ap = float(np.trapz(precision_interp, recall_points))
    return ap, recall, precision


def build_confusion_matrix(
    gt_by_image: dict[str, list[BoxRecord]],
    predictions: dict[str, list[BoxRecord]],
    num_classes: int,
    iou_threshold: float,
) -> np.ndarray:
    background = num_classes
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    all_image_keys = sorted(set(gt_by_image) | set(predictions))

    for image_path in all_image_keys:
        gt_boxes = gt_by_image.get(image_path, [])
        pred_boxes = predictions.get(image_path, [])
        matches, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes, iou_threshold, class_aware=False)
        for match in matches:
            gt_box = gt_boxes[match.gt_index]
            pred_box = pred_boxes[match.pred_index]
            matrix[gt_box.class_id, pred_box.class_id] += 1
        for gt_index in unmatched_gt:
            matrix[gt_boxes[gt_index].class_id, background] += 1
        for pred_index in unmatched_pred:
            matrix[background, pred_boxes[pred_index].class_id] += 1
    return matrix


def analyze_metrics(
    context: DatasetContext,
    predictions: dict[str, list[BoxRecord]],
    iou_threshold: float,
    gt_by_image: dict[str, list[BoxRecord]] | None = None,
    progress_callback: ProgressCallback = None,
) -> dict:
    if gt_by_image is None:
        gt_by_image = load_ground_truth(context, progress_callback=progress_callback, start=0.0, end=0.2)
    else:
        emit_progress(progress_callback, 0.2, "复用已有 GT 标注缓存")
    class_ids = sorted({box.class_id for boxes in gt_by_image.values() for box in boxes} | set(context.class_names.keys()))
    if not class_ids:
        class_ids = sorted({box.class_id for boxes in predictions.values() for box in boxes})

    per_class_counts: dict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    all_image_paths = sorted(set(gt_by_image) | set(predictions))
    for image_index, image_path in enumerate(all_image_paths, start=1):
        emit_progress(progress_callback, 0.2 + 0.2 * ((image_index - 1) / max(len(all_image_paths), 1)), f"统计类别 TP/FP/FN: {image_index - 1}/{len(all_image_paths)}")
        gt_boxes = gt_by_image.get(image_path, [])
        pred_boxes = predictions.get(image_path, [])
        matches, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes, iou_threshold, class_aware=True)
        for match in matches:
            class_id = gt_boxes[match.gt_index].class_id
            per_class_counts[class_id]["tp"] += 1
        for gt_index in unmatched_gt:
            class_id = gt_boxes[gt_index].class_id
            per_class_counts[class_id]["fn"] += 1
        for pred_index in unmatched_pred:
            class_id = pred_boxes[pred_index].class_id
            per_class_counts[class_id]["fp"] += 1

    class_rows: list[dict] = []
    pr_curves: dict[str, dict[str, list[float]]] = {}
    ap50_values: list[float] = []
    ap5095_values: list[float] = []

    for class_index, class_id in enumerate(class_ids, start=1):
        emit_progress(progress_callback, 0.4 + 0.4 * ((class_index - 1) / max(len(class_ids), 1)), f"计算每类别 AP: {class_index - 1}/{len(class_ids)}")
        class_name = context.class_names.get(class_id, str(class_id))
        tp_value = per_class_counts[class_id]["tp"]
        fp_value = per_class_counts[class_id]["fp"]
        fn_value = per_class_counts[class_id]["fn"]
        precision = safe_div(tp_value, tp_value + fp_value)
        recall = safe_div(tp_value, tp_value + fn_value)

        ap_values = []
        recall_curve = np.asarray([])
        precision_curve = np.asarray([])
        gt_class, pred_class, gt_count = _collect_class_boxes(gt_by_image, predictions, class_id)
        pred_count = len(pred_class)

        for threshold in IOU_THRESHOLDS:
            ap_value, class_recall_curve, class_precision_curve = compute_ap_curve(gt_by_image, predictions, class_id, float(threshold))
            ap_values.append(ap_value)
            if abs(float(threshold) - 0.5) < 1e-8:
                recall_curve = class_recall_curve
                precision_curve = class_precision_curve

        ap50 = ap_values[0] if ap_values else 0.0
        ap5095 = float(np.mean(ap_values)) if ap_values else 0.0
        if gt_count > 0:
            ap50_values.append(ap50)
            ap5095_values.append(ap5095)

        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "precision": precision,
                "recall": recall,
                "ap50": ap50,
                "ap50_95": ap5095,
            }
        )
        pr_curves[class_name] = {
            "recall": recall_curve.tolist(),
            "precision": precision_curve.tolist(),
        }

    emit_progress(progress_callback, 0.85, "构建混淆矩阵")
    class_df = pd.DataFrame(class_rows).sort_values("class_id") if class_rows else pd.DataFrame()

    total_tp = sum(item["tp"] for item in per_class_counts.values())
    total_fp = sum(item["fp"] for item in per_class_counts.values())
    total_fn = sum(item["fn"] for item in per_class_counts.values())
    uses_rotated_iou = any(box.meta.get("polygon") for boxes in gt_by_image.values() for box in boxes if isinstance(box.meta, dict))
    if not uses_rotated_iou:
        uses_rotated_iou = any(box.meta.get("polygon") for boxes in predictions.values() for box in boxes if isinstance(box.meta, dict))
    summary = {
        "precision": safe_div(total_tp, total_tp + total_fp),
        "recall": safe_div(total_tp, total_tp + total_fn),
        "mAP50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "mAP50_95": float(np.mean(ap5095_values)) if ap5095_values else 0.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "iou_mode": "rotated_polygon" if uses_rotated_iou else "axis_aligned_bbox",
    }

    confusion_matrix = build_confusion_matrix(
        gt_by_image=gt_by_image,
        predictions=predictions,
        num_classes=max(class_ids) + 1 if class_ids else 0,
        iou_threshold=iou_threshold,
    )
    emit_progress(progress_callback, 1.0, "指标分析完成")

    return {
        "summary": summary,
        "class_df": class_df,
        "pr_curves": pr_curves,
        "confusion_matrix": confusion_matrix,
        "confusion_labels": [context.class_names.get(class_id, str(class_id)) for class_id in range(confusion_matrix.shape[0] - 1)]
        + ["background"],
    }

