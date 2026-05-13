from __future__ import annotations

from .geometry import box_iou
from .schemas import BoxRecord, MatchPair


def match_boxes(
    gt_boxes: list[BoxRecord],
    pred_boxes: list[BoxRecord],
    iou_threshold: float,
    *,
    class_aware: bool = True,
) -> tuple[list[MatchPair], list[int], list[int]]:
    remaining_gt = set(range(len(gt_boxes)))
    matched_pairs: list[MatchPair] = []
    unmatched_pred: list[int] = []
    pred_order = sorted(
        range(len(pred_boxes)),
        key=lambda index: pred_boxes[index].confidence if pred_boxes[index].confidence is not None else 0.0,
        reverse=True,
    )

    for pred_index in pred_order:
        pred_box = pred_boxes[pred_index]
        best_gt_index = None
        best_iou = 0.0

        for gt_index in remaining_gt:
            gt_box = gt_boxes[gt_index]
            if class_aware and gt_box.class_id != pred_box.class_id:
                continue
            iou_value = box_iou(gt_box, pred_box)
            if iou_value > best_iou:
                best_iou = iou_value
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            remaining_gt.remove(best_gt_index)
            matched_pairs.append(MatchPair(gt_index=best_gt_index, pred_index=pred_index, iou=best_iou))
        else:
            unmatched_pred.append(pred_index)

    unmatched_gt = sorted(remaining_gt)
    return matched_pairs, unmatched_gt, unmatched_pred
