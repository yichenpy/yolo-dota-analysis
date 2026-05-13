from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .geometry import box_metric_dimensions, polygon_area, polygon_bounds, polygon_edge_dimensions
from .io import image_to_label_path, read_image_size
from .labels import parse_yolo_label_file
from .preprocess import assign_size_bucket, box_metric, letterbox_params, scale_polygon_to_letterbox
from .schemas import DatasetContext
from .utils import ProgressCallback, emit_progress, summarize_numeric


DEFAULT_DETECTION_HEADS = {
    "source": "default",
    "heads": [
        {"head_index": 0, "head_name": "P3/8", "stride": 8},
        {"head_index": 1, "head_name": "P4/16", "stride": 16},
        {"head_index": 2, "head_name": "P5/32", "stride": 32},
    ],
}


def _normalize_detection_heads(detection_head_info: dict | None) -> tuple[list[dict], str, str | None]:
    payload = detection_head_info or DEFAULT_DETECTION_HEADS
    raw_heads = payload.get("heads") or DEFAULT_DETECTION_HEADS["heads"]
    source = str(payload.get("source", "default"))
    warning = payload.get("warning")

    heads: list[dict] = []
    seen_strides: set[int] = set()
    for index, item in enumerate(sorted(raw_heads, key=lambda row: int(row.get("stride", 0)))):
        stride = int(item.get("stride", 0))
        if stride <= 0 or stride in seen_strides:
            continue
        seen_strides.add(stride)
        heads.append(
            {
                "head_index": int(item.get("head_index", index)),
                "head_name": str(item.get("head_name") or f"P{index + 3}/{stride}"),
                "stride": stride,
            }
        )

    if not heads:
        heads = DEFAULT_DETECTION_HEADS["heads"]
        source = "default"
    return heads, source, warning


def _head_cells_band(cells_metric: float) -> str:
    if cells_metric < 2.0:
        return "<2 cells"
    if cells_metric < 4.0:
        return "2-4 cells"
    if cells_metric < 8.0:
        return "4-8 cells"
    return ">=8 cells"


def _assign_detection_head(
    scaled_width: float,
    scaled_height: float,
    scaled_area: float,
    imgsz: int,
    detection_heads: list[dict],
    target_cells: float,
) -> dict:
    scale_value = math.sqrt(max(scaled_area, 0.0))
    best_head = detection_heads[0]
    best_score = float("inf")

    for head in detection_heads:
        stride = float(head["stride"])
        cells_metric = scale_value / stride if stride else 0.0
        score = abs(math.log(max(cells_metric, 1e-6) / max(target_cells, 1e-6)))
        if score < best_score:
            best_score = score
            best_head = head

    stride = float(best_head["stride"])
    short_side = min(scaled_width, scaled_height)
    long_side = max(scaled_width, scaled_height)
    cells_metric = scale_value / stride if stride else 0.0
    short_cells = short_side / stride if stride else 0.0
    long_cells = long_side / stride if stride else 0.0
    area_cells = scaled_area / (stride * stride) if stride else 0.0
    feature_map_size = max(1, int(round(imgsz / stride))) if stride else imgsz
    return {
        "head_name": str(best_head["head_name"]),
        "head_stride": int(best_head["stride"]),
        "head_feature_map": f"{feature_map_size}x{feature_map_size}",
        "head_label": f"{best_head['head_name']} ({feature_map_size}x{feature_map_size})",
        "head_cells_metric": cells_metric,
        "head_short_cells": short_cells,
        "head_long_cells": long_cells,
        "head_area_cells": area_cells,
        "head_cells_band": _head_cells_band(cells_metric),
    }


def analyze_dataset(
    context: DatasetContext,
    imgsz: int,
    size_metric: str,
    small_thr: float,
    medium_thr: float,
    detection_head_info: dict | None = None,
    head_target_cells: float = 4.0,
    progress_callback: ProgressCallback = None,
) -> dict:
    detection_heads, head_source, head_warning = _normalize_detection_heads(detection_head_info)
    image_rows: list[dict] = []
    box_rows: list[dict] = []
    missing_labels: list[str] = []
    total_images = len(context.image_paths)

    for index, image_path in enumerate(context.image_paths, start=1):
        emit_progress(progress_callback, (index - 1) / max(total_images, 1), f"数据集分析: {index - 1}/{total_images}")
        width, height = read_image_size(image_path)
        params = letterbox_params(width, height, imgsz)
        image_rows.append(
            {
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height else 0.0,
                "imgsz": imgsz,
                "resize_ratio": params["ratio"],
                "resized_width": params["resized_width"],
                "resized_height": params["resized_height"],
                "pad_left": params["pad_left"],
                "pad_top": params["pad_top"],
            }
        )

        label_path = image_to_label_path(image_path, context.image_dir, context.label_dir)
        if not label_path.exists():
            missing_labels.append(str(label_path))
            continue

        boxes = parse_yolo_label_file(label_path, image_path, (width, height), context.class_names)
        for box in boxes:
            orig_dims = box_metric_dimensions(box)
            scaled_polygon = scale_polygon_to_letterbox(box, params)
            scaled_left, scaled_top, scaled_right, scaled_bottom = polygon_bounds(scaled_polygon)
            scaled_width_metric, scaled_height_metric = polygon_edge_dimensions(scaled_polygon)
            scaled_area = polygon_area(scaled_polygon)
            metric_value = box_metric(orig_dims["width"], orig_dims["height"], size_metric) if size_metric not in {"area", "sqrt_area"} else (orig_dims["area"] if size_metric == "area" else orig_dims["area"] ** 0.5)
            scaled_metric_value = box_metric(scaled_width_metric, scaled_height_metric, size_metric) if size_metric not in {"area", "sqrt_area"} else (scaled_area if size_metric == "area" else scaled_area ** 0.5)
            head_assignment = _assign_detection_head(scaled_width_metric, scaled_height_metric, scaled_area, imgsz, detection_heads, head_target_cells)
            box_rows.append(
                {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "class_id": box.class_id,
                    "class_name": box.class_name,
                    "orig_width": orig_dims["width"],
                    "orig_height": orig_dims["height"],
                    "orig_area": orig_dims["area"],
                    "orig_aspect_ratio": orig_dims["aspect_ratio"],
                    "scaled_width": scaled_width_metric,
                    "scaled_height": scaled_height_metric,
                    "scaled_area": scaled_area,
                    "scaled_aspect_ratio": max(scaled_width_metric, scaled_height_metric) / max(min(scaled_width_metric, scaled_height_metric), 1e-9) if scaled_width_metric > 0 and scaled_height_metric > 0 else 0.0,
                    "orig_metric": metric_value,
                    "scaled_metric": scaled_metric_value,
                    "size_bucket": assign_size_bucket(metric_value, small_thr, medium_thr),
                    "scaled_size_bucket": assign_size_bucket(scaled_metric_value, small_thr, medium_thr),
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                    "scaled_x1": scaled_left,
                    "scaled_y1": scaled_top,
                    "scaled_x2": scaled_right,
                    "scaled_y2": scaled_bottom,
                    "assigned_head_name": head_assignment["head_name"],
                    "assigned_head_stride": head_assignment["head_stride"],
                    "assigned_head_feature_map": head_assignment["head_feature_map"],
                    "assigned_head_label": head_assignment["head_label"],
                    "assigned_head_cells_metric": head_assignment["head_cells_metric"],
                    "assigned_head_short_cells": head_assignment["head_short_cells"],
                    "assigned_head_long_cells": head_assignment["head_long_cells"],
                    "assigned_head_area_cells": head_assignment["head_area_cells"],
                    "assigned_head_band": head_assignment["head_cells_band"],
                    "polygon": box.meta.get("polygon") if isinstance(box.meta, dict) else None,
                    "scaled_polygon": [coord for point in scaled_polygon for coord in point],
                }
            )

    emit_progress(progress_callback, 1.0, f"数据集分析完成: {total_images}/{total_images}")
    image_df = pd.DataFrame(image_rows)
    box_df = pd.DataFrame(box_rows)

    if not image_df.empty:
        object_count_df = box_df.groupby(["image_path", "image_name"]).size().reset_index(name="object_count") if not box_df.empty else pd.DataFrame(columns=["image_path", "image_name", "object_count"])
        image_df = image_df.merge(object_count_df[["image_path", "object_count"]], on="image_path", how="left")
        image_df["object_count"] = image_df["object_count"].fillna(0).astype(int)
    else:
        image_df["object_count"] = []

    summary = {
        "dataset_name": context.dataset_name,
        "num_images": len(image_df),
        "num_boxes": len(box_df),
        "missing_label_files": missing_labels,
        "image_width_stats": summarize_numeric(image_df["width"]) if not image_df.empty else {"count": 0},
        "image_height_stats": summarize_numeric(image_df["height"]) if not image_df.empty else {"count": 0},
        "image_aspect_ratio_stats": summarize_numeric(image_df["aspect_ratio"]) if not image_df.empty else {"count": 0},
        "objects_per_image_stats": summarize_numeric(image_df["object_count"]) if not image_df.empty else {"count": 0},
        "box_width_stats": summarize_numeric(box_df["orig_width"]) if not box_df.empty else {"count": 0},
        "box_height_stats": summarize_numeric(box_df["orig_height"]) if not box_df.empty else {"count": 0},
        "box_area_stats": summarize_numeric(box_df["orig_area"]) if not box_df.empty else {"count": 0},
        "box_metric_stats": summarize_numeric(box_df["orig_metric"]) if not box_df.empty else {"count": 0},
        "scaled_metric_stats": summarize_numeric(box_df["scaled_metric"]) if not box_df.empty else {"count": 0},
        "detection_head_source": head_source,
        "detection_head_warning": head_warning,
        "detection_head_target_cells": head_target_cells,
        "detection_heads": detection_heads,
    }

    per_class_df = (
        box_df.groupby("class_name")
        .agg(
            instances=("class_name", "count"),
            mean_orig_width=("orig_width", "mean"),
            mean_orig_height=("orig_height", "mean"),
            mean_orig_area=("orig_area", "mean"),
            median_orig_metric=("orig_metric", "median"),
            mean_scaled_metric=("scaled_metric", "mean"),
            mean_aspect_ratio=("orig_aspect_ratio", "mean"),
            small_ratio=("size_bucket", lambda series: (series == "small").mean()),
            medium_ratio=("size_bucket", lambda series: (series == "medium").mean()),
            large_ratio=("size_bucket", lambda series: (series == "large").mean()),
        )
        .reset_index()
        if not box_df.empty
        else pd.DataFrame()
    )

    size_bucket_df = (
        box_df.groupby("size_bucket").size().reset_index(name="count")
        if not box_df.empty
        else pd.DataFrame(columns=["size_bucket", "count"])
    )

    per_class_bucket_df = (
        box_df.groupby(["class_name", "size_bucket"]).size().reset_index(name="count")
        if not box_df.empty
        else pd.DataFrame(columns=["class_name", "size_bucket", "count"])
    )

    if not box_df.empty:
        head_summary_df = (
            box_df.groupby(["assigned_head_name", "assigned_head_stride", "assigned_head_feature_map", "assigned_head_label"])
            .agg(
                assigned_count=("assigned_head_name", "count"),
                mean_scaled_metric=("scaled_metric", "mean"),
                median_scaled_metric=("scaled_metric", "median"),
                mean_cells_metric=("assigned_head_cells_metric", "mean"),
                median_cells_metric=("assigned_head_cells_metric", "median"),
                mean_short_cells=("assigned_head_short_cells", "mean"),
                mean_long_cells=("assigned_head_long_cells", "mean"),
                under_2_cells_ratio=("assigned_head_cells_metric", lambda series: (series < 2.0).mean()),
                cells_2_4_ratio=("assigned_head_cells_metric", lambda series: ((series >= 2.0) & (series < 4.0)).mean()),
                cells_4_8_ratio=("assigned_head_cells_metric", lambda series: ((series >= 4.0) & (series < 8.0)).mean()),
                over_8_cells_ratio=("assigned_head_cells_metric", lambda series: (series >= 8.0).mean()),
                small_ratio=("size_bucket", lambda series: (series == "small").mean()),
                medium_ratio=("size_bucket", lambda series: (series == "medium").mean()),
                large_ratio=("size_bucket", lambda series: (series == "large").mean()),
            )
            .reset_index()
            .sort_values("assigned_head_stride")
        )
        head_summary_df["assigned_ratio"] = head_summary_df["assigned_count"] / len(box_df)

        head_class_df = box_df.groupby(["assigned_head_name", "assigned_head_stride", "assigned_head_label", "class_name"]).size().reset_index(name="count")
        head_totals = head_class_df.groupby("assigned_head_name")["count"].sum().rename("head_total").reset_index()
        class_totals = head_class_df.groupby("class_name")["count"].sum().rename("class_total").reset_index()
        head_class_df = head_class_df.merge(head_totals, on="assigned_head_name", how="left").merge(class_totals, on="class_name", how="left")
        head_class_df["head_ratio"] = head_class_df["count"] / head_class_df["head_total"]
        head_class_df["class_ratio"] = head_class_df["count"] / head_class_df["class_total"]
        head_class_df = head_class_df.sort_values(["assigned_head_stride", "count"], ascending=[True, False])

        head_bucket_df = box_df.groupby(["assigned_head_name", "assigned_head_stride", "assigned_head_label", "size_bucket"]).size().reset_index(name="count")
        head_bucket_df = head_bucket_df.sort_values(["assigned_head_stride", "count"], ascending=[True, False])

        head_band_df = box_df.groupby(["assigned_head_name", "assigned_head_stride", "assigned_head_label", "assigned_head_band"]).size().reset_index(name="count")
        head_band_df = head_band_df.sort_values(["assigned_head_stride", "count"], ascending=[True, False])
    else:
        head_summary_df = pd.DataFrame(columns=[
            "assigned_head_name",
            "assigned_head_stride",
            "assigned_head_feature_map",
            "assigned_head_label",
            "assigned_count",
            "assigned_ratio",
            "mean_scaled_metric",
            "median_scaled_metric",
            "mean_cells_metric",
            "median_cells_metric",
            "mean_short_cells",
            "mean_long_cells",
            "under_2_cells_ratio",
            "cells_2_4_ratio",
            "cells_4_8_ratio",
            "over_8_cells_ratio",
            "small_ratio",
            "medium_ratio",
            "large_ratio",
        ])
        head_class_df = pd.DataFrame(columns=["assigned_head_name", "assigned_head_stride", "assigned_head_label", "class_name", "count", "head_total", "class_total", "head_ratio", "class_ratio"])
        head_bucket_df = pd.DataFrame(columns=["assigned_head_name", "assigned_head_stride", "assigned_head_label", "size_bucket", "count"])
        head_band_df = pd.DataFrame(columns=["assigned_head_name", "assigned_head_stride", "assigned_head_label", "assigned_head_band", "count"])

    return {
        "image_df": image_df,
        "box_df": box_df,
        "per_class_df": per_class_df,
        "size_bucket_df": size_bucket_df,
        "per_class_bucket_df": per_class_bucket_df,
        "head_summary_df": head_summary_df,
        "head_class_df": head_class_df,
        "head_bucket_df": head_bucket_df,
        "head_band_df": head_band_df,
        "summary": summary,
    }
