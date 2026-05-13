from __future__ import annotations

import math

import pandas as pd

from .utils import safe_div

DEFAULT_COVERAGE_MIN_CELLS = 2.0
DEFAULT_COVERAGE_MAX_CELLS = 8.0
DEFAULT_BOUNDARY_MARGIN = 0.35
DEFAULT_DETECTION_HEADS = [
    {"head_index": 0, "head_name": "P3/8", "stride": 8},
    {"head_index": 1, "head_name": "P4/16", "stride": 16},
    {"head_index": 2, "head_name": "P5/32", "stride": 32},
]


def _normalize_heads(summary: dict, imgsz: int | None) -> list[dict]:
    raw_heads = summary.get("detection_heads") or DEFAULT_DETECTION_HEADS
    normalized: list[dict] = []
    seen: set[int] = set()
    for index, item in enumerate(sorted(raw_heads, key=lambda row: int(row.get("stride", 0) or 0))):
        stride = int(item.get("stride", 0) or 0)
        if stride <= 0 or stride in seen:
            continue
        seen.add(stride)
        head_name = str(item.get("head_name") or f"P{index + 3}/{stride}")
        feature_map = f"{max(1, int(round(imgsz / stride)))}x{max(1, int(round(imgsz / stride)))}" if imgsz else f"1/{stride}"
        normalized.append(
            {
                "head_index": int(item.get("head_index", index)),
                "head_name": head_name,
                "head_stride": stride,
                "head_feature_map": feature_map,
                "head_label": f"{head_name} ({feature_map})",
            }
        )
    return normalized or [
        {
            "head_index": item["head_index"],
            "head_name": item["head_name"],
            "head_stride": item["stride"],
            "head_feature_map": f"{max(1, int(round(imgsz / item['stride'])))}x{max(1, int(round(imgsz / item['stride'])))}" if imgsz else f"1/{item['stride']}",
            "head_label": f"{item['head_name']} ({max(1, int(round(imgsz / item['stride'])))}x{max(1, int(round(imgsz / item['stride'])))} )" if imgsz else f"{item['head_name']} (1/{item['stride']})",
        }
        for item in DEFAULT_DETECTION_HEADS
    ]


def _head_score(cells_metric: float, target_cells: float) -> float:
    return abs(math.log(max(cells_metric, 1e-6) / max(target_cells, 1e-6)))


def _coverage_band(cells_metric: float, min_cells: float, max_cells: float) -> str:
    if cells_metric < min_cells:
        return f"<{min_cells:g} cells"
    if cells_metric <= max_cells:
        return f"{min_cells:g}-{max_cells:g} cells"
    return f">{max_cells:g} cells"


def _coverage_state(covered_head_count: int) -> str:
    if covered_head_count <= 0:
        return "无有效检测头"
    if covered_head_count == 1:
        return "单头覆盖"
    return "多头重叠"


def analyze_detection_heads(
    dataset_analysis: dict,
    *,
    coverage_min_cells: float = DEFAULT_COVERAGE_MIN_CELLS,
    coverage_max_cells: float = DEFAULT_COVERAGE_MAX_CELLS,
    boundary_margin: float = DEFAULT_BOUNDARY_MARGIN,
) -> dict:
    image_df = dataset_analysis.get("image_df", pd.DataFrame())
    box_df = dataset_analysis.get("box_df", pd.DataFrame()).copy()
    base_summary = dict(dataset_analysis.get("summary", {}))
    imgsz = int(image_df["imgsz"].iloc[0]) if isinstance(image_df, pd.DataFrame) and not image_df.empty and "imgsz" in image_df else None
    heads = _normalize_heads(base_summary, imgsz)
    target_cells = float(base_summary.get("detection_head_target_cells", 4.0))

    empty_result = {
        "per_box_df": pd.DataFrame(),
        "head_summary_df": pd.DataFrame(),
        "assigned_class_df": pd.DataFrame(),
        "coverage_class_df": pd.DataFrame(),
        "assigned_bucket_df": pd.DataFrame(),
        "coverage_band_df": pd.DataFrame(),
        "overlap_df": pd.DataFrame(),
        "range_miss_type_df": pd.DataFrame(),
        "boundary_df": pd.DataFrame(),
        "range_miss_df": pd.DataFrame(),
        "summary": {
            **base_summary,
            "coverage_min_cells": coverage_min_cells,
            "coverage_max_cells": coverage_max_cells,
            "boundary_margin": boundary_margin,
            "num_heads": len(heads),
            "total_targets": 0,
            "range_miss_ratio": 0.0,
            "multi_head_overlap_ratio": 0.0,
        },
    }
    if box_df.empty:
        return empty_result

    label_map = {}
    feature_map_map = {}
    if {"assigned_head_name", "assigned_head_label", "assigned_head_feature_map"}.issubset(box_df.columns):
        for _, row in box_df[["assigned_head_name", "assigned_head_label", "assigned_head_feature_map"]].drop_duplicates().iterrows():
            label_map[str(row["assigned_head_name"])] = str(row["assigned_head_label"])
            feature_map_map[str(row["assigned_head_name"])] = str(row["assigned_head_feature_map"])
    for head in heads:
        head_name = str(head["head_name"])
        if head_name in label_map:
            head["head_label"] = label_map[head_name]
        if head_name in feature_map_map:
            head["head_feature_map"] = feature_map_map[head_name]

    per_box_rows: list[dict] = []
    coverage_rows: list[dict] = []

    for _, row in box_df.iterrows():
        scaled_width = float(row.get("scaled_width", 0.0) or 0.0)
        scaled_height = float(row.get("scaled_height", 0.0) or 0.0)
        scaled_area = float(row.get("scaled_area", 0.0) or 0.0)
        scale_value = math.sqrt(max(scaled_area, 0.0))
        head_metrics: list[dict] = []
        for head in heads:
            stride = float(head["head_stride"])
            cells_metric = scale_value / stride if stride > 0 else 0.0
            short_cells = min(scaled_width, scaled_height) / stride if stride > 0 else 0.0
            long_cells = max(scaled_width, scaled_height) / stride if stride > 0 else 0.0
            score = _head_score(cells_metric, target_cells)
            head_metrics.append(
                {
                    "head_name": str(head["head_name"]),
                    "head_stride": int(head["head_stride"]),
                    "head_feature_map": str(head["head_feature_map"]),
                    "head_label": str(head["head_label"]),
                    "cells_metric": cells_metric,
                    "short_cells": short_cells,
                    "long_cells": long_cells,
                    "score": score,
                    "is_effective": coverage_min_cells <= cells_metric <= coverage_max_cells,
                    "coverage_band": _coverage_band(cells_metric, coverage_min_cells, coverage_max_cells),
                }
            )

        head_metrics.sort(key=lambda item: (item["score"], item["head_stride"]))
        assigned_head_name = str(row.get("assigned_head_name") or head_metrics[0]["head_name"])
        primary_metric = next((item for item in head_metrics if item["head_name"] == assigned_head_name), head_metrics[0])
        second_metric = next((item for item in head_metrics if item["head_name"] != primary_metric["head_name"]), None)
        margin = (second_metric["score"] - primary_metric["score"]) if second_metric is not None else float("inf")
        covered_metrics = [item for item in head_metrics if item["is_effective"]]
        covered_head_names = [item["head_name"] for item in covered_metrics]
        covered_head_labels = [item["head_label"] for item in covered_metrics]
        covered_head_count = len(covered_metrics)
        max_cells_metric = max(item["cells_metric"] for item in head_metrics)
        min_cells_metric = min(item["cells_metric"] for item in head_metrics)
        if covered_head_count == 0 and max_cells_metric < coverage_min_cells:
            range_miss_type = "目标过小，所有检测头 cells 偏少"
        elif covered_head_count == 0 and min_cells_metric > coverage_max_cells:
            range_miss_type = "目标过大，所有检测头 cells 偏多"
        elif covered_head_count == 0:
            range_miss_type = "落在检测头过渡带"
        else:
            range_miss_type = "有效检测头覆盖"

        per_box_rows.append(
            {
                "image_path": row.get("image_path"),
                "image_name": row.get("image_name"),
                "class_name": row.get("class_name"),
                "size_bucket": row.get("size_bucket"),
                "scaled_size_bucket": row.get("scaled_size_bucket"),
                "scaled_metric": float(row.get("scaled_metric", 0.0) or 0.0),
                "scaled_width": scaled_width,
                "scaled_height": scaled_height,
                "scaled_area": scaled_area,
                "assigned_head_name": primary_metric["head_name"],
                "assigned_head_stride": primary_metric["head_stride"],
                "assigned_head_feature_map": primary_metric["head_feature_map"],
                "assigned_head_label": primary_metric["head_label"],
                "primary_head_cells_metric": primary_metric["cells_metric"],
                "primary_head_short_cells": primary_metric["short_cells"],
                "primary_head_long_cells": primary_metric["long_cells"],
                "covered_head_count": covered_head_count,
                "covered_head_names": covered_head_names,
                "covered_head_labels": covered_head_labels,
                "coverage_state": _coverage_state(covered_head_count),
                "is_range_miss": covered_head_count == 0,
                "range_miss_type": range_miss_type,
                "best_head_margin": margin,
                "is_boundary": covered_head_count > 0 and margin < boundary_margin,
                "best_alternative_head_name": second_metric["head_name"] if second_metric is not None else None,
                "best_alternative_head_label": second_metric["head_label"] if second_metric is not None else None,
            }
        )

        for metric in covered_metrics:
            coverage_rows.append(
                {
                    "image_path": row.get("image_path"),
                    "image_name": row.get("image_name"),
                    "class_name": row.get("class_name"),
                    "size_bucket": row.get("size_bucket"),
                    "scaled_size_bucket": row.get("scaled_size_bucket"),
                    "assigned_head_name": primary_metric["head_name"],
                    "assigned_head_label": primary_metric["head_label"],
                    "covered_head_count": covered_head_count,
                    "is_boundary": covered_head_count > 0 and margin < boundary_margin,
                    "head_name": metric["head_name"],
                    "head_stride": metric["head_stride"],
                    "head_feature_map": metric["head_feature_map"],
                    "head_label": metric["head_label"],
                    "cells_metric": metric["cells_metric"],
                    "short_cells": metric["short_cells"],
                    "long_cells": metric["long_cells"],
                    "coverage_band": metric["coverage_band"],
                }
            )

    per_box_df = pd.DataFrame(per_box_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    total_targets = len(per_box_df)
    head_group = ["head_name", "head_stride", "head_feature_map", "head_label"]
    assigned_group = ["assigned_head_name", "assigned_head_stride", "assigned_head_feature_map", "assigned_head_label"]

    assigned_summary_df = (
        per_box_df.groupby(assigned_group)
        .agg(
            assigned_count=("assigned_head_name", "count"),
            median_scaled_metric=("scaled_metric", "median"),
            median_primary_cells=("primary_head_cells_metric", "median"),
            mean_primary_cells=("primary_head_cells_metric", "mean"),
            mean_short_cells=("primary_head_short_cells", "mean"),
            mean_long_cells=("primary_head_long_cells", "mean"),
            boundary_assigned_count=("is_boundary", "sum"),
            range_miss_assigned_count=("is_range_miss", "sum"),
            single_cover_assigned_count=("covered_head_count", lambda series: int((series == 1).sum())),
            multi_cover_assigned_count=("covered_head_count", lambda series: int((series > 1).sum())),
        )
        .reset_index()
        .rename(
            columns={
                "assigned_head_name": "head_name",
                "assigned_head_stride": "head_stride",
                "assigned_head_feature_map": "head_feature_map",
                "assigned_head_label": "head_label",
            }
        )
    )

    if coverage_df.empty:
        coverage_summary_df = pd.DataFrame(columns=head_group + ["effective_coverage_count", "effective_boundary_count", "exclusive_effective_count", "multi_effective_count", "median_effective_cells", "mean_effective_cells"])
    else:
        coverage_summary_df = (
            coverage_df.groupby(head_group)
            .agg(
                effective_coverage_count=("head_name", "count"),
                effective_boundary_count=("is_boundary", "sum"),
                exclusive_effective_count=("covered_head_count", lambda series: int((series == 1).sum())),
                multi_effective_count=("covered_head_count", lambda series: int((series > 1).sum())),
                median_effective_cells=("cells_metric", "median"),
                mean_effective_cells=("cells_metric", "mean"),
            )
            .reset_index()
        )

    all_heads_df = pd.DataFrame(heads).rename(columns={"head_name": "head_name", "head_stride": "head_stride", "head_feature_map": "head_feature_map", "head_label": "head_label"})
    head_summary_df = all_heads_df.merge(assigned_summary_df, on=head_group, how="left").merge(coverage_summary_df, on=head_group, how="left")
    for column in [
        "assigned_count",
        "median_scaled_metric",
        "median_primary_cells",
        "mean_primary_cells",
        "mean_short_cells",
        "mean_long_cells",
        "boundary_assigned_count",
        "range_miss_assigned_count",
        "single_cover_assigned_count",
        "multi_cover_assigned_count",
        "effective_coverage_count",
        "effective_boundary_count",
        "exclusive_effective_count",
        "multi_effective_count",
        "median_effective_cells",
        "mean_effective_cells",
    ]:
        if column in head_summary_df:
            head_summary_df[column] = head_summary_df[column].fillna(0.0)
    head_summary_df["assigned_count"] = head_summary_df["assigned_count"].astype(int)
    head_summary_df["boundary_assigned_count"] = head_summary_df["boundary_assigned_count"].astype(int)
    head_summary_df["range_miss_assigned_count"] = head_summary_df["range_miss_assigned_count"].astype(int)
    head_summary_df["single_cover_assigned_count"] = head_summary_df["single_cover_assigned_count"].astype(int)
    head_summary_df["multi_cover_assigned_count"] = head_summary_df["multi_cover_assigned_count"].astype(int)
    head_summary_df["effective_coverage_count"] = head_summary_df["effective_coverage_count"].astype(int)
    head_summary_df["effective_boundary_count"] = head_summary_df["effective_boundary_count"].astype(int)
    head_summary_df["exclusive_effective_count"] = head_summary_df["exclusive_effective_count"].astype(int)
    head_summary_df["multi_effective_count"] = head_summary_df["multi_effective_count"].astype(int)
    head_summary_df["assigned_ratio"] = head_summary_df["assigned_count"].apply(lambda value: safe_div(value, total_targets))
    head_summary_df["effective_coverage_ratio"] = head_summary_df["effective_coverage_count"].apply(lambda value: safe_div(value, total_targets))
    head_summary_df["exclusive_effective_ratio"] = head_summary_df["exclusive_effective_count"].apply(lambda value: safe_div(value, total_targets))
    head_summary_df["boundary_assigned_ratio"] = [safe_div(row["boundary_assigned_count"], row["assigned_count"]) for _, row in head_summary_df.iterrows()]
    head_summary_df["range_miss_assigned_ratio"] = [safe_div(row["range_miss_assigned_count"], row["assigned_count"]) for _, row in head_summary_df.iterrows()]
    head_summary_df["single_cover_assigned_ratio"] = [safe_div(row["single_cover_assigned_count"], row["assigned_count"]) for _, row in head_summary_df.iterrows()]
    head_summary_df["multi_cover_assigned_ratio"] = [safe_div(row["multi_cover_assigned_count"], row["assigned_count"]) for _, row in head_summary_df.iterrows()]
    head_summary_df["effective_multi_overlap_ratio"] = [safe_div(row["multi_effective_count"], row["effective_coverage_count"]) for _, row in head_summary_df.iterrows()]
    head_summary_df = head_summary_df.sort_values("head_stride").reset_index(drop=True)

    assigned_class_df = (
        per_box_df.groupby(["assigned_head_name", "assigned_head_stride", "assigned_head_label", "class_name"]).size().reset_index(name="assigned_count")
        .rename(columns={"assigned_head_name": "head_name", "assigned_head_stride": "head_stride", "assigned_head_label": "head_label"})
    )
    if not assigned_class_df.empty:
        head_totals = assigned_class_df.groupby("head_name")["assigned_count"].sum().rename("head_total").reset_index()
        class_totals = assigned_class_df.groupby("class_name")["assigned_count"].sum().rename("class_total").reset_index()
        assigned_class_df = assigned_class_df.merge(head_totals, on="head_name", how="left").merge(class_totals, on="class_name", how="left")
        assigned_class_df["head_ratio"] = assigned_class_df["assigned_count"] / assigned_class_df["head_total"]
        assigned_class_df["class_ratio"] = assigned_class_df["assigned_count"] / assigned_class_df["class_total"]
        assigned_class_df = assigned_class_df.sort_values(["head_stride", "assigned_count"], ascending=[True, False])

    if coverage_df.empty:
        coverage_class_df = pd.DataFrame(columns=["head_name", "head_stride", "head_label", "class_name", "coverage_count", "head_total", "class_total", "head_ratio", "class_ratio"])
        coverage_band_df = pd.DataFrame(columns=["head_name", "head_stride", "head_label", "coverage_band", "count"])
    else:
        coverage_class_df = coverage_df.groupby(["head_name", "head_stride", "head_label", "class_name"]).size().reset_index(name="coverage_count")
        head_totals = coverage_class_df.groupby("head_name")["coverage_count"].sum().rename("head_total").reset_index()
        class_totals = coverage_class_df.groupby("class_name")["coverage_count"].sum().rename("class_total").reset_index()
        coverage_class_df = coverage_class_df.merge(head_totals, on="head_name", how="left").merge(class_totals, on="class_name", how="left")
        coverage_class_df["head_ratio"] = coverage_class_df["coverage_count"] / coverage_class_df["head_total"]
        coverage_class_df["class_ratio"] = coverage_class_df["coverage_count"] / coverage_class_df["class_total"]
        coverage_class_df = coverage_class_df.sort_values(["head_stride", "coverage_count"], ascending=[True, False])
        coverage_band_df = coverage_df.groupby(["head_name", "head_stride", "head_label", "coverage_band"]).size().reset_index(name="count").sort_values(["head_stride", "count"], ascending=[True, False])

    assigned_bucket_df = per_box_df.groupby(["assigned_head_name", "assigned_head_stride", "assigned_head_label", "size_bucket"]).size().reset_index(name="count")
    assigned_bucket_df = assigned_bucket_df.rename(columns={"assigned_head_name": "head_name", "assigned_head_stride": "head_stride", "assigned_head_label": "head_label"}).sort_values(["head_stride", "count"], ascending=[True, False])

    overlap_df = per_box_df.groupby("covered_head_count").size().reset_index(name="count").sort_values("covered_head_count")
    overlap_df["ratio"] = overlap_df["count"].apply(lambda value: safe_div(value, total_targets))
    overlap_df["overlap_label"] = overlap_df["covered_head_count"].apply(lambda value: f"{int(value)} 个检测头")

    range_miss_type_df = per_box_df.groupby("range_miss_type").size().reset_index(name="count").sort_values("count", ascending=False)
    range_miss_type_df["ratio"] = range_miss_type_df["count"].apply(lambda value: safe_div(value, total_targets))

    boundary_df = per_box_df[per_box_df["is_boundary"]].copy().sort_values(["best_head_margin", "scaled_metric"], ascending=[True, False])
    range_miss_df = per_box_df[per_box_df["is_range_miss"]].copy().sort_values(["scaled_metric", "image_name"], ascending=[True, True])

    result_summary = {
        **base_summary,
        "coverage_min_cells": coverage_min_cells,
        "coverage_max_cells": coverage_max_cells,
        "boundary_margin": boundary_margin,
        "num_heads": len(heads),
        "total_targets": total_targets,
        "range_miss_ratio": safe_div(int(per_box_df["is_range_miss"].sum()), total_targets),
        "multi_head_overlap_ratio": safe_div(int((per_box_df["covered_head_count"] > 1).sum()), total_targets),
        "single_head_ratio": safe_div(int((per_box_df["covered_head_count"] == 1).sum()), total_targets),
    }

    return {
        "per_box_df": per_box_df,
        "head_summary_df": head_summary_df,
        "assigned_class_df": assigned_class_df,
        "coverage_class_df": coverage_class_df,
        "assigned_bucket_df": assigned_bucket_df,
        "coverage_band_df": coverage_band_df,
        "overlap_df": overlap_df,
        "range_miss_type_df": range_miss_type_df,
        "boundary_df": boundary_df,
        "range_miss_df": range_miss_df,
        "summary": result_summary,
    }
