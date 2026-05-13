from __future__ import annotations

from typing import Any

from .geometry import box_polygon, polygon_bounds, scale_polygon
from .schemas import BoxRecord


def letterbox_params(width: int, height: int, imgsz: int) -> dict[str, Any]:
    ratio = min(imgsz / float(width), imgsz / float(height))
    resized_w = int(round(width * ratio))
    resized_h = int(round(height * ratio))
    pad_w = imgsz - resized_w
    pad_h = imgsz - resized_h
    pad_left = pad_w / 2.0
    pad_top = pad_h / 2.0
    return {
        "ratio": ratio,
        "resized_width": resized_w,
        "resized_height": resized_h,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_w - pad_left,
        "pad_bottom": pad_h - pad_top,
        "input_width": imgsz,
        "input_height": imgsz,
    }


def scale_box_to_letterbox(box: BoxRecord, params: dict[str, Any]) -> tuple[float, float, float, float]:
    ratio = params["ratio"]
    pad_left = params["pad_left"]
    pad_top = params["pad_top"]
    x1 = box.x1 * ratio + pad_left
    y1 = box.y1 * ratio + pad_top
    x2 = box.x2 * ratio + pad_left
    y2 = box.y2 * ratio + pad_top
    return (x1, y1, x2, y2)


def scale_polygon_to_letterbox(box: BoxRecord, params: dict[str, Any]) -> list[tuple[float, float]]:
    polygon = box_polygon(box)
    return scale_polygon(polygon, ratio=float(params["ratio"]), pad_left=float(params["pad_left"]), pad_top=float(params["pad_top"]))


def scaled_box_bounds(box: BoxRecord, params: dict[str, Any]) -> tuple[float, float, float, float]:
    polygon = scale_polygon_to_letterbox(box, params)
    return polygon_bounds(polygon)


def box_metric(width: float, height: float, metric: str) -> float:
    area = width * height
    if metric == "area":
        return area
    if metric == "width":
        return width
    if metric == "height":
        return height
    if metric == "long_side":
        return max(width, height)
    if metric == "short_side":
        return min(width, height)
    return area ** 0.5


def assign_size_bucket(metric_value: float, small_thr: float, medium_thr: float) -> str:
    if metric_value < small_thr:
        return "small"
    if metric_value < medium_thr:
        return "medium"
    return "large"
