from __future__ import annotations

import math
from typing import Iterable

from .schemas import BoxRecord


Point = tuple[float, float]
EPS = 1e-9


def _pairwise_polygon(values: Iterable[float]) -> list[Point]:
    coords = list(values)
    if len(coords) < 6 or len(coords) % 2 != 0:
        return []
    return [(float(coords[index]), float(coords[index + 1])) for index in range(0, len(coords), 2)]


def rectangle_polygon(box: BoxRecord) -> list[Point]:
    return [
        (float(box.x1), float(box.y1)),
        (float(box.x2), float(box.y1)),
        (float(box.x2), float(box.y2)),
        (float(box.x1), float(box.y2)),
    ]


def box_polygon(box: BoxRecord, *, fallback_bbox: bool = True) -> list[Point]:
    polygon = box.meta.get("polygon") if isinstance(box.meta, dict) else None
    points = _pairwise_polygon(polygon) if polygon is not None else []
    if points:
        return points
    return rectangle_polygon(box) if fallback_bbox else []


def polygon_flatten(points: list[Point]) -> list[float]:
    flattened: list[float] = []
    for x_value, y_value in points:
        flattened.extend([float(x_value), float(y_value)])
    return flattened


def polygon_signed_area(points: list[Point]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def polygon_area(points: list[Point]) -> float:
    return abs(polygon_signed_area(points))


def normalize_polygon(points: list[Point]) -> list[Point]:
    if len(points) < 3:
        return points
    return points if polygon_signed_area(points) >= 0 else list(reversed(points))


def polygon_bounds(points: list[Point]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_edge_dimensions(points: list[Point]) -> tuple[float, float]:
    if len(points) < 4:
        left, top, right, bottom = polygon_bounds(points)
        return max(0.0, right - left), max(0.0, bottom - top)
    distances = []
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        distances.append(math.dist(point, next_point))
    even_edges = distances[0::2] or distances
    odd_edges = distances[1::2] or distances
    width = sum(even_edges) / len(even_edges)
    height = sum(odd_edges) / len(odd_edges)
    return float(width), float(height)


def box_area(box: BoxRecord) -> float:
    points = box_polygon(box, fallback_bbox=False)
    if points:
        return polygon_area(points)
    return box.area


def box_metric_dimensions(box: BoxRecord) -> dict[str, float]:
    points = box_polygon(box, fallback_bbox=False)
    if points:
        width, height = polygon_edge_dimensions(points)
        area = polygon_area(points)
    else:
        width = box.width
        height = box.height
        area = box.area
    long_side = max(width, height)
    short_side = min(width, height)
    aspect_ratio = long_side / short_side if short_side > 0 else 0.0
    return {
        "width": float(width),
        "height": float(height),
        "area": float(area),
        "long_side": float(long_side),
        "short_side": float(short_side),
        "aspect_ratio": float(aspect_ratio),
    }


def scale_polygon(points: list[Point], *, ratio: float, pad_left: float, pad_top: float) -> list[Point]:
    return [(point[0] * ratio + pad_left, point[1] * ratio + pad_top) for point in points]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _inside(point: Point, edge_start: Point, edge_end: Point, orientation: float) -> bool:
    return orientation * _cross(edge_start, edge_end, point) >= -EPS


def _line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Point:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < EPS:
        return p2
    determinant1 = x1 * y2 - y1 * x2
    determinant2 = x3 * y4 - y3 * x4
    x_value = (determinant1 * (x3 - x4) - (x1 - x2) * determinant2) / denominator
    y_value = (determinant1 * (y3 - y4) - (y1 - y2) * determinant2) / denominator
    return float(x_value), float(y_value)


def polygon_intersection(subject_polygon: list[Point], clip_polygon: list[Point]) -> list[Point]:
    subject = normalize_polygon(subject_polygon)
    clip = normalize_polygon(clip_polygon)
    if len(subject) < 3 or len(clip) < 3:
        return []

    output = subject
    clip_orientation = 1.0 if polygon_signed_area(clip) >= 0 else -1.0
    for clip_index, clip_end in enumerate(clip):
        clip_start = clip[clip_index - 1]
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if _inside(e, clip_start, clip_end, clip_orientation):
                if not _inside(s, clip_start, clip_end, clip_orientation):
                    output.append(_line_intersection(s, e, clip_start, clip_end))
                output.append(e)
            elif _inside(s, clip_start, clip_end, clip_orientation):
                output.append(_line_intersection(s, e, clip_start, clip_end))
            s = e
    return output


def polygon_iou(points_a: list[Point], points_b: list[Point]) -> float:
    if len(points_a) < 3 or len(points_b) < 3:
        return 0.0
    area_a = polygon_area(points_a)
    area_b = polygon_area(points_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    intersection_polygon = polygon_intersection(points_a, points_b)
    intersection_area = polygon_area(intersection_polygon)
    if intersection_area <= 0:
        return 0.0
    union_area = area_a + area_b - intersection_area
    if union_area <= 0:
        return 0.0
    return float(intersection_area / union_area)


def box_iou(box_a: BoxRecord, box_b: BoxRecord) -> float:
    polygon_a = box_polygon(box_a)
    polygon_b = box_polygon(box_b)
    return polygon_iou(polygon_a, polygon_b)
