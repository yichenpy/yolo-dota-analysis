from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
import yaml
from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SIZE_METRICS = ("area", "equivalent_side", "long_edge", "short_edge")
_OFFICIAL_IOU_WARNED = False


class AnalysisConfigurationError(RuntimeError):
    """Raised when the dataset or model configuration is not usable."""


@dataclass
class OBBInstance:
    class_id: int
    polygon: np.ndarray
    confidence: Optional[float] = None
    image_path: Optional[Path] = None
    source: str = ""

    def metric_value(self, metric: str) -> float:
        return metric_value_from_polygon(self.polygon, metric)

    @property
    def area(self) -> float:
        return polygon_metrics(self.polygon)["area"]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def maybe_import_ultralytics_dataset_loader():
    try:
        from ultralytics.data.utils import check_det_dataset  # type: ignore
    except Exception:
        return None
    return check_det_dataset


def maybe_import_ultralytics_obb_ops():
    try:
        import torch  # type: ignore
        from ultralytics.utils.metrics import batch_probiou  # type: ignore
        from ultralytics.utils.ops import xyxyxyxy2xywhr  # type: ignore
    except Exception:
        return None
    return torch, batch_probiou, xyxyxyxy2xywhr


def normalize_class_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {idx: str(value) for idx, value in enumerate(raw_names)}
    return {}


def resolve_existing_path(candidate: Path) -> Path:
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def resolve_path(value: Any, primary_root: Path, secondary_root: Path) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return resolve_existing_path(candidate)
    primary = resolve_existing_path(primary_root / candidate)
    if primary.exists():
        return primary
    secondary = resolve_existing_path(secondary_root / candidate)
    if secondary.exists():
        return secondary
    return primary


def load_dataset_config(dataset_yaml: Path) -> dict[str, Any]:
    dataset_yaml = resolve_existing_path(Path(dataset_yaml).expanduser())
    yaml_dir = dataset_yaml.parent
    raw_data = load_yaml(dataset_yaml)

    checked = maybe_import_ultralytics_dataset_loader()
    if checked is not None:
        try:
            raw_data = checked(str(dataset_yaml))
        except Exception as exc:
            warnings.warn(
                f"Ultralytics check_det_dataset failed for {dataset_yaml}: {exc}. "
                "Falling back to plain YAML parsing.",
                RuntimeWarning,
            )

    dataset_root = raw_data.get("path")
    if dataset_root:
        dataset_root = resolve_path(dataset_root, yaml_dir, yaml_dir)
    else:
        dataset_root = yaml_dir

    data = dict(raw_data)
    data["path"] = dataset_root
    data["_yaml_path"] = dataset_yaml
    data["_yaml_dir"] = yaml_dir
    data["names"] = normalize_class_names(data.get("names", {}))
    return data


def _iter_source_entries(split_spec: Any) -> list[Any]:
    if split_spec is None:
        return []
    if isinstance(split_spec, list):
        return split_spec
    return [split_spec]


def resolve_split_sources(dataset_config: dict[str, Any], split: str) -> list[Path]:
    if split not in dataset_config:
        raise AnalysisConfigurationError(
            f"Split '{split}' is not present in dataset yaml {dataset_config['_yaml_path']}."
        )

    yaml_dir = Path(dataset_config["_yaml_dir"])
    dataset_root = Path(dataset_config["path"])
    sources: list[Path] = []
    for entry in _iter_source_entries(dataset_config.get(split)):
        sources.append(resolve_path(entry, dataset_root, yaml_dir))
    return sources


def _iter_images_from_list_file(list_file: Path, dataset_root: Path) -> Iterator[Path]:
    with list_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(line).expanduser()
            if candidate.is_absolute():
                yield resolve_existing_path(candidate)
                continue
            relative_to_list = resolve_existing_path(list_file.parent / candidate)
            if relative_to_list.exists():
                yield relative_to_list
                continue
            yield resolve_existing_path(dataset_root / candidate)


def iter_image_paths(dataset_config: dict[str, Any], split: str) -> Iterator[Path]:
    dataset_root = Path(dataset_config["path"])
    for source in resolve_split_sources(dataset_config, split):
        if source.is_dir():
            for image_path in sorted(source.rglob("*")):
                if image_path.suffix.lower() in IMAGE_SUFFIXES and image_path.is_file():
                    yield image_path
            continue
        if source.is_file() and source.suffix.lower() == ".txt":
            yield from _iter_images_from_list_file(source, dataset_root)
            continue
        if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            yield source
            continue
        warnings.warn(
            f"Skipping unsupported split source {source}. Expected directory, image path, or .txt list file.",
            RuntimeWarning,
        )


def collect_image_paths(
    dataset_config: dict[str, Any],
    split: str,
    max_images: Optional[int] = None,
) -> list[Path]:
    images: list[Path] = []
    for image_path in iter_image_paths(dataset_config, split):
        images.append(image_path)
        if max_images is not None and len(images) >= max_images:
            break
    return images


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def label_path_for_image(image_path: Path, label_dirname: str = "labels") -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = label_dirname
            return Path(*parts).with_suffix(".txt")
    if image_path.parent.name != label_dirname:
        return image_path.with_suffix(".txt")
    return image_path


def normalize_polygon(points: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    center = array.mean(axis=0)
    angles = np.arctan2(array[:, 1] - center[1], array[:, 0] - center[0])
    order = np.argsort(angles)
    array = array[order]
    if signed_polygon_area(array) < 0:
        array = array[::-1]
    return array


def signed_polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(points: np.ndarray) -> float:
    return abs(signed_polygon_area(normalize_polygon(points)))


def polygon_metrics(points: np.ndarray) -> dict[str, float]:
    ordered = normalize_polygon(points)
    edges = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    opposite_a = 0.5 * (edges[0] + edges[2])
    opposite_b = 0.5 * (edges[1] + edges[3])
    long_edge = float(max(opposite_a, opposite_b))
    short_edge = float(max(min(opposite_a, opposite_b), 1e-9))
    area = polygon_area(ordered)
    return {
        "area": area,
        "equivalent_side": float(math.sqrt(max(area, 0.0))),
        "long_edge": long_edge,
        "short_edge": short_edge,
        "aspect_ratio": float(long_edge / short_edge),
    }


def metric_value_from_polygon(points: np.ndarray, metric: str) -> float:
    metrics = polygon_metrics(points)
    if metric not in metrics:
        raise KeyError(f"Unsupported size metric '{metric}'.")
    return float(metrics[metric])


def _cross_z(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _line_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    r = p2 - p1
    s = q2 - q1
    denominator = _cross_z(r, s)
    if abs(denominator) < 1e-9:
        return (p2 + q1) * 0.5
    t = _cross_z(q1 - p1, s) / denominator
    return p1 + t * r


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    return _cross_z(edge_end - edge_start, point - edge_start) >= -1e-9


def convex_polygon_intersection(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    output = normalize_polygon(subject)
    clipper = normalize_polygon(clipper)
    for clip_index in range(len(clipper)):
        edge_start = clipper[clip_index]
        edge_end = clipper[(clip_index + 1) % len(clipper)]
        input_points = output
        if len(input_points) == 0:
            break
        output_points: list[np.ndarray] = []
        prev_point = input_points[-1]
        for curr_point in input_points:
            curr_inside = _inside(curr_point, edge_start, edge_end)
            prev_inside = _inside(prev_point, edge_start, edge_end)
            if curr_inside:
                if not prev_inside:
                    output_points.append(_line_intersection(prev_point, curr_point, edge_start, edge_end))
                output_points.append(curr_point)
            elif prev_inside:
                output_points.append(_line_intersection(prev_point, curr_point, edge_start, edge_end))
            prev_point = curr_point
        output = np.asarray(output_points, dtype=np.float64)
    if len(output) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return normalize_polygon(output)


def polygon_iou(points_a: np.ndarray, points_b: np.ndarray) -> float:
    points_a = normalize_polygon(points_a)
    points_b = normalize_polygon(points_b)
    intersection = convex_polygon_intersection(points_a, points_b)
    intersection_area = polygon_area(intersection) if len(intersection) >= 3 else 0.0
    union = polygon_area(points_a) + polygon_area(points_b) - intersection_area
    if union <= 0:
        return 0.0
    return float(intersection_area / union)


def scale_polygon_if_needed(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scaled = np.asarray(points, dtype=np.float64).copy()
    if scaled.size == 0:
        return scaled
    if np.max(np.abs(scaled)) <= 1.5:
        scaled[:, 0] *= width
        scaled[:, 1] *= height
    return scaled


def parse_label_line(
    tokens: Sequence[str],
    image_size_xy: tuple[int, int],
    layout: str,
) -> OBBInstance:
    width, height = image_size_xy
    if layout == "gt":
        if len(tokens) != 9:
            raise ValueError(f"Expected 9 values for GT OBB labels, received {len(tokens)}.")
        class_id = int(float(tokens[0]))
        coords = np.asarray([float(value) for value in tokens[1:9]], dtype=np.float64).reshape(4, 2)
        coords = scale_polygon_if_needed(coords, width, height)
        return OBBInstance(class_id=class_id, polygon=normalize_polygon(coords))

    if layout == "class_xyxyxyxy_conf":
        if len(tokens) < 10:
            raise ValueError(f"Expected at least 10 values for prediction layout {layout}, received {len(tokens)}.")
        class_id = int(float(tokens[0]))
        coords = np.asarray([float(value) for value in tokens[1:9]], dtype=np.float64).reshape(4, 2)
        confidence = float(tokens[9])
    elif layout == "class_conf_xyxyxyxy":
        if len(tokens) < 10:
            raise ValueError(f"Expected at least 10 values for prediction layout {layout}, received {len(tokens)}.")
        class_id = int(float(tokens[0]))
        confidence = float(tokens[1])
        coords = np.asarray([float(value) for value in tokens[2:10]], dtype=np.float64).reshape(4, 2)
    else:
        raise KeyError(f"Unsupported prediction text layout '{layout}'.")

    coords = scale_polygon_if_needed(coords, width, height)
    return OBBInstance(
        class_id=class_id,
        polygon=normalize_polygon(coords),
        confidence=confidence,
    )


def read_label_file(
    label_path: Path,
    image_size_xy: tuple[int, int],
    layout: str,
) -> list[OBBInstance]:
    instances: list[OBBInstance] = []
    if not label_path.exists():
        return instances

    with label_path.open("r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            try:
                instances.append(parse_label_line(tokens, image_size_xy, layout))
            except Exception as exc:
                raise ValueError(f"Failed to parse {label_path}:{line_index}: {exc}") from exc
    return instances


def _warn_official_iou_fallback(reason: str) -> None:
    global _OFFICIAL_IOU_WARNED
    if _OFFICIAL_IOU_WARNED:
        return
    warnings.warn(
        f"Official Ultralytics OBB IoU is unavailable, falling back to custom polygon IoU. Reason: {reason}",
        RuntimeWarning,
    )
    _OFFICIAL_IOU_WARNED = True



def _instances_to_xywhr(instances: Sequence[OBBInstance], torch_module: Any, xyxyxyxy2xywhr: Any) -> Any:
    if not instances:
        return torch_module.zeros((0, 5), dtype=torch_module.float32)
    polygons = np.stack([normalize_polygon(instance.polygon).reshape(-1) for instance in instances]).astype(np.float32)
    polygon_tensor = torch_module.from_numpy(polygons)
    return xyxyxyxy2xywhr(polygon_tensor)



def official_obb_iou_matrix(
    ground_truth: Sequence[OBBInstance],
    predictions: Sequence[OBBInstance],
) -> Optional[np.ndarray]:
    if not ground_truth or not predictions:
        return np.zeros((len(ground_truth), len(predictions)), dtype=np.float64)

    imported = maybe_import_ultralytics_obb_ops()
    if imported is None:
        return None

    torch_module, batch_probiou, xyxyxyxy2xywhr = imported
    try:
        gt_boxes = _instances_to_xywhr(ground_truth, torch_module, xyxyxyxy2xywhr)
        pred_boxes = _instances_to_xywhr(predictions, torch_module, xyxyxyxy2xywhr)
        iou = batch_probiou(gt_boxes, pred_boxes)
        return iou.detach().cpu().numpy().astype(np.float64)
    except Exception as exc:
        _warn_official_iou_fallback(str(exc))
        return None



def obb_iou(instance_a: OBBInstance, instance_b: OBBInstance) -> float:
    iou_matrix = official_obb_iou_matrix([instance_a], [instance_b])
    if iou_matrix is not None and iou_matrix.size:
        return float(iou_matrix[0, 0])
    return polygon_iou(instance_a.polygon, instance_b.polygon)



def match_detections(
    ground_truth: Sequence[OBBInstance],
    predictions: Sequence[OBBInstance],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    iou_matrix = official_obb_iou_matrix(ground_truth, predictions)
    if iou_matrix is not None:
        gt_classes = np.asarray([instance.class_id for instance in ground_truth], dtype=np.int64)
        pred_classes = np.asarray([instance.class_id for instance in predictions], dtype=np.int64)
        if gt_classes.size and pred_classes.size:
            iou_matrix = iou_matrix * (gt_classes[:, None] == pred_classes[None, :])
        matches = np.argwhere(iou_matrix >= iou_threshold)
        if matches.size:
            scores = iou_matrix[matches[:, 0], matches[:, 1]]
            order = np.argsort(scores)[::-1]
            matches = np.concatenate([matches[order], scores[order, None]], axis=1)
            if matches.shape[0] > 1:
                matches = matches[np.unique(matches[:, 1].astype(int), return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0].astype(int), return_index=True)[1]]
            matched_rows = [
                (int(gt_index), int(pred_index), float(score))
                for gt_index, pred_index, score in matches.tolist()
            ]
            used_gt = {gt_index for gt_index, _, _ in matched_rows}
            used_pred = {pred_index for _, pred_index, _ in matched_rows}
        else:
            matched_rows = []
            used_gt = set()
            used_pred = set()
        unmatched_gt = [index for index in range(len(ground_truth)) if index not in used_gt]
        unmatched_pred = [index for index in range(len(predictions)) if index not in used_pred]
        return matched_rows, unmatched_gt, unmatched_pred

    candidate_pairs: list[tuple[float, int, int]] = []
    for gt_index, gt_instance in enumerate(ground_truth):
        for pred_index, pred_instance in enumerate(predictions):
            if gt_instance.class_id != pred_instance.class_id:
                continue
            iou = polygon_iou(gt_instance.polygon, pred_instance.polygon)
            if iou >= iou_threshold:
                candidate_pairs.append((iou, gt_index, pred_index))

    candidate_pairs.sort(key=lambda item: item[0], reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gt_index, pred_index in candidate_pairs:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))

    unmatched_gt = [index for index in range(len(ground_truth)) if index not in used_gt]
    unmatched_pred = [index for index in range(len(predictions)) if index not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def summarize_numeric(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    summary = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }
    for quantile in (0.05, 0.1, 0.25, 0.75, 0.9, 0.95):
        summary[f"q{int(quantile * 100):02d}"] = float(np.quantile(array, quantile))
    return summary


def safe_name(class_id: int, names: dict[int, str]) -> str:
    return names.get(class_id, f"class_{class_id}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


