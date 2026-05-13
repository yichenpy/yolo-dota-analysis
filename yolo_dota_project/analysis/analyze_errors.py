from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from common_obb import (
        AnalysisConfigurationError,
        OBBInstance,
        collect_image_paths,
        ensure_directory,
        image_size,
        label_path_for_image,
        load_dataset_config,
        match_detections,
        metric_value_from_polygon,
        normalize_polygon,
        obb_iou,
        polygon_iou,
        read_label_file,
        safe_name,
        scale_polygon_if_needed,
        write_json,
    )
except ImportError:
    from analysis.common_obb import (
        AnalysisConfigurationError,
        OBBInstance,
        collect_image_paths,
        ensure_directory,
        image_size,
        label_path_for_image,
        load_dataset_config,
        match_detections,
        metric_value_from_polygon,
        normalize_polygon,
        obb_iou,
        polygon_iou,
        read_label_file,
        safe_name,
        scale_polygon_if_needed,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze OBB missed detections and false detections on a YOLO validation split.",
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Path to data.yaml.")
    parser.add_argument("--split", default="val", help="Dataset split to analyze. Default: val.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Prediction result path. Supports a directory of .txt files or a .json file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional YOLO OBB weights. If provided and --predictions is omitted, the script runs inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/outputs/errors"),
        help="Directory for plots and summary files.",
    )
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for matching.")
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.001,
        help="Confidence threshold for prediction filtering.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size when --weights is used.")
    parser.add_argument("--batch", type=int, default=4, help="Inference batch size when --weights is used.")
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0 or cpu.")
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference when supported by the server GPU.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections kept per image during inference.",
    )
    parser.add_argument(
        "--prediction-format",
        choices=("auto", "txt", "json"),
        default="auto",
        help="How to interpret --predictions.",
    )
    parser.add_argument(
        "--prediction-layout",
        choices=("class_xyxyxyxy_conf", "class_conf_xyxyxyxy"),
        default="class_xyxyxyxy_conf",
        help="Text prediction layout used by .txt result files.",
    )
    parser.add_argument(
        "--size-metric",
        choices=("area", "equivalent_side", "long_edge", "short_edge"),
        default="equivalent_side",
        help="OBB size definition used for size statistics.",
    )
    parser.add_argument(
        "--size-bin-strategy",
        choices=("quantile", "linear"),
        default="quantile",
        help="Binning strategy for missed-size charts.",
    )
    parser.add_argument("--size-bins", type=int, default=10, help="Number of bins for missed-size plots.")
    parser.add_argument("--label-dirname", default="labels", help="Label directory name. Default: labels.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="CPU worker count for GT loading and per-image error matching.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for debugging on the server.",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save per-instance missed/false rows to CSV.",
    )
    parser.add_argument(
        "--skip-official-val",
        action="store_true",
        help=(
            "Skip a separate official Ultralytics val() pass. "
            "Without this flag, official comparable metrics are recorded when --weights is used."
        ),
    )
    parser.add_argument(
        "--official-conf",
        type=float,
        default=None,
        help="Confidence passed to official Ultralytics val(). Default: None.",
    )
    parser.add_argument(
        "--official-iou",
        type=float,
        default=0.7,
        help="NMS IoU passed to official Ultralytics val(). Default: 0.7.",
    )
    parser.add_argument(
        "--official-max-det",
        type=int,
        default=300,
        help="max_det passed to official Ultralytics val(). Default: 300.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def ensure_ultralytics_available() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is required when using --weights. Install it in the server environment, "
            "for example: pip install ultralytics"
        ) from exc
    return YOLO


def progress_iter(
    iterable: Iterable[Any],
    *,
    total: Optional[int] = None,
    desc: str = "",
    disable: bool = False,
) -> Iterable[Any]:
    if disable:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)


def create_progress_bar(
    total: int,
    *,
    desc: str,
    disable: bool,
) -> Any:
    if disable:
        return None
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)


def import_torch_if_available() -> Any:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def clear_cuda_cache(torch_module: Any) -> None:
    if torch_module is None:
        return
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
    except Exception:
        pass


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def extract_official_metrics_payload(metrics: Any, run_kwargs: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "metrics_class": metrics.__class__.__name__,
        "run_kwargs": to_jsonable(run_kwargs),
    }

    for attr_name in ("results_dict", "speed", "save_dir", "task", "names"):
        attr_value = getattr(metrics, attr_name, None)
        if attr_value is not None:
            payload[attr_name] = to_jsonable(attr_value)

    metric_head = getattr(metrics, "box", None) or getattr(metrics, "obb", None)
    if metric_head is not None:
        head_payload: dict[str, Any] = {}
        for attr_name in ("mp", "mr", "map50", "map", "maps", "p", "r", "f1"):
            attr_value = getattr(metric_head, attr_name, None)
            if attr_value is not None:
                head_payload[attr_name] = to_jsonable(attr_value)
        if head_payload:
            payload["metric_head"] = head_payload

    if hasattr(metrics, "mean_results"):
        try:
            payload["mean_results"] = to_jsonable(metrics.mean_results())
        except Exception:
            pass
    return payload


def run_official_ultralytics_val(args: argparse.Namespace, output_dir: Path) -> Optional[dict[str, Any]]:
    if args.weights is None or args.skip_official_val:
        return None

    YOLO = ensure_ultralytics_available()
    model = YOLO(str(args.weights))
    run_kwargs: dict[str, Any] = {
        "data": str(args.dataset.expanduser().resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "half": args.half,
        "conf": args.official_conf,
        "iou": args.official_iou,
        "max_det": args.official_max_det,
        "plots": False,
        "verbose": False,
        "save_json": False,
        "project": str(output_dir),
        "name": "official_val",
        "exist_ok": True,
    }

    try:
        metrics = model.val(**run_kwargs)
    except TypeError as exc:
        warnings.warn(
            "model.val() rejected one or more optional arguments. Retrying with a smaller argument set. "
            f"Original error: {exc}",
            RuntimeWarning,
        )
        fallback_kwargs = {
            key: value
            for key, value in run_kwargs.items()
            if key not in {"project", "name", "exist_ok", "save_json", "max_det", "conf"}
        }
        metrics = model.val(**fallback_kwargs)
        run_kwargs = fallback_kwargs
    except Exception as exc:
        warnings.warn(
            "Official Ultralytics validation failed. The custom error breakdown will still be generated. "
            f"Error: {exc}",
            RuntimeWarning,
        )
        payload = {
            "status": "failed",
            "error": str(exc),
            "run_kwargs": to_jsonable(run_kwargs),
        }
        write_json(output_dir / "official_val_metrics.json", payload)
        return payload

    payload = extract_official_metrics_payload(metrics, run_kwargs)
    write_json(output_dir / "official_val_metrics.json", payload)
    return payload


def _load_single_image_gt(task: tuple[Path, str]) -> tuple[Path, tuple[int, int], list[OBBInstance]]:
    image_path, label_dirname = task
    size_xy = image_size(image_path)
    label_path = label_path_for_image(image_path, label_dirname=label_dirname)
    instances = read_label_file(label_path, size_xy, layout="gt")
    for instance in instances:
        instance.image_path = image_path
        instance.source = "ground_truth"
    return image_path, size_xy, instances


def _match_single_image(task: tuple[Path, list[OBBInstance], list[OBBInstance], float, str, dict[int, str]]) -> dict[str, Any]:
    image_path, gt_instances, pred_instances, iou_thres, size_metric, names = task
    total_gt = len(gt_instances)
    total_pred = len(pred_instances)
    total_matches = 0
    all_gt_sizes: list[float] = []
    missed_sizes: list[float] = []
    missed_by_class: Counter[str] = Counter()
    false_by_class: Counter[str] = Counter()
    miss_reason_counter: Counter[str] = Counter()
    false_reason_counter: Counter[str] = Counter()
    missed_rows: list[dict[str, Any]] = []
    false_rows: list[dict[str, Any]] = []

    for gt_instance in gt_instances:
        all_gt_sizes.append(metric_value_from_polygon(gt_instance.polygon, size_metric))

    matches, unmatched_gt_indices, unmatched_pred_indices = match_detections(
        ground_truth=gt_instances,
        predictions=pred_instances,
        iou_threshold=iou_thres,
    )
    total_matches += len(matches)

    for gt_index in unmatched_gt_indices:
        gt_instance = gt_instances[gt_index]
        best_same, best_other = max_iou_by_class(gt_instance, pred_instances)
        reason = classify_miss_reason(best_same, best_other, iou_thres)
        size_value = metric_value_from_polygon(gt_instance.polygon, size_metric)
        class_name = safe_name(gt_instance.class_id, names)
        missed_by_class[class_name] += 1
        miss_reason_counter[reason] += 1
        missed_sizes.append(size_value)
        missed_rows.append(
            {
                "image": str(image_path),
                "class_id": gt_instance.class_id,
                "class_name": class_name,
                f"{size_metric}_px": round(size_value, 4),
                "reason": reason,
                "best_same_class_iou": round(best_same, 4),
                "best_other_class_iou": round(best_other, 4),
            }
        )

    for pred_index in unmatched_pred_indices:
        pred_instance = pred_instances[pred_index]
        best_same, best_other = max_iou_by_class(pred_instance, gt_instances)
        reason = classify_false_reason(best_same, best_other, iou_thres)
        class_name = safe_name(pred_instance.class_id, names)
        false_by_class[class_name] += 1
        false_reason_counter[reason] += 1
        false_rows.append(
            {
                "image": str(image_path),
                "class_id": pred_instance.class_id,
                "class_name": class_name,
                "confidence": round(float(pred_instance.confidence or 0.0), 4),
                "reason": reason,
                "best_same_class_iou": round(best_same, 4),
                "best_other_class_iou": round(best_other, 4),
            }
        )

    return {
        "image_path": image_path,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_matches": total_matches,
        "all_gt_sizes": all_gt_sizes,
        "missed_sizes": missed_sizes,
        "missed_by_class": dict(missed_by_class),
        "false_by_class": dict(false_by_class),
        "missed_by_reason": dict(miss_reason_counter),
        "false_by_reason": dict(false_reason_counter),
        "missed_rows": missed_rows,
        "false_rows": false_rows,
    }


def load_ground_truth(
    dataset_config: dict[str, Any],
    split: str,
    label_dirname: str,
    max_images: Optional[int],
    workers: int,
    show_progress: bool,
) -> tuple[list[Path], dict[Path, list[OBBInstance]], dict[Path, tuple[int, int]]]:
    images = collect_image_paths(dataset_config, split, max_images=max_images)
    if not images:
        raise AnalysisConfigurationError(
            f"No images were found for split '{split}'. Check {dataset_config['_yaml_path']} and server paths."
        )

    gt_map: dict[Path, list[OBBInstance]] = {}
    image_sizes: dict[Path, tuple[int, int]] = {}
    if workers <= 1:
        for image_path in progress_iter(
            images,
            total=len(images),
            desc="Loading GT",
            disable=not show_progress,
        ):
            loaded_image_path, size_xy, instances = _load_single_image_gt((image_path, label_dirname))
            image_sizes[loaded_image_path] = size_xy
            gt_map[loaded_image_path] = instances
        return images, gt_map, image_sizes

    progress_bar = create_progress_bar(
        len(images),
        desc="Loading GT",
        disable=not show_progress,
    )
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_load_single_image_gt, (image_path, label_dirname)) for image_path in images]
            for future in as_completed(futures):
                loaded_image_path, size_xy, instances = future.result()
                image_sizes[loaded_image_path] = size_xy
                gt_map[loaded_image_path] = instances
                if progress_bar is not None:
                    progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return images, gt_map, image_sizes


def build_image_lookup(image_paths: Iterable[Path]) -> dict[str, Any]:
    image_paths = [path.resolve() for path in image_paths]
    common_root = Path(os.path.commonpath([str(path) for path in image_paths])) if image_paths else None
    exact: dict[str, Path] = {}
    relative_without_suffix: dict[str, Path] = {}
    stem_lookup: dict[str, list[Path]] = defaultdict(list)

    for image_path in image_paths:
        exact[str(image_path)] = image_path
        exact[image_path.name] = image_path
        stem_lookup[image_path.stem].append(image_path)
        if common_root is not None:
            try:
                rel = image_path.relative_to(common_root).with_suffix("").as_posix()
                relative_without_suffix[rel] = image_path
            except ValueError:
                pass

    return {
        "common_root": common_root,
        "exact": exact,
        "relative_without_suffix": relative_without_suffix,
        "stem_lookup": stem_lookup,
    }


def resolve_prediction_key(raw_key: str, lookup: dict[str, Any]) -> Optional[Path]:
    candidate = raw_key.replace("\\", "/").strip()
    if not candidate:
        return None

    if candidate in lookup["exact"]:
        return lookup["exact"][candidate]

    normalized_path = Path(candidate)
    if normalized_path.is_absolute():
        resolved = str(normalized_path.resolve())
        if resolved in lookup["exact"]:
            return lookup["exact"][resolved]

    candidate_without_suffix = Path(candidate).with_suffix("").as_posix()
    if candidate_without_suffix in lookup["relative_without_suffix"]:
        return lookup["relative_without_suffix"][candidate_without_suffix]

    filename = Path(candidate).name
    if filename in lookup["exact"]:
        return lookup["exact"][filename]

    stem = Path(candidate).stem
    stem_matches = lookup["stem_lookup"].get(stem, [])
    if len(stem_matches) == 1:
        return stem_matches[0]
    if len(stem_matches) > 1:
        warnings.warn(
            f"Prediction key '{raw_key}' matches multiple images by stem. "
            "Use unique filenames or nested prediction paths that mirror the image layout.",
            RuntimeWarning,
        )
    return None


def load_predictions_from_text_dir(
    prediction_dir: Path,
    image_paths: list[Path],
    image_sizes: dict[Path, tuple[int, int]],
    prediction_layout: str,
    conf_threshold: float,
    show_progress: bool,
) -> dict[Path, list[OBBInstance]]:
    lookup = build_image_lookup(image_paths)
    prediction_map: dict[Path, list[OBBInstance]] = {image_path: [] for image_path in image_paths}
    text_files = sorted(prediction_dir.rglob("*.txt"))
    if not text_files:
        warnings.warn(f"No .txt prediction files were found under {prediction_dir}.", RuntimeWarning)

    for text_file in progress_iter(
        text_files,
        total=len(text_files),
        desc="Reading predictions",
        disable=not show_progress,
    ):
        rel_stem = text_file.relative_to(prediction_dir).with_suffix("").as_posix()
        image_path = (
            resolve_prediction_key(rel_stem, lookup)
            or resolve_prediction_key(text_file.stem, lookup)
            or resolve_prediction_key(text_file.name, lookup)
        )
        if image_path is None:
            warnings.warn(
                f"Could not map prediction file {text_file} to any validation image. Skipping it.",
                RuntimeWarning,
            )
            continue

        predictions = read_label_file(text_file, image_sizes[image_path], layout=prediction_layout)
        filtered: list[OBBInstance] = []
        for instance in predictions:
            if instance.confidence is not None and instance.confidence < conf_threshold:
                continue
            instance.image_path = image_path
            instance.source = str(text_file)
            filtered.append(instance)
        prediction_map[image_path] = filtered
    return prediction_map


def flatten_polygon(raw_polygon: Any) -> np.ndarray:
    if isinstance(raw_polygon, list) and raw_polygon and isinstance(raw_polygon[0], list):
        if len(raw_polygon) == 1 and len(raw_polygon[0]) == 8:
            raw_polygon = raw_polygon[0]
        elif len(raw_polygon) == 4 and len(raw_polygon[0]) == 2:
            return np.asarray(raw_polygon, dtype=np.float64)
    array = np.asarray(raw_polygon, dtype=np.float64).reshape(-1)
    if array.size != 8:
        raise ValueError(f"Expected 8 numeric polygon values, received {array.size}.")
    return array.reshape(4, 2)


def load_predictions_from_json(
    prediction_file: Path,
    image_paths: list[Path],
    image_sizes: dict[Path, tuple[int, int]],
    conf_threshold: float,
    show_progress: bool,
) -> dict[Path, list[OBBInstance]]:
    lookup = build_image_lookup(image_paths)
    prediction_map: dict[Path, list[OBBInstance]] = {image_path: [] for image_path in image_paths}
    with prediction_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    items: list[dict[str, Any]] = []
    if isinstance(payload, list):
        items = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("predictions"), list):
            items = [item for item in payload["predictions"] if isinstance(item, dict)]
        elif isinstance(payload.get("annotations"), list):
            items = [item for item in payload["annotations"] if isinstance(item, dict)]
        else:
            for image_key, value in payload.items():
                if isinstance(value, list):
                    for item in value:
                        if not isinstance(item, dict):
                            continue
                        item = dict(item)
                        item.setdefault("image", image_key)
                        items.append(item)

    for item in progress_iter(
        items,
        total=len(items),
        desc="Reading predictions",
        disable=not show_progress,
    ):
        raw_image_key = item.get("image") or item.get("image_id") or item.get("file_name")
        if raw_image_key is None:
            continue
        image_path = resolve_prediction_key(str(raw_image_key), lookup)
        if image_path is None:
            continue

        class_id = item.get("class_id", item.get("category_id"))
        if class_id is None:
            raise ValueError(
                f"Prediction item for image {raw_image_key} does not contain class_id/category_id."
            )
        confidence = float(item.get("confidence", item.get("score", 1.0)))
        if confidence < conf_threshold:
            continue

        raw_polygon = (
            item.get("points")
            or item.get("polygon")
            or item.get("obb")
            or item.get("segmentation")
        )
        if raw_polygon is None:
            raise ValueError(
                f"Prediction item for image {raw_image_key} does not contain points/polygon/obb/segmentation."
            )

        polygon = flatten_polygon(raw_polygon)
        width, height = image_sizes[image_path]
        polygon = scale_polygon_if_needed(polygon, width, height)
        prediction_map[image_path].append(
            OBBInstance(
                class_id=int(class_id),
                polygon=normalize_polygon(polygon),
                confidence=confidence,
                image_path=image_path,
                source=str(prediction_file),
            )
        )
    return prediction_map


def load_predictions_from_model(
    weights: Path,
    image_paths: list[Path],
    conf_threshold: float,
    imgsz: int,
    batch: int,
    device: Optional[str],
    half: bool,
    max_det: int,
    show_progress: bool,
) -> dict[Path, list[OBBInstance]]:
    YOLO = ensure_ultralytics_available()
    torch = import_torch_if_available()
    model = YOLO(str(weights))
    resolved_to_original = {image_path.resolve(): image_path for image_path in image_paths}
    prediction_map: dict[Path, list[OBBInstance]] = {image_path: [] for image_path in image_paths}
    progress_bar = create_progress_bar(
        len(image_paths),
        desc="Running inference",
        disable=not show_progress,
    )

    index = 0
    default_batch = max(1, batch)
    try:
        while index < len(image_paths):
            current_batch = min(default_batch, len(image_paths) - index)
            while True:
                chunk_paths = image_paths[index : index + current_batch]
                chunk_start = time.time()
                try:
                    results = model.predict(
                        source=[str(path) for path in chunk_paths],
                        stream=True,
                        conf=conf_threshold,
                        imgsz=imgsz,
                        batch=current_batch,
                        device=device,
                        verbose=False,
                        save=False,
                        half=half,
                        max_det=max_det,
                    )
                    chunk_results = list(results)
                    if len(chunk_results) != len(chunk_paths):
                        warnings.warn(
                            f"Expected {len(chunk_paths)} prediction results, got {len(chunk_results)}. "
                            "The remaining images in this chunk will be left empty.",
                            RuntimeWarning,
                        )

                    for result in chunk_results:
                        resolved_image_path = Path(result.path).resolve()
                        image_path = resolved_to_original.get(resolved_image_path, resolved_image_path)
                        obb = getattr(result, "obb", None)
                        if obb is None:
                            prediction_map[image_path] = []
                            continue

                        # TODO: If your server-side Ultralytics version exposes OBB polygons under a different
                        # attribute name, adjust this block first.
                        polygons = obb.xyxyxyxy.cpu().numpy()
                        class_ids = obb.cls.cpu().numpy().astype(int)
                        confidences = obb.conf.cpu().numpy()
                        instances: list[OBBInstance] = []
                        for polygon, class_id, confidence in zip(polygons, class_ids, confidences):
                            instances.append(
                                OBBInstance(
                                    class_id=int(class_id),
                                    polygon=normalize_polygon(np.asarray(polygon, dtype=np.float64)),
                                    confidence=float(confidence),
                                    image_path=image_path,
                                    source=str(weights),
                                )
                            )
                        prediction_map[image_path] = instances

                    index += len(chunk_paths)
                    if progress_bar is not None:
                        elapsed = max(time.time() - chunk_start, 1e-6)
                        images_per_second = len(chunk_paths) / elapsed
                        progress_bar.set_postfix(
                            batch=current_batch,
                            ips=f"{images_per_second:.2f}",
                        )
                        progress_bar.update(len(chunk_paths))
                    clear_cuda_cache(torch)
                    break
                except RuntimeError as exc:
                    clear_cuda_cache(torch)
                    if not is_oom_error(exc):
                        raise
                    if current_batch <= 1:
                        raise RuntimeError(
                            "CUDA OOM even with batch=1. Try one or more of: "
                            "--half, smaller --imgsz, --max-det 100, or --device cpu."
                        ) from exc
                    new_batch = max(1, current_batch // 2)
                    warnings.warn(
                        f"CUDA OOM with batch={current_batch}. Retrying the same images with batch={new_batch}. "
                        "You can also pass --half or reduce --imgsz manually.",
                        RuntimeWarning,
                    )
                    current_batch = new_batch
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return prediction_map


def load_predictions(
    args: argparse.Namespace,
    image_paths: list[Path],
    image_sizes: dict[Path, tuple[int, int]],
) -> dict[Path, list[OBBInstance]]:
    if args.predictions is None and args.weights is None:
        raise AnalysisConfigurationError("Provide either --predictions or --weights.")

    if args.predictions is None:
        return load_predictions_from_model(
            weights=args.weights,
            image_paths=image_paths,
            conf_threshold=args.conf_thres,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            half=args.half,
            max_det=args.max_det,
            show_progress=not args.no_progress,
        )

    prediction_path = args.predictions.expanduser().resolve()
    prediction_format = args.prediction_format
    if prediction_format == "auto":
        if prediction_path.is_dir():
            prediction_format = "txt"
        elif prediction_path.suffix.lower() == ".json":
            prediction_format = "json"
        else:
            raise AnalysisConfigurationError(
                f"Could not infer prediction format for {prediction_path}. "
                "Use --prediction-format txt or --prediction-format json."
            )

    if prediction_format == "txt":
        if not prediction_path.is_dir():
            raise AnalysisConfigurationError("--prediction-format txt expects --predictions to be a directory.")
        return load_predictions_from_text_dir(
            prediction_dir=prediction_path,
            image_paths=image_paths,
            image_sizes=image_sizes,
            prediction_layout=args.prediction_layout,
            conf_threshold=args.conf_thres,
            show_progress=not args.no_progress,
        )
    if prediction_format == "json":
        if not prediction_path.is_file():
            raise AnalysisConfigurationError("--prediction-format json expects --predictions to be a .json file.")
        return load_predictions_from_json(
            prediction_file=prediction_path,
            image_paths=image_paths,
            image_sizes=image_sizes,
            conf_threshold=args.conf_thres,
            show_progress=not args.no_progress,
        )
    raise KeyError(f"Unsupported prediction format {prediction_format}.")


def max_iou_by_class(target: OBBInstance, others: list[OBBInstance]) -> tuple[float, float]:
    best_same = 0.0
    best_other = 0.0
    for other in others:
        iou = obb_iou(target, other)
        if other.class_id == target.class_id:
            best_same = max(best_same, iou)
        else:
            best_other = max(best_other, iou)
    return best_same, best_other


def classify_miss_reason(best_same: float, best_other: float, iou_threshold: float) -> str:
    if best_other >= iou_threshold:
        return "wrong_class_overlap"
    if best_same >= 0.1:
        return "same_class_low_iou"
    if best_other >= 0.1:
        return "nearby_wrong_class"
    return "no_overlap_candidate"


def classify_false_reason(best_same: float, best_other: float, iou_threshold: float) -> str:
    if best_other >= iou_threshold:
        return "wrong_class_overlap"
    if best_same >= 0.1:
        return "same_class_low_iou"
    if best_other >= 0.1:
        return "nearby_object_wrong_class"
    return "background_false_positive"


def write_rows_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_counter(counter: Counter[str], title: str, xlabel: str, output_path: Path) -> None:
    labels = list(counter.keys())
    values = [counter[label] for label in labels]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color="#2c7fb8")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_size_bins(values: list[float], num_bins: int, strategy: str) -> np.ndarray:
    if not values:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1 or math.isclose(float(array.min()), float(array.max())):
        value = float(array[0])
        return np.asarray([max(0.0, value - 0.5), value + 0.5], dtype=np.float64)

    if strategy == "quantile":
        edges = np.quantile(array, np.linspace(0.0, 1.0, num_bins + 1))
        edges = np.unique(edges)
        if edges.size >= 2:
            return edges

    return np.linspace(float(array.min()), float(array.max()), num_bins + 1)


def format_bin_labels(edges: np.ndarray) -> list[str]:
    labels: list[str] = []
    for left, right in zip(edges[:-1], edges[1:]):
        labels.append(f"[{left:.1f}, {right:.1f})")
    return labels


def plot_histogram(values: list[float], edges: np.ndarray, title: str, xlabel: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.hist(values, bins=edges, color="#4daf4a", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_miss_rate_by_size(
    all_sizes: list[float],
    missed_sizes: list[float],
    edges: np.ndarray,
    output_path: Path,
    metric_name: str,
) -> list[dict[str, float]]:
    total_counts, _ = np.histogram(all_sizes, bins=edges)
    missed_counts, _ = np.histogram(missed_sizes, bins=edges)
    miss_rates = np.divide(
        missed_counts,
        total_counts,
        out=np.zeros_like(missed_counts, dtype=np.float64),
        where=total_counts > 0,
    )
    labels = format_bin_labels(edges)
    x = np.arange(len(labels))

    plt.figure(figsize=(13, 6))
    plt.plot(x, miss_rates, marker="o", color="#d95f02")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Miss Rate")
    plt.xlabel(metric_name)
    plt.title(f"Miss Rate vs {metric_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    rows: list[dict[str, float]] = []
    for left, right, total, missed, miss_rate in zip(
        edges[:-1],
        edges[1:],
        total_counts.tolist(),
        missed_counts.tolist(),
        miss_rates.tolist(),
    ):
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "total_gt": int(total),
                "missed_gt": int(missed),
                "miss_rate": float(miss_rate),
            }
        )
    return rows


def analyze(args: argparse.Namespace) -> None:
    output_dir = ensure_directory(args.output_dir.expanduser().resolve())
    dataset_config = load_dataset_config(args.dataset)
    names = dataset_config.get("names", {})
    official_val_summary = run_official_ultralytics_val(args, output_dir)

    image_paths, gt_map, image_sizes = load_ground_truth(
        dataset_config=dataset_config,
        split=args.split,
        label_dirname=args.label_dirname,
        max_images=args.max_images,
        workers=args.workers,
        show_progress=not args.no_progress,
    )
    prediction_map = load_predictions(args, image_paths, image_sizes)

    missed_by_class: Counter[str] = Counter()
    false_by_class: Counter[str] = Counter()
    miss_reason_counter: Counter[str] = Counter()
    false_reason_counter: Counter[str] = Counter()
    missed_sizes: list[float] = []
    all_gt_sizes: list[float] = []

    missed_rows: list[dict[str, Any]] = []
    false_rows: list[dict[str, Any]] = []

    total_gt = 0
    total_pred = 0
    total_matches = 0

    match_tasks = [
        (image_path, gt_map.get(image_path, []), prediction_map.get(image_path, []), args.iou_thres, args.size_metric, names)
        for image_path in image_paths
    ]

    if args.workers <= 1:
        results_iter = progress_iter(
            (_match_single_image(task) for task in match_tasks),
            total=len(match_tasks),
            desc="Matching errors",
            disable=args.no_progress,
        )
        partial_results = list(results_iter)
    else:
        partial_results = []
        progress_bar = create_progress_bar(
            len(match_tasks),
            desc="Matching errors",
            disable=args.no_progress,
        )
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_match_single_image, task) for task in match_tasks]
                for future in as_completed(futures):
                    partial_results.append(future.result())
                    if progress_bar is not None:
                        progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

    for result in partial_results:
        total_gt += result["total_gt"]
        total_pred += result["total_pred"]
        total_matches += result["total_matches"]
        all_gt_sizes.extend(result["all_gt_sizes"])
        missed_sizes.extend(result["missed_sizes"])
        missed_rows.extend(result["missed_rows"])
        false_rows.extend(result["false_rows"])
        missed_by_class.update(result["missed_by_class"])
        false_by_class.update(result["false_by_class"])
        miss_reason_counter.update(result["missed_by_reason"])
        false_reason_counter.update(result["false_by_reason"])

    size_edges = build_size_bins(all_gt_sizes, args.size_bins, args.size_bin_strategy)
    miss_rate_rows = plot_miss_rate_by_size(
        all_sizes=all_gt_sizes,
        missed_sizes=missed_sizes,
        edges=size_edges,
        output_path=output_dir / "miss_rate_by_size.png",
        metric_name=f"{args.size_metric} (pixels)",
    )

    plot_counter(
        missed_by_class,
        title="Missed Detections by Class",
        xlabel="Class",
        output_path=output_dir / "missed_by_class.png",
    )
    plot_counter(
        false_by_class,
        title="False Detections by Class",
        xlabel="Class",
        output_path=output_dir / "false_by_class.png",
    )
    plot_counter(
        miss_reason_counter,
        title="Missed Detection Reasons",
        xlabel="Reason",
        output_path=output_dir / "missed_by_reason.png",
    )
    plot_counter(
        false_reason_counter,
        title="False Detection Reasons",
        xlabel="Reason",
        output_path=output_dir / "false_by_reason.png",
    )
    plot_histogram(
        values=missed_sizes,
        edges=size_edges,
        title=f"Missed Target Size Histogram ({args.size_metric})",
        xlabel=f"{args.size_metric} (pixels)",
        output_path=output_dir / "missed_size_histogram.png",
    )

    summary = {
        "dataset": str(args.dataset.expanduser().resolve()),
        "split": args.split,
        "prediction_source": str(args.predictions.expanduser().resolve()) if args.predictions else str(args.weights),
        "size_metric": args.size_metric,
        "analysis_settings": {
            "prediction_conf_threshold": args.conf_thres,
            "matching_iou_threshold": args.iou_thres,
            "analysis_max_det": args.max_det,
            "prediction_format": args.prediction_format,
            "prediction_layout": args.prediction_layout,
            "matching_method": "Ultralytics official-style OBB matching (batch_probiou + class-aware greedy assignment) when available, otherwise custom polygon IoU fallback",
        },
        "image_count": len(image_paths),
        "total_gt": total_gt,
        "total_predictions": total_pred,
        "matched_pairs": total_matches,
        "missed_count": len(missed_rows),
        "false_count": len(false_rows),
        "miss_rate": float(len(missed_rows) / total_gt) if total_gt else None,
        "false_per_prediction": float(len(false_rows) / total_pred) if total_pred else None,
        "missed_by_class": dict(missed_by_class),
        "false_by_class": dict(false_by_class),
        "missed_by_reason": dict(miss_reason_counter),
        "false_by_reason": dict(false_reason_counter),
        "miss_rate_by_size_bins": miss_rate_rows,
        "official_validation": official_val_summary,
        "interpretation_notes": [
            "Per-image TP/FP/FN matching now prefers the same official Ultralytics OBB IoU backend used during validation.",
            "When Ultralytics is unavailable, the script falls back to custom polygon IoU and the summary will still be generated.",
            "When --weights is used, an additional official Ultralytics val() pass is saved to official_val_metrics.json unless --skip-official-val is set.",
        ],
    }
    write_json(output_dir / "summary.json", summary)

    if args.save_details:
        write_rows_csv(
            output_dir / "missed_details.csv",
            fieldnames=[
                "image",
                "class_id",
                "class_name",
                f"{args.size_metric}_px",
                "reason",
                "best_same_class_iou",
                "best_other_class_iou",
            ],
            rows=missed_rows,
        )
        write_rows_csv(
            output_dir / "false_details.csv",
            fieldnames=[
                "image",
                "class_id",
                "class_name",
                "confidence",
                "reason",
                "best_same_class_iou",
                "best_other_class_iou",
            ],
            rows=false_rows,
        )

    print(f"[analyze_errors] images={len(image_paths)} gt={total_gt} pred={total_pred} matched={total_matches}")
    print(f"[analyze_errors] missed={len(missed_rows)} false={len(false_rows)} output_dir={output_dir}")
    if official_val_summary is not None:
        print(f"[analyze_errors] official_validation_status={official_val_summary.get('status', 'unknown')}")
        print(f"  - {output_dir / 'official_val_metrics.json'}")
    print("[analyze_errors] Main outputs:")
    print(f"  - {output_dir / 'summary.json'}")
    print(f"  - {output_dir / 'missed_by_class.png'}")
    print(f"  - {output_dir / 'false_by_class.png'}")
    print(f"  - {output_dir / 'missed_size_histogram.png'}")
    print(f"  - {output_dir / 'miss_rate_by_size.png'}")


def main() -> None:
    args = parse_args()
    analyze(args)


if __name__ == "__main__":
    main()

