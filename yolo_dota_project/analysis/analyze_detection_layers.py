from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

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
        read_label_file,
        safe_name,
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
        read_label_file,
        safe_name,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe YOLO OBB detection layers with forward hooks and size-based heuristics.",
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Path to data.yaml.")
    parser.add_argument("--weights", required=True, type=Path, help="YOLO OBB weights file.")
    parser.add_argument("--split", default="val", help="Dataset split to sample. Default: val.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/outputs/detection_layers"),
        help="Directory for JSON summaries and plots.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size. Keep 1 for simpler tracing.")
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
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Prediction confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for GT/pred matching.")
    parser.add_argument(
        "--size-metric",
        choices=("area", "equivalent_side", "long_edge", "short_edge"),
        default="equivalent_side",
        help="OBB size definition used for heuristic layer assignment.",
    )
    parser.add_argument(
        "--small-object-threshold",
        type=float,
        default=32.0,
        help="Threshold in pixels of the selected size metric used to mark small objects.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="How many split images to sample for hook tracing and heuristic evaluation.",
    )
    parser.add_argument(
        "--sample-random",
        action="store_true",
        help="Randomly sample images instead of taking the first N.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --sample-random.")
    parser.add_argument("--label-dirname", default="labels", help="Label directory name. Default: labels.")
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Only collect hook outputs and feature map shapes. Skip GT/pred heuristic analysis.",
    )
    parser.add_argument(
        "--skip-official-val",
        action="store_true",
        help=(
            "Skip a separate official Ultralytics val() pass. "
            "Without this flag, official comparable metrics are recorded alongside the heuristic layer summary."
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
    return parser.parse_args()


def ensure_ultralytics_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "This script requires torch and ultralytics in the server environment. "
            "Install them before running detection-layer analysis."
        ) from exc
    return YOLO, torch, nn


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


def run_official_ultralytics_val(YOLO: Any, args: argparse.Namespace, output_dir: Path) -> Optional[dict[str, Any]]:
    if args.skip_official_val:
        return None

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
        fallback_kwargs = {
            key: value
            for key, value in run_kwargs.items()
            if key not in {"project", "name", "exist_ok", "save_json", "max_det", "conf"}
        }
        metrics = model.val(**fallback_kwargs)
        run_kwargs = fallback_kwargs
    except Exception as exc:
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


def sample_image_paths(image_paths: list[Path], max_images: int, sample_random: bool, seed: int) -> list[Path]:
    if len(image_paths) <= max_images:
        return image_paths
    if not sample_random:
        return image_paths[:max_images]
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(image_paths)), max_images))
    return [image_paths[index] for index in indices]


def find_detection_head(torch_model: Any, nn: Any) -> tuple[str, Any]:
    if hasattr(torch_model, "model") and isinstance(torch_model.model, nn.Module):
        named_children = list(torch_model.model.named_children())
        for child_name, child_module in reversed(named_children):
            class_name = child_module.__class__.__name__.lower()
            if "detect" in class_name or "obb" in class_name:
                return f"model.{child_name}", child_module

    for module_name, module in reversed(list(torch_model.named_modules())):
        if not module_name:
            continue
        class_name = module.__class__.__name__.lower()
        if "detect" in class_name or "obb" in class_name:
            return module_name, module

    raise AnalysisConfigurationError(
        "Could not locate the detection head in the loaded model. "
        "Update find_detection_head() for your server-side Ultralytics version."
    )


def shape_signature(obj: Any, torch: Any) -> Any:
    if torch.is_tensor(obj):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [shape_signature(item, torch) for item in obj]
    if isinstance(obj, dict):
        return {key: shape_signature(value, torch) for key, value in obj.items()}
    if obj is None:
        return None
    return str(type(obj).__name__)


def extract_feature_maps_from_inputs(inputs: Any, torch: Any) -> list[Any]:
    if not inputs:
        return []
    candidate = inputs[0] if len(inputs) == 1 else inputs
    if isinstance(candidate, (list, tuple)):
        return [item for item in candidate if torch.is_tensor(item)]
    return [candidate] if torch.is_tensor(candidate) else []


def register_hooks(head: Any, nn: Any, torch: Any) -> tuple[dict[str, Any], list[Any]]:
    captures: dict[str, Any] = {
        "head_input_shapes": None,
        "head_output_signature": None,
        "branch_output_shapes": defaultdict(set),
    }
    hooks: list[Any] = []

    def head_hook(module: Any, inputs: Any, output: Any) -> None:
        feature_maps = extract_feature_maps_from_inputs(inputs, torch)
        if captures["head_input_shapes"] is None:
            captures["head_input_shapes"] = [list(feature_map.shape) for feature_map in feature_maps]
        if captures["head_output_signature"] is None:
            captures["head_output_signature"] = shape_signature(output, torch)

    hooks.append(head.register_forward_hook(head_hook))

    for child_name, child_module in head.named_children():
        if isinstance(child_module, nn.ModuleList):
            for branch_index, branch_module in enumerate(child_module):
                label = f"{child_name}[{branch_index}]"

                def _hook(_: Any, __: Any, output: Any, branch_label: str = label) -> None:
                    captures["branch_output_shapes"][branch_label].add(str(shape_signature(output, torch)))

                hooks.append(branch_module.register_forward_hook(_hook))
    return captures, hooks


def remove_hooks(hooks: list[Any]) -> None:
    for hook in hooks:
        hook.remove()


def infer_strides(head: Any, head_input_shapes: list[list[int]], imgsz: int) -> list[float]:
    raw_stride = getattr(head, "stride", None)
    if raw_stride is not None:
        if hasattr(raw_stride, "tolist"):
            values = raw_stride.tolist()
        else:
            values = list(raw_stride)
        strides = [float(value) for value in values]
        if len(strides) == len(head_input_shapes):
            return strides

    strides: list[float] = []
    for shape in head_input_shapes:
        if len(shape) < 4 or shape[2] == 0:
            strides.append(float("nan"))
            continue
        strides.append(float(imgsz) / float(shape[2]))
    return strides


def anchors_per_location(head: Any) -> int:
    raw = getattr(head, "na", 1)
    try:
        value = int(raw)
    except Exception:
        value = 1
    return max(1, value)


def collect_ground_truth(
    dataset_config: dict[str, Any],
    image_paths: list[Path],
    label_dirname: str,
) -> dict[Path, list[OBBInstance]]:
    gt_map: dict[Path, list[OBBInstance]] = {}
    for image_path in image_paths:
        label_path = label_path_for_image(image_path, label_dirname=label_dirname)
        instances = read_label_file(label_path, image_size(image_path), layout="gt")
        for instance in instances:
            instance.image_path = image_path
            instance.source = "ground_truth"
        gt_map[image_path] = instances
    return gt_map


def run_predictions(model: Any, image_paths: list[Path], args: argparse.Namespace) -> dict[Path, list[OBBInstance]]:
    results = model.predict(
        source=[str(path) for path in image_paths],
        stream=True,
        conf=args.conf_thres,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
        save=False,
        half=args.half,
        max_det=args.max_det,
    )
    prediction_map: dict[Path, list[OBBInstance]] = {path.resolve(): [] for path in image_paths}
    for result in results:
        image_path = Path(result.path).resolve()
        obb = getattr(result, "obb", None)
        if obb is None:
            continue
        # TODO: If your Ultralytics version exposes a different OBB polygon field, update this block.
        polygons = obb.xyxyxyxy.cpu().numpy()
        class_ids = obb.cls.cpu().numpy().astype(int)
        confidences = obb.conf.cpu().numpy()
        instances: list[OBBInstance] = []
        for polygon, class_id, confidence in zip(polygons, class_ids, confidences):
            instances.append(
                OBBInstance(
                    class_id=int(class_id),
                    polygon=np.asarray(polygon, dtype=np.float64),
                    confidence=float(confidence),
                    image_path=image_path,
                    source=str(args.weights),
                )
            )
        prediction_map[image_path] = instances
    return prediction_map


def assign_layer_by_size(size_value: float, strides: list[float]) -> int:
    canonical_sizes = [max(stride * 4.0, 1e-6) for stride in strides]
    if len(canonical_sizes) == 1:
        return 0
    boundaries = [math.sqrt(left * right) for left, right in zip(canonical_sizes[:-1], canonical_sizes[1:])]
    for index, boundary in enumerate(boundaries):
        if size_value <= boundary:
            return index
    return len(canonical_sizes) - 1


def layer_range_text(layer_index: int, strides: list[float]) -> str:
    canonical_sizes = [max(stride * 4.0, 1e-6) for stride in strides]
    if len(canonical_sizes) == 1:
        return f"~{canonical_sizes[0]:.1f}px"
    boundaries = [math.sqrt(left * right) for left, right in zip(canonical_sizes[:-1], canonical_sizes[1:])]
    if layer_index == 0:
        return f"<= {boundaries[0]:.1f}px"
    if layer_index == len(canonical_sizes) - 1:
        return f"> {boundaries[-1]:.1f}px"
    return f"({boundaries[layer_index - 1]:.1f}, {boundaries[layer_index]:.1f}] px"


def plot_bar(values: list[float], labels: list[str], title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color="#3f7f93")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def analyze(args: argparse.Namespace) -> None:
    output_dir = ensure_directory(args.output_dir.expanduser().resolve())
    dataset_config = load_dataset_config(args.dataset)
    all_images = collect_image_paths(dataset_config, args.split)
    if not all_images:
        raise AnalysisConfigurationError(
            f"No images found for split '{args.split}'. Check the server dataset paths in {args.dataset}."
        )
    image_paths = sample_image_paths(all_images, args.max_images, args.sample_random, args.seed)

    YOLO, torch, nn = ensure_ultralytics_stack()
    official_val_summary = run_official_ultralytics_val(YOLO, args, output_dir)
    model = YOLO(str(args.weights))
    torch_model = getattr(model, "model", model)
    head_name, head_module = find_detection_head(torch_model, nn)
    captures, hooks = register_hooks(head_module, nn, torch)

    try:
        # Use a small batch to trigger hooks without modifying the training/validation code path.
        _ = list(
            model.predict(
                source=[str(path) for path in image_paths[: max(1, args.batch)]],
                stream=True,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                conf=args.conf_thres,
                verbose=False,
                save=False,
                half=args.half,
                max_det=args.max_det,
            )
        )
    finally:
        remove_hooks(hooks)

    head_input_shapes = captures["head_input_shapes"]
    if not head_input_shapes:
        raise AnalysisConfigurationError(
            "Forward hooks did not capture head input shapes. "
            "Update register_hooks() for your server-side model structure."
        )

    strides = infer_strides(head_module, head_input_shapes, args.imgsz)
    per_location = anchors_per_location(head_module)

    layer_rows: list[dict[str, Any]] = []
    plot_labels: list[str] = []
    candidate_counts: list[float] = []
    assigned_gt_counts: list[float] = []
    miss_rates: list[float] = []

    performance_by_layer: dict[int, dict[str, Any]] = {}
    for layer_index, shape in enumerate(head_input_shapes):
        _, channels, height, width = shape
        stride = strides[layer_index] if layer_index < len(strides) else float("nan")
        candidate_count = int(height * width * per_location)
        plot_labels.append(f"L{layer_index} / s{stride:.0f}")
        candidate_counts.append(candidate_count)
        performance_by_layer[layer_index] = {
            "assigned_gt": 0,
            "matched_gt": 0,
            "missed_gt": 0,
            "small_gt": 0,
            "small_missed_gt": 0,
            "heuristic_size_range": layer_range_text(layer_index, strides),
        }
        layer_rows.append(
            {
                "layer_index": layer_index,
                "feature_shape": shape,
                "channels": int(channels),
                "height": int(height),
                "width": int(width),
                "stride": float(stride),
                "anchors_per_location": int(per_location),
                "candidate_predictions": candidate_count,
            }
        )

    if not args.skip_performance:
        gt_map = collect_ground_truth(dataset_config, image_paths, args.label_dirname)
        prediction_map = run_predictions(model, image_paths, args)
        objects_below_smallest_nominal = 0
        smallest_nominal = min(strides) * 4.0 if strides else None

        for image_path in image_paths:
            gt_instances = gt_map.get(image_path, [])
            pred_instances = prediction_map.get(image_path.resolve(), [])
            matches, unmatched_gt_indices, _ = match_detections(gt_instances, pred_instances, args.iou_thres)
            matched_gt_indices = {gt_index for gt_index, _, _ in matches}

            for gt_index, gt_instance in enumerate(gt_instances):
                size_value = metric_value_from_polygon(gt_instance.polygon, args.size_metric)
                layer_index = assign_layer_by_size(size_value, strides)
                stats = performance_by_layer[layer_index]
                stats["assigned_gt"] += 1
                if size_value <= args.small_object_threshold:
                    stats["small_gt"] += 1
                if smallest_nominal is not None and size_value < smallest_nominal:
                    objects_below_smallest_nominal += 1
                if gt_index in matched_gt_indices:
                    stats["matched_gt"] += 1
                else:
                    stats["missed_gt"] += 1
                    if size_value <= args.small_object_threshold:
                        stats["small_missed_gt"] += 1

        for layer_index in range(len(head_input_shapes)):
            stats = performance_by_layer[layer_index]
            assigned = stats["assigned_gt"]
            assigned_gt_counts.append(assigned)
            miss_rate = float(stats["missed_gt"] / assigned) if assigned else 0.0
            miss_rates.append(miss_rate)
            layer_rows[layer_index]["heuristic_assigned_gt"] = assigned
            layer_rows[layer_index]["heuristic_matched_gt"] = stats["matched_gt"]
            layer_rows[layer_index]["heuristic_missed_gt"] = stats["missed_gt"]
            layer_rows[layer_index]["heuristic_miss_rate"] = miss_rate
            layer_rows[layer_index]["small_gt"] = stats["small_gt"]
            layer_rows[layer_index]["small_missed_gt"] = stats["small_missed_gt"]
            layer_rows[layer_index]["small_object_miss_rate"] = (
                float(stats["small_missed_gt"] / stats["small_gt"]) if stats["small_gt"] else None
            )
            layer_rows[layer_index]["heuristic_size_range"] = stats["heuristic_size_range"]
    else:
        for _ in range(len(head_input_shapes)):
            assigned_gt_counts.append(0.0)
            miss_rates.append(0.0)

    plot_bar(
        values=candidate_counts,
        labels=plot_labels,
        title="Candidate Predictions per Detection Layer",
        ylabel="Candidate Count",
        output_path=output_dir / "candidate_predictions_by_layer.png",
    )
    if not args.skip_performance:
        plot_bar(
            values=assigned_gt_counts,
            labels=plot_labels,
            title=f"Heuristic GT Assignment by Layer ({args.size_metric})",
            ylabel="Assigned GT Count",
            output_path=output_dir / "heuristic_gt_assignment_by_layer.png",
        )
        plot_bar(
            values=miss_rates,
            labels=plot_labels,
            title="Heuristic Miss Rate by Layer",
            ylabel="Miss Rate",
            output_path=output_dir / "heuristic_miss_rate_by_layer.png",
        )

    branch_output_shapes = {
        key: sorted(value_set) for key, value_set in captures["branch_output_shapes"].items()
    }
    summary = {
        "dataset": str(args.dataset.expanduser().resolve()),
        "weights": str(args.weights.expanduser().resolve()),
        "split": args.split,
        "analysis_settings": {
            "hook_conf_threshold": args.conf_thres,
            "heuristic_matching_iou_threshold": args.iou_thres,
            "analysis_max_det": args.max_det,
            "half": args.half,
            "batch": args.batch,
            "sample_random": args.sample_random,
            "skip_performance": args.skip_performance,
            "heuristic_matching_method": "Ultralytics official-style OBB matching (batch_probiou + class-aware greedy assignment) when available, otherwise custom polygon IoU fallback",
        },
        "sampled_images": [str(path) for path in image_paths],
        "sample_count": len(image_paths),
        "detection_head_name": head_name,
        "detection_head_class": head_module.__class__.__name__,
        "head_input_shapes": head_input_shapes,
        "head_output_signature": captures["head_output_signature"],
        "branch_output_shapes": branch_output_shapes,
        "layer_rows": layer_rows,
        "size_metric": args.size_metric,
        "official_validation": official_val_summary,
        "heuristic_notes": [
            "Hook outputs describe actual tensor shapes seen during inference.",
            "Per-layer GT responsibility is a heuristic based on target size and detection-layer stride, not a direct decode-path attribution.",
            "If many missed objects are smaller than the smallest nominal layer range, testing an extra higher-resolution feature map (for example stride 4) is reasonable.",
            "Official comparable metrics, when available, are stored in official_validation and official_val_metrics.json.",
        ],
    }
    write_json(output_dir / "layer_summary.json", summary)

    print(f"[analyze_detection_layers] head={head_name} class={head_module.__class__.__name__}")
    for row in layer_rows:
        print(
            "[analyze_detection_layers] "
            f"layer={row['layer_index']} feature_shape={row['feature_shape']} stride={row['stride']:.2f} "
            f"candidates={row['candidate_predictions']}"
        )
    if official_val_summary is not None:
        print(f"[analyze_detection_layers] official_validation_status={official_val_summary.get('status', 'unknown')}")
        print(f"  - {output_dir / 'official_val_metrics.json'}")
    print(f"[analyze_detection_layers] summary={output_dir / 'layer_summary.json'}")
    print(f"[analyze_detection_layers] candidate_plot={output_dir / 'candidate_predictions_by_layer.png'}")
    if not args.skip_performance:
        print(f"[analyze_detection_layers] gt_assignment_plot={output_dir / 'heuristic_gt_assignment_by_layer.png'}")
        print(f"[analyze_detection_layers] miss_rate_plot={output_dir / 'heuristic_miss_rate_by_layer.png'}")


def main() -> None:
    args = parse_args()
    analyze(args)


if __name__ == "__main__":
    main()

