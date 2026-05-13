from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from common_obb import (
        AnalysisConfigurationError,
        collect_image_paths,
        ensure_directory,
        image_size,
        label_path_for_image,
        load_dataset_config,
        metric_value_from_polygon,
        read_label_file,
        safe_name,
        summarize_numeric,
        write_json,
    )
except ImportError:
    from analysis.common_obb import (
        AnalysisConfigurationError,
        collect_image_paths,
        ensure_directory,
        image_size,
        label_path_for_image,
        load_dataset_config,
        metric_value_from_polygon,
        read_label_file,
        safe_name,
        summarize_numeric,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze OBB object-size distributions for a YOLO dataset.",
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Path to data.yaml.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to include. Default: train val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/outputs/object_sizes"),
        help="Directory for plots and summary files.",
    )
    parser.add_argument(
        "--size-metric",
        choices=("area", "equivalent_side", "long_edge", "short_edge"),
        default="equivalent_side",
        help=(
            "OBB size definition. Default equivalent_side = sqrt(OBB polygon area), "
            "which is rotation-invariant and easy to interpret in pixels."
        ),
    )
    parser.add_argument(
        "--binning",
        choices=("quantile", "preset"),
        default="quantile",
        help="How to group targets into coarse size buckets.",
    )
    parser.add_argument(
        "--quantile-bins",
        type=int,
        default=4,
        help="Number of quantile buckets when --binning quantile.",
    )
    parser.add_argument(
        "--preset-thresholds",
        nargs=3,
        type=float,
        default=[16.0, 32.0, 96.0],
        metavar=("VERY_SMALL", "SMALL", "MEDIUM"),
        help="Thresholds for preset buckets, in pixels of the chosen size metric.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=40,
        help="Histogram bin count.",
    )
    parser.add_argument("--label-dirname", default="labels", help="Label directory name. Default: labels.")
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional limit for debugging on the server.",
    )
    return parser.parse_args()


def write_rows_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_quantile_bucket_labels(values: list[float], num_bins: int) -> tuple[np.ndarray, list[str]]:
    edges = np.quantile(values, np.linspace(0.0, 1.0, num_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        value = float(values[0])
        edges = np.asarray([max(0.0, value - 0.5), value + 0.5])
    labels = [f"Q{i + 1}: [{left:.2f}, {right:.2f})" for i, (left, right) in enumerate(zip(edges[:-1], edges[1:]))]
    return edges, labels


def build_preset_bucket_labels(thresholds: list[float]) -> tuple[np.ndarray, list[str]]:
    very_small, small, medium = thresholds
    edges = np.asarray([0.0, very_small, small, medium, np.inf], dtype=np.float64)
    labels = [
        f"very_small (<{very_small:.1f})",
        f"small [{very_small:.1f}, {small:.1f})",
        f"medium [{small:.1f}, {medium:.1f})",
        f"large (>={medium:.1f})",
    ]
    return edges, labels


def assign_buckets(values: list[float], edges: np.ndarray, labels: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    finite_edges = edges.copy()
    if np.isinf(finite_edges[-1]):
        finite_edges[-1] = max(values) + 1.0
    bucket_indices = np.digitize(values, finite_edges[1:-1], right=False)
    for bucket_index in bucket_indices:
        counts[labels[int(bucket_index)]] += 1
    return counts


def plot_histogram(values: list[float], bins: int, metric: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.hist(values, bins=bins, color="#3b8bc2", edgecolor="black")
    plt.xlabel(f"{metric} (pixels)")
    plt.ylabel("Object Count")
    plt.title(f"Dataset OBB Size Distribution: {metric}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bucket_bar(counts: Counter[str], metric: str, output_path: Path) -> None:
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color="#6a9f58")
    plt.xlabel("Bucket")
    plt.ylabel("Object Count")
    plt.title(f"OBB Size Buckets: {metric}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def analyze(args: argparse.Namespace) -> None:
    output_dir = ensure_directory(args.output_dir.expanduser().resolve())
    dataset_config = load_dataset_config(args.dataset)
    names = dataset_config.get("names", {})

    size_values: list[float] = []
    split_rows: list[dict[str, Any]] = []
    class_size_values: dict[str, list[float]] = defaultdict(list)
    missing_label_count = 0

    for split in args.splits:
        image_paths = collect_image_paths(dataset_config, split, max_images=args.max_images_per_split)
        if not image_paths:
            print(f"[analyze_object_sizes] split={split} has no images. Skipping.")
            continue

        for image_path in image_paths:
            size_xy = image_size(image_path)
            label_path = label_path_for_image(image_path, label_dirname=args.label_dirname)
            if not label_path.exists():
                missing_label_count += 1
                continue
            instances = read_label_file(label_path, size_xy, layout="gt")
            for instance in instances:
                size_value = metric_value_from_polygon(instance.polygon, args.size_metric)
                class_name = safe_name(instance.class_id, names)
                size_values.append(size_value)
                class_size_values[class_name].append(size_value)
                split_rows.append(
                    {
                        "split": split,
                        "image": str(image_path),
                        "class_id": instance.class_id,
                        "class_name": class_name,
                        f"{args.size_metric}_px": round(size_value, 6),
                    }
                )

    if not size_values:
        raise AnalysisConfigurationError(
            "No labeled OBB targets were found. On the server, check the dataset YAML paths and label files."
        )

    if args.binning == "quantile":
        bucket_edges, bucket_labels = build_quantile_bucket_labels(size_values, args.quantile_bins)
    else:
        bucket_edges, bucket_labels = build_preset_bucket_labels(args.preset_thresholds)
    bucket_counts = assign_buckets(size_values, bucket_edges, bucket_labels)

    overall_summary = summarize_numeric(size_values)
    overall_summary["size_metric"] = args.size_metric
    overall_summary["size_metric_definition"] = (
        "area: polygon area in px^2; equivalent_side: sqrt(area) in px; "
        "long_edge/short_edge: averaged opposite-edge lengths in px."
    )
    overall_summary["main_range_p05_p95"] = [overall_summary.get("q05"), overall_summary.get("q95")]
    overall_summary["main_range_q25_q75"] = [overall_summary.get("q25"), overall_summary.get("q75")]
    overall_summary["missing_label_files"] = missing_label_count
    overall_summary["bucket_counts"] = dict(bucket_counts)
    overall_summary["splits"] = args.splits

    per_class_rows: list[dict[str, Any]] = []
    for class_name, values in sorted(class_size_values.items()):
        class_summary = summarize_numeric(values)
        per_class_rows.append(
            {
                "class_name": class_name,
                "count": class_summary.get("count", 0),
                "mean": round(class_summary.get("mean", 0.0), 6),
                "median": round(class_summary.get("median", 0.0), 6),
                "q05": round(class_summary.get("q05", 0.0), 6),
                "q25": round(class_summary.get("q25", 0.0), 6),
                "q75": round(class_summary.get("q75", 0.0), 6),
                "q95": round(class_summary.get("q95", 0.0), 6),
                "min": round(class_summary.get("min", 0.0), 6),
                "max": round(class_summary.get("max", 0.0), 6),
            }
        )

    plot_histogram(
        values=size_values,
        bins=args.hist_bins,
        metric=args.size_metric,
        output_path=output_dir / "object_size_histogram.png",
    )
    plot_bucket_bar(
        counts=bucket_counts,
        metric=args.size_metric,
        output_path=output_dir / "object_size_buckets.png",
    )

    write_json(output_dir / "object_size_summary.json", overall_summary)
    write_rows_csv(
        output_dir / "object_size_per_class.csv",
        fieldnames=["class_name", "count", "mean", "median", "q05", "q25", "q75", "q95", "min", "max"],
        rows=per_class_rows,
    )
    write_rows_csv(
        output_dir / "object_size_instances.csv",
        fieldnames=["split", "image", "class_id", "class_name", f"{args.size_metric}_px"],
        rows=split_rows,
    )

    print(
        f"[analyze_object_sizes] objects={len(size_values)} size_metric={args.size_metric} "
        f"output_dir={output_dir}"
    )
    print(f"[analyze_object_sizes] summary={output_dir / 'object_size_summary.json'}")
    print(f"[analyze_object_sizes] histogram={output_dir / 'object_size_histogram.png'}")
    print(f"[analyze_object_sizes] buckets={output_dir / 'object_size_buckets.png'}")


def main() -> None:
    args = parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
