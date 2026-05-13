from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_GROUPS = {
    "metrics_comparison.png": [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ],
    "train_loss_comparison.png": [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
    ],
    "val_loss_comparison.png": [
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ],
}

PRETTY_NAMES = {
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "train/box_loss": "Train Box Loss",
    "train/cls_loss": "Train Cls Loss",
    "train/dfl_loss": "Train DFL Loss",
    "val/box_loss": "Val Box Loss",
    "val/cls_loss": "Val Cls Loss",
    "val/dfl_loss": "Val DFL Loss",
}

RUN_COLORS = {
    "test": "#1f77b4",
    "yolo11s_obb_p2": "#d62728",
    "yolo11s_obb_p2_cbam_p3": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot grouped training metrics for multiple YOLO runs.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dota_runs",
        help="Directory that contains run folders with results.csv.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["test", "yolo11s_obb_p2", "yolo11s_obb_p2_cbam_p3"],
        help="Run directory names under runs-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "analysis" / "outputs" / "run_comparisons" / "test_p2_cbam_p3",
        help="Directory for generated comparison figures.",
    )
    return parser.parse_args()


def load_run_results(runs_root: Path, run_names: list[str]) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for run_name in run_names:
        csv_path = runs_root / run_name / "results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"results.csv not found for run: {csv_path}")
        frame = pd.read_csv(csv_path)
        frame.columns = [column.strip() for column in frame.columns]
        if "epoch" not in frame.columns:
            frame["epoch"] = range(1, len(frame) + 1)
        results[run_name] = frame
    return results


def make_figure(metric_names: list[str]) -> tuple[plt.Figure, list[plt.Axes]]:
    count = len(metric_names)
    if count == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)
        return fig, list(axes.flatten())
    fig, axes = plt.subplots(1, count, figsize=(5.2 * count, 4.6), dpi=180)
    if count == 1:
        axes = [axes]
    else:
        axes = list(axes)
    return fig, axes


def plot_group(output_path: Path, metric_names: list[str], run_results: dict[str, pd.DataFrame]) -> None:
    fig, axes = make_figure(metric_names)

    for axis, metric_name in zip(axes, metric_names):
        for run_name, frame in run_results.items():
            if metric_name not in frame.columns:
                continue
            axis.plot(
                frame["epoch"],
                frame[metric_name],
                label=run_name,
                color=RUN_COLORS.get(run_name),
                linewidth=2.0,
            )
        axis.set_title(PRETTY_NAMES.get(metric_name, metric_name))
        axis.set_xlabel("Epoch")
        axis.set_ylabel(PRETTY_NAMES.get(metric_name, metric_name))
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend(frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(output_dir: Path, run_results: dict[str, pd.DataFrame]) -> None:
    summary_rows = []
    for run_name, frame in run_results.items():
        best_map50_idx = frame["metrics/mAP50(B)"].idxmax()
        best_map5095_idx = frame["metrics/mAP50-95(B)"].idxmax()
        summary_rows.append(
            {
                "run": run_name,
                "epochs": int(frame["epoch"].max()),
                "best_mAP50": float(frame.loc[best_map50_idx, "metrics/mAP50(B)"]),
                "best_mAP50_epoch": int(frame.loc[best_map50_idx, "epoch"]),
                "best_mAP50_95": float(frame.loc[best_map5095_idx, "metrics/mAP50-95(B)"]),
                "best_mAP50_95_epoch": int(frame.loc[best_map5095_idx, "epoch"]),
                "final_train_box_loss": float(frame.iloc[-1]["train/box_loss"]),
                "final_train_cls_loss": float(frame.iloc[-1]["train/cls_loss"]),
                "final_train_dfl_loss": float(frame.iloc[-1]["train/dfl_loss"]),
                "final_val_box_loss": float(frame.iloc[-1]["val/box_loss"]),
                "final_val_cls_loss": float(frame.iloc[-1]["val/cls_loss"]),
                "final_val_dfl_loss": float(frame.iloc[-1]["val/dfl_loss"]),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)


def main() -> None:
    args = parse_args()
    run_results = load_run_results(args.runs_root, args.runs)

    for filename, metric_names in METRIC_GROUPS.items():
        plot_group(args.output_dir / filename, metric_names, run_results)

    write_summary(args.output_dir, run_results)


if __name__ == "__main__":
    main()
