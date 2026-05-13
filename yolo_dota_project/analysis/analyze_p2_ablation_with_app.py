from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
APP_ROOT = WORKSPACE_ROOT / "yolo11_analysis_app"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from yolo11_analysis.dataset_analysis import analyze_dataset
from yolo11_analysis.detection_head_analysis import analyze_detection_heads
from yolo11_analysis.error_analysis import analyze_errors
from yolo11_analysis.inference import get_detection_head_info, run_inference
from yolo11_analysis.io import build_dataset_context
from yolo11_analysis.metrics import analyze_metrics
from yolo11_analysis.schemas import InferenceConfig

try:
    import train as project_train
except Exception:
    project_train = None


SNAPSHOT_ROOT = APP_ROOT / "outputs" / "analysis_history"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "outputs" / "app_p2_ablation_detailed"

RUN_SPECS = {
    "test": {
        "weights_path": PROJECT_ROOT / "dota_runs" / "test" / "weights" / "best.pt",
        "snapshot_path": SNAPSHOT_ROOT / "20260324_193553_7d0c0c46" / "snapshot.pkl",
    },
    "yolo11s_obb_p2": {
        "weights_path": PROJECT_ROOT / "dota_runs" / "yolo11s_obb_p2" / "weights" / "best.pt",
        "snapshot_path": SNAPSHOT_ROOT / "20260324_185251_1c7e6d6a" / "snapshot.pkl",
    },
    "yolo11s_obb_p2_branch": {
        "weights_path": PROJECT_ROOT / "dota_runs" / "yolo11s_obb_p2_branch" / "weights" / "best.pt",
        "snapshot_path": SNAPSHOT_ROOT / "20260419_141409_66fc55bc" / "snapshot.pkl",
    },
    "yolo11s_obb_p2_fusion": {
        "weights_path": PROJECT_ROOT / "dota_runs" / "yolo11s_obb_p2_fusion" / "weights" / "best.pt",
        "snapshot_path": SNAPSHOT_ROOT / "20260419_131653_f5c191fe" / "snapshot.pkl",
    },
    "yolo11s_obb_p2_cbam_p3": {
        "weights_path": PROJECT_ROOT / "dota_runs" / "yolo11s_obb_p2_cbam_p3" / "weights" / "best.pt",
        "snapshot_path": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use yolo11_analysis_app backend to compare P2 ablation runs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(RUN_SPECS.keys()),
        help="Run names to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated comparison tables and report.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached snapshot results and recompute everything from weights.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Inference device for recomputed runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size for recomputed runs.",
    )
    return parser.parse_args()


def load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    with snapshot_path.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        "config": payload["config"],
        "results": payload["results"],
        "source": f"snapshot:{snapshot_path.name}",
    }


def compute_analysis(weights_path: Path, *, device: str, batch_size: int) -> dict[str, Any]:
    if project_train is not None and hasattr(project_train, "register_custom_modules"):
        project_train.register_custom_modules()

    data_yaml = PROJECT_ROOT / "datasets" / "DOTA-split-lite" / "data.yaml"
    context = build_dataset_context(
        data_yaml=data_yaml,
        image_dir=None,
        label_dir=None,
        split="val",
    )
    detection_head_info = get_detection_head_info(str(weights_path))
    dataset_result = analyze_dataset(
        context,
        imgsz=1024,
        size_metric="sqrt_area",
        small_thr=32.0,
        medium_thr=96.0,
        detection_head_info=detection_head_info,
        head_target_cells=4.0,
    )
    prediction_payload = run_inference(
        context.image_paths,
        InferenceConfig(
            weights_path=str(weights_path),
            imgsz=1024,
            conf=0.25,
            iou=0.7,
            device=device,
            max_det=1200,
            batch_size=batch_size,
            cpu_fallback=True,
        ),
    )
    predictions = prediction_payload["predictions"]
    detection_head_result = analyze_detection_heads(dataset_result)
    error_result = analyze_errors(
        context,
        predictions,
        iou_threshold=0.5,
        size_metric="sqrt_area",
        small_thr=32.0,
        medium_thr=96.0,
    )
    metrics_result = analyze_metrics(
        context,
        predictions,
        iou_threshold=0.5,
        gt_by_image=error_result["gt_by_image"],
    )
    return {
        "config": {
            "weights_path": str(weights_path),
            "data_yaml_path": str(data_yaml),
            "split": "val",
            "eval_split": "val",
            "imgsz": 1024,
            "conf": 0.25,
            "iou": 0.7,
            "match_iou": 0.5,
            "device": device,
            "max_det": 1200,
            "batch_size": batch_size,
            "size_metric": "sqrt_area",
            "small_thr": 32.0,
            "medium_thr": 96.0,
        },
        "results": {
            "dataset_result": dataset_result,
            "detection_head_result": detection_head_result,
            "prediction_payload": prediction_payload,
            "error_result": error_result,
            "metrics_result": metrics_result,
        },
        "source": "computed",
    }


def get_run_payload(run_name: str, *, force_recompute: bool, device: str, batch_size: int) -> dict[str, Any]:
    spec = RUN_SPECS[run_name]
    snapshot_path = spec.get("snapshot_path")
    if not force_recompute and snapshot_path is not None and Path(snapshot_path).exists():
        return load_snapshot(Path(snapshot_path))
    return compute_analysis(Path(spec["weights_path"]), device=device, batch_size=batch_size)


def fn_size_rates(dataset_result: dict[str, Any], error_result: dict[str, Any]) -> dict[str, float]:
    gt_size = dataset_result["size_bucket_df"].copy()
    if gt_size.empty:
        return {}
    gt_size = gt_size.set_index("size_bucket")["count"].to_dict()
    fn_df = error_result.get("fn_df", pd.DataFrame())
    if fn_df.empty:
        return {f"fn_rate_{bucket}": 0.0 for bucket in gt_size}
    fn_size = fn_df.groupby("size_bucket").size().to_dict()
    return {
        f"fn_rate_{bucket}": float(fn_size.get(bucket, 0) / max(gt_size.get(bucket, 1), 1))
        for bucket in sorted(gt_size)
    }


def head_summary_metrics(detection_head_result: dict[str, Any]) -> dict[str, float]:
    summary = detection_head_result["summary"]
    per_box_df = detection_head_result.get("per_box_df", pd.DataFrame())
    boundary_ratio = float(per_box_df["is_boundary"].mean()) if not per_box_df.empty and "is_boundary" in per_box_df else 0.0
    return {
        "num_heads": int(summary.get("num_heads", 0)),
        "range_miss_ratio": float(summary.get("range_miss_ratio", 0.0)),
        "multi_head_overlap_ratio": float(summary.get("multi_head_overlap_ratio", 0.0)),
        "single_head_ratio": float(summary.get("single_head_ratio", 0.0)),
        "boundary_ratio": boundary_ratio,
    }


def build_overall_summary(run_payloads: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, payload in run_payloads.items():
        results = payload["results"]
        metrics_summary = results["metrics_result"]["summary"]
        error_summary = results["error_result"]["summary"]
        prediction_payload = results["prediction_payload"]
        pred_count = sum(len(boxes) for boxes in prediction_payload["predictions"].values())
        row = {
            "run": run_name,
            "source": payload["source"],
            "weights_path": payload["config"]["weights_path"],
            "precision": float(metrics_summary["precision"]),
            "recall": float(metrics_summary["recall"]),
            "mAP50": float(metrics_summary["mAP50"]),
            "mAP50_95": float(metrics_summary["mAP50_95"]),
            "tp": int(metrics_summary["tp"]),
            "fp": int(metrics_summary["fp"]),
            "fn": int(metrics_summary["fn"]),
            "pred_count": int(pred_count),
            "avg_predictions_per_image": float(pred_count / max(error_summary["num_images"], 1)),
            "images_with_fn": int(error_summary["images_with_fn"]),
            "images_with_fp": int(error_summary["images_with_fp"]),
            "fp_per_tp": float(metrics_summary["fp"] / max(metrics_summary["tp"], 1)),
            "fn_per_gt": float(metrics_summary["fn"] / max(metrics_summary["tp"] + metrics_summary["fn"], 1)),
            **head_summary_metrics(results["detection_head_result"]),
            **fn_size_rates(results["dataset_result"], results["error_result"]),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("run").reset_index(drop=True)


def build_reason_table(run_payloads: dict[str, dict[str, Any]], key: str, total_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, payload in run_payloads.items():
        error_result = payload["results"]["error_result"]
        df = error_result[key].copy()
        total = max(int(error_result["summary"][total_key]), 1)
        for _, row in df.iterrows():
            rows.append(
                {
                    "run": run_name,
                    "reason": row["reason"],
                    "count": int(row["count"]),
                    "ratio": float(row["count"] / total),
                }
            )
    return pd.DataFrame(rows).sort_values(["run", "count"], ascending=[True, False]).reset_index(drop=True)


def build_class_metrics(run_payloads: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for run_name, payload in run_payloads.items():
        df = payload["results"]["metrics_result"]["class_df"].copy()
        df["run"] = run_name
        frames.append(df)
    class_metrics = pd.concat(frames, ignore_index=True)

    baseline = class_metrics[class_metrics["run"] == "test"][
        ["class_name", "precision", "recall", "ap50", "ap50_95", "gt_count", "pred_count"]
    ].rename(
        columns={
            "precision": "baseline_precision",
            "recall": "baseline_recall",
            "ap50": "baseline_ap50",
            "ap50_95": "baseline_ap50_95",
            "gt_count": "baseline_gt_count",
            "pred_count": "baseline_pred_count",
        }
    )
    deltas = class_metrics.merge(baseline, on="class_name", how="left")
    deltas["delta_precision"] = deltas["precision"] - deltas["baseline_precision"]
    deltas["delta_recall"] = deltas["recall"] - deltas["baseline_recall"]
    deltas["delta_ap50"] = deltas["ap50"] - deltas["baseline_ap50"]
    deltas["delta_ap50_95"] = deltas["ap50_95"] - deltas["baseline_ap50_95"]
    deltas = deltas[deltas["run"] != "test"].reset_index(drop=True)
    return class_metrics, deltas


def build_error_class_table(run_payloads: dict[str, dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_name, payload in run_payloads.items():
        df = payload["results"]["error_result"]["per_class_df"].copy()
        df["run"] = run_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_p2_head_table(run_payloads: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, payload in run_payloads.items():
        head_df = payload["results"]["detection_head_result"]["head_summary_df"].copy()
        if head_df.empty:
            continue
        for _, row in head_df.iterrows():
            rows.append(
                {
                    "run": run_name,
                    "head_name": row["head_name"],
                    "head_stride": int(row["head_stride"]),
                    "assigned_ratio": float(row["assigned_ratio"]),
                    "effective_coverage_ratio": float(row["effective_coverage_ratio"]),
                    "range_miss_assigned_ratio": float(row["range_miss_assigned_ratio"]),
                    "boundary_assigned_ratio": float(row["boundary_assigned_ratio"]),
                    "effective_multi_overlap_ratio": float(row["effective_multi_overlap_ratio"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["run", "head_stride"]).reset_index(drop=True)


def top_class_changes(delta_df: pd.DataFrame, run_name: str, metric: str, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = delta_df[delta_df["run"] == run_name].copy()
    gains = subset.sort_values(metric, ascending=False).head(n)
    drops = subset.sort_values(metric, ascending=True).head(n)
    return gains, drops


def write_report(output_dir: Path, overall_df: pd.DataFrame, delta_df: pd.DataFrame, fn_reason_df: pd.DataFrame, fp_reason_df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# P2 Ablation Detailed Analysis")
    lines.append("")
    lines.append("This report uses `yolo11_analysis_app` back-end analysis. The app uses rotated polygon IoU when polygon labels and OBB predictions are available, so the relative trends are more important than exact agreement with Ultralytics' built-in `val` outputs.")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    for _, row in overall_df.sort_values("mAP50_95", ascending=False).iterrows():
        lines.append(
            f"- `{row['run']}`: mAP50-95={row['mAP50_95']:.4f}, mAP50={row['mAP50']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, fp={int(row['fp'])}, fn={int(row['fn'])}, "
            f"range_miss_ratio={row['range_miss_ratio']:.4f}, boundary_ratio={row['boundary_ratio']:.4f}"
        )

    lines.append("")
    lines.append("## Key Readings")
    lines.append("")

    test_row = overall_df[overall_df["run"] == "test"].iloc[0]
    p2_row = overall_df[overall_df["run"] == "yolo11s_obb_p2"].iloc[0]
    branch_row = overall_df[overall_df["run"] == "yolo11s_obb_p2_branch"].iloc[0]
    fusion_row = overall_df[overall_df["run"] == "yolo11s_obb_p2_fusion"].iloc[0]
    cbam_row = overall_df[overall_df["run"] == "yolo11s_obb_p2_cbam_p3"].iloc[0]

    lines.append(
        f"- The raw `p2` model reduces theoretical small-object range miss from {test_row['range_miss_ratio']:.2%} to {p2_row['range_miss_ratio']:.2%}, "
        f"but still drops mAP50-95 by {(test_row['mAP50_95'] - p2_row['mAP50_95']):.4f}. Coverage improves, quality does not."
    )
    lines.append(
        f"- `p2_branch` keeps the extra P2 detection head but avoids rewriting all higher paths. It is better than raw `p2`, "
        f"yet still below baseline. This isolates the damage to the shallow detection head itself plus its supervision side effects."
    )
    lines.append(
        f"- `p2_fusion` is stronger than `p2_branch`, which means shallow information is more useful as feature assistance for `P3` than as a direct detection output."
    )
    lines.append(
        f"- `p2_cbam_p3` is the best P2 variant. Compared with `p2_fusion`, it improves mAP50-95 by {(cbam_row['mAP50_95'] - fusion_row['mAP50_95']):.4f} and lowers boundary_ratio from {fusion_row['boundary_ratio']:.2%} to {cbam_row['boundary_ratio']:.2%}. "
        "This is consistent with the earlier heatmap finding that shallow features need denoising before fusion."
    )

    lines.append("")
    lines.append("## FN / FP Reasons")
    lines.append("")
    for run_name in ["test", "yolo11s_obb_p2", "yolo11s_obb_p2_branch", "yolo11s_obb_p2_fusion", "yolo11s_obb_p2_cbam_p3"]:
        fn_rows = fn_reason_df[fn_reason_df["run"] == run_name].sort_values("count", ascending=False).head(3)
        fp_rows = fp_reason_df[fp_reason_df["run"] == run_name].sort_values("count", ascending=False).head(3)
        fn_text = ", ".join(f"{row.reason}:{row.ratio:.1%}" for row in fn_rows.itertuples())
        fp_text = ", ".join(f"{row.reason}:{row.ratio:.1%}" for row in fp_rows.itertuples())
        lines.append(f"- `{run_name}` FN top: {fn_text}; FP top: {fp_text}")

    lines.append("")
    lines.append("## Class-Level Changes vs `test`")
    lines.append("")
    for run_name in ["yolo11s_obb_p2", "yolo11s_obb_p2_branch", "yolo11s_obb_p2_fusion", "yolo11s_obb_p2_cbam_p3"]:
        gains, drops = top_class_changes(delta_df, run_name, "delta_ap50_95")
        gain_text = ", ".join(f"{row.class_name}:{row.delta_ap50_95:+.3f}" for row in gains.itertuples())
        drop_text = ", ".join(f"{row.class_name}:{row.delta_ap50_95:+.3f}" for row in drops.itertuples())
        lines.append(f"- `{run_name}` gains: {gain_text}")
        lines.append(f"- `{run_name}` drops: {drop_text}")

    lines.append("")
    lines.append("## Direction")
    lines.append("")
    lines.append("- Do not continue with a direct P2 detection head as the main line. Better geometric coverage alone does not translate into better OBB accuracy here.")
    lines.append("- Keep exploring shallow feature denoising plus controlled fusion into `P3` only. The current evidence favors `P2 -> refine -> assist P3` over `P2 -> detect`.")
    lines.append("- The next structural step should be residual or gated fusion, for example `P3_new = P3 + alpha * F(P2_refined)`, with small initial `alpha`.")
    lines.append("- Add a size-stratified evaluation after each run. The key remaining question is whether shallow fusion helps small targets enough to justify any medium/large target regressions.")
    lines.append("- If you still want explicit P2 supervision, try an auxiliary training-only head with reduced loss weight, not a permanent inference head.")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_payloads: dict[str, dict[str, Any]] = {}
    for run_name in args.runs:
        if run_name not in RUN_SPECS:
            raise KeyError(f"Unknown run: {run_name}")
        run_payloads[run_name] = get_run_payload(
            run_name,
            force_recompute=args.force_recompute,
            device=args.device,
            batch_size=args.batch_size,
        )

    overall_df = build_overall_summary(run_payloads)
    fn_reason_df = build_reason_table(run_payloads, "fn_reason_df", "total_fn")
    fp_reason_df = build_reason_table(run_payloads, "fp_reason_df", "total_fp")
    class_metrics_df, delta_df = build_class_metrics(run_payloads)
    error_class_df = build_error_class_table(run_payloads)
    head_summary_df = build_p2_head_table(run_payloads)

    overall_df.to_csv(output_dir / "overall_summary.csv", index=False)
    fn_reason_df.to_csv(output_dir / "fn_reason_compare.csv", index=False)
    fp_reason_df.to_csv(output_dir / "fp_reason_compare.csv", index=False)
    class_metrics_df.to_csv(output_dir / "class_metrics_compare.csv", index=False)
    delta_df.to_csv(output_dir / "class_delta_vs_test.csv", index=False)
    error_class_df.to_csv(output_dir / "error_class_compare.csv", index=False)
    head_summary_df.to_csv(output_dir / "detection_head_compare.csv", index=False)
    write_report(output_dir, overall_df, delta_df, fn_reason_df, fp_reason_df)


if __name__ == "__main__":
    main()
