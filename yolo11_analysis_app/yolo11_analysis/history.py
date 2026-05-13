from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_directory, load_pickle, save_pickle, to_serializable, write_json

SECTION_LABELS = {
    "dataset": "Dataset",
    "detection_head": "Detection Head",
    "prediction": "Prediction",
    "error": "Error Analysis",
    "metrics": "Metrics",
}


def _basename(path_value: str | None) -> str:
    if not path_value:
        return "-"
    return Path(path_value).name


def _prediction_summary(prediction_payload: dict[str, Any] | None) -> dict[str, Any]:
    predictions = (prediction_payload or {}).get("predictions", {}) or {}
    confidence_values: list[float] = []
    total_predictions = 0
    for boxes in predictions.values():
        total_predictions += len(boxes)
        for box in boxes:
            confidence = getattr(box, "confidence", None)
            if confidence is not None:
                confidence_values.append(float(confidence))
    return {
        "image_count": len(predictions),
        "prediction_count": total_predictions,
        "avg_confidence": (sum(confidence_values) / len(confidence_values)) if confidence_values else None,
    }


def _build_signature(config: dict[str, Any], dataset_result, prediction_payload, error_result, metrics_result) -> str:
    signature_payload = {
        "config": config,
        "dataset_summary": (dataset_result or {}).get("summary") if isinstance(dataset_result, dict) else None,
        "prediction_summary": _prediction_summary(prediction_payload),
        "prediction_meta": (prediction_payload or {}).get("meta") if isinstance(prediction_payload, dict) else None,
        "error_summary": (error_result or {}).get("summary") if isinstance(error_result, dict) else None,
        "metrics_summary": (metrics_result or {}).get("summary") if isinstance(metrics_result, dict) else None,
    }
    text = json.dumps(to_serializable(signature_payload), ensure_ascii=False, sort_keys=True)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_analysis_snapshot(
    *,
    note: str,
    config: dict[str, Any],
    dataset_context,
    eval_context,
    dataset_result,
    detection_head_result,
    prediction_payload,
    error_result,
    metrics_result,
) -> dict[str, Any]:
    available_sections = {
        "dataset": dataset_result is not None,
        "detection_head": detection_head_result is not None,
        "prediction": prediction_payload is not None,
        "error": error_result is not None,
        "metrics": metrics_result is not None,
    }
    return {
        "snapshot_version": 1,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": note.strip(),
        "signature": _build_signature(config, dataset_result, prediction_payload, error_result, metrics_result),
        "config": config,
        "available_sections": available_sections,
        "contexts": {
            "dataset_context": dataset_context,
            "eval_context": eval_context,
        },
        "results": {
            "dataset_result": dataset_result,
            "detection_head_result": detection_head_result,
            "prediction_payload": prediction_payload,
            "error_result": error_result,
            "metrics_result": metrics_result,
        },
    }


def snapshot_has_results(snapshot: dict[str, Any] | None) -> bool:
    if not snapshot:
        return False
    return any(bool(value) for value in snapshot.get("available_sections", {}).values())


def _build_snapshot_meta(snapshot: dict[str, Any], run_id: str) -> dict[str, Any]:
    config = snapshot.get("config", {})
    dataset_context = (snapshot.get("contexts", {}) or {}).get("dataset_context")
    eval_context = (snapshot.get("contexts", {}) or {}).get("eval_context")
    results = snapshot.get("results", {}) or {}
    dataset_result = results.get("dataset_result") or {}
    error_result = results.get("error_result") or {}
    metrics_result = results.get("metrics_result") or {}

    dataset_summary = dataset_result.get("summary", {}) if isinstance(dataset_result, dict) else {}
    error_summary = error_result.get("summary", {}) if isinstance(error_result, dict) else {}
    metrics_summary = metrics_result.get("summary", {}) if isinstance(metrics_result, dict) else {}
    prediction_summary = _prediction_summary(results.get("prediction_payload"))
    available_sections = snapshot.get("available_sections", {})
    available_labels = [label for key, label in SECTION_LABELS.items() if available_sections.get(key)]

    return {
        "run_id": run_id,
        "saved_at": snapshot.get("saved_at"),
        "note": snapshot.get("note", ""),
        "signature": snapshot.get("signature"),
        "weights_name": _basename(config.get("weights_path")),
        "data_yaml_name": _basename(config.get("data_yaml_path")),
        "image_dir_name": _basename(config.get("image_dir")),
        "label_dir_name": _basename(config.get("label_dir")),
        "dataset_name": getattr(dataset_context, "dataset_name", None) or getattr(eval_context, "dataset_name", None) or "-",
        "split": config.get("split") or getattr(dataset_context, "split", "-"),
        "eval_split": config.get("eval_split") or getattr(eval_context, "split", "-"),
        "imgsz": config.get("imgsz"),
        "conf": config.get("conf"),
        "iou": config.get("iou"),
        "match_iou": config.get("match_iou"),
        "available_sections": available_sections,
        "available_labels": available_labels,
        "available_text": ", ".join(available_labels),
        "dataset_images": dataset_summary.get("num_images"),
        "dataset_boxes": dataset_summary.get("num_boxes"),
        "prediction_images": prediction_summary.get("image_count"),
        "prediction_count": prediction_summary.get("prediction_count"),
        "avg_confidence": prediction_summary.get("avg_confidence"),
        "tp": error_summary.get("total_tp"),
        "fp": error_summary.get("total_fp"),
        "fn": error_summary.get("total_fn"),
        "precision": metrics_summary.get("precision"),
        "recall": metrics_summary.get("recall"),
        "mAP50": metrics_summary.get("mAP50"),
        "mAP50_95": metrics_summary.get("mAP50_95"),
    }


def save_analysis_snapshot(history_root: str | Path, snapshot: dict[str, Any]) -> Path:
    history_dir = ensure_directory(history_root)
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{snapshot['signature'][:8]}"
    run_dir = ensure_directory(history_dir / run_id)
    meta = _build_snapshot_meta(snapshot, run_id=run_id)
    write_json(run_dir / "meta.json", meta)
    save_pickle(run_dir / "snapshot.pkl", snapshot)
    return run_dir


def list_saved_analysis_runs(history_root: str | Path) -> list[dict[str, Any]]:
    history_dir = Path(history_root).expanduser().resolve()
    if not history_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for run_dir in sorted(history_dir.iterdir(), reverse=True):
        meta_path = run_dir / "meta.json"
        if not run_dir.is_dir() or not meta_path.exists():
            continue
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["run_dir"] = str(run_dir)
            records.append(payload)
        except Exception:
            continue
    return records


def load_saved_analysis_snapshot(run_dir: str | Path) -> dict[str, Any]:
    return load_pickle(Path(run_dir).expanduser().resolve() / "snapshot.pkl")


def build_history_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "run_id": record.get("run_id"),
                "Saved At": record.get("saved_at"),
                "Note": record.get("note") or "-",
                "Weights": record.get("weights_name") or "-",
                "Dataset": record.get("dataset_name") or "-",
                "Modules": record.get("available_text") or "-",
                "Images": record.get("dataset_images"),
                "Boxes": record.get("dataset_boxes"),
                "Predictions": record.get("prediction_count"),
                "Precision": record.get("precision"),
                "Recall": record.get("recall"),
                "mAP@0.5": record.get("mAP50"),
                "mAP@0.5:0.95": record.get("mAP50_95"),
                "TP": record.get("tp"),
                "FP": record.get("fp"),
                "FN": record.get("fn"),
            }
        )
    return pd.DataFrame(rows)


def format_history_option(record: dict[str, Any]) -> str:
    note = (record.get("note") or "").strip()
    title = note if note else f"{record.get('weights_name', '-')}"
    dataset_name = record.get("dataset_name") or "-"
    saved_at = record.get("saved_at") or "-"
    metrics_text = ""
    if record.get("mAP50") is not None:
        metrics_text = f" | mAP50={float(record['mAP50']):.4f}"
    return f"{saved_at} | {title} | {dataset_name}{metrics_text}"
