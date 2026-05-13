from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .exceptions import ConfigurationError

LOGGER_NAME = "yolo11_analysis"
ProgressCallback = Optional[Callable[[float, str], None]]


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_existing_path(path_value: str | Path | None, description: str) -> Optional[Path]:
    if path_value in (None, ""):
        return None
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise ConfigurationError(f"{description} 不存在: {path}")
    return path


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return to_serializable(vars(value))
    return str(value)


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, ensure_ascii=False, indent=2)
    return target


def save_pickle(path: str | Path, payload: Any) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load_pickle(path: str | Path) -> Any:
    target = Path(path).expanduser().resolve()
    with target.open("rb") as handle:
        return pickle.load(handle)


def summarize_numeric(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p5": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def choose_default_device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_threshold_text(value: str, default: tuple[float, float]) -> tuple[float, float]:
    if not value.strip():
        return default
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ConfigurationError("目标尺寸阈值请输入两个数字，格式如: 32, 96")
    try:
        small_thr = float(parts[0])
        medium_thr = float(parts[1])
    except ValueError as exc:
        raise ConfigurationError("目标尺寸阈值必须是数字") from exc
    if small_thr <= 0 or medium_thr <= 0 or small_thr >= medium_thr:
        raise ConfigurationError("目标尺寸阈值需满足 0 < small < medium")
    return small_thr, medium_thr


def emit_progress(callback: ProgressCallback, fraction: float, message: str) -> None:
    if callback is None:
        return
    callback(max(0.0, min(1.0, float(fraction))), message)


def emit_progress_step(callback: ProgressCallback, current: int, total: int, message: str) -> None:
    if total <= 0:
        emit_progress(callback, 1.0, message)
        return
    emit_progress(callback, current / float(total), message)
