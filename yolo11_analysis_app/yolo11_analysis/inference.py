from __future__ import annotations

import gc
from functools import lru_cache
from pathlib import Path

import numpy as np

from .exceptions import DependencyError, InferenceError
from .schemas import BoxRecord, InferenceConfig
from .utils import ProgressCallback, emit_progress


@lru_cache(maxsize=4)
def load_yolo_model(weights_path: str):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise DependencyError(
            "未检测到 ultralytics。请先安装 requirements.txt 中的依赖。"
        ) from exc

    path = Path(weights_path).expanduser().resolve()
    if not path.exists():
        raise InferenceError(f"权重文件不存在: {path}")
    return YOLO(str(path))


def _build_detection_head_info(strides: list[int], *, source: str, warning: str | None = None) -> dict:
    heads = []
    for index, stride in enumerate(sorted(dict.fromkeys(int(item) for item in strides if int(item) > 0))):
        heads.append(
            {
                "head_index": index,
                "head_name": f"P{index + 3}/{stride}",
                "stride": int(stride),
            }
        )
    payload = {"source": source, "heads": heads}
    if warning:
        payload["warning"] = warning
    return payload


def get_detection_head_info(weights_path: str | None, default_strides: tuple[int, ...] = (8, 16, 32)) -> dict:
    if not weights_path:
        return _build_detection_head_info(list(default_strides), source="default")

    try:
        model = load_yolo_model(weights_path)
        model_core = model.model
        stride_attr = getattr(model_core, "stride", None)
        if stride_attr is None and hasattr(model_core, "model") and len(getattr(model_core, "model", [])) > 0:
            stride_attr = getattr(model_core.model[-1], "stride", None)
        if stride_attr is None:
            return _build_detection_head_info(list(default_strides), source="default", warning="模型未暴露检测头步长，已回退到默认 stride。")

        if hasattr(stride_attr, "tolist"):
            raw_values = stride_attr.tolist()
        elif isinstance(stride_attr, (list, tuple)):
            raw_values = list(stride_attr)
        else:
            raw_values = [stride_attr]

        strides = [int(round(float(item))) for item in raw_values if float(item) > 0]
        if not strides:
            return _build_detection_head_info(list(default_strides), source="default", warning="模型步长为空，已回退到默认 stride。")
        return _build_detection_head_info(strides, source="model")
    except Exception as exc:
        return _build_detection_head_info(list(default_strides), source="default", warning=f"检测头步长解析失败，已回退默认 stride: {exc}")


def _axis_aligned_xyxy_from_polygon(polygon: np.ndarray) -> tuple[float, float, float, float]:
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _clear_torch_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda out of memory" in message


def result_to_records(result, names: dict) -> list[BoxRecord]:
    image_path = str(Path(result.path).resolve())
    image_id = Path(image_path).stem
    records: list[BoxRecord] = []

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        for index in range(len(xyxy)):
            class_id = int(cls[index])
            class_name = str(names.get(class_id, class_id))
            x1, y1, x2, y2 = xyxy[index].tolist()
            records.append(
                BoxRecord(
                    image_id=image_id,
                    image_path=image_path,
                    class_id=class_id,
                    class_name=class_name,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(conf[index]),
                    source="pred",
                    meta={"pred_format": "bbox"},
                )
            )
        return records

    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        conf = obb.conf.cpu().numpy()
        cls = obb.cls.cpu().numpy()
        polygons_raw = getattr(obb, "xyxyxyxy", None)
        polygons = polygons_raw.cpu().numpy() if polygons_raw is not None else None
        xyxy_raw = getattr(obb, "xyxy", None)
        xyxy = xyxy_raw.cpu().numpy() if xyxy_raw is not None else None

        if xyxy is None and polygons is not None:
            xyxy = np.asarray([_axis_aligned_xyxy_from_polygon(polygon) for polygon in polygons], dtype=np.float32)

        if xyxy is not None:
            for index in range(len(xyxy)):
                class_id = int(cls[index])
                class_name = str(names.get(class_id, class_id))
                x1, y1, x2, y2 = xyxy[index].tolist()
                meta = {"pred_format": "obb_projected_bbox"}
                if polygons is not None:
                    meta["polygon"] = polygons[index].reshape(-1).tolist()
                records.append(
                    BoxRecord(
                        image_id=image_id,
                        image_path=image_path,
                        class_id=class_id,
                        class_name=class_name,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=float(conf[index]),
                        source="pred",
                        meta=meta,
                    )
                )
        return records

    return records


def run_inference(
    image_paths: list[str],
    config: InferenceConfig,
    progress_callback: ProgressCallback = None,
) -> dict:
    if not image_paths:
        return {"predictions": {}, "meta": {"num_images": 0, "device": config.device, "batch_size": config.batch_size, "cpu_fallback_used": False}}

    model = load_yolo_model(config.weights_path)
    names = getattr(model, "names", {}) or {}
    predictions: dict[str, list[BoxRecord]] = {}
    total_images = len(image_paths)
    current_device = config.device
    current_batch_size = max(1, int(config.batch_size))
    cpu_fallback_used = False
    processed_images = 0
    oom_retries = 0

    while processed_images < total_images:
        batch_paths = image_paths[processed_images : processed_images + current_batch_size]
        emit_progress(progress_callback, processed_images / max(total_images, 1), f"推理中: {processed_images}/{total_images} | device={current_device} | batch={current_batch_size}")
        try:
            results = model.predict(
                source=batch_paths,
                imgsz=config.imgsz,
                conf=config.conf,
                iou=config.iou,
                device=current_device,
                max_det=config.max_det,
                batch=current_batch_size,
                verbose=False,
                stream=True,
            )
            count = 0
            for result in results:
                image_path = str(Path(result.path).resolve())
                predictions[image_path] = result_to_records(result, names)
                count += 1
            processed_images += count
            _clear_torch_cache()
        except Exception as exc:
            if _is_cuda_oom(exc):
                oom_retries += 1
                _clear_torch_cache()
                if current_batch_size > 1:
                    current_batch_size = 1
                    emit_progress(progress_callback, processed_images / max(total_images, 1), "CUDA 显存不足，已降为单图推理后重试")
                    continue
                if current_device != "cpu" and config.cpu_fallback:
                    current_device = "cpu"
                    cpu_fallback_used = True
                    emit_progress(progress_callback, processed_images / max(total_images, 1), "CUDA 显存不足，已切换到 CPU 继续推理")
                    continue
                raise InferenceError(
                    "推理失败: CUDA 显存不足。建议降低 imgsz、减小 batch_size，或改用 CPU。"
                ) from exc
            raise InferenceError(f"推理失败: {exc}") from exc

    emit_progress(progress_callback, 1.0, f"推理完成: {total_images}/{total_images} | device={current_device}")
    return {
        "predictions": predictions,
        "meta": {
            "num_images": total_images,
            "device_requested": config.device,
            "device_used": current_device,
            "batch_size_requested": config.batch_size,
            "batch_size_used": current_batch_size,
            "cpu_fallback_used": cpu_fallback_used,
            "oom_retries": oom_retries,
        },
    }


def predictions_to_rows(predictions: dict[str, list[BoxRecord]]) -> list[dict]:
    rows: list[dict] = []
    for image_path, boxes in predictions.items():
        for box in boxes:
            row = box.to_dict()
            row["image_name"] = Path(image_path).name
            rows.append(row)
    return rows
