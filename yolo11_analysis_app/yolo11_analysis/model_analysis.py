from __future__ import annotations

from typing import Any

from .exceptions import DependencyError, InferenceError
from .inference import load_yolo_model, result_to_records
from .schemas import FeatureMapResult, InferenceConfig


def _describe_output_shape(output: Any) -> str:
    if hasattr(output, "shape"):
        return str(tuple(int(item) for item in output.shape))
    if isinstance(output, (list, tuple)):
        return "[" + ", ".join(_describe_output_shape(item) for item in output) + "]"
    if isinstance(output, dict):
        return "{" + ", ".join(f"{key}: {_describe_output_shape(value)}" for key, value in output.items()) + "}"
    return str(type(output).__name__)


def _extract_feature_tensor(output: Any):
    try:
        import torch
    except Exception as exc:
        raise DependencyError("未检测到 torch。特征图分析需要 torch。") from exc

    if isinstance(output, torch.Tensor):
        tensor = output.detach().cpu()
        if tensor.ndim == 4:
            return tensor[0]
        if tensor.ndim == 3:
            return tensor
        raise InferenceError("所选层输出不是 3D/4D 特征图，无法直接可视化。")
    if isinstance(output, (list, tuple)):
        for item in output:
            try:
                return _extract_feature_tensor(item)
            except InferenceError:
                continue
    if isinstance(output, dict):
        for item in output.values():
            try:
                return _extract_feature_tensor(item)
            except InferenceError:
                continue
    raise InferenceError("未能从该层输出中提取可视化特征图。")


def _classify_stage(index: int, backbone_len: int, detect_indices: set[int]) -> str:
    if index < backbone_len:
        return "backbone"
    if index in detect_indices:
        return "head"
    return "neck"


def model_summary(weights_path: str, image_path: str, imgsz: int, device: str) -> list[dict]:
    model = load_yolo_model(weights_path)
    model_core = model.model
    if not hasattr(model_core, "model"):
        raise InferenceError("当前权重对应的模型结构不包含可遍历的顶层模块。")

    modules = list(enumerate(model_core.model))
    yaml_cfg = getattr(model_core, "yaml", {}) or {}
    backbone_len = len(yaml_cfg.get("backbone", []))
    detect_indices = {index for index, module in modules if "Detect" in module.__class__.__name__}
    output_shapes: dict[int, str] = {}
    hooks = []

    def register_hook(index: int):
        def hook(_module, _inputs, output):
            output_shapes[index] = _describe_output_shape(output)

        return hook

    for index, module in modules:
        hooks.append(module.register_forward_hook(register_hook(index)))

    try:
        model.predict(
            source=[image_path],
            imgsz=imgsz,
            conf=0.25,
            iou=0.7,
            device=device,
            max_det=50,
            verbose=False,
            stream=False,
        )
    except Exception as exc:
        raise InferenceError(f"模型摘要前向执行失败: {exc}") from exc
    finally:
        for hook in hooks:
            hook.remove()

    rows: list[dict] = []
    for index, module in modules:
        rows.append(
            {
                "layer_index": index,
                "layer_name": getattr(module, "type", module.__class__.__name__),
                "module_type": module.__class__.__name__,
                "stage": _classify_stage(index, backbone_len, detect_indices),
                "params": int(sum(parameter.numel() for parameter in module.parameters())),
                "output_shape": output_shapes.get(index, "n/a"),
            }
        )
    return rows


def capture_feature_map(
    image_path: str,
    config: InferenceConfig,
    layer_index: int,
) -> FeatureMapResult:
    model = load_yolo_model(config.weights_path)
    model_core = model.model
    if not hasattr(model_core, "model"):
        raise InferenceError("当前模型不支持顶层特征图捕获。")
    modules = list(enumerate(model_core.model))
    if layer_index < 0 or layer_index >= len(modules):
        raise InferenceError("指定层索引超出范围。")

    yaml_cfg = getattr(model_core, "yaml", {}) or {}
    backbone_len = len(yaml_cfg.get("backbone", []))
    detect_indices = {index for index, module in modules if "Detect" in module.__class__.__name__}

    target_index, target_module = modules[layer_index]
    captured: dict[str, Any] = {}

    def hook(_module, _inputs, output):
        captured["output"] = output

    handle = target_module.register_forward_hook(hook)
    try:
        results = model.predict(
            source=[image_path],
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            max_det=config.max_det,
            verbose=False,
            stream=False,
        )
    except Exception as exc:
        raise InferenceError(f"特征图提取失败: {exc}") from exc
    finally:
        handle.remove()

    if "output" not in captured:
        raise InferenceError("未捕获到指定层输出。")

    tensor = _extract_feature_tensor(captured["output"])
    feature_map = tensor.numpy()

    names = getattr(model, "names", {}) or {}
    detections = result_to_records(results[0], names)

    return FeatureMapResult(
        layer_index=target_index,
        layer_name=getattr(target_module, "type", target_module.__class__.__name__),
        stage=_classify_stage(target_index, backbone_len, detect_indices),
        feature_map=feature_map,
        detections=detections,
    )
