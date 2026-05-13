from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import streamlit as st

from yolo11_analysis.dataset_analysis import analyze_dataset
from yolo11_analysis.detection_head_analysis import analyze_detection_heads
from yolo11_analysis.error_analysis import analyze_errors
from yolo11_analysis.exceptions import AnalysisError
from yolo11_analysis.history import (
    build_analysis_snapshot,
    build_history_dataframe,
    format_history_option,
    list_saved_analysis_runs,
    load_saved_analysis_snapshot,
    save_analysis_snapshot,
    snapshot_has_results,
)
from yolo11_analysis.inference import get_detection_head_info, load_yolo_model, run_inference
from yolo11_analysis.io import build_dataset_context
from yolo11_analysis.metrics import analyze_metrics
from yolo11_analysis.model_analysis import capture_feature_map, model_summary
from yolo11_analysis.pages import (
    render_dataset_page,
    render_detection_head_page,
    render_error_page,
    render_metrics_page,
    render_model_page,
    render_prediction_page,
)
from yolo11_analysis.schemas import InferenceConfig
from yolo11_analysis.utils import (
    choose_default_device,
    ensure_directory,
    load_pickle,
    parse_threshold_text,
    save_pickle,
    to_serializable,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

st.set_page_config(page_title="YOLO11 检测分析应用", layout="wide")
st.title("目标检测数据集与模型分析应用")
st.caption("面向 Ultralytics YOLO11 的本地分析工具，支持数据集分析、检测头分析、预测分析、漏检/虚检分析、指标分析、模型结构与特征图可视化。")

HISTORY_STATE_KEYS = [
    "dataset_case_image",
    "dataset_bucket_class",
    "prediction_preview_image",
    "prediction_class_filter",
    "error_case_image",
    "fn_class_filter",
    "fn_reason_filter",
    "fp_class_filter",
    "fp_reason_filter",
    "error_class_filter",
    "head_class_select",
    "head_select",
    "model_page_image_select",
    "feature_layer_select",
    "model_summary_rows",
    "feature_map_result",
]


def _persist_uploaded_file(uploaded_file, target_dir: Path) -> str:
    ensure_directory(target_dir)
    suffix = Path(uploaded_file.name).suffix
    safe_name = f"{hashlib.md5(uploaded_file.name.encode('utf-8')).hexdigest()[:8]}_{Path(uploaded_file.name).stem}{suffix}"
    target_path = target_dir / safe_name
    target_path.write_bytes(uploaded_file.getbuffer())
    return str(target_path.resolve())


def _persist_uploaded_files(uploaded_files, target_dir: Path) -> str | None:
    if not uploaded_files:
        return None
    ensure_directory(target_dir)
    for uploaded_file in uploaded_files:
        target_path = target_dir / Path(uploaded_file.name).name
        target_path.write_bytes(uploaded_file.getbuffer())
    return str(target_dir.resolve())


def _cache_key(*parts: Any) -> str:
    text = json.dumps(to_serializable(parts), ensure_ascii=False, sort_keys=True)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _disk_cache_file(cache_root: Path, key: str) -> Path:
    return cache_root / f"{key}.pkl"


def _cached_compute(label: str, key: str, cache_root: Path, factory):
    cache_file = _disk_cache_file(cache_root, key)
    if cache_file.exists():
        return load_pickle(cache_file)

    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0.0, text=label)

    def progress_callback(fraction: float, message: str):
        progress_bar.progress(max(0.0, min(1.0, float(fraction))), text=message or label)

    try:
        result = factory(progress_callback)
        progress_bar.progress(1.0, text=f"{label} 完成")
        save_pickle(cache_file, result)
        return result
    finally:
        progress_placeholder.empty()


def _resolve_uploaded_image(uploaded_file, upload_root: Path) -> str | None:
    if uploaded_file is None:
        return None
    return _persist_uploaded_file(uploaded_file, upload_root / "model_images")


def _clear_view_state() -> None:
    for key in HISTORY_STATE_KEYS:
        st.session_state.pop(key, None)


with st.sidebar:
    st.header("输入与参数")
    st.caption("长任务会显示进度条，并优先使用磁盘缓存；推理默认使用小批处理，降低 OOM 风险。")
    weights_path_input = st.text_input("模型权重路径 (.pt)", value="")
    data_yaml_input = st.text_input("data.yaml 路径", value="")
    image_dir_input = st.text_input("图像目录", value="")
    label_dir_input = st.text_input("标签目录", value="")
    eval_image_dir_input = st.text_input("可选验证/测试图像目录", value="")
    eval_label_dir_input = st.text_input("可选验证/测试标签目录", value="")

    split = st.selectbox("数据集分析 split", ["train", "val", "test"], index=1)
    eval_split = st.selectbox("评估 split", ["val", "test", "train"], index=0)

    st.markdown("**上传文件**")
    uploaded_weights = st.file_uploader("上传权重", type=["pt"], key="uploaded_weights")
    uploaded_yaml = st.file_uploader("上传 data.yaml", type=["yaml", "yml"], key="uploaded_yaml")
    uploaded_images = st.file_uploader(
        "上传图像文件（可多选）",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="uploaded_images",
    )
    uploaded_labels = st.file_uploader(
        "上传标签文件（YOLO txt，可多选）",
        type=["txt"],
        accept_multiple_files=True,
        key="uploaded_labels",
    )

    st.markdown("**推理参数**")
    imgsz = int(st.number_input("imgsz", min_value=32, max_value=4096, value=640, step=32))
    conf = float(st.slider("conf", min_value=0.0, max_value=1.0, value=0.25, step=0.01))
    iou = float(st.slider("推理 NMS IoU", min_value=0.0, max_value=1.0, value=0.7, step=0.01))
    match_iou = float(st.slider("GT/预测匹配 IoU", min_value=0.1, max_value=1.0, value=0.5, step=0.01))
    device = st.text_input("device", value=choose_default_device())
    max_det = int(st.number_input("max_det", min_value=1, max_value=3000, value=300, step=10))
    batch_size = int(st.number_input("batch_size", min_value=1, max_value=64, value=1, step=1))
    cpu_fallback = st.checkbox("CUDA OOM 时自动切换到 CPU", value=True)

    st.markdown("**目标尺寸规则**")
    size_metric = st.selectbox("尺寸度量", ["sqrt_area", "area", "width", "height", "long_side", "short_side"], index=0)
    size_threshold_text = st.text_input("small, medium 阈值", value="32, 96")

    output_dir = st.text_input("分析结果输出目录", value=str((Path.cwd() / "outputs").resolve()))

output_root = ensure_directory(output_dir)
cache_root = ensure_directory(output_root / ".cache")
upload_root = ensure_directory(output_root / "uploads")
history_root = ensure_directory(output_root / "analysis_history")
history_records = list_saved_analysis_runs(history_root)
history_record_map = {record["run_id"]: record for record in history_records}
history_df = build_history_dataframe(history_records)
history_display_df = history_df.drop(columns=["run_id"], errors="ignore") if not history_df.empty else history_df

with st.sidebar:
    st.markdown("**运行历史**")
    history_feedback = st.empty()
    feedback_message = st.session_state.pop("_history_feedback_message", None)
    if feedback_message:
        level, message = feedback_message
        getattr(history_feedback, level)(message)

    history_note = st.text_input("保存备注（可选）", value="", key="analysis_history_note")
    view_mode = st.radio("查看来源", ["当前运行", "历史运行"], index=0, key="analysis_view_mode")

    history_options = list(history_record_map.keys())
    selected_history_run_id = None
    if history_options:
        if st.session_state.get("analysis_history_selected_run") not in history_options:
            st.session_state["analysis_history_selected_run"] = history_options[0]
        selected_history_run_id = st.selectbox(
            "选择历史运行",
            history_options,
            format_func=lambda item: format_history_option(history_record_map[item]),
            key="analysis_history_selected_run",
            disabled=view_mode != "历史运行",
        )
    else:
        st.caption("当前还没有保存过完整分析快照。")

    save_snapshot_clicked = st.button(
        "保存当前分析快照",
        use_container_width=True,
        disabled=view_mode == "历史运行" and bool(history_options),
    )
    if not history_display_df.empty:
        with st.expander("最近保存记录", expanded=False):
            st.dataframe(history_display_df, use_container_width=True, hide_index=True)

    st.caption(f"磁盘缓存目录: {cache_root}")
    if st.button("清空缓存"):
        if cache_root.exists():
            shutil.rmtree(cache_root, ignore_errors=True)
        ensure_directory(cache_root)
        st.session_state.pop("model_summary_rows", None)
        st.session_state.pop("feature_map_result", None)
        load_yolo_model.cache_clear()
        st.rerun()

is_history_view = view_mode == "历史运行" and selected_history_run_id is not None
view_state_token = f"history:{selected_history_run_id}" if is_history_view else "current"
if st.session_state.get("_analysis_view_token") != view_state_token:
    _clear_view_state()
    st.session_state["_analysis_view_token"] = view_state_token

active_history_record = history_record_map.get(selected_history_run_id) if is_history_view else None
current_snapshot = None
size_metric_internal = size_metric
small_thr = 32.0
medium_thr = 96.0

effective_weights_path = None
effective_data_yaml_path = None
effective_image_dir = None
effective_label_dir = None
effective_eval_image_dir = None
effective_eval_label_dir = None
effective_imgsz = imgsz
effective_conf = conf
effective_iou = iou
effective_match_iou = match_iou
effective_device = device
effective_max_det = max_det
effective_batch_size = batch_size
effective_cpu_fallback = cpu_fallback

dataset_context = None
eval_context = None
dataset_result = None
detection_head_result = None
prediction_payload = None
predictions = None
error_result = None
metrics_result = None
inference_config = InferenceConfig(
    weights_path="",
    imgsz=effective_imgsz,
    conf=effective_conf,
    iou=effective_iou,
    device=effective_device,
    max_det=effective_max_det,
    batch_size=effective_batch_size,
    cpu_fallback=effective_cpu_fallback,
)

if is_history_view and active_history_record is not None:
    try:
        selected_snapshot = load_saved_analysis_snapshot(active_history_record["run_dir"])
        snapshot_config = selected_snapshot.get("config", {})
        snapshot_contexts = selected_snapshot.get("contexts", {})
        snapshot_results = selected_snapshot.get("results", {})

        effective_weights_path = snapshot_config.get("weights_path")
        effective_data_yaml_path = snapshot_config.get("data_yaml_path")
        effective_image_dir = snapshot_config.get("image_dir")
        effective_label_dir = snapshot_config.get("label_dir")
        effective_eval_image_dir = snapshot_config.get("eval_image_dir")
        effective_eval_label_dir = snapshot_config.get("eval_label_dir")
        effective_imgsz = int(snapshot_config.get("imgsz", imgsz))
        effective_conf = float(snapshot_config.get("conf", conf))
        effective_iou = float(snapshot_config.get("iou", iou))
        effective_match_iou = float(snapshot_config.get("match_iou", match_iou))
        effective_device = str(snapshot_config.get("device", device))
        effective_max_det = int(snapshot_config.get("max_det", max_det))
        effective_batch_size = int(snapshot_config.get("batch_size", batch_size))
        effective_cpu_fallback = bool(snapshot_config.get("cpu_fallback", cpu_fallback))
        size_metric_internal = snapshot_config.get("size_metric", size_metric)
        small_thr = float(snapshot_config.get("small_thr", 32.0))
        medium_thr = float(snapshot_config.get("medium_thr", 96.0))

        inference_config = InferenceConfig(
            weights_path=effective_weights_path or "",
            imgsz=effective_imgsz,
            conf=effective_conf,
            iou=effective_iou,
            device=effective_device,
            max_det=effective_max_det,
            batch_size=effective_batch_size,
            cpu_fallback=effective_cpu_fallback,
        )

        dataset_context = snapshot_contexts.get("dataset_context")
        eval_context = snapshot_contexts.get("eval_context")
        dataset_result = snapshot_results.get("dataset_result")
        detection_head_result = snapshot_results.get("detection_head_result")
        prediction_payload = snapshot_results.get("prediction_payload")
        predictions = (prediction_payload or {}).get("predictions", {})
        error_result = snapshot_results.get("error_result")
        metrics_result = snapshot_results.get("metrics_result")
    except Exception as exc:
        st.error(f"加载历史运行失败: {exc}")
else:
    effective_weights_path = _persist_uploaded_file(uploaded_weights, upload_root / "weights") if uploaded_weights else weights_path_input.strip() or None
    if uploaded_yaml:
        effective_data_yaml_path = _persist_uploaded_file(uploaded_yaml, upload_root / "configs")
    else:
        effective_data_yaml_path = data_yaml_input.strip() or None

    uploaded_image_dir = _persist_uploaded_files(uploaded_images, upload_root / "images") if uploaded_images else None
    uploaded_label_dir = _persist_uploaded_files(uploaded_labels, upload_root / "labels") if uploaded_labels else None
    effective_image_dir = uploaded_image_dir or image_dir_input.strip() or None
    effective_label_dir = uploaded_label_dir or label_dir_input.strip() or None
    effective_eval_image_dir = eval_image_dir_input.strip() or effective_image_dir
    effective_eval_label_dir = eval_label_dir_input.strip() or effective_label_dir

    try:
        small_thr, medium_thr = parse_threshold_text(size_threshold_text, (32.0, 96.0))
    except AnalysisError as exc:
        st.error(str(exc))
        st.stop()

    inference_config = InferenceConfig(
        weights_path=effective_weights_path or "",
        imgsz=effective_imgsz,
        conf=effective_conf,
        iou=effective_iou,
        device=effective_device,
        max_det=effective_max_det,
        batch_size=effective_batch_size,
        cpu_fallback=effective_cpu_fallback,
    )
    detection_head_info = get_detection_head_info(effective_weights_path)

    try:
        if effective_data_yaml_path or effective_image_dir:
            dataset_context = _cached_compute(
                "加载数据集信息",
                _cache_key("dataset_context", effective_data_yaml_path, effective_image_dir, effective_label_dir, split),
                cache_root,
                lambda _cb: build_dataset_context(
                    data_yaml=effective_data_yaml_path,
                    image_dir=effective_image_dir,
                    label_dir=effective_label_dir,
                    split=split,
                ),
            )
            dataset_result = _cached_compute(
                "统计数据集分布",
                _cache_key(
                    "dataset_result",
                    dataset_context.__dict__,
                    effective_imgsz,
                    size_metric_internal,
                    small_thr,
                    medium_thr,
                    detection_head_info,
                ),
                cache_root,
                lambda cb: analyze_dataset(
                    dataset_context,
                    effective_imgsz,
                    size_metric_internal,
                    small_thr,
                    medium_thr,
                    detection_head_info=detection_head_info,
                    progress_callback=cb,
                ),
            )
            detection_head_result = _cached_compute(
                "统计检测头覆盖",
                _cache_key(
                    "detection_head_result",
                    dataset_context.__dict__,
                    effective_imgsz,
                    size_metric_internal,
                    small_thr,
                    medium_thr,
                    detection_head_info,
                ),
                cache_root,
                lambda _cb: analyze_detection_heads(dataset_result),
            )

        if effective_data_yaml_path or effective_eval_image_dir:
            eval_context = _cached_compute(
                "加载评估数据集",
                _cache_key("eval_context", effective_data_yaml_path, effective_eval_image_dir, effective_eval_label_dir, eval_split),
                cache_root,
                lambda _cb: build_dataset_context(
                    data_yaml=effective_data_yaml_path,
                    image_dir=effective_eval_image_dir,
                    label_dir=effective_eval_label_dir,
                    split=eval_split,
                ),
            )

        if effective_weights_path and eval_context is not None:
            prediction_payload = _cached_compute(
                "执行模型推理",
                _cache_key(
                    "predictions",
                    effective_weights_path,
                    tuple(eval_context.image_paths),
                    effective_imgsz,
                    effective_conf,
                    effective_iou,
                    effective_device,
                    effective_max_det,
                    effective_batch_size,
                    effective_cpu_fallback,
                ),
                cache_root,
                lambda cb: run_inference(eval_context.image_paths, inference_config, progress_callback=cb),
            )
            predictions = prediction_payload.get("predictions", {})

        if predictions is not None and eval_context is not None:
            error_result = _cached_compute(
                "执行漏检/虚检分析",
                _cache_key(
                    "error_result",
                    eval_context.__dict__,
                    tuple(eval_context.image_paths),
                    effective_weights_path,
                    effective_imgsz,
                    effective_conf,
                    effective_iou,
                    effective_device,
                    effective_max_det,
                    effective_batch_size,
                    effective_cpu_fallback,
                    effective_match_iou,
                ),
                cache_root,
                lambda cb: analyze_errors(
                    eval_context,
                    predictions,
                    effective_match_iou,
                    size_metric_internal,
                    small_thr,
                    medium_thr,
                    progress_callback=cb,
                ),
            )
            metrics_result = _cached_compute(
                "计算指标",
                _cache_key(
                    "metrics_result",
                    eval_context.__dict__,
                    tuple(eval_context.image_paths),
                    effective_weights_path,
                    effective_imgsz,
                    effective_conf,
                    effective_iou,
                    effective_device,
                    effective_max_det,
                    effective_batch_size,
                    effective_cpu_fallback,
                    effective_match_iou,
                ),
                cache_root,
                lambda cb: analyze_metrics(
                    eval_context,
                    predictions,
                    effective_match_iou,
                    gt_by_image=error_result.get("gt_by_image") if error_result is not None else None,
                    progress_callback=cb,
                ),
            )
    except AnalysisError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"运行过程中出现异常: {exc}")

    current_snapshot = build_analysis_snapshot(
        note=history_note,
        config={
            "weights_path": effective_weights_path,
            "data_yaml_path": effective_data_yaml_path,
            "image_dir": effective_image_dir,
            "label_dir": effective_label_dir,
            "eval_image_dir": effective_eval_image_dir,
            "eval_label_dir": effective_eval_label_dir,
            "split": split,
            "eval_split": eval_split,
            "imgsz": effective_imgsz,
            "conf": effective_conf,
            "iou": effective_iou,
            "match_iou": effective_match_iou,
            "device": effective_device,
            "max_det": effective_max_det,
            "batch_size": effective_batch_size,
            "cpu_fallback": effective_cpu_fallback,
            "size_metric": size_metric_internal,
            "small_thr": small_thr,
            "medium_thr": medium_thr,
        },
        dataset_context=dataset_context,
        eval_context=eval_context,
        dataset_result=dataset_result,
        detection_head_result=detection_head_result,
        prediction_payload=prediction_payload,
        error_result=error_result,
        metrics_result=metrics_result,
    )

if save_snapshot_clicked:
    if current_snapshot and snapshot_has_results(current_snapshot):
        run_dir = save_analysis_snapshot(history_root, current_snapshot)
        st.session_state["_history_feedback_message"] = ("success", f"已保存分析快照: {run_dir.name}")
        st.rerun()
    else:
        history_feedback.warning("当前没有可保存的分析结果。")

if view_mode == "历史运行" and not history_options:
    st.warning("当前还没有历史运行记录，页面展示的是实时分析结果。")
elif is_history_view and active_history_record is not None:
    st.info(f"当前正在查看历史运行: {format_history_option(active_history_record)}")
else:
    st.caption("左侧“运行历史”支持保存当前完整分析快照，并切换查看之前的分析结果。")

if prediction_payload and prediction_payload.get("meta", {}).get("cpu_fallback_used"):
    if is_history_view:
        st.warning("该历史运行在原始推理过程中发生过 CUDA 显存不足，并已自动切换到 CPU 完成推理。")
    else:
        st.warning("本次推理过程中发生 CUDA 显存不足，系统已自动切换到 CPU。若想继续使用 GPU，请尝试减小 imgsz 或保持 batch_size=1。")

active_run_label = format_history_option(active_history_record) if is_history_view and active_history_record is not None else "当前运行（未保存）"


def _get_model_summary_data(image_path: str):
    return _cached_compute(
        "生成模型摘要",
        _cache_key("model_summary", effective_weights_path, image_path, effective_imgsz, effective_device),
        cache_root,
        lambda _cb: model_summary(effective_weights_path, image_path, effective_imgsz, effective_device),
    )


def _get_feature_map_data(image_path: str, layer_index: int):
    return _cached_compute(
        "提取特征图",
        _cache_key(
            "feature_map",
            effective_weights_path,
            image_path,
            effective_imgsz,
            effective_conf,
            effective_iou,
            effective_device,
            effective_max_det,
            layer_index,
        ),
        cache_root,
        lambda _cb: capture_feature_map(image_path, inference_config, layer_index),
    )


tab_dataset, tab_head, tab_prediction, tab_error, tab_metrics, tab_model = st.tabs(
    ["数据集分析", "检测头分析", "预测结果分析", "漏检 / 虚检分析", "指标分析", "模型结构与特征图可视化"]
)

with tab_dataset:
    render_dataset_page(dataset_context, dataset_result, effective_imgsz, size_metric_internal, small_thr, medium_thr)

with tab_head:
    render_detection_head_page(detection_head_result)

with tab_prediction:
    render_prediction_page(eval_context, prediction_payload)

with tab_error:
    render_error_page(eval_context, error_result, effective_match_iou)

with tab_metrics:
    render_metrics_page(metrics_result, history_display_df, active_run_label, is_history_view)

with tab_model:
    image_candidates = []
    if eval_context is not None:
        image_candidates = eval_context.image_paths[:100]
    elif dataset_context is not None:
        image_candidates = dataset_context.image_paths[:100]
    render_model_page(
        weights_path=effective_weights_path,
        image_candidates=image_candidates,
        inference_config=inference_config,
        resolve_uploaded_image=lambda uploaded_file: _resolve_uploaded_image(uploaded_file, upload_root),
        get_model_summary_data=_get_model_summary_data,
        get_feature_map_data=_get_feature_map_data,
    )
