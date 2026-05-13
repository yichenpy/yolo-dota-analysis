from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..exports import dataframe_to_csv_bytes
from ..inference import predictions_to_rows
from ..visualization import overlay_prediction_boxes, plot_bar, plot_histogram


def render_prediction_page(context, prediction_payload):
    st.subheader("预测结果分析")
    if not prediction_payload:
        st.info("请先提供权重文件与推理图像。")
        return

    predictions = prediction_payload.get("predictions", {})
    meta = prediction_payload.get("meta", {})
    if not predictions:
        st.warning("当前参数下没有任何预测结果。请尝试降低 conf、降低 imgsz 或检查权重。")
        return

    pred_df = pd.DataFrame(predictions_to_rows(predictions))
    if pred_df.empty:
        st.warning("当前参数下没有任何预测结果。请尝试降低 conf 或检查权重。")
        return

    image_count = len(predictions)
    total_pred = len(pred_df)
    avg_conf = float(pred_df["confidence"].mean()) if not pred_df.empty else 0.0
    avg_det_per_image = total_pred / max(image_count, 1)

    metric_cols = st.columns(4)
    metric_cols[0].metric("图像数", image_count)
    metric_cols[1].metric("预测框数", total_pred)
    metric_cols[2].metric("平均置信度", f"{avg_conf:.4f}")
    metric_cols[3].metric("平均每图检测数", f"{avg_det_per_image:.2f}")

    st.caption(
        f"实际推理设备: {meta.get('device_used', 'n/a')} | "
        f"请求 batch: {meta.get('batch_size_requested', 'n/a')} | "
        f"实际 batch: {meta.get('batch_size_used', 'n/a')} | "
        f"OOM 重试次数: {meta.get('oom_retries', 0)}"
    )
    if meta.get("cpu_fallback_used"):
        st.warning("推理过程中检测到 CUDA 显存不足，应用已自动切换到 CPU 完成剩余推理。")

    col1, col2 = st.columns(2)
    fig_conf = plot_histogram(pred_df["confidence"], "置信度分布", "Confidence")
    class_counts = pred_df.groupby("class_name").size().reset_index(name="count")
    fig_class = plot_bar(class_counts["class_name"].astype(str).tolist(), class_counts["count"].astype(float).tolist(), "预测类别分布", "Count")
    col1.pyplot(fig_conf, clear_figure=True)
    col2.pyplot(fig_class, clear_figure=True)

    image_counts = pred_df.groupby("image_name").size().sort_values(ascending=False).head(20)
    fig_image = plot_bar(image_counts.index.tolist(), image_counts.values.astype(float).tolist(), "Top-20 图片预测数", "Count")
    st.pyplot(fig_image, clear_figure=True)

    st.markdown("**单图查看**")
    selectable_images = sorted(predictions.keys())
    selected_image = st.selectbox("选择图片", selectable_images, format_func=lambda item: Path(item).name, key="prediction_preview_image")
    st.image(overlay_prediction_boxes(selected_image, predictions[selected_image]), caption="预测框叠加图", use_container_width=True)

    st.markdown("**表格明细**")
    class_options = ["全部"] + sorted(pred_df["class_name"].astype(str).unique().tolist())
    class_filter = st.selectbox("按类别筛选", class_options, key="prediction_class_filter")
    filtered_df = pred_df if class_filter == "全部" else pred_df[pred_df["class_name"] == class_filter]
    st.dataframe(filtered_df, use_container_width=True)

    st.download_button(
        "导出预测明细 CSV",
        data=dataframe_to_csv_bytes(filtered_df),
        file_name="prediction_details.csv",
        mime="text/csv",
    )
