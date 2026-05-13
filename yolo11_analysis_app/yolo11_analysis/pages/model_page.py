from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..exports import dataframe_to_csv_bytes
from ..visualization import (
    overlay_prediction_boxes,
    plot_feature_average_heatmap,
    plot_feature_channel,
    plot_feature_grid,
)


def render_model_page(
    *,
    weights_path: str | None,
    image_candidates: list[str],
    inference_config,
    resolve_uploaded_image,
    get_model_summary_data,
    get_feature_map_data,
):
    st.subheader("模型结构与特征图可视化")
    if not weights_path:
        st.info("请先提供模型权重文件。")
        return

    uploaded_image = st.file_uploader("上传单张分析图片", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="model_single_image")
    uploaded_image_path = resolve_uploaded_image(uploaded_image) if uploaded_image else None

    selected_image = uploaded_image_path
    if not selected_image and image_candidates:
        selected_image = st.selectbox(
            "或从数据集中选择图片",
            image_candidates,
            format_func=lambda item: Path(item).name,
            key="model_page_image_select",
        )

    if not selected_image:
        st.warning("请上传单张图片，或在侧边栏提供可用图像目录。")
        return

    if st.button("生成模型摘要", key="model_summary_button"):
        st.session_state["model_summary_rows"] = get_model_summary_data(selected_image)

    summary_rows = st.session_state.get("model_summary_rows", [])
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        total_params = int(summary_df["params"].sum()) if not summary_df.empty else 0
        stage_counts = summary_df.groupby("stage").size().to_dict() if not summary_df.empty else {}

        metric_cols = st.columns(4)
        metric_cols[0].metric("层数", len(summary_df))
        metric_cols[1].metric("总参数量", f"{total_params:,}")
        metric_cols[2].metric("backbone 层数", int(stage_counts.get("backbone", 0)))
        metric_cols[3].metric("neck/head 层数", int(stage_counts.get("neck", 0) + stage_counts.get("head", 0)))

        st.dataframe(summary_df, use_container_width=True, height=420)
        st.download_button(
            "导出模型摘要 CSV",
            data=dataframe_to_csv_bytes(summary_df),
            file_name="model_summary.csv",
            mime="text/csv",
        )

        layer_labels = [f"{row['layer_index']} | {row['stage']} | {row['module_type']} | {row['output_shape']}" for row in summary_rows]
        selected_label = st.selectbox("选择指定层", layer_labels, key="feature_layer_select")
        selected_layer_index = summary_rows[layer_labels.index(selected_label)]["layer_index"]

        if st.button("提取所选层特征图", key="feature_extract_button"):
            st.session_state["feature_map_result"] = get_feature_map_data(selected_image, selected_layer_index)

        feature_result = st.session_state.get("feature_map_result")
        if feature_result is not None and feature_result.layer_index == selected_layer_index:
            feature_map = feature_result.feature_map
            channels = int(feature_map.shape[0])
            if channels <= 0:
                st.warning("当前层特征图没有可视化通道。")
                return

            max_grid_channels = min(32, channels)
            col1, col2 = st.columns(2)
            channel_index = col1.number_input("查看指定通道", min_value=0, max_value=max(0, channels - 1), value=0, step=1)
            if max_grid_channels <= 1:
                col2.metric("前 N 个通道", 1)
                grid_count = 1
            else:
                grid_count = col2.slider("前 N 个通道", min_value=1, max_value=max_grid_channels, value=min(8, max_grid_channels))

            avg_fig = plot_feature_average_heatmap(feature_map, f"平均激活热力图 | layer {feature_result.layer_index}")
            ch_fig = plot_feature_channel(feature_map, int(channel_index), f"特征图 | layer {feature_result.layer_index}")
            grid_fig = plot_feature_grid(feature_map, int(grid_count), f"前 {grid_count} 个通道")
            st.pyplot(avg_fig, clear_figure=True)
            st.pyplot(ch_fig, clear_figure=True)
            st.pyplot(grid_fig, clear_figure=True)

            st.image(
                overlay_prediction_boxes(selected_image, feature_result.detections),
                caption="最终检测结果",
                use_container_width=True,
            )
            st.caption(
                f"当前层: index={feature_result.layer_index}, name={feature_result.layer_name}, stage={feature_result.stage}, feature_shape={feature_map.shape}"
            )
