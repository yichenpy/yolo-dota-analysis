from __future__ import annotations

import pandas as pd
import streamlit as st

from ..exports import dataframe_to_csv_bytes, json_to_bytes
from ..visualization import figure_to_image_bytes, plot_bar, plot_confusion_matrix, plot_pr_curves


def render_metrics_page(metrics_result, history_df: pd.DataFrame | None, active_run_label: str, is_history_view: bool):
    st.subheader("指标分析")
    st.caption(f"当前结果来源: {active_run_label}")

    if metrics_result is None:
        st.info("请先提供权重和带标注的验证/测试集。")
        if history_df is not None and not history_df.empty:
            st.markdown("**已保存的分析快照**")
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        return

    summary = metrics_result["summary"]
    class_df = metrics_result["class_df"]
    pr_curves = metrics_result["pr_curves"]
    confusion_matrix = metrics_result["confusion_matrix"]
    confusion_labels = metrics_result["confusion_labels"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Precision", f"{summary['precision']:.4f}")
    metric_cols[1].metric("Recall", f"{summary['recall']:.4f}")
    metric_cols[2].metric("mAP@0.5", f"{summary['mAP50']:.4f}")
    metric_cols[3].metric("mAP@0.5:0.95", f"{summary['mAP50_95']:.4f}")

    if summary.get("iou_mode") == "rotated_polygon":
        st.info("当前指标计算使用旋转框多边形 IoU；如果预测或标注缺失多边形，系统会自动退化为矩形多边形。")

    if is_history_view:
        st.info("当前为历史运行浏览模式。切回左侧栏的“当前运行”后，可以继续产生新结果并保存快照。")

    pr_fig = plot_pr_curves(pr_curves)
    st.pyplot(pr_fig, clear_figure=True)

    if not class_df.empty:
        ap_fig = plot_bar(
            class_df["class_name"].astype(str).tolist(),
            class_df["ap50_95"].astype(float).tolist(),
            "每类别 AP@0.5:0.95",
            "AP",
        )
        st.pyplot(ap_fig, clear_figure=True)
        st.dataframe(class_df, use_container_width=True)
    else:
        ap_fig = None

    cm_fig = plot_confusion_matrix(confusion_matrix, confusion_labels)
    st.pyplot(cm_fig, clear_figure=True)

    st.markdown("**导出**")
    export_cols = st.columns(5)
    export_cols[0].download_button(
        "导出类别指标 CSV",
        data=dataframe_to_csv_bytes(class_df),
        file_name="metrics_per_class.csv",
        mime="text/csv",
    )
    export_cols[1].download_button(
        "导出指标 JSON",
        data=json_to_bytes(metrics_result),
        file_name="metrics_summary.json",
        mime="application/json",
    )
    export_cols[2].download_button(
        "导出 PR 曲线 PNG",
        data=figure_to_image_bytes(pr_fig),
        file_name="pr_curve.png",
        mime="image/png",
    )
    export_cols[3].download_button(
        "导出 AP 柱状图 PNG",
        data=figure_to_image_bytes(ap_fig) if ap_fig is not None else b"",
        file_name="ap_bar.png",
        mime="image/png",
        disabled=ap_fig is None,
    )
    export_cols[4].download_button(
        "导出混淆矩阵 PNG",
        data=figure_to_image_bytes(cm_fig),
        file_name="confusion_matrix.png",
        mime="image/png",
    )

    st.markdown("**已保存的分析快照**")
    if history_df is None or history_df.empty:
        st.caption("当前还没有保存过完整分析快照。")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
