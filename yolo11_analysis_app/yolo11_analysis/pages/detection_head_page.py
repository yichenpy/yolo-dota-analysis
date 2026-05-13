from __future__ import annotations

import pandas as pd
import streamlit as st

from ..exports import dataframe_to_csv_bytes, json_to_bytes
from ..visualization import plot_bar


def render_detection_head_page(detection_result):
    st.subheader("检测头分析")
    if detection_result is None:
        st.info("请先在侧边栏提供数据集与可选权重。")
        return

    head_summary_df = detection_result.get("head_summary_df", pd.DataFrame())
    assigned_class_df = detection_result.get("assigned_class_df", pd.DataFrame())
    coverage_class_df = detection_result.get("coverage_class_df", pd.DataFrame())
    assigned_bucket_df = detection_result.get("assigned_bucket_df", pd.DataFrame())
    coverage_band_df = detection_result.get("coverage_band_df", pd.DataFrame())
    overlap_df = detection_result.get("overlap_df", pd.DataFrame())
    range_miss_type_df = detection_result.get("range_miss_type_df", pd.DataFrame())
    boundary_df = detection_result.get("boundary_df", pd.DataFrame())
    range_miss_df = detection_result.get("range_miss_df", pd.DataFrame())
    summary = detection_result.get("summary", {})

    if head_summary_df.empty:
        st.warning("当前数据集中没有可用于检测头分析的目标统计。")
        return

    dominant_assigned = head_summary_df.sort_values("assigned_count", ascending=False).iloc[0]
    dominant_covered = head_summary_df.sort_values("effective_coverage_count", ascending=False).iloc[0]

    metric_cols = st.columns(6)
    metric_cols[0].metric("检测头数", int(summary.get("num_heads", len(head_summary_df))))
    metric_cols[1].metric("主负责检测头", str(dominant_assigned["head_label"]))
    metric_cols[2].metric("主负责头占比", f"{float(dominant_assigned['assigned_ratio']):.2%}")
    metric_cols[3].metric("覆盖最多检测头", str(dominant_covered["head_label"]))
    metric_cols[4].metric("无有效检测头占比", f"{float(summary.get('range_miss_ratio', 0.0)):.2%}")
    metric_cols[5].metric("多头重叠占比", f"{float(summary.get('multi_head_overlap_ratio', 0.0)):.2%}")

    source_text = "来自当前模型权重" if summary.get("detection_head_source") == "model" else "默认 stride 配置"
    st.info(
        f"这里区分两套统计: 1) 主负责检测头: 用缩放后目标的 `sqrt(area) / stride` 与 {float(summary.get('detection_head_target_cells', 4.0)):.1f} cells 的接近程度分配。"
        f" 2) 有效覆盖检测头: 如果目标在某个 head 上满足 {float(summary.get('coverage_min_cells', 2.0)):.1f} 到 {float(summary.get('coverage_max_cells', 8.0)):.1f} cells，则认为该 head 对它处于有效尺度。"
        f" 当前步长来源: {source_text}。"
    )
    if summary.get("detection_head_warning"):
        st.warning(str(summary["detection_head_warning"]))

    top_row_1 = st.columns(2)
    top_row_1[0].pyplot(
        plot_bar(
            head_summary_df["head_label"].astype(str).tolist(),
            head_summary_df["assigned_count"].astype(float).tolist(),
            "各检测头主负责目标数",
            "Count",
            color="#4c78a8",
        ),
        clear_figure=True,
    )
    top_row_1[1].pyplot(
        plot_bar(
            head_summary_df["head_label"].astype(str).tolist(),
            (head_summary_df["effective_coverage_ratio"].astype(float) * 100.0).tolist(),
            "各检测头有效覆盖占比",
            "Percent",
            color="#54a24b",
        ),
        clear_figure=True,
    )

    top_row_2 = st.columns(2)
    top_row_2[0].pyplot(
        plot_bar(
            head_summary_df["head_label"].astype(str).tolist(),
            (head_summary_df["exclusive_effective_ratio"].astype(float) * 100.0).tolist(),
            "各检测头独占覆盖占比",
            "Percent",
            color="#f58518",
        ),
        clear_figure=True,
    )
    top_row_2[1].pyplot(
        plot_bar(
            head_summary_df["head_label"].astype(str).tolist(),
            (head_summary_df["boundary_assigned_ratio"].astype(float) * 100.0).tolist(),
            "各检测头边界样本占比",
            "Percent",
            color="#e45756",
        ),
        clear_figure=True,
    )

    bottom_row = st.columns(2)
    if not overlap_df.empty:
        bottom_row[0].pyplot(
            plot_bar(
                overlap_df["overlap_label"].astype(str).tolist(),
                (overlap_df["ratio"].astype(float) * 100.0).tolist(),
                "每个目标被多少个检测头有效覆盖",
                "Percent",
                color="#72b7b2",
            ),
            clear_figure=True,
        )
    if not range_miss_type_df.empty:
        bottom_row[1].pyplot(
            plot_bar(
                range_miss_type_df["range_miss_type"].astype(str).tolist(),
                (range_miss_type_df["ratio"].astype(float) * 100.0).tolist(),
                "目标落在各尺度状态的占比",
                "Percent",
                color="#9c755f",
            ),
            clear_figure=True,
        )

    st.markdown("**检测头汇总统计**")
    st.dataframe(
        head_summary_df[
            [
                "head_label",
                "assigned_count",
                "assigned_ratio",
                "effective_coverage_count",
                "effective_coverage_ratio",
                "exclusive_effective_ratio",
                "median_primary_cells",
                "median_effective_cells",
                "range_miss_assigned_ratio",
                "boundary_assigned_ratio",
                "single_cover_assigned_ratio",
                "multi_cover_assigned_ratio",
            ]
        ],
        use_container_width=True,
    )

    if not assigned_class_df.empty:
        dominant_assigned_class_df = (
            assigned_class_df.sort_values(["class_name", "class_ratio", "assigned_count"], ascending=[True, False, False])
            .groupby("class_name", as_index=False)
            .first()[["class_name", "head_label", "assigned_count", "class_total", "class_ratio"]]
            .rename(
                columns={
                    "head_label": "dominant_assigned_head",
                    "assigned_count": "dominant_assigned_count",
                    "class_total": "class_total_count",
                    "class_ratio": "dominant_assigned_ratio",
                }
            )
        )
        st.markdown("**类别主负责检测头**")
        st.dataframe(dominant_assigned_class_df.sort_values(["dominant_assigned_ratio", "class_total_count"], ascending=[False, False]), use_container_width=True)

    if not assigned_class_df.empty or not coverage_class_df.empty:
        st.markdown("**类别视角**")
        class_options = sorted(set(assigned_class_df.get("class_name", pd.Series(dtype=str)).astype(str).tolist()) | set(coverage_class_df.get("class_name", pd.Series(dtype=str)).astype(str).tolist()))
        if class_options:
            selected_class = st.selectbox("查看指定类别在各检测头上的分配与覆盖", class_options, key="head_class_select")
            assigned_subset = assigned_class_df[assigned_class_df["class_name"] == selected_class].sort_values("head_stride") if not assigned_class_df.empty else pd.DataFrame()
            covered_subset = coverage_class_df[coverage_class_df["class_name"] == selected_class].sort_values("head_stride") if not coverage_class_df.empty else pd.DataFrame()
            class_cols = st.columns(2)
            if not assigned_subset.empty:
                class_cols[0].pyplot(
                    plot_bar(
                        assigned_subset["head_label"].astype(str).tolist(),
                        (assigned_subset["class_ratio"].astype(float) * 100.0).tolist(),
                        f"{selected_class} 的主负责头占比",
                        "Percent",
                        color="#4c78a8",
                    ),
                    clear_figure=True,
                )
            if not covered_subset.empty:
                class_cols[1].pyplot(
                    plot_bar(
                        covered_subset["head_label"].astype(str).tolist(),
                        (covered_subset["class_ratio"].astype(float) * 100.0).tolist(),
                        f"{selected_class} 的有效覆盖头占比",
                        "Percent",
                        color="#54a24b",
                    ),
                    clear_figure=True,
                )
            merged_class_df = assigned_subset[["head_name", "head_label", "assigned_count", "class_ratio"]].rename(columns={"class_ratio": "assigned_class_ratio"}) if not assigned_subset.empty else pd.DataFrame(columns=["head_name", "head_label", "assigned_count", "assigned_class_ratio"])
            if not covered_subset.empty:
                merged_class_df = merged_class_df.merge(
                    covered_subset[["head_name", "coverage_count", "class_ratio"]].rename(columns={"class_ratio": "coverage_class_ratio"}),
                    on="head_name",
                    how="outer",
                )
                if "head_label" not in merged_class_df and "head_label" in covered_subset:
                    merged_class_df = merged_class_df.merge(covered_subset[["head_name", "head_label"]].drop_duplicates(), on="head_name", how="left")
            if not merged_class_df.empty:
                st.dataframe(merged_class_df.fillna(0), use_container_width=True)

    head_options = head_summary_df.sort_values("head_stride")["head_name"].astype(str).tolist()
    head_label_map = {row["head_name"]: row["head_label"] for _, row in head_summary_df[["head_name", "head_label"]].drop_duplicates().iterrows()}
    selected_head = st.selectbox("查看指定检测头明细", head_options, format_func=lambda item: head_label_map.get(item, item), key="head_select")

    head_class_subset = assigned_class_df[assigned_class_df["head_name"] == selected_head].sort_values("assigned_count", ascending=False).head(15) if not assigned_class_df.empty else pd.DataFrame()
    head_bucket_subset = assigned_bucket_df[assigned_bucket_df["head_name"] == selected_head] if not assigned_bucket_df.empty else pd.DataFrame()
    head_band_subset = coverage_band_df[coverage_band_df["head_name"] == selected_head] if not coverage_band_df.empty else pd.DataFrame()

    head_cols = st.columns(3)
    if not head_class_subset.empty:
        head_cols[0].pyplot(
            plot_bar(
                head_class_subset["class_name"].astype(str).tolist(),
                head_class_subset["assigned_count"].astype(float).tolist(),
                f"{head_label_map.get(selected_head, selected_head)} 主负责的主要类别",
                "Count",
                color="#9c755f",
            ),
            clear_figure=True,
        )
    if not head_bucket_subset.empty:
        head_cols[1].pyplot(
            plot_bar(
                head_bucket_subset["size_bucket"].astype(str).tolist(),
                head_bucket_subset["count"].astype(float).tolist(),
                f"{head_label_map.get(selected_head, selected_head)} 主负责尺度分桶",
                "Count",
                color="#b279a2",
            ),
            clear_figure=True,
        )
    if not head_band_subset.empty:
        head_cols[2].pyplot(
            plot_bar(
                head_band_subset["coverage_band"].astype(str).tolist(),
                head_band_subset["count"].astype(float).tolist(),
                f"{head_label_map.get(selected_head, selected_head)} 有效覆盖 cells 区间",
                "Count",
                color="#ff9da6",
            ),
            clear_figure=True,
        )

    focus_cols = st.columns(2)
    if not range_miss_df.empty:
        focus_cols[0].markdown("**无有效检测头样本**")
        focus_cols[0].dataframe(
            range_miss_df[
                [
                    "image_name",
                    "class_name",
                    "size_bucket",
                    "assigned_head_label",
                    "primary_head_cells_metric",
                    "range_miss_type",
                    "best_alternative_head_label",
                ]
            ].head(100),
            use_container_width=True,
        )
    if not boundary_df.empty:
        focus_cols[1].markdown("**边界样本**")
        focus_cols[1].dataframe(
            boundary_df[
                [
                    "image_name",
                    "class_name",
                    "size_bucket",
                    "assigned_head_label",
                    "primary_head_cells_metric",
                    "best_head_margin",
                    "best_alternative_head_label",
                ]
            ].head(100),
            use_container_width=True,
        )

    st.markdown("**导出**")
    export_cols = st.columns(3)
    export_cols[0].download_button(
        "导出检测头汇总 CSV",
        data=dataframe_to_csv_bytes(head_summary_df),
        file_name="detection_head_summary.csv",
        mime="text/csv",
    )
    export_cols[1].download_button(
        "导出检测头类别分布 CSV",
        data=dataframe_to_csv_bytes(assigned_class_df),
        file_name="detection_head_class_distribution.csv",
        mime="text/csv",
    )
    export_cols[2].download_button(
        "导出检测头分析 JSON",
        data=json_to_bytes(
            {
                "summary": summary,
                "head_summary": head_summary_df.to_dict(orient="records"),
                "assigned_class": assigned_class_df.to_dict(orient="records"),
                "coverage_class": coverage_class_df.to_dict(orient="records"),
                "assigned_bucket": assigned_bucket_df.to_dict(orient="records"),
                "coverage_band": coverage_band_df.to_dict(orient="records"),
                "overlap": overlap_df.to_dict(orient="records"),
                "range_miss_type": range_miss_type_df.to_dict(orient="records"),
            }
        ),
        file_name="detection_head_analysis.json",
        mime="application/json",
    )
