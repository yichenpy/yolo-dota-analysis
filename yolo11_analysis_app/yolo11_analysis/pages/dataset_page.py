from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..exports import dataframe_to_csv_bytes, json_to_bytes
from ..schemas import BoxRecord
from ..visualization import draw_scaled_box_view, overlay_error_boxes, plot_bar, plot_histogram, plot_scatter


def _rows_to_boxes(box_df: pd.DataFrame, image_path: str) -> list[BoxRecord]:
    subset = box_df[box_df["image_path"] == image_path]
    boxes: list[BoxRecord] = []
    for _, row in subset.iterrows():
        meta = {}
        if "polygon" in row and isinstance(row["polygon"], list):
            meta["polygon"] = row["polygon"]
        boxes.append(
            BoxRecord(
                image_id=Path(image_path).stem,
                image_path=image_path,
                class_id=int(row["class_id"]),
                class_name=str(row["class_name"]),
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
                source="gt",
                meta=meta,
            )
        )
    return boxes


def _set_selected_image(state_key: str, image_path: str | None) -> None:
    if image_path:
        st.session_state[state_key] = image_path


def _extract_selected_rows(event) -> list[int]:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if selection is None:
        return []
    rows = selection.get("rows", []) if isinstance(selection, dict) else getattr(selection, "rows", [])
    return [int(row) for row in rows]


def _render_selectable_table(
    source_df: pd.DataFrame,
    display_df: pd.DataFrame,
    *,
    state_key: str,
    table_key: str,
    selection_guard: dict[str, bool],
    height: int = 320,
) -> None:
    if source_df.empty or display_df.empty:
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=height)
        return
    try:
        event = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=table_key,
            height=height,
        )
        selected_rows = _extract_selected_rows(event)
        signature_key = f"{table_key}__selected_rows"
        previous_rows = tuple(st.session_state.get(signature_key, []))
        current_rows = tuple(selected_rows)
        if current_rows != previous_rows:
            st.session_state[signature_key] = list(current_rows)
            if current_rows and not selection_guard.get("updated", False):
                _set_selected_image(state_key, source_df.iloc[current_rows[0]]["image_path"])
                selection_guard["updated"] = True
    except TypeError:
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=height)
        st.caption("当前 Streamlit 版本不支持表格行选择联动。")


def _render_linkable_table(
    source_df: pd.DataFrame,
    *,
    state_key: str,
    table_key: str,
    columns: list[str],
    caption: str,
    selection_guard: dict[str, bool],
    height: int = 280,
) -> None:
    if source_df.empty:
        return
    st.caption(caption)
    _render_selectable_table(
        source_df.reset_index(drop=True),
        source_df[columns].reset_index(drop=True),
        state_key=state_key,
        table_key=table_key,
        selection_guard=selection_guard,
        height=height,
    )


def _render_paged_image_stats_table(image_df: pd.DataFrame, *, state_key: str, selection_guard: dict[str, bool]) -> None:
    if image_df.empty:
        return
    control_cols = st.columns([2.0, 1.0, 1.0])
    search_text = control_cols[0].text_input("按图片名筛选", value="", key="dataset_stats_search")
    page_size = int(control_cols[1].selectbox("每页显示", [50, 100, 200, 500], index=1, key="dataset_stats_page_size"))

    filtered_df = image_df.copy()
    if search_text:
        filtered_df = filtered_df[filtered_df["image_name"].str.contains(search_text, case=False, na=False)]
    filtered_df = filtered_df.sort_values(["object_count", "image_name"], ascending=[False, True]).reset_index(drop=True)
    if filtered_df.empty:
        st.warning("筛选后没有图片。")
        return

    total_pages = max(1, (len(filtered_df) + page_size - 1) // page_size)
    page_value = int(control_cols[2].number_input("页码", min_value=1, max_value=total_pages, value=1, step=1, key="dataset_stats_page"))
    start = (page_value - 1) * page_size
    end = start + page_size
    page_df = filtered_df.iloc[start:end].reset_index(drop=True)

    st.caption(f"综合图片统计表: 共 {len(filtered_df)} 张图，当前第 {page_value}/{total_pages} 页。点选任意一行会同步到下方单图案例。")
    _render_selectable_table(
        page_df,
        page_df[["image_name", "object_count", "width", "height", "resized_width", "resized_height"]],
        state_key=state_key,
        table_key="dataset_stats_table",
        selection_guard=selection_guard,
        height=360,
    )


def render_dataset_page(context, analysis, imgsz: int, size_metric: str, small_thr: float, medium_thr: float):
    st.subheader("数据集分析")
    if analysis is None:
        st.info("请先在侧边栏提供数据集路径或上传图像/标签。")
        return

    selection_guard = {"updated": False}

    image_df = analysis["image_df"]
    box_df = analysis["box_df"]
    per_class_df = analysis["per_class_df"]
    size_bucket_df = analysis["size_bucket_df"]
    per_class_bucket_df = analysis["per_class_bucket_df"]
    summary = analysis["summary"]

    if not image_df.empty and st.session_state.get("dataset_case_image") not in image_df["image_path"].tolist():
        default_image = image_df.sort_values(["object_count", "image_name"], ascending=[False, True]).iloc[0]["image_path"]
        st.session_state["dataset_case_image"] = default_image

    total_boxes = summary["num_boxes"]
    small_ratio = 0.0
    if not size_bucket_df.empty and total_boxes > 0:
        small_ratio = float(size_bucket_df[size_bucket_df["size_bucket"] == "small"]["count"].sum()) / total_boxes
    imbalance_ratio = 0.0
    if not per_class_df.empty and per_class_df["instances"].min() > 0:
        imbalance_ratio = float(per_class_df["instances"].max() / per_class_df["instances"].min())

    metric_cols = st.columns(6)
    metric_cols[0].metric("图像数", summary["num_images"])
    metric_cols[1].metric("目标框数", summary["num_boxes"])
    metric_cols[2].metric("平均每图目标数", f"{image_df['object_count'].mean():.2f}" if not image_df.empty else "0.00")
    metric_cols[3].metric("小目标占比", f"{small_ratio:.2%}")
    metric_cols[4].metric("类别不均衡比", f"{imbalance_ratio:.2f}" if imbalance_ratio else "n/a")
    metric_cols[5].metric("当前 imgsz", imgsz)

    st.info(
        f"当前页以目标分布和单图样本为主。模型输入缩放规则: 保持长宽比缩放到 {imgsz}x{imgsz} 范围内，再对四周补边(letterbox)。"
        f" 尺寸度量使用 `{size_metric}`，分桶阈值为 small<{small_thr}, medium<{medium_thr}。"
    )

    with st.expander("基础图像统计", expanded=False):
        st.write(
            {
                "image_width_stats": summary["image_width_stats"],
                "image_height_stats": summary["image_height_stats"],
                "image_aspect_ratio_stats": summary["image_aspect_ratio_stats"],
                "objects_per_image_stats": summary["objects_per_image_stats"],
                "missing_label_files": summary["missing_label_files"][:20],
            }
        )

    if not box_df.empty:
        col1, col2 = st.columns(2)
        col1.pyplot(plot_histogram(box_df["orig_metric"], f"原图目标 {size_metric} 分布", size_metric), clear_figure=True)
        col2.pyplot(plot_histogram(box_df["scaled_metric"], f"缩放后目标 {size_metric} 分布", f"scaled {size_metric}"), clear_figure=True)

        col3, col4 = st.columns(2)
        col3.pyplot(plot_scatter(box_df["orig_width"], box_df["orig_height"], "目标宽高散点图", "Width", "Height"), clear_figure=True)
        col4.pyplot(plot_histogram(box_df["orig_aspect_ratio"], "目标长宽比分布", "long / short"), clear_figure=True)

        scaled_bucket_df = box_df.groupby("scaled_size_bucket").size().reset_index(name="count").rename(columns={"scaled_size_bucket": "size_bucket"})
        col5, col6 = st.columns(2)
        if not size_bucket_df.empty:
            col5.pyplot(
                plot_bar(
                    size_bucket_df["size_bucket"].astype(str).tolist(),
                    size_bucket_df["count"].astype(float).tolist(),
                    "原图目标尺寸分桶",
                    "Count",
                    color="#e45756",
                ),
                clear_figure=True,
            )
        if not scaled_bucket_df.empty:
            col6.pyplot(
                plot_bar(
                    scaled_bucket_df["size_bucket"].astype(str).tolist(),
                    scaled_bucket_df["count"].astype(float).tolist(),
                    "缩放后目标尺寸分桶",
                    "Count",
                    color="#54a24b",
                ),
                clear_figure=True,
            )

        col7, col8 = st.columns(2)
        if not image_df.empty:
            col7.pyplot(plot_histogram(image_df["object_count"], "每图目标数分布", "objects per image"), clear_figure=True)
            top_images_df = image_df.sort_values(["object_count", "image_name"], ascending=[False, True]).head(20)
            col8.pyplot(
                plot_bar(
                    top_images_df["image_name"].astype(str).tolist(),
                    top_images_df["object_count"].astype(float).tolist(),
                    "Top-20 图像目标数",
                    "Objects",
                    color="#4c78a8",
                ),
                clear_figure=True,
            )

        if not per_class_df.empty:
            class_rank_df = per_class_df.sort_values("instances", ascending=False).head(15)
            class_size_df = per_class_df.sort_values("median_orig_metric", ascending=False).head(15)
            col9, col10 = st.columns(2)
            col9.pyplot(
                plot_bar(
                    class_rank_df["class_name"].astype(str).tolist(),
                    class_rank_df["instances"].astype(float).tolist(),
                    "Top-15 类别目标数量",
                    "Instances",
                ),
                clear_figure=True,
            )
            col10.pyplot(
                plot_bar(
                    class_size_df["class_name"].astype(str).tolist(),
                    class_size_df["median_orig_metric"].astype(float).tolist(),
                    f"Top-15 类别中位 {size_metric}",
                    size_metric,
                    color="#72b7b2",
                ),
                clear_figure=True,
            )

            st.markdown("**类别级目标统计**")
            st.dataframe(per_class_df.sort_values("instances", ascending=False), use_container_width=True)

            if not per_class_bucket_df.empty:
                st.markdown("**类别尺寸分布联动**")
                bucket_classes = sorted(per_class_bucket_df["class_name"].astype(str).unique().tolist())
                selected_class = st.selectbox("查看指定类别的尺寸分桶", bucket_classes, key="dataset_bucket_class")
                selected_bucket_df = per_class_bucket_df[per_class_bucket_df["class_name"] == selected_class]
                st.pyplot(
                    plot_bar(
                        selected_bucket_df["size_bucket"].astype(str).tolist(),
                        selected_bucket_df["count"].astype(float).tolist(),
                        f"{selected_class} 尺寸分桶统计",
                        "Count",
                        color="#9c755f",
                    ),
                    clear_figure=True,
                )
                pivot_df = per_class_bucket_df.pivot(index="class_name", columns="size_bucket", values="count").fillna(0).reset_index()
                st.dataframe(pivot_df, use_container_width=True)

        st.markdown("**统计表联动到单图案例**")
        top_image_candidates = image_df.sort_values(["object_count", "image_name"], ascending=[False, True]).head(50)
        smallest_df = box_df.sort_values(["orig_metric", "image_name"], ascending=[True, True]).head(30)
        largest_df = box_df.sort_values(["orig_metric", "image_name"], ascending=[False, True]).head(30)
        link_col1, link_col2, link_col3 = st.columns(3)
        with link_col1:
            _render_linkable_table(
                top_image_candidates,
                state_key="dataset_case_image",
                table_key="dataset_top_image_table",
                columns=["image_name", "object_count", "width", "height"],
                caption="高密度图片统计表。",
                selection_guard=selection_guard,
            )
        with link_col2:
            _render_linkable_table(
                smallest_df,
                state_key="dataset_case_image",
                table_key="dataset_smallest_table",
                columns=["image_name", "class_name", "orig_width", "orig_height", "orig_metric", "size_bucket"],
                caption="最小目标样本表。",
                selection_guard=selection_guard,
            )
        with link_col3:
            _render_linkable_table(
                largest_df,
                state_key="dataset_case_image",
                table_key="dataset_largest_table",
                columns=["image_name", "class_name", "orig_width", "orig_height", "orig_metric", "size_bucket"],
                caption="最大目标样本表。",
                selection_guard=selection_guard,
            )

        _render_paged_image_stats_table(
            image_df[["image_path", "image_name", "object_count", "width", "height", "resized_width", "resized_height"]],
            state_key="dataset_case_image",
            selection_guard=selection_guard,
        )

    st.markdown("**单图案例分析**")
    selected_image = st.session_state.get("dataset_case_image")
    if selected_image:
        sample_boxes = _rows_to_boxes(box_df, selected_image)
        sample_meta = image_df[image_df["image_path"] == selected_image].iloc[0].to_dict()
        st.caption(f"当前案例图: {sample_meta['image_name']}。点选上方任意统计表行可切换。")
        cols = st.columns(2)
        cols[0].write(
            {
                "image_name": sample_meta["image_name"],
                "object_count": int(sample_meta.get("object_count", 0)),
                "orig_width": int(sample_meta["width"]),
                "orig_height": int(sample_meta["height"]),
                "resized_width": int(sample_meta["resized_width"]),
                "resized_height": int(sample_meta["resized_height"]),
                "resize_ratio": float(sample_meta["resize_ratio"]),
            }
        )
        cols[0].image(overlay_error_boxes(selected_image, sample_boxes, [], [], []), caption="原图与 GT 框", use_container_width=True)
        cols[1].image(draw_scaled_box_view(selected_image, sample_boxes, imgsz), caption="缩放到模型输入尺寸后的框", use_container_width=True)
        sample_box_df = box_df[box_df["image_path"] == selected_image][
            [
                "class_name",
                "orig_width",
                "orig_height",
                "orig_metric",
                "size_bucket",
                "scaled_metric",
                "scaled_size_bucket",
            ]
        ]
        st.dataframe(sample_box_df.sort_values(["orig_metric", "class_name"], ascending=[False, True]), use_container_width=True)

    st.markdown("**导出**")
    export_cols = st.columns(3)
    export_cols[0].download_button(
        "导出图像统计 CSV",
        data=dataframe_to_csv_bytes(image_df),
        file_name="dataset_image_stats.csv",
        mime="text/csv",
    )
    export_cols[1].download_button(
        "导出框统计 CSV",
        data=dataframe_to_csv_bytes(box_df),
        file_name="dataset_box_stats.csv",
        mime="text/csv",
    )
    export_cols[2].download_button(
        "导出摘要 JSON",
        data=json_to_bytes(summary),
        file_name="dataset_summary.json",
        mime="application/json",
    )
