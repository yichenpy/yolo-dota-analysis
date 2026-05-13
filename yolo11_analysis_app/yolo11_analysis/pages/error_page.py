from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..exports import dataframe_to_csv_bytes
from ..visualization import COLOR_FN, COLOR_FP, extract_box_crops, overlay_error_boxes, plot_bar


FN_COLUMNS = ["image_name", "class_name", "size_bucket", "reason", "best_same_iou", "best_any_iou", "best_match_class", "gt_xyxy"]
FP_COLUMNS = ["image_name", "class_name", "confidence", "confidence_bin", "size_bucket", "reason", "best_same_iou", "best_any_iou", "best_match_class", "pred_xyxy"]


def _format_case_table(case_df: pd.DataFrame) -> pd.DataFrame:
    if case_df.empty:
        return case_df
    formatted = case_df.copy()
    for column in ["gt_xyxy", "pred_xyxy"]:
        if column in formatted:
            formatted[column] = formatted[column].apply(lambda value: None if value is None else [round(float(item), 2) for item in value])
    for column in ["confidence", "iou", "metric_value", "best_same_iou", "best_any_iou"]:
        if column in formatted:
            formatted[column] = formatted[column].apply(lambda value: None if pd.isna(value) else round(float(value), 4))
    return formatted


def _filter_detail(detail_df: pd.DataFrame, *, image_keyword: str, class_filter: str, reason_filter: str) -> pd.DataFrame:
    filtered = detail_df.copy()
    if image_keyword:
        filtered = filtered[filtered["image_name"].str.contains(image_keyword, case=False, na=False)]
    if class_filter != "全部":
        filtered = filtered[filtered["class_name"] == class_filter]
    if reason_filter != "全部":
        filtered = filtered[filtered["reason"] == reason_filter]
    return filtered


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


def _render_crop_gallery(image_path: str, boxes, *, color: str, label_prefix: str, title: str, empty_text: str) -> None:
    st.markdown(f"**{title}**")
    crop_items = extract_box_crops(image_path, list(boxes), color=color, label_prefix=label_prefix, max_items=12)
    if not crop_items:
        st.caption(empty_text)
        return
    columns = st.columns(3)
    for index, (crop, caption) in enumerate(crop_items):
        columns[index % 3].image(crop, caption=caption, use_container_width=True)


def render_error_page(context, error_result, iou_threshold: float):
    st.subheader("漏检 / 虚检分析")
    if error_result is None:
        st.info("请先提供权重和带标注的验证/测试集。")
        return

    selection_guard = {"updated": False}

    image_df = error_result["image_df"]
    detail_df = error_result["detail_df"]
    summary = error_result["summary"]
    image_cases = error_result["image_cases"]
    fn_df = error_result["fn_df"]
    fp_df = error_result["fp_df"]
    fn_reason_df = error_result["fn_reason_df"]
    fn_size_df = error_result["fn_size_df"]
    fn_class_df = error_result["fn_class_df"]
    fp_reason_df = error_result["fp_reason_df"]
    fp_conf_df = error_result["fp_conf_df"]
    fp_class_df = error_result["fp_class_df"]
    hard_images_df = error_result["hard_images_df"]

    if not image_df.empty and st.session_state.get("error_case_image") not in image_df["image_path"].tolist():
        default_image = hard_images_df.iloc[0]["image_path"] if not hard_images_df.empty else image_df.iloc[0]["image_path"]
        st.session_state["error_case_image"] = default_image

    total_gt = summary["total_tp"] + summary["total_fn"]
    total_pred = summary["total_tp"] + summary["total_fp"]
    overall_miss_rate = summary["total_fn"] / total_gt if total_gt else 0.0
    overall_false_rate = summary["total_fp"] / total_pred if total_pred else 0.0

    metric_cols = st.columns(6)
    metric_cols[0].metric("IoU 阈值", f"{iou_threshold:.2f}")
    metric_cols[1].metric("TP", summary["total_tp"])
    metric_cols[2].metric("FN", summary["total_fn"])
    metric_cols[3].metric("FP", summary["total_fp"])
    metric_cols[4].metric("整体漏检率", f"{overall_miss_rate:.2%}")
    metric_cols[5].metric("整体虚检率", f"{overall_false_rate:.2%}")

    current_case_path = st.session_state.get("error_case_image")
    if current_case_path:
        st.caption(f"当前单图案例: {Path(current_case_path).name}。点选下方任意统计表行可切换。")

    tab_fn, tab_fp, tab_case, tab_detail = st.tabs(["漏检分析", "虚检分析", "单图案例", "总明细"])

    with tab_fn:
        st.markdown("**漏检分析**")
        fn_metric_cols = st.columns(4)
        fn_metric_cols[0].metric("漏检总数", summary["total_fn"])
        fn_metric_cols[1].metric("发生漏检的图片数", summary["images_with_fn"])
        fn_metric_cols[2].metric("平均每张问题图漏检数", f"{summary['total_fn'] / max(summary['images_with_fn'], 1):.2f}")
        fn_metric_cols[3].metric("整体漏检率", f"{overall_miss_rate:.2%}")

        col1, col2 = st.columns(2)
        if not fn_reason_df.empty:
            col1.pyplot(plot_bar(fn_reason_df["reason"].astype(str).tolist(), fn_reason_df["count"].astype(float).tolist(), "漏检原因分布", "Count", color="#d62728"), clear_figure=True)
        if not fn_size_df.empty:
            col2.pyplot(plot_bar(fn_size_df["size_bucket"].astype(str).tolist(), fn_size_df["count"].astype(float).tolist(), "漏检尺寸分布", "Count", color="#f58518"), clear_figure=True)

        fn_image_rank_df = (
            fn_df.groupby(["image_path", "image_name"]).size().reset_index(name="fn_count").sort_values(["fn_count", "image_name"], ascending=[False, True]).head(30)
            if not fn_df.empty
            else pd.DataFrame()
        )
        if not fn_image_rank_df.empty:
            _render_linkable_table(
                fn_image_rank_df,
                state_key="error_case_image",
                table_key="error_fn_rank_table",
                columns=["image_name", "fn_count"],
                caption="漏检最多的图片: 点选后同步到单图案例。",
                selection_guard=selection_guard,
            )

        if not fn_class_df.empty:
            st.markdown("**按类别漏检**")
            st.dataframe(fn_class_df, use_container_width=True)

        fn_col1, fn_col2, fn_col3 = st.columns(3)
        fn_image_keyword = fn_col1.text_input("图片名包含", value="", key="fn_image_keyword")
        fn_class_options = ["全部"] + sorted(fn_df["class_name"].astype(str).unique().tolist()) if not fn_df.empty else ["全部"]
        fn_reason_options = ["全部"] + sorted(fn_df["reason"].astype(str).unique().tolist()) if not fn_df.empty else ["全部"]
        fn_class_filter = fn_col2.selectbox("类别", fn_class_options, key="fn_class_filter")
        fn_reason_filter = fn_col3.selectbox("漏检原因", fn_reason_options, key="fn_reason_filter")
        filtered_fn_df = _filter_detail(fn_df, image_keyword=fn_image_keyword, class_filter=fn_class_filter, reason_filter=fn_reason_filter)
        _render_selectable_table(
            filtered_fn_df.reset_index(drop=True),
            _format_case_table(filtered_fn_df[FN_COLUMNS] if not filtered_fn_df.empty else filtered_fn_df).reset_index(drop=True),
            state_key="error_case_image",
            table_key="error_fn_detail_table",
            selection_guard=selection_guard,
            height=320,
        )

    with tab_fp:
        st.markdown("**虚检分析**")
        fp_metric_cols = st.columns(4)
        fp_metric_cols[0].metric("虚检总数", summary["total_fp"])
        fp_metric_cols[1].metric("发生虚检的图片数", summary["images_with_fp"])
        fp_metric_cols[2].metric("平均每张问题图虚检数", f"{summary['total_fp'] / max(summary['images_with_fp'], 1):.2f}")
        fp_metric_cols[3].metric("整体虚检率", f"{overall_false_rate:.2%}")

        col3, col4 = st.columns(2)
        if not fp_reason_df.empty:
            col3.pyplot(plot_bar(fp_reason_df["reason"].astype(str).tolist(), fp_reason_df["count"].astype(float).tolist(), "虚检原因分布", "Count", color="#ff7f0e"), clear_figure=True)
        if not fp_conf_df.empty:
            col4.pyplot(plot_bar(fp_conf_df["confidence_bin"].astype(str).tolist(), fp_conf_df["count"].astype(float).tolist(), "虚检置信度分布", "Count", color="#4c78a8"), clear_figure=True)

        fp_image_rank_df = (
            fp_df.groupby(["image_path", "image_name"]).size().reset_index(name="fp_count").sort_values(["fp_count", "image_name"], ascending=[False, True]).head(30)
            if not fp_df.empty
            else pd.DataFrame()
        )
        if not fp_image_rank_df.empty:
            _render_linkable_table(
                fp_image_rank_df,
                state_key="error_case_image",
                table_key="error_fp_rank_table",
                columns=["image_name", "fp_count"],
                caption="虚检最多的图片: 点选后同步到单图案例。",
                selection_guard=selection_guard,
            )

        if not fp_class_df.empty:
            st.markdown("**按类别虚检**")
            st.dataframe(fp_class_df, use_container_width=True)

        fp_col1, fp_col2, fp_col3 = st.columns(3)
        fp_image_keyword = fp_col1.text_input("图片名包含", value="", key="fp_image_keyword")
        fp_class_options = ["全部"] + sorted(fp_df["class_name"].astype(str).unique().tolist()) if not fp_df.empty else ["全部"]
        fp_reason_options = ["全部"] + sorted(fp_df["reason"].astype(str).unique().tolist()) if not fp_df.empty else ["全部"]
        fp_class_filter = fp_col2.selectbox("类别", fp_class_options, key="fp_class_filter")
        fp_reason_filter = fp_col3.selectbox("虚检原因", fp_reason_options, key="fp_reason_filter")
        filtered_fp_df = _filter_detail(fp_df, image_keyword=fp_image_keyword, class_filter=fp_class_filter, reason_filter=fp_reason_filter)
        _render_selectable_table(
            filtered_fp_df.reset_index(drop=True),
            _format_case_table(filtered_fp_df[FP_COLUMNS] if not filtered_fp_df.empty else filtered_fp_df).reset_index(drop=True),
            state_key="error_case_image",
            table_key="error_fp_detail_table",
            selection_guard=selection_guard,
            height=320,
        )

    with tab_case:
        st.markdown("**单图案例分析**")
        if not hard_images_df.empty:
            rank_col1, rank_col2, rank_col3 = st.columns(3)
            with rank_col1:
                _render_linkable_table(
                    hard_images_df[["image_path", "image_name", "error_total", "fn", "fp"]].head(30),
                    state_key="error_case_image",
                    table_key="error_hard_total_table",
                    columns=["image_name", "error_total", "fn", "fp"],
                    caption="综合错误最多的图片。",
                    selection_guard=selection_guard,
                )
            with rank_col2:
                _render_linkable_table(
                    hard_images_df.sort_values(["fn", "error_total", "image_name"], ascending=[False, False, True])[["image_path", "image_name", "fn", "tp", "fp"]].head(30),
                    state_key="error_case_image",
                    table_key="error_hard_fn_table",
                    columns=["image_name", "fn", "tp", "fp"],
                    caption="漏检最多的案例图。",
                    selection_guard=selection_guard,
                )
            with rank_col3:
                _render_linkable_table(
                    hard_images_df.sort_values(["fp", "error_total", "image_name"], ascending=[False, False, True])[["image_path", "image_name", "fp", "tp", "fn"]].head(30),
                    state_key="error_case_image",
                    table_key="error_hard_fp_table",
                    columns=["image_name", "fp", "tp", "fn"],
                    caption="虚检最多的案例图。",
                    selection_guard=selection_guard,
                )

        selected_image = st.session_state.get("error_case_image")
        if selected_image:
            case = image_cases[selected_image]
            selected_meta = image_df[image_df["image_path"] == selected_image].iloc[0].to_dict()

            summary_cols = st.columns(6)
            summary_cols[0].metric("图片", selected_meta["image_name"])
            summary_cols[1].metric("GT", int(selected_meta["gt_count"]))
            summary_cols[2].metric("Pred", int(selected_meta["pred_count"]))
            summary_cols[3].metric("TP", int(selected_meta["tp"]))
            summary_cols[4].metric("FN", int(selected_meta["fn"]))
            summary_cols[5].metric("FP", int(selected_meta["fp"]))

            st.image(
                overlay_error_boxes(selected_image, case["gt_boxes"], case["tp_boxes"], case["fn_boxes"], case["fp_boxes"]),
                caption=f"GT / TP / FN / FP 叠加图 | {selected_meta['image_name']}",
                use_container_width=True,
            )

            split_cols = st.columns(4)
            split_cols[0].image(overlay_error_boxes(selected_image, case["gt_boxes"], [], [], []), caption=f"GT ({len(case['gt_boxes'])})", use_container_width=True)
            split_cols[1].image(overlay_error_boxes(selected_image, [], case["tp_boxes"], [], []), caption=f"TP ({len(case['tp_boxes'])})", use_container_width=True)
            split_cols[2].image(overlay_error_boxes(selected_image, [], [], case["fn_boxes"], []), caption=f"FN ({len(case['fn_boxes'])})", use_container_width=True)
            split_cols[3].image(overlay_error_boxes(selected_image, [], [], [], case["fp_boxes"]), caption=f"FP ({len(case['fp_boxes'])})", use_container_width=True)

            crop_tab1, crop_tab2, crop_tab3 = st.tabs(["问题局部", "案例明细", "原子图层"])
            with crop_tab1:
                crop_cols = st.columns(2)
                with crop_cols[0]:
                    _render_crop_gallery(selected_image, case["fn_boxes"], color=COLOR_FN, label_prefix="FN:", title="漏检目标局部", empty_text="当前图片没有漏检目标。")
                with crop_cols[1]:
                    _render_crop_gallery(selected_image, case["fp_boxes"], color=COLOR_FP, label_prefix="FP:", title="虚检目标局部", empty_text="当前图片没有虚检框。")
            with crop_tab2:
                case_df = detail_df[detail_df["image_path"] == selected_image]
                st.dataframe(_format_case_table(case_df), use_container_width=True, height=360)
            with crop_tab3:
                atom_cols = st.columns(2)
                atom_cols[0].image(overlay_error_boxes(selected_image, case["gt_boxes"], [], [], []), caption="GT 单独图层", use_container_width=True)
                atom_cols[0].image(overlay_error_boxes(selected_image, [], case["tp_boxes"], [], []), caption="TP 单独图层", use_container_width=True)
                atom_cols[1].image(overlay_error_boxes(selected_image, [], [], case["fn_boxes"], []), caption="FN 单独图层", use_container_width=True)
                atom_cols[1].image(overlay_error_boxes(selected_image, [], [], [], case["fp_boxes"]), caption="FP 单独图层", use_container_width=True)

    with tab_detail:
        st.markdown("**总明细**")
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        image_keyword = detail_col1.text_input("图片名包含", value="", key="error_image_keyword")
        class_options = ["全部"] + sorted(detail_df["class_name"].astype(str).unique().tolist()) if not detail_df.empty else ["全部"]
        error_options = ["全部", "TP", "FN", "FP"]
        class_filter = detail_col2.selectbox("类别", class_options, key="error_class_filter")
        error_filter = detail_col3.selectbox("错误类型", error_options, key="error_type_filter")

        filtered_detail = detail_df.copy()
        if image_keyword:
            filtered_detail = filtered_detail[filtered_detail["image_name"].str.contains(image_keyword, case=False, na=False)]
        if class_filter != "全部":
            filtered_detail = filtered_detail[filtered_detail["class_name"] == class_filter]
        if error_filter != "全部":
            filtered_detail = filtered_detail[filtered_detail["error_type"] == error_filter]
        _render_selectable_table(
            filtered_detail.reset_index(drop=True),
            _format_case_table(filtered_detail).reset_index(drop=True),
            state_key="error_case_image",
            table_key="error_total_detail_table",
            selection_guard=selection_guard,
            height=320,
        )

    st.markdown("**导出**")
    export_cols = st.columns(4)
    export_cols[0].download_button(
        "导出图片级统计 CSV",
        data=dataframe_to_csv_bytes(image_df),
        file_name="error_image_summary.csv",
        mime="text/csv",
    )
    export_cols[1].download_button(
        "导出漏检明细 CSV",
        data=dataframe_to_csv_bytes(fn_df),
        file_name="fn_details.csv",
        mime="text/csv",
    )
    export_cols[2].download_button(
        "导出虚检明细 CSV",
        data=dataframe_to_csv_bytes(fp_df),
        file_name="fp_details.csv",
        mime="text/csv",
    )
    export_cols[3].download_button(
        "导出总明细 CSV",
        data=dataframe_to_csv_bytes(detail_df),
        file_name="error_details.csv",
        mime="text/csv",
    )

