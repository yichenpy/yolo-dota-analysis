from __future__ import annotations

import io
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import box_polygon, polygon_bounds
from .io import read_image
from .preprocess import letterbox_params, scale_polygon_to_letterbox
from .schemas import BoxRecord

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

COLOR_GT = "#1f77b4"
COLOR_TP = "#2ca02c"
COLOR_FN = "#d62728"
COLOR_FP = "#ff7f0e"
COLOR_PRED = "#17becf"


def _load_font(size: int = 16):
    for font_name in ["msyh.ttc", "simhei.ttf", "arialuni.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _clean_numeric(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return array
    return array[np.isfinite(array)]


def _adaptive_bin_count(values: np.ndarray, min_bins: int = 10, max_bins: int = 80) -> int:
    if values.size <= 1:
        return 1
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        bin_width = 2.0 * iqr / np.cbrt(values.size)
        if bin_width > 0:
            bins = int(np.ceil((values.max() - values.min()) / bin_width))
        else:
            bins = int(np.sqrt(values.size))
    else:
        bins = int(np.sqrt(values.size))
    return max(min_bins, min(max_bins, bins))


def _adaptive_focus_limits(values: np.ndarray) -> tuple[float, float, int, int, bool]:
    if values.size == 0:
        return 0.0, 1.0, 0, 0, False

    data_min = float(values.min())
    data_max = float(values.max())
    if np.isclose(data_min, data_max):
        pad = max(abs(data_min) * 0.05, 1.0)
        return data_min - pad, data_max + pad, 0, 0, False

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        low = float(q1 - 1.5 * iqr)
        high = float(q3 + 1.5 * iqr)
    else:
        q5, q95 = np.percentile(values, [5, 95])
        spread = float(q95 - q5)
        pad = spread * 0.1 if spread > 0 else max(abs(data_max) * 0.05, 1.0)
        low = float(q5 - pad)
        high = float(q95 + pad)

    if data_min >= 0 and low < 0:
        low = 0.0

    left_outliers = int((values < low).sum())
    right_outliers = int((values > high).sum())
    has_outliers = left_outliers + right_outliers > 0
    if not has_outliers:
        return data_min, data_max, 0, 0, False

    if np.isclose(low, high):
        pad = max(abs(high) * 0.05, 1.0)
        low -= pad
        high += pad
    return low, high, left_outliers, right_outliers, True


def _draw_distribution_note(ax, low: float, high: float, left_outliers: int, right_outliers: int) -> None:
    notes: list[str] = [f"主分布区间: [{low:.3g}, {high:.3g}]"]
    if left_outliers > 0:
        notes.append(f"左侧离群点: {left_outliers}")
    if right_outliers > 0:
        notes.append(f"右侧离群点: {right_outliers}")
    ax.text(
        0.98,
        0.96,
        "\n".join(notes),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )


def _draw_outline(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, width: int) -> None:
    if not points:
        return
    if len(points) == 4:
        draw.line(points + [points[0]], fill=color, width=width)
        return
    if len(points) >= 2:
        draw.line(points + [points[0]], fill=color, width=width)


def _label_anchor(points: list[tuple[float, float]]) -> tuple[float, float]:
    left, top, _, _ = polygon_bounds(points)
    return left + 2.0, max(0.0, top - 14.0)


def draw_box_layer(
    draw: ImageDraw.ImageDraw,
    boxes: list[BoxRecord],
    color: str,
    *,
    label_prefix: str = "",
    width: int = 3,
) -> None:
    font = _load_font()
    for box in boxes:
        points = box_polygon(box)
        _draw_outline(draw, points, color, width)
        confidence_text = f" {box.confidence:.2f}" if box.confidence is not None else ""
        text = f"{label_prefix}{box.class_name}{confidence_text}"
        text_position = _label_anchor(points)
        draw.text(text_position, text, fill=color, font=font)


def overlay_error_boxes(
    image_path: str,
    gt_boxes: list[BoxRecord],
    tp_boxes: list[BoxRecord],
    fn_boxes: list[BoxRecord],
    fp_boxes: list[BoxRecord],
) -> Image.Image:
    image = read_image(image_path).copy()
    draw = ImageDraw.Draw(image)
    draw_box_layer(draw, gt_boxes, COLOR_GT, label_prefix="GT:", width=2)
    draw_box_layer(draw, tp_boxes, COLOR_TP, label_prefix="TP:", width=3)
    draw_box_layer(draw, fn_boxes, COLOR_FN, label_prefix="FN:", width=4)
    draw_box_layer(draw, fp_boxes, COLOR_FP, label_prefix="FP:", width=4)
    return image


def overlay_prediction_boxes(image_path: str, pred_boxes: list[BoxRecord]) -> Image.Image:
    image = read_image(image_path).copy()
    draw = ImageDraw.Draw(image)
    draw_box_layer(draw, pred_boxes, COLOR_PRED, label_prefix="Pred:", width=3)
    return image


def draw_scaled_box_view(image_path: str, gt_boxes: list[BoxRecord], imgsz: int) -> Image.Image:
    image = read_image(image_path)
    width, height = image.size
    params = letterbox_params(width, height, imgsz)
    canvas = Image.new("RGB", (imgsz, imgsz), color=(35, 35, 35))
    resized = image.resize((params["resized_width"], params["resized_height"]))
    canvas.paste(resized, (int(params["pad_left"]), int(params["pad_top"])))

    draw = ImageDraw.Draw(canvas)
    font = _load_font()
    for box in gt_boxes:
        scaled_points = scale_polygon_to_letterbox(box, params)
        _draw_outline(draw, scaled_points, COLOR_GT, 2)
        draw.text(_label_anchor(scaled_points), box.class_name, fill=COLOR_GT, font=font)
    return canvas


def extract_box_crops(
    image_path: str,
    boxes: list[BoxRecord],
    *,
    color: str,
    label_prefix: str,
    max_items: int = 12,
    padding_ratio: float = 0.25,
) -> list[tuple[Image.Image, str]]:
    if not boxes:
        return []

    image = read_image(image_path).copy()
    width, height = image.size
    font = _load_font(size=14)
    crops: list[tuple[Image.Image, str]] = []
    for index, box in enumerate(boxes[:max_items], start=1):
        points = box_polygon(box)
        left_bound, top_bound, right_bound, bottom_bound = polygon_bounds(points)
        pad_x = max(8, int(max(1.0, right_bound - left_bound) * padding_ratio))
        pad_y = max(8, int(max(1.0, bottom_bound - top_bound) * padding_ratio))
        left = max(0, int(left_bound) - pad_x)
        top = max(0, int(top_bound) - pad_y)
        right = min(width, int(right_bound) + pad_x)
        bottom = min(height, int(bottom_bound) + pad_y)
        crop = image.crop((left, top, right, bottom)).copy()
        draw = ImageDraw.Draw(crop)
        local_points = [(point[0] - left, point[1] - top) for point in points]
        _draw_outline(draw, local_points, color, 3)
        confidence_text = f" {box.confidence:.2f}" if box.confidence is not None else ""
        text = f"{label_prefix}{box.class_name}{confidence_text}"
        text_x, text_y = _label_anchor(local_points)
        draw.text((text_x, text_y), text, fill=color, font=font)
        caption = f"{index}. {box.class_name} | ({box.width:.1f} x {box.height:.1f})"
        crops.append((crop, caption))
    return crops


def plot_histogram(values, title: str, xlabel: str, bins: int = 30):
    data = _clean_numeric(values)
    fig, ax = plt.subplots(figsize=(8, 4))
    if data.size == 0:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        return fig

    low, high, left_outliers, right_outliers, focused = _adaptive_focus_limits(data)
    visible = data[(data >= low) & (data <= high)] if focused else data
    if visible.size == 0:
        visible = data
        low = float(data.min())
        high = float(data.max())
        focused = False

    adaptive_bins = _adaptive_bin_count(visible) if bins == 30 else bins
    ax.hist(visible, bins=adaptive_bins, color="#4c78a8", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_xlim(low, high)
    ax.grid(alpha=0.2, linestyle="--")

    ax.axvline(float(np.median(data)), color="#f58518", linestyle="--", linewidth=1.2, label="median")
    ax.axvline(float(np.mean(data)), color="#54a24b", linestyle=":", linewidth=1.2, label="mean")
    ax.legend(fontsize=8)
    if focused:
        _draw_distribution_note(ax, low, high, left_outliers, right_outliers)

    fig.tight_layout()
    return fig


def plot_boxplot(data: list[np.ndarray], labels: list[str], title: str, ylabel: str):
    cleaned = [_clean_numeric(item) for item in data]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(cleaned, labels=labels, patch_artist=True, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linestyle="--", axis="y")
    fig.tight_layout()
    return fig


def plot_scatter(x, y, title: str, xlabel: str, ylabel: str, color="#e45756"):
    x_array = np.asarray(x, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[mask]
    y_array = y_array[mask]

    fig, ax = plt.subplots(figsize=(8, 4))
    if x_array.size == 0:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        return fig

    x_low, x_high, _, _, x_focused = _adaptive_focus_limits(x_array)
    y_low, y_high, _, _, y_focused = _adaptive_focus_limits(y_array)
    visible_mask = (x_array >= x_low) & (x_array <= x_high) & (y_array >= y_low) & (y_array <= y_high)
    hidden_points = int((~visible_mask).sum())

    plot_x = x_array[visible_mask] if hidden_points > 0 and visible_mask.any() else x_array
    plot_y = y_array[visible_mask] if hidden_points > 0 and visible_mask.any() else y_array

    ax.scatter(plot_x, plot_y, alpha=0.65, s=18, color=color, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linestyle="--")

    if hidden_points > 0:
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)
        ax.text(
            0.98,
            0.96,
            f"主分布视图\n隐藏离群点: {hidden_points}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
        )
    elif x_focused:
        ax.set_xlim(x_low, x_high)
    elif y_focused:
        ax.set_ylim(y_low, y_high)

    fig.tight_layout()
    return fig


def plot_bar(labels: list[str], values: list[float], title: str, ylabel: str, color: str = "#72b7b2"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.2, linestyle="--", axis="y")
    fig.tight_layout()
    return fig


def plot_pr_curves(pr_curves: dict[str, dict[str, list[float]]]):
    fig, ax = plt.subplots(figsize=(8, 6))
    has_curve = False
    for class_name, curve in pr_curves.items():
        recall = np.asarray(curve.get("recall", []))
        precision = np.asarray(curve.get("precision", []))
        if recall.size == 0 or precision.size == 0:
            continue
        ax.plot(recall, precision, label=class_name)
        has_curve = True
    ax.set_title("PR 曲线 @ IoU=0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2, linestyle="--")
    if has_curve:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str]):
    fig, ax = plt.subplots(figsize=(8, 7))
    display = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("混淆矩阵")
    fig.colorbar(display, ax=ax, fraction=0.046, pad=0.04)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, int(matrix[row_index, col_index]), ha="center", va="center", color="black")
    fig.tight_layout()
    return fig


def plot_feature_average_heatmap(feature_map: np.ndarray, title: str):
    activation = feature_map.mean(axis=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    display = ax.imshow(activation, cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(display, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_feature_channel(feature_map: np.ndarray, channel_index: int, title: str):
    channels = feature_map.shape[0]
    channel_index = max(0, min(channel_index, channels - 1))
    fig, ax = plt.subplots(figsize=(6, 6))
    display = ax.imshow(feature_map[channel_index], cmap="viridis")
    ax.set_title(f"{title} - channel {channel_index}")
    ax.axis("off")
    fig.colorbar(display, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_feature_grid(feature_map: np.ndarray, max_channels: int, title: str):
    count = min(max_channels, feature_map.shape[0])
    cols = min(4, max(1, count))
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).flatten()
    for index, axis in enumerate(axes):
        if index < count:
            axis.imshow(feature_map[index], cmap="viridis")
            axis.set_title(f"ch {index}")
            axis.axis("off")
        else:
            axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def figure_to_image_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()

