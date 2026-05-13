from __future__ import annotations

from pathlib import Path

from .exceptions import DatasetError
from .schemas import BoxRecord


def parse_yolo_label_file(
    label_path: str | Path,
    image_path: str | Path,
    image_size: tuple[int, int],
    class_names: dict[int, str],
) -> list[BoxRecord]:
    path = Path(label_path).expanduser().resolve()
    if not path.exists():
        return []

    image = Path(image_path).expanduser().resolve()
    width, height = image_size
    image_id = image.stem
    boxes: list[BoxRecord] = []

    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            fields = text.split()
            class_name = ""

            try:
                class_id = int(float(fields[0]))
                class_name = class_names.get(class_id, str(class_id))
            except ValueError as exc:
                raise DatasetError(f"标签类别解析失败: {path} 第 {row_index} 行") from exc

            if len(fields) == 5:
                try:
                    cx, cy, box_w, box_h = map(float, fields[1:5])
                except ValueError as exc:
                    raise DatasetError(f"标签数值解析失败: {path} 第 {row_index} 行") from exc

                x1 = (cx - box_w / 2.0) * width
                y1 = (cy - box_h / 2.0) * height
                x2 = (cx + box_w / 2.0) * width
                y2 = (cy + box_h / 2.0) * height
                meta = {"label_format": "bbox"}
            elif len(fields) == 9:
                try:
                    polygon = [float(value) for value in fields[1:9]]
                except ValueError as exc:
                    raise DatasetError(f"OBB 标签数值解析失败: {path} 第 {row_index} 行") from exc
                xs = [polygon[index] * width for index in range(0, 8, 2)]
                ys = [polygon[index] * height for index in range(1, 8, 2)]
                x1 = min(xs)
                y1 = min(ys)
                x2 = max(xs)
                y2 = max(ys)
                meta = {
                    "label_format": "obb",
                    "polygon": [coord for pair in zip(xs, ys) for coord in pair],
                }
            else:
                raise DatasetError(
                    f"暂不支持的标签格式: {path} 第 {row_index} 行共有 {len(fields)} 列。"
                    "当前支持普通检测框(5列)和 OBB 四点框(9列)。"
                )

            boxes.append(
                BoxRecord(
                    image_id=image_id,
                    image_path=str(image),
                    class_id=class_id,
                    class_name=class_name,
                    x1=max(0.0, x1),
                    y1=max(0.0, y1),
                    x2=min(width, x2),
                    y2=min(height, y2),
                    confidence=None,
                    source="gt",
                    meta=meta,
                )
            )
    return boxes
