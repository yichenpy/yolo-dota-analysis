from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from PIL import Image
import yaml

from .exceptions import ConfigurationError, DatasetError
from .schemas import DatasetContext

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_class_names(names: object) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def load_data_yaml(data_yaml_path: str | Path) -> dict:
    path = Path(data_yaml_path).expanduser().resolve()
    if not path.exists():
        raise ConfigurationError(f"data.yaml 不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError("data.yaml 内容无效，应为字典格式")
    return data


def _dataset_root(data_yaml: Path, data: dict) -> Path:
    raw_path = data.get("path", ".")
    if raw_path in (None, "", "."):
        return data_yaml.parent.resolve()

    base_path = Path(str(raw_path)).expanduser()
    candidate_paths: list[Path] = []
    if base_path.is_absolute():
        candidate_paths.append(base_path)
    else:
        candidate_paths.append(data_yaml.parent / base_path)
    candidate_paths.append(data_yaml.parent)

    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return data_yaml.parent.resolve()


def _resolve_source_to_images(source: object, root: Path) -> list[Path]:
    if isinstance(source, (list, tuple)):
        paths: list[Path] = []
        for item in source:
            paths.extend(_resolve_source_to_images(item, root))
        return paths

    source_path = Path(str(source))
    resolved = source_path if source_path.is_absolute() else (root / source_path)
    resolved = resolved.expanduser().resolve()
    if resolved.is_dir():
        return list_images(resolved)
    if resolved.is_file() and resolved.suffix.lower() == ".txt":
        image_paths: list[Path] = []
        with resolved.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                item_path = Path(text)
                image_paths.append(item_path if item_path.is_absolute() else (root / item_path).resolve())
        return [path for path in image_paths if path.exists() and path.suffix.lower() in IMAGE_SUFFIXES]
    if resolved.is_file() and resolved.suffix.lower() in IMAGE_SUFFIXES:
        return [resolved]
    raise ConfigurationError(f"无法解析数据源: {resolved}")


def list_images(image_dir: str | Path) -> list[Path]:
    directory = Path(image_dir).expanduser().resolve()
    if not directory.exists():
        raise ConfigurationError(f"图像目录不存在: {directory}")
    image_paths = sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise DatasetError(f"图像目录为空: {directory}")
    return image_paths


def infer_label_dir(image_dir: Path) -> Optional[Path]:
    parts = list(image_dir.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts)
    if image_dir.name == "images":
        return image_dir.parent / "labels"
    if image_dir.parent.name == "images":
        return image_dir.parent.parent / "labels" / image_dir.name
    sibling = image_dir.parent / "labels"
    return sibling if sibling.exists() else None


def image_to_label_path(image_path: str | Path, image_dir: str | Path | None, label_dir: str | Path | None) -> Path:
    image = Path(image_path).expanduser().resolve()
    if image_dir and label_dir:
        image_root = Path(image_dir).expanduser().resolve()
        label_root = Path(label_dir).expanduser().resolve()
        try:
            relative = image.relative_to(image_root)
            return (label_root / relative).with_suffix(".txt")
        except ValueError:
            pass
    parts = list(image.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def read_image_size(image_path: str | Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def read_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def build_dataset_context(
    *,
    data_yaml: str | Path | None,
    image_dir: str | Path | None,
    label_dir: str | Path | None,
    split: str,
) -> DatasetContext:
    resolved_image_dir = Path(image_dir).expanduser().resolve() if image_dir else None
    resolved_label_dir = Path(label_dir).expanduser().resolve() if label_dir else None
    class_names: dict[int, str] = {}
    image_paths: list[Path] = []
    dataset_name = "custom_dataset"
    data_yaml_path = Path(data_yaml).expanduser().resolve() if data_yaml else None

    if data_yaml_path:
        data = load_data_yaml(data_yaml_path)
        class_names = normalize_class_names(data.get("names"))
        dataset_name = data.get("name", data_yaml_path.stem)
        dataset_root = _dataset_root(data_yaml_path, data)
        source = data.get(split)
        if source:
            image_paths = _resolve_source_to_images(source, dataset_root)
            if image_paths and resolved_image_dir is None:
                resolved_image_dir = image_paths[0].parent
        elif resolved_image_dir:
            image_paths = list_images(resolved_image_dir)

    if resolved_image_dir and not image_paths:
        image_paths = list_images(resolved_image_dir)

    if not image_paths:
        raise DatasetError("未找到任何图像。请检查 data.yaml、图像目录或上传内容。")

    if not resolved_label_dir and resolved_image_dir:
        inferred = infer_label_dir(resolved_image_dir)
        if inferred and inferred.exists():
            resolved_label_dir = inferred.resolve()

    return DatasetContext(
        dataset_name=dataset_name,
        split=split,
        image_paths=[str(path) for path in sorted(image_paths)],
        class_names=class_names,
        image_dir=str(resolved_image_dir) if resolved_image_dir else None,
        label_dir=str(resolved_label_dir) if resolved_label_dir else None,
        data_yaml=str(data_yaml_path) if data_yaml_path else None,
    )


def validate_image_and_label_dirs(image_dir: str | Path | None, label_dir: str | Path | None) -> None:
    if image_dir:
        directory = Path(image_dir).expanduser().resolve()
        if not directory.exists():
            raise ConfigurationError(f"图像目录不存在: {directory}")
    if label_dir:
        directory = Path(label_dir).expanduser().resolve()
        if not directory.exists():
            raise ConfigurationError(f"标签目录不存在: {directory}")


def choose_images_for_preview(image_paths: Iterable[str], limit: int = 50) -> list[str]:
    return list(sorted(image_paths))[:limit]
