from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class InferenceConfig:
    weights_path: str
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    device: str = "cpu"
    max_det: int = 300
    batch_size: int = 1
    cpu_fallback: bool = True


@dataclass
class DatasetContext:
    dataset_name: str
    split: str
    image_paths: list[str]
    class_names: dict[int, str]
    image_dir: Optional[str] = None
    label_dir: Optional[str] = None
    data_yaml: Optional[str] = None

    @property
    def num_images(self) -> int:
        return len(self.image_paths)


@dataclass
class BoxRecord:
    image_id: str
    image_path: str
    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Optional[float] = None
    source: str = "gt"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "confidence": self.confidence,
            "source": self.source,
            **self.meta,
        }


@dataclass(frozen=True)
class MatchPair:
    gt_index: int
    pred_index: int
    iou: float


@dataclass
class FeatureMapResult:
    layer_index: int
    layer_name: str
    stage: str
    feature_map: Any
    detections: list[BoxRecord]


def path_to_str(path: Optional[Path]) -> Optional[str]:
    return str(path) if path else None
