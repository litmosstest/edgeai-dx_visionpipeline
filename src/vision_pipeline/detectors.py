from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vision_pipeline.models import Detection


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        ...


@dataclass
class YoloDetector:
    model_name: str
    device: str
    min_confidence: float

    def __post_init__(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -e '.[models]'"
            ) from error
        self.model = YOLO(self.model_name)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame_bgr,
            conf=self.min_confidence,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.min_confidence:
                    continue
                class_id = int(box.cls[0])
                label = str(names.get(class_id, class_id))
                xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
                detections.append(Detection(label=label, confidence=confidence, bbox_xyxy=xyxy))
        return detections


class NoopDetector:
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return []


class DemoDetector:
    def __init__(self) -> None:
        self.frame_count = 0

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        self.frame_count += 1
        height, width = frame_bgr.shape[:2]
        phase = (self.frame_count % 80) / 80
        box_width = width * 0.22
        box_height = height * 0.26
        x1 = width * (0.12 + 0.48 * phase)
        y1 = height * 0.20
        x2 = min(x1 + box_width, width - 1)
        y2 = min(y1 + box_height, height - 1)
        return [
            Detection("demo-target", 0.88, (x1, y1, x2, y2)),
            Detection(
                "watch-zone",
                0.64,
                (width * 0.58, height * 0.52, width * 0.88, height * 0.82),
            ),
        ]


def build_detector(backend: str, model_name: str, device: str, min_confidence: float) -> Detector:
    normalized = backend.lower()
    if normalized == "yolo":
        return YoloDetector(model_name=model_name, device=device, min_confidence=min_confidence)
    if normalized in {"demo", "test"}:
        return DemoDetector()
    if normalized in {"noop", "none", "disabled"}:
        return NoopDetector()
    raise ValueError(f"Unsupported detector backend: {backend}")
