from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
        }


@dataclass(frozen=True)
class VisualEvent:
    id: str
    camera_id: str
    timestamp: datetime
    label_summary: str
    confidence: float
    description: str
    description_backend: str
    image_path: Path
    detections: list[Detection]
    image_embedding: list[float]
    video_embedding: list[float]

    @property
    def embedding(self) -> list[float]:
        return self.image_embedding

    @classmethod
    def create(
        cls,
        camera_id: str,
        label_summary: str,
        confidence: float,
        description: str,
        image_path: Path,
        detections: list[Detection],
        description_backend: str = "unknown",
        embedding: list[float] | None = None,
        image_embedding: list[float] | None = None,
        video_embedding: list[float] | None = None,
    ) -> "VisualEvent":
        resolved_image_embedding = image_embedding if image_embedding is not None else embedding or []
        resolved_video_embedding = (
            video_embedding if video_embedding is not None else resolved_image_embedding
        )
        return cls(
            id=str(uuid4()),
            camera_id=camera_id,
            timestamp=utc_now(),
            label_summary=label_summary,
            confidence=confidence,
            description=description,
            description_backend=description_backend,
            image_path=image_path,
            detections=detections,
            image_embedding=resolved_image_embedding,
            video_embedding=resolved_video_embedding,
        )
