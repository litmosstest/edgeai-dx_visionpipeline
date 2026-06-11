from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from vision_pipeline.models import Detection


class EventDescriber(Protocol):
    def describe(self, image: Image.Image, detections: list[Detection]) -> str:
        ...


class TemplateDescriber:
    def describe(self, image: Image.Image, detections: list[Detection]) -> str:
        del image
        if not detections:
            return "No configured objects were detected in this sampled frame."
        labels = summarize_labels(detections)
        return f"Detected {labels} in the camera view."


@dataclass
class TransformersVlmDescriber:
    model_name: str
    device: str

    def __post_init__(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError("transformers is not installed. Run: pip install -e '.[models]'") from error
        device_map = "auto" if self.device == "cuda" else None
        self.pipeline = pipeline(
            "image-text-to-text",
            model=self.model_name,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    def describe(self, image: Image.Image, detections: list[Detection]) -> str:
        labels = summarize_labels(detections) if detections else "the scene"
        prompt = (
            "Write a concise security-camera event description. "
            f"Known detector outputs: {labels}. Mention only visible evidence."
        )
        result = self.pipeline(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_new_tokens=96,
        )
        return extract_generated_text(result)


def summarize_labels(detections: list[Detection]) -> str:
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.label] = counts.get(detection.label, 0) + 1
    parts = [f"{count} {label}" if count > 1 else label for label, count in sorted(counts.items())]
    return ", ".join(parts)


def build_describer(backend: str, model_name: str, device: str) -> EventDescriber:
    normalized = backend.lower()
    if normalized in {"template", "simple", "none"}:
        return TemplateDescriber()
    if normalized in {"transformers", "qwen", "qwen-vl"}:
        return TransformersVlmDescriber(model_name=model_name, device=device)
    raise ValueError(f"Unsupported VLM backend: {backend}")


def extract_generated_text(result: object) -> str:
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            generated = item.get("generated_text")
            if isinstance(generated, str):
                return generated.strip()
            if isinstance(generated, list) and generated:
                last = generated[-1]
                if isinstance(last, dict):
                    content = last.get("content")
                    if isinstance(content, str):
                        return content.strip()
    return "The VLM generated a response, but it could not be parsed into text."
