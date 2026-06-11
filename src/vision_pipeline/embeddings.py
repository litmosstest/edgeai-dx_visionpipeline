from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


class Embedder(Protocol):
    def embed_image(self, image: Image.Image) -> list[float]:
        ...

    def embed_video(self, frames: list[Image.Image]) -> list[float]:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_video_text(self, text: str) -> list[float]:
        ...


class VideoTextEmbedder(Protocol):
    def embed_video(self, frames: list[Image.Image]) -> list[float]:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...


@dataclass
class ClipEmbedder:
    model_name: str
    device: str

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install -e '.[models]'"
            ) from error
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_image(self, image: Image.Image) -> list[float]:
        vector = self.model.encode(image, normalize_embeddings=True)
        return vector.astype(np.float32).tolist()

    def embed_video(self, frames: list[Image.Image]) -> list[float]:
        if not frames:
            return []
        frame_vectors = np.asarray([self.embed_image(frame) for frame in frames], dtype=np.float32)
        return normalize_vector(frame_vectors.mean(axis=0)).tolist()

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.astype(np.float32).tolist()

    def embed_video_text(self, text: str) -> list[float]:
        return self.embed_text(text)


@dataclass
class HashingEmbedder:
    dimensions: int = 384

    def embed_image(self, image: Image.Image) -> list[float]:
        resized = image.convert("RGB").resize((32, 32))
        payload = resized.tobytes()
        return self._embed_bytes(payload)

    def embed_video(self, frames: list[Image.Image]) -> list[float]:
        if not frames:
            return []
        frame_vectors = np.asarray([self.embed_image(frame) for frame in frames], dtype=np.float32)
        return normalize_vector(frame_vectors.mean(axis=0)).tolist()

    def embed_text(self, text: str) -> list[float]:
        return self._embed_bytes(text.lower().encode("utf-8"))

    def embed_video_text(self, text: str) -> list[float]:
        return self.embed_text(text)

    def _embed_bytes(self, payload: bytes) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for offset in range(0, len(payload), 32):
            digest = hashlib.blake2b(payload[offset : offset + 32], digest_size=16).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()


@dataclass
class XClipVideoEmbedder:
    model_name: str
    device: str

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import XCLIPModel, XCLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "transformers and torch are required for X-CLIP video embeddings. "
                "Run: pip install -e '.[models]'"
            ) from error

        self.torch = torch
        self.processor = XCLIPProcessor.from_pretrained(self.model_name)
        self.model = XCLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def embed_video(self, frames: list[Image.Image]) -> list[float]:
        if not frames:
            return []
        rgb_frames = [frame.convert("RGB") for frame in frames]
        inputs = self.processor(videos=[rgb_frames], return_tensors="pt")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            features = self.model.get_video_features(pixel_values=inputs["pixel_values"])
        return torch_vector_to_list(features, self.torch)

    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return torch_vector_to_list(features, self.torch)


@dataclass
class RoutedEmbedder:
    image_embedder: Embedder
    video_embedder: VideoTextEmbedder

    def embed_image(self, image: Image.Image) -> list[float]:
        return self.image_embedder.embed_image(image)

    def embed_video(self, frames: list[Image.Image]) -> list[float]:
        return self.video_embedder.embed_video(frames)

    def embed_text(self, text: str) -> list[float]:
        return self.image_embedder.embed_text(text)

    def embed_video_text(self, text: str) -> list[float]:
        return self.video_embedder.embed_text(text)


def build_embedder(
    backend: str,
    model_name: str,
    device: str,
    video_backend: str = "frame_average",
    video_model_name: str = "",
) -> Embedder:
    image_embedder = build_image_embedder(backend, model_name, device)
    video_embedder = build_video_embedder(video_backend, video_model_name, device)
    if video_embedder is None:
        return image_embedder
    return RoutedEmbedder(image_embedder=image_embedder, video_embedder=video_embedder)


def build_image_embedder(backend: str, model_name: str, device: str) -> Embedder:
    normalized = backend.lower()
    if normalized in {"clip", "mobileclip", "siglip"}:
        return ClipEmbedder(model_name=model_name, device=device)
    if normalized in {"hash", "hashing", "demo"}:
        return HashingEmbedder()
    raise ValueError(f"Unsupported embedding backend: {backend}")


def build_video_embedder(
    backend: str,
    model_name: str,
    device: str,
) -> VideoTextEmbedder | None:
    normalized = backend.lower()
    if normalized in {"", "frame_average", "frame-average", "average", "averaged", "inherit"}:
        return None
    if normalized in {"xclip", "x-clip", "video"}:
        if not model_name:
            raise ValueError("X-CLIP video embeddings require a video embedding model name.")
        return XClipVideoEmbedder(model_name=model_name, device=device)
    raise ValueError(f"Unsupported video embedding backend: {backend}")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


def torch_vector_to_list(features: object, torch_module: object) -> list[float]:
    normalized = torch_module.nn.functional.normalize(features, dim=-1)
    vector = normalized[0].detach().cpu().float().numpy()
    return vector.astype(np.float32).tolist()
