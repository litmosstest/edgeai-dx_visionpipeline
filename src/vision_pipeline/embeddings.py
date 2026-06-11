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


def build_embedder(backend: str, model_name: str, device: str) -> Embedder:
    normalized = backend.lower()
    if normalized in {"clip", "mobileclip", "siglip"}:
        return ClipEmbedder(model_name=model_name, device=device)
    if normalized in {"hash", "hashing", "demo"}:
        return HashingEmbedder()
    raise ValueError(f"Unsupported embedding backend: {backend}")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector
