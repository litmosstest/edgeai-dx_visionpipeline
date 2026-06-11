from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISION_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8081
    database_path: Path = Path("data/events.db")
    media_dir: Path = Path("data/media")

    rtsp_url: str = "rtsp://localhost:8554/webcam"
    camera_id: str = "usb-webcam-1"
    device: str = "cuda"

    capture_fps: int = Field(default=8, ge=1, le=60)
    sample_every_n_frames: int = Field(default=8, ge=1)
    min_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    target_labels: str = ""
    event_cooldown_seconds: int = Field(default=8, ge=0)

    detector_backend: str = "yolo"
    detector_model: str = "yolo11n.pt"
    embedding_backend: str = "clip"
    embedding_model: str = "sentence-transformers/clip-ViT-B-32"
    video_embedding_backend: str = "frame_average"
    video_embedding_model: str = "microsoft/xclip-base-patch32"
    video_embedding_frames: int = Field(default=8, ge=1, le=64)
    vlm_backend: str = "template"
    vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_settings() -> Settings:
    return Settings()
