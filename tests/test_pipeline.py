from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from vision_pipeline.config import Settings
from vision_pipeline.db import EventStore
from vision_pipeline.pipeline import VisionPipeline, save_image_atomic


class EmptyDetector:
    def detect(self, frame_bgr):
        return []


class UnusedEmbedder:
    def embed_image(self, image):
        return []

    def embed_text(self, text):
        return []

    def embed_video(self, frames):
        return []

    def embed_video_text(self, text):
        return []


class UnusedDescriber:
    def describe(self, image, detections):
        return ""


def test_save_image_atomic_preserves_image_extension(tmp_path: Path) -> None:
    destination = tmp_path / "latest.jpg"
    image = Image.new("RGB", (8, 8), color="white")

    save_image_atomic(image, destination, quality=82)

    assert destination.exists()
    assert not list(tmp_path.glob(".latest.*.tmp.jpg"))
    with Image.open(destination) as saved_image:
        assert saved_image.format == "JPEG"


def test_save_image_atomic_replaces_existing_image(tmp_path: Path) -> None:
    destination = tmp_path / "latest.jpg"
    first_image = Image.new("RGB", (8, 8), color="white")
    second_image = Image.new("RGB", (8, 8), color="black")

    save_image_atomic(first_image, destination, quality=82)
    save_image_atomic(second_image, destination, quality=82)

    assert destination.exists()
    assert not list(tmp_path.glob(".latest.*.tmp.jpg"))
    with Image.open(destination) as saved_image:
        assert saved_image.format == "JPEG"


def test_pipeline_reconnects_after_repeated_rtsp_read_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeCapture:
        instances = []

        def __init__(self, url: str) -> None:
            self.url = url
            self.reads = 0
            self.released = False
            self.index = len(self.instances)
            self.instances.append(self)

        def isOpened(self) -> bool:
            return True

        def read(self):
            self.reads += 1
            if self.index == 0:
                return False, None
            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            return True, frame

        def release(self) -> None:
            self.released = True

    settings = Settings(
        database_path=tmp_path / "events.db",
        media_dir=tmp_path / "media",
        rtsp_url="rtsp://example.local/webcam",
        capture_fps=60,
        sample_every_n_frames=1,
        device="cpu",
    )
    pipeline = VisionPipeline(
        settings,
        EventStore(settings.database_path),
        EmptyDetector(),
        UnusedEmbedder(),
        UnusedDescriber(),
    )

    original_process_frame = pipeline._process_frame

    def stop_after_process(frame_bgr):
        original_process_frame(frame_bgr)
        pipeline._stop.set()

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=FakeCapture))
    monkeypatch.setattr("vision_pipeline.pipeline.time.sleep", lambda seconds: None)
    monkeypatch.setattr(pipeline, "_process_frame", stop_after_process)

    pipeline._consume_rtsp()

    assert len(FakeCapture.instances) == 2
    assert FakeCapture.instances[0].released is True
    assert pipeline.frames_seen == 1
    assert pipeline.last_error is None