from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from vision_pipeline.config import Settings
from vision_pipeline.db import EventStore
from vision_pipeline.detectors import Detector, build_detector
from vision_pipeline.embeddings import Embedder, build_embedder
from vision_pipeline.models import Detection, VisualEvent, utc_now
from vision_pipeline.vlm import EventDescriber, build_describer, summarize_labels

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineStatus:
    running: bool
    camera_id: str
    rtsp_url: str
    frames_seen: int
    events_seen: int
    last_error: str | None
    latest_frame_url: str | None
    latest_frame_width: int | None
    latest_frame_height: int | None
    latest_detections: list[dict[str, object]]


class VisionPipeline:
    def __init__(
        self,
        settings: Settings,
        store: EventStore,
        detector: Detector,
        embedder: Embedder,
        describer: EventDescriber,
    ) -> None:
        self.settings = settings
        self.store = store
        self.detector = detector
        self.embedder = embedder
        self.describer = describer
        self.frames_seen = 0
        self.events_seen = 0
        self.last_error: str | None = None
        self.latest_frame_path: Path | None = None
        self.latest_frame_width: int | None = None
        self.latest_frame_height: int | None = None
        self.latest_detections: list[Detection] = []
        self._recent_event_frames: deque[Image.Image] = deque(
            maxlen=self.settings.video_embedding_frames
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_event_at_by_label: dict[str, float] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.last_error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vision-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def status(self) -> PipelineStatus:
        return PipelineStatus(
            running=bool(self._thread and self._thread.is_alive()),
            camera_id=self.settings.camera_id,
            rtsp_url=self.settings.rtsp_url,
            frames_seen=self.frames_seen,
            events_seen=self.events_seen,
            last_error=self.last_error,
            latest_frame_url=(
                f"/media/{self.latest_frame_path.name}" if self.latest_frame_path else None
            ),
            latest_frame_width=self.latest_frame_width,
            latest_frame_height=self.latest_frame_height,
            latest_detections=[detection.as_dict() for detection in self.latest_detections],
        )

    def _run(self) -> None:
        try:
            self._consume_rtsp()
        except Exception as error:
            LOGGER.exception("Vision pipeline stopped after an error")
            self.last_error = str(error)

    def _consume_rtsp(self) -> None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is not installed. Run: pip install -e '.[models]'") from error

        capture = cv2.VideoCapture(self.settings.rtsp_url)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open RTSP stream: {self.settings.rtsp_url}. "
                "Make sure MediaMTX is running and FFmpeg is actively publishing the webcam "
                "to that exact RTSP path. Run: docker compose up -d mediamtx && "
                "./scripts/publish_webcam_rtsp.sh, then verify with ./scripts/check_rtsp.sh."
            )

        frame_interval = 1.0 / max(self.settings.capture_fps, 1)
        while not self._stop.is_set():
            started_at = time.monotonic()
            ok, frame_bgr = capture.read()
            if not ok:
                self.last_error = "Failed to read frame from RTSP stream"
                time.sleep(0.5)
                continue

            self.last_error = None
            self.frames_seen += 1
            if self.frames_seen % self.settings.sample_every_n_frames == 0:
                self._process_frame(frame_bgr)

            elapsed = time.monotonic() - started_at
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
        capture.release()

    def _process_frame(self, frame_bgr: np.ndarray) -> None:
        detections = self.detector.detect(frame_bgr)
        detections = filter_target_labels(detections, self.settings.target_labels)
        image = bgr_to_image(frame_bgr)
        self._recent_event_frames.append(image.copy())
        height, width = frame_bgr.shape[:2]
        self._update_latest_sample(image, detections, width, height)

        if not detections:
            return
        if not self._should_emit(detections):
            return

        timestamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        image_path = self.settings.media_dir / f"{self.settings.camera_id}-{timestamp}.jpg"
        save_image_atomic(image, image_path, quality=88)

        label_summary = summarize_labels(detections)
        confidence = max(detection.confidence for detection in detections)
        description = self.describer.describe(image, detections)
        image_embedding = self.embedder.embed_image(image)
        video_embedding = self.embedder.embed_video(list(self._recent_event_frames))
        event = VisualEvent.create(
            camera_id=self.settings.camera_id,
            label_summary=label_summary,
            confidence=confidence,
            description=description,
            image_path=image_path,
            detections=detections,
            image_embedding=image_embedding,
            video_embedding=video_embedding,
        )
        self.store.add_event(event)
        self.events_seen += 1
        LOGGER.info("Stored event %s: %s", event.id, event.label_summary)

    def _update_latest_sample(
        self,
        image: Image.Image,
        detections: list[Detection],
        width: int,
        height: int,
    ) -> None:
        self.settings.media_dir.mkdir(parents=True, exist_ok=True)
        latest_frame_path = self.settings.media_dir / f"{self.settings.camera_id}-latest.jpg"
        save_image_atomic(image, latest_frame_path, quality=82)
        self.latest_frame_path = latest_frame_path
        self.latest_frame_width = width
        self.latest_frame_height = height
        self.latest_detections = detections

    def _should_emit(self, detections: list[Detection]) -> bool:
        now = time.monotonic()
        cooldown = self.settings.event_cooldown_seconds
        labels = sorted({detection.label for detection in detections})
        event_key = ",".join(labels)
        last_event_at = self._last_event_at_by_label.get(event_key, 0.0)
        if now - last_event_at < cooldown:
            return False
        self._last_event_at_by_label[event_key] = now
        return True


class PipelineController:
    def __init__(self, settings: Settings, store: EventStore) -> None:
        self.settings = settings
        self.store = store
        self.pipeline: VisionPipeline | None = None

    def start(self) -> PipelineStatus:
        if self.pipeline is None:
            detector = build_detector(
                self.settings.detector_backend,
                self.settings.detector_model,
                self.settings.device,
                self.settings.min_confidence,
            )
            embedder = build_embedder(
                self.settings.embedding_backend,
                self.settings.embedding_model,
                self.settings.device,
            )
            describer = build_describer(
                self.settings.vlm_backend,
                self.settings.vlm_model,
                self.settings.device,
            )
            self.pipeline = VisionPipeline(self.settings, self.store, detector, embedder, describer)
        self.pipeline.start()
        return self.pipeline.status()

    def stop(self) -> PipelineStatus:
        if self.pipeline:
            self.pipeline.stop()
            return self.pipeline.status()
        return PipelineStatus(
            False,
            self.settings.camera_id,
            self.settings.rtsp_url,
            0,
            0,
            None,
            None,
            None,
            None,
            [],
        )

    def status(self) -> PipelineStatus:
        if self.pipeline:
            return self.pipeline.status()
        return PipelineStatus(
            False,
            self.settings.camera_id,
            self.settings.rtsp_url,
            0,
            0,
            None,
            None,
            None,
            None,
            [],
        )


def bgr_to_image(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = frame_bgr[:, :, ::-1]
    return Image.fromarray(frame_rgb)


def save_image_atomic(image: Image.Image, destination: Path, quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    image.save(temporary_path, quality=quality)
    temporary_path.replace(destination)


def filter_target_labels(detections: Sequence[Detection], target_labels: str) -> list[Detection]:
    labels = {label.strip().lower() for label in target_labels.split(",") if label.strip()}
    if not labels:
        return list(detections)
    return [detection for detection in detections if detection.label.lower() in labels]
