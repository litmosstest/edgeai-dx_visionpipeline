from pathlib import Path

from PIL import Image

from vision_pipeline.cli import describe_events
from vision_pipeline.db import EventStore
from vision_pipeline.models import Detection, VisualEvent


def test_describe_events_updates_saved_descriptions(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "events.db"
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    image_path = media_dir / "event.jpg"
    Image.new("RGB", (32, 24), color="white").save(image_path)

    store = EventStore(database_path)
    event = VisualEvent.create(
        camera_id="test-camera",
        label_summary="person",
        confidence=0.9,
        description="Detected person in the camera view.",
        image_path=image_path,
        detections=[Detection("person", 0.9, (1.0, 2.0, 3.0, 4.0))],
        embedding=[1.0, 0.0],
    )
    store.add_event(event)

    monkeypatch.setenv("VISION_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("VISION_MEDIA_DIR", str(media_dir))
    monkeypatch.setenv("VISION_VLM_BACKEND", "template")
    monkeypatch.setenv("VISION_DEVICE", "cpu")

    describe_events(limit=1)

    updated = store.get_event(event.id)
    assert updated is not None
    assert updated["description"] == "Camera frame 32x24: detected person; highest detector confidence 90%."