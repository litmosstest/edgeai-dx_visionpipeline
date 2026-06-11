from pathlib import Path
from datetime import timedelta

from vision_pipeline.db import EventStore
from vision_pipeline.models import Detection, VisualEvent, utc_now


def test_store_round_trips_events(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    event = VisualEvent.create(
        camera_id="test-camera",
        label_summary="person",
        confidence=0.92,
        description="Detected a person near the camera.",
        image_path=tmp_path / "frame.jpg",
        detections=[Detection("person", 0.92, (1.0, 2.0, 3.0, 4.0))],
        embedding=[1.0, 0.0, 0.0],
    )

    store.add_event(event)

    events = store.list_events()
    assert len(events) == 1
    assert events[0]["label_summary"] == "person"
    assert events[0]["detections"][0]["label"] == "person"
    assert events[0]["embeddings"]["image"]["dimensions"] == 3
    assert events[0]["embeddings"]["video"]["dimensions"] == 3
    assert events[0]["processing_status"] == {
        "image_embedding": True,
        "video_embedding": True,
        "vlm_description": True,
    }


def test_store_reports_processing_status_for_incomplete_events(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    event = VisualEvent.create(
        camera_id="test-camera",
        label_summary="person",
        confidence=0.92,
        description="",
        image_path=tmp_path / "frame.jpg",
        detections=[Detection("person", 0.92, (1.0, 2.0, 3.0, 4.0))],
        image_embedding=[1.0, 0.0, 0.0],
        video_embedding=[],
    )

    store.add_event(event)

    stored_event = store.get_event(event.id)
    assert stored_event is not None
    assert stored_event["processing_status"] == {
        "image_embedding": True,
        "video_embedding": False,
        "vlm_description": False,
    }


def test_store_updates_event_description(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    event = VisualEvent.create(
        camera_id="test-camera",
        label_summary="person",
        confidence=0.92,
        description="Detected person in the camera view.",
        image_path=tmp_path / "frame.jpg",
        detections=[Detection("person", 0.92, (1.0, 2.0, 3.0, 4.0))],
        embedding=[1.0, 0.0, 0.0],
    )
    store.add_event(event)

    store.update_event_description(event.id, "A person is visible near the camera.")

    updated = store.get_event(event.id)
    assert updated is not None
    assert updated["description"] == "A person is visible near the camera."


def test_vector_search_orders_by_cosine_similarity(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    first = VisualEvent.create(
        camera_id="test-camera",
        label_summary="forklift",
        confidence=0.9,
        description="Forklift in an aisle.",
        image_path=tmp_path / "forklift.jpg",
        detections=[Detection("forklift", 0.9, (0.0, 0.0, 1.0, 1.0))],
        embedding=[1.0, 0.0],
    )
    second = VisualEvent.create(
        camera_id="test-camera",
        label_summary="door",
        confidence=0.8,
        description="Door is open.",
        image_path=tmp_path / "door.jpg",
        detections=[Detection("door", 0.8, (0.0, 0.0, 1.0, 1.0))],
        embedding=[0.0, 1.0],
    )
    store.add_event(first)
    store.add_event(second)

    results = store.search_by_vector([0.9, 0.1])

    assert results[0]["id"] == first.id
    assert results[0]["score"] > results[1]["score"]


def test_vector_search_can_target_video_embeddings(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    first = VisualEvent.create(
        camera_id="test-camera",
        label_summary="person",
        confidence=0.9,
        description="Person entering.",
        image_path=tmp_path / "person.jpg",
        detections=[Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))],
        image_embedding=[1.0, 0.0],
        video_embedding=[0.0, 1.0],
    )
    second = VisualEvent.create(
        camera_id="test-camera",
        label_summary="person",
        confidence=0.8,
        description="Person leaving.",
        image_path=tmp_path / "leaving.jpg",
        detections=[Detection("person", 0.8, (0.0, 0.0, 1.0, 1.0))],
        image_embedding=[0.0, 1.0],
        video_embedding=[1.0, 0.0],
    )
    store.add_event(first)
    store.add_event(second)

    image_results = store.search_by_vector([1.0, 0.0], embedding_type="image")
    video_results = store.search_by_vector([1.0, 0.0], embedding_type="video")

    assert image_results[0]["id"] == first.id
    assert video_results[0]["id"] == second.id


def test_typed_vector_search_keeps_embedding_spaces_separate(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    image_match = VisualEvent.create(
        camera_id="test-camera",
        label_summary="image-match",
        confidence=0.9,
        description="Image match.",
        image_path=tmp_path / "image.jpg",
        detections=[Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))],
        image_embedding=[1.0, 0.0],
        video_embedding=[1.0, 0.0],
    )
    video_match = VisualEvent.create(
        camera_id="test-camera",
        label_summary="video-match",
        confidence=0.8,
        description="Video match.",
        image_path=tmp_path / "video.jpg",
        detections=[Detection("person", 0.8, (0.0, 0.0, 1.0, 1.0))],
        image_embedding=[0.0, 1.0],
        video_embedding=[0.0, 1.0],
    )
    store.add_event(image_match)
    store.add_event(video_match)

    result = store.search_by_typed_vectors_with_stats({"video": [0.0, 1.0]})

    assert result["events"][0]["id"] == video_match.id


def test_vector_search_skips_incompatible_dimensions(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    incompatible = VisualEvent.create(
        camera_id="test-camera",
        label_summary="hash-event",
        confidence=0.9,
        description="Old event.",
        image_path=tmp_path / "old.jpg",
        detections=[Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))],
        embedding=[1.0, 0.0],
    )
    compatible = VisualEvent.create(
        camera_id="test-camera",
        label_summary="clip-event",
        confidence=0.8,
        description="Current event.",
        image_path=tmp_path / "current.jpg",
        detections=[Detection("person", 0.8, (0.0, 0.0, 1.0, 1.0))],
        embedding=[0.0, 1.0, 0.0],
    )
    store.add_event(incompatible)
    store.add_event(compatible)

    result = store.search_by_vector_with_stats([0.0, 1.0, 0.0])

    assert result["events"][0]["id"] == compatible.id
    assert result["compatible_events"] == 1
    assert result["skipped_vectors"] == 2


def test_vector_search_boosts_matching_labels(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    vector_match = VisualEvent.create(
        camera_id="test-camera",
        label_summary="demo-target, watch-zone",
        confidence=0.9,
        description="Synthetic dashboard event.",
        image_path=tmp_path / "demo.jpg",
        detections=[Detection("demo-target", 0.9, (0.0, 0.0, 1.0, 1.0))],
        embedding=[1.0, 0.0],
    )
    label_match = VisualEvent.create(
        camera_id="test-camera",
        label_summary="person",
        confidence=0.8,
        description="Detected person in the camera view.",
        image_path=tmp_path / "person.jpg",
        detections=[Detection("person", 0.8, (0.0, 0.0, 1.0, 1.0))],
        embedding=[0.8, 0.2],
    )
    store.add_event(vector_match)
    store.add_event(label_match)

    result = store.search_by_vector_with_stats([1.0, 0.0], query_text="person")

    assert result["events"][0]["id"] == label_match.id
    assert result["events"][0]["text_score"] > 0


def test_update_event_embeddings_preserves_video_vector_by_default(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    event = VisualEvent.create(
        camera_id="test-camera",
        label_summary="person",
        confidence=0.9,
        description="Person detected.",
        image_path=tmp_path / "event.jpg",
        detections=[Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))],
        image_embedding=[1.0, 0.0],
        video_embedding=[0.0, 1.0],
    )
    store.add_event(event)

    store.update_event_embeddings(event.id, [0.0, 1.0, 0.0])

    embeddings = store.get_event_embeddings(event.id, include_values=True)
    assert embeddings is not None
    assert embeddings["image"]["dimensions"] == 3
    assert embeddings["vectors"]["video"] == [0.0, 1.0]


def test_update_event_embeddings_replaces_video_when_supplied(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    event = VisualEvent.create(
        camera_id="test-camera",
        label_summary="person",
        confidence=0.9,
        description="Person detected.",
        image_path=tmp_path / "event.jpg",
        detections=[Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))],
        image_embedding=[1.0, 0.0],
        video_embedding=[0.0, 1.0],
    )
    store.add_event(event)

    store.update_event_embeddings(
        event.id,
        image_embedding=[0.0, 1.0, 0.0],
        video_embedding=[0.5, 0.5, 0.5],
    )

    embeddings = store.get_event_embeddings(event.id, include_values=True)
    assert embeddings is not None
    assert embeddings["vectors"]["video"] == [0.5, 0.5, 0.5]


def test_delete_event_removes_row_and_media(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    image_path = tmp_path / "event.jpg"
    image_path.write_bytes(b"frame")
    event = VisualEvent.create(
        camera_id="test-camera",
        label_summary="person",
        confidence=0.9,
        description="Person detected.",
        image_path=image_path,
        detections=[Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))],
        embedding=[1.0, 0.0],
    )
    store.add_event(event)

    deleted = store.delete_event(event.id)

    assert deleted == 1
    assert store.count_events() == 0
    assert not image_path.exists()


def test_delete_events_before_cutoff(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    old = VisualEvent.create(
        camera_id="test-camera",
        label_summary="old",
        confidence=0.9,
        description="Old event.",
        image_path=tmp_path / "old.jpg",
        detections=[Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))],
        embedding=[1.0, 0.0],
    )
    recent = VisualEvent.create(
        camera_id="test-camera",
        label_summary="recent",
        confidence=0.9,
        description="Recent event.",
        image_path=tmp_path / "recent.jpg",
        detections=[Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))],
        embedding=[0.0, 1.0],
    )
    store.add_event(old)
    store.add_event(recent)

    with store.connect() as connection:
        connection.execute(
            "UPDATE events SET timestamp = ? WHERE id = ?",
            ((utc_now() - timedelta(days=10)).isoformat(), old.id),
        )
    deleted = store.delete_events_before(utc_now() - timedelta(days=7))

    events = store.list_events()
    assert deleted == 1
    assert len(events) == 1
    assert events[0]["id"] == recent.id


def test_get_event_embeddings_returns_summary_and_values(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    event = VisualEvent.create(
        camera_id="test-camera",
        label_summary="person",
        confidence=0.9,
        description="Person detected.",
        image_path=tmp_path / "event.jpg",
        detections=[Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))],
        image_embedding=[1.0, 0.0, 0.5],
        video_embedding=[0.0, 1.0, 0.5],
    )
    store.add_event(event)

    summary = store.get_event_embeddings(event.id)
    values = store.get_event_embeddings(event.id, include_values=True)

    assert summary is not None
    assert summary["image"]["dimensions"] == 3
    assert summary["video"]["preview"] == [0.0, 1.0, 0.5]
    assert "vectors" not in summary
    assert values is not None
    assert values["vectors"]["image"] == [1.0, 0.0, 0.5]
