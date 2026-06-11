from pathlib import Path

from fastapi.testclient import TestClient

from vision_pipeline.api import create_app
from vision_pipeline.config import Settings
from vision_pipeline.models import Detection, VisualEvent


def test_search_reuses_cached_embedder(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "events.db",
        media_dir=tmp_path / "media",
        embedding_backend="hash",
        embedding_model="unused",
        device="cpu",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        first_response = client.post("/api/search", json={"query": "person", "limit": 3})
        state = app.state.pipeline
        first_embedder = state._search_embedder
        second_response = client.post("/api/search", json={"query": "door", "limit": 3})

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert first_embedder is not None
        assert state._search_embedder is first_embedder


def test_search_reports_incompatible_embeddings(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "events.db",
        media_dir=tmp_path / "media",
        embedding_backend="hash",
        embedding_model="unused",
        device="cpu",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        event = VisualEvent.create(
            camera_id="test-camera",
            label_summary="person",
            confidence=0.9,
            description="Person detected.",
            image_path=tmp_path / "event.jpg",
            detections=[Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))],
            embedding=[1.0, 0.0],
        )
        app.state.pipeline.store.add_event(event)

        response = client.post("/api/search", json={"query": "person", "limit": 3})

        assert response.status_code == 200
        payload = response.json()
        assert payload["events"] == []
        assert payload["query_dimensions"] == 384
        assert payload["skipped_vectors"] == 2


def test_delete_event_endpoint(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "events.db",
        media_dir=tmp_path / "media",
        embedding_backend="hash",
        embedding_model="unused",
        device="cpu",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        event = VisualEvent.create(
            camera_id="test-camera",
            label_summary="person",
            confidence=0.9,
            description="Person detected.",
            image_path=tmp_path / "event.jpg",
            detections=[Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))],
            embedding=[1.0, 0.0],
        )
        app.state.pipeline.store.add_event(event)

        response = client.delete(f"/api/events/{event.id}")

        assert response.status_code == 200
        assert response.json()["deleted"] == 1
        assert app.state.pipeline.store.count_events() == 0


def test_delete_events_endpoint_requires_scope(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "events.db",
        media_dir=tmp_path / "media",
        embedding_backend="hash",
        embedding_model="unused",
        device="cpu",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post("/api/events/delete", json={})

        assert response.status_code == 400


def test_event_embeddings_endpoint(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "events.db",
        media_dir=tmp_path / "media",
        embedding_backend="hash",
        embedding_model="unused",
        device="cpu",
    )
    app = create_app(settings)

    with TestClient(app) as client:
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
        app.state.pipeline.store.add_event(event)

        events_response = client.get("/api/events?limit=1")
        response = client.get(f"/api/events/{event.id}/embeddings")
        values_response = client.get(f"/api/events/{event.id}/embeddings?include_values=true")

        assert events_response.status_code == 200
        assert events_response.json()["events"][0]["processing_status"] == {
            "image_embedding": True,
            "video_embedding": True,
            "vlm_description": True,
        }
        assert response.status_code == 200
        assert response.json()["image"]["dimensions"] == 2
        assert "vectors" not in response.json()
        assert values_response.status_code == 200
        assert values_response.json()["vectors"]["video"] == [0.0, 1.0]