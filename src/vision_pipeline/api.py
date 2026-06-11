from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vision_pipeline.config import Settings, load_settings
from vision_pipeline.db import EventStore
from vision_pipeline.embeddings import Embedder, build_embedder
from vision_pipeline.models import utc_now
from vision_pipeline.pipeline import PipelineController


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=100)
    embedding_type: str = Field(default="any", pattern="^(any|image|video|legacy)$")


class DeleteEventsRequest(BaseModel):
    older_than_days: int | None = Field(default=None, ge=1, le=3650)
    all: bool = False
    delete_media: bool = True


class PipelineAppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = EventStore(settings.database_path)
        self.controller = PipelineController(settings, self.store)
        self._search_embedder: Embedder | None = None
        self._search_embedder_lock = threading.Lock()

    def search_embedder(self) -> Embedder:
        if self._search_embedder is None:
            with self._search_embedder_lock:
                if self._search_embedder is None:
                    self._search_embedder = build_embedder(
                        self.settings.embedding_backend,
                        self.settings.embedding_model,
                        self.settings.device,
                        self.settings.video_embedding_backend,
                        self.settings.video_embedding_model,
                    )
        return self._search_embedder


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or load_settings()
    resolved_settings.media_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = PipelineAppState(resolved_settings)
        yield
        app.state.pipeline.controller.stop()

    app = FastAPI(title="Vision Pipeline", version="0.1.0", lifespan=lifespan)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/media", StaticFiles(directory=resolved_settings.media_dir), name="media")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        state = get_state(app)
        return {
            "ok": True,
            "events": state.store.count_events(),
            "pipeline": state.controller.status().__dict__,
        }

    @app.get("/api/events")
    def events(limit: int = 50, offset: int = 0) -> dict[str, Any]:
        state = get_state(app)
        return {"events": state.store.list_events(limit=limit, offset=offset)}

    @app.get("/api/events/{event_id}")
    def event(event_id: str) -> dict[str, Any]:
        state = get_state(app)
        payload = state.store.get_event(event_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Event not found")
        return payload

    @app.get("/api/events/{event_id}/embeddings")
    def event_embeddings(event_id: str, include_values: bool = False) -> dict[str, Any]:
        state = get_state(app)
        payload = state.store.get_event_embeddings(event_id, include_values=include_values)
        if payload is None:
            raise HTTPException(status_code=404, detail="Event not found")
        return payload

    @app.delete("/api/events/{event_id}")
    def delete_event(event_id: str, delete_media: bool = True) -> dict[str, Any]:
        state = get_state(app)
        deleted = state.store.delete_event(event_id, delete_media=delete_media)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Event not found")
        return {"deleted": deleted}

    @app.post("/api/events/delete")
    def delete_events(request: DeleteEventsRequest) -> dict[str, Any]:
        state = get_state(app)
        if request.all:
            deleted = state.store.clear_events(delete_media=request.delete_media)
            return {"deleted": deleted}
        if request.older_than_days is None:
            raise HTTPException(
                status_code=400,
                detail="Provide older_than_days or set all=true.",
            )
        cutoff = utc_now() - timedelta(days=request.older_than_days)
        deleted = state.store.delete_events_before(cutoff, delete_media=request.delete_media)
        return {"deleted": deleted, "cutoff": cutoff.isoformat()}

    @app.post("/api/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        state = get_state(app)
        embedder = state.search_embedder()
        query_vectors = search_query_vectors(embedder, request.query, request.embedding_type)
        return state.store.search_by_typed_vectors_with_stats(
            query_vectors,
            limit=request.limit,
            query_text=request.query,
        )

    @app.post("/api/pipeline/start")
    def start_pipeline() -> dict[str, Any]:
        state = get_state(app)
        return state.controller.start().__dict__

    @app.post("/api/pipeline/stop")
    def stop_pipeline() -> dict[str, Any]:
        state = get_state(app)
        return state.controller.stop().__dict__

    @app.get("/api/pipeline/status")
    def pipeline_status() -> dict[str, Any]:
        state = get_state(app)
        return state.controller.status().__dict__

    return app


def get_state(app: FastAPI) -> PipelineAppState:
    return app.state.pipeline


def search_query_vectors(
    embedder: Embedder,
    query: str,
    embedding_type: str,
) -> dict[str, list[float]]:
    normalized = embedding_type.lower()
    if normalized == "image":
        return {"image": embedder.embed_text(query)}
    if normalized == "video":
        return {"video": embedder.embed_video_text(query)}
    if normalized == "legacy":
        return {"legacy": embedder.embed_text(query)}
    return {
        "image": embedder.embed_text(query),
        "video": embedder.embed_video_text(query),
    }


app = create_app()
