from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from vision_pipeline.models import Detection, VisualEvent


class EventStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def migrate(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    label_summary TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    description TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    detections_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    image_embedding_json TEXT,
                    video_embedding_json TEXT
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(events)").fetchall()
            }
            if "image_embedding_json" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN image_embedding_json TEXT")
            if "video_embedding_json" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN video_embedding_json TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)"
            )

    def add_event(self, event: VisualEvent) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO events (
                    id, camera_id, timestamp, label_summary, confidence, description,
                    image_path, detections_json, embedding_json, image_embedding_json,
                    video_embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.camera_id,
                    event.timestamp.isoformat(),
                    event.label_summary,
                    event.confidence,
                    event.description,
                    str(event.image_path),
                    json.dumps([detection.as_dict() for detection in event.detections]),
                    json.dumps(event.embedding),
                    json.dumps(event.image_embedding),
                    json.dumps(event.video_embedding),
                ),
            )

    def list_events(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, camera_id, timestamp, label_summary, confidence, description,
                      image_path, detections_json, embedding_json, image_embedding_json,
                      video_embedding_json
                FROM events
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._row_to_public_dict(row) for row in rows]

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, camera_id, timestamp, label_summary, confidence, description,
                      image_path, detections_json, embedding_json, image_embedding_json,
                      video_embedding_json
                FROM events
                WHERE id = ?
                """,
                (event_id,),
            ).fetchone()
        return None if row is None else self._row_to_public_dict(row)

    def get_event_embeddings(
        self,
        event_id: str,
        include_values: bool = False,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, embedding_json, image_embedding_json, video_embedding_json
                FROM events
                WHERE id = ?
                """,
                (event_id,),
            ).fetchone()
        if row is None:
            return None

        legacy_vector = json.loads(row["embedding_json"])
        image_vector = json.loads(row["image_embedding_json"] or row["embedding_json"])
        video_vector = json.loads(row["video_embedding_json"] or row["embedding_json"])
        payload: dict[str, Any] = {
            "id": row["id"],
            "image": embedding_summary(image_vector),
            "video": embedding_summary(video_vector),
            "legacy": embedding_summary(legacy_vector),
        }
        if include_values:
            payload["vectors"] = {
                "image": image_vector,
                "video": video_vector,
                "legacy": legacy_vector,
            }
        return payload

    def search_by_vector(
        self,
        query_vector: list[float],
        limit: int = 20,
        embedding_type: str = "any",
        query_text: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.search_by_vector_with_stats(
            query_vector,
            limit=limit,
            embedding_type=embedding_type,
            query_text=query_text,
        )["events"]

    def search_by_vector_with_stats(
        self,
        query_vector: list[float],
        limit: int = 20,
        embedding_type: str = "any",
        query_text: str | None = None,
    ) -> dict[str, Any]:
        if not query_vector:
            return {
                "events": [],
                "query_dimensions": 0,
                "compatible_events": 0,
                "skipped_vectors": 0,
            }

        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM events ORDER BY timestamp DESC").fetchall()

        scored: list[tuple[float, sqlite3.Row, float, float]] = []
        skipped_vectors = 0
        for row in rows:
            vectors = event_vectors(row, embedding_type)
            compatible_vectors = [vector for vector in vectors if len(vector) == len(query_vector)]
            skipped_vectors += len(vectors) - len(compatible_vectors)
            if not compatible_vectors:
                continue
            vector_score = max(cosine_similarity(query_vector, vector) for vector in compatible_vectors)
            text_score = text_relevance(query_text or "", row)
            score = vector_score + text_score
            scored.append((score, row, vector_score, text_score))

        scored.sort(key=lambda item: item[0], reverse=True)
        return {
            "events": [
                self._row_to_public_dict(
                    row,
                    score=score,
                    vector_score=vector_score,
                    text_score=text_score,
                )
                for score, row, vector_score, text_score in scored[:limit]
            ],
            "query_dimensions": len(query_vector),
            "compatible_events": len(scored),
            "skipped_vectors": skipped_vectors,
        }

    def count_events(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM events").fetchone()
        return int(row["count"])

    def delete_event(self, event_id: str, delete_media: bool = True) -> int:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT image_path FROM events WHERE id = ?",
                (event_id,),
            ).fetchall()
            connection.execute("DELETE FROM events WHERE id = ?", (event_id,))
        if delete_media:
            unlink_event_media(rows)
        return len(rows)

    def delete_events_before(self, cutoff: datetime, delete_media: bool = True) -> int:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT image_path FROM events WHERE timestamp < ?",
                (cutoff.isoformat(),),
            ).fetchall()
            connection.execute("DELETE FROM events WHERE timestamp < ?", (cutoff.isoformat(),))
        if delete_media:
            unlink_event_media(rows)
        return len(rows)

    def clear_events(self, delete_media: bool = True) -> int:
        with self.connect() as connection:
            rows = connection.execute("SELECT image_path FROM events").fetchall()
            connection.execute("DELETE FROM events")
        if delete_media:
            unlink_event_media(rows)
        return len(rows)

    def list_event_media(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT id, image_path FROM events ORDER BY timestamp DESC"
        params: tuple[int, ...] = ()
        if limit is not None:
            query = f"{query} LIMIT ?"
            params = (limit,)
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [{"id": row["id"], "image_path": Path(row["image_path"])} for row in rows]

    def update_event_embeddings(
        self,
        event_id: str,
        image_embedding: list[float],
        video_embedding: list[float] | None = None,
    ) -> None:
        resolved_video_embedding = video_embedding if video_embedding is not None else image_embedding
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE events
                SET embedding_json = ?, image_embedding_json = ?, video_embedding_json = ?
                WHERE id = ?
                """,
                (
                    json.dumps(image_embedding),
                    json.dumps(image_embedding),
                    json.dumps(resolved_video_embedding),
                    event_id,
                ),
            )

    def iter_embeddings(self) -> Iterable[list[float]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT embedding_json FROM events").fetchall()
        for row in rows:
            yield json.loads(row["embedding_json"])

    def _row_to_public_dict(
        self,
        row: sqlite3.Row,
        score: float | None = None,
        vector_score: float | None = None,
        text_score: float | None = None,
    ) -> dict[str, Any]:
        image_path = Path(row["image_path"])
        payload = {
            "id": row["id"],
            "camera_id": row["camera_id"],
            "timestamp": row["timestamp"],
            "label_summary": row["label_summary"],
            "confidence": row["confidence"],
            "description": row["description"],
            "image_path": str(image_path),
            "image_url": f"/media/{image_path.name}",
            "detections": json.loads(row["detections_json"]),
            "embeddings": embedding_metadata(row),
        }
        if score is not None:
            payload["score"] = score
        if vector_score is not None:
            payload["vector_score"] = vector_score
        if text_score is not None:
            payload["text_score"] = text_score
        return payload


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    denominator = left_norm * right_norm
    if denominator == 0.0:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right)) / denominator


def max_embedding_score(
    query_vector: list[float], row: sqlite3.Row, embedding_type: str = "any"
) -> float:
    vectors = event_vectors(row, embedding_type)
    if not vectors:
        return 0.0
    return max(cosine_similarity(query_vector, vector) for vector in vectors)


def event_vectors(row: sqlite3.Row, embedding_type: str = "any") -> list[list[float]]:
    legacy_vector = json.loads(row["embedding_json"])
    image_vector = json.loads(row["image_embedding_json"] or row["embedding_json"])
    video_vector = json.loads(row["video_embedding_json"] or row["embedding_json"])

    normalized = embedding_type.lower()
    if normalized == "image":
        return [image_vector]
    if normalized == "video":
        return [video_vector]
    if normalized == "legacy":
        return [legacy_vector]
    return [image_vector, video_vector]


def embedding_metadata(row: sqlite3.Row) -> dict[str, Any]:
    image_vector = json.loads(row["image_embedding_json"] or row["embedding_json"])
    video_vector = json.loads(row["video_embedding_json"] or row["embedding_json"])
    return {
        "image": embedding_summary(image_vector),
        "video": embedding_summary(video_vector),
    }


def embedding_summary(vector: list[float]) -> dict[str, Any]:
    return {
        "present": bool(vector),
        "dimensions": len(vector),
        "preview": vector[:8],
    }


def text_relevance(query_text: str, row: sqlite3.Row) -> float:
    query_terms = tokenize(query_text)
    if not query_terms:
        return 0.0

    detections = json.loads(row["detections_json"])
    labels = tokenize(row["label_summary"])
    labels.update(tokenize(" ".join(item["label"] for item in detections)))
    description = tokenize(row["description"])

    label_matches = len(query_terms & labels)
    description_matches = len(query_terms & description)
    if label_matches == 0 and description_matches == 0:
        return 0.0
    label_score = 0.45 * label_matches / len(query_terms)
    description_score = 0.2 * description_matches / len(query_terms)
    return min(0.65, label_score + description_score)


def tokenize(value: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", value.lower()) if len(term) >= 2}


def unlink_event_media(rows: list[sqlite3.Row]) -> None:
    for row in rows:
        image_path = Path(row["image_path"])
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            continue


def detections_from_json(payload: str) -> list[Detection]:
    return [
        Detection(
            label=item["label"],
            confidence=float(item["confidence"]),
            bbox_xyxy=tuple(item["bbox_xyxy"]),
        )
        for item in json.loads(payload)
    ]


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)
