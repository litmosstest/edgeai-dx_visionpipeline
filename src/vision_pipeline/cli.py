from __future__ import annotations

import argparse
import logging
import socket
import sys

import uvicorn
from PIL import Image

from vision_pipeline.config import load_settings
from vision_pipeline.db import EventStore
from vision_pipeline.embeddings import build_image_embedder
from vision_pipeline.vlm import build_describer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI vision pipeline services.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("api", help="Run the FastAPI dashboard and pipeline API")
    reembed_parser = subparsers.add_parser(
        "reembed-events",
        help="Rebuild stored event embeddings from saved event images using current settings",
    )
    reembed_parser.add_argument("--limit", type=int, default=None)
    reembed_parser.add_argument("--dry-run", action="store_true")
    describe_parser = subparsers.add_parser(
        "describe-events",
        help="Generate VLM descriptions for stored event images using current settings",
    )
    describe_parser.add_argument("--limit", type=int, default=None)
    describe_parser.add_argument("--dry-run", action="store_true")
    describe_parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Skip rows that already have a non-empty description",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "api":
        settings = load_settings()
        if not is_port_available(settings.host, settings.port):
            print(
                f"Port {settings.port} is already in use on {settings.host}. "
                f"Stop the existing service or run with VISION_PORT={settings.port + 1}.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        uvicorn.run("vision_pipeline.api:app", host=settings.host, port=settings.port, reload=False)
    elif args.command == "reembed-events":
        reembed_events(limit=args.limit, dry_run=args.dry_run)
    elif args.command == "describe-events":
        describe_events(
            limit=args.limit,
            dry_run=args.dry_run,
            keep_existing=args.keep_existing,
        )


def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def reembed_events(limit: int | None = None, dry_run: bool = False) -> None:
    settings = load_settings()
    store = EventStore(settings.database_path)
    rows = store.list_event_media(limit=limit)
    if dry_run:
        print(f"Would re-embed {len(rows)} events using {settings.embedding_model}.")
        return

    embedder = build_image_embedder(
        settings.embedding_backend,
        settings.embedding_model,
        settings.device,
    )
    updated = 0
    missing = 0
    for row in rows:
        image_path = row["image_path"]
        if not image_path.exists():
            missing += 1
            continue
        with Image.open(image_path) as image:
            image_embedding = embedder.embed_image(image.convert("RGB"))
        store.update_event_embeddings(row["id"], image_embedding)
        updated += 1
        print(f"Re-embedded {updated}/{len(rows)} events", end="\r", flush=True)
    print(f"Re-embedded {updated} events; skipped {missing} missing images.")


def describe_events(
    limit: int | None = None,
    dry_run: bool = False,
    keep_existing: bool = False,
) -> None:
    settings = load_settings()
    store = EventStore(settings.database_path)
    rows = store.list_event_description_inputs(limit=limit)
    if keep_existing:
        rows = [row for row in rows if not row["description"].strip()]
    if dry_run:
        print(f"Would describe {len(rows)} events using {settings.vlm_backend}:{settings.vlm_model}.")
        return

    describer = build_describer(
        settings.vlm_backend,
        settings.vlm_model,
        settings.device,
    )
    updated = 0
    missing = 0
    for row in rows:
        image_path = row["image_path"]
        if not image_path.exists():
            missing += 1
            continue
        with Image.open(image_path) as image:
            description = describer.describe(image.convert("RGB"), row["detections"])
        store.update_event_description(row["id"], description, settings.vlm_backend)
        updated += 1
        print(f"Described {updated}/{len(rows)} events", end="\r", flush=True)
    print(f"Described {updated} events; skipped {missing} missing images.")


if __name__ == "__main__":
    main()
