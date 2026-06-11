# Vision Pipeline

Open-source edge AI vision pipeline for an NVIDIA DGX Spark with a USB webcam. The starter turns the webcam into an RTSP source, samples the stream for object-detection events, stores event frames and metadata, embeds events for semantic recall, and exposes a web dashboard for search and review.

## AI Pipeline At A Glance

```text
USB webcam or test pattern
  1280x720 at 30 FPS default publisher, RTSP over TCP
    -> MediaMTX RTSP stream: rtsp://localhost:8554/webcam
    -> Python capture loop: 8 FPS target, sample every 8 frames
    -> Object detection: Ultralytics YOLO, yolo11n.pt by default
    -> Event filter: target labels, confidence >= 0.45, 8 second cooldown
    -> CLIP image embedding: sentence-transformers/clip-ViT-B-32
    -> Video embedding: average of the latest 8 CLIP frame embeddings
    -> VLM description: template backend by default, Qwen/Qwen2.5-VL-3B-Instruct for transformers
    -> SQLite event memory + saved media frames
    -> FastAPI dashboard, event search, and review UI
```

Default video publication is `/dev/video0` at `1280x720` and `30 FPS`; override with `DEVICE`, `SIZE`, and `FPS` when running `./scripts/publish_webcam_rtsp.sh`. The full model path defaults are also captured in `.env.example` and `configs/pipeline.yaml`; lightweight demos can swap in `noop`, `demo`, `hash`, and `template` backends without downloading GPU model weights.

## What It Builds

- USB webcam to RTSP with MediaMTX and FFmpeg.
- Object detection with a lazy-loaded Ultralytics YOLO adapter.
- Image and text embeddings with a lazy-loaded CLIP-compatible Sentence Transformers adapter.
- Video event embeddings by averaging a short rolling window of frame embeddings.
- SQLite event memory with cosine vector search in Python for the MVP.
- VLM descriptions through a lightweight template backend or a Transformers image-to-text adapter for local VLMs.
- FastAPI API and built-in dashboard at `http://localhost:8081` by default.
- Dashboard latest-frame and event-card overlays render detector bounding boxes.
- Event cards show stored image/video embedding dimensions and support deletion.

## System Architecture

```mermaid
flowchart LR
  subgraph Edge_Device[Edge AI Device]
    Camera[USB webcam<br/>/dev/video0]
    FFmpeg[FFmpeg publisher<br/>scripts/publish_webcam_rtsp.sh]
    MediaMTX[MediaMTX RTSP server<br/>rtsp://localhost:8554/webcam]
    API[FastAPI service<br/>vision-pipeline api]
    Controller[PipelineController]
    Capture[VisionPipeline<br/>RTSP capture loop]
    Detector[Detector adapter<br/>YOLO / demo / noop]
    Embedder[Embedding adapter<br/>CLIP / hash]
    VLM[VLM describer<br/>template / transformers]
    Store[(SQLite event store<br/>data/events.db)]
    Media[(Media files<br/>data/media)]
    Static[Built-in dashboard<br/>static HTML/CSS/JS]
  end

  Browser[Browser dashboard] -->|GET / and static assets| API
  Browser -->|REST API<br/>health, start/stop, events, search| API
  Camera --> FFmpeg --> MediaMTX --> Capture
  API --> Static
  API --> Controller --> Capture
  Capture --> Detector
  Capture --> Embedder
  Capture --> VLM
  Detector --> Event[Visual event composer]
  Embedder --> Event
  VLM --> Event
  Capture -->|latest frame| Media
  Event --> Store
  Event --> Media
  API -->|list/search/delete events| Store
  API -->|serve latest frames and event images| Media
```

The pipeline is designed as a small edge stack. FFmpeg publishes the local webcam into MediaMTX, the Python service samples the RTSP stream, model adapters enrich interesting frames, and FastAPI exposes both the API and a local dashboard. SQLite and the media directory are intentionally local-first for the MVP, making the system easy to run on a single DGX Spark or Linux workstation.

## Repository Layout

```text
src/vision_pipeline/
  api.py             FastAPI app and dashboard API
  pipeline.py        RTSP capture loop and event creation
  detectors.py       YOLO object detection adapter
  embeddings.py      CLIP and demo embedding adapters
  vlm.py             event description adapters
  db.py              SQLite event store and vector search
  static/            dashboard UI
scripts/
  publish_webcam_rtsp.sh
configs/
  pipeline.yaml
```

## Publish Target

This repository is prepared to publish to:

```text
https://github.com/litmosstest/vision_pipeline
```

Runtime data, virtual environments, downloaded model weights, `.env`, build outputs, and Python caches are ignored so the public repo contains source, tests, scripts, config, and documentation only.

## Quick Start

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[models,dev]'
docker compose up -d mediamtx
./scripts/publish_webcam_rtsp.sh
vision-pipeline api
```

Open `http://localhost:8081`, then press the start button in the dashboard or call:

```bash
curl -X POST http://localhost:8081/api/pipeline/start
```

For a lightweight UI/API demo without GPU model dependencies, set these in `.env`:

```bash
VISION_DETECTOR_BACKEND=noop
VISION_EMBEDDING_BACKEND=hash
VISION_VLM_BACKEND=template
```

Use `VISION_DETECTOR_BACKEND=demo` with `./scripts/publish_test_rtsp.sh` to see moving sample boxes without downloading model weights.

To store only people events from YOLO, set:

```bash
VISION_DETECTOR_BACKEND=yolo
VISION_TARGET_LABELS=person
VISION_EMBEDDING_BACKEND=clip
VISION_EMBEDDING_MODEL=sentence-transformers/clip-ViT-B-32
VISION_VIDEO_EMBEDDING_FRAMES=8
```

`demo-target` and `watch-zone` are generated only by the demo detector; they are not real object classes.

## RTSP Webcam Input

The default flow is:

1. MediaMTX listens on `rtsp://localhost:8554`.
2. FFmpeg publishes `/dev/video0` to `rtsp://localhost:8554/webcam`.
3. The Python pipeline reads `VISION_RTSP_URL` and samples frames.

Override camera settings when publishing:

```bash
DEVICE=/dev/video2 SIZE=1920x1080 FPS=30 ./scripts/publish_webcam_rtsp.sh
```

Before starting the pipeline, verify that the RTSP path is live:

```bash
./scripts/check_rtsp.sh
```

If this reports `404 Not Found`, MediaMTX is running but no publisher is currently sending video to `/webcam`. Start the publisher in a second terminal and leave it running:

```bash
./scripts/publish_webcam_rtsp.sh
```

To validate the RTSP and pipeline flow before camera permissions are fixed, publish a generated test pattern instead:

```bash
./scripts/publish_test_rtsp.sh
```

If FFmpeg prints `Cannot open video device /dev/video0: Permission denied`, your user probably cannot read the camera device. Check it with:

```bash
id
ls -l /dev/video*
```

On Linux, webcams are commonly owned by `root:video`. Add your user to that group, then log out and back in so the new group is visible to VS Code and your shell:

```bash
sudo usermod -aG video "$USER"
```

For a temporary fix that lasts until the device is recreated, use an ACL:

```bash
sudo setfacl -m u:$USER:rw /dev/video0
```

If the API prints that port `8081` is already in use, find the process or run on another port:

```bash
ss -ltnp 'sport = :8081'
VISION_PORT=8082 vision-pipeline api
```

## Model Curation Targets

| Stage | Starter | Edge candidates | Notes |
| --- | --- | --- | --- |
| Object detection | YOLO11n | YOLO11s, RT-DETR, RF-DETR, TensorRT-exported YOLO | Track latency, mAP, VRAM, and false positives per scene. |
| Image embeddings | CLIP ViT-B/32 | MobileCLIP, SigLIP, OpenCLIP | Stores a key-frame embedding for each event. |
| Video embeddings | Averaged CLIP frame window | X-CLIP, VideoCLIP, LanguageBind, SigLIP frame pooling | Stores a temporal embedding from the latest sampled event window for motion/context recall. |
| Vector search | SQLite + cosine scan | sqlite-vec, sqlite-vss, LanceDB | SQLite scan is simple for thousands of events; switch once event counts grow. |
| VLM descriptions | Template backend | Qwen2.5-VL, Qwen3-VL when available, InternVL | Run asynchronously for lower capture latency. |

## API

- `GET /api/health` returns service status and pipeline counters.
- `POST /api/pipeline/start` starts RTSP capture and event processing.
- `POST /api/pipeline/stop` stops capture.
- `GET /api/events?limit=50` lists recent events.
- `GET /api/events/{event_id}/embeddings` returns image/video embedding dimensions and vector previews.
- Add `?include_values=true` to return full stored vectors for inspection.
- `POST /api/search` with `{ "query": "person at the door", "limit": 20 }` searches both image and video embeddings.
- Add `"embedding_type": "image"` or `"embedding_type": "video"` to search one embedding space explicitly.
- Event cards show embedding chips such as `image 384d` and `video 384d`; those are the stored vector dimensions.
- If search returns no compatible events after changing embedding backends/models, rebuild saved event vectors with `vision-pipeline reembed-events`.
- `DELETE /api/events/{event_id}` deletes one event and its saved event image.
- `POST /api/events/delete` with `{ "older_than_days": 30 }` deletes events older than the chosen retention window.
- `POST /api/events/delete` with `{ "all": true }` clears event history. The live `*-latest.jpg` camera sample is not an event and is left in place.

## Next Engineering Steps

- Add sqlite-vec or sqlite-vss as the vector index while keeping SQLite as the metadata store.
- Replace averaged frame embeddings with a dedicated video embedding model once latency and VRAM budgets are measured.
- Split VLM description generation into a worker queue so detection stays low-latency.
- Add event windows with pre-roll/post-roll clips, not just key frames.
- Export YOLO to TensorRT on DGX Spark and record latency, accuracy, and VRAM in `configs/`.
- Add auth, camera management, and retention policies before exposing the dashboard beyond localhost.
