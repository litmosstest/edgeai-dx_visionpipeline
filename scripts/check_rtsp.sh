#!/usr/bin/env bash
set -euo pipefail

RTSP_URL="${RTSP_URL:-rtsp://localhost:8554/webcam}"
DEVICE="${DEVICE:-/dev/video0}"

echo "Checking RTSP URL: $RTSP_URL"

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe is required for this check. Install ffmpeg first." >&2
  exit 1
fi

if ! ss -ltn 'sport = :8554' 2>/dev/null | grep -q ':8554'; then
  echo "No RTSP server is listening on port 8554." >&2
  echo "Start MediaMTX with: docker compose up -d mediamtx" >&2
  exit 1
fi

if ffprobe \
  -rtsp_transport tcp \
  -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height \
  -of default=nw=1 \
  "$RTSP_URL"; then
  echo "RTSP stream is readable."
else
  echo "RTSP server is running, but this stream path is not readable: $RTSP_URL" >&2
  echo "If ffprobe showed 404, publish the webcam first:" >&2
  echo "  ./scripts/publish_webcam_rtsp.sh" >&2
  if [[ -e "$DEVICE" && ! -r "$DEVICE" ]]; then
    echo "" >&2
    echo "Camera device exists but is not readable: $DEVICE" >&2
    echo "Fix: sudo usermod -aG video \"$USER\", then log out and back in." >&2
    echo "Temporary fix: sudo setfacl -m u:$USER:rw \"$DEVICE\"" >&2
    ls -l "$DEVICE" >&2
  fi
  exit 1
fi
