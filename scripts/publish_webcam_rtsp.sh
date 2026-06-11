#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-/dev/video0}"
RTSP_URL="${RTSP_URL:-rtsp://localhost:8554/webcam}"
SIZE="${SIZE:-1280x720}"
FPS="${FPS:-30}"

if [[ ! -e "$DEVICE" ]]; then
  echo "Camera device not found: $DEVICE" >&2
  echo "Available video devices:" >&2
  ls -l /dev/video* 2>/dev/null >&2 || true
  exit 1
fi

if [[ ! -r "$DEVICE" ]]; then
  echo "Cannot read camera device: $DEVICE" >&2
  echo "This usually means your user is not in the Linux 'video' group." >&2
  echo "Permanent fix: sudo usermod -aG video \"$USER\", then log out and back in." >&2
  echo "Temporary fix: sudo setfacl -m u:$USER:rw \"$DEVICE\"" >&2
  echo "Current device permissions:" >&2
  ls -l "$DEVICE" >&2
  exit 1
fi

exec ffmpeg \
  -f v4l2 \
  -framerate "$FPS" \
  -video_size "$SIZE" \
  -i "$DEVICE" \
  -an \
  -c:v libx264 \
  -preset veryfast \
  -tune zerolatency \
  -pix_fmt yuv420p \
  -f rtsp \
  -rtsp_transport tcp \
  "$RTSP_URL"
