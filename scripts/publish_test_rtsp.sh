#!/usr/bin/env bash
set -euo pipefail

RTSP_URL="${RTSP_URL:-rtsp://localhost:8554/webcam}"
SIZE="${SIZE:-1280x720}"
FPS="${FPS:-30}"

exec ffmpeg \
  -re \
  -f lavfi \
  -i "testsrc=size=$SIZE:rate=$FPS" \
  -an \
  -c:v libx264 \
  -preset veryfast \
  -tune zerolatency \
  -pix_fmt yuv420p \
  -f rtsp \
  -rtsp_transport tcp \
  "$RTSP_URL"
