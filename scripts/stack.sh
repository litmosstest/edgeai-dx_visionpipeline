#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${VISION_STACK_RUN_DIR:-$ROOT_DIR/data/run}"
LOG_DIR="${VISION_STACK_LOG_DIR:-$ROOT_DIR/data/logs}"
API_PID="$RUN_DIR/api.pid"
PUBLISHER_PID="$RUN_DIR/publisher.pid"

mkdir -p "$RUN_DIR" "$LOG_DIR"

api_bin() {
  if [[ -x "$ROOT_DIR/.venv/bin/vision-pipeline" ]]; then
    printf '%s\n' "$ROOT_DIR/.venv/bin/vision-pipeline"
  else
    printf '%s\n' "vision-pipeline"
  fi
}

publisher_script() {
  case "${VISION_PUBLISHER:-webcam}" in
    webcam)
      printf '%s\n' "$ROOT_DIR/scripts/publish_webcam_rtsp.sh"
      ;;
    test)
      printf '%s\n' "$ROOT_DIR/scripts/publish_test_rtsp.sh"
      ;;
    *)
      printf '%s\n' "${VISION_PUBLISHER}"
      ;;
  esac
}

publisher_url() {
  local rtsp_url
  rtsp_url="${RTSP_URL:-$(env_value VISION_RTSP_URL rtsp://localhost:8554/webcam)}"
  printf '%s\n' "$rtsp_url"
}

env_value() {
  local name="$1"
  local default_value="$2"
  local value="${!name-}"
  if [[ -n "$value" ]]; then
    printf '%s\n' "$value"
    return
  fi

  if [[ -f "$ROOT_DIR/.env" ]]; then
    value="$(awk -F= -v key="$name" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$ROOT_DIR/.env")"
    if [[ -n "$value" ]]; then
      printf '%s\n' "$value"
      return
    fi
  fi

  printf '%s\n' "$default_value"
}

api_url() {
  local configured_url port
  configured_url="$(env_value VISION_API_URL "")"
  if [[ -n "$configured_url" ]]; then
    printf '%s\n' "$configured_url"
    return
  fi

  port="$(env_value VISION_PORT 8081)"
  printf 'http://127.0.0.1:%s\n' "$port"
}

find_publisher_pids() {
  local rtsp_url="$1"
  pgrep -af ffmpeg 2>/dev/null | grep -F -- "$rtsp_url" | awk '{print $1}' || true
}

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

stop_pid() {
  local name="$1"
  local pid_file="$2"
  if ! [[ -f "$pid_file" ]]; then
    echo "$name is not running."
    return
  fi

  local pid
  pid="$(cat "$pid_file")"
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    echo "$name was not running."
    return
  fi

  kill "$pid" 2>/dev/null || true
  for _ in {1..30}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
      echo "Stopped $name."
      return
    fi
    sleep 0.1
  done

  kill -9 "$pid" 2>/dev/null || true
  rm -f "$pid_file"
  echo "Stopped $name."
}

start_mediamtx() {
  (cd "$ROOT_DIR" && docker compose up -d mediamtx)
}

start_publisher() {
  if is_running "$PUBLISHER_PID"; then
    echo "FFmpeg publisher is already running with PID $(cat "$PUBLISHER_PID")."
    return
  fi

  local script rtsp_url existing_pids
  script="$(publisher_script)"
  rtsp_url="$(publisher_url)"
  if ! [[ -x "$script" ]]; then
    echo "Publisher script is not executable: $script" >&2
    exit 1
  fi

  existing_pids="$(find_publisher_pids "$rtsp_url" | paste -sd, -)"
  if [[ -n "$existing_pids" ]]; then
    echo "FFmpeg publisher is already running outside this stack with PID(s) $existing_pids."
    return
  fi

  (
    cd "$ROOT_DIR"
    nohup env RTSP_URL="$rtsp_url" "$script" >"$LOG_DIR/publisher.log" 2>&1 &
    echo $! >"$PUBLISHER_PID"
  )
  echo "Started FFmpeg publisher with PID $(cat "$PUBLISHER_PID"). Log: $LOG_DIR/publisher.log"
}

wait_for_publisher() {
  if ! [[ -f "$PUBLISHER_PID" ]]; then
    return 0
  fi

  for _ in {1..20}; do
    if is_running "$PUBLISHER_PID"; then
      sleep 0.25
      continue
    fi

    rm -f "$PUBLISHER_PID"
    echo "FFmpeg publisher exited during startup. Recent publisher log:" >&2
    tail -n 40 "$LOG_DIR/publisher.log" >&2 || true
    return 1
  done
}

start_api() {
  if is_running "$API_PID"; then
    echo "API is already running with PID $(cat "$API_PID")."
    return
  fi

  local bin
  bin="$(api_bin)"
  (
    cd "$ROOT_DIR"
    nohup "$bin" api >"$LOG_DIR/api.log" 2>&1 &
    echo $! >"$API_PID"
  )
  echo "Started API with PID $(cat "$API_PID"). Log: $LOG_DIR/api.log"
}

wait_for_api() {
  local url
  url="$(api_url)"
  for _ in {1..60}; do
    if curl -fsS "$url/api/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done

  echo "API did not become healthy at $url. Check $LOG_DIR/api.log" >&2
  return 1
}

wait_for_rtsp() {
  local rtsp_url
  rtsp_url="$(publisher_url)"
  if ! command -v ffprobe >/dev/null 2>&1; then
    echo "ffprobe is not installed; skipping RTSP readiness check for $rtsp_url."
    return 0
  fi

  for _ in {1..30}; do
    if ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
      -show_entries stream=codec_name -of csv=p=0 "$rtsp_url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done

  echo "RTSP stream did not become readable at $rtsp_url. Recent publisher log:" >&2
  tail -n 40 "$LOG_DIR/publisher.log" >&2 || true
  return 1
}

start_pipeline() {
  case "${VISION_STACK_START_PIPELINE:-1}" in
    0|false|no)
      echo "Pipeline auto-start skipped because VISION_STACK_START_PIPELINE=${VISION_STACK_START_PIPELINE}."
      return
      ;;
  esac

  local url
  url="$(api_url)"
  wait_for_api
  wait_for_rtsp
  nohup curl -fsS -X POST "$url/api/pipeline/start" >"$LOG_DIR/pipeline-start.log" 2>&1 &
  echo "Pipeline start requested. Log: $LOG_DIR/pipeline-start.log"
}

start_all() {
  start_mediamtx
  start_publisher
  wait_for_publisher
  start_api
  start_pipeline
  echo "Stack start requested. Open the dashboard on VISION_PORT, default http://localhost:8081."
}

stop_all() {
  stop_pid "API" "$API_PID"
  stop_pid "FFmpeg publisher" "$PUBLISHER_PID"
  (cd "$ROOT_DIR" && docker compose down)
}

status_one() {
  local name="$1"
  local pid_file="$2"
  if is_running "$pid_file"; then
    echo "$name: running with PID $(cat "$pid_file")"
  else
    rm -f "$pid_file"
    echo "$name: stopped"
  fi
}

status_all() {
  local rtsp_url existing_pids
  status_one "API" "$API_PID"
  if is_running "$PUBLISHER_PID"; then
    echo "FFmpeg publisher: running with PID $(cat "$PUBLISHER_PID")"
  else
    rm -f "$PUBLISHER_PID"
    rtsp_url="$(publisher_url)"
    existing_pids="$(find_publisher_pids "$rtsp_url" | paste -sd, -)"
    if [[ -n "$existing_pids" ]]; then
      echo "FFmpeg publisher: running outside this stack with PID(s) $existing_pids"
    else
      echo "FFmpeg publisher: stopped"
    fi
  fi
  (cd "$ROOT_DIR" && docker compose ps mediamtx)
}

logs_all() {
  touch "$LOG_DIR/api.log" "$LOG_DIR/publisher.log"
  tail -n "${LINES:-80}" -f "$LOG_DIR/api.log" "$LOG_DIR/publisher.log"
}

case "${1:-}" in
  start|up)
    start_all
    ;;
  stop|down)
    stop_all
    ;;
  restart)
    stop_all
    start_all
    ;;
  status)
    status_all
    ;;
  logs)
    logs_all
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}" >&2
    exit 2
    ;;
esac