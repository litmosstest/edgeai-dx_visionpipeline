const eventsEl = document.querySelector("#events");
const template = document.querySelector("#eventTemplate");
const statusLine = document.querySelector("#statusLine");
const eventCount = document.querySelector("#eventCount");
const frameCount = document.querySelector("#frameCount");
const cameraId = document.querySelector("#cameraId");
const rtspUrl = document.querySelector("#rtspUrl");
const queryInput = document.querySelector("#queryInput");
const embeddingType = document.querySelector("#embeddingType");
const searchButton = document.querySelector("#searchButton");
const deleteRange = document.querySelector("#deleteRange");
const sampleImage = document.querySelector("#sampleImage");
const sampleOverlay = document.querySelector("#sampleOverlay");
const sampleMeta = document.querySelector("#sampleMeta");
const svgNamespace = "http://www.w3.org/2000/svg";
const completionItems = [
  ["image_embedding", "Image"],
  ["video_embedding", "Video"],
  ["vlm_description", "VLM"],
];
let activeSearch = false;

document.querySelector("#startButton").addEventListener("click", async () => {
  await fetch("/api/pipeline/start", { method: "POST" });
  await refresh();
});

document.querySelector("#stopButton").addEventListener("click", async () => {
  await fetch("/api/pipeline/stop", { method: "POST" });
  await refresh();
});

searchButton.addEventListener("click", search);
document.querySelector("#clearButton").addEventListener("click", async () => {
  queryInput.value = "";
  activeSearch = false;
  await refresh();
});
document.querySelector("#deleteRangeButton").addEventListener("click", deleteEventRange);

queryInput.addEventListener("keydown", async (event) => {
  if (event.key === "Enter") {
    await search();
  }
});

async function refresh() {
  const [healthResponse, eventsResponse] = await Promise.all([
    fetch("/api/health"),
    fetch("/api/events?limit=60"),
  ]);
  const health = await healthResponse.json();
  const eventPayload = await eventsResponse.json();
  renderStatus(health);
  renderSample(health.pipeline);
  renderEvents(eventPayload.events);
}

async function search() {
  const query = queryInput.value.trim();
  if (!query) {
    activeSearch = false;
    await refresh();
    return;
  }
  activeSearch = true;
  setSearchLoading(true);
  renderEvents([], false, `Searching for "${query}"...`);
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit: 40, embedding_type: embeddingType.value }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `Search failed with HTTP ${response.status}.`);
    }
    if (!Array.isArray(payload.events)) {
      throw new Error("Search response did not include an events list.");
    }
    renderEvents(payload.events, true, searchMessage(payload, query));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Search failed.";
    renderEvents([], false, message);
  } finally {
    setSearchLoading(false);
  }
}

function setSearchLoading(isLoading) {
  searchButton.disabled = isLoading;
  searchButton.textContent = isLoading ? "Searching" : "Search";
}

async function deleteEventRange() {
  const selectedRange = deleteRange.value;
  const confirmed = window.confirm(
    selectedRange === "all"
      ? "Delete all stored events?"
      : `Delete events older than ${selectedRange} days?`,
  );
  if (!confirmed) {
    return;
  }
  const body = selectedRange === "all"
    ? { all: true }
    : { older_than_days: Number(selectedRange) };
  const response = await fetch("/api/events/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    return;
  }
  await refresh();
}

async function deleteEvent(eventId) {
  if (!window.confirm("Delete this event?")) {
    return;
  }
  const response = await fetch(`/api/events/${encodeURIComponent(eventId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    return;
  }
  await refresh();
}

function renderStatus(health) {
  const pipeline = health.pipeline;
  statusLine.textContent = pipeline.running
    ? `Running on ${pipeline.rtsp_url}`
    : pipeline.last_error || "Stopped";
  eventCount.textContent = String(health.events);
  frameCount.textContent = String(pipeline.frames_seen);
  cameraId.textContent = pipeline.camera_id;
  rtspUrl.textContent = pipeline.rtsp_url;
}

function renderSample(pipeline) {
  if (!pipeline.latest_frame_url) {
    sampleImage.removeAttribute("src");
    sampleOverlay.replaceChildren();
    sampleMeta.textContent = "No frame";
    return;
  }

  sampleImage.src = `${pipeline.latest_frame_url}?frame=${pipeline.frames_seen}`;
  sampleMeta.textContent = `${pipeline.latest_detections.length} boxes`;
  drawBoxes(
    sampleOverlay,
    pipeline.latest_detections,
    pipeline.latest_frame_width,
    pipeline.latest_frame_height,
  );
}

function renderEvents(events, showScore = false, emptyMessage = "No events captured yet.") {
  eventsEl.replaceChildren();
  if (activeSearch) {
    const notice = document.createElement("div");
    notice.className = "search-notice";
    notice.textContent = emptyMessage;
    eventsEl.append(notice);
  }
  if (!events.length) {
    if (!activeSearch) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = emptyMessage;
      eventsEl.append(empty);
    }
    return;
  }

  for (const event of events) {
    const node = template.content.cloneNode(true);
    const image = node.querySelector(".event-image");
    const overlay = node.querySelector(".box-overlay");
    image.addEventListener("load", () => {
      drawBoxes(overlay, event.detections, image.naturalWidth, image.naturalHeight);
    });
    image.src = event.image_url;
    node.querySelector("h2").textContent = event.label_summary;
    node.querySelector(".confidence").textContent = `${Math.round(event.confidence * 100)}%`;
    node.querySelector(".delete-event-button").addEventListener("click", async () => {
      await deleteEvent(event.id);
    });
    renderCompletionStatus(node.querySelector(".completion-status"), event.processing_status || {});
    node.querySelector(".description").textContent = event.description;
    const timestamp = new Date(event.timestamp).toLocaleString();
    node.querySelector(".meta").textContent = showScore && typeof event.score === "number"
      ? `${timestamp} · score ${event.score.toFixed(3)}`
      : timestamp;

    const chips = node.querySelector(".chips");
    for (const detection of event.detections) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = `${detection.label} ${Math.round(detection.confidence * 100)}%`;
      chips.append(chip);
    }
    for (const [kind, embedding] of Object.entries(event.embeddings || {})) {
      if (!embedding.present) {
        continue;
      }
      const chip = document.createElement("span");
      chip.className = "chip embedding-chip";
      chip.textContent = `${kind} ${embedding.dimensions}d`;
      chips.append(chip);
    }
    eventsEl.append(node);
  }
}

function renderCompletionStatus(container, status) {
  container.replaceChildren();
  for (const [key, label] of completionItems) {
    const complete = Boolean(status[key]);
    const item = document.createElement("span");
    item.className = `status-tick ${complete ? "is-complete" : "is-pending"}`;
    item.textContent = `${complete ? "✓ " : ""}${label}`;
    item.title = `${label}: ${complete ? "created" : "pending"}`;
    item.setAttribute("aria-label", item.title);
    container.append(item);
  }
}

function drawBoxes(svg, detections, width, height) {
  svg.replaceChildren();
  if (!width || !height || !detections.length) {
    svg.removeAttribute("viewBox");
    return;
  }

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "none");

  for (const detection of detections) {
    const [rawX1, rawY1, rawX2, rawY2] = detection.bbox_xyxy.map(Number);
    const x1 = clamp(rawX1, 0, width);
    const y1 = clamp(rawY1, 0, height);
    const x2 = clamp(rawX2, x1, width);
    const y2 = clamp(rawY2, y1, height);
    const boxWidth = Math.max(1, x2 - x1);
    const boxHeight = Math.max(1, y2 - y1);

    const rect = document.createElementNS(svgNamespace, "rect");
    rect.classList.add("bbox");
    rect.setAttribute("x", String(x1));
    rect.setAttribute("y", String(y1));
    rect.setAttribute("width", String(boxWidth));
    rect.setAttribute("height", String(boxHeight));
    svg.append(rect);

    const label = document.createElementNS(svgNamespace, "text");
    label.classList.add("bbox-label");
    label.setAttribute("x", String(x1 + 8));
    label.setAttribute("y", String(Math.max(24, y1 + 24)));
    label.textContent = `${detection.label} ${Math.round(detection.confidence * 100)}%`;
    svg.append(label);
  }
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function searchMessage(payload, query) {
  const count = payload.events?.length ?? 0;
  if (count > 0) {
    return count === 1
      ? `Showing 1 result for "${query}".`
      : `Showing ${count} results for "${query}".`;
  }
  if (payload.skipped_vectors) {
    return `No compatible ${payload.query_dimensions}d embeddings found. Re-embed older events with the current model.`;
  }
  return `No matching events found for "${query}".`;
}

refresh();
setInterval(() => {
  if (!activeSearch) {
    refresh();
  }
}, 5000);
