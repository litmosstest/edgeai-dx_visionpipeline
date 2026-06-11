FROM python:3.11-slim

ARG INSTALL_EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN if [ -n "$INSTALL_EXTRAS" ]; then pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"; else pip install --no-cache-dir .; fi

CMD ["vision-pipeline", "api"]
