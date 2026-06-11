.PHONY: install install-models api mediamtx publish test lint

install:
	python -m pip install -e .

install-models:
	python -m pip install -e '.[models,dev]'

api:
	vision-pipeline api

mediamtx:
	docker compose up mediamtx

publish:
	./scripts/publish_webcam_rtsp.sh

test:
	python -m pytest

lint:
	python -m ruff check src tests
