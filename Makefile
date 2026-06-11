.PHONY: install install-models up down restart status logs api mediamtx publish test lint

install:
	python -m pip install -e .

install-models:
	python -m pip install -e '.[models,dev]'

up:
	bash scripts/stack.sh start

down:
	bash scripts/stack.sh stop

restart:
	bash scripts/stack.sh restart

status:
	bash scripts/stack.sh status

logs:
	bash scripts/stack.sh logs

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
