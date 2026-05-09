.PHONY: help install install-dev \
        build-image rebuild-image \
        lint format test check

# Workspace image tag. Override with `make build-image IMAGE_TAG=foo:bar`.
IMAGE_TAG ?= rlm-workspace:0.1.0

help:
	@echo "RLM Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install base dependencies with uv"
	@echo "  make install-dev    - Install dev dependencies with uv"
	@echo ""
	@echo "Workspace image:"
	@echo "  make build-image    - Build the workspace Docker image (tag: $(IMAGE_TAG))"
	@echo "  make rebuild-image  - Rebuild without cache"
	@echo ""
	@echo "Development:"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Run ruff formatter"
	@echo "  make test           - Run tests"
	@echo "  make check          - Run lint + format + tests"

install:
	uv sync

install-dev:
	uv sync --group dev --group test

lint: install-dev
	uv run ruff check .

format: install-dev
	uv run ruff format .

test: install-dev
	uv run pytest

build-image:
	docker build -t $(IMAGE_TAG) -f docker/workspace.Dockerfile docker

rebuild-image:
	docker build --no-cache -t $(IMAGE_TAG) -f docker/workspace.Dockerfile docker

check: lint format test
