IMAGE_NAME = llmx
IMAGE_TAG = build

.PHONY: build test lint format docker-build docker-tag docker-push

build:
	rm -rf .build_tmp dist
	mkdir -p .build_tmp dist
	cp -r llmx .build_tmp/llmx
	find .build_tmp -name '__pycache__' -type d -exec rm -rf {} +
	printf 'import asyncio\n\nfrom llmx.cli import main\n\nasyncio.run(main())\n' > .build_tmp/__main__.py
	uv run python -m zipapp .build_tmp -p "/usr/bin/env python3" -o dist/llm
	chmod +x dist/llm

test:
	uv run python -m unittest discover -s tests -v

lint:
	uv run ruff check llmx tests

format:
	uv run ruff format llmx tests

docker-build:
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile .

docker-tag: docker-build
	docker tag ${IMAGE_NAME}:${IMAGE_TAG} praktiskt/${IMAGE_NAME}:latest

docker-push: docker-tag
	docker push praktiskt/${IMAGE_NAME}:latest

