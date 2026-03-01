IMAGE_NAME = llmx
IMAGE_TAG = build

docker-build:
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile .

docker-tag: docker-build
	docker tag ${IMAGE_NAME}:${IMAGE_TAG} praktiskt/${IMAGE_NAME}:latest

docker-push: docker-tag
	docker push praktiskt/${IMAGE_NAME}:latest

