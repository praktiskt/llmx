FROM python:3.14-slim

WORKDIR /app
COPY . /app

RUN pip install fastapi httpx uvicorn[standard] mistune requests

ENTRYPOINT ["python", "-m", "src.server"]
