FROM python:3.14-slim

WORKDIR /app
COPY . /app

RUN pip install mistune requests

ENTRYPOINT ["python", "-m", "src.server"]
