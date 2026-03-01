FROM astral/uv:python3.14-bookworm-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev
COPY . .
CMD ["uv", "run", "python", "-m", "llmx.server"]
ENTRYPOINT ["uv", "run", "python", "-m", "llmx.server"]
