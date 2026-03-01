FROM astral/uv:python3.14-bookworm-slim
WORKDIR /app

RUN useradd -m -u 1001 user

COPY pyproject.toml uv.lock ./
RUN chown -R user:user /app

USER user
RUN uv sync --locked --no-install-project

COPY . .
CMD ["uv", "run", "python", "-m", "llmx.server"]
ENTRYPOINT ["uv", "run", "python", "-m", "llmx.server"]
