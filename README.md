# llmx

LLM assistant with tool use — a dependency-free CLI and a small web chat server.

## What's inside

| Path | Purpose |
|---|---|
| `dist/llm` | Standalone executable (zipapp). Pure stdlib — runs on any Linux with `python3`. |
| `llmx/cli.py` | CLI entrypoint |
| `llmx/client.py` | Chat loop, streaming reader, context-window truncation |
| `llmx/transport.py` | HTTP client (keep-alive pool), retry helper |
| `llmx/tools.py` | Tool schemas + execution (fetch, search, read_file, grep_file, summarize) |
| `llmx/search.py` | DuckDuckGo / markdown-proxy search |
| `llmx/cache.py` | Bounded LRU cache for fetched documents (file_ids) |
| `llmx/server.py` | Web UI backend (FastAPI, SSE streaming) |

## Usage

### CLI

```sh
export LLM_HOST="https://your-gateway/v1/chat/completions"
export LLM_API_KEY="..."
export LLM_MODEL="..."

llm "Explain quines"
echo "some text" | llm "summarize this"
```

The model can call tools: it fetches URLs (stored under 6-char file_ids), searches
the web, greps/reads cached content, and summarizes files via sub-requests.
Reasoning ("thinking") streams to stderr, answers to stdout.

### Web server

```sh
uv run uvicorn llmx.server:app --port 8080
```

SSE-streamed responses (thinking + answer deltas live), same tool set,
sessions expire after 1h (`LLM_SESSION_TTL`).

## Build & install

```sh
make build          # -> dist/llm (zipapp)
cp dist/llm ~/.local/bin/llm
make test           # unit tests (stdlib unittest)
```

## Configuration (env)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_HOST` | – | Chat-completions endpoint (required) |
| `LLM_API_KEY` | – | Bearer token (required) |
| `LLM_MODEL` | – | Model name (required) |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `LLM_STREAM` | `True` | Stream tokens (CLI) |
| `LLM_SHOW_THINKING` | `True` | Print reasoning to stderr |
| `LLM_MAX_CONTEXT_CHARS` | `200000` | History truncation budget |
| `LLM_MARKDOWN_FETCH_PROXY` | – | Fetch URLs through this markdown proxy |
| `LLM_MARKDOWN_SEARCH_PROXY` | – | Search through this proxy |
| `LLM_MARKDOWN_IMAGE_SEARCH_PROXY` | – | Image search proxy |
| `LLM_FETCH_ALLOW_PRIVATE` | `False` | Allow fetching loopback/private hosts |
| `LLM_SUMMARIZE_CONCURRENCY` | `4` | Parallel summarize requests |
| `LLM_SESSION_TTL` | `3600` | Web session lifetime (s) |
| `LLM_MAX_SESSIONS` | `200` | Web session cap |

Retries: API calls retry up to 5× on any 4xx/5xx or network error, aborting early
when the identical error repeats (deterministic failure).
