# llmx

LLM assistant with tool use: dependency-free CLI + small web chat server.

- **CLI** (`dist/llm`): standalone zipapp, pure stdlib, runs on any Linux with `python3`
- **Server** (`llmx/server.py`): FastAPI web UI, SSE-streamed thinking + answers
- **Tools**: fetch URLs (cached as 6-char file_ids), web search, read/grep cached docs, summarize via sub-requests
- **Resilience**: 5x retry on all >=400/network errors, keep-alive connection pool, context-window truncation, harness-artifact filtering

## CLI

```sh
export LLM_HOST="https://gateway/v1/chat/completions"  # required
export LLM_API_KEY="..."                               # required
export LLM_MODEL="..."                                 # required

llm "Explain quines"
echo "text" | llm "summarize this"
```

Thinking -> stderr, answer -> stdout.

## Server

```sh
uv run uvicorn llmx.server:app --port 8080
```

Sessions expire after 1h (`LLM_SESSION_TTL`).

## Build / test

```sh
make build   # -> dist/llm
cp dist/llm ~/.local/bin/llm
make test    # stdlib unittest
```

## Env (optional)

Core:

- `LLM_TOOLS` — unset = all tools, empty = none, else comma-separated allowlist: `fetch,search,read_file,grep_file,summarize,list_files`
- `LLM_TEMPERATURE` (0.1), `LLM_STREAM` (True), `LLM_RESPONSE_FORMAT` (unset; setting it disables streaming)
- `LLM_SHOW_THINKING` (True), `LLM_MAX_CONTEXT_CHARS` (200000)
- `LLM_SYSTEM_PROMPT` — overrides the built-in tool-usage prompt
- `NO_COLOR` / `LLM_DISABLE_COLOR_OUTPUT` (False) — disable ANSI colors
- `LOG_LEVEL` (WARNING for CLI, INFO for server)

Network / tools:

- `LLM_MARKDOWN_FETCH_PROXY`, `LLM_MARKDOWN_SEARCH_PROXY`, `LLM_MARKDOWN_IMAGE_SEARCH_PROXY`
- `LLM_FETCH_ALLOW_PRIVATE` (False) — allow fetch to private/loopback hosts
- `LLM_SUMMARIZE_CONCURRENCY` (4)

Server:

- `LLM_BIND_ADDRESS` (0.0.0.0), `LLM_SERVER_PORT` (8080)
- `LLM_SESSION_TTL` (3600), `LLM_MAX_SESSIONS` (200)

Local files:

- `LLM_LOCAL_FILES` (False) — when true, `read_file`, `grep_file`, and `summarize`
  accept `paths=[...]` relative to the current directory (globs allowed) and
  `list_files` lists local files by glob/regex; paths cannot be absolute,
  contain `..`, or resolve outside the cwd, symlinks included
- `LLM_LOCAL_MAX_FILES` (100), `LLM_LOCAL_MAX_FILE_BYTES` (2000000)

Interactive:

- `LLM_INTERACTIVE` (False) — when true and stdin is a tty, keeps reading new user
  lines after each answer as one continuing conversation; Ctrl-D, `quit`, or
  `exit` ends the session
