# llmx

LLM with tool use - dependency-free CLI (`dist/llm` zipapp, stdlib only) + FastAPI web UI (SSE streaming).

Tools: `fetch`/`search`->`memory://<id>`, `read`/`grep`/`summarize` over `sources=[memory://..., "path/*.py"]`, `list_files`, MCP `server__tool`. Resilience: 5x retry, keep-alive pool, context truncation, harness filter.

## Quickstart

```sh
export LLM_HOST="https://gateway/v1/chat/completions" LLM_API_KEY="..." LLM_MODEL="..." # required
llm "Explain quines"
echo "text" | llm "summarize this"          # thinking->stderr, answer->stdout
uv run uvicorn llmx.server:app --port 8080  # sessions 1h (LLM_SESSION_TTL)
make build && cp dist/llm ~/.local/bin/llm; make test
```

## Env

`LLM_TOOLS` (all; `fetch,search,read,grep,summarize,list_files` or `server__tool` - `read_file`/`grep_file` aliases kept for compat), `LLM_TEMPERATURE=0.1`, `LLM_STREAM=True`, `LLM_RESPONSE_FORMAT` (set disables stream), `LLM_SHOW_THINKING=True`, `LLM_MAX_CONTEXT_CHARS=200000`, `LLM_SYSTEM_PROMPT`, `NO_COLOR`/`LLM_DISABLE_COLOR_OUTPUT=False`, `LOG_LEVEL=WARNING` (CLI) / `INFO` (server).

`LLM_MARKDOWN_FETCH_PROXY`, `LLM_MARKDOWN_SEARCH_PROXY`, `LLM_MARKDOWN_IMAGE_SEARCH_PROXY`, `LLM_FETCH_ALLOW_PRIVATE=False`, `LLM_SUMMARIZE_CONCURRENCY=4`.

`LLM_BIND_ADDRESS=0.0.0.0`, `LLM_SERVER_PORT=8080`, `LLM_SESSION_TTL=3600`, `LLM_MAX_SESSIONS=200`.

`LLM_LOCAL_FILES=False` (+ `LLM_LOCAL_MAX_FILES=100`, `LLM_LOCAL_MAX_FILE_BYTES=2000000`) - enables `sources` local paths/globs (relative, no `..`/absolute/symlink escape).

`LLM_INTERACTIVE=False` - tty REPL (`> `, Ctrl-D/`quit`/`exit`).

`LLM_MCP_SERVERS` (unset) - `{"fs":{"command":"npx","args":[...]},"remote":{"url":"http://host/mcp","headers":{}}}` (stdio: `command`+`args`/`env`/`cwd`; http: `url`+`headers`) -> `fs__tool`, `LLM_TOOLS`-filterable, large->`memory://` (`[a-zA-Z0-9_-]{1,32}`). `LLM_MCP_TIMEOUT=30`.
