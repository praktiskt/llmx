import asyncio
import ipaddress
import json
import logging
import os
import socket
from urllib.parse import urlsplit

from .cache import Cache
from .config import Config
from .localfs import contents as _localfs_contents
from .localfs import grep_entries as _localfs_grep_entries
from .localfs import list_entries as _localfs_list_entries
from .localfs import read_entries as _localfs_read_entries
from .search import search as _search
from .transport import AsyncHttp, _iri_to_uri, request_with_retries

try:
    from .mcp import get_mcp_schemas_sync, mcp_call
except ImportError:

    def get_mcp_schemas_sync() -> list:  # type: ignore[no-redef]
        return []

    mcp_call = None  # type: ignore

logger = logging.getLogger(__name__)


def _repair_json(s: str) -> str:
    out = []
    in_string = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\":
            out.append(ch)
            if i + 1 < n:
                out.append(s[i + 1])
            i += 2
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
            else:
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                if j >= n or s[j] in ",]}:":
                    in_string = False
                    out.append(ch)
                else:
                    out.append("\\")
                    out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _unwrap_args(args: dict) -> dict:
    out = {}
    for key, value in args.items():
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed[:1] in "[{":
                try:
                    value = json.loads(trimmed)
                except json.JSONDecodeError:
                    try:
                        value = json.loads(_repair_json(trimmed))
                    except json.JSONDecodeError:
                        pass
        out[key] = value
    return out


def parse_tool_args(args_str: str) -> dict:
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse tool arguments, attempting repair: %r",
            args_str[:200],
        )
        try:
            args = json.loads(_repair_json(args_str))
        except json.JSONDecodeError:
            args = {}
    return _unwrap_args(args)


async def summarize(
    file_ids: list[str],
    local_paths: list[str],
    resolve_errors: list[str],
    directives: list[str],
    max_length: int = 1000,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    concurrency = int(os.environ.get("LLM_SUMMARIZE_CONCURRENCY", "4"))
    semaphore = asyncio.Semaphore(concurrency)

    def _slice(content: str) -> str:
        if offset is not None or limit is not None:
            lines = content.splitlines()
            if offset is not None and offset > 0:
                lines = lines[offset - 1 :]
            if limit is not None and limit > 0:
                lines = lines[:limit]
            return "\n".join(lines)
        return content

    async def summarize_task(task: tuple) -> tuple:
        file_id, directive, content = task
        if content.startswith("Error:") and not directive:
            return (file_id, directive, content)
        summarize_limit = max(50_000, Config.MAX_CONTEXT_CHARS - 20_000)
        if len(content) > summarize_limit:
            return (
                file_id,
                directive,
                f"Error: Content too large ({len(content)} chars, max {summarize_limit // 1000}k). Use summarize with offset/limit to select a smaller section.",
            )
        tokens_for_summary = max(250, max_length * 2)
        messages = [
            {
                "role": "system",
                "content": f"Summarize content concisely. Aim for ~{max_length} characters max.",
            },
            {"role": "user", "content": f"{directive}\n\n---\n\n{content}"},
        ]

        payload = {
            "messages": messages,
            "model": os.environ["LLM_MODEL"],
            "temperature": 0.1,
            "stream": False,
            "max_tokens": tokens_for_summary,
        }

        headers = {
            "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
            "Content-Type": "application/json",
        }

        async with semaphore:
            response = await request_with_retries(
                lambda: AsyncHttp.post(
                    os.environ["LLM_HOST"],
                    headers=headers,
                    json=payload,
                    timeout=Config.LLM_TIMEOUT,
                    reuse=False,
                ),
                attempts=5,
                fail_fast=True,
                on_exception=lambda a, e: logger.warning(
                    f"Summarize attempt {a}/5 failed for {file_id}: {e}"
                ),
                on_retry=lambda a, r: logger.warning(
                    "Summarize API %d (attempt %d/5) for %s: %s, retrying...",
                    r.status_code,
                    a,
                    file_id,
                    r.text[:120],
                ),
            )

        if response is None or response.status_code != 200:
            status = response.status_code if response else "no response"
            body = response.text[:200] if response else "all attempts raised"
            logger.error(
                "Summarize failed for file %s (status %s): %s",
                file_id,
                status,
                body,
            )
            return (
                file_id,
                directive,
                "Error summarizing: all 5 attempts failed. Do you want me to try again?",
            )

        try:
            summary = response.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, ValueError):
            logger.error(
                "Summarize got malformed response for file %s: %.200s",
                file_id,
                response.text,
            )
            return (
                file_id,
                directive,
                "Error summarizing: unexpected API response shape",
            )

        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return (file_id, directive, summary)

    tasks = []
    for error in resolve_errors:
        tasks.append(("sources", "", error))

    for file_id in file_ids:
        content = Cache.get(file_id)
        if content is None:
            tasks.append((file_id, "", f"Error: file {file_id} not found"))
            continue
        content = _slice(content)
        for directive in directives:
            tasks.append((file_id, directive, content))

    if local_paths:
        pairs = await asyncio.to_thread(_localfs_contents, local_paths)
        local_errors = []
        for key, text in pairs:
            if key is None:
                local_errors.append(text)
                continue
            content = _slice(text)
            for directive in directives:
                tasks.append((key, directive, content))
        if local_errors:
            tasks.append(("sources", "", "\n\n".join(local_errors)))

    async def run_task(task: tuple) -> tuple:
        return await summarize_task(task)

    results = await asyncio.gather(*(run_task(task) for task in tasks))
    results_map: dict[str, list[str]] = {}
    for file_id, _, result in results:
        if file_id not in results_map:
            results_map[file_id] = []
        results_map[file_id].append(result)

    final_results = []
    keys_order: list[str] = list(file_ids)
    for key, _, _ in results:
        if key not in keys_order:
            keys_order.append(key)
    for file_id in keys_order:
        if file_id not in results_map:
            continue
        summaries = results_map[file_id]
        file_summaries = [f"{i}. {s}" for i, s in enumerate(summaries, 1)]
        final_results.append(f"File {file_id}:\n" + "\n\n".join(file_summaries))

    return "\n\n---\n\n".join(final_results)


class Tools:
    @staticmethod
    def _is_private_url(url: str) -> bool:
        # Use urlsplit to handle unicode host before DNS lookup (IDNA)
        try:
            host = urlsplit(url).hostname
        except ValueError:
            return True
        if not host:
            return True
        try:
            host = host.encode("idna").decode("ascii")
        except Exception:
            pass
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return True
        for info in infos:
            try:
                ip = ipaddress.ip_address(info[4][0])
            except ValueError:
                return True
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return True
        return False

    SCHEMA = [
        {
            "type": "function",
            "function": {
                "name": "fetch",
                "description": "Fetch content from URLs." + ""
                if not Config.markdown_fetch_proxy()
                else " Response is always Markdown.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of URLs to fetch (at least one)" + ""
                            if not Config.markdown_fetch_proxy()
                            else " as markdown",
                        }
                    },
                    "required": ["urls"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search DuckDuckGo and return results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of search queries (at least one)",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max results per query (default 5)",
                        },
                        "images_only": {
                            "type": "boolean",
                            "description": "Search for images only (default false)",
                        },
                    },
                    "required": ["queries"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read content from one or more sources: cached documents (memory://<id> from fetch/search) and/or local files (paths relative to the current directory, if local access is enabled)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Sources to read, e.g. ['memory://abc123', 'src/main.py', 'docs/*.md']. memory:// ids come from fetch/search results; local paths must be relative to the current directory.",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Start line per source (1-indexed, optional)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max lines per source to return (optional)",
                        },
                    },
                    "required": ["sources"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize",
                "description": "Summarize one or more sources (memory://<id> from fetch/search, and/or local files) according to multiple directives",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Sources to summarize, e.g. ['memory://abc123', 'src/main.py']. memory:// ids come from fetch/search results; local paths must be relative to the current directory.",
                        },
                        "directives": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of summary instructions (at least one)",
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Max characters per summary (default 1000)",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Start line per source for summarization (1-indexed, optional)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max lines per source to summarize (optional)",
                        },
                    },
                    "required": ["sources", "directives"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_file",
                "description": "Search for a pattern in one or more sources: cached documents (memory://<id> from fetch/search) and/or local files (paths relative to the current directory, if local access is enabled)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Sources to search, e.g. ['memory://abc123', 'src/**/*.py']. memory:// ids come from fetch/search results; local paths must be relative to the current directory.",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern",
                        },
                        "is_regex": {
                            "type": "boolean",
                            "description": "Treat pattern as regex (default false, literal search)",
                        },
                        "ignore_case": {
                            "type": "boolean",
                            "description": "Case-insensitive search (default false)",
                        },
                        "context": {
                            "type": "integer",
                            "description": "Lines of context before/after match (default 0)",
                        },
                    },
                    "required": ["sources", "pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List local files under the current directory matching an optional glob pattern and/or path regex (requires local file access to be enabled)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob relative to the current directory (default '**/*', e.g., 'src/**/*.py')",
                        },
                        "regex": {
                            "type": "string",
                            "description": "Optional regex filtering returned paths (matched against the relative path)",
                        },
                        "ignore_case": {
                            "type": "boolean",
                            "description": "Case-insensitive regex (default false)",
                        },
                    },
                    "required": [],
                },
            },
        },
    ]

    @staticmethod
    async def _fetch_single(url: str) -> str:
        def store_and_return(content: str) -> str:
            file_id = Cache.new_id()
            Cache.store(file_id, content)
            return f"Stored as memory://{file_id} ({len(content)} chars). Tools: read_file, grep_file, summarize (memory://{file_id})"

        # Normalize IRI -> URI so http.client ASCII path succeeds
        url = _iri_to_uri(url)

        proxy = Config.markdown_fetch_proxy()
        if proxy:
            fetch_url = f"{proxy.rstrip('/')}/{url}"
        else:
            fetch_url = url

        if not Config.fetch_allow_private() and await asyncio.to_thread(
            Tools._is_private_url, url
        ):
            logger.warning("Fetch blocked private/resolved-private URL: %s", url)
            return "fetch blocked: private, loopback, or unresolvable host"

        headers = {"User-Agent": "Mozilla/5.0"}
        response = await request_with_retries(
            lambda: AsyncHttp.get(
                fetch_url, timeout=Config.FETCH_TIMEOUT, headers=headers
            ),
            attempts=3,
            delay=0.3,
            retriable=lambda r: False,
            on_exception=lambda a, e: logger.warning(
                "Fetch attempt %d/3 failed for %s: %s", a, url, e
            ),
        )

        if response is None:
            logger.error("Fetch failed for %s after 3 attempts", url)
            return "fetch failed after 3 attempts"

        if response.status_code == 200:
            return store_and_return(response.text)
        return f"fetch failed (status {response.status_code})"

    @staticmethod
    async def fetch(urls: list[str]) -> str:
        results = []
        for url in urls:
            result = await Tools._fetch_single(url)
            results.append(f"{url}: {result}")
        return "\n\n---\n\n".join(results)

    @staticmethod
    def _as_list(value) -> list:
        if isinstance(value, str):
            return [value]
        return value

    @staticmethod
    def schema() -> list[dict]:
        """SCHEMA filtered down to Config.allowed_tools(), including MCP tools."""
        try:
            mcp_schemas = get_mcp_schemas_sync()
        except Exception:
            mcp_schemas = []
        all_schemas = Tools.SCHEMA + mcp_schemas
        allowed = Config.allowed_tools()
        if allowed is None:
            return all_schemas
        return [tool for tool in all_schemas if tool["function"]["name"] in allowed]

    @staticmethod
    async def execute(tool_name: str, tool_args: dict) -> str:
        allowed = Config.allowed_tools()
        if allowed is not None and tool_name not in allowed:
            return f"Error: tool '{tool_name}' is not enabled"
        handlers = {
            "fetch": Tools._exec_fetch,
            "search": Tools._exec_search,
            "read_file": Tools._exec_read_file,
            "summarize": Tools._exec_summarize,
            "grep_file": Tools._exec_grep_file,
            "list_files": Tools._exec_list_files,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            # MCP tools are prefixed server__tool
            if (
                "__" in tool_name
                and mcp_call is not None
                and Config.mcp_servers() is not None
            ):
                return await mcp_call(tool_name, tool_args)
            return f"Unknown tool: {tool_name}"
        return await handler(tool_args)

    @staticmethod
    async def _exec_fetch(args: dict) -> str:
        return await Tools.fetch(Tools._as_list(args.get("urls", [])))

    @staticmethod
    async def _exec_search(args: dict) -> str:
        return await _search(
            Tools._as_list(args.get("queries", [])),
            args.get("max_results", 5),
            args.get("images_only", False),
        )

    @staticmethod
    def _local_disabled_error() -> str:
        return "Error: local file access is disabled (LLM_LOCAL_FILES)"

    @staticmethod
    def _resolve_sources(args: dict) -> tuple[list[str], list[str], list[str]]:
        """Split args into (cache_ids, local_paths, errors).

        sources entries are memory://<id> (cache) or relative local paths/globs;
        anything else is rejected with a teaching error.
        """
        if "file_ids" in args or "paths" in args:
            return (
                [],
                [],
                [
                    "Error: file_ids/paths parameters were removed - use "
                    "sources=['memory://<id>', 'relative/path'] instead"
                ],
            )

        raw = args.get("sources")
        if not raw:
            return [], [], []

        cache_ids: list[str] = []
        local_paths: list[str] = []
        errors: list[str] = []
        local_enabled = Config.local_files_enabled()
        for source in Tools._as_list(raw):
            if not isinstance(source, str) or not source.strip():
                errors.append("Error: empty source provided")
                continue
            source = source.strip()
            if source.startswith("memory://"):
                file_id = source[len("memory://") :]
                error = Config.validate_file_id(file_id)
                if error:
                    errors.append(f"Error: {error}")
                else:
                    cache_ids.append(file_id)
            elif "://" in source:
                scheme = source.split("://", 1)[0]
                errors.append(
                    f"Error: unsupported scheme '{scheme}://' - use fetch for URLs; "
                    "sources accepts memory://<id> (from fetch/search) or relative local paths"
                )
            elif not local_enabled:
                errors.append(Tools._local_disabled_error())
            else:
                local_paths.append(source)
        return cache_ids, local_paths, errors

    @staticmethod
    async def _exec_read_file(args: dict) -> str:
        parts: list[str] = []
        file_ids, local_paths, errors = Tools._resolve_sources(args)
        parts.extend(errors)
        if file_ids:
            parts.append(Cache.read(file_ids, args.get("offset"), args.get("limit")))
        if local_paths:
            parts.extend(
                await asyncio.to_thread(
                    _localfs_read_entries,
                    local_paths,
                    args.get("offset"),
                    args.get("limit"),
                )
            )
        if not parts:
            return "Error: provide sources=['memory://<id>', 'relative/path']"
        return "\n\n---\n\n".join(parts)

    @staticmethod
    async def _exec_summarize(args: dict) -> str:
        file_ids, local_paths, errors = Tools._resolve_sources(args)
        if not file_ids and not local_paths and not errors:
            return "Error: provide sources=['memory://<id>', 'relative/path']"
        return await summarize(
            file_ids,
            local_paths,
            errors,
            Tools._as_list(args.get("directives", [])),
            args.get("max_length", 1000),
            args.get("offset"),
            args.get("limit"),
        )

    @staticmethod
    async def _exec_grep_file(args: dict) -> str:
        parts: list[str] = []
        file_ids, local_paths, errors = Tools._resolve_sources(args)
        parts.extend(errors)
        if file_ids:
            parts.append(
                Cache.grep(
                    file_ids,
                    args.get("pattern", ""),
                    args.get("is_regex", False),
                    args.get("ignore_case", False),
                    args.get("context", 0),
                )
            )
        if local_paths:
            parts.extend(
                await asyncio.to_thread(
                    _localfs_grep_entries,
                    local_paths,
                    args.get("pattern", ""),
                    args.get("is_regex", False),
                    args.get("ignore_case", False),
                    args.get("context", 0),
                )
            )
        if not parts:
            return "Error: provide sources=['memory://<id>', 'relative/path']"
        return "\n\n---\n\n".join(parts)

    @staticmethod
    async def _exec_list_files(args: dict) -> str:
        if not Config.local_files_enabled():
            return Tools._local_disabled_error()
        return await asyncio.to_thread(
            _localfs_list_entries,
            args.get("pattern") or "**/*",
            args.get("regex"),
            args.get("ignore_case", False),
        )

    @staticmethod
    async def execute_wrapper(tool_call: dict) -> tuple[str, str]:
        tool_id = tool_call.get("id", "")
        func = tool_call.get("function", {})
        tool_name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        args = parse_tool_args(args_str)
        result = await Tools.execute(tool_name, args)

        if len(result) <= Config.MAX_TOOL_RESULT_CHARS:
            return (tool_id, result)

        num_sources = len(Tools._as_list(args["sources"])) if args.get("sources") else 0
        if tool_name == "read_file":
            current_limit = args.get("limit") or 50
            if current_limit > 10:
                suggested_limit = max(
                    10,
                    int(
                        current_limit * Config.MAX_TOOL_RESULT_CHARS / len(result) * 0.8
                    ),
                )
                args["limit"] = suggested_limit
                result = await Tools.execute(tool_name, args)
                return (
                    tool_id,
                    f"[Truncated from limit={current_limit} to limit={suggested_limit}]\n{result}",
                )

        num_directives = len(args.get("directives", []))
        total_summaries = max(1, num_sources) * max(1, num_directives)
        suggested_max_length = max(500, Config.MAX_TOOL_RESULT_CHARS // total_summaries)

        return (
            tool_id,
            f"Result too large ({len(result)} chars, max {Config.MAX_TOOL_RESULT_CHARS}). "
            f"Suggestions:\n"
            f"1. Reduce max_length (currently {args.get('max_length', 1000)}, try {suggested_max_length})\n"
            f"2. Summarize fewer sources at a time (currently {num_sources})\n"
            f"3. Summarize with different directives in separate calls",
        )
