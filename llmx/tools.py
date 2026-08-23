import asyncio
import ipaddress
import json
import logging
import os
import socket
from urllib.parse import urlparse

from .cache import Cache
from .config import Config
from .search import search as _search
from .transport import AsyncHttp, request_with_retries

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
    directives: list[str],
    max_length: int = 1000,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    concurrency = int(os.environ.get("LLM_SUMMARIZE_CONCURRENCY", "4"))
    semaphore = asyncio.Semaphore(concurrency)

    async def summarize_task(task: tuple) -> tuple:
        file_id, directive, content = task
        if len(content) > 180_000:
            return (
                file_id,
                directive,
                f"Error: Content too large ({len(content)} chars, max 180k). Use summarize with offset/limit to select a smaller section.",
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
    for file_id in file_ids:
        validation_error = Config.validate_file_id(file_id)
        if validation_error:
            tasks.append((file_id, "", f"Error: {validation_error}"))
            continue

        content = Cache.get(file_id)
        if content is None:
            tasks.append((file_id, "", f"Error: file {file_id} not found"))
            continue

        if offset is not None or limit is not None:
            lines = content.splitlines()
            if offset is not None and offset > 0:
                lines = lines[offset - 1 :]
            if limit is not None and limit > 0:
                lines = lines[:limit]
            content = "\n".join(lines)

        for directive in directives:
            tasks.append((file_id, directive, content))

    async def run_task(task: tuple) -> tuple:
        return await summarize_task(task)

    results = await asyncio.gather(*(run_task(task) for task in tasks))
    results_map: dict[str, list[str]] = {}
    for file_id, _, result in results:
        if file_id not in results_map:
            results_map[file_id] = []
        results_map[file_id].append(result)

    final_results = []
    for file_id in file_ids:
        if file_id not in results_map:
            continue
        summaries = results_map[file_id]
        file_summaries = [f"{i}. {s}" for i, s in enumerate(summaries, 1)]
        final_results.append(f"File {file_id}:\n" + "\n\n".join(file_summaries))

    return "\n\n---\n\n".join(final_results)


class Tools:
    @staticmethod
    def _is_private_url(url: str) -> bool:
        host = urlparse(url).hostname
        if not host:
            return True
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
                "description": "Read content from one or more cached files (file_id from fetch)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 6-character lowercase alphanumeric IDs returned by fetch (e.g., ['abc123', 'xyz789']). Do NOT invent file_ids.",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Start line per file (1-indexed, optional)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max lines per file to return (optional)",
                        },
                    },
                    "required": ["file_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize",
                "description": "Summarize one or more cached files according to multiple directives",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 6-character lowercase alphanumeric IDs returned by fetch (e.g., ['abc123', 'xyz789']). Do NOT invent file_ids.",
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
                            "description": "Start line per file for summarization (1-indexed, optional)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max lines per file to summarize (optional)",
                        },
                    },
                    "required": ["file_ids", "directives"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_file",
                "description": "Search for a pattern in one or more cached files (file_id from fetch)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 6-character lowercase alphanumeric IDs returned by fetch (e.g., ['abc123', 'xyz789']). Do NOT invent file_ids.",
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
                    "required": ["file_ids", "pattern"],
                },
            },
        },
    ]

    @staticmethod
    async def _fetch_single(url: str) -> str:
        def store_and_return(content: str) -> str:
            file_id = Cache.new_id()
            Cache.store(file_id, content)
            return f'Stored as {file_id} ({len(content)} chars). Tools: read_file, grep_file, summarize (file_id="{file_id}")'

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
    async def execute(tool_name: str, tool_args: dict) -> str:
        handlers = {
            "fetch": Tools._exec_fetch,
            "search": Tools._exec_search,
            "read_file": Tools._exec_read_file,
            "summarize": Tools._exec_summarize,
            "grep_file": Tools._exec_grep_file,
        }
        handler = handlers.get(tool_name)
        if handler is None:
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
    async def _exec_read_file(args: dict) -> str:
        return Cache.read(
            Tools._as_list(args.get("file_ids", [])),
            args.get("offset"),
            args.get("limit"),
        )

    @staticmethod
    async def _exec_summarize(args: dict) -> str:
        return await summarize(
            Tools._as_list(args.get("file_ids", [])),
            Tools._as_list(args.get("directives", [])),
            args.get("max_length", 1000),
            args.get("offset"),
            args.get("limit"),
        )

    @staticmethod
    async def _exec_grep_file(args: dict) -> str:
        return Cache.grep(
            Tools._as_list(args.get("file_ids", [])),
            args.get("pattern", ""),
            args.get("is_regex", False),
            args.get("ignore_case", False),
            args.get("context", 0),
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

        file_ids = args.get("file_ids", [])
        if isinstance(file_ids, str):
            file_ids = [file_ids]

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

        num_files = len(file_ids)
        num_directives = len(args.get("directives", []))
        total_summaries = max(1, num_files) * max(1, num_directives)
        suggested_max_length = max(500, Config.MAX_TOOL_RESULT_CHARS // total_summaries)

        return (
            tool_id,
            f"Result too large ({len(result)} chars, max {Config.MAX_TOOL_RESULT_CHARS}). "
            f"Suggestions:\n"
            f"1. Reduce max_length (currently {args.get('max_length', 1000)}, try {suggested_max_length})\n"
            f"2. Summarize fewer files at a time (currently {num_files})\n"
            f"3. Summarize with different directives in separate calls",
        )
