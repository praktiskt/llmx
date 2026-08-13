#!/usr/bin/python3
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import string
import sys
from datetime import date
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import requests

logger = logging.getLogger(__name__)


class AsyncHttp:
    @staticmethod
    async def get(url: str, **kwargs) -> requests.Response:
        return await asyncio.to_thread(requests.get, url, **kwargs)

    @staticmethod
    async def post(url: str, **kwargs) -> requests.Response:
        return await asyncio.to_thread(requests.post, url, **kwargs)

    @staticmethod
    async def head(url: str) -> requests.Response:
        return await asyncio.to_thread(
            requests.head, url, timeout=5, headers={"User-Agent": "Mozilla/5.0"}
        )


class Log:
    @staticmethod
    def stdout(msg, **kwargs) -> None:
        print(msg, file=sys.stdout, **kwargs)

    @staticmethod
    def stderr(msg, **kwargs) -> None:
        print(msg, file=sys.stderr, **kwargs)


class Color:
    RESET = "\033[0m"
    TOOL_COLORS = {
        "fetch": "\033[2;36m",
        "search": "\033[2;32m",
        "read_file": "\033[2;34m",
        "summarize": "\033[2;33m",
        "grep_file": "\033[2;35m",
    }
    DEFAULT = "\033[2m"
    THINKING = "\033[2;3m"
    ERROR = "\033[2;31m"

    @staticmethod
    def tool(name: str, text: str) -> str:
        if Config.color_output_enabled():
            color = Color.TOOL_COLORS.get(name, Color.DEFAULT)
            return f"{color}{text}{Color.RESET}"
        return text

    @staticmethod
    def dim(text: str) -> str:
        return Color.tool("default", text)

    @staticmethod
    def thinking(text: str) -> str:
        if Config.color_output_enabled():
            return f"{Color.THINKING}{text}{Color.RESET}"
        return text


class Config:
    CONTENT_THRESHOLD = 5000
    MAX_TOOL_RESULT_CHARS = 8000
    GREP_MAX_MATCHES = 50

    @staticmethod
    def response_format():
        return os.environ.get("LLM_RESPONSE_FORMAT", None)

    @staticmethod
    def is_stream():
        if Config.response_format() is not None:
            return False
        return os.environ.get("LLM_STREAM", "True").lower() == "true"

    @staticmethod
    def tools_enabled():
        return os.environ.get("LLM_ENABLE_TOOLS", "True").lower() == "true"

    @staticmethod
    def color_output_enabled():
        if os.environ.get("NO_COLOR"):
            return False
        return os.environ.get("LLM_DISABLE_COLOR_OUTPUT", "False").lower() != "true"

    @staticmethod
    def thinking_enabled():
        return os.environ.get("LLM_SHOW_THINKING", "True").lower() == "true"

    @staticmethod
    def markdown_fetch_proxy() -> str | None:
        return os.environ.get("LLM_MARKDOWN_FETCH_PROXY")

    @staticmethod
    def markdown_search_proxy() -> str | None:
        return os.environ.get("LLM_MARKDOWN_SEARCH_PROXY")

    @staticmethod
    def markdown_image_search_proxy() -> str | None:
        return os.environ.get("LLM_MARKDOWN_IMAGE_SEARCH_PROXY")

    @staticmethod
    def get_system_prompt() -> str:
        today = date.today().isoformat()
        return (
            f"Today is: {today}. "
            "When making multiple independent tool calls, batch them in a single response for efficiency. "
            "IMPORTANT: Use fetch() to get content. If content is large, it returns a file_id (6 lowercase alphanumeric chars). "
            "When you need to read, grep, or summarize multiple files, use file_ids=[...] in a single call for efficiency. "
            "Only use read_file, grep_file, or summarize with that exact file_id returned by fetch. "
            "NEVER invent or guess file_ids - only use IDs explicitly returned by fetch."
        )

    @staticmethod
    def generate_file_id() -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

    @staticmethod
    def validate_file_id(file_id: str) -> str | None:
        if not file_id:
            return "file_id is required"
        if not re.match(r"^[a-z0-9]{6}$", file_id):
            return f"invalid file_id '{file_id}'. Must be 6 lowercase alphanumeric characters (e.g., 'abc123'). Do NOT invent file_ids - only use IDs returned by fetch."
        return None


class DuckDuckGoLiteSearch(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current_url = ""
        self.current_title = ""
        self.in_link = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "a":
            href = attrs_dict.get("href", "")
            if isinstance(href, str) and href.startswith("//duckduckgo.com/l/?uddg="):
                self.in_link = True
                self.current_url = href
                self.current_title = ""

    def handle_endtag(self, tag):
        if tag == "a" and self.in_link:
            self.in_link = False
            if self.current_url and self.current_title:
                parsed = urlparse(self.current_url)
                params = parse_qs(parsed.query)
                actual_url = params.get("uddg", [self.current_url])[0]
                self.results.append(
                    {
                        "url": unquote(actual_url),
                        "title": self.current_title.strip(),
                        "snippet": "",
                    }
                )

    def handle_data(self, data):
        if self.in_link:
            self.current_title += data

    def get_results(self, max_results=5):
        return self.results[:max_results]


class Cache:
    _storage: dict[str, str] = {}

    @staticmethod
    def store(file_id: str, content: str) -> None:
        wrapped_lines = []
        for line in content.splitlines():
            line = re.sub(r"(data:[^,]+,)[^)\s]+", r"\1[TRUNCATED]", line)

            if len(line) <= 200:
                wrapped_lines.append(line)
            else:
                start = 0
                while start < len(line):
                    chunk = line[start : start + 200]
                    last_space = chunk.rfind(" ")
                    if last_space > 0:
                        wrapped_lines.append(chunk[:last_space])
                        start += last_space + 1
                    else:
                        wrapped_lines.append(line[start : start + 200])
                        break
        Cache._storage[file_id] = "\n".join(wrapped_lines)

    @staticmethod
    def get(file_id: str) -> str | None:
        return Cache._storage.get(file_id)

    @staticmethod
    def read(
        file_ids: list[str], offset: int | None = None, limit: int | None = None
    ) -> str:
        results = []
        for file_id in file_ids:
            validation_error = Config.validate_file_id(file_id)
            if validation_error:
                results.append(f"Error: {validation_error}")
                continue

            content = Cache.get(file_id)
            if content is None:
                results.append(f"Error: file {file_id} not found")
                continue

            lines = content.splitlines()
            total_lines = len(lines)

            if offset is None:
                offset = 1
            if offset < 1:
                offset = 1

            if limit is None:
                limit = 50
            if limit < 1:
                limit = 1

            start = offset - 1
            end = start + limit

            selected = lines[start:end]
            result = "\n".join(
                f"{i + offset}: {line}" for i, line in enumerate(selected)
            )

            header = f"File {file_id} (lines {offset}-{min(end, total_lines)} of {total_lines})\n"
            results.append(header + result)

        return "\n\n---\n\n".join(results)

    @staticmethod
    def grep(
        file_ids: list[str],
        pattern: str,
        is_regex: bool = False,
        ignore_case: bool = False,
        context: int = 0,
    ) -> str:
        results = []
        for file_id in file_ids:
            validation_error = Config.validate_file_id(file_id)
            if validation_error:
                results.append(f"Error: {validation_error}")
                continue

            content = Cache.get(file_id)
            if content is None:
                results.append(f"Error: file {file_id} not found")
                continue

            lines = content.splitlines()
            total_lines = len(lines)

            flags = re.IGNORECASE if ignore_case else 0
            if is_regex:
                try:
                    regex = re.compile(pattern, flags)
                except re.error as e:
                    results.append(f"Error: invalid regex: {e}")
                    continue

                def matcher(line: str, regex=regex) -> bool:
                    return regex.search(line) is not None

            elif ignore_case:
                pattern_lower = pattern.lower()

                def matcher(line: str, pattern_lower=pattern_lower) -> bool:
                    return pattern_lower in line.lower()

            else:

                def matcher(line: str, pattern=pattern) -> bool:
                    return pattern in line

            matched_indices = set()
            for i, line in enumerate(lines):
                if matcher(line):
                    matched_indices.add(i)

            if not matched_indices:
                results.append(f'File {file_id}: no matches for "{pattern}"')
                continue

            all_matched_indices = matched_indices.copy()

            if context > 0:
                context_indices = set()
                for idx in matched_indices:
                    for j in range(
                        max(0, idx - context), min(total_lines, idx + context + 1)
                    ):
                        context_indices.add(j)
                matched_indices = context_indices

            sorted_indices = sorted(matched_indices)

            groups = []
            current_group = []
            for i, idx in enumerate(sorted_indices):
                if not current_group or idx == sorted_indices[i - 1] + 1:
                    current_group.append(idx)
                else:
                    groups.append(current_group)
                    current_group = [idx]
            if current_group:
                groups.append(current_group)

            output_lines = [
                f'File {file_id}: {len(all_matched_indices)} matches for "{pattern}"'
            ]
            match_count = 0
            truncated = False

            for group in groups:
                if truncated:
                    break
                output_lines.append("--")
                for idx in group:
                    if match_count >= Config.GREP_MAX_MATCHES:
                        truncated = True
                        break
                    line_num = idx + 1
                    prefix = ">" if idx in all_matched_indices else " "
                    output_lines.append(f"{prefix}{line_num}: {lines[idx]}")
                    match_count += 1

            if len(all_matched_indices) > Config.GREP_MAX_MATCHES:
                output_lines.append("--")
                output_lines.append(
                    f"... {len(all_matched_indices) - Config.GREP_MAX_MATCHES} more matches not shown"
                )

            results.append("\n".join(output_lines))

        return "\n\n---\n\n".join(results)


class Tools:
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
            file_id = Config.generate_file_id()
            Cache.store(file_id, content)
            return f'Stored as {file_id} ({len(content)} chars). Tools: read_file, grep_file, summarize (file_id="{file_id}")'

        proxy = Config.markdown_fetch_proxy()
        if proxy:
            fetch_url = f"{proxy.rstrip('/')}/{url}"
        else:
            fetch_url = url

        headers = {"User-Agent": "Mozilla/5.0"}
        last_error = None
        for attempt in range(3):
            try:
                response = await AsyncHttp.get(fetch_url, timeout=30, headers=headers)
                if response.status_code == 200:
                    return store_and_return(response.text)
                return f"fetch failed (status {response.status_code})"
            except Exception as e:
                last_error = e
                logger.warning(
                    "Fetch attempt %d/3 failed for %s: %s", attempt + 1, url, e
                )
                if attempt < 2:
                    await asyncio.sleep(0.3 * (attempt + 1))
                    continue

        logger.error("Fetch failed for %s after 3 attempts: %s", url, last_error)
        return f"fetch failed: {last_error}"

    @staticmethod
    async def fetch(urls: list[str]) -> str:
        results = []
        for url in urls:
            result = await Tools._fetch_single(url)
            results.append(f"{url}: {result}")
        return "\n\n---\n\n".join(results)

    @staticmethod
    async def _search_images(query: str, max_results: int = 5) -> str:
        from urllib.parse import quote

        proxy = Config.markdown_image_search_proxy()
        if proxy:
            url = f"{proxy.rstrip('/')}/{quote(query)}"
        else:
            url = f"https://duckduckgo.com/?q={quote(query)}&ia=images&iax=images"

        try:
            response = await AsyncHttp.get(
                url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            if proxy:
                return response.text

            html = response.text
            results = []
            seen_urls = set()

            img_pattern = re.compile(r"!\[Image \d+:")
            for img_match in img_pattern.finditer(html):
                search_start = img_match.end()
                link_match = re.search(r"\]\((https?://[^)]+)\)", html[search_start:])
                if not link_match:
                    continue

                img_url = link_match.group(1)
                if "duckduckgo.com" in img_url and "/iu/" not in img_url:
                    continue
                if img_url in seen_urls:
                    continue
                seen_urls.add(img_url)

                parsed = urlparse(img_url)
                params = parse_qs(parsed.query)
                if "u" in params:
                    target_url = unquote(params["u"][0])
                else:
                    target_url = img_url

                title_start = img_match.end()
                title_end = search_start + link_match.start()
                title = html[title_start:title_end].strip()

                results.append(f"{len(results) + 1}. [{title}]({target_url})")
                if len(results) >= max_results:
                    break

            if not results:
                return "No images found."
            return "\n".join(results)
        except Exception:
            logger.error("Image search failed for query: %s", query, exc_info=True)
            return "Image search failed."

    @staticmethod
    async def _run_search(query: str, max_results: int = 5) -> str:
        from urllib.parse import quote

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "sv-SE,sv;q=0.9",
            "Cache-Control": "max-age=0",
            "Priority": "u=0, i",
            "Sec-Ch-Ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Linux"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "DNT": "1",
            "Connection": "keep-alive",
        }

        proxy = Config.markdown_search_proxy()
        if proxy:
            url = f"{proxy.rstrip('/')}/{quote(query)}"
            for attempt in range(3):
                try:
                    response = await AsyncHttp.get(url, timeout=10, headers=headers)
                    response.raise_for_status()
                    return response.text.strip()
                except Exception:
                    if attempt < 2:
                        logger.warning(
                            "Search proxy attempt %d failed for '%s', retrying...",
                            attempt + 1,
                            query,
                        )
                        await asyncio.sleep(0.3 * (attempt + 1))
                        continue
                    logger.error(
                        "Search proxy failed after retries for '%s'",
                        query,
                        exc_info=True,
                    )
            return "Search failed after retries."

        url = f"https://lite.duckduckgo.com/lite/?q={quote(query)}"
        for attempt in range(3):
            try:
                response = await AsyncHttp.get(url, timeout=10, headers=headers)
                response.raise_for_status()

                parser = DuckDuckGoLiteSearch()
                parser.feed(response.text)
                results = parser.get_results(max_results)
                if results:
                    lines = []
                    for i, r in enumerate(results, 1):
                        lines.append(f"{i}. [{r['title']}]({r['url']})")
                        if r["snippet"]:
                            lines.append(f"   {r['snippet']}")
                        lines.append("")
                    return "\n".join(lines).strip()
            except Exception:
                if attempt < 2:
                    logger.warning(
                        "DuckDuckGo search attempt %d failed for '%s', retrying...",
                        attempt + 1,
                        query,
                    )
                    await asyncio.sleep(0.3 * (attempt + 1))
                    continue
                logger.error(
                    "DuckDuckGo search failed after retries for '%s'",
                    query,
                    exc_info=True,
                )

        return "Search failed after retries."

    @staticmethod
    async def search(
        queries: list[str], max_results: int = 5, images_only: bool = False
    ) -> str:
        async def search_task(query: str) -> tuple[str, str]:
            if images_only:
                result = await Tools._search_images(query, max_results)
            else:
                result = await Tools._run_search(query, max_results)
            file_id = Config.generate_file_id()
            Cache.store(file_id, result)
            return (query, file_id)

        file_ids = await asyncio.gather(*(search_task(query) for query in queries))

        lines = [f"Query '{q}' stored in file_id={fid}" for q, fid in file_ids]
        lines.append("")
        lines.append("Use read_file, grep_file or summarize to get the content.")
        return "\n".join(lines)

    @staticmethod
    async def summarize(
        file_ids: list[str],
        directives: list[str],
        max_length: int = 1000,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
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

            last_error = None
            for attempt in range(5):
                try:
                    response = await AsyncHttp.post(
                        os.environ["LLM_HOST"],
                        headers=headers,
                        json=payload,
                        timeout=60,
                    )
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Summarize attempt {attempt + 1}/5 failed for {file_id}: {e}"
                    )
                    if attempt < 4:
                        await asyncio.sleep(0.5 * (attempt + 1))
                    continue

            if last_error is not None:
                logger.error(
                    "Summarize failed for file %s after 5 attempts: %s",
                    file_id,
                    last_error,
                )
                return (
                    file_id,
                    directive,
                    "Error summarizing: All 5 attempts timed out. Do you want me to try again?",
                )

            if response.status_code != 200:
                logger.error(
                    "Summarize API error %d for file %s",
                    response.status_code,
                    file_id,
                )
                return (file_id, directive, f"Error: {response.status_code}")

            summary = response.json()["choices"][0]["message"]["content"]

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

    @staticmethod
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

    @staticmethod
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
                            value = json.loads(Tools._repair_json(trimmed))
                        except json.JSONDecodeError:
                            pass
            out[key] = value
        return out

    @staticmethod
    def _parse_tool_args(args_str: str) -> dict:
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse tool arguments, attempting repair: %r",
                args_str[:200],
            )
            try:
                args = json.loads(Tools._repair_json(args_str))
            except json.JSONDecodeError:
                args = {}
        return Tools._unwrap_args(args)

    @staticmethod
    async def execute(tool_name: str, tool_args: dict) -> str:
        if tool_name == "fetch":
            urls = tool_args.get("urls", [])
            if isinstance(urls, str):
                urls = [urls]
            return await Tools.fetch(urls)
        if tool_name == "search":
            queries = tool_args.get("queries", [])
            if isinstance(queries, str):
                queries = [queries]
            return await Tools.search(
                queries,
                tool_args.get("max_results", 5),
                tool_args.get("images_only", False),
            )
        if tool_name == "read_file":
            file_ids = tool_args.get("file_ids", [])
            if isinstance(file_ids, str):
                file_ids = [file_ids]
            return Cache.read(
                file_ids,
                tool_args.get("offset"),
                tool_args.get("limit"),
            )
        if tool_name == "summarize":
            file_ids = tool_args.get("file_ids", [])
            if isinstance(file_ids, str):
                file_ids = [file_ids]
            return await Tools.summarize(
                file_ids,
                tool_args.get("directives", []),
                tool_args.get("max_length", 1000),
                tool_args.get("offset"),
                tool_args.get("limit"),
            )
        if tool_name == "grep_file":
            file_ids = tool_args.get("file_ids", [])
            if isinstance(file_ids, str):
                file_ids = [file_ids]
            return Cache.grep(
                file_ids,
                tool_args.get("pattern", ""),
                tool_args.get("is_regex", False),
                tool_args.get("ignore_case", False),
                tool_args.get("context", 0),
            )
        return f"Unknown tool: {tool_name}"

    @staticmethod
    async def execute_wrapper(tool_call: dict) -> tuple[str, str]:
        tool_id = tool_call.get("id", "")
        func = tool_call.get("function", {})
        tool_name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        args = Tools._parse_tool_args(args_str)
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


_STREAM_EOF = object()


def _safe_next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_EOF


class LLMClient:
    @staticmethod
    def body(messages: list) -> dict:
        d = {
            "messages": messages,
            "model": os.environ["LLM_MODEL"],
            "temperature": float(os.environ.get("LLM_TEMPERATURE", 0.1)),
            "stream": Config.is_stream(),
        }

        if Config.response_format() is not None:
            d["response_format"] = Config.response_format()

        if Config.tools_enabled():
            d["tools"] = Tools.SCHEMA

        return d

    @staticmethod
    async def stream(prompt: list[str]) -> None:
        messages = [
            {
                "role": "system",
                "content": os.environ.get(
                    "LLM_SYSTEM_PROMPT", Config.get_system_prompt()
                ),
            },
            {"role": "user", "content": " ".join(prompt)},
        ]

        headers = {
            "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        while True:
            msg = LLMClient.body(messages=messages)

            response = await AsyncHttp.post(
                os.environ["LLM_HOST"],
                headers=headers,
                data=json.dumps(msg),
                stream=Config.is_stream(),
                timeout=60,
            )

            if response.status_code in (408, 429) or response.status_code >= 500:
                Log.stderr(
                    f"{Color.ERROR}[error]: {response.status_code}: {response.content.decode()}, retrying{Color.RESET}"
                )
                await asyncio.sleep(1)
                response = await AsyncHttp.post(
                    os.environ["LLM_HOST"],
                    headers=headers,
                    data=json.dumps(msg),
                    stream=False,
                    timeout=60,
                )

            if response.status_code != 200:
                Log.stderr(
                    f"{Color.ERROR}[error]: {response.status_code}: {response.content.decode()}{Color.RESET}"
                )
                sys.exit(1)

            if Config.is_stream():
                message, printed = await LLMClient._read_stream(response)
            else:
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {})
                printed = False

                if Config.thinking_enabled():
                    reasoning = (
                        message.get("reasoning_content")
                        or message.get("reasoning")
                        or ""
                    )
                    if reasoning:
                        Log.stderr(
                            f"{Color.dim('[thinking]')} {Color.thinking(reasoning)}"
                        )

            tool_calls = message.get("tool_calls", [])
            if not tool_calls or not Config.tools_enabled():
                content = message.get("content", "")
                if content and not printed:
                    Log.stdout(content)
                return

            message.pop("reasoning_content", None)
            message.pop("reasoning", None)
            message.pop("provider_specific_fields", None)
            messages.append(message)

            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                args_str = func.get("arguments", "{}")
                args = Tools._parse_tool_args(args_str)
                tool_name = func.get("name", "unknown")
                Log.stderr(
                    Color.tool(
                        tool_name,
                        f"[tool] {tool_name}({', '.join(f'{k}={repr(v)}' for k, v in args.items())})",
                    )
                )

            results = {}
            exec_tasks = [Tools.execute_wrapper(tc) for tc in tool_calls]
            exec_results = await asyncio.gather(*exec_tasks)
            for tool_id, result in exec_results:
                results[tool_id] = result

            for tool_call in tool_calls:
                tool_id = tool_call.get("id", "")
                if not tool_id:
                    continue
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": results.get(tool_id, ""),
                    }
                )

    @staticmethod
    async def _read_stream(response) -> dict:
        content_parts = []
        reasoning_parts = []
        tool_calls: dict[int, dict] = {}
        final_message = None
        reasoning_flushed = False

        def flush_thinking() -> None:
            nonlocal reasoning_flushed
            if reasoning_flushed or not reasoning_parts:
                return
            reasoning_flushed = True
            if Config.color_output_enabled():
                Log.stderr(Color.RESET, end="")
            if not "".join(reasoning_parts).endswith("\n"):
                Log.stderr("")

        def append_thinking(fragment: str) -> None:
            if not Config.thinking_enabled() or not fragment:
                return
            if not reasoning_parts:
                prefix = f"{Color.dim('[thinking]')} "
                if Config.color_output_enabled():
                    prefix += Color.THINKING
                Log.stderr(prefix, end="", flush=True)
            reasoning_parts.append(fragment)
            Log.stderr(fragment, end="", flush=True)

        def append_content(fragment: str) -> None:
            if not fragment:
                return
            if reasoning_parts:
                flush_thinking()
            content_parts.append(fragment)
            Log.stdout(fragment, end="", flush=True)

        try:
            response.encoding = "utf-8"
            lines = response.iter_lines(decode_unicode=True)
            while True:
                line = await asyncio.to_thread(_safe_next, lines)
                if line is _STREAM_EOF:
                    break
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices")
                if not choices:
                    if chunk.get("message"):
                        final_message = chunk["message"]
                    continue
                choice = choices[0]

                if choice.get("message"):
                    final_message = choice["message"]

                delta = choice.get("delta", {})
                reasoning = (
                    delta.get("reasoning_content") or delta.get("reasoning") or ""
                )
                if reasoning:
                    append_thinking(reasoning)

                content = delta.get("content") or ""
                if content:
                    append_content(content)

                for tc in delta.get("tool_calls", []):
                    index = tc.get("index", 0)
                    entry = tool_calls.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    entry["id"] = tc.get("id") or entry["id"] or f"call_{index}"
                    fn = tc.get("function", {})
                    entry["function"]["name"] += fn.get("name", "") or ""
                    entry["function"]["arguments"] += fn.get("arguments", "") or ""
        finally:
            response.close()

        content = "".join(content_parts)
        if content and not content.endswith("\n"):
            Log.stdout("", flush=True)
        reasoning = "".join(reasoning_parts)
        flush_thinking()

        if not content and not reasoning and not tool_calls and final_message:
            return (final_message, False)

        message = {
            "role": "assistant",
            "content": None if tool_calls and not content else content,
        }
        if reasoning:
            message["reasoning"] = reasoning
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        return (message, bool(content_parts))


async def main() -> None:
    prompt = [*sys.argv[1:]]
    if not sys.stdin.isatty():
        prompt.extend(["\n\n", *sys.stdin.read().splitlines()])
    await LLMClient.stream(prompt)


if __name__ == "__main__":
    asyncio.run(main())
