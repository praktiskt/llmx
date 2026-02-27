#!/usr/bin/python3
from __future__ import annotations

import json
import os
import random
import re
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from datetime import date
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import requests


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
    MAX_TOOL_RESULT_CHARS = 5000
    GREP_MAX_MATCHES = 50
    BINARY_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".epub", ".doc"}

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
    def get_system_prompt() -> str:
        today = date.today().isoformat()
        return (
            f"Today: {today}. "
            "Briefly respond to the user, being crystal clear and helpful. Never lie. "
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


class HTMLToMarkdown(HTMLParser):
    HEADERS = {
        "h1": "\n# ",
        "h2": "\n## ",
        "h3": "\n### ",
        "h4": "\n#### ",
        "h5": "\n##### ",
        "h6": "\n###### ",
    }
    INLINE = {"strong": "**", "b": "**", "em": "*", "i": "*"}
    SELF_CLOSING = {"br": "\n", "hr": "\n---\n"}
    BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre"}

    def __init__(self):
        super().__init__()
        self.result = []
        self.tag_stack = []
        self.list_stack = []
        self.ignore_tags = {
            "script",
            "style",
            "noscript",
            "head",
            "meta",
            "link",
            "base",
            "template",
            "object",
            "embed",
            "applet",
            "canvas",
            "source",
            "track",
            "map",
            "area",
            "portal",
        }
        self.in_pre = self.in_code = self.in_blockquote = False

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        attrs_dict = dict(attrs)
        self.tag_stack.append(t)

        if t in self.ignore_tags:
            return

        if t in self.HEADERS:
            self.result.append(self.HEADERS[t])
        elif t in self.SELF_CLOSING:
            self.result.append(self.SELF_CLOSING[t])
        elif t in self.INLINE:
            self.result.append(self.INLINE[t])
        elif t == "p":
            self.result.append("\n\n")
        elif t == "a":
            self.href = attrs_dict.get("href", "")
            self.result.append("[")
        elif t == "code":
            self.in_code = True
            self.result.append("`" if not self.in_pre else "")
        elif t == "pre":
            self.in_pre = True
            self.result.append("\n```\n")
        elif t == "blockquote":
            self.in_blockquote = True
            self.result.append("\n> ")
        elif t == "ul":
            self.list_stack.append("ul")
            self.result.append("\n")
        elif t == "ol":
            self.list_stack.append(("ol", 1))
            self.result.append("\n")
        elif t == "li" and self.list_stack:
            lst = self.list_stack[-1]
            if lst == "ul":
                self.result.append("- ")
            else:
                self.result.append(f"{lst[1]}. ")
                self.list_stack[-1] = ("ol", lst[1] + 1)
        elif t == "img":
            self.result.append(
                f"![{attrs_dict.get('alt', '')}]({attrs_dict.get('src', '')})"
            )

    def handle_endtag(self, tag):
        t = tag.lower()
        if self.tag_stack and self.tag_stack[-1] == t:
            self.tag_stack.pop()

        if t in self.ignore_tags:
            return

        if t in self.HEADERS:
            self.result.append("\n")
        elif t in self.INLINE:
            self.result.append(self.INLINE[t])
        elif t == "a":
            self.result.append(f"]({getattr(self, 'href', '')})")
        elif t == "code":
            self.in_code = False
            self.result.append("`" if not self.in_pre else "")
        elif t == "pre":
            self.in_pre = False
            self.result.append("\n```\n")
        elif t == "blockquote":
            self.in_blockquote = False
            self.result.append("\n")
        elif t == "li":
            self.result.append("\n")
        elif t in ("ul", "ol") and self.list_stack:
            self.list_stack.pop()

    def handle_data(self, data):
        for tag in self.tag_stack:
            if tag in self.ignore_tags:
                return
        if self.in_pre:
            self.result.append(data)
            return
        if self.in_blockquote:
            for line in data.split("\n"):
                if line.strip():
                    self.result.append(line)
            return

        in_inline_only = any(t in self.INLINE for t in self.tag_stack)
        in_block = any(t in self.BLOCKS for t in self.tag_stack)

        if in_inline_only and not in_block:
            self.result.append(data)
        elif in_block:
            self.result.append(data)
        else:
            text = " ".join(data.split())
            if text:
                self.result.append(text)

    def get_markdown(self):
        return "".join(self.result).strip()


class DuckDuckGoLiteSearch(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current_url = ""
        self.current_title = ""
        self.in_link = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "a" and attrs_dict.get("href", "").startswith(
            "//duckduckgo.com/l/?uddg="
        ):
            self.in_link = True
            self.current_url = attrs_dict.get("href", "")
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


class FileCache:
    _storage: dict[str, str] = {}

    @staticmethod
    def store(file_id: str, content: str) -> None:
        FileCache._storage[file_id] = content

    @staticmethod
    def get(file_id: str) -> str | None:
        return FileCache._storage.get(file_id)


class Cache:
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

            content = FileCache.get(file_id)
            if content is None:
                results.append(f"Error: file {file_id} not found")
                continue

            lines = content.splitlines()
            total_lines = len(lines)

            if offset is None:
                offset = 1
            if offset < 1:
                offset = 1

            start = offset - 1
            end = total_lines if limit is None else start + limit

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

            content = FileCache.get(file_id)
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
                matcher = lambda line: regex.search(line) is not None
            else:
                if ignore_case:
                    pattern_lower = pattern.lower()
                    matcher = lambda line: pattern_lower in line.lower()
                else:
                    matcher = lambda line: pattern in line

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
                "description": "Fetch content from URLs and return as markdown",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of URLs to fetch (at least one)",
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
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {
                            "type": "integer",
                            "description": "Max results (default 5)",
                        },
                    },
                    "required": ["query"],
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
    def _fetch_single(url: str) -> str:
        def fetch_and_process(fetch_url: str) -> str | None:
            response = requests.get(
                fetch_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()

            if fetch_url.startswith("https://r.jina.ai/"):
                content = response.text
            elif "text/html" in content_type or url.endswith((".html", ".htm")):
                parser = HTMLToMarkdown()
                parser.feed(response.text)
                content = parser.get_markdown()
            else:
                content = response.text

            return content if content else None

        content = None
        try:
            fetch_url = url
            url_lower = url.lower()
            if any(url_lower.endswith(ext) for ext in Config.BINARY_EXTENSIONS):
                fetch_url = f"https://r.jina.ai/{url}"
                content = fetch_and_process(fetch_url)
            else:
                content = fetch_and_process(fetch_url)
                if not content:
                    fallback_url = f"https://r.jina.ai/{url}"
                    content = fetch_and_process(fallback_url)
        except requests.HTTPError:
            try:
                fallback_url = f"https://r.jina.ai/{url}"
                content = fetch_and_process(fallback_url)
            except Exception:
                pass
        except Exception:
            try:
                fallback_url = f"https://r.jina.ai/{url}"
                content = fetch_and_process(fallback_url)
            except Exception:
                pass

        if not content:
            content = "fetch failed"

        file_id = Config.generate_file_id()
        FileCache.store(file_id, content)
        return f'Stored as {file_id} ({len(content)} chars). Tools: read_file, grep_file, summarize (file_id="{file_id}")'

    @staticmethod
    def fetch(urls: list[str]) -> str:
        results = []
        for url in urls:
            result = Tools._fetch_single(url)
            results.append(f"{url}: {result}")
        return "\n\n---\n\n".join(results)

    @staticmethod
    def search(query: str, max_results: int = 5) -> str:
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

        delays = [0.2, 0.3, 0.4]
        endpoints = [
            ("https://lite.duckduckgo.com/lite/?q=", None),
            (
                "https://r.jina.ai/https://lite.duckduckgo.com/lite/?q=",
                "jina",
            ),
        ]

        for url_prefix, endpoint_type in endpoints:
            url = f"{url_prefix}{query}"
            for attempt, delay in enumerate(delays):
                try:
                    response = requests.get(url, timeout=30, headers=headers)
                    response.raise_for_status()

                    if endpoint_type == "jina":
                        return response.text.strip()

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
                    if attempt < len(delays) - 1:
                        time.sleep(delay)
                        continue

        return "Search failed after retries."

    @staticmethod
    def summarize(
        file_ids: list[str],
        directives: list[str],
        max_length: int = 1000,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        def summarize_task(task: tuple) -> tuple:
            file_id, directive, content = task
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
            }

            headers = {
                "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
                "Content-Type": "application/json",
            }

            try:
                response = requests.post(
                    os.environ["LLM_HOST"],
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if response.status_code != 200:
                    return (file_id, directive, f"Error: {response.status_code}")

                summary = response.json()["choices"][0]["message"]["content"]

                if len(summary) > max_length:
                    summary = summary[: max_length - 3] + "..."

                return (file_id, directive, summary)
            except Exception as e:
                return (file_id, directive, f"Error summarizing: {str(e)}")

        tasks = []
        for file_id in file_ids:
            validation_error = Config.validate_file_id(file_id)
            if validation_error:
                tasks.append((file_id, "", f"Error: {validation_error}"))
                continue

            content = FileCache.get(file_id)
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

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(summarize_task, task): task[0] for task in tasks}
            results_map = {}
            for future in as_completed(futures):
                file_id, directive, result = future.result()
                if file_id not in results_map:
                    results_map[file_id] = []
                results_map[file_id].append(result)

        results = []
        for file_id in file_ids:
            if file_id not in results_map:
                continue
            summaries = results_map[file_id]
            file_summaries = [f"{i}. {s}" for i, s in enumerate(summaries, 1)]
            results.append(f"File {file_id}:\n" + "\n\n".join(file_summaries))

        return "\n\n---\n\n".join(results)

    @staticmethod
    def execute(tool_name: str, tool_args: dict) -> str:
        if tool_name == "fetch":
            return Tools.fetch(tool_args.get("urls", []))
        if tool_name == "search":
            return Tools.search(
                tool_args.get("query", ""), tool_args.get("max_results", 5)
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
            return Tools.summarize(
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
    def execute_wrapper(tool_call: dict) -> tuple[str, str]:
        tool_id = tool_call.get("id", "")
        func = tool_call.get("function", {})
        tool_name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        result = Tools.execute(tool_name, args)

        if len(result) > Config.MAX_TOOL_RESULT_CHARS:
            file_ids = args.get("file_ids", [])
            if isinstance(file_ids, str):
                file_ids = [file_ids]
            if file_ids:
                result = f'Result too large ({len(result)} chars). Use summarize(file_ids={file_ids}, directives=["..."]) to extract what you need.'
                return (tool_id, result)
            result = f"Result too large ({len(result)} chars). Use summarize to extract what you need."

        return (tool_id, result)


class LLMClient:
    @staticmethod
    def body(prompt: list[str] | None = None, messages: list | None = None) -> str:
        if prompt is None:
            prompt = []
        if messages is None:
            messages = [
                {
                    "role": "system",
                    "content": os.environ.get(
                        "LLM_SYSTEM_PROMPT", Config.get_system_prompt()
                    ),
                },
                {"role": "user", "content": " ".join(prompt)},
            ]

        d = {
            "messages": messages,
            "model": os.environ["LLM_MODEL"],
            "temperature": float(os.environ.get("LLM_TEMPERATURE", 0.1)),
            "stream": False,
        }

        if Config.response_format() is not None:
            d["response_format"] = Config.response_format()

        if Config.tools_enabled():
            d["tools"] = Tools.SCHEMA

        return json.dumps(d)

    @staticmethod
    def stream(prompt: list[str]) -> None:
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
            msg = json.loads(LLMClient.body(messages=messages))

            response = requests.post(
                os.environ["LLM_HOST"],
                headers=headers,
                data=json.dumps(msg),
                stream=False,
            )

            if response.status_code != 200:
                Log.stderr(f"{response.status_code}: {response.content.decode()}")
                sys.exit(1)

            data = response.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})

            if Config.thinking_enabled():
                reasoning = (
                    message.get("reasoning_content") or message.get("reasoning") or ""
                )
                if reasoning:
                    Log.stderr(f"{Color.dim('[thinking]')} {Color.thinking(reasoning)}")

            tool_calls = message.get("tool_calls", [])
            if not tool_calls or not Config.tools_enabled():
                content = message.get("content", "")
                if content:
                    Log.stdout(content)
                return

            messages.append(message)

            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                tool_name = func.get("name", "unknown")
                Log.stderr(
                    Color.tool(
                        tool_name,
                        f"[tool] {tool_name}({', '.join(f'{k}={repr(v)}' for k, v in args.items())})",
                    )
                )

            results = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(Tools.execute_wrapper, tc): tc for tc in tool_calls
                }
                for future in as_completed(futures):
                    tool_id, result = future.result()
                    results[tool_id] = result

            for tool_call in tool_calls:
                tool_id = tool_call.get("id")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": results[tool_id],
                    }
                )


def main() -> None:
    with suppress(KeyboardInterrupt):
        prompt = [*sys.argv[1:]]
        if not sys.stdin.isatty():
            prompt.extend(["\n\n", *sys.stdin.read().splitlines()])
        LLMClient.stream(prompt)


if __name__ == "__main__":
    main()
