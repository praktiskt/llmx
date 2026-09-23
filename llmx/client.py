import asyncio
import json
import logging
import os
import sys
import threading
from collections.abc import Awaitable

from .config import Config
from .output import Color, Log
from .tools import Tools, parse_tool_args, sanitize_tool_call_args
from .transport import AsyncHttp, LLMAPIError, Response, request_with_retries

logger = logging.getLogger(__name__)

_STREAM_EOF = object()

_TRUNCATION_MARKER = "[earlier messages truncated to fit the context window]"

_HARNESS_TAGS = (("<system-reminder>", "</system-reminder>"),)


class StreamFilter:
    """Suppresses harness-injected wrapper blocks (e.g. <system-reminder>)
    that some gateway models echo into their completions.

    Tags split across SSE fragments are handled via a small rolling buffer;
    call flush() once the stream ends to release any held-back tail.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._close_tag: str | None = None

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buf += text
        out: list[str] = []
        while True:
            if self._close_tag is not None:
                end = self._buf.find(self._close_tag)
                if end == -1:
                    keep = min(len(self._close_tag) - 1, len(self._buf))
                    self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                    return "".join(out)
                self._buf = self._buf[end + len(self._close_tag) :]
                self._close_tag = None
                continue

            matched_open = None
            for open_tag, close_tag in _HARNESS_TAGS:
                idx = self._buf.find(open_tag)
                if idx != -1:
                    matched_open = (open_tag, close_tag)
                    out.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(open_tag) :]
                    self._close_tag = close_tag
                    break
            if matched_open is None:
                break
        hold = 0
        for open_tag, _ in _HARNESS_TAGS:
            for size in range(1, len(open_tag)):
                if self._buf.endswith(open_tag[:size]):
                    hold = max(hold, size)
        emit = len(self._buf) - hold
        if emit > 0:
            out.append(self._buf[:emit])
            self._buf = self._buf[emit:]
        return "".join(out)

    def flush(self) -> str:
        if self._close_tag is not None:
            # Unterminated block at end of stream: discard everything held.
            self._buf = ""
            self._close_tag = None
            return ""
        rest, self._buf = self._buf, ""
        return rest


def truncate_history(messages: list[dict]) -> list[dict]:
    """Drop oldest whole turns (assistant+its tool results stay together) until
    the serialized history fits Config.MAX_CONTEXT_CHARS. System prompt kept."""
    if len(messages) <= 1:
        return messages
    system, rest = messages[0], messages[1:]

    units: list[list[dict]] = []
    i = 0
    while i < len(rest):
        msg = rest[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            j = i + 1
            while j < len(rest) and rest[j].get("role") == "tool":
                j += 1
            units.append(rest[i:j])
            i = j
        else:
            units.append([msg])
            i += 1

    def size(unit: list[dict] | dict) -> int:
        return len(json.dumps(unit, default=str))

    total = size(system) + sum(size(u) for u in units)
    if total <= Config.MAX_CONTEXT_CHARS:
        return messages

    while units and total > Config.MAX_CONTEXT_CHARS:
        total -= size(units.pop(0))

    return [
        system,
        {"role": "user", "content": _TRUNCATION_MARKER},
        *[m for unit in units for m in unit],
    ]


def repair_history_tool_calls(messages: list) -> None:
    """Fix malformed tool_call arguments in an in-flight history, in place.

    Repairable arguments are replaced with canonically serialized JSON so
    strict providers don't reject the whole replay. Unrepairable arguments
    are left untouched — the provider's 400 then surfaces against the
    original payload instead of a silently rewritten one. Internal markers
    are stripped either way.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        msg.pop("_malformed_tool_calls_dropped", None)
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue
        for tc in tool_calls:
            func = tc.get("function") or {}
            args = func.get("arguments")
            if args is None:
                continue
            sanitized = sanitize_tool_call_args(args)
            if sanitized is None and args:
                logger.error(
                    "Unrepairable tool_call arguments left in history: name=%s args=%.200r",
                    func.get("name") or "unknown",
                    args,
                )
                continue
            if sanitized is None:
                continue
            if sanitized != args:
                logger.warning(
                    "Repaired malformed tool_call arguments in history: name=%s -> %.200s",
                    func.get("name") or "unknown",
                    sanitized,
                )
                func["arguments"] = sanitized


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

        tools = Tools.schema()
        if tools:
            d["tools"] = tools

        return d

    @staticmethod
    async def _next_user_message() -> str | None:
        """Prompt until a non-blank tty line; None on EOF or quit command."""
        try:
            import readline  # noqa: F401 — enables Alt+Backspace etc.
        except ImportError:
            pass
        while True:
            try:
                line = await asyncio.to_thread(input, "> ")
            except EOFError:
                return None
            text = line.strip()
            if text.lower() in {"exit", "quit", "/exit", "/quit"}:
                return None
            if text:
                return text

    @staticmethod
    async def stream(prompt: list[str]) -> None:
        prompt_text = " ".join(prompt).strip()
        if not prompt_text and Config.interactive_enabled() and sys.stdin.isatty():
            first = await LLMClient._next_user_message()
            if first is None:
                return
            prompt_text = first

        messages = [
            {
                "role": "system",
                "content": os.environ.get(
                    "LLM_SYSTEM_PROMPT", Config.get_system_prompt()
                ),
            },
            {"role": "user", "content": prompt_text},
        ]

        headers = {
            "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        while True:
            messages = truncate_history(messages)
            repair_history_tool_calls(messages)
            msg = LLMClient.body(messages=messages)

            def send(msg=msg) -> Awaitable[Response]:
                # reuse=False: fresh dial per request so retries rotate
                # load-balancer backends instead of pinning one.
                return AsyncHttp.post(
                    os.environ["LLM_HOST"],
                    headers=headers,
                    data=json.dumps(msg),
                    stream=Config.is_stream(),
                    timeout=Config.LLM_TIMEOUT,
                    reuse=False,
                )

            def log_failure(text: str) -> None:
                Log.stderr(f"{Color.ERROR}[error]: {text}{Color.RESET}")

            response = await request_with_retries(
                send,
                attempts=5,
                on_exception=lambda a, e: log_failure(f"attempt {a}/5 failed: {e}"),
                on_retry=lambda a, r: log_failure(
                    f"API {r.status_code} (attempt {a}/5): "
                    f"{r.content.decode(errors='replace')[:200]}, retrying"
                ),
            )

            if response is None or response.status_code != 200:
                status = response.status_code if response else "no response"
                body = (
                    response.content.decode(errors="replace")[:500]
                    if response
                    else "all attempts raised"
                )
                raise LLMAPIError(
                    f"API request failed after 5 attempts (last status: {status}): {body}"
                )

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

            tool_calls = message.get("tool_calls") or []
            if not tool_calls or not Tools.schema():
                content = message.get("content", "")
                if content and not printed:
                    Log.stdout(content)
                if not Config.interactive_enabled() or not sys.stdin.isatty():
                    return
                message.pop("reasoning_content", None)
                message.pop("reasoning", None)
                message.pop("provider_specific_fields", None)
                messages.append(message)
                user_line = await LLMClient._next_user_message()
                if user_line is None:
                    return
                messages.append({"role": "user", "content": user_line})
                continue

            message.pop("reasoning_content", None)
            message.pop("reasoning", None)
            message.pop("provider_specific_fields", None)
            messages.append(message)

            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                args_str = func.get("arguments", "{}")
                args = parse_tool_args(args_str)
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
    async def _read_stream(response) -> tuple[dict, bool]:
        content_parts = []
        reasoning_parts = []
        tool_calls: dict[int, dict] = {}
        final_message = None
        reasoning_len = 0
        reasoning_len_flushed = 0
        reasoning_pending_ends_nl = True
        reasoning_filter = StreamFilter()
        content_filter = StreamFilter()

        def flush_thinking() -> None:
            nonlocal reasoning_len_flushed, reasoning_pending_ends_nl
            if reasoning_len == reasoning_len_flushed:
                return
            if Config.color_output_enabled():
                Log.stderr(Color.RESET, end="")
            if not reasoning_pending_ends_nl:
                Log.stderr("")
            reasoning_len_flushed = reasoning_len
            reasoning_pending_ends_nl = True

        def append_thinking(fragment: str) -> None:
            nonlocal reasoning_len, reasoning_len_flushed, reasoning_pending_ends_nl
            fragment = reasoning_filter.feed(fragment)
            if not Config.thinking_enabled() or not fragment:
                return
            if reasoning_len == reasoning_len_flushed:
                prefix = f"{Color.dim('[thinking]')} "
                if Config.color_output_enabled():
                    prefix += Color.THINKING
                Log.stderr(prefix, end="", flush=True)
            reasoning_parts.append(fragment)
            reasoning_len += len(fragment)
            reasoning_pending_ends_nl = fragment.endswith("\n")
            Log.stderr(fragment, end="", flush=True)

        def append_content(fragment: str) -> None:
            if not fragment:
                return
            fragment = content_filter.feed(fragment)
            if not fragment:
                return
            if reasoning_parts:
                flush_thinking()
            content_parts.append(fragment)
            Log.stdout(fragment, end="", flush=True)

        try:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def pump() -> None:
                try:
                    response.encoding = "utf-8"
                    for line in response.iter_lines(decode_unicode=True):
                        loop.call_soon_threadsafe(queue.put_nowait, line)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, _STREAM_EOF)

            threading.Thread(target=pump, daemon=True).start()

            while True:
                item = await queue.get()
                if item is _STREAM_EOF:
                    break
                if isinstance(item, Exception):
                    raise item
                line = item
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

                delta = choice.get("delta") or {}
                reasoning = (
                    delta.get("reasoning_content") or delta.get("reasoning") or ""
                )
                if reasoning:
                    append_thinking(reasoning)

                content = delta.get("content") or ""
                if content:
                    append_content(content)

                for tc in delta.get("tool_calls") or []:
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

        tail = reasoning_filter.flush()
        if tail:
            reasoning_parts.append(tail)
            reasoning_len += len(tail)
            Log.stderr(tail, end="", flush=True)
        tail = content_filter.flush()
        if tail:
            content_parts.append(tail)
            Log.stdout(tail, end="", flush=True)

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
