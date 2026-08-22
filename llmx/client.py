import asyncio
import json
import os
import threading
from collections.abc import Awaitable

from .config import Config
from .output import Color, Log
from .tools import Tools, parse_tool_args
from .transport import AsyncHttp, LLMAPIError, Response, request_with_retries

_STREAM_EOF = object()

_TRUNCATION_MARKER = "[earlier messages truncated to fit the context window]"


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
            messages = truncate_history(messages)
            msg = LLMClient.body(messages=messages)

            def send(msg=msg) -> Awaitable[Response]:
                return AsyncHttp.post(
                    os.environ["LLM_HOST"],
                    headers=headers,
                    data=json.dumps(msg),
                    stream=Config.is_stream(),
                    timeout=Config.LLM_TIMEOUT,
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
        reasoning_len_flushed = 0

        def flush_thinking() -> None:
            nonlocal reasoning_len_flushed
            joined = "".join(reasoning_parts)
            if len(joined) == reasoning_len_flushed:
                return
            if Config.color_output_enabled():
                Log.stderr(Color.RESET, end="")
            if not joined[reasoning_len_flushed:].endswith("\n"):
                Log.stderr("")
            reasoning_len_flushed = len(joined)

        def append_thinking(fragment: str) -> None:
            nonlocal reasoning_len_flushed
            if not Config.thinking_enabled() or not fragment:
                return
            if len("".join(reasoning_parts)) == reasoning_len_flushed:
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
