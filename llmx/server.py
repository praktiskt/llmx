#!/usr/bin/python3
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import traceback
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import AsyncGenerator

STATIC_DIR = Path(__file__).parent / "static"

logger = logging.getLogger(__name__)

import httpx
import mistune
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .llm import Config, Tools, UrlRedirect


def resolve_redirects(text: str) -> str:
    def replace_redirect(match):
        url = match.group(0)
        resolved = UrlRedirect.resolve(url)
        return resolved if resolved else url

    return re.sub(
        r"https://redirect/[a-f0-9]{8}(\.png|\.jpg|\.jpeg|\.gif)?",
        replace_redirect,
        text,
    )


for name in ("uvicorn.error", "uvicorn.asgi", "asyncio"):
    logging.getLogger(name).addFilter(
        lambda r: not (r.exc_info and isinstance(r.exc_info[1], asyncio.CancelledError))
    )


_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=120,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _http_client


class M(HTMLParser):
    o: list[str]
    p: bool

    def __init__(self, escape_code: bool = False):
        super().__init__()
        self.o = []
        self.p = True
        self.escape_code = escape_code
        self.in_code = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.escape_code and tag == "pre":
            self.in_code = True
        a = "".join(f' {k}="{v}"' for k, v in attrs if v)
        self.o.append(f"<{tag}{a}>")
        self.p = False

    def handle_endtag(self, tag: str) -> None:
        if self.escape_code and tag == "pre":
            self.in_code = False
        self.o.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self.in_code:
            data = html.escape(data)
        self.o.append(data)
        self.p = False

    def handle_entityref(self, name: str) -> None:
        self.o.append(f"&{name};")

    @staticmethod
    def minify_html(h: str, escape_code: bool = False) -> str:
        m = M(escape_code=escape_code)
        m.feed(h)
        return re.sub(r">\s+<", "><", "".join(m.o)).strip()


class Session:
    def __init__(self):
        self._system_prompt = os.environ.get(
            "LLM_SYSTEM_PROMPT", Config.get_system_prompt()
        )
        self._system_prompt += """\n\n
Extra capabilities:
* You can render HTML when asked for custom styling.
    * No custom HTML components, only basic HTML without comments.
    * Page is in dark mode, using catppuccin-mocha CSS variables: --base, --mantle, --crust, --text, --subtext1, --surface0, --surface1, --surface2, --overlay0, --blue, --lavender, --mauve, --red, --peach, --yellow, --green, --teal, --sky, --sapphire.
    * Use inline styles like style="color: var(--blue)" or style="background: var(--surface1)". Do not use global style changes.
    * Response must be contained within a single div.
    * If you include remote content (e.g., images), use the fetch tool to verify the content exists.
    * If tables are included, make sure they are horizontably scrollable.
* You can fetch recent news from news.praktiskt.dev/ with query params:
    * keywords=<comma,separated,list>
    * since=<1w, 1d, 1h, 2h, 60m and so on, set to whatever you need.>
    * format=markdown
    * Use fetch on URLs from the site to get more details and images when asked.
* When asked about news, write a short news article.
    * Focus on mobile-first layout.
    * If images are not present in the content you have, research the story to locate relevant images.
    * Clearly outline the timeline of events on developing stories.
"""
        self.messages = [
            {"role": "system", "content": self._system_prompt},
        ]


sessions: dict[str, Session] = {}


def generate_session_id() -> str:
    return str(uuid.uuid4())


def get_session(session_id: str | None = None) -> tuple[Session, str]:
    global sessions
    if session_id and session_id in sessions:
        return sessions[session_id], session_id
    new_session = Session()
    new_id = generate_session_id()
    sessions[new_id] = new_session
    return new_session, new_id


def format_tool_call(tool_name: str, args: dict, result: str | None = None) -> str:
    if tool_name == "fetch" and "urls" in args:
        args = {**args, "urls": [UrlRedirect.resolve(u) or u for u in args["urls"]]}
    args_str = json.dumps(args, indent=2)
    escaped_args = html.escape(args_str)

    if result:
        result = resolve_redirects(result)
        truncated = result[:500] + ("..." if len(result) > 500 else "")
        escaped_result = html.escape(truncated)
        result_html = f"""<span class="result-toggle" onclick="this.classList.toggle('expanded'); const c = this.nextElementSibling; c.classList.toggle('collapsed'); this.textContent = this.classList.contains('expanded') ? '[▲ result]' : '[▼ result]'">[▼ result]</span><pre class="result-content collapsed"><code>{escaped_result}</code></pre>"""
    else:
        result_html = ""

    return f'<span class="tool-name">{html.escape(tool_name)}</span>\n<div class="tool-args"><pre><code>{escaped_args}</code></pre></div>{result_html}'


_markdown = mistune.create_markdown(
    escape=False,
    plugins=[
        "strikethrough",
        "footnotes",
        "table",
        "url",
        "task_lists",
        "def_list",
        "abbr",
        "mark",
        "insert",
        "superscript",
        "subscript",
        "math",
        "ruby",
        "spoiler",
    ],
)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_session_id_from_cookie(request: Request) -> str | None:
    cookie = request.headers.get("Cookie", "")
    for part in cookie.split(";"):
        part = part.strip()
        if part.startswith("session="):
            return part[8:]
    return None


def format_message(content: str) -> str:
    content = content.rstrip()
    content = re.sub(r"•\s*", "- ", content)
    content = resolve_redirects(content)

    has_backticks = "```" in content
    md = mistune.create_markdown(
        escape=has_backticks,
        plugins=[
            "strikethrough",
            "footnotes",
            "table",
            "url",
            "task_lists",
            "def_list",
            "abbr",
            "mark",
            "insert",
            "superscript",
            "subscript",
            "math",
            "ruby",
            "spoiler",
        ],
    )
    content = md(content)  # type: ignore[assignment]
    content = content.replace("<a href=", '<a target="_blank" href=')
    content = re.sub(r"(<table>)", r"<div style='overflow-x:auto'>\1", content)
    content = re.sub(r"(</table>)", r"\1</div>", content)
    content = M.minify_html(content, escape_code=has_backticks)
    return content.rstrip()


@app.get("/", response_class=HTMLResponse)
async def get_index():
    session, session_id = get_session(None)
    return RedirectResponse(f"/{session_id}")


@app.get("/{session_id}", response_class=HTMLResponse)
async def get_session_page(session_id: str):
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/session/{session_id}/messages")
async def get_session_messages(session_id: str):
    session, _ = get_session(session_id)
    return {"messages": session.messages}


@app.post("/chat")
async def post_chat(request: Request, session_id: str | None = None):
    try:
        data = await request.json()
        user_message = data.get("message", "")
    except Exception:
        return {"error": "Invalid JSON"}

    if not session_id:
        session_id = get_session_id_from_cookie(request)
    session, new_session_id = get_session(session_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        session.messages.append({"role": "user", "content": user_message})

        try:
            async for event in stream_response(session, request):
                yield event
        except Exception as e:
            yield f"data: {json.dumps({'type': 'message', 'content': f'Error: {str(e)}'})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def execute_with_retry(tool_call: dict, max_retries: int = 5) -> tuple[str, str]:
    tool_id = tool_call.get("id", "")
    func = tool_call.get("function", {})
    tool_name = func.get("name", "unknown")

    for attempt in range(max_retries):
        try:
            return await Tools.execute_wrapper(tool_call)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Tool '{tool_name}' failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}"
                )
            else:
                logger.error(
                    f"Tool '{tool_name}' failed after {max_retries} attempts: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                )
                return (
                    tool_id,
                    f"Error: {tool_name} failed after {max_retries} attempts: {type(e).__name__}: {str(e)}",
                )

    return tool_id, "Error: Max retries exceeded"


async def stream_response(
    session: Session, request: Request
) -> AsyncGenerator[str, None]:
    headers = {
        "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    max_iterations = 100

    client = get_http_client()
    for _ in range(max_iterations):
        payload = {
            "messages": session.messages,
            "model": os.environ["LLM_MODEL"],
            "temperature": float(os.environ.get("LLM_TEMPERATURE", 0.1)),
            "stream": False,
        }

        if Config.tools_enabled():
            payload["tools"] = Tools.SCHEMA

        response = None
        for attempt in range(5):
            response = await client.post(
                os.environ["LLM_HOST"],
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                break

            if response.status_code == 429:
                logger.warning(
                    f"API rate limited (attempt {attempt + 1}/5), retrying..."
                )
                continue

            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "message",
                        "content": "API Error "
                        + str(response.status_code)
                        + ": "
                        + response.text[:200],
                    }
                )
                + "\n\n"
            )
            return

        if response is None or response.status_code != 200:
            yield f"data: {json.dumps({'type': 'message', 'content': 'API Error: Max retries exceeded'})}\n\n"
            return

        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if reasoning:
            escaped_reasoning = html.escape(reasoning)
            content_val = (
                f"<span class='thinking-header'>thinking</span>{escaped_reasoning}"
            )
            yield f"data: {json.dumps({'type': 'thinking', 'content': content_val})}\n\n"

        tool_calls = message.get("tool_calls", [])
        if not tool_calls or not Config.tools_enabled():
            content = message.get("content", "")
            if content:
                yield f"data: {json.dumps({'type': 'message', 'content': format_message(content)})}\n\n"
                session.messages.append(
                    {"role": "assistant", "content": content, "reasoning": reasoning}
                )
            return

        session.messages.append({**message, "reasoning": reasoning})

        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            tool_name = func.get("name", "unknown")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            tool_id, result = await execute_with_retry(tool_call)
            yield f"data: {json.dumps({'type': 'tool_call', 'content': format_tool_call(tool_name, args, result)})}\n\n"

            if await request.is_disconnected():
                return

            session.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                }
            )

        if await request.is_disconnected():
            return


def main():
    bind = os.environ.get("LLM_BIND_ADDRESS", "0.0.0.0")
    port = int(os.environ.get("LLM_SERVER_PORT", "8080"))
    uvicorn.run(
        "llmx.server:app",
        host=bind,
        port=port,
        timeout_graceful_shutdown=0,
    )


if __name__ == "__main__":
    main()
