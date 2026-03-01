#!/usr/bin/python3
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import secrets
import traceback
from html.parser import HTMLParser
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

import httpx
import mistune
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from .llm import Config, Tools

for name in ("uvicorn.error", "uvicorn.asgi", "asyncio"):
    logging.getLogger(name).addFilter(
        lambda r: not (r.exc_info and isinstance(r.exc_info[1], asyncio.CancelledError))
    )


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


CATPPUCCIN_MOCHA = """
:root {
    --base: #1e1e2e;
    --mantle: #181825;
    --crust: #11111b;
    --text: #cdd6f4;
    --subtext1: #a6adc8;
    --subtext0: #9399b2;
    --overlay2: #7f849c;
    --overlay1: #6c7086;
    --overlay0: #585b70;
    --surface0: #1e1e2e;
    --surface1: #313244;
    --surface2: #45475a;
    --blue: #89b4fa;
    --lavender: #b4befe;
    --mauve: #cba6f7;
    --red: #f38ba8;
    --maroon: #eba6ac;
    --peach: #fab387;
    --yellow: #f9e2af;
    --green: #a6e3a1;
    --teal: #94e2d5;
    --sky: #89dceb;
    --sapphire: #74c7ec;
    --font-mono: 'SF Mono', Monaco, monospace;
    --radius: 0.5rem;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: var(--base);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    height: 100dvh;
    display: flex;
    flex-direction: column;
}
#chat {
    flex: 1;
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
    padding: 0.5rem;
    padding-bottom: 4.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.message {
    max-width: 100%;
    padding: 0.5rem 0.75rem;
    border-radius: var(--radius);
    white-space: pre-wrap;
    word-break: break-word;
    contain: content;
    animation: messageIn 0.2s ease-out;
    scroll-margin-bottom: 60px;
}
@keyframes messageIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.message.user { align-self: flex-end; background: var(--blue); color: var(--crust); }
.message.assistant, .message.thinking { align-self: flex-start; background: var(--surface1); white-space: normal; line-height: 1.25; }
.message.assistant pre,
.message.assistant code { white-space: pre-wrap; }
.message.assistant pre { margin: 0.25rem 0; padding: 0.375rem; }
.message.assistant th, .message.assistant td { padding: 0.25rem 0.375rem; }
.message.thinking { border-left: 3px solid var(--mauve); font-style: italic; }
.message.thinking .thinking-header { display: block; color: var(--mauve); font-weight: 600; font-style: normal; font-family: var(--font-mono); font-size: 0.875rem; margin-bottom: 0.25rem; }
.message.system { align-self: flex-start; background: var(--surface2); color: var(--subtext1); font-style: italic; }
.message.tool-call, .message.tool-result {
    align-self: flex-start;
    font-family: var(--font-mono);
}
.message.tool-call { background: var(--surface1); border-left: 3px solid var(--peach); font-size: 0.875rem; }
.message.tool-result { background: var(--crust); border-left: 3px solid var(--teal); font-size: 0.8125rem; min-width: 200px; }
.tool-call .tool-name { color: var(--peach); font-weight: 600; }
.tool-call .tool-args { color: var(--subtext1); margin-top: 0.25rem; }
.result-toggle {
    color: var(--teal);
    cursor: pointer;
    font-size: 0.75rem;
    user-select: none;
}
.result-toggle.expanded { color: var(--mauve); }
.result-content.collapsed { display: none; }
.tool-result .result-label { color: var(--teal); }
#input-area {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--mantle);
    padding: 0.75rem;
    border-top: 1px solid var(--surface1);
    display: flex;
    gap: 0.5rem;
}
#input-area input {
    flex: 1;
    background: var(--surface1);
    border: 1px solid var(--surface2);
    border-radius: var(--radius);
    padding: 0.5rem 0.75rem;
    color: var(--text);
    font-size: 1rem;
    outline: none;
    cursor: pointer;
    touch-action: manipulation;
}
#input-area input:focus { border-color: var(--blue); }
#input-area button {
    background: var(--blue);
    color: var(--crust);
    border: none;
    border-radius: var(--radius);
    padding: 0.625rem 1.25rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
}
#input-area button:hover { background: var(--sapphire); }
#input-area button:disabled { background: var(--surface2); cursor: not-allowed; }
#input-area button#trash {
    background: var(--red);
    color: var(--crust);
    padding: 0.625rem 1rem;
}
#input-area button .spinner { display: none; width: 18px; height: 18px; border: 2px solid var(--text); border-top-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite; }
#input-area button.loading .spinner { display: inline-block; }
#input-area button.loading .btn-text { display: none; }
@keyframes spin { to { transform: rotate(360deg); } }
.typing { display: inline-block; }
.typing::after {
    content: '';
    animation: dots 1.5s infinite;
}
@keyframes dots {
    0%, 20% { content: '.'; }
    40% { content: '..'; }
    60%, 100% { content: '...'; }
}
a { color: var(--sky); }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--crust); }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--overlay1); }
code {
    background: var(--surface2);
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    font-family: var(--font-mono);
    font-size: 0.875em;
}
pre {
    background: var(--crust);
    padding: 0.75rem;
    border-radius: var(--radius);
    overflow-x: auto;
    margin: 0.5rem 0;
    white-space: pre-wrap;
    word-break: break-word;
}
pre code { background: none; padding: 0; }
img { max-width: 100%; height: auto; max-height: 80vh; object-fit: contain; cursor: pointer; }
#image-modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 1000; align-items: center; justify-content: center; }
#image-modal.active { display: flex; }
#image-modal img { max-width: 95vw; max-height: 95vh; object-fit: contain; }
#image-modal:active { cursor: zoom-out; }
iframe { max-width: 100%; width: 100%; aspect-ratio: 16 / 9; height: auto; }
table { border-collapse: collapse; width: 100%; margin: 0.25rem 0; }
th, td { border: 1px solid var(--surface2); padding: 0.25rem 0.5rem; text-align: left; min-width: 80px; overflow-wrap: break-word; word-break: break-word; }
th { background: var(--surface1); }
ul, ol { padding-left: 1rem; margin: 0.25rem 0; list-style-type: disc; }
li { margin: 0.125rem 0; display: list-item; }
p { margin: 0.125rem 0; }
blockquote { border-left: 3px solid var(--overlay0); padding-left: 0.5rem; margin: 0.25rem 0; color: var(--subtext1); font-style: italic; }
hr { border: none; border-top: 1px solid var(--surface2); margin: 0.5rem 0; }
h1, h2, h3, h4, h5, h6 { margin: 0.5rem 0 0.25rem; color: var(--text); }
h1 { font-size: 1.25rem; }
h2, h3 { font-size: 1rem; }
h2 { font-size: 1.1rem; }
"""

HTML_PAGE = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0, user-scalable=no, minimal-ui">
    <title>server</title>
    <style>{CATPPUCCIN_MOCHA}</style>
</head>
<body>
    <div id="chat"></div>
    <div id="input-area">
        <input type="text" id="msg" placeholder="Type your message..." autocomplete="off" autofocus>
        <button id="send"><span class="btn-text">Send</span><span class="spinner"></span></button>
        <button id="trash" title="Clear session">🗑️</button>
    </div>
    <div id="image-modal"></div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('msg');
        const sendBtn = document.getElementById('send');
        const trashBtn = document.getElementById('trash');
        const inputArea = document.getElementById('input-area');
        const imageModal = document.getElementById('image-modal');
        
        imageModal.addEventListener('click', () => imageModal.classList.remove('active'));
        document.addEventListener('keydown', e => {{ if (e.key === 'Escape') imageModal.classList.remove('active'); }});
        
        function addMessage(content, type = 'assistant') {{
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.innerHTML = content;
            chat.appendChild(div);
            div.querySelectorAll('img').forEach(img => {{
                img.addEventListener('click', e => {{
                    imageModal.innerHTML = '';
                    const fullImg = document.createElement('img');
                    fullImg.src = img.src;
                    imageModal.appendChild(fullImg);
                    imageModal.classList.add('active');
                }});
            }});
            const last = chat.lastElementChild;
            if (last) last.scrollIntoView({{block: 'nearest', behavior: 'auto'}});
        }}
        
        function setTyping() {{
            addMessage('<span class="typing">Thinking</span>', 'system');
        }}
        
        function clearTyping() {{
            chat.querySelectorAll('.message.system').forEach(el => el.remove());
        }}
        
        async function send() {{
            const msg = input.value.trim();
            if (!msg) return;
            
            addMessage(msg, 'user');
            input.value = '';
            sendBtn.disabled = true;
            sendBtn.classList.add('loading');
            
            setTyping();
            
            try {{
                const res = await fetch('/chat', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{message: msg}})
                }});
                
                if (!res.ok) {{
                    clearTyping();
                    addMessage('Error: ' + res.status, 'system');
                    sendBtn.disabled = false;
                    sendBtn.classList.remove('loading');
                    return;
                }}
                
                clearTyping();
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let done = false;
                
                while (!done) {{
                    const result = await reader.read();
                    done = result.done;
                    
                    if (result.value) {{
                        buffer += decoder.decode(result.value, {{stream: !done}});
                        const lines = buffer.split('\\n');
                        buffer = lines.pop() || '';
                        
                        for (const line of lines) {{
                            if (line.startsWith('data: ')) {{
                                const data = line.slice(6);
                                if (data === '[DONE]') {{
                                    done = true;
                                    break;
                                }}
                                try {{
                                    const event = JSON.parse(data);
                                    handleEvent(event);
                                }} catch (e) {{
                                }}
                            }}
                        }}
                    }}
                }}
            }} catch (e) {{
                clearTyping();
                addMessage('Error: ' + e.message, 'system');
            }}
            
            sendBtn.disabled = false;
            sendBtn.classList.remove('loading');
        }}
        
        function handleEvent(event) {{
            const type = event.type;
            const content = event.content || '';
            
            if (type === 'thinking') {{
                addMessage(content, 'thinking');
            }} else if (type === 'tool_call') {{
                addMessage(content, 'tool-call');
            }} else if (type === 'message') {{
                addMessage(content, 'assistant');
            }}
        }}
        
        sendBtn.addEventListener('click', send);
        trashBtn.addEventListener('click', () => {{
            if (confirm('Are you sure?')) {{
                window.location.href = '/';
            }}
        }});
        input.addEventListener('keydown', e => {{
            if (e.key === 'Enter' && !e.shiftKey) {{
                e.preventDefault();
                send();
            }}
        }});
        
        window.addEventListener('load', () => input.focus());
        inputArea.addEventListener('click', () => input.focus());
        input.addEventListener('touchstart', () => input.focus());
    </script>
</body>
</html>"""


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
    return secrets.token_hex(8)


def get_session(session_id: str | None = None) -> tuple[Session, str]:
    global sessions
    if session_id and session_id in sessions:
        return sessions[session_id], session_id
    new_session = Session()
    new_id = generate_session_id()
    sessions[new_id] = new_session
    return new_session, new_id


def format_tool_call(tool_name: str, args: dict, result: str | None = None) -> str:
    args_str = json.dumps(args, indent=2)
    escaped_args = html.escape(args_str)

    if result:
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

    # Check for code fences BEFORE markdown processing
    has_backticks = "```" in content

    # Use escape=True if backticks present (show code), else escape=False (render HTML)
    md = (
        mistune.create_markdown(
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
        if has_backticks
        else _markdown
    )
    content = md(content)  # type: ignore[assignment]

    content = content.replace("<a href=", '<a target="_blank" href=')
    content = re.sub(r"(<table>)", r"<div style='overflow-x:auto'>\1", content)
    content = re.sub(r"(</table>)", r"\1</div>", content)
    content = M.minify_html(content, escape_code=has_backticks)
    return content.rstrip()


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return HTMLResponse(
        content=HTML_PAGE,
        headers={"Set-Cookie": "session=; Path=/; Max-Age=0"},
    )


@app.post("/chat")
async def post_chat(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
    except Exception:
        return {"error": "Invalid JSON"}

    session_id = get_session_id_from_cookie(request)
    session, new_session_id = get_session(session_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        session.messages.append({"role": "user", "content": user_message})

        try:
            async for event in stream_response(session):
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
            "Set-Cookie": f"session={new_session_id}; Path=/",
        },
    )


async def execute_with_retry(tool_call: dict, max_retries: int = 5) -> tuple[str, str]:
    tool_id = tool_call.get("id", "")
    func = tool_call.get("function", {})
    tool_name = func.get("name", "unknown")

    for attempt in range(max_retries):
        try:
            return Tools.execute_wrapper(tool_call)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Tool '{tool_name}' failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}"
                )
            else:
                logger.error(
                    f"Tool '{tool_name}' failed after {max_retries} attempts: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                )
                return tool_id, f"Error: {str(e)}"

    return tool_id, "Error: Max retries exceeded"


async def stream_response(session: Session) -> AsyncGenerator[str, None]:
    headers = {
        "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    max_iterations = 100

    async with httpx.AsyncClient(timeout=120) as client:
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

                yield f"data: {json.dumps({'type': 'message', 'content': f'API Error {response.status_code}: {response.text[:200]}'})}\n\n"
                return

            if response is None or response.status_code != 200:
                yield f"data: {json.dumps({'type': 'message', 'content': 'API Error: Max retries exceeded'})}\n\n"
                return

            data = response.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})

            reasoning = (
                message.get("reasoning_content") or message.get("reasoning") or ""
            )
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
                    session.messages.append({"role": "assistant", "content": content})
                return

            session.messages.append(message)

            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                tool_name = func.get("name", "unknown")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                tool_id, result = await execute_with_retry(tool_call)
                yield f"data: {json.dumps({'type': 'tool_call', 'content': format_tool_call(tool_name, args, result)})}\n\n"

                session.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result,
                    }
                )


def main():
    bind = os.environ.get("LLM_BIND_ADDRESS", "0.0.0.0")
    port = int(os.environ.get("LLM_SERVER_PORT", "8080"))
    uvicorn.run(app, host=bind, port=port, timeout_graceful_shutdown=0)


if __name__ == "__main__":
    main()
