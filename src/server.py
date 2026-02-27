#!/usr/bin/python3
from __future__ import annotations

import html
import json
import os
import re
import secrets
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

import mistune

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm import Config, Tools


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
    --surface2: #45475a;
    --surface1: #313244;
    --surface0: #313244;
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
    --pink: #f5c2e7;
    --flamingo: #f2cdcd;
    --rosewater: #f5e0dc;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
:root { --vh: 1vh; }
body {
    background: var(--base);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    height: calc(var(--vh, 1vh) * 100);
    display: flex;
    flex-direction: column;
}
#chat {
    flex: 1;
    overflow-y: auto;
    padding: 0.5rem;
    padding-bottom: 6rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.message {
    max-width: 90%;
    padding: 0.5rem 0.75rem;
    border-radius: 0.5rem;
    white-space: pre-wrap;
    word-break: break-word;
}
.message.user { align-self: flex-end; background: var(--blue); color: var(--crust); }
.message.assistant, .message.thinking { align-self: flex-start; background: var(--surface1); }
.message.thinking { border-left: 3px solid var(--mauve); }
.message.system { align-self: flex-start; background: var(--surface2); color: var(--subtext1); font-style: italic; }
.message.tool-call, .message.tool-result {
    align-self: flex-start;
    font-family: 'SF Mono', Monaco, monospace;
    white-space: pre-wrap;
    word-break: break-word;
}
.message.tool-call { background: var(--surface0); border-left: 3px solid var(--peach); font-size: 0.875rem; }
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
    border-radius: 0.5rem;
    padding: 0.5rem 0.75rem;
    color: var(--text);
    font-size: 1rem;
    outline: none;
    cursor: pointer;
}
#input-area input:focus { border-color: var(--blue); }
#input-area button {
    background: var(--blue);
    color: var(--crust);
    border: none;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
}
#input-area button:hover { background: var(--sapphire); }
#input-area button:disabled { background: var(--surface2); cursor: not-allowed; }
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
    font-family: 'SF Mono', Monaco, monospace;
    font-size: 0.875em;
}
pre {
    background: var(--crust);
    padding: 0.75rem;
    border-radius: 0.5rem;
    overflow-x: auto;
    margin: 0.5rem 0;
    white-space: pre-wrap;
    word-break: break-word;
}
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.25rem 0; }
th, td { border: 1px solid var(--surface2); padding: 0.25rem 0.5rem; text-align: left; min-width: 80px; overflow-wrap: break-word; word-break: break-word; }
th { background: var(--surface1); }
ul, ol { padding-left: 1rem; margin: 0.25rem 0; }
li { margin: 0.125rem 0; }
blockquote { border-left: 3px solid var(--overlay0); padding-left: 0.5rem; margin: 0.25rem 0; color: var(--subtext1); font-style: italic; }
hr { border: none; border-top: 1px solid var(--surface2); margin: 0.5rem 0; }
h1, h2, h3, h4, h5, h6 { margin: 0.5rem 0 0.25rem; color: var(--text); }
h1 { font-size: 1.25rem; }
h2 { font-size: 1.1rem; }
h3 { font-size: 1rem; }
.message.assistant { line-height: 1.25; }
.message.assistant p { margin: 0; }
.message.assistant li p { margin: 0; }
.message.assistant pre { margin: 0.25rem 0; padding: 0.375rem; }
.message.assistant table { margin: 0.25rem 0; }
.message.assistant th, .message.assistant td { padding: 0.25rem 0.375rem; }
.message.assistant ul, .message.assistant ol { padding-left: 1rem; margin: 0.25rem 0; }
.message.assistant li { margin: 0.125rem 0; }
.message.assistant blockquote { margin: 0.25rem 0; padding-left: 0.5rem; }
"""

HTML_PAGE = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0, user-scalable=no, minimal-ui">
    <title>LLM Chat</title>
    <style>{CATPPUCCIN_MOCHA}</style>
</head>
<body>
    <div id="chat"></div>
    <div id="input-area">
        <input type="text" id="msg" placeholder="Type your message..." autocomplete="off" autofocus>
        <button id="send"><span class="btn-text">Send</span><span class="spinner"></span></button>
    </div>
    <script>
        function setVh() {{
            document.documentElement.style.setProperty('--vh', (window.innerHeight * 0.01) + 'px');
        }}
        setVh();
        window.addEventListener('resize', setVh);
        if (visualViewport) {{
            visualViewport.addEventListener('resize', () => chat.scrollTop = chat.scrollHeight);
        }}
        
        const chat = document.getElementById('chat');
        const input = document.getElementById('msg');
        const sendBtn = document.getElementById('send');
        
        function addMessage(content, type = 'assistant') {{
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.innerHTML = content;
            chat.appendChild(div);
            setTimeout(() => chat.scrollTop = chat.scrollHeight, 100);
        }}
        
        function setTyping() {{
            addMessage('<span class="typing">Thinking</span>', 'system');
        }}
        
        function clearTyping() {{
            const typing = chat.querySelector('.typing');
            if (typing) typing.parentElement.remove();
        }}
        
        async function send() {{
            const msg = input.value.trim();
            if (!msg) return;
            
            input.blur();
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
                                    console.error('Parse error:', e);
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
            input.blur();
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
        input.addEventListener('keydown', e => {{
            if (e.key === 'Enter' && !e.shiftKey) {{
                e.preventDefault();
                send();
            }}
        }});
        
        window.addEventListener('load', () => input.focus());
        document.getElementById('input-area').addEventListener('click', () => input.focus());
        input.addEventListener('touchstart', () => input.focus());
    </script>
</body>
</html>"""


class Session:
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": os.environ.get(
                    "LLM_SYSTEM_PROMPT", Config.get_system_prompt()
                ),
            },
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


def format_message(content: str) -> str:
    md = mistune.create_markdown(plugins=["strikethrough", "table"])
    content = md(content)
    content = content.replace("<a href=", '<a target="_blank" href=')
    content = re.sub(r"<li>\s*<p>", "<li>", content)
    content = re.sub(r"</p>\s*</li>", "</li>", content)
    content = re.sub(r"</li>\s+<li>", "</li><li>", content)
    content = re.sub(r"<(ul|ol)>\s+", r"<\1>", content)
    content = re.sub(r"\s+</(ul|ol)>", r"</\1>", content)
    content = re.sub(r"(<table>)", r"<div style='overflow-x:auto'>\1", content)
    content = re.sub(r"(</table>)", r"\1</div>", content)
    return content


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def get_session_id(self) -> str | None:
        cookie = self.headers.get("Cookie", "")
        for part in cookie.split(";"):
            part = part.strip()
            if part.startswith("session="):
                return part[8:]
        return None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(body)
            user_message = data.get("message", "")
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        session_id = self.get_session_id()
        session, new_session_id = get_session(session_id)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Set-Cookie", f"session={new_session_id}; Path=/")
        self.end_headers()

        session.messages.append({"role": "user", "content": user_message})

        try:
            self._stream_response(session)
        except Exception as e:
            self._send_event("message", f"Error: {str(e)}")

        self._send_event(None, "[DONE]")

    def _send_event(self, event_type: str | None, data: str):
        if event_type:
            event_json = json.dumps({"type": event_type, "content": data})
            self.wfile.write(f"data: {event_json}\n\n".encode("utf-8"))
        else:
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_response(self, session: Session):
        import requests

        headers = {
            "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        max_iterations = 100

        for iteration in range(max_iterations):
            payload = {
                "messages": session.messages,
                "model": os.environ["LLM_MODEL"],
                "temperature": float(os.environ.get("LLM_TEMPERATURE", 0.1)),
                "stream": False,
            }

            if Config.tools_enabled():
                payload["tools"] = Tools.SCHEMA

            response = requests.post(
                os.environ["LLM_HOST"],
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code != 200:
                self._send_event(
                    "message",
                    f"API Error {response.status_code}: {response.text[:200]}",
                )
                return

            data = response.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})

            reasoning = (
                message.get("reasoning_content") or message.get("reasoning") or ""
            )
            if reasoning:
                self._send_event("thinking", html.escape(reasoning))

            tool_calls = message.get("tool_calls", [])
            if not tool_calls or not Config.tools_enabled():
                content = message.get("content", "")
                if content:
                    self._send_event("message", format_message(content))
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

                tool_id, result = Tools.execute_wrapper(tool_call)
                self._send_event("tool_call", format_tool_call(tool_name, args, result))

                session.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result,
                    }
                )


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    bind = os.environ.get("LLM_BIND_ADDRESS", "0.0.0.0")
    port = int(os.environ.get("LLM_SERVER_PORT", "8080"))

    print(f"Starting server at http://{bind}:{port}")
    server = ThreadingHTTPServer((bind, port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
