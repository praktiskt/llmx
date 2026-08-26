import asyncio
import http.server
import json
import os
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

from llmx.cache import Cache
from llmx.tools import Tools

FAKE_STDIO_ECHO = textwrap.dedent("""
    import sys, json
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg=json.loads(line)
        except Exception:
            continue
        mid=msg.get("id")
        method=msg.get("method")
        if method=="initialize":
            resp={"jsonrpc":"2.0","id":mid,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"demo","version":"1.0"}}}
            sys.stdout.write(json.dumps(resp)+"\\n"); sys.stdout.flush()
        elif method=="tools/list":
            resp={"jsonrpc":"2.0","id":mid,"result":{"tools":[{"name":"echo","description":"Echo","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}]}}
            sys.stdout.write(json.dumps(resp)+"\\n"); sys.stdout.flush()
        elif method=="tools/call":
            text=msg["params"]["arguments"].get("text","")
            if text=="large":
                big="x"*9000
                resp={"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":big}]}}
            elif text=="error":
                resp={"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"oops"}],"isError":True}}
            else:
                resp={"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"echo: " + text}]}}
            sys.stdout.write(json.dumps(resp)+"\\n"); sys.stdout.flush()
        elif method=="notifications/initialized":
            pass
""")

FAKE_STDIO_SLOW = textwrap.dedent("""
    import sys, json, time
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg=json.loads(line)
        except Exception:
            continue
        mid=msg.get("id")
        method=msg.get("method")
        if method=="initialize":
            resp={"jsonrpc":"2.0","id":mid,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"slow","version":"1.0"}}}
            sys.stdout.write(json.dumps(resp)+"\\n"); sys.stdout.flush()
        elif method=="tools/list":
            resp={"jsonrpc":"2.0","id":mid,"result":{"tools":[{"name":"slowtool","description":"slow","inputSchema":{"type":"object","properties":{}}}]}}
            sys.stdout.write(json.dumps(resp)+"\\n"); sys.stdout.flush()
        elif method=="tools/call":
            time.sleep(5)
            resp={"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"should not arrive"}]}}
            sys.stdout.write(json.dumps(resp)+"\\n"); sys.stdout.flush()
        elif method=="notifications/initialized":
            pass
""")

FAKE_STDIO_NULL = textwrap.dedent("""
    import sys, json
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg=json.loads(line)
        except Exception:
            continue
        mid=msg.get("id")
        method=msg.get("method")
        if method=="initialize":
            resp={"jsonrpc":"2.0","id":mid,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"nulltest","version":"1.0"}}}
            sys.stdout.write(json.dumps(resp)+"\\n"); sys.stdout.flush()
        elif method=="tools/list":
            resp={"jsonrpc":"2.0","id":mid,"result":{"tools":[{"name":"nulltool","description":"null","inputSchema":{"type":"object","properties":{}}}]}}
            sys.stdout.write(json.dumps(resp)+"\\n"); sys.stdout.flush()
        elif method=="tools/call":
            resp={"jsonrpc":"2.0","id":mid,"result":None}
            sys.stdout.write(json.dumps(resp)+"\\n"); sys.stdout.flush()
        elif method=="notifications/initialized":
            pass
""")


def _write_fake(path: str, content: str) -> None:
    Path(path).write_text(content)
    Path(path).chmod(0o755)


class _HttpHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass


class MCPTimeoutTest(unittest.TestCase):
    def setUp(self):
        self._saved = {
            k: os.environ.get(k)
            for k in ("LLM_MCP_SERVERS", "LLM_MCP_TIMEOUT", "LLM_TOOLS")
        }
        self._tmpdir = tempfile.TemporaryDirectory()
        import llmx.mcp as mcp_mod

        self.mcp_mod = mcp_mod
        mcp_mod._global_manager = None

    def tearDown(self):
        import llmx.mcp as mcp_mod

        try:
            asyncio.run(mcp_mod.close_mcp())
        except Exception:
            pass
        mcp_mod._global_manager = None
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmpdir.cleanup()

    def _start_stdio(self, content: str, server_name: str, timeout: str = "5") -> None:
        path = str(Path(self._tmpdir.name) / f"{server_name}.py")
        _write_fake(path, content)
        os.environ["LLM_MCP_SERVERS"] = json.dumps(
            {server_name: {"command": sys.executable, "args": [path]}}
        )
        os.environ["LLM_MCP_TIMEOUT"] = timeout

    def test_stdio_basic_echo(self):
        self._start_stdio(FAKE_STDIO_ECHO, "demo")
        import llmx.mcp as mcp_mod

        async def run():
            mgr = await mcp_mod.get_mcp_manager()
            self.assertIn(
                "demo__echo", [s["function"]["name"] for s in mgr.get_schemas()]
            )
            res = await Tools.execute("demo__echo", {"text": "hello"})
            self.assertEqual(res, "echo: hello")

        asyncio.run(run())

    def test_large_result_stored_as_memory(self):
        self._start_stdio(FAKE_STDIO_ECHO, "demo")

        async def run():
            await self.mcp_mod.get_mcp_manager()
            res = await Tools.execute("demo__echo", {"text": "large"})
            self.assertIn("memory://", res)
            self.assertIn("9000 chars", res)
            # Extract id and verify cache
            import re

            m = re.search(r"memory://([a-z0-9]{6})", res)
            self.assertIsNotNone(m)
            fid = m.group(1)
            cached = Cache.get(fid)
            self.assertIsNotNone(cached)
            # Cache wraps long lines, so stored length > 9000 with newlines
            self.assertGreater(len(cached), 8000)

        asyncio.run(run())

    def test_stdio_timeout_returns_error_not_null(self):
        self._start_stdio(FAKE_STDIO_SLOW, "slow", timeout="1")

        async def run():
            await self.mcp_mod.get_mcp_manager()
            res = await Tools.execute("slow__slowtool", {})
            self.assertNotEqual(res.strip(), "null")
            self.assertIn("timed out after", res)
            self.assertIn("Error", res)

        asyncio.run(run())

    def test_null_result_returns_error_not_null(self):
        self._start_stdio(FAKE_STDIO_NULL, "nulltest")

        async def run():
            await self.mcp_mod.get_mcp_manager()
            res = await Tools.execute("nulltest__nulltool", {})
            self.assertNotEqual(res.strip(), "null")
            self.assertIn("no content", res.lower())
            self.assertIn("error", res.lower())

        asyncio.run(run())

    def test_http_basic_echo(self):
        # spin simple http fake
        class Handler(_HttpHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    msg = json.loads(body)
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                method = msg.get("method")
                mid = msg.get("id")
                if method == "initialize":
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "serverInfo": {"name": "httpdemo", "version": "1.0"},
                        },
                    }
                elif method == "tools/list":
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "tools": [
                                {
                                    "name": "echo",
                                    "description": "Echo via http",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"text": {"type": "string"}},
                                        "required": ["text"],
                                    },
                                }
                            ]
                        },
                    }
                elif method == "tools/call":
                    text = msg["params"]["arguments"].get("text", "")
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "content": [{"type": "text", "text": f"http echo: {text}"}]
                        },
                    }
                elif method == "notifications/initialized":
                    self.send_response(202)
                    self.end_headers()
                    return
                else:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "error": {"code": -32601, "message": "not found"},
                    }
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        port = server.server_address[1]
        os.environ["LLM_MCP_SERVERS"] = json.dumps(
            {"httpdemo": {"url": f"http://127.0.0.1:{port}/mcp"}}
        )
        os.environ["LLM_MCP_TIMEOUT"] = "5"
        self.mcp_mod._global_manager = None

        async def run():
            mgr = await self.mcp_mod.get_mcp_manager()
            self.assertIn(
                "httpdemo__echo", [s["function"]["name"] for s in mgr.get_schemas()]
            )
            res = await Tools.execute("httpdemo__echo", {"text": "hello http"})
            self.assertEqual(res, "http echo: hello http")

        try:
            asyncio.run(run())
        finally:
            server.shutdown()
            server.server_close()

    def test_http_timeout_returns_error(self):
        class Handler(_HttpHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    msg = json.loads(body)
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                method = msg.get("method")
                mid = msg.get("id")
                if method == "initialize":
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "serverInfo": {"name": "httpslow", "version": "1.0"},
                        },
                    }
                    data = json.dumps(resp).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if method == "tools/list":
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "tools": [
                                {
                                    "name": "slowtool",
                                    "description": "slow",
                                    "inputSchema": {"type": "object", "properties": {}},
                                }
                            ]
                        },
                    }
                    data = json.dumps(resp).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if method == "tools/call":
                    time.sleep(5)
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {"content": [{"type": "text", "text": "slow"}]},
                    }
                    data = json.dumps(resp).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    try:
                        self.wfile.write(data)
                    except BrokenPipeError:
                        pass
                    return
                if method == "notifications/initialized":
                    self.send_response(202)
                    self.end_headers()
                    return
                self.send_response(404)
                self.end_headers()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        port = server.server_address[1]
        os.environ["LLM_MCP_SERVERS"] = json.dumps(
            {"httpslow": {"url": f"http://127.0.0.1:{port}/mcp"}}
        )
        os.environ["LLM_MCP_TIMEOUT"] = "1"
        self.mcp_mod._global_manager = None

        async def run():
            await self.mcp_mod.get_mcp_manager()
            res = await Tools.execute("httpslow__slowtool", {})
            self.assertNotEqual(res.strip(), "null")
            self.assertIn("timed out after", res)

        try:
            asyncio.run(run())
        finally:
            server.shutdown()
            server.server_close()

    def test_stateful_http_session_id(self):
        SESSION = "test-session-abc123"

        class Handler(_HttpHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    msg = json.loads(body)
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                method = msg.get("method")
                mid = msg.get("id")
                if method == "initialize":
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "serverInfo": {"name": "stateful", "version": "1.0"},
                        },
                    }
                    data = json.dumps(resp).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Mcp-Session-Id", SESSION)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                sid = self.headers.get("Mcp-Session-Id")
                if sid != SESSION:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "error": {
                            "code": -32000,
                            "message": f"missing session {sid!r}",
                        },
                    }
                    data = json.dumps(resp).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if method == "tools/list":
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "tools": [
                                {
                                    "name": "echo",
                                    "description": "Echo",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"text": {"type": "string"}},
                                        "required": ["text"],
                                    },
                                }
                            ]
                        },
                    }
                elif method == "tools/call":
                    text = msg["params"]["arguments"].get("text", "")
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {
                            "content": [
                                {"type": "text", "text": f"stateful echo: {text}"}
                            ]
                        },
                    }
                elif method == "notifications/initialized":
                    self.send_response(202)
                    self.end_headers()
                    return
                else:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "error": {"code": -32601, "message": "not found"},
                    }
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        port = server.server_address[1]
        os.environ["LLM_MCP_SERVERS"] = json.dumps(
            {"stateful": {"url": f"http://127.0.0.1:{port}/mcp"}}
        )
        os.environ["LLM_MCP_TIMEOUT"] = "5"
        self.mcp_mod._global_manager = None

        async def run():
            mgr = await self.mcp_mod.get_mcp_manager()
            self.assertEqual(mgr.clients["stateful"].session_id, SESSION)
            res = await Tools.execute("stateful__echo", {"text": "hello"})
            self.assertEqual(res, "stateful echo: hello")

        try:
            asyncio.run(run())
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
