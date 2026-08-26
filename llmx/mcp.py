from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "llmx", "version": "0.1.0"}

# Tool names must be <=64 chars, alphanumeric + _- for OpenAI.
# Prefixed name = server__tool must satisfy that.
_PREFIX_RE = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")


class MCPError(RuntimeError):
    pass


def _sanitize_tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]


def _mcp_headers(extra: dict | None) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if extra:
        headers.update(extra)
    return headers


def _extract_content(result: dict | None) -> tuple[str, bool]:
    """Return (text, is_error) from tools/call result."""
    if result is None:
        return ("MCP tool returned no content (timed out or empty result)", True)
    if not isinstance(result, dict):
        return (json.dumps(result, ensure_ascii=False), False)
    is_error = bool(result.get("isError"))
    content = result.get("content", result)
    if content is None:
        return ("MCP tool returned no content (timed out or empty result)", True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            t = item.get("type")
            if t == "text":
                parts.append(item.get("text", ""))
            elif t == "image":
                parts.append(f"[image: {item.get('mimeType', '')}]")
            elif t == "resource":
                res = item.get("resource", {})
                parts.append(res.get("text", json.dumps(res, ensure_ascii=False)))
            else:
                parts.append(item.get("text", json.dumps(item, ensure_ascii=False)))
        return ("\n".join(p for p in parts if p), is_error)
    if isinstance(content, str):
        return (content, is_error)
    return (json.dumps(content, ensure_ascii=False), is_error)


def _parse_sse_text(text: str) -> dict | None:
    """Try to extract JSON-RPC object from SSE or plain JSON text."""
    text = text.strip()
    if not text:
        return None
    # Plain JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except Exception:
            pass
    # SSE: look for data: lines
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                return json.loads(payload)
            except Exception:
                continue
    return None


class _MCPStdioClient:
    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.proc: asyncio.subprocess.Process | None = None
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        command = self.cfg.get("command")
        if not command:
            raise MCPError(f"server '{self.name}' missing command")
        args = self.cfg.get("args", [])
        if args is None:
            args = []
        if not isinstance(args, list):
            raise MCPError(f"server '{self.name}' args must be a list")
        env = os.environ.copy()
        extra_env = self.cfg.get("env")
        if isinstance(extra_env, dict):
            for k, v in extra_env.items():
                env[str(k)] = str(v)
        cwd = self.cfg.get("cwd")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                command,
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )
        except FileNotFoundError as e:
            raise MCPError(f"server '{self.name}' failed to start: {e}") from e
        except Exception as e:
            raise MCPError(f"server '{self.name}' failed to start: {e}") from e
        self._reader_task = asyncio.create_task(self._reader_loop())
        self._stderr_task = asyncio.create_task(self._stderr_loop())
        timeout = Config.mcp_timeout()
        await self._handshake(timeout)

    async def _handshake(self, timeout: float) -> None:
        result = await self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": _CLIENT_INFO,
            },
            timeout=timeout,
        )
        # Server may return protocolVersion; ignore mismatch.
        logger.debug("MCP server '%s' initialized: %s", self.name, result)
        # Notifications/initialized has no id — fire-and-forget.
        try:
            await self._notify("notifications/initialized", {})
        except Exception as e:
            logger.debug(
                "MCP server '%s' initialized notification failed: %s", self.name, e
            )

    async def _reader_loop(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                logger.debug("MCP server '%s' non-JSON line: %r", self.name, line[:200])
                continue
            msg_id = msg.get("id")
            if msg_id is not None:
                fut = self._pending.pop(msg_id, None)
                if fut and not fut.done():
                    if "error" in msg:
                        fut.set_exception(MCPError(msg["error"]))
                    else:
                        fut.set_result(msg.get("result"))
            else:
                # Notification — ignore.
                pass
        # Process ended — fail pending.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPError(f"server '{self.name}' closed"))
        self._pending.clear()

    async def _stderr_loop(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            try:
                text = line.decode(errors="replace").rstrip()
            except Exception:
                text = repr(line[:200])
            if text:
                logger.debug("MCP server '%s' stderr: %s", self.name, text)

    async def _notify(self, method: str, params: dict) -> None:
        assert self.proc and self.proc.stdin
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        data = (json.dumps(msg) + "\n").encode()
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    async def _request(
        self, method: str, params: dict, timeout: float | None = None
    ) -> Any:
        if timeout is None:
            timeout = Config.mcp_timeout()
        assert self.proc and self.proc.stdin
        async with self._lock:
            self._id += 1
            msg_id = self._id
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            self._pending[msg_id] = fut
            msg = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
            data = (json.dumps(msg) + "\n").encode()
            try:
                self.proc.stdin.write(data)
                await self.proc.stdin.drain()
            except Exception as e:
                self._pending.pop(msg_id, None)
                raise MCPError(f"server '{self.name}' write failed: {e}") from e
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError as e:
            self._pending.pop(msg_id, None)
            raise MCPError(
                f"server '{self.name}' request '{method}' timed out after {timeout}s"
            ) from e

    async def list_tools(self) -> list[dict]:
        result = await self._request("tools/list", {})
        if not isinstance(result, dict):
            return []
        return result.get("tools", []) or []

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        return await self._request(
            "tools/call", {"name": tool_name, "arguments": arguments}
        )

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                # Give process a moment to exit gracefully.
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=2.0)
                except TimeoutError:
                    self.proc.kill()
                    await self.proc.wait()
            except Exception:
                pass
            self.proc = None


class _MCPHttpClient:
    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.url: str = cfg.get("url", "")
        self.headers: dict = cfg.get("headers", {}) or {}
        self.session_id: str | None = None
        self._id = 0
        self._lock = asyncio.Lock()

    def _headers_with_session(self) -> dict:
        h = _mcp_headers(self.headers)
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        return h

    def _capture_session(self, resp) -> None:
        # Headers are lowercased in transport.Response.headers
        try:
            sid = resp.headers.get("mcp-session-id")
        except Exception:
            sid = None
        if sid:
            if sid != self.session_id:
                logger.debug("MCP HTTP server '%s' session id: %s", self.name, sid)
            self.session_id = sid

    async def start(self) -> None:
        if not self.url:
            raise MCPError(f"server '{self.name}' missing url")
        timeout = Config.mcp_timeout()
        # Handshake — two RPCs over HTTP.
        result = await self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": _CLIENT_INFO,
            },
            timeout=timeout,
        )
        logger.debug("MCP HTTP server '%s' initialized: %s", self.name, result)
        # notifications/initialized — POST without id, ignore response.
        try:
            await self._notify("notifications/initialized", {})
        except Exception as e:
            logger.debug(
                "MCP HTTP server '%s' initialized notification failed: %s", self.name, e
            )

    async def _notify(self, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        from .transport import AsyncHttp

        try:
            resp = await AsyncHttp.post(
                self.url,
                json=payload,
                headers=self._headers_with_session(),
                timeout=Config.mcp_timeout(),
                reuse=False,
            )
            self._capture_session(resp)
        except Exception:
            pass

    async def _request(
        self, method: str, params: dict, timeout: float | None = None
    ) -> Any:
        if timeout is None:
            timeout = Config.mcp_timeout()
        from .transport import AsyncHttp

        async with self._lock:
            self._id += 1
            msg_id = self._id
        payload = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        try:
            resp = await asyncio.wait_for(
                AsyncHttp.post(
                    self.url,
                    json=payload,
                    headers=self._headers_with_session(),
                    timeout=timeout,
                    reuse=False,
                ),
                timeout=timeout + 2,
            )
        except TimeoutError as e:
            raise MCPError(
                f"server '{self.name}' request '{method}' timed out after {timeout}s"
            ) from e
        except Exception as e:
            # http.client socket timeout surfaces as TimeoutError with "timed out" text
            if isinstance(e, TimeoutError) or "timed out" in str(e).lower():
                raise MCPError(
                    f"server '{self.name}' request '{method}' timed out after {timeout}s"
                ) from e
            raise
        self._capture_session(resp)
        if resp.status_code >= 400:
            raise MCPError(
                f"server '{self.name}' HTTP {resp.status_code}: {resp.text[:500]}"
            )
        # Try plain JSON, then SSE.
        text = resp.text
        parsed = _parse_sse_text(text)
        if parsed is None:
            raise MCPError(f"server '{self.name}' invalid response: {text[:300]}")
        if "error" in parsed:
            raise MCPError(parsed["error"])
        # Match id if present.
        if parsed.get("id") not in (None, msg_id):
            # Some servers echo different id; accept any result for single-flight HTTP.
            pass
        return parsed.get("result")

    async def list_tools(self) -> list[dict]:
        result = await self._request("tools/list", {})
        if not isinstance(result, dict):
            return []
        return result.get("tools", []) or []

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        return await self._request(
            "tools/call", {"name": tool_name, "arguments": arguments}
        )

    async def close(self) -> None:
        pass


class MCPManager:
    def __init__(self, servers_cfg: dict | None = None):
        if servers_cfg is None:
            servers_cfg = Config.mcp_servers() or {}
        self.servers_cfg = servers_cfg
        self.clients: dict[str, _MCPStdioClient | _MCPHttpClient] = {}
        self.tool_map: dict[
            str, tuple[str, str]
        ] = {}  # prefixed_name -> (server_name, original_tool_name)
        self.schemas: list[dict] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for server_name, cfg in self.servers_cfg.items():
            if not isinstance(cfg, dict):
                logger.warning(
                    "MCP server '%s' config must be an object, skipping", server_name
                )
                continue
            if not _PREFIX_RE.match(server_name):
                logger.warning(
                    "MCP server name '%s' invalid (use [a-zA-Z0-9_-]{1,32}), skipping",
                    server_name,
                )
                continue
            has_command = "command" in cfg
            has_url = "url" in cfg
            if has_command and has_url:
                logger.warning(
                    "MCP server '%s' has both command and url, skipping", server_name
                )
                continue
            if not has_command and not has_url:
                logger.warning(
                    "MCP server '%s' missing command or url, skipping", server_name
                )
                continue
            try:
                client: _MCPStdioClient | _MCPHttpClient
                if has_command:
                    client = _MCPStdioClient(server_name, cfg)
                else:
                    client = _MCPHttpClient(server_name, cfg)
                await asyncio.wait_for(client.start(), timeout=Config.mcp_timeout() + 5)
                tools = await asyncio.wait_for(
                    client.list_tools(), timeout=Config.mcp_timeout()
                )
                self.clients[server_name] = client
                for tool in tools:
                    if not isinstance(tool, dict):
                        continue
                    raw_name = tool.get("name", "")
                    if not raw_name:
                        continue
                    safe_name = _sanitize_tool_name(raw_name)
                    prefixed = f"{server_name}__{safe_name}"
                    if len(prefixed) > 64:
                        prefixed = prefixed[:64]
                    if prefixed in self.tool_map:
                        logger.warning(
                            "MCP tool name collision '%s', skipping", prefixed
                        )
                        continue
                    self.tool_map[prefixed] = (server_name, raw_name)
                    input_schema = tool.get("inputSchema") or {
                        "type": "object",
                        "properties": {},
                    }
                    # Ensure parameters is an object schema.
                    if (
                        not isinstance(input_schema, dict)
                        or input_schema.get("type") != "object"
                    ):
                        input_schema = {"type": "object", "properties": {}}
                    description = tool.get("description", "")
                    if description:
                        description = f"[{server_name}] {description}"
                    else:
                        description = f"[{server_name}] {raw_name}"
                    self.schemas.append(
                        {
                            "type": "function",
                            "function": {
                                "name": prefixed,
                                "description": description,
                                "parameters": input_schema,
                            },
                        }
                    )
                logger.info(
                    "MCP server '%s' started, %d tools", server_name, len(tools)
                )
            except TimeoutError:
                logger.warning("MCP server '%s' timed out during startup", server_name)
            except Exception as e:
                logger.warning("MCP server '%s' failed to start: %s", server_name, e)

    def get_schemas(self) -> list[dict]:
        return list(self.schemas)

    async def call(self, prefixed_name: str, arguments: dict) -> str:
        entry = self.tool_map.get(prefixed_name)
        if not entry:
            return f"Error: unknown MCP tool '{prefixed_name}'"
        server_name, original_name = entry
        client = self.clients.get(server_name)
        if not client:
            return f"Error: MCP server '{server_name}' not available"
        try:
            result = await client.call_tool(original_name, arguments)
        except MCPError as e:
            return f"Error: MCP tool '{prefixed_name}' failed: {e}"
        except Exception as e:
            logger.warning(
                "MCP tool '%s' unexpected error: %s", prefixed_name, e, exc_info=True
            )
            return f"Error: MCP tool '{prefixed_name}' failed: {e}"
        text, is_error = _extract_content(result)
        prefix = "MCP tool error: " if is_error else ""
        full = (
            f"{prefix}{text}" if text else (f"{prefix}(no content)" if is_error else "")
        )
        # Auto memory:// for large results.
        if len(full) > Config.MAX_TOOL_RESULT_CHARS:
            try:
                from .cache import Cache

                file_id = Cache.new_id()
                Cache.store(file_id, full)
                return f"Stored as memory://{file_id} ({len(full)} chars). Use read_file with sources=['memory://{file_id}'] to read or summarize."
            except Exception as e:
                logger.warning("MCP large result cache failed: %s", e)
        return full

    async def close(self) -> None:
        for _name, client in list(self.clients.items()):
            try:
                await client.close()
            except Exception:
                pass
        self.clients.clear()
        self.tool_map.clear()
        self.schemas.clear()
        self._started = False


# Global singleton, lazily started.
_global_manager: MCPManager | None = None
_global_lock = asyncio.Lock()


async def get_mcp_manager() -> MCPManager | None:
    cfg = Config.mcp_servers()
    if not cfg:
        return None
    global _global_manager
    async with _global_lock:
        if _global_manager is None:
            _global_manager = MCPManager(cfg)
            await _global_manager.start()
        elif not _global_manager._started:
            await _global_manager.start()
    return _global_manager


def get_mcp_schemas_sync() -> list[dict]:
    """Sync accessor for Tools.schema(); returns cached schemas (may be empty before startup)."""
    if _global_manager is None:
        # Try to trigger background start without awaiting; caller will get empty this turn.
        cfg = Config.mcp_servers()
        if cfg:
            # Schedule start for next iteration; don't block.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(get_mcp_manager())
            except RuntimeError:
                pass
        return []
    return _global_manager.get_schemas()


async def mcp_call(prefixed_name: str, arguments: dict) -> str:
    manager = await get_mcp_manager()
    if not manager:
        return "Error: MCP not configured"
    return await manager.call(prefixed_name, arguments)


async def close_mcp() -> None:
    global _global_manager
    if _global_manager is not None:
        try:
            await _global_manager.close()
        except Exception:
            pass
        _global_manager = None
