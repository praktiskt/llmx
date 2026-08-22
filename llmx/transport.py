import asyncio
import json
import urllib.error
import urllib.request

from .config import Config


class LLMAPIError(RuntimeError):
    pass


class Response:
    def __init__(self, fp, preloaded: bytes | None = None):
        self._fp = fp
        self._content = preloaded
        self.status_code = getattr(fp, "status", None) or getattr(fp, "code", 0)
        self.encoding: str | None = None

    @property
    def content(self) -> bytes:
        if self._content is None:
            self._content = self._fp.read()
        return self._content

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "utf-8", errors="replace")

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text[:200]}")

    def iter_lines(self, decode_unicode: bool = False):
        for raw in self._fp:
            if decode_unicode:
                yield raw.decode(self.encoding or "utf-8", errors="replace").rstrip(
                    "\r\n"
                )
            else:
                yield raw.rstrip(b"\r\n")

    def close(self) -> None:
        self._fp.close()


class AsyncHttp:
    @staticmethod
    async def _request(method: str, url: str, **kwargs) -> Response:
        def do() -> Response:
            headers = dict(kwargs.get("headers") or {})
            body = kwargs.get("data")
            if isinstance(body, str):
                body = body.encode("utf-8")
            payload = kwargs.get("json")
            if payload is not None:
                body = json.dumps(payload).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"User-Agent": "Mozilla/5.0", **headers},
                method=method,
            )
            try:
                fp = urllib.request.urlopen(req, timeout=kwargs.get("timeout"))
                if kwargs.get("stream"):
                    return Response(fp)
                content = fp.read()
                fp.close()
                return Response(fp, content)
            except urllib.error.HTTPError as e:
                if kwargs.get("stream"):
                    return Response(e)
                return Response(e, e.read())

        return await asyncio.to_thread(do)

    @classmethod
    async def get(cls, url: str, **kwargs) -> Response:
        return await cls._request("GET", url, **kwargs)

    @classmethod
    async def post(cls, url: str, **kwargs) -> Response:
        return await cls._request("POST", url, **kwargs)

    @classmethod
    async def head(cls, url: str) -> Response:
        return await cls._request(
            "HEAD",
            url,
            timeout=Config.HEAD_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
