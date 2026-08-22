import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable

from .config import Config


class LLMAPIError(RuntimeError):
    pass


async def request_with_retries(
    send: Callable[[], Awaitable[Response]],
    attempts: int = 5,
    delay: float = 0.0,
    retriable: Callable[[Response], bool] | None = None,
    on_exception: Callable[[int, Exception], None] | None = None,
    on_retry: Callable[[int, Response], None] | None = None,
) -> Response | None:
    """Run send() up to `attempts` times.

    Returns the final Response (200, non-retriable, or exhausted retries), or
    None if every attempt raised an exception. Sleeps delay*attempt between tries.
    """
    if retriable is None:

        def retriable(response: Response) -> bool:
            return response.status_code >= 400

    last_response: Response | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = await send()
        except Exception as e:
            if on_exception is not None:
                on_exception(attempt, e)
            if attempt < attempts and delay:
                await asyncio.sleep(delay * attempt)
            continue

        last_response = response
        if response.status_code != 200 and retriable(response) and attempt < attempts:
            if on_retry is not None:
                on_retry(attempt, response)
            if delay:
                await asyncio.sleep(delay * attempt)
            continue
        return response

    return last_response


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
