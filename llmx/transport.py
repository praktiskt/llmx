from __future__ import annotations

import asyncio
import http.client
import io
import json
import logging
from collections.abc import Awaitable, Callable
from urllib.parse import quote, unquote, urljoin, urlparse, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0"

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class LLMAPIError(RuntimeError):
    pass


async def request_with_retries(
    send: Callable[[], Awaitable[Response]],
    attempts: int = 5,
    delay: float = 0.0,
    retriable: Callable[[Response], bool] | None = None,
    on_exception: Callable[[int, Exception], None] | None = None,
    on_retry: Callable[[int, Response], None] | None = None,
    fail_fast: bool = False,
) -> Response | None:
    """Run send() up to `attempts` times.

    Returns the final Response (200, non-retriable, or exhausted retries), or
    None if every attempt raised an exception. Sleeps delay*attempt between tries.

    With fail_fast, stops retrying when two attempts return the same
    (status, body-prefix); a deterministic error will not heal on retry.
    """
    if retriable is None:

        def retriable(response: Response) -> bool:
            return response.status_code >= 400

    last_response: Response | None = None
    seen_signatures: set[tuple[int, str]] = set()
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
        if response.status_code != 200 and retriable(response):
            signature = (response.status_code, response.text[:120])
            if fail_fast and signature in seen_signatures:
                return response
            seen_signatures.add(signature)
            if attempt < attempts:
                if on_retry is not None:
                    on_retry(attempt, response)
                if delay:
                    await asyncio.sleep(delay * attempt)
                continue
        return response

    return last_response


class _Stream:
    """One-shot streaming body that closes connection on close."""

    def __init__(
        self, conn: http.client.HTTPConnection, resp: http.client.HTTPResponse
    ):
        self._conn = conn
        self._resp = resp
        self.status = resp.status

    def read(self) -> bytes:
        return self._resp.read()

    def __iter__(self):
        return iter(self._resp)

    def close(self) -> None:
        try:
            self._resp.close()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass


class Response:
    def __init__(
        self,
        fp,
        preloaded: bytes | None = None,
        status_code: int | None = None,
        headers: dict | None = None,
    ):
        self._fp = fp
        self._content = preloaded
        self.status_code = (
            status_code
            if status_code is not None
            else getattr(fp, "status", None) or getattr(fp, "code", 0)
        )
        self.headers: dict = {k.lower(): v for k, v in (headers or {}).items()}
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


def _new_connection(
    scheme: str, host: str, port: int, timeout: float | None
) -> http.client.HTTPConnection:
    if scheme == "https":
        return http.client.HTTPSConnection(host, port=port, timeout=timeout)
    return http.client.HTTPConnection(host, port=port, timeout=timeout)


def _iri_to_uri(url: str) -> str:
    """Convert IRI with unicode to URI with percent-encoding and punycode.

    Keeps already-encoded %XX intact, encodes unicode path/query/fragment
    and IDNA-encodes hostname. Safe for http.client ASCII path.
    """
    if not url or url.startswith("data:"):
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.scheme:
        return url
    # IDNA host
    netloc = parts.netloc
    if parts.hostname:
        try:
            host_ascii = parts.hostname.encode("idna").decode("ascii")
        except Exception:
            host_ascii = parts.hostname
        # rebuild netloc with userinfo/port
        rebuilt = ""
        if parts.username:
            rebuilt += quote(unquote(parts.username), safe="._~-")
            if parts.password:
                rebuilt += ":" + quote(unquote(parts.password), safe="._~-")
            rebuilt += "@"
        # handle IPv6 brackets
        if "[" in parts.netloc and parts.hostname:
            rebuilt += f"[{host_ascii}]"
        else:
            rebuilt += host_ascii
        if parts.port:
            rebuilt += f":{parts.port}"
        netloc = rebuilt
    # path / query / fragment: unquote then quote to normalize
    # safe keeps RFC 3986 sub-delims + unreserved; % kept to avoid double-encode
    # when not unquoting, but we unquote so % safe not needed for correctness
    # yet keep it to preserve stray % handling via unquote->quote cycle
    path = quote(unquote(parts.path), safe="/:@!$&'()*+,;=-._~%")
    query = quote(unquote(parts.query), safe="/:@!$&'()*+,;=-._~%?=&")
    fragment = quote(unquote(parts.fragment), safe="/:@!$&'()*+,;=-._~%?=&%#")
    return urlunsplit((parts.scheme, netloc, path, query, fragment))


def _sync_request(method: str, url: str, **kwargs):
    kwargs.pop("reuse", None)
    url = _iri_to_uri(url)
    parsed = urlparse(url)

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    headers.update(kwargs.get("headers") or {})

    body = kwargs.get("data")
    if isinstance(body, str):
        body = body.encode("utf-8")
    payload = kwargs.get("json")
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")

    timeout = kwargs.get("timeout")
    stream = bool(kwargs.get("stream"))
    max_redirects = 5

    for _ in range(max_redirects + 1):
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        conn = _new_connection(parsed.scheme, host, port, timeout)
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            raw_headers = resp.headers

            if resp.status in _REDIRECT_STATUSES and "location" in raw_headers:
                location = raw_headers["location"]
                next_url = _iri_to_uri(urljoin(url, location))
                resp.read()
                try:
                    conn.close()
                except Exception:
                    pass
                parsed, url = urlparse(next_url), next_url
                if resp.status in (303, 301, 302):
                    method, body = "GET", None
                continue

            if stream:
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                return Response(
                    _Stream(conn, resp), status_code=resp.status, headers=hdrs
                )

            data = resp.read()
            status = resp.status
            hdrs = {k.lower(): v for k, v in raw_headers.items()} if raw_headers else {}
            try:
                conn.close()
            except Exception:
                pass
            return Response(io.BytesIO(data), data, status_code=status, headers=hdrs)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

    raise RuntimeError(f"Too many redirects requesting {url}")


class AsyncHttp:
    @staticmethod
    async def _request(method: str, url: str, **kwargs) -> Response:
        return await asyncio.to_thread(_sync_request, method, url, **kwargs)

    @classmethod
    async def get(cls, url: str, **kwargs) -> Response:
        return await cls._request("GET", url, **kwargs)

    @classmethod
    async def post(cls, url: str, **kwargs) -> Response:
        return await cls._request("POST", url, **kwargs)
