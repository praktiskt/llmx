import asyncio
import http.server
import threading
import unittest

from llmx import transport
from llmx.transport import AsyncHttp


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/redir":
            self.send_response(302)
            self.send_header("Location", "/final")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/final":
            body = b"final-body"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = b"not found"
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(length)
        body = b"posted:" + payload
        self.send_response(201)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def run(coro):
    return asyncio.run(coro)


class TransportPoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        transport._pool.clear()

    def setUp(self):
        transport._pool.clear()

    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def test_get_follows_redirect(self):
        resp = run(AsyncHttp.get(f"{self.base_url()}/redir", timeout=5))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text, "final-body")

    def test_post_body_roundtrip(self):
        resp = run(
            AsyncHttp.post(
                f"{self.base_url()}/final",
                json={"a": 1},
                timeout=5,
            )
        )
        self.assertEqual(resp.status_code, 201)
        self.assertIn(b'"a"', resp.content)

    def test_404_status_preserved_not_raised(self):
        resp = run(AsyncHttp.get(f"{self.base_url()}/missing", timeout=5))
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.text, "not found")

    def test_connection_reused_from_pool(self):
        run(AsyncHttp.get(f"{self.base_url()}/final", timeout=5))
        idle = transport._pool.get(("http", "127.0.0.1", self.port), [])
        self.assertEqual(len(idle), 1)
        run(AsyncHttp.get(f"{self.base_url()}/final", timeout=5))
        idle = transport._pool.get(("http", "127.0.0.1", self.port), [])
        self.assertEqual(len(idle), 1)

    def test_reuse_false_discards_connection(self):
        run(AsyncHttp.get(f"{self.base_url()}/final", timeout=5, reuse=False))
        idle = transport._pool.get(("http", "127.0.0.1", self.port), [])
        self.assertEqual(idle, [])
        run(
            AsyncHttp.post(
                f"{self.base_url()}/final",
                json={"a": 1},
                timeout=5,
                reuse=False,
            )
        )
        idle = transport._pool.get(("http", "127.0.0.1", self.port), [])
        self.assertEqual(idle, [])

    def test_streaming_iter_lines(self):
        async def consume():
            resp = await AsyncHttp.get(
                f"{self.base_url()}/final", timeout=5, stream=True
            )
            try:
                lines = list(resp.iter_lines(decode_unicode=True))
                return resp.status_code, lines
            finally:
                resp.close()

        status, lines = run(consume())
        self.assertEqual(status, 200)
        self.assertEqual(lines, ["final-body"])

    def test_streaming_close_without_read_does_not_pool(self):
        async def early_close():
            resp = await AsyncHttp.get(
                f"{self.base_url()}/final", timeout=5, stream=True
            )
            await asyncio.sleep(0)  # let headers arrive
            resp.close()

        run(early_close())
        idle = transport._pool.get(("http", "127.0.0.1", self.port), [])
        self.assertEqual(idle, [])

    def test_streaming_close_after_full_read_does_not_pool(self):
        async def consume_and_close():
            resp = await AsyncHttp.get(
                f"{self.base_url()}/final", timeout=5, stream=True
            )
            try:
                assert resp.content
            finally:
                resp.close()

        run(consume_and_close())
        # fully-drained streams are still discarded: a poisoned pooled
        # connection costs more than a handshake
        idle = transport._pool.get(("http", "127.0.0.1", self.port), [])
        self.assertEqual(idle, [])

    def test_stale_pooled_connection_silently_retried_on_fresh_socket(self):
        class _PoisonConn:
            host = "127.0.0.1"

            def __init__(self, port):
                self.port = port
                self.closed = False

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                raise http.client.BadStatusLine("0")

            def close(self):
                self.closed = True

        poison = _PoisonConn(self.port)
        transport._pool[("http", "127.0.0.1", self.port)] = [poison]

        resp = run(AsyncHttp.get(f"{self.base_url()}/final", timeout=5))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text, "final-body")
        self.assertTrue(poison.closed, "poisoned connection must be discarded")
        idle = transport._pool.get(("http", "127.0.0.1", self.port), [])
        self.assertEqual(len(idle), 1)
        self.assertIsNot(idle[0], poison)


if __name__ == "__main__":
    unittest.main()
