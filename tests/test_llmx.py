import asyncio
import time
import unittest

from llmx.cache import Cache
from llmx.client import StreamFilter, truncate_history
from llmx.config import Config
from llmx.tools import _repair_json, parse_tool_args
from llmx.transport import Response, request_with_retries


def run(coro):
    return asyncio.run(coro)


class FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.text = f"body-{status_code}"


class RepairJsonTest(unittest.TestCase):
    def test_valid_json_untouched(self):
        s = '{"a": "b", "c": [1, 2]}'
        self.assertEqual(_repair_json(s), s)

    def test_inner_quotes_escaped(self):
        s = '{"a": "say "hi" now", "b": 1}'
        import json

        self.assertEqual(json.loads(_repair_json(s)), {"a": 'say "hi" now', "b": 1})


class ParseToolArgsTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(
            parse_tool_args('{"urls": ["https://x"]}'), {"urls": ["https://x"]}
        )

    def test_broken_json_returns_empty(self):
        self.assertEqual(parse_tool_args("not json at all {"), {})

    def test_string_encoded_list_unwrapped(self):
        result = parse_tool_args('{"file_ids": "[\\"abc123\\"]"}')
        self.assertEqual(result["file_ids"], ["abc123"])


class ValidateFileIdTest(unittest.TestCase):
    def test_ok(self):
        self.assertIsNone(Config.validate_file_id("abc123"))

    def test_bad_chars(self):
        err = Config.validate_file_id("ABC123")
        self.assertIsNotNone(err)

    def test_empty(self):
        err = Config.validate_file_id("")
        self.assertIsNotNone(err)


class CacheTest(unittest.TestCase):
    def setUp(self):
        Cache._storage.clear()

    def test_store_read_roundtrip(self):
        Cache.store("abc123", "line1\nline2")
        self.assertIn("line1", Cache.read(["abc123"]))

    def test_wraps_long_lines(self):
        Cache.store("abc123", "x" * 500)
        content = Cache.get("abc123")
        self.assertTrue(all(len(line) <= 200 for line in content.splitlines()))

    def test_lru_eviction(self):
        for i in range(60):
            Cache.store(f"id{i:03d}", "data")
        self.assertEqual(len(Cache._storage), Cache.MAX_ENTRIES)
        self.assertNotIn("id000", Cache._storage)
        self.assertIn("id059", Cache._storage)

    def test_get_touches_lru(self):
        for i in range(50):
            Cache.store(f"id{i:03d}", "data")
        Cache.get("id000")
        Cache.store("zzzzzz", "new")
        self.assertIn("id000", Cache._storage)
        self.assertNotIn("id001", Cache._storage)

    def test_new_id_unique(self):
        ids = {Cache.new_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_read_offset_limit(self):
        Cache.store("abc123", "\n".join(str(i) for i in range(100)))
        out = Cache.read(["abc123"], offset=10, limit=5)
        lines = [ln for ln in out.splitlines() if ": " in ln and ln[0].isdigit()]
        self.assertEqual(lines[0], "10: 9")
        self.assertEqual(len(lines), 5)
        self.assertIn("lines 10-14 of 100", out)

    def test_read_missing_file(self):
        self.assertIn("not found", Cache.read(["zzz999"]))

    def test_grep_literal_and_context(self):
        Cache.store("abc123", "alpha\nbeta match\ngamma\nbeta again\ndelta")
        out = Cache.grep(["abc123"], "beta", context=0)
        self.assertIn("2 matches", out)
        out = Cache.grep(["abc123"], "beta", context=1)
        self.assertIn("alpha", out)
        self.assertIn("gamma", out)

    def test_grep_regex_and_ignore_case(self):
        Cache.store("abc123", "Hello\nworld\nHELLO")
        out = Cache.grep(["abc123"], "^hello$", is_regex=True)
        self.assertIn("no matches", out.lower())
        out = Cache.grep(["abc123"], "^hello$", is_regex=True, ignore_case=True)
        self.assertIn("2 matches", out)


class _FakeFP:
    def __init__(self, status=200, body=b"", lines=None):
        self.status = status
        self.code = status
        self._body = body
        self._lines = lines or []

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        pass


class ResponseTest(unittest.TestCase):
    def test_status_from_fp(self):
        r = Response(_FakeFP(status=201), b"data")
        self.assertEqual(r.status_code, 201)

    def test_text_decode(self):
        r = Response(_FakeFP(), "héllo".encode())
        self.assertEqual(r.text, "héllo")

    def test_raise_for_status(self):
        r = Response(_FakeFP(status=404), b"x")
        with self.assertRaises(RuntimeError):
            r.raise_for_status()
        ok = Response(_FakeFP(status=200), b"x")
        ok.raise_for_status()

    def test_iter_lines_strips_newlines(self):
        r = Response(_FakeFP(lines=[b"a\n", b"b\r\n", b"c"]))
        self.assertEqual(list(r.iter_lines(decode_unicode=True)), ["a", "b", "c"])


class RequestWithRetriesTest(unittest.TestCase):
    def test_success_first_try(self):
        calls = []

        async def send():
            calls.append(1)
            return FakeResponse(200)

        resp = run(request_with_retries(send))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(calls), 1)

    def test_retries_then_succeeds(self):
        seq = [FakeResponse(503), FakeResponse(200)]

        async def send():
            return seq.pop(0)

        retries = []
        resp = run(
            request_with_retries(
                send,
                attempts=3,
                on_retry=lambda a, r: retries.append((a, r.status_code)),
            )
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(retries, [(1, 503)])

    def test_exhaustion_returns_last_response(self):
        async def send():
            return FakeResponse(500)

        resp = run(request_with_retries(send, attempts=3))
        self.assertIsNotNone(resp)
        self.assertEqual(resp.status_code, 500)

    def test_all_exceptions_returns_none(self):
        async def send():
            raise OSError("boom")

        resp = run(request_with_retries(send, attempts=2))
        self.assertIsNone(resp)

    def test_non_retriable_returned_immediately(self):
        calls = []

        async def send():
            calls.append(1)
            return FakeResponse(400)

        resp = run(request_with_retries(send, attempts=5, retriable=lambda r: False))
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(len(calls), 1)

    def test_delay_between_attempts(self):
        import time

        attempts = []

        async def send():
            attempts.append(time.monotonic())
            if len(attempts) < 3:
                return FakeResponse(500)
            return FakeResponse(200)

        run(request_with_retries(send, attempts=3, delay=0.05))
        self.assertGreaterEqual(attempts[1] - attempts[0], 0.04)
        self.assertGreaterEqual(attempts[2] - attempts[1], 0.09)

    def test_fail_fast_stops_on_identical_error(self):
        calls = []

        async def send():
            calls.append(1)
            return FakeResponse(422)

        resp = run(request_with_retries(send, attempts=5, fail_fast=True))
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(len(calls), 2)

    def test_fail_fast_ignores_distinct_errors(self):
        codes = [500, 502, 503]

        async def send():
            return FakeResponse(codes.pop(0))

        resp = run(request_with_retries(send, attempts=3))
        self.assertEqual(resp.status_code, 503)


class TruncateHistoryTest(unittest.TestCase):
    def _msg(self, role, content, **kw):
        m = {"role": role, "content": content}
        m.update(kw)
        return m

    def test_short_history_untouched(self):
        msgs = [self._msg("system", "sys"), self._msg("user", "hi")]
        self.assertIs(truncate_history(msgs), msgs)

    def test_drops_oldest_turns_first(self):
        old_limit = Config.MAX_CONTEXT_CHARS
        Config.MAX_CONTEXT_CHARS = 400
        try:
            msgs = [self._msg("system", "sys")]
            for i in range(10):
                msgs.append(self._msg("user", f"u{i}" * 40))
                msgs.append(self._msg("assistant", f"a{i}" * 40))
            out = truncate_history(msgs)
            self.assertEqual(out[0], msgs[0])
            joined = " ".join(m["content"] or "" for m in out[1:])
            self.assertNotIn("u0", joined)
            self.assertNotIn("u1", joined)
            self.assertIn("a9", joined)
            self.assertTrue(any("truncated" in (m["content"] or "") for m in out))
        finally:
            Config.MAX_CONTEXT_CHARS = old_limit

    def test_assistant_tool_group_kept_together(self):
        old_limit = Config.MAX_CONTEXT_CHARS
        Config.MAX_CONTEXT_CHARS = 300
        try:
            tool_call_msg = self._msg(
                "assistant",
                None,
                tool_calls=[
                    {"id": "t1", "function": {"name": "fetch", "arguments": "x" * 80}}
                ],
            )
            tool_result = self._msg("tool", "r" * 80, tool_call_id="t1")
            filler = self._msg("user", "f" * 100)
            msgs = [
                self._msg("system", "s"),
                tool_call_msg,
                tool_result,
                filler,
            ]
            out = truncate_history(msgs)
            for i, m in enumerate(out):
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    self.assertLess(i, len(out) - 1)
                    self.assertEqual(out[i + 1].get("role"), "tool")
        finally:
            Config.MAX_CONTEXT_CHARS = old_limit


class SessionStoreTest(unittest.TestCase):
    def test_cap_evicts_oldest(self):
        import llmx.server as srv

        old_max = srv.MAX_SESSIONS
        old_ttl = srv.SESSION_TTL_SECONDS
        srv.MAX_SESSIONS = 3
        srv.SESSION_TTL_SECONDS = 3600
        try:
            srv.sessions.clear()
            for _ in range(5):
                srv.get_session(None)
            self.assertEqual(len(srv.sessions), 3)
        finally:
            srv.MAX_SESSIONS = old_max
            srv.SESSION_TTL_SECONDS = old_ttl
            srv.sessions.clear()

    def test_expired_session_recreated(self):
        import llmx.server as srv

        old_ttl = srv.SESSION_TTL_SECONDS
        srv.SESSION_TTL_SECONDS = 3600
        try:
            srv.sessions.clear()
            s, sid = srv.get_session(None)
            s.last_access -= 7200
            s2, sid2 = srv.get_session(sid)
            self.assertIsNot(s, s2)
            self.assertNotEqual(sid2, sid)
        finally:
            srv.SESSION_TTL_SECONDS = old_ttl
            srv.sessions.clear()

    def test_live_session_touched(self):
        import llmx.server as srv

        old_ttl = srv.SESSION_TTL_SECONDS
        srv.SESSION_TTL_SECONDS = 3600
        try:
            srv.sessions.clear()
            s, sid = srv.get_session(None)
            s.last_access -= 1800
            s2, sid2 = srv.get_session(sid)
            self.assertIs(s, s2)
            self.assertGreater(s2.last_access, time.monotonic() - 5)
        finally:
            srv.SESSION_TTL_SECONDS = old_ttl
            srv.sessions.clear()


class ExecuteWithRetryTest(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_deterministic_error_no_retry(self):
        import llmx.server as srv

        calls = []

        async def boom(tool_call):
            calls.append(1)
            raise ValueError("bad args")

        orig = srv.Tools.execute_wrapper
        srv.Tools.execute_wrapper = staticmethod(boom)
        try:
            tool_id, result = self._run(
                srv.execute_with_retry({"id": "t1", "function": {"name": "fetch"}})
            )
        finally:
            srv.Tools.execute_wrapper = staticmethod(orig)
        self.assertEqual(len(calls), 1)
        self.assertIn("ValueError", result)

    def test_transient_error_retries(self):
        import llmx.server as srv

        calls = []

        async def flaky(tool_call):
            calls.append(1)
            if len(calls) < 3:
                raise OSError("network down")
            return ("t1", "fine")

        orig = srv.Tools.execute_wrapper
        srv.Tools.execute_wrapper = staticmethod(flaky)
        try:
            tool_id, result = self._run(
                srv.execute_with_retry({"id": "t1", "function": {"name": "fetch"}})
            )
        finally:
            srv.Tools.execute_wrapper = staticmethod(orig)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result, "fine")


class StreamFilterTest(unittest.TestCase):
    def test_plain_text_passthrough(self):
        f = StreamFilter()
        self.assertEqual(f.feed("hello world\nsecond line"), "hello world\nsecond line")
        self.assertEqual(f.flush(), "")

    def test_block_single_fragment_removed(self):
        f = StreamFilter()
        text = "A<system-reminder>secret plans</system-reminder>B"
        self.assertEqual(f.feed(text), "AB")

    def test_block_split_across_fragments(self):
        f = StreamFilter()
        out = f.feed("answer <system-")
        out += f.feed("reminder>injected junk</system-")
        out += f.feed("reminder> done")
        self.assertEqual(out, "answer  done")

    def test_partial_open_tag_held_then_completed(self):
        f = StreamFilter()
        out = f.feed("keep me <syst")
        self.assertEqual(out, "keep me ")
        out += f.feed("em-reminder>junk</system-reminder>tail")
        self.assertEqual(out + f.flush(), "keep me tail")

    def test_unterminated_block_suppressed_and_flush_empty(self):
        f = StreamFilter()
        out = f.feed("ok <system-reminder>never closed...")
        self.assertEqual(out, "ok ")
        self.assertEqual(f.flush(), "")

    def test_multiple_blocks(self):
        f = StreamFilter()
        text = (
            "<system-reminder>a</system-reminder>X<system-reminder>b</system-reminder>Y"
        )
        self.assertEqual(f.feed(text), "XY")

    def test_case_sensitive(self):
        f = StreamFilter()
        self.assertEqual(
            f.feed("<System-Reminder>kept</System-Reminder>"),
            "<System-Reminder>kept</System-Reminder>",
        )

    def test_split_tag_across_three_tiny_fragments(self):
        f = StreamFilter()
        pieces = ["hi ", "<system", "-remi", "nder>x</system", "-reminder>", "bye"]
        out = "".join(f.feed(p) for p in pieces) + f.flush()
        self.assertEqual(out, "hi bye")


class ReadStreamFilterWiringTest(unittest.TestCase):
    class _FakeSSE:
        def __init__(self, sse_lines):
            self._lines = [ln.encode("utf-8") for ln in sse_lines]
            self.encoding = None

        def iter_lines(self, decode_unicode=False):
            for raw in self._lines:
                yield raw.decode("utf-8") if decode_unicode else raw

        def close(self):
            pass

    def test_reader_strips_reminders_from_content_and_history(self):
        from llmx.client import LLMClient

        lines = [
            'data: {"choices":[{"delta":{"content":"pre <system-"}}]}',
            'data: {"choices":[{"delta":{"content":"reminder>secret tail</system-reminder>post"}}]}',
            "data: [DONE]",
        ]
        message, printed = asyncio.run(LLMClient._read_stream(self._FakeSSE(lines)))
        self.assertEqual(message["content"], "pre post")
        self.assertTrue(printed)


if __name__ == "__main__":
    unittest.main()
