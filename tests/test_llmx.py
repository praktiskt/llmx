import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from llmx.cache import Cache
from llmx.client import StreamFilter, truncate_history
from llmx.config import Config
from llmx.localfs import contents as localfs_contents
from llmx.localfs import grep_entries as localfs_grep
from llmx.localfs import list_entries as localfs_list
from llmx.localfs import read_entries as localfs_read
from llmx.localfs import resolve as localfs_resolve
from llmx.output import Color
from llmx.tools import Tools, _repair_json, parse_tool_args
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


class ColorToolTest(unittest.TestCase):
    def setUp(self):
        self._old = {
            k: os.environ.get(k) for k in ("NO_COLOR", "LLM_DISABLE_COLOR_OUTPUT")
        }
        for k in ("NO_COLOR", "LLM_DISABLE_COLOR_OUTPUT"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_builtin_uses_mapped_color(self):
        out = Color.tool("grep", "[tool] grep(q='x')")
        self.assertTrue(out.startswith(Color.TOOL_COLORS["grep"]))
        self.assertTrue(out.endswith(Color.RESET))

    def test_mcp_name_gets_mcp_color(self):
        out = Color.tool("fs__read_file", "[tool] fs__read_file(path='a')")
        self.assertTrue(out.startswith(Color.MCP))
        self.assertTrue(out.endswith(Color.RESET))
        self.assertNotEqual(Color.MCP, Color.DEFAULT)

    def test_unknown_non_mcp_falls_back_dim(self):
        out = Color.tool("mystery", "x")
        self.assertTrue(out.startswith(Color.DEFAULT))

    def test_disabled_color_plain(self):
        os.environ["LLM_DISABLE_COLOR_OUTPUT"] = "true"
        out = Color.tool("fs__read_file", "[tool] fs__read_file(path='a')")
        self.assertEqual(out, "[tool] fs__read_file(path='a')")


class CacheTest(unittest.TestCase):
    def setUp(self):
        Cache._storage.clear()

    def test_store_read_roundtrip(self):
        Cache.store("abc123", "line1\nline2")
        self.assertIn("line1", Cache.read(["abc123"]))

    def test_wraps_long_lines(self):
        Cache.store("abc123", "x" * 500)
        content = Cache.get("abc123")
        # Long lines are preserved; only data: URIs are truncated
        self.assertEqual(content, "x" * 500)

    def test_truncates_data_uris(self):
        Cache.store("abc123", "data:image/png,aaaaabbbbbccccc rest")
        content = Cache.get("abc123")
        self.assertIn("data:image/png,[TRUNCATED]", content)
        self.assertNotIn("aaaaa", content)

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


class ToolsAllowlistTest(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("LLM_TOOLS")
        os.environ.pop("LLM_TOOLS", None)

    def tearDown(self):
        if self._env is None:
            os.environ.pop("LLM_TOOLS", None)
        else:
            os.environ["LLM_TOOLS"] = self._env

    def test_unset_allows_all(self):

        self.assertEqual(len(Tools.schema()), len(Tools.SCHEMA))

    def test_allowlist_filters_schema(self):

        os.environ["LLM_TOOLS"] = "fetch, read"
        names = {t["function"]["name"] for t in Tools.schema()}
        self.assertEqual(names, {"fetch", "read"})

    def test_allowlist_alias_compat(self):

        os.environ["LLM_TOOLS"] = "fetch, read_file"
        names = {t["function"]["name"] for t in Tools.schema()}
        self.assertEqual(names, {"fetch", "read"})
        os.environ["LLM_TOOLS"] = "fetch, read"
        # old alias should still be executable
        Cache.store("abc123", "hello alias")
        result = asyncio.run(
            Tools.execute("read_file", {"sources": ["memory://abc123"]})
        )
        self.assertIn("hello alias", result)

    def test_empty_value_disables_all(self):

        os.environ["LLM_TOOLS"] = ""
        self.assertEqual(Tools.schema(), [])

    def test_execute_blocks_disabled_tool(self):
        os.environ["LLM_TOOLS"] = "fetch"
        result = asyncio.run(Tools.execute("grep_file", {}))
        self.assertIn("not enabled", result)

    def test_execute_allows_enabled_tool(self):
        Cache.store("abc123", "hello")
        os.environ["LLM_TOOLS"] = "read_file"
        result = asyncio.run(
            Tools.execute("read_file", {"sources": ["memory://abc123"]})
        )
        self.assertIn("hello", result)


class LocalFilesTest(unittest.TestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self._old_flag = os.environ.get("LLM_LOCAL_FILES")
        os.environ["LLM_LOCAL_FILES"] = "true"
        Path("src").mkdir()
        Path("src/main.py").write_text("alpha\nbeta\ngamma\n")
        Path("docs").mkdir()
        Path("docs/note.md").write_text("hello world\nbeta again\n")
        Path("docker").write_text("not a file_id\n")
        Path("blob.bin").write_bytes(b"\x00\x01\x02binary")

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmp.cleanup()
        if self._old_flag is None:
            os.environ.pop("LLM_LOCAL_FILES", None)
        else:
            os.environ["LLM_LOCAL_FILES"] = self._old_flag

    def test_read_offset_limit(self):
        out = "\n".join(localfs_read(["src/main.py"], offset=2, limit=1))
        self.assertIn("File src/main.py (lines 2-2 of 3)", out)
        self.assertIn("2: beta", out)
        self.assertNotIn("alpha", out)

    def test_grep_literal_context_and_regex(self):
        out = "\n".join(localfs_grep(["src/main.py"], "beta", context=1))
        self.assertIn('1 matches for "beta"', out)
        self.assertIn(" alpha", out)
        self.assertIn(">2: beta", out)
        out = "\n".join(localfs_grep(["src/main.py"], "^a", is_regex=True))
        self.assertIn(">1: alpha", out)

    def test_glob_multiple_files(self):
        out = "\n\n---\n\n".join(localfs_grep(["**/*.py", "docs/*.md"], "beta"))
        self.assertIn("src/main.py: 1 matches", out)
        self.assertIn("docs/note.md: 1 matches", out)

    def test_reject_absolute_and_traversal(self):
        for bad in ["/etc/passwd", "../outside.txt", "src/../../../etc/passwd"]:
            files, errors = localfs_resolve([bad])
            self.assertEqual(files, [])
            self.assertTrue(any("Error" in e for e in errors), bad)

    def test_symlink_escape_rejected(self):
        outside = Path(self._tmp.name).parent / f"outside_{os.getpid()}.txt"
        outside.write_text("secret\n")
        try:
            os.symlink(outside.resolve(), "link.txt")
            files, errors = localfs_resolve(["link.txt"])
            self.assertEqual(files, [])
            self.assertTrue(errors)
        finally:
            outside.unlink(missing_ok=True)

    def test_binary_file_refused(self):
        out = "\n".join(localfs_read(["blob.bin"]))
        self.assertIn("binary file", out)

    def test_missing_and_directory_messages(self):
        out = "\n".join(localfs_read(["nope.txt"]))
        self.assertIn("file nope.txt not found", out)
        out = "\n".join(localfs_read(["src"]))
        self.assertIn("is a directory", out)

    def test_docker_named_file_vs_cache_ids(self):
        Cache.store("docker", "cached content here")
        cache_part = Cache.read(["docker"])
        local_part = "\n".join(localfs_read(["docker"]))
        self.assertIn("cached content here", cache_part)
        self.assertIn("not a file_id", local_part)

    def test_contents_pairs_for_summarize(self):
        pairs = localfs_contents(["src/main.py", "missing.txt"])
        labels = [label for label, _ in pairs]
        texts = [text for _, text in pairs]
        self.assertIn("src/main.py", labels)
        self.assertIn("gamma", dict(pairs)["src/main.py"])
        self.assertTrue(any(text.startswith("Error:") for text in texts))
        self.assertIn(None, labels)

    def test_list_entries_glob_and_regex(self):
        out = localfs_list("**/*")
        self.assertIn("src/main.py", out)
        self.assertIn("docs/note.md", out)
        out = localfs_list("**/*.py")
        self.assertIn("main.py", out)
        self.assertNotIn("note.md", out)
        out = localfs_list(regex=r"\.md$")
        self.assertIn("note.md", out)
        self.assertNotIn("main.py", out)
        out = localfs_list(pattern="../*")
        self.assertIn("must stay under", out)

    def test_tools_read_merge_and_disabled(self):
        Cache.store("abc123", "cached line")
        result = asyncio.run(
            Tools.execute("read_file", {"sources": ["memory://abc123", "src/main.py"]})
        )
        self.assertIn("File abc123", result)
        self.assertIn("File src/main.py", result)

        os.environ["LLM_LOCAL_FILES"] = "false"
        result = asyncio.run(Tools.execute("read_file", {"sources": ["src/main.py"]}))
        self.assertIn("disabled (LLM_LOCAL_FILES)", result)

    def test_sources_reject_legacy_keys(self):
        for legacy in ({"file_ids": ["abc123"]}, {"paths": ["x.txt"]}):
            result = asyncio.run(Tools.execute("read_file", legacy))
            self.assertIn("were removed", result)
            self.assertIn("sources=", result)

    def test_sources_unknown_scheme(self):
        result = asyncio.run(
            Tools.execute("read_file", {"sources": ["https://example.com/x"]})
        )
        self.assertIn("unsupported scheme 'https://'", result)
        self.assertIn("use fetch for URLs", result)

    def test_sources_bad_memory_id(self):
        result = asyncio.run(
            Tools.execute("read_file", {"sources": ["memory://BAD-ID!"]})
        )
        self.assertIn("invalid id", result)
        self.assertIn("memory://<id>", result)

    def test_docker_named_source_vs_memory(self):
        Cache.store("docker", "cached content here")
        via_memory = asyncio.run(
            Tools.execute("read_file", {"sources": ["memory://docker"]})
        )
        via_path = asyncio.run(Tools.execute("read_file", {"sources": ["docker"]}))
        self.assertIn("cached content here", via_memory)
        self.assertNotIn("not a file_id", via_memory)
        self.assertIn("not a file_id", via_path)
        self.assertNotIn("cached content here", via_path)

    def test_tools_teaching_error_when_neither_source(self):
        result = asyncio.run(Tools.execute("read_file", {}))
        self.assertIn("provide sources=", result)
        result = asyncio.run(Tools.execute("grep_file", {}))
        self.assertIn("provide sources=", result)

    def test_tools_list_files(self):
        result = asyncio.run(Tools.execute("list_files", {"regex": r"\.py$"}))
        self.assertIn("src/main.py", result)
        os.environ["LLM_LOCAL_FILES"] = "false"
        result = asyncio.run(Tools.execute("list_files", {}))
        self.assertIn("disabled", result)

    def test_summarize_with_paths(self):
        from llmx import tools as tools_mod

        class FakeResp:
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"message": {"content": "SUM"}}]}

        async def fake_retry(factory, **kwargs):
            return FakeResp()

        saved_env = {
            k: os.environ.get(k) for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_HOST")
        }
        os.environ.update(LLM_MODEL="m", LLM_API_KEY="k", LLM_HOST="http://h")
        original = tools_mod.request_with_retries
        tools_mod.request_with_retries = fake_retry
        try:
            result = asyncio.run(
                tools_mod.summarize(
                    [],
                    ["src/main.py", "missing.txt"],
                    [],
                    ["d1"],
                )
            )
        finally:
            tools_mod.request_with_retries = original
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertIn("File src/main.py:", result)
        self.assertIn("SUM", result)
        self.assertIn("File sources:", result)
        self.assertIn("missing.txt not found", result)

    def test_summarize_with_memory_ids(self):
        from llmx import tools as tools_mod

        class FakeResp:
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"message": {"content": "MSUM"}}]}

        async def fake_retry(factory, **kwargs):
            return FakeResp()

        saved_env = {
            k: os.environ.get(k) for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_HOST")
        }
        os.environ.update(LLM_MODEL="m", LLM_API_KEY="k", LLM_HOST="http://h")
        Cache.store("abc123", "cached body for summary")
        original = tools_mod.request_with_retries
        tools_mod.request_with_retries = fake_retry
        try:
            result = asyncio.run(tools_mod.summarize(["abc123"], [], [], ["d1"]))
        finally:
            tools_mod.request_with_retries = original
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertIn("File abc123:", result)
        self.assertIn("MSUM", result)

    def test_schema_sources_replaces_legacy(self):
        names = {t["function"]["name"] for t in Tools.SCHEMA}
        self.assertIn("list_files", names)
        by_name = {t["function"]["name"]: t for t in Tools.SCHEMA}
        for tool in ("read", "grep", "summarize"):
            params = by_name[tool]["function"]["parameters"]
            props = params["properties"]
            self.assertIn("sources", props, tool)
            self.assertNotIn("file_ids", props, tool)
            self.assertNotIn("paths", props, tool)

    def test_system_prompt_conditional(self):
        os.environ["LLM_LOCAL_FILES"] = "true"
        prompt = Config.get_system_prompt()
        self.assertIn("sources=[...]", prompt)
        self.assertIn("memory://", prompt)
        self.assertIn("current working directory", prompt)
        os.environ["LLM_LOCAL_FILES"] = "false"
        prompt = Config.get_system_prompt()
        self.assertIn("sources=[...]", prompt)
        self.assertNotIn("current working directory", prompt)


class InteractiveTest(unittest.TestCase):
    def setUp(self):
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "LLM_INTERACTIVE",
                "LLM_RESPONSE_FORMAT",
                "LLM_MODEL",
                "LLM_API_KEY",
                "LLM_HOST",
                "LLM_TOOLS",
            )
        }
        os.environ.update(
            LLM_RESPONSE_FORMAT="json_object",
            LLM_MODEL="m",
            LLM_API_KEY="k",
            LLM_HOST="http://h",
            LLM_TOOLS="",
        )
        from llmx import client as client_mod

        self.client_mod = client_mod
        self._old_stdin = sys.stdin

    def tearDown(self):
        sys.stdin = self._old_stdin
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    class _FakeStdin:
        def __init__(self, lines):
            import io

            self._buf = io.StringIO("".join(line + "\n" for line in lines))

        def isatty(self):
            return True

        def readline(self):
            return self._buf.readline()

    def test_flag_default_off(self):
        os.environ.pop("LLM_INTERACTIVE", None)
        self.assertFalse(Config.interactive_enabled())

    def test_flag_env_toggle(self):
        os.environ["LLM_INTERACTIVE"] = "true"
        self.assertTrue(Config.interactive_enabled())
        os.environ["LLM_INTERACTIVE"] = "false"
        self.assertFalse(Config.interactive_enabled())

    def test_next_user_message_reads_and_skips_blanks(self):
        import unittest.mock

        with unittest.mock.patch("builtins.input", side_effect=["", "  ", "hello"]):
            out = run(self.client_mod.LLMClient._next_user_message())
        self.assertEqual(out, "hello")

    def test_next_user_message_quit_commands(self):
        import unittest.mock

        for cmd in ["quit", "exit", "/quit", "/exit"]:
            with unittest.mock.patch("builtins.input", return_value=cmd):
                self.assertIsNone(
                    run(self.client_mod.LLMClient._next_user_message()), cmd
                )

    def test_next_user_message_eof(self):
        import unittest.mock

        with unittest.mock.patch("builtins.input", side_effect=EOFError):
            self.assertIsNone(run(self.client_mod.LLMClient._next_user_message()))

    def _run_stream(self, argv, contents):
        """Patch AsyncHttp.post; returns list of request bodies."""
        calls = []

        async def fake_post(url, **kwargs):
            calls.append(json.loads(kwargs["data"]))

            class Resp:
                status_code = 200

                @staticmethod
                def json():
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": contents.pop(0) if contents else "x",
                                }
                            }
                        ]
                    }

            return Resp()

        http = self.client_mod.AsyncHttp
        original_post = http.post
        http.post = staticmethod(fake_post)
        try:
            run(self.client_mod.LLMClient.stream(argv))
        finally:
            http.post = original_post
        return calls

    def test_non_interactive_single_shot(self):
        os.environ["LLM_INTERACTIVE"] = "false"
        calls = self._run_stream(["hi"], ["one"])
        self.assertEqual(len(calls), 1)

    def test_interactive_continues_conversation(self):
        import unittest.mock

        os.environ["LLM_INTERACTIVE"] = "true"
        sys.stdin = self._FakeStdin([])
        with unittest.mock.patch("builtins.input", side_effect=["follow-up", "quit"]):
            calls = self._run_stream(["hi"], ["one", "two"])
        self.assertEqual(len(calls), 2)
        roles = [m["role"] for m in calls[1]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(calls[1]["messages"][2]["content"], "one")
        self.assertEqual(calls[1]["messages"][3]["content"], "follow-up")

    def test_interactive_first_input_when_no_args(self):
        import unittest.mock

        os.environ["LLM_INTERACTIVE"] = "true"
        sys.stdin = self._FakeStdin([])
        with unittest.mock.patch(
            "builtins.input", side_effect=["first prompt", "quit"]
        ):
            calls = self._run_stream([], ["answer"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["messages"][1]["content"], "first prompt")


if __name__ == "__main__":
    unittest.main()
