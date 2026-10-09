import json
import os
import pty
import select
import termios
import time
import unittest

from llmx.client import EscWatcher


def _report(wfd, **kwargs):
    os.write(wfd, (json.dumps(kwargs) + "\n").encode())


@unittest.skipUnless(
    hasattr(os, "forkpty") and hasattr(termios, "tcgetattr"), "needs termios + pty"
)
class EscWatcherTest(unittest.TestCase):
    """Regression tests for the interactive tty mode.

    EscWatcher flips the tty to cbreak on a background thread. The attrs must
    be restored before __exit__ returns; a late restore lands while readline
    owns the terminal at the next prompt and input silently breaks.
    """

    def _run_on_tty(self, child_fn, on_ready=None, timeout=10.0):
        """Run child_fn(report_wfd) with stdin/stdout on a pty.

        on_ready(master_fd) is called once the child has signalled READY.
        Returns the list of JSON report dicts the child wrote.
        """
        rfd, wfd = os.pipe()
        pid, master = pty.fork()
        if pid == 0:
            os.close(rfd)
            try:
                child_fn(wfd)
            except BaseException:
                os._exit(1)
            os._exit(0)
        os.close(wfd)
        reports = []
        buf = b""
        injected = on_ready is None
        deadline = time.time() + timeout
        while time.time() < deadline:
            r, _, _ = select.select([rfd], [], [], 0.1)
            if r:
                chunk = os.read(rfd, 4096)
                if chunk:
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        reports.append(json.loads(line))
            if not injected and any(r.get("ready") for r in reports):
                time.sleep(0.1)  # let the watcher settle into cbreak mode
                on_ready(master)
                injected = True
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                while True:
                    r, _, _ = select.select([rfd], [], [], 0.05)
                    if not r:
                        break
                    chunk = os.read(rfd, 4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        reports.append(json.loads(line))
                break
        else:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            self.fail(f"child timed out after {timeout}s; reports: {reports}")
        os.close(rfd)
        os.close(master)
        return reports

    def test_attrs_restored_before_exit(self):
        def child(wfd):
            os.environ["LLM_INTERACTIVE"] = "true"
            before = termios.tcgetattr(0)
            watcher = EscWatcher()
            with watcher:
                during = termios.tcgetattr(0)
                _report(wfd, cbreak_during=not (during[3] & termios.ICANON))
            after = termios.tcgetattr(0)
            _report(
                wfd,
                restored_after_exit=after == before,
                thread_dead_after_exit=not watcher._thread.is_alive(),
            )
            os._exit(0)

        reports = self._run_on_tty(child)
        by_key = {k: v for r in reports for k, v in r.items()}
        self.assertTrue(by_key["cbreak_during"], "cbreak mode not engaged")
        self.assertTrue(
            by_key["restored_after_exit"],
            "termios attrs still cbreak after __exit__ returned",
        )
        self.assertTrue(
            by_key["thread_dead_after_exit"],
            "watcher thread still alive after __exit__",
        )

    def test_esc_interrupts_and_restores(self):
        def child(wfd):
            os.environ["LLM_INTERACTIVE"] = "true"
            before = termios.tcgetattr(0)
            watcher = EscWatcher()
            with watcher:
                _report(wfd, ready=True)
                deadline = time.time() + 2.0
                while time.time() < deadline and not watcher.interrupted:
                    time.sleep(0.01)
            after = termios.tcgetattr(0)
            _report(
                wfd,
                interrupted=watcher.interrupted,
                restored_after_exit=after == before,
                thread_dead_after_exit=not watcher._thread.is_alive(),
            )
            os._exit(0)

        reports = self._run_on_tty(child, on_ready=lambda m: os.write(m, b"\x1b"))
        by_key = {k: v for r in reports for k, v in r.items()}
        self.assertTrue(by_key.get("interrupted"), "ESC not detected")
        self.assertTrue(by_key.get("restored_after_exit"), "attrs not restored")
        self.assertTrue(by_key.get("thread_dead_after_exit"), "thread still alive")

    def test_input_after_exit_reaches_prompt(self):
        def child(wfd):
            os.environ["LLM_INTERACTIVE"] = "true"
            watcher = EscWatcher()
            with watcher:
                time.sleep(0.3)
            _report(wfd, ready=True)
            data = b""
            deadline = time.time() + 1.5
            while time.time() < deadline:
                r, _, _ = select.select([0], [], [], 0.1)
                if r:
                    data = os.read(0, 100)
                    break
            _report(wfd, data=data.decode())
            os._exit(0)

        reports = self._run_on_tty(child, on_ready=lambda m: os.write(m, b"y\n"))
        by_key = {k: v for r in reports for k, v in r.items()}
        self.assertEqual(by_key.get("data"), "y\n")

    def test_non_tty_is_noop(self):
        import unittest.mock

        os.environ["LLM_INTERACTIVE"] = "true"
        self.addCleanup(os.environ.pop, "LLM_INTERACTIVE", None)
        with unittest.mock.patch("sys.stdin") as fake:
            fake.isatty.return_value = False
            watcher = EscWatcher()
            with watcher as ctx:
                self.assertFalse(ctx.interrupted)
            self.assertFalse(watcher.interrupted)
            self.assertFalse(hasattr(watcher, "_thread"))


if __name__ == "__main__":
    unittest.main()
