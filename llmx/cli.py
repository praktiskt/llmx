import asyncio
import logging
import os
import signal
import sys

from .client import LLMClient
from .output import Color, Log
from .transport import LLMAPIError


def _install_quit_handler() -> None:
    """Hard-exit on Ctrl+C.

    A KeyboardInterrupt raised into asyncio.run stalls in executor shutdown
    while a blocked stdin-reader thread lingers; a direct handler avoids
    teardown entirely.
    """

    def handler(signum, frame):
        Log.stderr(Color.dim("\n[quit]"), flush=True)
        sys.stdout.flush()
        os._exit(130)

    signal.signal(signal.SIGINT, handler)


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "WARNING").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    _install_quit_handler()
    # Start MCP servers if configured (non-blocking; tools appear once available).
    try:
        from .config import Config as _Cfg
        from .mcp import get_mcp_manager as _get_mcp

        if _Cfg.mcp_servers():
            await _get_mcp()
    except Exception as e:
        logging.getLogger(__name__).debug("MCP startup failed: %s", e)
    prompt = [*sys.argv[1:]]
    if not sys.stdin.isatty():
        prompt.extend(["\n\n", *sys.stdin.read().splitlines()])
    try:
        await LLMClient.stream(prompt)
    except LLMAPIError as e:
        Log.stderr(f"{Color.ERROR}[error]: {e}{Color.RESET}")
        sys.exit(1)
    finally:
        try:
            from .mcp import close_mcp as _close

            await _close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
