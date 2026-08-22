import asyncio
import logging
import os
import sys

from .client import LLMClient
from .output import Color, Log
from .transport import LLMAPIError


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "WARNING").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    prompt = [*sys.argv[1:]]
    if not sys.stdin.isatty():
        prompt.extend(["\n\n", *sys.stdin.read().splitlines()])
    try:
        await LLMClient.stream(prompt)
    except LLMAPIError as e:
        Log.stderr(f"{Color.ERROR}[error]: {e}{Color.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
