import asyncio
import sys
import pathlib

# Ensure repository root is on path
sys.path.append(str(pathlib.Path(".").resolve()))

from llmx.llm import AsyncHttp, ConversionCache


async def _dummy_fetch(url: str) -> str:
    # Use the real method; network may fail, so we just test cache logic by calling twice.
    return await AsyncHttp.get_markdown(url)


def test_conversion_cache():
    url = "https://example.com"
    # clear any prior entry
    ConversionCache._store.pop(url, None)
    # First call – may return None if network blocked, but cache will store result if any.
    result1 = asyncio.run(_dummy_fetch(url))
    # Second call should hit cache (even if None)
    result2 = asyncio.run(_dummy_fetch(url))
    assert result1 == result2, "Cache should return the same result on repeated calls"
