import asyncio
import logging
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, unquote, urlparse

from .cache import Cache
from .config import Config
from .transport import AsyncHttp

logger = logging.getLogger(__name__)


class DuckDuckGoLiteSearch(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current_url = ""
        self.current_title = ""
        self.in_link = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "a":
            href = attrs_dict.get("href", "")
            if isinstance(href, str) and href.startswith("//duckduckgo.com/l/?uddg="):
                self.in_link = True
                self.current_url = href
                self.current_title = ""

    def handle_endtag(self, tag):
        if tag == "a" and self.in_link:
            self.in_link = False
            if self.current_url and self.current_title:
                parsed = urlparse(self.current_url)
                params = parse_qs(parsed.query)
                actual_url = params.get("uddg", [self.current_url])[0]
                self.results.append(
                    {
                        "url": unquote(actual_url),
                        "title": self.current_title.strip(),
                        "snippet": "",
                    }
                )

    def handle_data(self, data):
        if self.in_link:
            self.current_title += data

    def get_results(self, max_results=5):
        return self.results[:max_results]


async def _search_images(query: str, max_results: int = 5) -> str:
    proxy = Config.markdown_image_search_proxy()
    if proxy:
        url = f"{proxy.rstrip('/')}/{quote(query)}"
    else:
        url = f"https://duckduckgo.com/?q={quote(query)}&ia=images&iax=images"

    try:
        response = await AsyncHttp.get(
            url,
            timeout=Config.SEARCH_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        if proxy:
            return response.text

        html = response.text
        results = []
        seen_urls = set()

        img_pattern = re.compile(r"!\[Image \d+:")
        for img_match in img_pattern.finditer(html):
            search_start = img_match.end()
            link_match = re.search(r"\]\((https?://[^)]+)\)", html[search_start:])
            if not link_match:
                continue

            img_url = link_match.group(1)
            if "duckduckgo.com" in img_url and "/iu/" not in img_url:
                continue
            if img_url in seen_urls:
                continue
            seen_urls.add(img_url)

            parsed = urlparse(img_url)
            params = parse_qs(parsed.query)
            if "u" in params:
                target_url = unquote(params["u"][0])
            else:
                target_url = img_url

            title_start = img_match.end()
            title_end = search_start + link_match.start()
            title = html[title_start:title_end].strip()

            results.append(f"{len(results) + 1}. [{title}]({target_url})")
            if len(results) >= max_results:
                break

        if not results:
            return "No images found."
        return "\n".join(results)
    except Exception:
        logger.error("Image search failed for query: %s", query, exc_info=True)
        return "Image search failed."


async def _run_search(query: str, max_results: int = 5) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "sv-SE,sv;q=0.9",
        "Cache-Control": "max-age=0",
        "Priority": "u=0, i",
        "Sec-Ch-Ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Linux"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
        "Connection": "keep-alive",
    }

    proxy = Config.markdown_search_proxy()
    if proxy:
        url = f"{proxy.rstrip('/')}/{quote(query)}"
        for attempt in range(3):
            try:
                response = await AsyncHttp.get(
                    url, timeout=Config.SEARCH_TIMEOUT, headers=headers
                )
                response.raise_for_status()
                return response.text.strip()
            except Exception:
                if attempt < 2:
                    logger.warning(
                        "Search proxy attempt %d failed for '%s', retrying...",
                        attempt + 1,
                        query,
                    )
                    await asyncio.sleep(0.3 * (attempt + 1))
                    continue
                logger.error(
                    "Search proxy failed after retries for '%s'",
                    query,
                    exc_info=True,
                )
        return "Search failed after retries."

    url = f"https://lite.duckduckgo.com/lite/?q={quote(query)}"
    for attempt in range(3):
        try:
            response = await AsyncHttp.get(
                url, timeout=Config.SEARCH_TIMEOUT, headers=headers
            )
            response.raise_for_status()

            parser = DuckDuckGoLiteSearch()
            parser.feed(response.text)
            results = parser.get_results(max_results)
            if results:
                lines = []
                for i, r in enumerate(results, 1):
                    lines.append(f"{i}. [{r['title']}]({r['url']})")
                    if r["snippet"]:
                        lines.append(f"   {r['snippet']}")
                    lines.append("")
                return "\n".join(lines).strip()
        except Exception:
            if attempt < 2:
                logger.warning(
                    "DuckDuckGo search attempt %d failed for '%s', retrying...",
                    attempt + 1,
                    query,
                )
                await asyncio.sleep(0.3 * (attempt + 1))
                continue
            logger.error(
                "DuckDuckGo search failed after retries for '%s'",
                query,
                exc_info=True,
            )

    return "Search failed after retries."


async def search(
    queries: list[str], max_results: int = 5, images_only: bool = False
) -> str:
    async def search_task(query: str) -> tuple[str, str | None]:
        if images_only:
            result = await _search_images(query, max_results)
        else:
            result = await _run_search(query, max_results)
        if result.startswith(("Search failed", "Image search failed")):
            return (query, None)
        file_id = Cache.new_id()
        Cache.store(file_id, result)
        return (query, file_id)

    results = await asyncio.gather(*(search_task(query) for query in queries))

    lines = []
    for q, fid in results:
        if fid is None:
            lines.append(f"Query '{q}': search failed, do not read_file this query")
        else:
            lines.append(f"Query '{q}' stored in file_id={fid}")
    lines.append("")
    lines.append("Use read_file, grep_file or summarize to get the content.")
    return "\n".join(lines)
