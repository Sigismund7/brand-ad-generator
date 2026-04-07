from __future__ import annotations

import asyncio
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from models import ProductNotFoundError

# crawl4ai is imported lazily so the rest of the app still loads if it's
# not installed yet (the env-setup step documents it in requirements.txt).
try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
    CRAWL4AI_AVAILABLE = True
except Exception:
    # crawl4ai can raise RuntimeError on import in non-main threads (e.g. Streamlit)
    # because it creates asyncio locks at module level. Fall back to requests.
    CRAWL4AI_AVAILABLE = False

_TIMEOUT = 20          # seconds per crawl attempt
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_url(href: str, base: str) -> str | None:
    """Turn a relative href into an absolute URL under the same domain."""
    try:
        full = urljoin(base, href)
        parsed = urlparse(full)
        base_parsed = urlparse(base)
        if parsed.netloc == base_parsed.netloc and parsed.scheme in ("http", "https"):
            return full
    except Exception:
        pass
    return None


def _slug_score(product_name: str, url: str, anchor: str) -> float:
    """
    Return a 0–1 relevance score between a product name and a (url, anchor) pair.
    Higher = better match.
    """
    words = re.findall(r"[a-z0-9]+", product_name.lower())
    if not words:
        return 0.0

    target = (url.lower() + " " + anchor.lower())
    hits = sum(1 for w in words if w in target)
    return hits / len(words)


def _extract_links_bs4(html: str, base_url: str) -> list[tuple[str, str]]:
    """Return [(absolute_url, anchor_text)] from raw HTML via BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    links: list[tuple[str, str]] = []
    for tag in soup.find_all("a", href=True):
        url = _normalize_url(tag["href"], base_url)
        if url:
            links.append((url, tag.get_text(" ", strip=True)))
    return links


async def _crawl_with_crawl4ai(url: str) -> str:
    """Fetch a page with crawl4ai and return cleaned markdown."""
    browser_cfg = BrowserConfig(headless=True)
    run_cfg = CrawlerRunConfig(page_timeout=_TIMEOUT * 1000)
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)
        if result.success:
            return result.markdown or result.cleaned_html or ""
    return ""


def _crawl_with_requests(url: str) -> str:
    """Fallback fetch via requests + BeautifulSoup → plain text."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove boilerplate elements
        for tag in soup(["nav", "footer", "header", "script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_product_url(brand_url: str, product_name: str) -> str:
    """
    Crawl the brand homepage and return the URL of the page that best
    matches `product_name`.  Raises ProductNotFoundError if nothing
    scores above a minimum threshold.
    """
    raw_html = ""

    if CRAWL4AI_AVAILABLE:
        try:
            markdown = asyncio.run(_crawl_with_crawl4ai(brand_url))
            # crawl4ai markdown won't have hrefs; fall through to requests
            # for link extraction. We just need the rendered HTML for links.
        except Exception:
            pass

    # Always use requests for link extraction — it's more reliable for hrefs
    try:
        resp = requests.get(brand_url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        raw_html = resp.text
    except Exception as exc:
        raise ProductNotFoundError(
            f"Could not reach {brand_url}: {exc}"
        ) from exc

    links = _extract_links_bs4(raw_html, brand_url)

    # Score every link
    scored = [
        (score, url, anchor)
        for url, anchor in links
        if (score := _slug_score(product_name, url, anchor)) > 0
    ]
    scored.sort(key=lambda t: t[0], reverse=True)

    if not scored or scored[0][0] < 0.3:
        raise ProductNotFoundError(
            f"No product page found for '{product_name}' on {brand_url}. "
            "Try pasting the product URL directly."
        )

    return scored[0][1]


def scrape_product_page(url: str) -> str:
    """
    Scrape a product page and return its content as clean text/markdown.
    Tries crawl4ai first (handles JS-rendered sites); falls back to requests.
    """
    content = ""

    if CRAWL4AI_AVAILABLE:
        try:
            content = asyncio.run(_crawl_with_crawl4ai(url))
        except Exception:
            content = ""

    if not content or len(content.strip()) < 100:
        content = _crawl_with_requests(url)

    if not content or len(content.strip()) < 50:
        raise ProductNotFoundError(
            f"Could not extract content from {url}. "
            "The page may require a login or block automated access."
        )

    # Trim to a reasonable size so we don't blow Gemini's context window
    return content[:12_000]
