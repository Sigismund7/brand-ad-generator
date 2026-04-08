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


def extract_product_image_url(url: str) -> str | None:
    """
    Fetch a product page and return the best candidate product image URL.

    Priority order:
    1. <meta property="og:image"> — most reliable across e-commerce sites
    2. <meta name="twitter:image">
    3. First "image" key in schema.org Product JSON-LD
    4. Largest <img> whose src path contains a product-signal keyword
    """
    import json as _json

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # 1. og:image
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return _normalize_url(og["content"], url) or og["content"]

    # 2. twitter:image
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    if not tw:
        tw = soup.find("meta", property="twitter:image")
    if tw and tw.get("content"):
        return _normalize_url(tw["content"], url) or tw["content"]

    # 3. schema.org Product JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = _json.loads(script.string or "")
            # data can be a list or a dict
            items = data if isinstance(data, list) else [data]
            for item in items:
                # Handle @graph wrapper
                if isinstance(item, dict) and "@graph" in item:
                    items = item["@graph"]
                    break
            for item in items:
                if not isinstance(item, dict):
                    continue
                types = item.get("@type", "")
                if isinstance(types, str):
                    types = [types]
                if "Product" in types:
                    img = item.get("image")
                    if isinstance(img, str) and img:
                        return _normalize_url(img, url) or img
                    if isinstance(img, list) and img:
                        candidate = img[0] if isinstance(img[0], str) else img[0].get("url", "")
                        if candidate:
                            return _normalize_url(candidate, url) or candidate
                    if isinstance(img, dict):
                        candidate = img.get("url", "")
                        if candidate:
                            return _normalize_url(candidate, url) or candidate
        except Exception:
            continue

    # 4. Largest <img> with a product-signal in its src
    _PRODUCT_SIGNALS = ("product", "pdp", "/images/", "/img/", "/media/", "/photos/")
    _SKIP_SIGNALS = ("logo", "icon", "sprite", "placeholder", "blank", "pixel")
    best_url: str | None = None
    best_area = 0
    for img_tag in soup.find_all("img"):
        src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-lazy-src") or ""
        if not src:
            continue
        src_lower = src.lower()
        if any(s in src_lower for s in _SKIP_SIGNALS):
            continue
        if not any(s in src_lower for s in _PRODUCT_SIGNALS):
            continue
        try:
            w = int(img_tag.get("width", 0) or 0)
            h = int(img_tag.get("height", 0) or 0)
            area = w * h
        except (ValueError, TypeError):
            area = 0
        if area > best_area:
            best_area = area
            best_url = src
    if best_url:
        return _normalize_url(best_url, url) or best_url

    return None


def fetch_brand_logo_url(brand_url: str) -> str | None:
    """
    Fetch the brand homepage and return the best candidate logo URL.

    Priority order:
    1. <img> whose class, alt, id, or src contains "logo"
    2. <link rel="apple-touch-icon"> — high-res square icon, ideal for avatars
    3. <link rel="icon"> or <link rel="shortcut icon"> — favicon fallback
    """
    try:
        resp = requests.get(brand_url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # 1. <img> with "logo" in class / alt / id / src
    for img_tag in soup.find_all("img"):
        src = img_tag.get("src") or img_tag.get("data-src") or ""
        alt = img_tag.get("alt") or ""
        cls = " ".join(img_tag.get("class") or [])
        tag_id = img_tag.get("id") or ""
        combined = (src + " " + alt + " " + cls + " " + tag_id).lower()
        if "logo" in combined and src:
            absolute = _normalize_url(src, brand_url) or src
            if absolute:
                return absolute

    # 2. apple-touch-icon (high-res, square — great for circular avatars)
    for rel_value in ("apple-touch-icon", "apple-touch-icon-precomposed"):
        link = soup.find("link", rel=lambda r: r and rel_value in (r if isinstance(r, list) else [r]))
        if link and link.get("href"):
            absolute = _normalize_url(link["href"], brand_url) or link["href"]
            if absolute:
                return absolute

    # 3. Standard favicon (SVG/PNG only — .ico files look bad at small sizes)
    for rel_value in ("icon", "shortcut icon"):
        link = soup.find("link", rel=lambda r: r and rel_value in (r if isinstance(r, list) else [r]))
        if link and link.get("href"):
            href = link["href"]
            if not href.lower().endswith(".ico"):
                absolute = _normalize_url(href, brand_url) or href
                if absolute:
                    return absolute

    return None
