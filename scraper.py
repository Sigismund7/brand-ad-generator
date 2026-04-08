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


def _is_svg_url(href: str) -> bool:
    u = href.lower().split("?")[0]
    return u.endswith(".svg") or u.endswith(".svgz")


def _is_likely_non_product_image(href: str, alt: str = "") -> bool:
    """True if URL/alt look like a brand mark, icon, or tiny asset — not the PDP hero."""
    u = href.lower()
    a = (alt or "").lower()
    if _is_svg_url(href):
        return True
    # Path / filename signals (avoid matching unrelated words: use path segments)
    bad_fragments = (
        "logo", "wordmark", "swoosh", "brand-mark", "brandmark", "/icons/", "/icon/",
        "favicon", "sprite", "placeholder", "pixel", "apple-touch", "badge",
        "/branding/", "/brand-", "-logo", "logo.", "/logos/", "og-image-brand",
        "social-share", "share-image-brand",
    )
    for frag in bad_fragments:
        if frag in u:
            return True
    bad_alt = ("logo", "brand logo", "icon", "wordmark", "badge")
    for b in bad_alt:
        if b in a and len(a) < 80:
            return True
    return False


def _img_intrinsic_area(img_tag) -> int:
    try:
        w = int(img_tag.get("width", 0) or 0)
        h = int(img_tag.get("height", 0) or 0)
        if w >= 200 and h >= 200:
            return w * h
    except (ValueError, TypeError):
        pass
    # Unknown dimensions — neutral area so URL heuristics still compete
    return 10_000


def _gallery_parent_bonus(img_tag) -> float:
    """Boost score when the <img> sits inside typical product / gallery markup."""
    bonus = 0.0
    p = img_tag.parent
    for _ in range(8):
        if p is None or not getattr(p, "name", None):
            break
        cls = " ".join(p.get("class") or []).lower()
        pid = (p.get("id") or "").lower()
        role = (p.get("role") or "").lower()
        blob = f"{cls} {pid} {role}"
        if p.get("itemprop") == "image":
            bonus = max(bonus, 45.0)
        if any(
            k in blob
            for k in (
                "product",
                "gallery",
                "pdp",
                "media-gallery",
                "product-media",
                "product__",
                "featured-image",
                "image-zoom",
            )
        ):
            bonus = max(bonus, 28.0)
        if p.name == "main":
            bonus = max(bonus, 12.0)
        p = p.parent
    return bonus


def extract_product_image_url(url: str) -> str | None:
    """
    Fetch a product page and return the best candidate **product hero** image URL.

    Does **not** return the first og:image blindly (often a brand/social image).
    Scores candidates: JSON-LD Product images and gallery <img> tags win over og/twitter
    when the latter look like logos or lack product context.
    """
    import json as _json

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    best_score: dict[str, float] = {}

    def add(u: str | None, score: float) -> None:
        if not u:
            return
        nu = _normalize_url(u, url) or u
        if not nu or _is_likely_non_product_image(nu, ""):
            return
        best_score[nu] = max(best_score.get(nu, 0.0), score)

    # --- schema.org Product JSON-LD (highest trust for PDP) ---
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = _json.loads(script.string or "")
            stack: list = data if isinstance(data, list) else [data]
            expanded: list = []
            for item in stack:
                if isinstance(item, dict) and "@graph" in item:
                    expanded.extend(item["@graph"] if isinstance(item["@graph"], list) else [item["@graph"]])
                else:
                    expanded.append(item)
            for item in expanded:
                if not isinstance(item, dict):
                    continue
                types = item.get("@type", "")
                if isinstance(types, str):
                    type_list = [types]
                elif isinstance(types, list):
                    type_list = [str(x) for x in types]
                else:
                    type_list = []
                if not any("Product" in t for t in type_list):
                    continue
                img = item.get("image")
                imgs: list = []
                if isinstance(img, str) and img:
                    imgs = [img]
                elif isinstance(img, list):
                    for el in img:
                        if isinstance(el, str):
                            imgs.append(el)
                        elif isinstance(el, dict):
                            uu = el.get("url") or el.get("contentUrl")
                            if uu:
                                imgs.append(uu)
                elif isinstance(img, dict):
                    uu = img.get("url") or img.get("contentUrl")
                    if uu:
                        imgs.append(uu)
                for iu in imgs:
                    if iu and not _is_likely_non_product_image(iu, ""):
                        add(iu, 100.0)
        except Exception:
            continue

    # --- <img itemprop="image"> (common on Shopify / structured PDPs) ---
    for img_tag in soup.find_all("img", itemprop="image"):
        src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-lazy-src") or ""
        if not src or _is_likely_non_product_image(src, img_tag.get("alt") or ""):
            continue
        area = _img_intrinsic_area(img_tag)
        add(src, 85.0 + min(area / 50_000.0, 30.0) + _gallery_parent_bonus(img_tag))

    # --- Gallery / product section images ---
    _PRODUCT_SIGNALS = ("product", "pdp", "/images/", "/img/", "/media/", "/photos/", "cdn", "assets")
    _SKIP = ("logo", "icon", "sprite", "placeholder", "blank", "pixel", "wordmark", "swoosh")
    for img_tag in soup.find_all("img"):
        src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-lazy-src") or ""
        if not src:
            continue
        sl = src.lower()
        if any(s in sl for s in _SKIP):
            continue
        if _is_likely_non_product_image(src, img_tag.get("alt") or ""):
            continue
        if not any(s in sl for s in _PRODUCT_SIGNALS):
            continue
        w = int(img_tag.get("width", 0) or 0)
        h = int(img_tag.get("height", 0) or 0)
        if w and h and (w < 200 or h < 200):
            continue
        area = _img_intrinsic_area(img_tag)
        base = 40.0 + min(area / 40_000.0, 35.0) + _gallery_parent_bonus(img_tag)
        add(src, base)

    # --- og / twitter (lower priority; skip if logo-like) ---
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        c = og["content"]
        if not _is_likely_non_product_image(c, ""):
            add(c, 42.0)

    tw = soup.find("meta", attrs={"name": "twitter:image"})
    if not tw:
        tw = soup.find("meta", property="twitter:image")
    if tw and tw.get("content"):
        c = tw["content"]
        if not _is_likely_non_product_image(c, ""):
            add(c, 40.0)

    if not best_score:
        return None

    return max(best_score.items(), key=lambda kv: kv[1])[0]


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
