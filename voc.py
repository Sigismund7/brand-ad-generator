"""
Voice-of-Customer (VoC) research module.

Gathers consumer language from three public sources:
  1. Reddit  — real discussions about the brand / product
  2. YouTube — comments on product review videos
  3. Google Autocomplete — what consumers search before buying

All three sources run concurrently.  Each one fails silently and logs
a warning string rather than crashing the pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import quote_plus

import requests

from models import GenerateRequest, VocSummary

_AUTOCOMPLETE_URL = "https://suggestqueries.google.com/complete/search"
_AUTOCOMPLETE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------

async def _fetch_reddit(
    brand_name: str,
    product_name: str,
    errors: list[str],
) -> list[str]:
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        errors.append("Reddit skipped: REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set.")
        return []

    try:
        import praw  # noqa: PLC0415
    except ImportError:
        errors.append("Reddit skipped: 'praw' package not installed.")
        return []

    def _sync_fetch() -> list[str]:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="brand-ad-generator/1.0 (by u/brand_ad_gen)",
        )
        query = f"{brand_name} {product_name}"
        snippets: list[str] = []
        try:
            for submission in reddit.subreddit("all").search(
                query, sort="relevance", limit=10
            ):
                # Include post title
                snippets.append(submission.title[:200])
                # Include top 3 comments
                submission.comments.replace_more(limit=0)
                for comment in list(submission.comments)[:3]:
                    if hasattr(comment, "body") and len(comment.body) > 20:
                        snippets.append(comment.body[:300])
        except Exception as exc:
            errors.append(f"Reddit partial error: {exc}")
        return snippets

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_fetch)


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

async def _fetch_youtube(
    brand_name: str,
    product_name: str,
    errors: list[str],
) -> list[str]:
    api_key = os.getenv("YOUTUBE_API_KEY", "")

    if not api_key:
        errors.append("YouTube skipped: YOUTUBE_API_KEY not set.")
        return []

    try:
        from googleapiclient.discovery import build  # noqa: PLC0415
        from googleapiclient.errors import HttpError  # noqa: PLC0415
    except ImportError:
        errors.append("YouTube skipped: 'google-api-python-client' package not installed.")
        return []

    def _sync_fetch() -> list[str]:
        snippets: list[str] = []
        try:
            youtube = build("youtube", "v3", developerKey=api_key)

            # Search for review videos
            search_resp = (
                youtube.search()
                .list(
                    q=f"{brand_name} {product_name} review",
                    part="id",
                    type="video",
                    maxResults=3,
                    relevanceLanguage="en",
                )
                .execute()
            )
            video_ids = [
                item["id"]["videoId"]
                for item in search_resp.get("items", [])
                if item["id"].get("videoId")
            ]

            for video_id in video_ids:
                try:
                    comments_resp = (
                        youtube.commentThreads()
                        .list(
                            videoId=video_id,
                            part="snippet",
                            maxResults=50,
                            order="relevance",
                            textFormat="plainText",
                        )
                        .execute()
                    )
                    for item in comments_resp.get("items", []):
                        text = (
                            item["snippet"]["topLevelComment"]["snippet"]
                            .get("textDisplay", "")
                        )
                        if len(text) > 20:
                            snippets.append(text[:400])
                except HttpError:
                    continue  # skip unavailable comment threads

        except HttpError as exc:
            errors.append(f"YouTube API error: {exc}")
        except Exception as exc:
            errors.append(f"YouTube unexpected error: {exc}")

        return snippets[:150]  # cap total comments

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_fetch)


# ---------------------------------------------------------------------------
# Google Autocomplete
# ---------------------------------------------------------------------------

async def _fetch_autocomplete(
    product_name: str,
    errors: list[str],
) -> list[str]:
    seeds = [
        f"{product_name} for people who",
        f"{product_name} vs",
        f"{product_name} is",
        f"best {product_name} for",
        f"why {product_name}",
    ]

    def _query(seed: str) -> list[str]:
        try:
            resp = requests.get(
                _AUTOCOMPLETE_URL,
                params={"q": seed, "client": "firefox", "hl": "en"},
                headers=_AUTOCOMPLETE_HEADERS,
                timeout=8,
            )
            data = json.loads(resp.text)
            # Response format: [query_string, [suggestion1, suggestion2, ...]]
            return data[1] if len(data) > 1 else []
        except Exception:
            return []

    def _sync_fetch() -> list[str]:
        results: list[str] = []
        for seed in seeds:
            results.extend(_query(seed))
        return list(dict.fromkeys(results))  # deduplicate, preserve order

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_fetch)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def gather_voc_async(
    request: GenerateRequest,
    errors: list[str],
) -> VocSummary:
    """
    Run all three VoC sources concurrently and return a VocSummary.
    Appends non-fatal warning strings to `errors`.
    """
    brand = request.brand_name or _extract_brand_name(request.brand_url)

    reddit_task = _fetch_reddit(brand, request.product_name, errors)
    youtube_task = _fetch_youtube(brand, request.product_name, errors)
    autocomplete_task = _fetch_autocomplete(request.product_name, errors)

    reddit_results, youtube_results, autocomplete_results = await asyncio.gather(
        reddit_task, youtube_task, autocomplete_task
    )

    return VocSummary(
        reddit_findings=reddit_results,
        youtube_findings=youtube_results,
        autocomplete_queries=autocomplete_results,
    )


def gather_voc(request: GenerateRequest, errors: list[str]) -> VocSummary:
    """Synchronous wrapper — safe to call from non-async code (including Streamlit threads)."""
    import concurrent.futures  # noqa: PLC0415

    try:
        asyncio.get_running_loop()
        # A loop is already running (rare, but guard against it)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, gather_voc_async(request, errors))
            return future.result()
    except RuntimeError:
        # No running loop — this is the normal Streamlit case; create a fresh one
        pass

    try:
        return asyncio.run(gather_voc_async(request, errors))
    except Exception as exc:
        errors.append(f"VoC gather failed: {exc}")
        return VocSummary()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_brand_name(brand_url: str) -> str:
    """Best-effort brand name extraction from a URL."""
    from urllib.parse import urlparse  # noqa: PLC0415
    host = urlparse(brand_url).hostname or ""
    # Strip www. and TLD
    parts = host.replace("www.", "").split(".")
    return parts[0] if parts else brand_url
