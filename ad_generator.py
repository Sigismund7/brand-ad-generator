"""
3-step Gemini pipeline for Meta ad generation.

Step 1 — Product Intelligence Extraction
    Input : scraped product page text
    Output: structured JSON (name, price, features, benefits, brand voice, etc.)

Step 2 — VoC Synthesis
    Input : Step 1 JSON + raw Reddit / YouTube / Autocomplete text
    Output: structured JSON (pain points, desired outcomes, exact phrases, persona)

Step 3 — Meta Ad Generation
    Input : Step 1 + Step 2 JSON
    Output: 3 AdVariation objects following the full copywriting constitution

Public entry point: generate_ads(request) -> AdOutput
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time

from google import genai
from google.genai import types as genai_types

from renderer import compose, merge_creative_spec_defaults

from models import (
    AdOutput,
    AdVariation,
    GenerateRequest,
    ProductNotFoundError,
    VocSummary,
)
from scraper import find_product_url, scrape_product_page
from voc import gather_voc

_MODEL = "gemini-2.5-flash"
_IMAGE_MODEL = "imagen-4.0-generate-001"

# Imagen GenerateImagesConfig.aspect_ratio must be one of: 1:1, 3:4, 4:3, 9:16, 16:9 (Gemini API docs).
# 4:5 (common Meta feed) is NOT accepted — use 1:1 for square (1080×1080-class) placements.
_IMAGEN_ASPECT_RATIO = "1:1"

# Matches ui/components.py char_badge(..., 500) for Primary Text — model often overshoots.
_PRIMARY_TEXT_MAX_CHARS = 500

log = logging.getLogger(__name__)

# region agent log
_DEBUG_AGENT_LOG = (
    "/Users/daschelgorgenyi/Desktop/Test project Speed run/brand-ad-generator/.cursor/debug-06d5a7.log"
)
_DEBUG_SESSION_ID = "06d5a7"


def _http_status_from_exception(exc: BaseException) -> int | None:
    """Best-effort HTTP status from google-genai / httpx-style exceptions."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    for _ in range(5):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        for name in ("status_code", "http_status", "status"):
            v = getattr(cur, name, None)
            if isinstance(v, int) and 100 <= v <= 599:
                return v
        code_attr = getattr(cur, "code", None)
        if code_attr is not None:
            if isinstance(code_attr, int) and 100 <= code_attr <= 599:
                return code_attr
            val = getattr(code_attr, "value", None)
            if isinstance(val, int) and 100 <= val <= 599:
                return val
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    msg = str(exc)
    m = re.search(r"\b(4\d{2}|5\d{2})\b", msg)
    if m:
        try:
            c = int(m.group(1))
            if 400 <= c <= 599:
                return c
        except ValueError:
            pass
    return None


def _debug_agent_log(
    hypothesis_id: str,
    message: str,
    data: dict,
) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": "ad_generator.py",
            "message": message,
            "data": data,
        }
        with open(_DEBUG_AGENT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


# endregion agent log


def _clamp_primary_text_for_ui(text: str, max_chars: int = _PRIMARY_TEXT_MAX_CHARS) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    sp = cut.rfind(" ")
    if sp > max_chars // 2:
        cut = cut[:sp].rstrip()
    return cut + "…"


# ---------------------------------------------------------------------------
# Gemini client — cached singleton
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file or the sidebar."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _transient_api_error(exc: BaseException) -> bool:
    """True if the error may succeed on retry (overload, rate limit, transient server errors)."""
    msg = f"{type(exc).__name__} {exc}".lower()
    if "503" in msg or "unavailable" in msg or "429" in msg:
        return True
    if "resource_exhausted" in msg or ("rate" in msg and "limit" in msg):
        return True
    if "too many requests" in msg or "overloaded" in msg or "try again" in msg:
        return True
    if "500" in msg or "502" in msg or "504" in msg:
        return True
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    return False


def _is_rate_limit_error(exc: BaseException) -> bool:
    """429 / quota / RESOURCE_EXHAUSTED — use longer backoff between retries."""
    msg = f"{type(exc).__name__} {exc}".lower()
    if "429" in msg:
        return True
    if "resource_exhausted" in msg:
        return True
    if "quota" in msg and ("exceed" in msg or "exceeded" in msg):
        return True
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code == 429:
        return True
    status = getattr(exc, "status", None)
    if isinstance(status, str) and "resource_exhausted" in status.lower():
        return True
    return False


def _retry_sleep_after_failure(exc: BaseException | None, attempt_index: int) -> float:
    """
    Seconds to wait before the next retry. attempt_index is 0 after the first failure.
    Rate limits get longer waits + jitter.
    """
    if exc is not None and _is_rate_limit_error(exc):
        base = 8.0 * (2 ** min(attempt_index, 5))
        return float(min(base + random.uniform(0.5, 4.0), 120.0))
    if exc is not None and _transient_api_error(exc):
        base = 1.5 * (2 ** min(attempt_index, 5))
        return float(min(base + random.uniform(0.2, 1.2), 45.0))
    base = 2.0 * (2 ** min(attempt_index, 4))
    return float(min(base + random.uniform(0.2, 1.0), 35.0))


def imagen_stagger_seconds() -> float:
    """Pause between each variation's Imagen call to reduce burst rate limits (env override)."""
    raw = os.getenv("IMAGEN_STAGGER_SECONDS", "5")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 5.0


def _call(client: genai.Client, system: str, user: str, *, max_attempts: int = 6) -> str:
    """Send a single turn and return the text response. Retries transient API errors."""
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=_MODEL,
                contents=f"SYSTEM:\n{system}\n\nUSER:\n{user}",
                config=genai_types.GenerateContentConfig(temperature=0.7),
            )
            return response.text.strip()
        except Exception as exc:
            last_exc = exc
            # region agent log
            st = _http_status_from_exception(exc)
            _debug_agent_log(
                "H3",
                "gemini_generate_content_failed",
                {
                    "http_status": st,
                    "exc_type": type(exc).__name__,
                    "transient": _transient_api_error(exc),
                    "attempt": attempt + 1,
                    "msg_excerpt": str(exc)[:280],
                },
            )
            # endregion agent log
            if not _transient_api_error(exc) or attempt >= max_attempts - 1:
                raise
            sleep_s = _retry_sleep_after_failure(exc, attempt)
            log.warning(
                "Gemini request failed (attempt %s/%s), retrying in %.1fs: %s",
                attempt + 1,
                max_attempts,
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _parse_json(raw: str) -> dict | list:
    """
    Extract and parse a JSON block from a Gemini response.
    Handles ```json ... ``` fences and bare JSON.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e1:
        # Last resort: find the first { or [ and try from there
        match = re.search(r"[{\[]", cleaned)
        if match:
            try:
                return json.loads(cleaned[match.start():])
            except json.JSONDecodeError:
                log.debug("JSON fallback parse failed. Raw prefix: %s", cleaned[:300])
                raise
        raise


# ---------------------------------------------------------------------------
# Step 1 — Product Intelligence
# ---------------------------------------------------------------------------

_STEP1_SYSTEM = """
You are a senior product analyst and brand strategist.
Your job is to read a scraped product page and extract structured intelligence
that a direct-response copywriter will use to write high-converting Meta ads.

Return ONLY valid JSON — no prose, no markdown, no explanation.
""".strip()

_STEP1_USER_TMPL = """
Product page content:
---
{page_content}
---

Extract the following and return as JSON:
{{
  "name": "product name",
  "price": "price or price range as a string",
  "key_features": ["feature 1", "feature 2", ...],
  "core_benefits": ["benefit 1", "benefit 2", ...],
  "brand_voice": "one sentence describing brand tone and style",
  "target_audience_signals": "who this product appears to be made for",
  "pain_points_solved": ["pain point 1", "pain point 2", ...],
  "social_proof_signals": ["any reviews, awards, press mentions, customer counts found"],
  "visual_appearance": "EXTREMELY detailed physical description — include ALL of: exact colors with specifics (not 'blue' but 'navy blue with teal accents'), all visible materials (mesh, leather, foam, metal, plastic, knit, suede), shape and silhouette, every distinctive visual feature (logos and their placement, patterns, stitching, textures, hardware, laces, zippers, buttons), proportions and form factor. Write it as if describing the product to a photographer who has never seen it and must recreate it visually. Example: 'low-profile running shoe with white engineered mesh upper, bright orange Nike React foam midsole visible from the lateral side, black rubber outsole with waffle-pattern traction, small black swoosh logo on outer lateral panel, white flat laces, padded collar with teal interior lining, rounded toe box, approximately standard men's shoe proportions'"
}}
""".strip()


def _step1_extract_product_intel(
    model: genai.Client,
    page_content: str,
) -> dict:
    user_prompt = _STEP1_USER_TMPL.format(page_content=page_content[:8000])
    raw = _call(model, _STEP1_SYSTEM, user_prompt)
    return _parse_json(raw)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Step 2 — VoC Synthesis
# ---------------------------------------------------------------------------

_STEP2_SYSTEM = """
You are a consumer research analyst and direct-response copywriter.
You have been given raw, unfiltered consumer language collected from Reddit,
YouTube comments, and Google search autocomplete.

Your job is to synthesise this into a compact brief that a copywriter can use
immediately — without doing any more research — to write ads that sound like
they were written by someone who deeply understands this customer.

Return ONLY valid JSON — no prose, no markdown, no explanation.
""".strip()

_STEP2_USER_TMPL = """
PRODUCT INTELLIGENCE (from Step 1):
{product_intel}

RAW CONSUMER LANGUAGE:

Reddit discussions ({reddit_count} snippets):
{reddit_text}

YouTube review comments ({youtube_count} snippets):
{youtube_text}

Google Autocomplete suggestions:
{autocomplete_text}

Synthesise the above into this JSON structure:
{{
  "top_pain_points": [
    "3-6 specific frustrations real people have, in their own words"
  ],
  "top_desired_outcomes": [
    "3-6 specific results or feelings people want, in their own words"
  ],
  "exact_phrases_to_use": [
    "8-12 verbatim or near-verbatim phrases lifted from the consumer text that
     would resonate in ad copy — conversational, emotional, specific"
  ],
  "primary_persona": "One sentence: who this person is, their defining struggle, and what they want",
  "common_objections": [
    "3-5 doubts or barriers that come up before purchase"
  ]
}}

If consumer data is sparse or missing for a source, draw on the product
intelligence to make reasonable inferences, but prefer real consumer language
where available.
""".strip()


def _step2_synthesise_voc(
    model: genai.GenerativeModel,
    product_intel: dict,
    voc: VocSummary,
) -> dict:
    def _join(items: list[str], limit: int = 60) -> str:
        return "\n".join(f"- {s[:300]}" for s in items[:limit]) or "(none)"

    user_prompt = _STEP2_USER_TMPL.format(
        product_intel=json.dumps(product_intel, indent=2),
        reddit_count=len(voc.reddit_findings),
        reddit_text=_join(voc.reddit_findings),
        youtube_count=len(voc.youtube_findings),
        youtube_text=_join(voc.youtube_findings),
        autocomplete_text=_join(voc.autocomplete_queries, limit=30),
    )
    raw = _call(model, _STEP2_SYSTEM, user_prompt)
    result = _parse_json(raw)  # type: ignore[assignment]
    # Attach persona back to VocSummary
    if isinstance(result, dict):
        voc.synthesized_persona = result.get("primary_persona", "")
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Product image download (scraped asset for compositor)
# ---------------------------------------------------------------------------

def _download_image_bytes(url: str) -> bytes | None:
    """Download an image from a URL and return raw bytes, or None on failure."""
    import requests as _requests

    try:
        resp = _requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        log.warning("Could not download product image from %s: %s", url, exc)
        return None


def _parse_creative_spec_field(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            log.warning("creative_spec string was not valid JSON")
    return {}


# ---------------------------------------------------------------------------
# Image brief (Gemini) → Imagen prompt
# ---------------------------------------------------------------------------

_IMAGE_BRIEF_SYSTEM = """
You are an elite creative director specializing in Meta (Facebook/Instagram)
static ad creatives for e-commerce brands. You direct product photography
for cold-prospecting feed ads that must stop the scroll and convert.

You understand how Meta's Andromeda algorithm reads creative signals and
how images perform in the mobile feed environment.

Your job: given product intelligence and an ad angle, output a structured
JSON image brief that a photographer (or image generation model) can execute.

═══════════════════════════════════════════════════════════════
META AD CREATIVE PERFORMANCE RULES (non-negotiable)
═══════════════════════════════════════════════════════════════

SCROLL-STOPPING PRINCIPLES:
  • The product must be identifiable within 0.3 seconds at thumbnail size
    on a mobile phone. If someone scrolling at speed can't tell what the
    product is, the ad fails before the copy is even read.
  • High contrast between the product and its background. The product must
    visually "pop" against whatever is behind it. Dark product = light
    background. Light product = dark background. Neutral product = bold
    colored or textured background.
  • 1:1 square format (1080x1080) works across Meta Feed and many placements.
    Compose for a square frame: hero product centered with strong thumbnail
    readability.
  • The product or key visual must read clearly in the top/center of the
    square — assume mobile thumbnail crop is tight.
  • Static images still drive 60-70% of conversions on Meta. These briefs
    are for static product photography, not video.

PRODUCT PHOTOGRAPHY RULES:
  • THE PRODUCT IS ALWAYS THE SUBJECT. Every composition decision exists
    to showcase the product. The product is never a secondary element.
  • Product must fill at least 50% of the frame. Bigger is almost always
    better for feed ads. A tiny product in a vast landscape = invisible.
  • Product must be tack-sharp. If anything has shallow depth of field,
    it's the background, never the product.
  • Show the product from an angle that reveals its distinctive features,
    but AVOID the default "product page hero" crop when possible — pick a
    camera height, distance, or environmental context that would NOT match
    a typical e-commerce packshot. For shoes, consider low hero, asymmetric
    framing, or in-context wear vs. the usual lateral studio hero.
  • Lighting must reveal the product's materials, textures, and colors
    accurately. A matte product needs soft diffused light. A glossy
    product needs a key light plus fill to show the sheen without
    blowing out highlights.

PEOPLE IN ADS:
  • People are OPTIONAL. Many top-performing product ads have no people.
  • When people appear, they are there to demonstrate the product in use.
    The camera focuses on the product, not the person.
  • For wearable products (shoes, clothing, watches, headphones): show
    the body part wearing/using the product. Feet for shoes. Wrist for
    watch. Torso for jacket. The product is the focal point.
  • NEVER generate a portrait or headshot as a product ad. A person's
    face looking at the camera is a brand awareness ad, not a product ad.
  • If a person's face is partially visible, it must be out of focus or
    at the edge of frame. The eye is drawn to faces — if a face is sharp
    and centered, the product becomes secondary.

WHAT META'S ALGORITHM REWARDS:
  • Visual diversity across ad variations is mandatory: three angles must
    yield three COMPLETELY different creative concepts — not crops of the
    same shot. See VISUAL DIVERSITY ACROSS THE 3 VARIATIONS below.
  • Clean images with minimal text overlay. The algorithm's computer vision
    deprioritizes cluttered, text-heavy images. Keep the image clean —
    copy goes in the text fields, not on the image.
  • Images that blend with organic content. Feed ads that look like
    polished versions of organic posts outperform ads that look like ads.
  • High resolution, correct aspect ratio. Anything below 1080px wide or
    with wrong aspect ratio gets deprioritized.

CREATIVE FREEDOM:
  • Do NOT default to white/beige studio backgrounds. Those are for
    product pages, not ads. Ads need to stop the scroll.
  • Use interesting environments: textured surfaces, real-world locations,
    atmospheric scenes, colored lighting, dramatic settings.
  • Be bold with lighting. Side light, rim light, colored gels, golden
    hour, blue hour, neon reflections, spotlight in darkness. Flat even
    lighting is boring.
  • Be bold with angles. Low angles make products look powerful. Overhead
    flat lays feel editorial. Dynamic tilted angles feel energetic.
    Don't default to straight-on eye level.
  • The product should still be the hero and clearly visible, but the
    CONTEXT around it should be visually interesting and scroll-stopping.
  • Think about what makes someone stop scrolling on Instagram. It's
    never a product floating on white. It's a striking image with mood,
    atmosphere, and visual tension.

VISUAL DIVERSITY ACROSS THE 3 VARIATIONS (CRITICAL):
  The three ad angles MUST produce three completely different images.
  Not variations of the same shot — entirely different creative concepts.

  Pain Point: Dark, dramatic, high-contrast. The product in a moody,
  intense environment. Think: gritty urban surface, dramatic shadows,
  noir lighting, the product looking powerful and serious. OR the product
  in an action context that implies the pain (cracked ground, harsh
  terrain, extreme conditions).

  Aspiration: Warm, beautiful, scroll-stopping — but still a PRODUCT ad.
  The sellable item stays huge, sharp, and unmistakable (same 0.3s thumbnail
  rule). Think: golden hour outdoors, beautiful environment, warm tones.
  If a person appears, they demonstrate the product (feet for footwear,
  hands for handheld) — NEVER a face-forward portrait or fashion shot where
  the product is missing, tiny, or off-frame. "Lifestyle" must not replace
  product visibility.

  Social Proof: Clean, approachable, real. The product in an everyday
  context that feels authentic. Think: on a doorstep being unboxed, on
  a desk next to everyday objects, casually placed in a lived-in space,
  natural light, the product looking accessible and trustworthy.

  If the three images could be mistaken for the same photo with different
  crops, you have failed. Each must tell a different visual story.

ORIGINALITY — NOT A STORE PACKSHOT OR REFERENCE RESHOOT:
  • You have NOT seen the brand's on-site product photos. Do NOT aim to
    recreate the exact composition, crop, lighting, or "listing hero" look
    of a product-detail page — that reads as a copy of the reference image.
  • Invent a NEW advertising photograph: fresh environment, camera
    relationship, and framing, while keeping colors, materials, and iconic
    product traits accurate vs. visual_appearance.
  • The "subject" field must DISTILL the product in 2–4 tight sentences
    (key colors, materials, logos/placement, silhouette). Do NOT paste
    visual_appearance verbatim — long pasted prose encourages copycat renders.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — JSON IMAGE BRIEF
═══════════════════════════════════════════════════════════════

Return ONLY valid JSON. No prose, no markdown, no explanation.

The "technical" object must use valid JSON: product_frame_pct is an integer between 50 and 70.

{
  "subject": "Tight 2–4 sentence hero description: product name plus distilled
    colors, materials, logos/placement, silhouette from visual_appearance — accurate
    but NOT a verbatim paste of the full paragraph (verbatim encourages catalog
    copycat shots). This is the most important field.",
  "product_angle": "Which angle/side of the product to show (e.g. '3/4 lateral view
    showing the side profile', 'front-facing hero shot', 'top-down flat lay',
    'low angle looking up at the product')",
  "scene": "Where the product is placed — be specific about the surface, environment,
    and spatial context (e.g. 'on rough concrete pavement with visible texture',
    'on a sunlit wooden deck at golden hour overlooking green hills', 'on a lived-in
    oak desk beside a laptop and coffee mug, soft window light')",
  "lighting": "Specific lighting setup using photography language (e.g. 'dramatic
    rim light from upper right with soft fill from left, hard shadows', 'natural
    golden hour sunlight from behind and left, warm color temperature 5500K',
    'soft diffused window light from the right, gentle shadows')",
  "camera": "Camera settings and framing (e.g. 'shot at f/2.8, product fills 65%
    of frame, slight low angle at 10 degrees', 'overhead flat lay shot, f/5.6,
    product centered', 'eye-level, f/1.8 shallow DOF, product sharp, background
    creamy bokeh')",
  "mood": "2-3 word mood label (e.g. 'bold premium', 'warm aspirational',
    'clean trustworthy', 'raw dramatic')",
  "color_palette": "Dominant colors in the image (e.g. 'dark charcoal background
    with warm orange accent from the product', 'soft cream and natural wood tones
    with green foliage', 'deep teal shadows with amber rim light on the product')",
  "people": "none OR a specific description of how a person interacts with the
    product (e.g. 'none', 'only feet visible in black athletic leggings, wearing
    the shoe, mid-stride on pavement', 'hands holding the product, no face visible').
    NEVER write a portrait description.",
  "props": "Other objects in the scene that add context without competing with the
    product (e.g. 'none — product only', 'scattered autumn leaves on the ground',
    'a gym bag and water bottle blurred in background', 'coffee mug and notebook
    nearby, slightly out of focus')",
  "do_not_include": "Explicit list of things to exclude (always include: 'text
    overlays, logos, watermarks, brand names on image, replica of e-commerce packshot
    or product-page hero framing'. Add angle-specific exclusions like 'no faces' or
    'no other products')",
  "technical": {
    "aspect_ratio": "1:1",
    "resolution": "1080x1080",
    "format": "photorealistic commercial product photography",
    "product_frame_pct": 60
  }
}
""".strip()

_IMAGE_BRIEF_USER_TMPL = """
PRODUCT INTELLIGENCE:
{product_intel}

AD VARIATION:
- Angle: {angle}
- Format label (from copy deck): {format_type}
- Headline: {headline}
- Primary text hook: {hook}

IMPORTANT: This is the {angle} angle. The image must look COMPLETELY
DIFFERENT from the other two angles (Pain Point, Aspiration, Social Proof).
Do not generate a plain studio product photo. Create a visually striking,
scroll-stopping ad image with a real environment and interesting lighting.

If the format label is "Flat Lay", the photograph must be a top-down or
table-style shot where the actual product (from product intelligence) is
the arranged hero on a surface — not a portrait, not a generic outfit or
office-fashion scene unless the product is apparel.

The final image must feel like a NEW campaign photoshoot — NOT a reshoot or
digital twin of typical on-site product listing / PDP "hero" photography.

Generate the image brief JSON for this ad variation. The product must match
visual_appearance for accuracy, but distill that into the "subject" field —
do not paste the full visual_appearance text verbatim. The scene and mood
should align with the {angle} angle.

Use visual_appearance as the source of truth for product accuracy; use
product_angle, scene, and lighting to make the shot unmistakably different
from a catalog reference frame.
""".strip()

_FOOTWEAR_HINTS = frozenset(
    (
        "shoe",
        "sneaker",
        "boot",
        "footwear",
        "trainer",
        "cleat",
        "sandal",
        "slide",
        "pegasus",  # common Nike line name in running copy
    )
)


def _is_footwear_product(product_intel: dict) -> bool:
    blob = (
        f"{product_intel.get('name') or ''} "
        f"{product_intel.get('visual_appearance') or ''} "
        f"{product_intel.get('category') or ''}"
    ).lower()
    return any(h in blob for h in _FOOTWEAR_HINTS)


def _people_field_risky_for_product_ad(people: str) -> bool:
    pl = (people or "").lower()
    if not pl or pl == "none":
        return False
    risky = (
        "face",
        "portrait",
        "headshot",
        "head ",
        "shoulders",
        "torso",
        "upper body",
        "waist",
        "looking at camera",
        "model",
        "blazer",
        "turtleneck",
    )
    return any(x in pl for x in risky)


def _normalize_image_brief(brief: dict, product_intel: dict, variation: dict) -> dict:
    """Tighten LLM briefs so the hero SKU stays visible (fixes e.g. footwear + aspiration portraits)."""
    if not isinstance(brief, dict):
        return brief

    angle = (variation.get("angle") or "").strip()
    footwear = _is_footwear_product(product_intel)
    people = str(brief.get("people") or "").strip()
    people_l = people.lower()

    dni = str(brief.get("do_not_include") or "")
    extra_exclude = (
        "face-centered portrait or fashion editorial where the hero product is absent, "
        "tiny, or cropped out; office or formal outfit focus without the product visible"
    )
    if extra_exclude not in dni:
        brief["do_not_include"] = f"{dni}, {extra_exclude}" if dni else extra_exclude

    if footwear:
        shoe_extra = (
            "upper-body or head-and-shoulders shots without the shoes clearly visible; "
            "generic lifestyle model not wearing or displaying the footwear"
        )
        if shoe_extra not in brief["do_not_include"]:
            brief["do_not_include"] = f"{brief['do_not_include']}, {shoe_extra}"

    feet_wearing_shoes = (
        "only feet and lower legs in athletic wear, clearly wearing the product "
        "shoes, mid-stride or planted on the ground; shoes tack-sharp and large in "
        "frame; no face, no upper body, no office or fashion styling"
    )
    if footwear and _people_field_risky_for_product_ad(people):
        brief["people"] = feet_wearing_shoes
    elif footwear and angle == "Aspiration" and (not people or people_l == "none"):
        brief["people"] = feet_wearing_shoes
    elif _people_field_risky_for_product_ad(people):
        brief["people"] = (
            "hands interacting with the product or holding it; no face-centered portrait"
        )

    tech = brief.get("technical")
    if not isinstance(tech, dict):
        tech = {}
        brief["technical"] = tech
    # Align with Imagen API (1:1 only — brief text may still mention legacy 4:5).
    tech["aspect_ratio"] = "1:1"
    tech["resolution"] = "1080x1080"

    if footwear and angle == "Aspiration":
        try:
            pct = int(tech.get("product_frame_pct", 0))
        except (TypeError, ValueError):
            pct = 0
        if pct < 58:
            tech["product_frame_pct"] = 60

    return brief


def _fallback_image_brief(product_intel: dict, variation: dict) -> dict:
    """Non-studio defaults when Gemini brief generation fails; varies by ad angle."""
    name = product_intel.get("name", "product")
    vis = product_intel.get("visual_appearance", "")
    if vis:
        vis_tight = vis if len(vis) <= 280 else vis[:277].rsplit(" ", 1)[0] + "…"
        subj = (
            f"{name}: {vis_tight} "
            f"(new campaign shoot — not a store packshot reproduction)."
        )
    else:
        subj = name
    angle = (variation.get("angle") or "Pain Point").strip()

    if angle == "Aspiration":
        return {
            "subject": subj,
            "product_angle": "dynamic 3/4 view showing distinctive features",
            "scene": (
                "on a sun-warmed coastal path at golden hour, blurred ocean and sky behind, "
                "real outdoor athletic context — not a studio"
            ),
            "lighting": (
                "warm golden-hour key from camera left, soft rim from behind, "
                "rich amber tones — not flat softbox"
            ),
            "camera": "low angle f/2.8, product fills ~60% of frame, background creamy bokeh",
            "mood": "warm aspirational",
            "color_palette": "amber, teal shadows, natural earth tones",
            "people": "none",
            "props": "subtle sand or trail texture at edge of frame, out of focus",
            "do_not_include": (
                "text overlays, logos, watermarks, plain white seamless backdrop, "
                "gray studio sweep, catalog floating product, PDP hero crop replica"
            ),
            "technical": {
                "aspect_ratio": "1:1",
                "resolution": "1080x1080",
                "product_frame_pct": 62,
                "format": "editorial outdoor advertising photograph",
            },
        }
    if angle == "Social Proof":
        return {
            "subject": subj,
            "product_angle": "clear hero angle appropriate to the product category",
            "scene": (
                "on a real oak desk beside a mug and notebook, soft daylight from a window — "
                "lived-in authentic home or office, not a studio set"
            ),
            "lighting": "soft natural window light from left, gentle shadows, ~4500K",
            "camera": "eye-level f/2.8, product sharp, props slightly defocused",
            "mood": "clean trustworthy",
            "color_palette": "warm neutrals, natural wood, soft whites — not stark clinical white",
            "people": "none",
            "props": "everyday objects blurred in background",
            "do_not_include": (
                "text overlays, logos, watermarks, plain white seamless, empty pure white void, "
                "product listing packshot framing"
            ),
            "technical": {
                "aspect_ratio": "1:1",
                "resolution": "1080x1080",
                "product_frame_pct": 58,
                "format": "authentic lifestyle product photograph",
            },
        }
    # Pain Point (default)
    return {
        "subject": subj,
        "product_angle": "powerful low 3/4 angle emphasizing form",
        "scene": (
            "on cracked dark asphalt with gritty texture, moody urban night-adjacent feel — "
            "environment suggests struggle or intensity, not a white studio"
        ),
        "lighting": (
            "hard side light with deep shadows, cool rim from behind, high contrast — "
            "no flat even lighting"
        ),
        "camera": "low angle f/2.8, product dominates frame, background falls off dark",
        "mood": "raw dramatic",
        "color_palette": "charcoal, deep blue shadows, product colors as accents",
        "people": "none",
        "props": "none — texture from ground only",
        "do_not_include": (
            "text overlays, logos, watermarks, white seamless, bright catalog backdrop, "
            "e-commerce hero image reshoot"
        ),
        "technical": {
            "aspect_ratio": "1:1",
            "resolution": "1080x1080",
            "product_frame_pct": 65,
            "format": "dramatic editorial advertising photograph",
        },
    }


def _generate_image_brief(
    client: genai.Client,
    product_intel: dict,
    variation: dict,
) -> dict:
    """Ask Gemini to create a structured image brief for one ad variation."""
    hook = (variation.get("primary_text") or "")[:120]
    user_prompt = _IMAGE_BRIEF_USER_TMPL.format(
        product_intel=json.dumps(product_intel, indent=2),
        angle=variation.get("angle", "Pain Point"),
        format_type=(variation.get("format_type") or "").strip() or "(not specified)",
        headline=variation.get("headline", ""),
        hook=hook,
    )
    try:
        raw = _call(client, _IMAGE_BRIEF_SYSTEM, user_prompt)
        brief = _parse_json(raw)
        if isinstance(brief, dict):
            return _normalize_image_brief(brief, product_intel, variation)
    except Exception as exc:
        log.warning("Image brief generation failed: %s", exc)

    return _normalize_image_brief(
        _fallback_image_brief(product_intel, variation),
        product_intel,
        variation,
    )


def _brief_to_imagen_prompt(brief: dict, product_intel: dict | None = None) -> str:
    """Convert a structured image brief JSON into an Imagen prompt string."""
    parts = []

    # Lead with hero + anti-catalog cues — early tokens matter most for Imagen
    subject = (brief.get("subject") or "product").strip()
    # Very long "subject" text tracks PDP prose and encourages copycat framing
    if len(subject) > 500:
        cut = subject[:497]
        subject = (cut.rsplit(" ", 1)[0] + "…") if " " in cut else cut + "…"

    parts.append(
        f"Original environmental advertising photograph for a social campaign: "
        f"{subject} is the unmistakable hero — shot on location with real-world "
        f"context and mood. This is NOT a reshoot of online store or product-listing "
        f"packshot photography, NOT the same crop or composition as a PDP hero image, "
        f"and not a white seamless e-commerce studio sweep."
    )

    if product_intel and _is_footwear_product(product_intel):
        parts.append(
            "Footwear rule: the shoes must be unmistakably visible—worn on feet from the "
            "ankles down or as a large product hero. Do not depict a portrait, office outfit, "
            "or fashion editorial where the footwear is absent, tiny, or cropped out."
        )

    # Product angle
    product_angle = brief.get("product_angle", "")
    if product_angle:
        parts.append(f"Shown from {product_angle}.")

    # Scene and environment
    scene = brief.get("scene", "")
    if scene:
        parts.append(f"The product is placed {scene}.")

    # People (or lack thereof)
    people = brief.get("people", "none")
    if people and people.lower() != "none":
        parts.append(f"{people}.")
    else:
        parts.append("No people in the image.")

    # Lighting
    lighting = brief.get("lighting", "")
    if lighting:
        parts.append(f"Lighting: {lighting}.")

    # Camera
    camera = brief.get("camera", "")
    if camera:
        parts.append(f"Camera: {camera}.")

    # Props
    props = brief.get("props", "")
    if props and props.lower() not in ("none", "none — product only", ""):
        parts.append(f"Scene includes {props}.")

    # Color palette
    palette = brief.get("color_palette", "")
    if palette:
        parts.append(f"Color palette: {palette}.")

    # Mood
    mood = brief.get("mood", "")
    if mood:
        parts.append(f"Overall mood: {mood}.")

    # Technical requirements
    tech = brief.get("technical", {}) if isinstance(brief.get("technical"), dict) else {}
    pct = tech.get("product_frame_pct", 60)
    if not isinstance(pct, int):
        try:
            pct = int(pct)
        except (TypeError, ValueError):
            pct = 60
    parts.append(f"The product fills {pct}% of the frame.")

    parts.append(
        "Match the product's true colors and distinctive traits above — without "
        "duplicating stock listing composition, symmetric packshot layout, or "
        "typical marketplace hero framing."
    )

    parts.append(
        "Avoid plain white or light gray seamless backdrops, empty void backgrounds, "
        "and catalog floating-product compositions."
    )

    # Exclusions
    exclude = brief.get(
        "do_not_include",
        "text overlays, logos, watermarks, e-commerce packshot replica framing",
    )
    parts.append(f"Do not include: {exclude}.")

    # Format — match Imagen 1:1 (square) placement
    fmt = tech.get("format", "editorial advertising photograph for social feeds")
    ar = tech.get("aspect_ratio", "1:1")
    res = tech.get("resolution", "1080x1080")
    parts.append(
        f"{fmt}. Square {ar} aspect ratio ({res}), scroll-stopping social ad composition."
    )

    return " ".join(parts)


def _generate_ad_image(
    client: genai.Client,
    product_intel: dict,
    variation: dict,
) -> str | None:
    """Generate an ad image: brief -> prompt -> Imagen. Returns base64 JPEG or None."""
    import base64

    # Step A: Gemini generates the structured brief
    brief = _generate_image_brief(client, product_intel, variation)

    # Step B: Convert brief to Imagen prompt
    prompt = _brief_to_imagen_prompt(brief, product_intel)

    angle = variation.get("angle", "")
    print(f"[IMAGE] angle={angle}")
    print(f"[IMAGE] brief subject={brief.get('subject', 'MISSING')!r}")
    print(f"[IMAGE] brief scene={brief.get('scene', '')!r}")
    print(f"[IMAGE] prompt length={len(prompt)} chars")

    # Step C: Call Imagen (retry transient overload / rate limits)
    img_config = genai_types.GenerateImagesConfig(
        number_of_images=1,
        aspect_ratio=_IMAGEN_ASPECT_RATIO,
        output_mime_type="image/jpeg",
        person_generation="ALLOW_ADULT",
    )

    max_attempts = 6
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            # region agent log
            _debug_agent_log(
                "H1",
                "imagen_request_start",
                {
                    "attempt": attempt + 1,
                    "model": _IMAGE_MODEL,
                    "aspect_ratio": _IMAGEN_ASPECT_RATIO,
                    "prompt_len": len(prompt),
                    "person_generation": getattr(img_config, "person_generation", None),
                },
            )
            # endregion agent log
            response = client.models.generate_images(
                model=_IMAGE_MODEL,
                prompt=prompt,
                config=img_config,
            )
            if response.generated_images:
                raw = response.generated_images[0].image.image_bytes
                # region agent log
                _debug_agent_log(
                    "H1",
                    "imagen_success",
                    {
                        "attempt": attempt + 1,
                        "bytes_len": len(raw) if raw else 0,
                    },
                )
                # endregion agent log
                return base64.b64encode(raw).decode("utf-8")
            log.warning(
                "[IMAGE] Imagen returned no images for %s (attempt %s/%s)",
                angle,
                attempt + 1,
                max_attempts,
            )
            if attempt >= max_attempts - 1:
                break
            sleep_s = _retry_sleep_after_failure(None, attempt)
            log.warning("[IMAGE] Imagen empty response, sleeping %.1fs", sleep_s)
            time.sleep(sleep_s)
            continue
        except Exception as exc:
            last_exc = exc
            print(f"[IMAGE] Imagen generation failed for {angle}: {exc}")
            # region agent log
            st = _http_status_from_exception(exc)
            tr = _transient_api_error(exc)
            _debug_agent_log(
                "H2" if st == 400 else "H1",
                "imagen_generate_images_failed",
                {
                    "http_status": st,
                    "exc_type": type(exc).__name__,
                    "transient_classified": tr,
                    "attempt": attempt + 1,
                    "will_retry": tr and attempt < max_attempts - 1,
                    "msg_excerpt": str(exc)[:400],
                },
            )
            _debug_agent_log(
                "H5",
                "transient_vs_400_check",
                {
                    "http_status": st,
                    "transient_classified": tr,
                    "status_is_400": st == 400,
                    "status_is_503": st == 503,
                },
            )
            # endregion agent log
            if not _transient_api_error(exc) or attempt >= max_attempts - 1:
                break
            sleep_s = _retry_sleep_after_failure(exc, attempt)
            log.warning(
                "[IMAGE] Imagen retry in %.1fs (attempt %s/%s): %s",
                sleep_s,
                attempt + 1,
                max_attempts,
                exc,
            )
            time.sleep(sleep_s)
            continue

    if last_exc:
        log.warning("[IMAGE] Imagen gave up for %s: %s", angle, last_exc)
        # region agent log
        st = _http_status_from_exception(last_exc)
        _debug_agent_log(
            "H4",
            "imagen_gave_up_final",
            {
                "http_status": st,
                "exc_type": type(last_exc).__name__,
                "msg_excerpt": str(last_exc)[:400],
            },
        )
        # endregion agent log
    return None


def _render_ad_creative(
    creative_spec: dict,
    product_image_bytes: bytes | None,
    logo_bytes: bytes | None,
    *,
    product_image_url: str | None = None,
) -> str | None:
    """Compose a final ad image from the creative spec and scraped product image."""
    if not product_image_bytes:
        return None
    try:
        from io import BytesIO

        from PIL import Image

        _im = Image.open(BytesIO(product_image_bytes))
        _dims = _im.size
    except Exception:
        _dims = None
    print(
        f"[ad_generator] _render_ad_creative product_image_url={product_image_url!r} "
        f"product_bytes_len={len(product_image_bytes)} logo_bytes_len={len(logo_bytes or b'')}"
    )
    print(f"[ad_generator] product_image dimensions before compose: {_dims}")
    try:
        return compose(
            spec=creative_spec,
            product_image=product_image_bytes,
            logo_image=logo_bytes,
        )
    except ValueError as exc:
        log.warning("Ad composition skipped: %s", exc)
        return None
    except Exception as exc:
        log.warning("Ad composition failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Step 3 — Meta Ad Generation (Copywriting Constitution)
# ---------------------------------------------------------------------------

_STEP3_SYSTEM = """
You are one of the world's best direct-response copywriters AND a senior
performance creative strategist specialising in high-converting static Meta
(Facebook / Instagram) ads for cold prospecting audiences.

You follow every rule in the copywriting constitution (Part 1) and output a
structured creative_spec JSON object (Part 2) for each variation without exception.

═══════════════════════════════════════════════════════════════
PART 1 — COPYWRITING CONSTITUTION
═══════════════════════════════════════════════════════════════

━━━ THE HOOK IS EVERYTHING ━━━
The hook is the first line of primary_text. If it fails, the ad is invisible.
You have under 3 seconds. The hook must land entirely within the first
80–115 characters. The first 5 words must front-load the benefit or tension.

SIX HOOK FORMULAS — use a different one for each variation:

1. Pain-Point Question — get inside their head with a question they're asking
   "Tired of morning puffiness?"
   "Still waking up with back pain every single day?"
   "Why does every moisturiser leave your skin greasy by noon?"

2. Bold Outcome Statement — lead with the result, not the product
   "Wake up glowing. Every single morning."
   "Flat-to-voluminous hair in 60 seconds."
   "Finally, a wallet that fits in your front pocket."

3. Specific Proof Hook — open with a number or data point
   "12,000+ five-star reviews can't be wrong."
   "Sold out 3x. Back in stock TODAY."
   "Results in 14 days — or your money back."

4. Curiosity Gap — leave something unsaid that makes them need to know more
   "The one thing dermatologists won't tell you about SPF."
   "We almost didn't launch this product."
   "This $29 tool replaced my entire skincare shelf."

5. Us vs. Them — position against the status quo
   "Your old blender wishes it could do this."
   "What $200 serums do — for $34."

6. Social Proof Lead — let a customer speak first
   "'I didn't think it would work — until it did.' — Sarah, verified buyer"
   "'Best purchase I made all year.' 4.9★ from 8,400 reviews."

HOOK RULES (non-negotiable):
  • Always product-led, never a generic slogan ("Quality you can trust" = skip)
  • Must connect to a real pain point, desire, or proof element
  • If the hook would work with a competitor's name swapped in, rewrite it
  • Never start with the brand name — lead with value, not the logo

━━━ COPYWRITING FRAMEWORKS ━━━
Pick ONE framework per variation. Do not overstuff or mix frameworks.

PAS — Problem, Agitate, Solve
  Best for: Pain Point angle, cold audiences who already feel the pain
  Problem:  Name the pain the audience already feels
  Agitate:  Make it worse — describe consequences of inaction
  Solve:    Present the product as the clear, simple fix
  Example: "Tired of waking up with back pain? (P) It's not just discomfort —
  it's ruining your sleep, mood, and whole day. (A) Our orthopedic pillow is
  designed by spine specialists to fix alignment overnight. (S)"

BAB — Before, After, Bridge
  Best for: Aspiration angle, transformation products
  Before:   Describe the current frustrating state
  After:    Paint the desired outcome vividly
  Bridge:   Position the product as the path between the two
  Example: "Dull, tired-looking skin every morning. (B) Imagine waking up with
  a genuine glow — no filter needed. (A) Our Vitamin C serum bridges the gap
  with clinical-strength brightening in just 2 weeks. (Br)"

AIDA — Attention, Interest, Desire, Action
  Best for: Mid-to-high ticket, multi-step funnels, longer primary text
  Attention: Hook with bold statement or question
  Interest:  Relatable benefit or surprising fact
  Desire:    Emotional want — transformation or social proof
  Action:    Clear, specific CTA

FAB — Features, Advantages, Benefits
  Best for: Product-forward ads, technical products, comparison ads
  Feature:     What the product has or does
  Advantage:   Why that feature matters vs. alternatives
  Benefit:     How it improves the buyer's life
  Example: "100% organic bamboo fabric. (F) 3x softer than cotton, naturally
  antibacterial. (A) Sheets that feel like a luxury hotel and stay fresh all
  week. (B)"

Value Stack
  Best for: Justifying price, bundles, offers
  List what they get — product, bonuses, guarantees, savings
  Make total perceived value far exceed the price
  Close with "All for just $X" or "Try it risk-free"

━━━ TONE & VOICE ━━━
  • Write like a smart friend texting a recommendation, not a brand announcing
  • Second person always: "you" and "your" — never "our customers" or "one"
  • Short sentences. Fragments are fine. Rhythm matters more than grammar rules
  • Conversational over corporate: "This stuff actually works" not "Experience
    our innovative solution"
  • Never start with "we" in the first line — the ad is about THEM, not you
  • Never use superlatives without proof: "best ever" = empty claim
  • Emojis: 1–2 max, purposeful only (never decorative, never a wall of them)
  • No ALL CAPS sentences; one word for emphasis is fine

━━━ FIELD-LEVEL RULES ━━━
primary_text:
  • Hook lands in the first 80–115 characters
  • Structure: Hook → Problem/Desire → Evidence or bridge → Stinger + CTA
  • 3–5 short paragraphs maximum. Every line must earn its place.
  • No period on the final CTA line
  • No walls of text with zero line breaks — mobile readers scan in chunks
  • Short-form (50–150 chars): best for simple offers and product hero creatives
  • Long-form (300+ chars): can work for PAS/BAB on cold audiences, but the
    first line still must earn the scroll

headline (target ≤27 chars, hard max 40):
  • Benefit-driven, never feature-driven
  • Use a number when possible
  • Action verbs: Get, Stop, Start, End, Finally, Try, See, Feel
  • Never end with a question mark
  • Think of it as the "caption" for the image
  • Examples: "Run Faster. Hurt Less." / "Rated #1 by Runners" / "30% Off Today"

description (≤30 chars):
  • One supporting detail only: price, offer, USP, or risk reversal
  • Treat as reinforcement — never put critical info here (often hidden)
  • Examples: "Free shipping. 30-day returns." / "4.9★ — 12K reviews"

cta_button — pick exactly one STRONG CTA (FORBIDDEN: "Learn More", "Click Here",
"Check It Out", "See More" — these kill conversions):
  • "Shop Now"                — product discovery, warm audiences
  • "Get Yours"               — scarcity angle
  • "Get Yours Before They're Gone" — scarcity + urgency
  • "Try It Risk-Free"        — risk reversal
  • "Get Offer"               — genuine discount available
  • "Claim Your [X]% Off"     — specific discount
  • "Order Now"               — impulse urgency
  • "Shop the Collection"     — multi-product or catalogue
  • "See the Difference"      — transformation products
  • "Build Your Bundle"       — multi-product upsell

audience_note:
  • One sentence for Meta Ads Manager targeting
  • Must be specific: demographics + interests + behaviours
  • Never generic ("people interested in the product category")

━━━ PERSUASION TRIGGERS ━━━
Each ad MUST contain ≥2 of (specific beats vague every time):
  ✓ Specificity — real numbers: "dropped 2 sizes in 6 weeks" not "lost weight"
  ✓ Urgency — real deadlines: "Ends tonight" beats "while supplies last"
  ✓ Scarcity — specific counts: "23 left in stock" beats "selling fast"
  ✓ Social proof — volume + rating: "4.8★ from 11,000+ reviews"
  ✓ Loss framing — "Don't let another morning start with back pain"
  ✓ Identity signal — who they become, not just what they get
  ✓ Risk reversal — "love it or return it, no questions asked"

━━━ WHAT NOT TO DO ━━━
Violating any of these will produce a bad ad. Check every variation:
  ✗ No generic slogans disconnected from the product
  ✗ No feature dumps — every feature must connect to a benefit
  ✗ No copy that would work with a competitor's name swapped in
  ✗ No walls of text with zero line breaks
  ✗ Never lead with "we" in the first line
  ✗ No repeating the same text in primary text, headline, AND on-image —
    each zone has a distinct job and must complement, not duplicate
  ✗ No ALL CAPS sentences
  ✗ No superlatives without proof ("world's greatest" = instant skip)
  ✗ Never "Learn More", "Click Here", "Check It Out" as CTA

═══════════════════════════════════════════════════════════════
PART 2 — CREATIVE SPEC (JSON FOR DETERMINISTIC LAYOUT RENDERER)
═══════════════════════════════════════════════════════════════

The product photograph is supplied separately by the pipeline — you NEVER
describe or prompt generative image models for the product. Your job is ONLY to
output layout, colours, typography sizes, and text that will be composited ON
TOP of the real product image.

Each variation MUST include a complete "creative_spec" object with this shape:

  canvas: { width: 1080, height: 1350, aspect_ratio: "4:5" }

  background.mode — one of:
    • "solid"     → value: { "color": "#F5F5F0" }  (hex or rgba only)
    • "gradient"  → value: { "type": "linear"|"radial", "angle_deg": number (linear only),
          "stops": [{"position": 0.0, "color": "#..."}, {"position": 1.0, "color": "#..."}] }
    • "blur"      → value: { "radius": 40, "tint": "rgba(0,0,0,0.3)" }  (blurred product as bg)
    • "ai_generated" — NOT SUPPORTED YET; do not use (use gradient or solid instead)

  product_zone:
    anchor: "center"
    y_offset_pct: number (-15 to 15) — vertical nudge as % of canvas height
    max_width_pct, max_height_pct: 40–60 (product should occupy ~40–60% of canvas area)
    crop_mode: "contain" | "cover" | "original"
    shadow: { "enabled": true/false, "offset_y": number, "blur": number, "color": "rgba(...)" }

  text_zones — array of 2–4 objects, max 4. Each MUST have:
    id: "headline" | "subtext" | "cta" | "trust"
    content: string (for headline/subtext/trust use on-image copy; cta = button label only)
    font_weight: "bold" | "normal" | "medium"
    font_size_px: integer 16–64
    color: hex or rgba
    anchor: one of
      "top-center", "top-left", "top-right",
      "bottom-center", "bottom-left", "bottom-right",
      "below-headline", "below-subtext", "above-cta"
    y_offset_pct: 0–25 (extra offset as % of canvas height after anchor)
    max_width_pct: 20–95
    alignment: "center" | "left" | "right" (for multi-line text)
    For id "cta" ONLY, also set:
      background_color (hex/rgba), corner_radius (number), padding: [top, right, bottom, left]

  brand_badge:
    position: "top-left" | "top-right"
    margin_px: number
    use_logo: true/false (pipeline pastes logo when true)
    fallback_text: SHORT BRAND NAME if no logo
    max_height_px: number

  style:
    font_family: "sans-serif-bold" | "sans-serif" (informational)
    mood: short label
    dominant_color, accent_color: hex (for CTA tinting; must contrast with background)

PLACEMENT RULES:
  • Choose background by angle: Pain Point → dark gradient + high contrast;
    Aspiration → warm light gradient OR "blur"; Social Proof → solid clean light.
  • Text zones MUST stay in top OR bottom bands — never overlap the central product.
    Put headline + subtext top; trust + cta bottom (trust above-cta).
  • creative_headline / creative_subtext / creative_cta / trust_element in the top-level
    JSON MUST match the "content" strings in text_zones (same wording).
  • Colours: only #RRGGBB or rgba(...). No named colours.
  • Do NOT output visual_description or visual_style (deprecated).

EXAMPLE creative_spec (Pain Point — structure reference only):
{
  "canvas": {"width": 1080, "height": 1350, "aspect_ratio": "4:5"},
  "background": {
    "mode": "gradient",
    "value": {
      "type": "linear",
      "angle_deg": 160,
      "stops": [
        {"position": 0.0, "color": "#1a1a2e"},
        {"position": 1.0, "color": "#16213e"}
      ]
    }
  },
  "product_zone": {
    "anchor": "center",
    "y_offset_pct": 5,
    "max_width_pct": 62,
    "max_height_pct": 48,
    "crop_mode": "contain",
    "shadow": {"enabled": true, "offset_y": 12, "blur": 28, "color": "rgba(0,0,0,0.35)"}
  },
  "text_zones": [
    {"id": "headline", "content": "Finally shoes that do not destroy your knees",
      "font_weight": "bold", "font_size_px": 50, "color": "#FFFFFF",
      "anchor": "top-center", "y_offset_pct": 6, "max_width_pct": 85, "alignment": "center"},
    {"id": "subtext", "content": "Orthopedic-grade support meets all-day comfort",
      "font_weight": "normal", "font_size_px": 24, "color": "rgba(255,255,255,0.78)",
      "anchor": "below-headline", "y_offset_pct": 2, "max_width_pct": 78, "alignment": "center"},
    {"id": "cta", "content": "Shop Now", "font_weight": "bold", "font_size_px": 22, "color": "#FFFFFF",
      "background_color": "#E63946", "corner_radius": 8, "padding": [12, 32, 12, 32],
      "anchor": "bottom-center", "y_offset_pct": 8},
    {"id": "trust", "content": "4.8★ from 3,200+ reviews", "font_weight": "normal",
      "font_size_px": 18, "color": "rgba(255,255,255,0.62)", "anchor": "above-cta",
      "y_offset_pct": 2, "max_width_pct": 80, "alignment": "center"}
  ],
  "brand_badge": {
    "position": "top-left", "margin_px": 24, "use_logo": true,
    "fallback_text": "BRAND", "max_height_px": 36
  },
  "style": {
    "font_family": "sans-serif-bold",
    "mood": "urgent-premium",
    "dominant_color": "#1a1a2e",
    "accent_color": "#E63946"
  }
}

EXAMPLE creative_spec (Social Proof — solid light background):
{
  "canvas": {"width": 1080, "height": 1350, "aspect_ratio": "4:5"},
  "background": {"mode": "solid", "value": {"color": "#F4F1EA"}},
  "product_zone": {
    "anchor": "center", "y_offset_pct": 4, "max_width_pct": 58, "max_height_pct": 50,
    "crop_mode": "contain",
    "shadow": {"enabled": true, "offset_y": 10, "blur": 22, "color": "rgba(0,0,0,0.2)"}
  },
  "text_zones": [
    {"id": "headline", "content": "Rated by thousands of happy customers",
      "font_weight": "bold", "font_size_px": 44, "color": "#1a1a1a",
      "anchor": "top-center", "y_offset_pct": 7, "max_width_pct": 88, "alignment": "center"},
    {"id": "cta", "content": "Get Yours", "font_weight": "bold", "font_size_px": 22,
      "color": "#FFFFFF", "background_color": "#2A9D8F", "corner_radius": 8,
      "padding": [12, 36, 12, 36], "anchor": "bottom-center", "y_offset_pct": 9},
    {"id": "trust", "content": "4.9★ average · 8,400+ verified reviews",
      "font_weight": "normal", "font_size_px": 18, "color": "rgba(26,26,26,0.65)",
      "anchor": "above-cta", "y_offset_pct": 2, "max_width_pct": 85, "alignment": "center"}
  ],
  "brand_badge": {
    "position": "top-left", "margin_px": 24, "use_logo": true,
    "fallback_text": "BRAND", "max_height_px": 34
  },
  "style": {
    "font_family": "sans-serif",
    "mood": "trust-clean",
    "dominant_color": "#F4F1EA",
    "accent_color": "#2A9D8F"
  }
}
═══════════════════════════════════════════════════════════════

Return ONLY valid JSON — no prose, no markdown, no explanation.
""".strip()

_STEP3_USER_TMPL = """
PRODUCT INTELLIGENCE:
{product_intel}

CONSUMER VOICE-OF-CUSTOMER BRIEF:
{voc_brief}

CAMPAIGN CONTEXT:
- Platform: {platform}
- Campaign Goal: {campaign_goal}
- Offer (if any): {offer}
- Landing Page URL (if provided): {landing_page_url}

Using the copywriting constitution (Part 1) AND a full creative_spec (Part 2)
for each variation, write exactly 3 Meta ad variations.

Rules for this run:
- Each variation must use a DIFFERENT hook formula (Pain-Point Question, Bold
  Outcome, Specific Proof, Curiosity Gap, Us vs. Them, or Social Proof Lead)
- Each variation must use the most appropriate framework (PAS, BAB, AIDA, FAB,
  or Value Stack) — do not default to PAS for all three
- headline: target ≤27 chars (hard max 40). Short and punchy wins.
- primary_text: hard max 500 characters (enforced in the app UI — stay within this)
- cta: choose from the approved strong CTA list — NEVER use "Learn More"
- format_type: pick one label (Product Hero, Bold Text Overlay, Testimonial Card,
  Before/After, Problem-Agitate-Solution, Offer/Promo, Flat Lay) for your records
- creative_headline, creative_subtext, creative_cta, trust_element MUST match the
  corresponding text_zones[].content strings inside creative_spec (identical text)
- On-image copy and primary_text must use DIFFERENT words — each zone has a distinct job
- If an offer is provided, incorporate urgency into at least one variation
- Omit visual_description and visual_style (deprecated)

Return this exact JSON structure (array of 3 objects):
[
  {{
    "angle": "Pain Point",
    "framework": "PAS or FAB or AIDA — whichever fits best",
    "primary_text": "...",
    "headline": "...(≤27 chars target)...",
    "description": "...(≤30 chars)...",
    "cta": "...(strong CTA — not Learn More)...",
    "audience_note": "...",
    "format_type": "Product Hero",
    "creative_headline": "...",
    "creative_subtext": "...",
    "creative_cta": "...",
    "trust_element": "...",
    "creative_spec": {{ ... full object per Part 2 — required ... }}
  }},
  {{
    "angle": "Aspiration",
    "framework": "BAB or AIDA or Value Stack — whichever fits best",
    "primary_text": "...",
    "headline": "...(≤27 chars target)...",
    "description": "...(≤30 chars)...",
    "cta": "...(strong CTA — not Learn More)...",
    "audience_note": "...",
    "format_type": "Flat Lay",
    "creative_headline": "...",
    "creative_subtext": "...",
    "creative_cta": "...",
    "trust_element": "...",
    "creative_spec": {{ ... }}
  }},
  {{
    "angle": "Social Proof",
    "framework": "Social Proof + FOMO or AIDA — whichever fits best",
    "primary_text": "...",
    "headline": "...(≤27 chars target)...",
    "description": "...(≤30 chars)...",
    "cta": "...(strong CTA — not Learn More)...",
    "audience_note": "...",
    "format_type": "Testimonial Card",
    "creative_headline": "...",
    "creative_subtext": "...",
    "creative_cta": "...",
    "trust_element": "...",
    "creative_spec": {{ ... }}
  }}
]
""".strip()


def _step3_generate_ads(
    model: genai.Client,
    product_intel: dict,
    voc_brief: dict,
    generate_images: bool = True,
    platform: str = "Meta Feed",
    campaign_goal: str = "Conversions",
    offer: str = "",
    landing_page_url: str = "",
    product_image_bytes: bytes | None = None,
    logo_bytes: bytes | None = None,
    brand_name: str = "",
) -> list[AdVariation]:
    user_prompt = _STEP3_USER_TMPL.format(
        product_intel=json.dumps(product_intel, indent=2),
        voc_brief=json.dumps(voc_brief, indent=2),
        platform=platform or "Meta Feed",
        campaign_goal=campaign_goal or "Conversions",
        offer=offer or "None",
        landing_page_url=landing_page_url or "Not provided",
    )
    raw = _call(model, _STEP3_SYSTEM, user_prompt)
    data = _parse_json(raw)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array from Step 3, got: {type(data)}")

    variations: list[AdVariation] = []
    for item in data:
        angle = item.get("angle", "")
        format_type = item.get("format_type", "")
        raw_spec = _parse_creative_spec_field(item.get("creative_spec"))
        creative_spec = merge_creative_spec_defaults(raw_spec, brand_name, item)
        image_b64: str | None = None
        if generate_images and product_image_bytes:
            image_b64 = _render_ad_creative(
                creative_spec,
                product_image_bytes,
                logo_bytes,
                product_image_url=None,
            )
        variations.append(
            AdVariation(
                angle=angle,
                framework=item.get("framework", ""),
                primary_text=_clamp_primary_text_for_ui(item.get("primary_text", "")),
                headline=item.get("headline", "")[:40],  # hard max 40; target ≤27
                description=item.get("description", "")[:30],
                cta=item.get("cta", "Learn More"),
                audience_note=item.get("audience_note", ""),
                image_b64=image_b64,
                creative_spec=creative_spec,
                format_type=format_type,
                visual_description=item.get("visual_description", ""),
                creative_headline=item.get("creative_headline", ""),
                creative_subtext=item.get("creative_subtext", ""),
                creative_cta=item.get("creative_cta", ""),
                trust_element=item.get("trust_element", ""),
                visual_style=item.get("visual_style", ""),
            )
        )
    return variations


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_ads(request: GenerateRequest) -> AdOutput:
    """
    Full pipeline:
      1. Resolve product URL
      2. Scrape product page
      3. Gather VoC data (Reddit / YouTube / Autocomplete)
      4. Run 3-step Gemini pipeline
      5. Return AdOutput

    Raises:
      ValueError        — missing GEMINI_API_KEY
      ProductNotFoundError — auto-discovery failed and no product_url provided
    """
    errors: list[str] = []
    model = _get_client()

    # --- Resolve product URL ---
    if request.product_url:
        product_url = request.product_url
    else:
        product_url = find_product_url(request.brand_url, request.product_name)

    # --- Scrape ---
    page_content = scrape_product_page(product_url)

    # --- VoC ---
    voc_summary = gather_voc(request, errors)

    # --- Step 1 ---
    product_intel = _step1_extract_product_intel(model, page_content)

    # Derive brand name if not provided
    brand_name = (
        request.brand_name
        or product_intel.get("brand_voice", "")[:30]
        or request.brand_url.split("//")[-1].split(".")[0].title()
    )

    # --- Step 2 ---
    voc_brief = _step2_synthesise_voc(model, product_intel, voc_summary)

    # --- Step 3 ---
    variations = _step3_generate_ads(
        model,
        product_intel,
        voc_brief,
        platform=request.platform,
        campaign_goal=request.campaign_goal,
        offer=request.offer,
        landing_page_url=request.landing_page_url,
        brand_name=brand_name,
    )

    return AdOutput(
        product_name=product_intel.get("name", request.product_name),
        brand_name=brand_name,
        variations=variations,
        voc_summary=voc_summary,
        product_intel=product_intel,
        errors=errors,
    )
