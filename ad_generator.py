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

from renderer import merge_creative_spec_defaults

from models import (
    AdOutput,
    AdVariation,
    GenerateRequest,
    ProductNotFoundError,
    VocSummary,
)
from scraper import find_product_url, scrape_product_page
from voc import gather_voc

_MODEL = "gemini-3-flash-preview"
_IMAGE_MODEL = "gemini-3.1-flash-image-preview"  # Nano Banana — native Gemini image generation

# Matches ui/components.py char_badge(..., 500) for Primary Text — model often overshoots.
_PRIMARY_TEXT_MAX_CHARS = 500

log = logging.getLogger(__name__)

# region agent log
_DEBUG_AGENT_LOG = (
    "/Users/daschelgorgenyi/Desktop/Test project Speed run/brand-ad-generator/.cursor/debug-da0215.log"
)
_DEBUG_SESSION_ID = "da0215"


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


def _call(client: genai.Client, system: str, user: str, *, max_attempts: int = 6) -> str:
    """Send a single turn and return the text response. Retries transient API errors."""
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            # region agent log
            _debug_agent_log(
                "H1",
                "generate_content_attempt",
                {
                    "model": _MODEL,
                    "attempt": attempt + 1,
                    "runId": "post-fix",
                },
            )
            # endregion agent log
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
  "visual_appearance": "EXTREMELY detailed physical description — include ALL of: exact colors (not 'blue' but 'navy blue with teal accents'), all materials (mesh, leather, foam, metal, plastic, knit), shape and silhouette, every distinctive visual feature (logos, patterns, stitching, textures, hardware), proportions. Write as if describing to a photographer who must recreate the product visually. Example: 'low-profile running shoe with coral-pink engineered mesh upper, black swoosh logo on lateral panel, pale pink React foam midsole with visible Air Zoom unit text, black rubber outsole, black padded collar and tongue, black flat laces'"
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
# Product image download (scraped asset for UI reference / previews)
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
# Image brief (Gemini) → Nano Banana prompt
# ---------------------------------------------------------------------------

_IMAGE_BRIEF_SYSTEM = """
You are an elite ad creative director and consumer psychologist. You don't just 
make pretty product photos — you design images that trigger specific psychological 
responses in specific people.

Your job: given a product, a target persona, and an ad angle, design an image 
that would make THAT SPECIFIC PERSON stop scrolling on Instagram.

═══════════════════════════════════════════════════════════════
STEP 1 — IDENTIFY THE PSYCHOLOGICAL TRIGGER
═══════════════════════════════════════════════════════════════

Before designing anything, decide which trigger to use for this persona + angle:

PROBLEM RECOGNITION — "That's me"
  The viewer sees their frustration in the image. The product is the escape.
  Best for: skeptics, people in pain, problem-aware audiences
  Visual approach: show the struggle or the before-state, product as relief
  Example: runner grimacing on a rainy morning → shoe positioned as the fix

IDENTITY ASPIRATION — "That's who I want to be"  
  The viewer sees the life they want. The product is the bridge.
  Best for: aspirational buyers, status-motivated, lifestyle-driven
  Visual approach: show the after-state, the lifestyle, the enviable moment
  Example: car in the driveway of a beautiful home at golden hour

SOCIAL VALIDATION — "People like me chose this"
  The viewer sees proof that their tribe already picked this product.
  Best for: researchers, cautious buyers, community-driven
  Visual approach: feels like a real customer photo, trust signals visible
  Example: product on a real kitchen counter, star rating overlay, natural light

SCARCITY/URGENCY — "I need to act now"
  The viewer feels time pressure or exclusivity.
  Best for: impulse buyers, deal hunters, FOMO-susceptible
  Visual approach: energy, movement, limited-edition framing, NOW feeling
  Example: product with "back in stock" energy, dynamic composition

CURIOSITY GAP — "What is that?"
  The viewer needs to know more. Something unexpected or incomplete.
  Best for: early adopters, innovation-seekers, contrarian types
  Visual approach: unexpected angle, surprising context, creates a question
  Example: close-up of one unusual product detail that demands explanation

Pick ONE trigger per brief. State it explicitly in your output.

═══════════════════════════════════════════════════════════════
STEP 2 — PRODUCT CATEGORY CREATIVE RULES
═══════════════════════════════════════════════════════════════

Identify the product category and follow these specific rules:

FOOTWEAR:
  • Show ONE shoe (or pair) on a real surface — never floating
  • Hero angle: 3/4 lateral showing full silhouette
  • For action context: feet on ground, motion suggested, real environment
  • Surfaces: track, trail, wet pavement, studio concrete, gym floor

AUTOMOTIVE:
  • EXACTLY ONE vehicle — never two cars, never a reflection showing a second car
  • On a real road or in a real location — never floating, never on a void
  • Hero angle: 3/4 front view
  • Environments: city street, mountain road, coastal highway, residential driveway

ELECTRONICS / WATCHES:
  • On a real surface or in a real hand
  • Show the screen/interface if it's a key feature
  • Hero angle: front or 3/4 showing the display
  • Surfaces: desk, table, wrist (for watches), being held

SKINCARE / BEAUTY:
  • Bottle/tube clearly showing the label
  • On a real surface: marble counter, bathroom shelf, held by hand
  • Can show skin texture and results if aspirational angle
  • Lighting: soft, flattering, beauty photography standards

CLOTHING / APPAREL:
  • On-body or styled flat lay on a real surface
  • Show fabric texture and fit
  • Lifestyle context for aspiration, detail shots for features

FOOD / BEVERAGE:
  • 45-degree angle, appetite appeal, real surface
  • Steam, condensation, fresh ingredients for context
  • Warm tones, inviting composition

ANY OTHER PRODUCT:
  • One product, real surface, real lighting
  • Apply the psychological trigger rules above

═══════════════════════════════════════════════════════════════
STEP 3 — IMAGE RULES (CRITICAL)
═══════════════════════════════════════════════════════════════

RULE 1: ONE PRODUCT ONLY. Never two instances of the same product.
RULE 2: PHOTOREALISTIC. Shot on a professional camera. Real materials, 
  real light, real textures. No illustration, no CGI, no cartoon, no AI look.
RULE 3: REAL SURFACES. Product sits on or is in a real environment. 
  Never floating in a void.
RULE 4: MINIMAL TEXT. Maximum 5 words headline. Brand name small at top. 
  One line subtext max. CTA in a button at bottom. Text never overlaps product.
RULE 5: DIVERSITY. If generating for Pain Point, Aspiration, and Social Proof, 
  they must look like three different photo shoots: different lighting, different 
  background, different color temperature, different product angle.
RULE 6: MOBILE-FIRST. The product must be identifiable at phone thumbnail size.
  The value proposition must be clear within 0.3 seconds.
RULE 7: NO AI ARTIFACTS. No merged objects, no extra limbs, no impossible 
  physics, no floating elements, no distorted faces.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Return ONLY valid JSON:

{
  "psychological_trigger": "problem_recognition | identity_aspiration | social_validation | scarcity_urgency | curiosity_gap",
  "trigger_rationale": "One sentence: why this trigger works for this persona",
  "image_description": "3-5 sentences describing the complete image. Be specific 
    enough that two photographers would produce similar shots. Include: product 
    placement, surface, environment, lighting direction and color, background, 
    mood, what the viewer's eye goes to first.",
  "product_description": "Exact physical description from product intel",
  "text_overlay": {
    "brand_name": "BRAND",
    "product_line": "Model name",
    "headline": "3-5 word headline that reinforces the trigger",
    "subtext": "One supporting line",
    "cta_text": "CTA button text"
  },
  "composition": {
    "product_position": "center | left-third | right-third",
    "product_angle": "specific angle",
    "surface": "what product sits on",
    "background": "environment behind product",
    "text_placement": "where text goes"
  },
  "mood": "2-3 word mood",
  "color_palette": "specific colors",
  "photography_style": "e.g. commercial product, lifestyle editorial, UGC-style"
}
""".strip()

_IMAGE_BRIEF_BATCH_APPEND = """

═══════════════════════════════════════════════════════════════
BATCH OUTPUT (this request only)
═══════════════════════════════════════════════════════════════
Generate 3 image briefs, one for each ad variation listed in the user message, in the same order.
Return ONLY a JSON array of exactly 3 brief objects. Each object must match the single-brief schema above.
No prose, no markdown.
""".strip()


def _nb_image_config(aspect_ratio: str) -> genai_types.ImageConfig:
    """Prefer 512 image_size when the installed SDK supports it (lower compute, fewer 503s)."""
    mf = getattr(genai_types.ImageConfig, "model_fields", {})
    kw: dict = {"aspect_ratio": aspect_ratio}
    if "image_size" in mf:
        kw["image_size"] = "512"
    return genai_types.ImageConfig(**kw)


def _inline_b64_from_response(response) -> str | None:
    import base64

    cands = getattr(response, "candidates", None) or []
    if not cands:
        return None
    content = cands[0].content
    if not content or not content.parts:
        return None
    for part in content.parts:
        if part.inline_data and part.inline_data.data:
            raw = part.inline_data.data
            return base64.b64encode(raw).decode("utf-8")
    return None


def _generate_ad_image_nb(
    client: genai.Client,
    prompt: str,
    aspect_ratio: str = "4:5",
) -> str | None:
    """
    Generate an ad image using Nano Banana (Gemini native image gen).
    Returns base64-encoded PNG string or None after up to 3 attempts with backoff.
    """
    # Up to 3 attempts; wait 10s then 20s before 2nd and 3rd try (transient 503s).
    backoff_before_retry = (10, 20)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=_IMAGE_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=_nb_image_config(aspect_ratio),
                ),
            )
            b64 = _inline_b64_from_response(response)
            if b64:
                return b64
            print("[IMAGE] No image data in response")
        except Exception as exc:
            print(f"[IMAGE] Nano Banana generation failed (attempt {attempt + 1}/3): {exc}")
        if attempt < 2:
            time.sleep(backoff_before_retry[attempt])

    return None


def _fallback_single_image_brief(
    product_intel: dict,
    angle: str,
    headline: str,
    primary_text: str,
    cta: str,
    brand_name: str,
) -> dict:
    visual = product_intel.get("visual_appearance", "product")
    name = product_intel.get("name", "Product")
    return {
        "image_description": (
            f"Professional ad photography of {name} ({visual}). "
            f"Product centered on a dark gradient background with dramatic side lighting. "
            f"Bold text '{headline}' above the product. Clean, premium Meta ad creative."
        ),
        "product_description": f"{name} — {visual}",
        "text_overlay": {
            "brand_name": brand_name.upper(),
            "headline": headline,
            "cta_text": cta or "Shop Now",
        },
        "mood": "premium dramatic",
    }


def _brief_to_prompt(brief: dict, include_text: bool = True) -> str:
    """Convert JSON image brief into a Nano Banana prompt."""
    parts = []

    desc = brief.get("image_description", "")
    if desc:
        parts.append(desc)

    product = brief.get("product_description", "")
    if product:
        parts.append(f"The product shown is: {product}. It must be clearly recognizable.")

    if include_text:
        text = brief.get("text_overlay", {})
        text_parts = []
        if text.get("brand_name"):
            text_parts.append(f'"{text["brand_name"]}" in small caps at the top')
        if text.get("product_line"):
            text_parts.append(
                f'"{text["product_line"]}" below the brand name in smaller text'
            )
        if text.get("headline"):
            text_parts.append(f'Bold headline text: "{text["headline"]}"')
        if text.get("subtext"):
            text_parts.append(f'Smaller subtext: "{text["subtext"]}"')
        if text.get("cta_text"):
            text_parts.append(f'A button or pill shape with "{text["cta_text"]}"')
        if text_parts:
            parts.append("Text on the image: " + ". ".join(text_parts) + ".")
            parts.append(
                "All text must be crisp, clean, perfectly spelled, and readable at phone size."
            )

    comp = brief.get("composition", {})
    if isinstance(comp, dict):
        if comp.get("surface"):
            parts.append(f"Surface: {comp['surface']}.")
        if comp.get("background_type"):
            parts.append(f"Background: {comp['background_type']}.")
        elif comp.get("background"):
            parts.append(f"Background: {comp['background']}.")
        if comp.get("text_placement"):
            parts.append(f"Text placement: {comp['text_placement']}.")

    mood = brief.get("mood", "")
    palette = brief.get("color_palette", "")
    photo_style = brief.get("photography_style", "")
    if mood:
        parts.append(f"Overall mood: {mood}.")
    if palette:
        parts.append(f"Color palette: {palette}.")
    if photo_style:
        parts.append(f"Photography style: {photo_style}.")

    parts.append(
        "This is a professional Meta/Instagram ad creative. "
        "4:5 vertical format. Photorealistic commercial quality. "
        "The product is the hero of the image. "
        "No watermarks. No stock photo artifacts."
    )

    return " ".join(parts)


def _generate_image_brief(
    client: genai.Client,
    product_intel: dict,
    angle: str,
    headline: str = "",
    primary_text: str = "",
    cta: str = "",
    brand_name: str = "",
    persona: str = "",
) -> dict:
    """Ask Gemini to create a structured image brief for one ad variation."""
    persona_context = f"\n- Target Persona: {persona}" if persona else ""

    user_prompt = f"""
PRODUCT INTELLIGENCE:
{json.dumps(product_intel, indent=2)}

AD VARIATION:
- Angle: {angle}
- Brand: {brand_name}
- Headline: {headline}
- CTA: {cta}
- Primary text hook: {(primary_text or '')[:150]}{persona_context}

Generate the image brief for this ad. 
First, identify the product category.
Then, choose the psychological trigger that would work best for this angle{' and this specific persona' if persona else ''}.
Design the image around that trigger — not just a pretty product photo, but an image 
that would make the target viewer stop scrolling because it resonates with their psychology.
""".strip()

    try:
        raw = _call(client, _IMAGE_BRIEF_SYSTEM, user_prompt)
        brief = _parse_json(raw)
        if isinstance(brief, dict):
            return brief
    except Exception as exc:
        log.warning("Image brief generation failed: %s", exc)

    return _fallback_single_image_brief(
        product_intel, angle, headline, primary_text, cta, brand_name
    )


def _generate_all_image_briefs(
    client: genai.Client,
    product_intel: dict,
    variations: list[AdVariation],
    brand_name: str,
    persona: str = "",
) -> list[dict]:
    """
    One LLM call: JSON array of 3 image briefs (same order as variations).
    """
    persona_block = (
        f"\nTARGET PERSONA (apply to every brief):\n{persona}\n"
        if persona
        else ""
    )
    rows = []
    for v in variations:
        h = v.creative_headline or v.headline
        c = v.creative_cta or v.cta
        hook = (v.primary_text or "")[:150]
        rows.append(
            f"  • Angle: {v.angle}\n"
            f"    Headline: {h}\n"
            f"    CTA: {c}\n"
            f"    Primary text hook: {hook}"
        )
    variations_block = "\n".join(rows)
    user_prompt = f"""
PRODUCT INTELLIGENCE:
{json.dumps(product_intel, indent=2)}
{persona_block}
AD VARIATIONS (produce exactly one brief per row, in this order):
{variations_block}

Brand: {brand_name}

Generate 3 image briefs, one for each angle above. Return a JSON array of exactly 3 brief objects.
Each brief must showcase the product from visual_appearance as the hero and use the headline/CTA in text_overlay.
For each brief, identify the product category, pick the best psychological trigger for this angle
{"and the target persona above" if persona else ""}, and design around that trigger.
The three briefs must be visually distinct (Pain Point vs Aspiration vs Social Proof): different lighting, backgrounds, and color temperature.
""".strip()

    system = _IMAGE_BRIEF_SYSTEM + _IMAGE_BRIEF_BATCH_APPEND

    try:
        raw = _call(client, system, user_prompt)
        parsed = _parse_json(raw)
        if isinstance(parsed, list) and len(parsed) == 3:
            out: list[dict] = []
            for i, item in enumerate(parsed):
                if isinstance(item, dict):
                    out.append(item)
                else:
                    v = variations[i]
                    out.append(
                        _fallback_single_image_brief(
                            product_intel,
                            v.angle,
                            v.creative_headline or v.headline,
                            v.primary_text,
                            v.creative_cta or v.cta,
                            brand_name,
                        )
                    )
            return out
        if isinstance(parsed, list) and len(parsed) != 3:
            log.warning("Batch image briefs: expected 3 items, got %s", len(parsed))
    except Exception as exc:
        log.warning("Batch image brief generation failed: %s", exc)

    return [
        _fallback_single_image_brief(
            product_intel,
            v.angle,
            v.creative_headline or v.headline,
            v.primary_text,
            v.creative_cta or v.cta,
            brand_name,
        )
        for v in variations
    ]


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
  • Emojis — Meta feed norm: put **one** relevant emoji at the **start of the first line**
    of primary_text (hook) when it adds energy or clarity (e.g. 🚀 ⚡ ✨ 🙌). Optionally one
    emoji in **headline** or **description** if it fits the character limits and improves
    scroll-stop; never more than 2 emojis total across all fields; never a wall of emojis
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
    platform: str = "Meta Feed",
    campaign_goal: str = "Conversions",
    offer: str = "",
    landing_page_url: str = "",
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


def _evaluate_with_persona(
    client: genai.Client,
    persona: str,
    product_intel: dict,
    variations: list,
) -> dict:
    """Evaluate existing ads through a specific persona's perspective."""
    system = """
You are a consumer psychologist. Given a specific persona and 3 ad variants,
evaluate each ad as if you ARE that person scrolling through Instagram.

Be specific, critical, and honest. A skeptical persona should be hard to impress.

Return ONLY valid JSON:
{
  "persona_summary": "one sentence restating who this person is",
  "evaluations": [
    {
      "angle": "Pain Point",
      "would_stop_scrolling": true/false,
      "hook_reaction": "what they think when they read the first line",
      "emotional_response": "how this ad makes them feel",
      "click_score": 7,
      "what_works": "the specific element that resonates",
      "what_fails": "the specific element that doesn't land",
      "missing": "what would make this ad irresistible for this persona"
    }
  ],
  "recommendations": {
    "copy_changes": ["specific change 1", "specific change 2"],
    "image_changes": ["specific change 1", "specific change 2"],
    "tone_shift": "how the overall tone should change for this persona"
  }
}
""".strip()

    ads_summary = []
    for v in variations:
        ads_summary.append({
            "angle": v.angle,
            "headline": v.headline,
            "primary_text": v.primary_text[:400],
            "description": v.description,
            "cta": v.cta,
            "trust": v.trust_element,
        })

    user = f"""
PERSONA: {persona}

PRODUCT: {json.dumps(product_intel, indent=2)}

ADS TO EVALUATE:
{json.dumps(ads_summary, indent=2)}

Evaluate each ad as this specific persona. Stay in character.
""".strip()

    raw = _call(client, system, user)
    data = _parse_json(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Persona evaluation expected a JSON object, got {type(data)}")
    return data


def _generate_persona_optimized_ads(
    client: genai.Client,
    product_intel: dict,
    voc_brief: dict,
    persona: str,
    evaluation: dict,
    brand_name: str = "",
) -> list:
    """Generate 3 improved ads based on persona feedback."""
    user = f"""
PRODUCT INTELLIGENCE:
{json.dumps(product_intel, indent=2)}

VOC BRIEF:
{json.dumps(voc_brief, indent=2)}

TARGET PERSONA: {persona}

PERSONA EVALUATION OF CURRENT ADS:
{json.dumps(evaluation, indent=2)}

Write 3 IMPROVED Meta ad variations specifically optimized for this persona.
Address every criticism from the evaluation. Incorporate the recommendations.
Each variation: different angle (Pain Point, Aspiration, Social Proof), different hook formula.

Return the same JSON array format as standard ad generation (array of 3 objects).
""".strip()

    raw = _call(client, _STEP3_SYSTEM, user)
    data = _parse_json(raw)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")

    variations: list[AdVariation] = []
    for item in data:
        angle = item.get("angle", "")
        format_type = item.get("format_type", "")
        raw_spec = _parse_creative_spec_field(item.get("creative_spec"))
        creative_spec = merge_creative_spec_defaults(raw_spec, brand_name, item)
        image_b64 = None
        try:
            brief = _generate_image_brief(
                client,
                product_intel,
                angle=angle,
                headline=item.get("creative_headline") or item.get("headline", ""),
                primary_text=item.get("primary_text", ""),
                cta=item.get("creative_cta") or item.get("cta", "Shop Now"),
                brand_name=brand_name,
                persona=persona or "",
            )
            prompt = _brief_to_prompt(brief)
            image_b64 = _generate_ad_image_nb(client, prompt, aspect_ratio="4:5")
        except Exception as exc:
            print(f"[PERSONA] Image gen failed for optimized ad: {exc}")

        variations.append(
            AdVariation(
                angle=angle,
                framework=item.get("framework", ""),
                primary_text=_clamp_primary_text_for_ui(item.get("primary_text", "")),
                headline=item.get("headline", "")[:40],
                description=item.get("description", "")[:30],
                cta=item.get("cta", "Shop Now"),
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
