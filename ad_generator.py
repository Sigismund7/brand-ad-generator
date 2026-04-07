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

import base64
import json
import os
import re

from google import genai
from google.genai import types as genai_types

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


# ---------------------------------------------------------------------------
# Gemini client setup
# ---------------------------------------------------------------------------

def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file or the sidebar."
        )
    return genai.Client(api_key=api_key)


def _get_model() -> genai.Client:
    """Kept for backward compatibility — returns the client."""
    return _get_client()


def _call(client: genai.Client, system: str, user: str) -> str:
    """Send a single turn and return the text response."""
    response = client.models.generate_content(
        model=_MODEL,
        contents=f"SYSTEM:\n{system}\n\nUSER:\n{user}",
        config=genai_types.GenerateContentConfig(temperature=0.7),
    )
    return response.text.strip()


def _parse_json(raw: str) -> dict | list:
    """
    Extract and parse a JSON block from a Gemini response.
    Handles ```json ... ``` fences and bare JSON.
    """
    import json as _json2, time as _time2
    _LOG = "/Users/daschelgorgenyi/Desktop/Test project Speed run/.cursor/debug-e762c2.log"

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        result = json.loads(cleaned)
        # #region agent log
        with open(_LOG, "a") as _f:
            _f.write(_json2.dumps({"sessionId": "e762c2", "timestamp": int(_time2.time() * 1000), "hypothesisId": "A", "location": "ad_generator.py:_parse_json:ok", "message": "json parsed ok", "data": {"cleaned_len": len(cleaned), "result_type": type(result).__name__}}) + "\n")
        # #endregion
        return result
    except json.JSONDecodeError as e1:
        # #region agent log
        with open(_LOG, "a") as _f:
            _f.write(_json2.dumps({"sessionId": "e762c2", "timestamp": int(_time2.time() * 1000), "hypothesisId": "A,B", "location": "ad_generator.py:_parse_json:first_fail", "message": "first json.loads failed", "data": {"error": str(e1), "cleaned_prefix": cleaned[:300], "cleaned_suffix": cleaned[-200:]}}) + "\n")
        # #endregion
        # Last resort: find the first { or [ and try from there
        match = re.search(r"[{\[]", cleaned)
        if match:
            try:
                result2 = json.loads(cleaned[match.start():])
                # #region agent log
                with open(_LOG, "a") as _f:
                    _f.write(_json2.dumps({"sessionId": "e762c2", "timestamp": int(_time2.time() * 1000), "hypothesisId": "A,B", "location": "ad_generator.py:_parse_json:fallback_ok", "message": "fallback parse succeeded", "data": {"start_pos": match.start()}}) + "\n")
                # #endregion
                return result2
            except json.JSONDecodeError as e2:
                # #region agent log
                with open(_LOG, "a") as _f:
                    _f.write(_json2.dumps({"sessionId": "e762c2", "timestamp": int(_time2.time() * 1000), "hypothesisId": "A,B", "location": "ad_generator.py:_parse_json:fallback_fail", "message": "BOTH parses failed — raising", "data": {"e1": str(e1), "e2": str(e2), "cleaned_around_error": cleaned[max(0, e2.pos - 80):e2.pos + 80]}}) + "\n")
                # #endregion
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
  "social_proof_signals": ["any reviews, awards, press mentions, customer counts found"]
}}
""".strip()


def _step1_extract_product_intel(
    model: genai.GenerativeModel,
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
# Image Generation — Imagen 3
# ---------------------------------------------------------------------------

_IMAGE_MODEL = "imagen-4.0-generate-001"

def _build_image_prompt(product_intel: dict, angle: str, format_type: str = "") -> str:
    name = product_intel.get("name", "product")
    features = ", ".join(product_intel.get("key_features", [])[:3])
    brand_voice = product_intel.get("brand_voice", "")
    audience = product_intel.get("target_audience_signals", "")

    feature_line = f" Key features: {features}." if features else ""
    brand_line = f" Brand feel: {brand_voice}." if brand_voice else ""
    audience_line = f" Target customer: {audience}." if audience else ""

    # Format-type-specific overrides (take precedence over angle defaults)
    if format_type == "Testimonial Card":
        return (
            f"Clean Meta/Instagram product photography for a testimonial ad for {name}.{feature_line}"
            f" The {name} is centered on a plain light grey or off-white background with generous "
            f"empty space on the lower third for a quote overlay. Studio lighting, soft shadows. "
            f"Product fills 50% of the frame, perfectly sharp. Photorealistic commercial quality."
            f"{brand_line} No people. No text. No logos. Square 1:1 format."
        )
    elif format_type == "Flat Lay":
        return (
            f"Editorial flat-lay product photograph for Meta/Instagram ad for {name}.{feature_line}"
            f" Top-down overhead shot: the {name} arranged elegantly with 2-3 complementary props "
            f"(neutral lifestyle objects that match the brand aesthetic) on a textured surface "
            f"(linen, marble, or natural wood). Warm, even studio lighting. "
            f"Looks like a high-end editorial magazine spread.{brand_line}"
            f" No text overlays. No logos. Square 1:1 format."
        )
    elif format_type in ("Bold Text Overlay", "Problem-Agitate-Solution"):
        return (
            f"High-contrast Meta/Instagram feed ad background for {name}.{feature_line}"
            f" The {name} is positioned dramatically — left or right third of frame — "
            f"against a deep dark background (charcoal, navy, or near-black) with a single "
            f"powerful directional light source creating strong shadows and highlights. "
            f"Leaves significant clear space (right or left half) for bold text overlay. "
            f"Photorealistic, high drama, premium commercial quality.{brand_line}"
            f" No text. No logos. Square 1:1 format."
        )
    elif format_type == "Before/After":
        return (
            f"Transformation Meta/Instagram ad creative for {name}.{feature_line}"
            f" Split composition: left half shows a muted, desaturated lifestyle scene "
            f"representing struggle or the 'before' state; right half shows the same scene "
            f"bright, vibrant, and joyful representing the transformed outcome. "
            f"The {name} is featured prominently in the right (after) half. "
            f"No before/after labels. Photorealistic.{brand_line}{audience_line}"
            f" No text overlays. No logos. Square 1:1 format."
        )
    elif format_type == "Offer/Promo":
        return (
            f"High-energy Meta/Instagram promotional ad creative for {name}.{feature_line}"
            f" The {name} is the hero, centred against a bold brand-coloured background "
            f"with a clean gradient or solid fill. Bright, energetic studio lighting. "
            f"Product fills 60% of the frame. Leaves clear space at top and bottom for "
            f"price/offer text overlays. Optimistic, urgent feel.{brand_line}"
            f" No text. No logos. Square 1:1 format."
        )

    # Angle-based fallbacks (Product Hero is also the Pain Point default)
    if angle == "Pain Point":
        return (
            f"High-impact Meta/Instagram feed ad creative for {name}.{feature_line}"
            f" Hero product shot: the {name} isolated on a clean gradient background "
            f"(light grey to white), dramatic studio rim lighting from the side and above "
            f"that reveals every texture and contour of the product. "
            f"The product fills 70% of the frame. Extremely sharp focus. "
            f"Subtle drop shadow grounds the product. "
            f"Photorealistic, ultra high detail, premium commercial advertising photography quality."
            f"{brand_line} No people. No text. No logos. Square 1:1 format."
        )
    elif angle == "Aspiration":
        return (
            f"Aspirational Meta/Instagram lifestyle ad creative for {name}.{feature_line}"
            f" The product is shown in a beautiful real-world setting — "
            f"outdoors in golden-hour afternoon light, warm amber and orange tones."
            f" A person's lower body (feet, legs) is actively using the product, "
            f"face not shown. Dynamic, in-motion feel."
            f" Shallow depth of field, bokeh background. "
            f"Looks like a premium editorial shoot for a major brand campaign."
            f"{brand_line}{audience_line}"
            f" No text overlays. No logos. Square 1:1 format."
        )
    else:  # Social Proof
        return (
            f"Authentic lifestyle Meta/Instagram ad creative for {name}.{feature_line}"
            f" Candid, real-world moment — the product in use in an everyday relatable setting."
            f" Natural indoor or outdoor light, feels warm and genuine."
            f" May show a person's hands or feet using the product; face optional but friendly if shown."
            f" Community vibe, feels like a real customer photo but shot professionally."
            f"{brand_line}{audience_line}"
            f" No text overlays. No logos. Square 1:1 format."
        )


def _generate_ad_image(
    client: genai.Client,
    product_intel: dict,
    angle: str,
    format_type: str = "",
) -> str | None:
    """
    Calls Imagen to generate a product image for the given ad angle.
    Returns a base64-encoded JPEG string, or None if generation fails.
    """
    import json as _json, time as _time, traceback as _tb
    _LOG_OLD = "/Users/daschelgorgenyi/Desktop/Test project Speed run/.cursor/debug-e762c2.log"
    _LOG = "/Users/daschelgorgenyi/Desktop/Test project Speed run/.cursor/debug-087295.log"

    prompt = _build_image_prompt(product_intel, angle, format_type)

    # #region agent log (087295)
    with open(_LOG, "a") as _f:
        _f.write(_json.dumps({"sessionId": "087295", "timestamp": int(_time.time() * 1000), "hypothesisId": "H-A,H-C", "location": "ad_generator.py:_generate_ad_image:entry", "message": "image gen entry", "data": {"angle": angle, "format_type": format_type, "format_type_matched_branch": "testimonial" if "testimonial" in prompt.lower() else ("flat" if "flat-lay" in prompt.lower() else ("contrast" if "high-contrast" in prompt.lower() else "other")), "prompt_full": prompt}}) + "\n")
    # #endregion

    # #region agent log (e762c2 — legacy)
    with open(_LOG_OLD, "a") as _f:
        _f.write(_json.dumps({"sessionId": "e762c2", "timestamp": int(_time.time() * 1000), "hypothesisId": "D", "location": "ad_generator.py:_generate_ad_image:entry", "message": "image gen called", "data": {"angle": angle, "model": _IMAGE_MODEL, "prompt_prefix": prompt[:120]}}) + "\n")
    # List available models on first call (angle == "Pain Point")
    if angle == "Pain Point":
        try:
            img_models = [m.name for m in client.models.list() if "generat" in m.name.lower() or "imagen" in m.name.lower() or "flash" in m.name.lower()]
            with open(_LOG_OLD, "a") as _f:
                _f.write(_json.dumps({"sessionId": "e762c2", "timestamp": int(_time.time() * 1000), "hypothesisId": "C", "location": "ad_generator.py:_generate_ad_image:listmodels", "message": "available models", "data": {"models": img_models}}) + "\n")
        except Exception as _le:
            with open(_LOG_OLD, "a") as _f:
                _f.write(_json.dumps({"sessionId": "e762c2", "timestamp": int(_time.time() * 1000), "hypothesisId": "C", "location": "ad_generator.py:_generate_ad_image:listmodels_err", "message": "list failed", "data": {"err": str(_le)}}) + "\n")
    # #endregion

    try:
        response = client.models.generate_images(
            model=_IMAGE_MODEL,
            prompt=prompt,
            config=genai_types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1",
                output_mime_type="image/jpeg",
                person_generation="ALLOW_ADULT",
            ),
        )
        count = len(response.generated_images) if response.generated_images else 0
        # #region agent log (087295)
        with open(_LOG, "a") as _f:
            _f.write(_json.dumps({"sessionId": "087295", "timestamp": int(_time.time() * 1000), "hypothesisId": "H-A,H-B,H-D", "location": "ad_generator.py:_generate_ad_image:response", "message": "image response received", "data": {"angle": angle, "format_type": format_type, "image_count": count, "has_generated_images_attr": hasattr(response, "generated_images")}}) + "\n")
        # #endregion
        # #region agent log (e762c2 — legacy)
        with open(_LOG_OLD, "a") as _f:
            _f.write(_json.dumps({"sessionId": "e762c2", "timestamp": int(_time.time() * 1000), "hypothesisId": "E", "location": "ad_generator.py:_generate_ad_image:response", "message": "got response", "data": {"image_count": count}}) + "\n")
        # #endregion
        if response.generated_images:
            raw = response.generated_images[0].image.image_bytes
            return base64.b64encode(raw).decode("utf-8")
    except Exception as exc:
        # #region agent log (087295)
        with open(_LOG, "a") as _f:
            _f.write(_json.dumps({"sessionId": "087295", "timestamp": int(_time.time() * 1000), "hypothesisId": "H-A,H-B,H-D", "location": "ad_generator.py:_generate_ad_image:exception", "message": "exception during generate_images", "data": {"angle": angle, "format_type": format_type, "error": str(exc), "type": type(exc).__name__, "traceback": _tb.format_exc()[-1000:]}}) + "\n")
        # #endregion
        # #region agent log (e762c2 — legacy)
        with open(_LOG_OLD, "a") as _f:
            _f.write(_json.dumps({"sessionId": "e762c2", "timestamp": int(_time.time() * 1000), "hypothesisId": "A,B,C", "location": "ad_generator.py:_generate_ad_image:exception", "message": "exception caught", "data": {"error": str(exc), "type": type(exc).__name__, "traceback": _tb.format_exc()[-800:]}}) + "\n")
        # #endregion
    return None


# ---------------------------------------------------------------------------
# Step 3 — Meta Ad Generation (Copywriting Constitution)
# ---------------------------------------------------------------------------

_STEP3_SYSTEM = """
You are one of the world's best direct-response copywriters AND a senior
performance creative strategist specialising in high-converting static Meta
(Facebook / Instagram) ads for cold prospecting audiences.

You follow both the copywriting constitution and the static creative direction
framework below without exception.

═══════════════════════════════════════════════════════════════
COPYWRITING CONSTITUTION
═══════════════════════════════════════════════════════════════

THE 5 NON-NEGOTIABLE RULES
1. Hook within 125 chars — The first 125 characters of primary_text are all
   that appear before "See More" on mobile. Everything critical must land there.
2. Never use "you" generically — Open with a specific persona callout.
   Right: "Attention runners over 40 who..."
   Wrong: "Are you tired of..."
3. One angle per ad — Each variation makes exactly one promise. No feature
   lists. No multi-benefit rambling.
4. Use their words, not marketing words — Draw directly from the
   exact_phrases_to_use list. Zero buzzwords (innovative, game-changing,
   revolutionary, powerful, seamless).
5. 6th-grade reading level — Every sentence under 15 words. Short paragraphs.
   Conversational. If it sounds like a brochure, rewrite it.

THE 3 VARIATION FRAMEWORKS
Variation 1 — PAS (Pain → Agitate → Solution)
  • Framework: Name the pain → make it feel worse → present the product as relief
  • Hook formula: investment hook or problem-question hook
    Examples: "I spent $400 on orthotics before I tried..."
              "Running 5Ks but your knees ache for days after?"
  • Psychological trigger: Loss aversion (pain of not acting > gain of acting)

Variation 2 — BAB (Before → After → Bridge)
  • Framework: Describe life now (struggle) → paint the destination (outcome)
    → introduce the product as the path between them
  • Hook formula: transformation hook
    Examples: "What if better sleep didn't require a prescription?"
              "Imagine finishing a half-marathon without the next-day limp..."
  • Psychological trigger: Aspiration — identity-level desire, who they become

Variation 3 — Social Proof + FOMO
  • Framework: Open with community signal or implied testimony → validate with
    specifics → create urgency
  • Hook formula: social proof hook or scarcity/surprise hook
    Examples: "47,000 runners switched. Here's why."
              "I can't believe I didn't find this sooner..."
              Use verbatim consumer phrases where possible.
  • Psychological trigger: Social proof + fear of missing out

FIELD-LEVEL RULES
primary_text:
  • Structure: Hook (persona callout) → Problem/Desire → Evidence or bridge
    → Stinger + CTA
  • Max: 3-5 short paragraphs. Every line must earn its place.
  • No period on the final CTA line.

headline (≤40 chars):
  • Benefit-driven, not feature-driven
  • Use a number when possible
  • Action verbs: Get, Stop, Start, End, Finally, Try
  • Never end with a question mark

description (≤30 chars):
  • One supporting detail: price point, offer, USP, or risk reversal
  • Examples: "Free shipping. 30-day returns."

cta_button — pick exactly one:
  • "Shop Now"   — warm or retargeting audiences
  • "Learn More" — cold traffic, complex or high-priced products
  • "Get Offer"  — when there is a genuine discount
  • "Order Now"  — impulse purchases with urgency

audience_note:
  • One sentence describing who to target in Meta Ads Manager
    (demographics, interests, behaviours — not generic)

PSYCHOLOGICAL TRIGGER CHECKLIST
Each ad MUST contain ≥2 of:
  ✓ Specificity — real numbers over vague claims ("dropped 2 sizes" > "lost weight")
  ✓ Loss framing — what they lose by not acting
  ✓ Social proof — real or implied community
  ✓ Identity signal — who they'll become, not just what they'll get
  ✓ Risk reversal — removes barrier to clicking
  ✓ Urgency — only if the product genuinely supports it; never fabricated
═══════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════
STATIC CREATIVE DIRECTION FRAMEWORK
═══════════════════════════════════════════════════════════════

FORMAT TYPES — assign exactly one per variation:
  • Product Hero        — clean product shot, branded background, minimal text
                          Best for: Pain Point angle (sharp, direct)
  • Bold Text Overlay   — large hook text over lifestyle/product image
                          Best for: Pain Point angle (urgent, confrontational)
  • Testimonial Card    — customer quote + star rating + product
                          Best for: Social Proof angle
  • Before/After        — visual transformation (do NOT use the words Before/After)
                          Best for: Pain Point or Aspiration angle
  • Problem-Agitate-Solution — pain imagery → intensified → product as relief
                          Best for: Pain Point angle
  • Offer/Promo         — discount + product + urgency element
                          Use only if a genuine offer is provided
  • Flat Lay            — styled editorial arrangement of product + accessories
                          Best for: Aspiration angle (premium feel)

VISUAL HIERARCHY — describe placement following this attention sequence:
  1st eye → Product or hero image (largest element, 30-40% of frame)
  2nd eye → Value proposition / hook (bold, 6-8 words max, on-image)
  3rd eye → Social proof (stars, review count, press badge, result claim)
  4th eye → CTA (specific action verb — "Shop the Set", not "Learn More")

COMPOSITION RULES:
  • Aspect ratio: 4:5 vertical (1080×1350px) — default for Feed
    Use 1:1 (1080×1080px) for Stories or square placements
  • Maintain 30-40% white/negative space
  • Maximum 3 focal points (3-Element Rule)
  • Product is the unmistakable hero
  • Text never covers the product
  • High-contrast text for mobile readability
  • Price visible if it qualifies intent — readable, not dominant

ON-IMAGE COPY RULES:
  creative_headline: 6-8 words, product-led hook, not a generic slogan
    Examples: "Finally shoes that don't destroy your knees"
              "The last pillow you'll ever need to buy"
  creative_subtext: one line of supporting proof or benefit (optional but preferred)
  creative_cta: specific action verb phrase — "Get Yours", "Try It Risk-Free",
    "Shop the Set", "Grab Yours Today" (NOT just "Learn More")
  trust_element — include exactly one:
    • Star rating + review count: "4.8★ from 3,200+ reviews"
    • Press credibility: "As Seen In Forbes · NYT · Vogue"
    • Customer result claim: "Results in 14 days — or your money back"
    • Third-party badge: "Trustpilot Excellent · 10,000+ reviews"
    • Specific outcome stat: "9 out of 10 customers reorder"

COLOR & STYLE DIRECTION per angle:
  Pain Point   → Bold & Urgent: dark or high-contrast background, sharp typography,
                 accent colour that pops against the Meta feed
  Aspiration   → Premium / Editorial: warm neutrals or brand colour, refined
                 sans-serif, generous white space
  Social Proof → Clean & Warm: light background, friendly photography vibe,
                 social-community feel; avoid pure white if Meta feed is white

visual_description: write 3-5 sentences describing the complete layout as if
  briefing a designer — what element is where, what size, what colour, what
  text appears on the canvas, and what the overall visual feel is.

visual_style: one concise sentence: background colour/texture, typography style,
  overall aesthetic label (e.g. "Dark charcoal gradient, bold white Helvetica
  headline, Premium/Urgent feel").
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

Using BOTH the copywriting constitution AND the static creative direction
framework, write exactly 3 Meta ad variations. Each variation must produce
complete copy fields AND a complete static creative brief.

For the creative brief, choose the most appropriate format_type for each angle,
write a full visual_description (3-5 sentences briefing a designer), and
provide all on-image copy fields. Ensure the creative brief is specific enough
that a designer could execute it without further clarification.

If an offer is provided, incorporate it into the Aspiration or Social Proof
variation's creative direction. If a landing page URL is provided, note in
visual_description how the ad's visual promise should match the landing page.

Return this exact JSON structure:
[
  {{
    "angle": "Pain Point",
    "framework": "PAS",
    "primary_text": "...",
    "headline": "...",
    "description": "...",
    "cta": "...",
    "audience_note": "...",
    "format_type": "...",
    "visual_description": "...",
    "creative_headline": "...",
    "creative_subtext": "...",
    "creative_cta": "...",
    "trust_element": "...",
    "visual_style": "..."
  }},
  {{
    "angle": "Aspiration",
    "framework": "BAB",
    "primary_text": "...",
    "headline": "...",
    "description": "...",
    "cta": "...",
    "audience_note": "...",
    "format_type": "...",
    "visual_description": "...",
    "creative_headline": "...",
    "creative_subtext": "...",
    "creative_cta": "...",
    "trust_element": "...",
    "visual_style": "..."
  }},
  {{
    "angle": "Social Proof",
    "framework": "Social Proof + FOMO",
    "primary_text": "...",
    "headline": "...",
    "description": "...",
    "cta": "...",
    "audience_note": "...",
    "format_type": "...",
    "visual_description": "...",
    "creative_headline": "...",
    "creative_subtext": "...",
    "creative_cta": "...",
    "trust_element": "...",
    "visual_style": "..."
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

    # #region agent log (087295)
    import json as _j2, time as _t2
    _LOG2 = "/Users/daschelgorgenyi/Desktop/Test project Speed run/.cursor/debug-087295.log"
    with open(_LOG2, "a") as _f2:
        _f2.write(_j2.dumps({"sessionId": "087295", "timestamp": int(_t2.time() * 1000), "hypothesisId": "H-C", "location": "ad_generator.py:_step3_generate_ads:format_types", "message": "AI returned format_types", "data": {"format_types": [i.get("format_type", "") for i in data], "angles": [i.get("angle", "") for i in data]}}) + "\n")
    # #endregion

    variations: list[AdVariation] = []
    for item in data:
        angle = item.get("angle", "")
        format_type = item.get("format_type", "")
        image_b64 = (
            _generate_ad_image(model, product_intel, angle, format_type)
            if generate_images
            else None
        )
        variations.append(
            AdVariation(
                angle=angle,
                framework=item.get("framework", ""),
                primary_text=item.get("primary_text", ""),
                headline=item.get("headline", "")[:40],
                description=item.get("description", "")[:30],
                cta=item.get("cta", "Learn More"),
                audience_note=item.get("audience_note", ""),
                image_b64=image_b64,
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
    model = _get_model()

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
    )

    return AdOutput(
        product_name=product_intel.get("name", request.product_name),
        brand_name=brand_name,
        variations=variations,
        voc_summary=voc_summary,
        product_intel=product_intel,
        errors=errors,
    )
