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
import logging
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
_IMAGE_MODEL = "imagen-4.0-generate-001"

log = logging.getLogger(__name__)

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
  "visual_appearance": "concise physical description of what the product looks like — colour, shape, material, size, form factor (e.g. 'white mesh running shoe with bright orange foam midsole and black outsole')"
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

def _build_image_prompt(product_intel: dict, angle: str, format_type: str = "") -> str:
    name = product_intel.get("name", "product")
    features = ", ".join(product_intel.get("key_features", [])[:3])
    brand_voice = product_intel.get("brand_voice", "")
    audience = product_intel.get("target_audience_signals", "")
    visual_appearance = product_intel.get("visual_appearance", "")

    feature_line = f" Key features: {features}." if features else ""
    brand_line = f" Brand feel: {brand_voice}." if brand_voice else ""
    audience_line = f" Target customer: {audience}." if audience else ""
    # Anchor every prompt with a visual description so Imagen renders the actual product
    product_ref = f"{name} ({visual_appearance})" if visual_appearance else name

    # Format-type-specific overrides (take precedence over angle defaults)
    if format_type == "Testimonial Card":
        return (
            f"Premium product photography for Meta/Instagram feed. "
            f"The product is a {product_ref}.{feature_line}"
            f" The {product_ref} is the clear hero, displayed on a clean off-white or warm neutral surface "
            f"with soft studio lighting and gentle shadows. Product fills 60% of the frame, "
            f"perfectly sharp with fine detail visible. Generous empty space above and below "
            f"the product. Warm, trustworthy, community feel.{brand_line}"
            f" No people. No text. No logos. Square 1:1 format."
        )
    elif format_type == "Flat Lay":
        return (
            f"Editorial flat-lay product photograph for Meta/Instagram ad. "
            f"The product is a {product_ref}.{feature_line}"
            f" Top-down overhead shot: the {product_ref} arranged elegantly with 2-3 complementary props "
            f"(neutral lifestyle objects that match the brand aesthetic) on a textured surface "
            f"(linen, marble, or natural wood). Warm, even studio lighting. "
            f"Looks like a high-end editorial magazine spread.{brand_line}"
            f" No text overlays. No logos. Square 1:1 format."
        )
    elif format_type in ("Bold Text Overlay", "Problem-Agitate-Solution"):
        return (
            f"High-contrast Meta/Instagram feed ad. The product is a {product_ref}.{feature_line}"
            f" The {product_ref} is positioned dramatically — left or right third of frame — "
            f"against a deep dark background (charcoal, navy, or near-black) with a single "
            f"powerful directional light source creating strong shadows and highlights on the product. "
            f"Leaves significant clear space (right or left half) for bold text overlay. "
            f"Photorealistic, high drama, premium commercial quality.{brand_line}"
            f" No text. No logos. Square 1:1 format."
        )
    elif format_type == "Before/After":
        return (
            f"Transformation Meta/Instagram ad. The product is a {product_ref}.{feature_line}"
            f" Split composition: left half shows a muted, desaturated lifestyle scene "
            f"representing struggle or the 'before' state; right half shows the same scene "
            f"bright, vibrant, and joyful representing the transformed outcome. "
            f"The {product_ref} is featured prominently and clearly visible in the right (after) half. "
            f"No before/after labels. Photorealistic.{brand_line}{audience_line}"
            f" No text overlays. No logos. Square 1:1 format."
        )
    elif format_type == "Offer/Promo":
        return (
            f"High-energy Meta/Instagram promotional ad. The product is a {product_ref}.{feature_line}"
            f" The {product_ref} is the hero, centred against a bold brand-coloured background "
            f"with a clean gradient or solid fill. Bright, energetic studio lighting. "
            f"Product fills 60% of the frame. Leaves clear space at top and bottom for "
            f"price/offer text overlays. Optimistic, urgent feel.{brand_line}"
            f" No text. No logos. Square 1:1 format."
        )

    # Angle-based fallbacks (Product Hero is also the Pain Point default)
    if angle == "Pain Point":
        return (
            f"High-impact Meta/Instagram feed ad. The product is a {product_ref}.{feature_line}"
            f" Hero product shot: the {product_ref} isolated on a clean gradient background "
            f"(light grey to white), dramatic studio rim lighting from the side and above "
            f"that reveals every texture and contour of the product. "
            f"The product fills 70% of the frame. Extremely sharp focus. "
            f"Subtle drop shadow grounds the product. "
            f"Photorealistic, ultra high detail, premium commercial advertising photography quality."
            f"{brand_line} No people. No text. No logos. Square 1:1 format."
        )
    elif angle == "Aspiration":
        return (
            f"Aspirational Meta/Instagram lifestyle ad. The product is a {product_ref}.{feature_line}"
            f" The {product_ref} is shown in active use in a beautiful real-world setting — "
            f"outdoors in golden-hour afternoon light, warm amber and orange tones."
            f" A person's lower body (feet, legs) is shown actively using the {product_ref}; "
            f"the product must be clearly visible and identifiable in the frame. Dynamic, in-motion feel."
            f" Shallow depth of field, bokeh background. "
            f"Looks like a premium editorial shoot for a major brand campaign."
            f"{brand_line}{audience_line}"
            f" No text overlays. No logos. Square 1:1 format."
        )
    else:  # Social Proof
        return (
            f"Authentic lifestyle Meta/Instagram ad. The product is a {product_ref}.{feature_line}"
            f" Candid, real-world moment — a person actively using the {product_ref} in an everyday "
            f"relatable setting. The {product_ref} must be clearly visible and identifiable in the frame. "
            f"Natural indoor or outdoor light, feels warm and genuine."
            f" The person's face may be shown — warm, happy, friendly expression."
            f" Community vibe, feels like a real customer photo but shot professionally."
            f"{brand_line}{audience_line}"
            f" No text overlays. No logos. Square 1:1 format."
        )


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


def _generate_ad_image(
    client: genai.Client,
    product_intel: dict,
    angle: str,
    format_type: str = "",
    product_image_bytes: bytes | None = None,
) -> str | None:
    """
    Calls Imagen to generate a product image for the given ad angle.
    When product_image_bytes is provided, attempts a subject-reference call first
    so Imagen anchors the creative to the real product; falls back to text-only.
    Returns a base64-encoded JPEG string, or None if generation fails.
    """
    prompt = _build_image_prompt(product_intel, angle, format_type)
    img_config = genai_types.GenerateImagesConfig(
        number_of_images=1,
        aspect_ratio="1:1",
        output_mime_type="image/jpeg",
        person_generation="ALLOW_ADULT",
    )

    # Attempt reference-image call when we have the real product photo
    if product_image_bytes:
        try:
            ref_image = genai_types.ReferenceImage(
                reference_type="SUBJECT",
                reference_image=genai_types.Image(image_bytes=product_image_bytes),
            )
            response = client.models.generate_images(
                model=_IMAGE_MODEL,
                prompt=prompt,
                reference_images=[ref_image],
                config=img_config,
            )
            if response.generated_images:
                raw = response.generated_images[0].image.image_bytes
                return base64.b64encode(raw).decode("utf-8")
            log.warning("Reference-image call returned 0 images for angle '%s'; falling back", angle)
        except Exception as exc:
            log.warning("Reference-image call failed for angle '%s': %s — falling back to text-only", angle, exc)

    # Text-only fallback (also the primary path when no product image is available)
    try:
        response = client.models.generate_images(
            model=_IMAGE_MODEL,
            prompt=prompt,
            config=img_config,
        )
        if response.generated_images:
            raw = response.generated_images[0].image.image_bytes
            return base64.b64encode(raw).decode("utf-8")
    except Exception as exc:
        log.warning("Image generation failed for angle '%s': %s", angle, exc)
    return None


# ---------------------------------------------------------------------------
# Step 3 — Meta Ad Generation (Copywriting Constitution)
# ---------------------------------------------------------------------------

_STEP3_SYSTEM = """
You are one of the world's best direct-response copywriters AND a senior
performance creative strategist specialising in high-converting static Meta
(Facebook / Instagram) ads for cold prospecting audiences.

You follow every rule in the copywriting constitution and static creative
direction framework below without exception.

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
PART 2 — STATIC CREATIVE DIRECTION FRAMEWORK
═══════════════════════════════════════════════════════════════

FORMAT TYPES — assign exactly one per variation:
  • Product Hero        — clean product shot, branded background, minimal text
                          Best for: Pain Point angle (sharp, direct)
  • Bold Text Overlay   — large hook text over lifestyle/product image
                          Best for: Pain Point angle (urgent, confrontational)
  • Testimonial Card    — customer quote + star rating + product
                          Best for: Social Proof angle
  • Before/After        — visual transformation (do NOT label "Before" or "After")
                          Best for: Pain Point or Aspiration angle
  • Problem-Agitate-Solution — pain imagery → intensified → product as relief
                          Best for: Pain Point angle
  • Offer/Promo         — discount + product + urgency element
                          Use only if a genuine offer is provided
  • Flat Lay            — styled editorial arrangement of product + accessories
                          Best for: Aspiration angle (premium feel)

VISUAL HIERARCHY — describe placement following this attention sequence:
  1st eye → Product or hero image (largest element, 30–40% of frame)
  2nd eye → Value proposition / hook (bold, 6–8 words max, on-image)
  3rd eye → Social proof (stars, review count, press badge, result claim)
  4th eye → CTA (specific action verb — "Shop the Set", never "Learn More")

COMPOSITION RULES:
  • Aspect ratio: 4:5 vertical (1080×1350px) default for Feed;
    1:1 (1080×1080px) for Stories
  • Maintain 30–40% white/negative space
  • Maximum 3 focal points (3-Element Rule)
  • Product is the unmistakable hero
  • Text never covers the product
  • High-contrast text for mobile readability
  • Price visible if it qualifies intent — readable, not dominant
  • On-image text: 6–8 words MAX (less text = better delivery algorithm score)
  • On-image text and primary text must NOT repeat the same words — they
    complement each other

ON-IMAGE COPY RULES:
  creative_headline: 6–8 words, product-led, punchy, not a generic slogan
    Examples: "Finally shoes that don't destroy your knees"
              "The last pillow you'll ever buy"
  creative_subtext: one line of supporting proof or benefit (preferred, not required)
  creative_cta: specific action — "Get Yours", "Try It Risk-Free", "Shop the Set"
    (NEVER "Learn More", "Check It Out", "Click Here")
  trust_element — include exactly one:
    • Star rating + volume: "4.8★ from 3,200+ reviews"
    • Press credibility: "As Seen In Forbes · NYT · Vogue"
    • Result claim: "Results in 14 days — or your money back"
    • Third-party: "Trustpilot Excellent · 10,000+ reviews"
    • Outcome stat: "9 out of 10 customers reorder"

COLOR & STYLE DIRECTION per angle:
  Pain Point   → Bold & Urgent: dark/high-contrast background, sharp typography,
                 accent colour that pops against the Meta feed
  Aspiration   → Premium / Editorial: warm neutrals or brand colour, refined
                 sans-serif, generous white space
  Social Proof → Clean & Warm: light background, friendly community photography,
                 avoid pure white if the Meta feed is white

visual_description: 3–5 sentences briefing a designer — what element is where,
  what size, what colour, what text appears on canvas, overall visual feel.

visual_style: one sentence: background, typography style, aesthetic label.
  Example: "Dark charcoal gradient, bold white Helvetica hook, Urgent/Premium feel"
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

Rules for this run:
- Each variation must use a DIFFERENT hook formula (Pain-Point Question, Bold
  Outcome, Specific Proof, Curiosity Gap, Us vs. Them, or Social Proof Lead)
- Each variation must use the most appropriate framework (PAS, BAB, AIDA, FAB,
  or Value Stack) — do not default to PAS for all three
- headline: target ≤27 chars (hard max 40). Short and punchy wins.
- cta: choose from the approved strong CTA list — NEVER use "Learn More"
- For the creative brief: choose the most appropriate format_type, write a full
  visual_description (3–5 sentences), and complete all on-image copy fields
- On-image copy and primary text must use DIFFERENT words — each zone has a
  distinct job
- If an offer is provided, incorporate urgency into at least one variation
- If a landing page URL is provided, note visual continuity in visual_description

Return this exact JSON structure:
[
  {{
    "angle": "Pain Point",
    "framework": "PAS or FAB or AIDA — whichever fits best",
    "primary_text": "...",
    "headline": "...(≤27 chars target)...",
    "description": "...(≤30 chars)...",
    "cta": "...(strong CTA — not Learn More)...",
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
    "framework": "BAB or AIDA or Value Stack — whichever fits best",
    "primary_text": "...",
    "headline": "...(≤27 chars target)...",
    "description": "...(≤30 chars)...",
    "cta": "...(strong CTA — not Learn More)...",
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
    "framework": "Social Proof + FOMO or AIDA — whichever fits best",
    "primary_text": "...",
    "headline": "...(≤27 chars target)...",
    "description": "...(≤30 chars)...",
    "cta": "...(strong CTA — not Learn More)...",
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
    product_image_bytes: bytes | None = None,
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
        image_b64 = (
            _generate_ad_image(model, product_intel, angle, format_type, product_image_bytes)
            if generate_images
            else None
        )
        variations.append(
            AdVariation(
                angle=angle,
                framework=item.get("framework", ""),
                primary_text=item.get("primary_text", ""),
                headline=item.get("headline", "")[:40],  # hard max 40; target ≤27
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
    )

    return AdOutput(
        product_name=product_intel.get("name", request.product_name),
        brand_name=brand_name,
        variations=variations,
        voc_summary=voc_summary,
        product_intel=product_intel,
        errors=errors,
    )
