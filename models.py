from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerateRequest:
    """Input from the frontend to kick off ad generation."""
    brand_url: str           # e.g. "https://allbirds.com"
    product_name: str        # e.g. "Men's Tree Runner"
    product_url: str | None = None   # direct URL override — skips auto-discovery
    brand_name: str | None = None    # fallback for VoC search if unclear from URL
    country: str = "US"
    # Creative context fields (from master prompt)
    platform: str = "Meta Feed"        # "Meta Feed" | "Stories" | "Both"
    campaign_goal: str = "Conversions" # "Conversions" | "Traffic" | "Awareness"
    offer: str = ""                    # e.g. "20% off, free shipping"
    landing_page_url: str = ""         # used for ad-to-landing-page alignment notes


@dataclass
class AdVariation:
    """One complete Meta ad unit."""
    angle: str               # "Pain Point" | "Aspiration" | "Social Proof"
    framework: str           # "PAS" | "BAB" | "Social Proof + FOMO"
    primary_text: str        # full body copy (hook within first 125 chars)
    headline: str            # ≤40 chars
    description: str         # ≤30 chars
    cta: str                 # "Shop Now" | "Learn More" | "Get Offer" | "Order Now"
    audience_note: str       # targeting recommendation for Meta Ads Manager
    image_b64: str | None = None  # base64-encoded JPEG from compositor, None if unavailable
    creative_spec: dict = field(default_factory=dict)  # structured layout spec (Phase 1: derived from copy)
    # Static creative brief fields (from master prompt)
    format_type: str = ""        # e.g. "Product Hero" | "Bold Text Overlay" | "Testimonial Card"
    visual_description: str = "" # detailed layout description of the 1080px canvas
    creative_headline: str = ""  # on-image hook text (6-8 words)
    creative_subtext: str = ""   # supporting line overlaid on the creative
    creative_cta: str = ""       # on-image CTA text (e.g. "Shop the Set")
    trust_element: str = ""      # e.g. "4.8★ from 3,200+ reviews"
    visual_style: str = ""       # background colour, typography feel, overall aesthetic


@dataclass
class VocSummary:
    """Structured consumer intelligence gathered before ad generation."""
    reddit_findings: list[str] = field(default_factory=list)
    youtube_findings: list[str] = field(default_factory=list)
    autocomplete_queries: list[str] = field(default_factory=list)
    synthesized_persona: str = ""   # Gemini's inferred target consumer description


@dataclass
class AdOutput:
    """Everything the frontend needs to render results."""
    product_name: str
    brand_name: str
    variations: list[AdVariation]     # always 3
    voc_summary: VocSummary
    product_intel: dict               # raw Step 1 JSON for display in research panel
    errors: list[str] = field(default_factory=list)  # non-fatal warnings
    product_image_url: str | None = None   # canonical product image URL from scraper
    product_image_b64: str | None = None   # base64 JPEG for use in ad preview
    brand_logo_b64: str | None = None      # base64-encoded brand logo for avatar


class ProductNotFoundError(Exception):
    """Raised when auto-discovery cannot locate a product page on the brand site."""
    pass
