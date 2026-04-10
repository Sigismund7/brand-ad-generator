"""Reusable UI components: ad cards, VoC panel, char counter."""

from __future__ import annotations

import base64
import html
import io
from textwrap import dedent

import streamlit as st
from PIL import Image

from models import AdVariation, VocSummary

_ANGLE_COLORS = {
    "Pain Point":   ("#00D4FF", "rgba(0, 212, 255, 0.08)"),
    "Aspiration":   ("#8B5CF6", "rgba(139, 92, 246, 0.08)"),
    "Social Proof": ("#22D3A0", "rgba(34, 211, 160, 0.08)"),
}
_ANGLE_LABELS = {
    "Pain Point":   "Pain Point",
    "Aspiration":   "Aspiration",
    "Social Proof": "Social Proof",
}


def _logo_src_from_b64(logo_b64: str) -> str | None:
    """
    Build a correct data: URL for the downloaded logo. Logos are often JPEG/WebP/ICO/SVG;
    assuming PNG breaks the browser image (broken icon + clipped alt text).
    """
    try:
        raw = base64.b64decode(logo_b64)
    except Exception:
        return None
    if len(raw) < 8:
        return None
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return f"data:image/png;base64,{logo_b64}"
    if raw.startswith(b"\xff\xd8\xff"):
        return f"data:image/jpeg;base64,{logo_b64}"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return f"data:image/gif;base64,{logo_b64}"
    if raw.startswith(b"RIFF") and len(raw) > 12 and raw[8:12] == b"WEBP":
        return f"data:image/webp;base64,{logo_b64}"
    strip = raw.lstrip()
    head = strip[:300].lower()
    if b"<svg" in head or head.startswith(b"<?xml") or head.startswith(b"<!doctype"):
        return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")
    try:
        im = Image.open(io.BytesIO(raw))
        buf = io.BytesIO()
        im.convert("RGBA").save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _teaser_upto_word_boundary(text: str, max_chars: int) -> str:
    """Truncate for preview so we do not split mid-word before 'See more'."""
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    last_break = max(chunk.rfind(" "), chunk.rfind("\n"), chunk.rfind("\t"))
    if last_break > max_chars // 4:
        return chunk[:last_break].rstrip()
    return chunk.rstrip()


def char_badge(text: str, limit: int) -> str:
    n = len(text)
    pct = n / limit
    cls = "adg-char-ok" if pct <= 0.85 else ("adg-char-warn" if pct <= 1.0 else "adg-char-over")
    icon = "OK" if n <= limit else "OVER"
    return f'<span class="{cls}">{icon} {n}/{limit}</span>'


def ad_card(variation: AdVariation, idx: int, brand_name: str = "", product_image_b64: str | None = None, logo_b64: str | None = None) -> None:
    color, bg = _ANGLE_COLORS.get(variation.angle, ("#4E7090", "rgba(78,112,144,0.08)"))
    framework_label = variation.framework or _ANGLE_LABELS.get(variation.angle, variation.angle)

    st.markdown(
        f'<div class="meta-preview-label">'
        f'<span class="adg-badge" style="background:{bg};color:{color};">{html.escape(framework_label)}</span>'
        f'<span class="adg-angle">{html.escape(variation.angle)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Wider copy column; avoid nested st.columns here — they shrink to unusable widths.
    preview_col, copy_col = st.columns([6, 14])

    with preview_col:
        PREVIEW_CHARS = 220
        if len(variation.primary_text) > PREVIEW_CHARS:
            teaser_raw = _teaser_upto_word_boundary(variation.primary_text, PREVIEW_CHARS)
            short = html.escape(teaser_raw)
            full_body = html.escape(variation.primary_text)
            if teaser_raw:
                text_html = (
                    '<details class="meta-primary-details">'
                    '<summary class="meta-primary-summary">'
                    f'<span class="meta-primary-teaser">{short}... </span>'
                    '<span class="meta-see-more-link">See more</span>'
                    '<span class="meta-see-less-link">See less</span></summary>'
                    f'<div class="meta-primary-rest">{full_body}</div>'
                    "</details>"
                )
            else:
                text_html = html.escape(variation.primary_text)
        else:
            text_html = html.escape(variation.primary_text)

        if variation.image_b64:
            placeholder_html = ""
            reference_html = ""
        else:
            placeholder_html = (
                '<div class="meta-image-placeholder meta-image-unavailable">'
                '<span style="font-size:1.75rem;line-height:1;">⚠</span>'
                '<span><strong>AI image not generated</strong> — the image model may be '
                "unavailable, overloaded, or rate-limited. Copy above is still valid; "
                "try again in a minute.</span>"
                "</div>"
            )
            if product_image_b64:
                reference_html = (
                    '<div class="meta-product-reference">'
                    '<span class="meta-product-reference-label">Reference — product on store</span>'
                    '<img class="meta-product-reference-img" '
                    f'src="data:image/jpeg;base64,{product_image_b64}" '
                    'alt="Store product reference" style="width:100%;display:block;" />'
                    "</div>"
                )
            else:
                reference_html = ""

        initial = (brand_name or "B")[0].upper()
        safe_brand = html.escape(brand_name or "Brand")
        domain = (brand_name or "brand").lower().replace(" ", "") + ".com"
        safe_domain = html.escape(domain)

        if logo_b64:
            logo_src = _logo_src_from_b64(logo_b64)
            if logo_src:
                avatar_content = (
                    f'<img src="{logo_src}" alt="" role="presentation" '
                    'width="40" height="40" loading="lazy" />'
                )
            else:
                avatar_content = initial
        else:
            avatar_content = initial

        # Indented HTML inside st.markdown is parsed as a Markdown code block (leading
        # 4+ spaces). Dedented markup renders as real HTML.
        # Large data: URIs break Streamlit markdown; render AI image via st.image above the strip.
        # Two valid HTML fragments (Streamlit blocks are separate DOM trees). Do not span
        # one <div class="meta-card"> across markdown + st.image + markdown — orphan tags
        # caused a duplicate “ghost” strip + reactions bar below the real preview.
        card_top_html = dedent(
            f"""
            <div class="meta-card meta-card--top">
              <div class="meta-post-header">
                <div class="meta-avatar">{avatar_content}</div>
                <div class="meta-page-info">
                  <div class="meta-page-name">{safe_brand}</div>
                  <div class="meta-sponsored">Sponsored &nbsp;·&nbsp; 🌐</div>
                </div>
                <div class="meta-more-btn">···</div>
              </div>
              <div class="meta-primary-text-block">{text_html}</div>
            </div>
            """
        ).strip()

        card_bottom_html = dedent(
            f"""
            <div class="meta-card meta-card--bottom">
              {reference_html}
              <div class="meta-info-strip">
                <div class="meta-strip-left">
                  <div class="meta-domain-text">{safe_domain}</div>
                  <div class="meta-headline-text">{html.escape(variation.headline)}</div>
                  <div class="meta-desc-text">{html.escape(variation.description)}</div>
                </div>
                <span class="meta-cta-btn" role="button">{html.escape(variation.cta)}</span>
              </div>
              <div class="meta-reactions-bar">
                <div class="meta-reaction-btn">👍 Like</div>
                <div class="meta-reaction-btn">💬 Comment</div>
                <div class="meta-reaction-btn">↗ Share</div>
              </div>
            </div>
            """
        ).strip()

        st.markdown(card_top_html, unsafe_allow_html=True)
        # st.image must sit between markdown blocks; wrapping in a separate open/close
        # <div> does not nest the image in Streamlit and adds a visible gap + broken HTML.
        if variation.image_b64:
            raw_bytes = base64.b64decode(variation.image_b64)
            pil_img = Image.open(io.BytesIO(raw_bytes))
            st.image(pil_img, use_container_width=True)
        else:
            st.markdown(
                f'<div class="meta-card meta-card--media">{placeholder_html}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(card_bottom_html, unsafe_allow_html=True)

    with copy_col:
        st.markdown('<div class="adg-field-label">Primary Text</div>', unsafe_allow_html=True)
        hook_preview = variation.primary_text[:125]
        rest = variation.primary_text[125:]
        annotated = (
            f"{hook_preview}\n↑ visible before 'See More' on mobile\n\n{rest}" if rest
            else variation.primary_text
        )
        st.code(annotated, language=None)
        st.markdown(
            char_badge(variation.primary_text, 500)
            + ' &nbsp;&middot;&nbsp; <span class="adg-char-ok" style="font-size:0.68rem">'
            + f"Hook: {len(hook_preview)} chars</span>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="adg-field-label">Headline</div>', unsafe_allow_html=True)
        st.code(variation.headline, language=None)
        st.markdown(char_badge(variation.headline, 27), unsafe_allow_html=True)

        st.markdown('<div class="adg-field-label">Description</div>', unsafe_allow_html=True)
        st.code(variation.description, language=None)
        st.markdown(char_badge(variation.description, 30), unsafe_allow_html=True)

        st.markdown('<div class="adg-field-label">CTA Button</div>', unsafe_allow_html=True)
        st.code(variation.cta, language=None)

        st.markdown(
            f'<div class="adg-audience"><strong>Targeting //</strong> {html.escape(variation.audience_note)}</div>',
            unsafe_allow_html=True,
        )

    has_brief = any([
        variation.format_type,
        variation.visual_description,
        variation.creative_headline,
        variation.trust_element,
    ])

    if has_brief:
        format_badge = (
            f'<span class="adg-format-badge">{html.escape(variation.format_type)}</span>'
            if variation.format_type else ""
        )
        style_badge = (
            f'<span class="adg-style-badge">{html.escape(variation.visual_style)}</span>'
            if variation.visual_style else ""
        )
        visual_desc_html = (
            f'<div class="adg-visual-desc">{html.escape(variation.visual_description)}</div>'
            if variation.visual_description else ""
        )

        def _oic(label: str, value: str, extra_cls: str = "") -> str:
            if not value:
                return ""
            return (
                f'<div class="adg-onimage-cell">'
                f'<div class="adg-onimage-label">{label}</div>'
                f'<div class="adg-onimage-value {extra_cls}">{html.escape(value)}</div>'
                f'</div>'
            )

        onimage_cells = "".join(filter(None, [
            _oic("On-Image Hook", variation.creative_headline),
            _oic("On-Image Subtext", variation.creative_subtext),
            _oic("On-Image CTA", variation.creative_cta),
            _oic("Trust Element", variation.trust_element, "adg-onimage-value--proof"),
        ]))

        grid_html = (
            f'<div class="adg-onimage-grid">{onimage_cells}</div>'
            if onimage_cells else ""
        )

        st.markdown(
            f'<div class="adg-creative-brief">'
            f'<div class="adg-brief-header">'
            f'<span class="adg-brief-title">Static Creative Brief</span>'
            f'{format_badge}{style_badge}'
            f'</div>'
            f'{visual_desc_html}'
            f'{grid_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)


def voc_panel(voc: VocSummary, product_intel: dict) -> None:
    with st.expander("Consumer Research // what informed these ads", expanded=False):
        if voc.synthesized_persona:
            st.markdown('<div class="adg-section-label">Synthesized Target Persona</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="adg-persona-box">{voc.synthesized_persona}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if voc.reddit_findings:
                st.markdown('<div class="adg-section-label">Reddit Discussions</div>', unsafe_allow_html=True)
                chips = "".join(
                    f'<span class="adg-voc-chip">{s[:80]}</span>'
                    for s in voc.reddit_findings[:12]
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="adg-section-label">Reddit</div>'
                    '<div style="font-size:0.75rem;color:#243548;font-family:var(--font-mono);">No data — key not set or no results found.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            if voc.autocomplete_queries:
                st.markdown('<div class="adg-section-label">Google Pre-Purchase Searches</div>', unsafe_allow_html=True)
                chips = "".join(
                    f'<span class="adg-voc-chip">{s}</span>'
                    for s in voc.autocomplete_queries[:15]
                )
                st.markdown(chips, unsafe_allow_html=True)

        with col2:
            if voc.youtube_findings:
                st.markdown('<div class="adg-section-label">YouTube Review Comments</div>', unsafe_allow_html=True)
                for comment in voc.youtube_findings[:6]:
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:#4E7090;padding:0.4rem 0;'
                        f'border-bottom:1px solid rgba(0,212,255,0.06);font-family:var(--font-mono);">'
                        f'"{comment[:200]}"</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="adg-section-label">YouTube</div>'
                    '<div style="font-size:0.75rem;color:#243548;font-family:var(--font-mono);">No data — key not set or no results found.</div>',
                    unsafe_allow_html=True,
                )

        if product_intel:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="adg-section-label">Extracted Product Intelligence</div>', unsafe_allow_html=True)
            st.json(product_intel, expanded=False)
