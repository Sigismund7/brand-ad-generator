"""Reusable UI components: ad cards, VoC panel, char counter."""

from __future__ import annotations

import html

import streamlit as st

from models import AdVariation, VocSummary

_ANGLE_COLORS = {
    "Pain Point":   ("#00D4FF", "rgba(0, 212, 255, 0.08)"),
    "Aspiration":   ("#8B5CF6", "rgba(139, 92, 246, 0.08)"),
    "Social Proof": ("#22D3A0", "rgba(34, 211, 160, 0.08)"),
}
_ANGLE_LABELS = {
    "Pain Point":   "PAS Framework",
    "Aspiration":   "BAB Framework",
    "Social Proof": "Social Proof + FOMO",
}


def char_badge(text: str, limit: int) -> str:
    n = len(text)
    pct = n / limit
    cls = "adg-char-ok" if pct <= 0.85 else ("adg-char-warn" if pct <= 1.0 else "adg-char-over")
    icon = "OK" if n <= limit else "OVER"
    return f'<span class="{cls}">{icon} {n}/{limit}</span>'


def ad_card(variation: AdVariation, idx: int, brand_name: str = "") -> None:
    color, bg = _ANGLE_COLORS.get(variation.angle, ("#4E7090", "rgba(78,112,144,0.08)"))
    framework_label = _ANGLE_LABELS.get(variation.angle, variation.framework)

    st.markdown(
        f'<div class="meta-preview-label">'
        f'<span class="adg-badge" style="background:{bg};color:{color};">{framework_label}</span>'
        f'<span class="adg-angle">{html.escape(variation.angle)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    preview_col, copy_col = st.columns([9, 11])

    with preview_col:
        PREVIEW_CHARS = 220
        if len(variation.primary_text) > PREVIEW_CHARS:
            short = html.escape(variation.primary_text[:PREVIEW_CHARS])
            text_html = f'{short}... <span class="meta-see-more-link">See more</span>'
        else:
            text_html = html.escape(variation.primary_text)

        if variation.image_b64:
            image_html = (
                f'<img class="meta-ad-image" '
                f'src="data:image/jpeg;base64,{variation.image_b64}" '
                f'alt="Ad creative" style="width:100%;display:block;" />'
            )
        else:
            image_html = (
                '<div class="meta-image-placeholder">'
                '<span style="font-size:2rem;line-height:1;">🖼</span>'
                '<span>Generating creative...</span>'
                '</div>'
            )

        initial = (brand_name or "B")[0].upper()
        safe_brand = html.escape(brand_name or "Brand")
        domain = (brand_name or "brand").lower().replace(" ", "") + ".com"

        st.markdown(
            f"""
            <div class="meta-card">
              <div class="meta-post-header">
                <div class="meta-avatar">{initial}</div>
                <div class="meta-page-info">
                  <div class="meta-page-name">{safe_brand}</div>
                  <div class="meta-sponsored">Sponsored &nbsp;·&nbsp; 🌐</div>
                </div>
                <div class="meta-more-btn">···</div>
              </div>
              <div class="meta-primary-text-block">{text_html}</div>
              {image_html}
              <div class="meta-info-strip">
                <div class="meta-strip-left">
                  <div class="meta-domain-text">{domain}</div>
                  <div class="meta-headline-text">{html.escape(variation.headline)}</div>
                  <div class="meta-desc-text">{html.escape(variation.description)}</div>
                </div>
                <button class="meta-cta-btn">{html.escape(variation.cta)}</button>
              </div>
              <div class="meta-reactions-bar">
                <div class="meta-reaction-btn">👍 Like</div>
                <div class="meta-reaction-btn">💬 Comment</div>
                <div class="meta-reaction-btn">↗ Share</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="adg-field-label">Headline</div>', unsafe_allow_html=True)
            st.code(variation.headline, language=None)
            st.markdown(char_badge(variation.headline, 40), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="adg-field-label">Description</div>', unsafe_allow_html=True)
            st.code(variation.description, language=None)
            st.markdown(char_badge(variation.description, 30), unsafe_allow_html=True)

        col3, _ = st.columns([1, 3])
        with col3:
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
