"""
Brand Ad Generator — Streamlit Frontend

Aesthetic: dark futuristic / deep blue grid with glass panels
Fonts: Orbitron (headings) · Syne (body) · JetBrains Mono (technical/code)
"""

from __future__ import annotations

import os
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from models import AdOutput, ProductNotFoundError
from ui.components import ad_card, voc_panel
from ui.pipeline import run_generation
from ui.styles import inject_css

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ad Generator",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

_defaults: dict = {
    "result": None,
    "running": False,
    "show_product_url": False,
    "last_error": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="adg-wordmark">Ad<span>Gen</span></div>'
            '<div class="adg-tagline">Meta Copy // Powered by Gemini</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown(
            '<div class="adg-sidebar-key-label">Gemini API Key <span style="color:#F43F5E">*</span></div>',
            unsafe_allow_html=True,
        )
        gemini_key = st.text_input(
            "gemini_key_input",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            placeholder="AIza...",
            label_visibility="collapsed",
        )
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
        st.markdown(
            '<div class="adg-sidebar-link"><a href="https://aistudio.google.com/app/apikey" target="_blank" '
            'style="color:#00D4FF;text-decoration:none;">// Get free key at AI Studio</a></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="adg-sidebar-key-label">Reddit Client ID <span style="color:#243548">(optional)</span></div>',
            unsafe_allow_html=True,
        )
        reddit_id = st.text_input(
            "reddit_id_input",
            value=os.getenv("REDDIT_CLIENT_ID", ""),
            type="password",
            placeholder="e.g. aBcDeFgH1234",
            label_visibility="collapsed",
        )
        if reddit_id:
            os.environ["REDDIT_CLIENT_ID"] = reddit_id

        st.markdown(
            '<div class="adg-sidebar-key-label">Reddit Client Secret <span style="color:#243548">(optional)</span></div>',
            unsafe_allow_html=True,
        )
        reddit_secret = st.text_input(
            "reddit_secret_input",
            value=os.getenv("REDDIT_CLIENT_SECRET", ""),
            type="password",
            placeholder="e.g. xYz_secret_here",
            label_visibility="collapsed",
        )
        if reddit_secret:
            os.environ["REDDIT_CLIENT_SECRET"] = reddit_secret

        st.markdown(
            '<div class="adg-sidebar-link"><a href="https://www.reddit.com/prefs/apps" target="_blank" '
            'style="color:#00D4FF;text-decoration:none;">// Create a free Reddit app (2 min)</a></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="adg-sidebar-key-label">YouTube API Key <span style="color:#243548">(optional)</span></div>',
            unsafe_allow_html=True,
        )
        yt_key = st.text_input(
            "yt_key_input",
            value=os.getenv("YOUTUBE_API_KEY", ""),
            type="password",
            placeholder="AIza...",
            label_visibility="collapsed",
        )
        if yt_key:
            os.environ["YOUTUBE_API_KEY"] = yt_key
        st.markdown(
            '<div class="adg-sidebar-link"><a href="https://console.cloud.google.com/" target="_blank" '
            'style="color:#00D4FF;text-decoration:none;">// Get free key at Google Cloud (5 min)</a></div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:0.62rem;color:#243548;line-height:1.7;">'
            'Reddit + YouTube are optional. Google Autocomplete is used as a free fallback. '
            'Keys are stored only in this session.'
            '</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────────────────────────────────────

def _input_form() -> None:
    st.markdown(
        '<div class="adg-hero-title">'
        '<span class="adg-glitch" data-text="WRITE ADS THAT">WRITE ADS THAT</span><br>'
        '<span class="adg-glitch adg-glitch--accent" data-text="ACTUALLY CONVERT.">ACTUALLY CONVERT.</span>'
        '</div>'
        '<div class="adg-hero-sub">Paste a brand URL, name a product — get three research-backed '
        'Meta ads in under a minute.</div>',
        unsafe_allow_html=True,
    )

    with st.form("generate_form"):
        col1, col2 = st.columns([3, 2])

        with col1:
            brand_url = st.text_input(
                "Brand Website URL",
                placeholder="https://allbirds.com",
                help="The brand's main website. We'll find the product page automatically.",
            )
            product_name = st.text_input(
                "Product Name",
                placeholder="Men's Tree Runner",
                help="Type the product name as it appears on the site.",
            )

        with col2:
            brand_name = st.text_input(
                "Brand Name (optional)",
                placeholder="Allbirds",
                help="Used for Reddit/YouTube research. Inferred from URL if left blank.",
            )
            show_url = st.checkbox(
                "Paste product URL directly (skip auto-find)",
                value=st.session_state.show_product_url,
            )

        product_url_override = ""
        if show_url:
            product_url_override = st.text_input(
                "Direct Product Page URL",
                placeholder="https://allbirds.com/products/mens-tree-runner",
            )

        with st.expander("Creative Options // platform, goal, offer", expanded=False):
            st.markdown(
                '<div style="font-family:var(--font-mono);font-size:0.65rem;color:#4E7090;'
                'margin-bottom:0.75rem;letter-spacing:0.06em;">'
                'These fields inform the visual creative brief generated alongside each ad variation.'
                '</div>',
                unsafe_allow_html=True,
            )
            cr_col1, cr_col2 = st.columns(2)
            with cr_col1:
                platform = st.selectbox(
                    "Platform",
                    options=["Meta Feed", "Stories", "Both"],
                    index=0,
                    help="Affects aspect ratio guidance in the creative brief (4:5 Feed vs 1:1 Stories).",
                )
                offer = st.text_input(
                    "Offer (optional)",
                    placeholder="e.g. 20% off, free shipping, bundle deal",
                    help="Include a specific promotion if running one. Left blank = no offer framing.",
                )
            with cr_col2:
                campaign_goal = st.selectbox(
                    "Campaign Goal",
                    options=["Conversions", "Traffic", "Awareness"],
                    index=0,
                    help="Shapes CTA and urgency choices in the creative brief.",
                )
                landing_page_url = st.text_input(
                    "Landing Page URL (optional)",
                    placeholder="https://allbirds.com/products/mens-tree-runner",
                    help="If provided, the creative brief will note how the ad should visually match this page.",
                )

        submitted = st.form_submit_button(
            "Generate Ads //",
            use_container_width=False,
            disabled=st.session_state.running,
        )

    if submitted:
        if not os.getenv("GEMINI_API_KEY", "").strip():
            st.error("Add your Gemini API key in the sidebar to continue.")
            return
        if not brand_url.strip():
            st.error("Brand URL is required.")
            return
        if not product_name.strip():
            st.error("Product name is required.")
            return

        st.session_state.result = None
        st.session_state.last_error = None
        st.session_state.running = True
        st.session_state.show_product_url = False

        run_generation(
            brand_url,
            product_name,
            brand_name,
            product_url_override,
            platform=platform,
            campaign_goal=campaign_goal,
            offer=offer,
            landing_page_url=landing_page_url,
        )
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Results renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_results(output: AdOutput) -> None:
    if output.errors:
        with st.expander(f"WARN {len(output.errors)} data source(s) skipped", expanded=False):
            for err in output.errors:
                st.markdown(f'<div class="adg-error-item">// {err}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="adg-results-header">{output.product_name}</div>'
        f'<div class="adg-results-sub">{output.brand_name} &nbsp;//&nbsp; 3 Meta ad variations ready</div>',
        unsafe_allow_html=True,
    )

    for i, variation in enumerate(output.variations):
        ad_card(variation, i, output.brand_name)

    st.divider()
    voc_panel(output.voc_summary, output.product_intel)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("<< Generate for another product"):
        st.session_state.result = None
        st.session_state.last_error = None
        st.session_state.show_product_url = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _sidebar()

    _, main_col, _ = st.columns([0.5, 9, 0.5])

    with main_col:
        if st.session_state.last_error:
            if st.session_state.show_product_url:
                st.warning(
                    "**Product page not found automatically.**\n\n"
                    f"{st.session_state.last_error}\n\n"
                    "Check the box below to paste the product URL directly."
                )
            else:
                st.error(st.session_state.last_error)

        if st.session_state.result is not None:
            _render_results(st.session_state.result)
        else:
            _input_form()


if __name__ == "__main__":
    main()
