"""Generation pipeline with live Streamlit progress updates."""

from __future__ import annotations

import os
import traceback

import streamlit as st

from ad_generator import (
    _get_client,
    _generate_ad_image,
    _step1_extract_product_intel,
    _step2_synthesise_voc,
    _step3_generate_ads,
)
from models import AdOutput, GenerateRequest, ProductNotFoundError
from scraper import find_product_url, scrape_product_page
from voc import gather_voc


def run_generation(
    brand_url: str,
    product_name: str,
    brand_name: str,
    product_url_override: str,
    platform: str = "Meta Feed",
    campaign_goal: str = "Conversions",
    offer: str = "",
    landing_page_url: str = "",
) -> None:
    """
    Runs the full ad generation pipeline with a live st.status() progress block.
    Saves AdOutput to st.session_state.result on success, or sets
    st.session_state.last_error on failure.
    """
    request = GenerateRequest(
        brand_url=brand_url.strip(),
        product_name=product_name.strip(),
        product_url=product_url_override.strip() or None,
        brand_name=brand_name.strip() or None,
        platform=platform,
        campaign_goal=campaign_goal,
        offer=offer.strip(),
        landing_page_url=landing_page_url.strip(),
    )

    try:
        with st.status("Generating your Meta ads...", expanded=True) as status:
            spinner_slot = st.empty()
            spinner_slot.markdown(
                '<div class="adg-spinner-wrap">'
                '<div class="adg-spinner"></div>'
                '<div class="adg-spinner-label">Processing...</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            result = _generate_with_progress(request, status, spinner_slot)

        st.session_state.result = result
        st.session_state.last_error = None
        st.session_state.running = False

    except ProductNotFoundError as exc:
        st.session_state.show_product_url = True
        st.session_state.last_error = str(exc)
        st.session_state.running = False

    except ValueError as exc:
        st.session_state.last_error = str(exc)
        st.session_state.running = False

    except Exception as exc:
        st.session_state.last_error = (
            f"Unexpected error: {exc}\n\n"
            f"```\n{traceback.format_exc()}\n```"
        )
        st.session_state.running = False


def _generate_with_progress(request: GenerateRequest, status, spinner_slot=None) -> AdOutput:
    errors: list[str] = []
    client = _get_client()

    st.write("// Locating product page...")
    if request.product_url:
        product_url = request.product_url
        st.write("OK Using provided product URL")
    else:
        product_url = find_product_url(request.brand_url, request.product_name)
        st.write("OK Found product page")

    st.write("// Scraping product content...")
    page_content = scrape_product_page(product_url)
    st.write("OK Product page scraped")

    st.write("// Gathering consumer voice data (Reddit · YouTube · Google)...")
    voc_summary = gather_voc(request, errors)
    reddit_n = len(voc_summary.reddit_findings)
    yt_n = len(voc_summary.youtube_findings)
    ac_n = len(voc_summary.autocomplete_queries)
    st.write(f"OK Consumer research — {reddit_n} Reddit · {yt_n} YouTube · {ac_n} autocomplete")

    st.write("// Extracting product intelligence...")
    product_intel = _step1_extract_product_intel(client, page_content)
    st.write("OK Product intelligence extracted")

    brand_name = (
        request.brand_name
        or str(product_intel.get("brand_voice", ""))[:30]
        or request.brand_url.split("//")[-1].split(".")[0].title()
    )

    st.write("// Synthesising consumer brief...")
    voc_brief = _step2_synthesise_voc(client, product_intel, voc_summary)
    st.write("OK Consumer brief synthesised")

    st.write("// Writing your Meta ads...")
    variations = _step3_generate_ads(
        client,
        product_intel,
        voc_brief,
        generate_images=False,
        platform=request.platform,
        campaign_goal=request.campaign_goal,
        offer=request.offer,
        landing_page_url=request.landing_page_url,
    )
    st.write("OK Ad copy written")

    st.write("// Generating ad creatives with Gemini Imagen...")
    images_ok = 0
    for variation in variations:
        img = _generate_ad_image(client, product_intel, variation.angle, variation.format_type)
        variation.image_b64 = img
        if img:
            images_ok += 1
    if images_ok == len(variations):
        st.write(f"OK {images_ok} images generated")
    elif images_ok > 0:
        st.write(f"WARN {images_ok}/{len(variations)} images generated")
    else:
        st.write("WARN Image generation unavailable — copy is ready")

    if spinner_slot is not None:
        spinner_slot.empty()

    status.update(label="Complete // your ads are ready", state="complete", expanded=False)

    return AdOutput(
        product_name=product_intel.get("name", request.product_name),
        brand_name=brand_name,
        variations=variations,
        voc_summary=voc_summary,
        product_intel=product_intel,
        errors=errors,
    )
