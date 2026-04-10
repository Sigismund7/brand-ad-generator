"""Generation pipeline with live Streamlit progress updates."""

from __future__ import annotations

import time
import traceback

import streamlit as st

import base64

from ad_generator import (
    _brief_to_prompt,
    _download_image_bytes,
    _generate_ad_image_nb,
    _generate_all_image_briefs,
    _get_client,
    _step1_extract_product_intel,
    _step2_synthesise_voc,
    _step3_generate_ads,
)
from models import AdOutput, AdVariation, GenerateRequest, ProductNotFoundError
from scraper import extract_product_image_url, fetch_brand_logo_url, find_product_url, scrape_product_page
from voc import gather_voc


def _run_nb_image_loop(
    client,
    variations: list[AdVariation],
    briefs: list[dict],
) -> int:
    """
    Generate image 1, sleep 5s, image 2, sleep 5s, image 3 (4 API calls total including the batch brief step).
    After each successful or failed _generate_ad_image_nb, wait 5s before the next image (not after the last).
    """
    images_ok = 0
    n = len(variations)

    for i, variation in enumerate(variations):
        prompt = _brief_to_prompt(briefs[i])
        print(f"[IMAGE] {variation.angle} prompt length={len(prompt)}")
        img = _generate_ad_image_nb(client, prompt, aspect_ratio="4:5")
        variation.image_b64 = img
        if img:
            images_ok += 1

        if i < n - 1:
            time.sleep(5)

    return images_ok


def run_generation(
    brand_url: str,
    product_name: str,
    brand_name: str,
    product_url_override: str,
    platform: str = "Meta Feed",
    campaign_goal: str = "Conversions",
    offer: str = "",
    landing_page_url: str = "",
    *,
    images_only: bool = False,
) -> None:
    """
    Runs the full ad generation pipeline with a live st.status() progress block.
    Saves AdOutput to st.session_state.result on success, or sets
    st.session_state.last_error on failure.

    When images_only=True, skips scraping/steps 1–3 and only runs Nano Banana
    using st.session_state.cached_image_briefs (3 calls).
    """
    if images_only:
        request = None
    else:
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
        with st.status(
            "Retrying images..." if images_only else "Generating your Meta ads...",
            expanded=True,
        ) as status:
            spinner_slot = st.empty()
            spinner_slot.markdown(
                '<div class="adg-spinner-wrap">'
                '<div class="adg-spinner"></div>'
                '<div class="adg-spinner-label">Processing...</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            if images_only:
                result = _generate_images_from_cache(status, spinner_slot)
            else:
                assert request is not None
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


def _generate_images_from_cache(status, spinner_slot=None) -> AdOutput:
    """Regenerate images only; requires result + cached_image_briefs in session_state."""
    result = st.session_state.result
    briefs = st.session_state.cached_image_briefs
    if result is None or not briefs or len(briefs) != len(result.variations):
        raise ValueError(
            "Cannot retry images — cached briefs missing. Run a full generation first."
        )

    client = _get_client()
    st.write("// Retrying images (cached briefs, no text call)...")
    images_ok = _run_nb_image_loop(client, result.variations, briefs)

    if images_ok == len(result.variations):
        st.write(f"OK {images_ok} images generated")
    elif images_ok > 0:
        st.write(f"WARN {images_ok}/{len(result.variations)} images generated")
    else:
        st.write("WARN Image generation unavailable — copy is ready")

    if spinner_slot is not None:
        spinner_slot.empty()

    status.update(label="Complete // your ads are ready", state="complete", expanded=False)
    return result


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

    st.write("// Extracting product image...")
    product_image_url = extract_product_image_url(product_url)
    print(f"[pipeline] product_image_url={product_image_url!r}")
    if product_image_url:
        st.write("OK Product image found")
    else:
        st.write("WARN Could not extract product image — text-only creative preview")
    product_image_bytes = _download_image_bytes(product_image_url) if product_image_url else None
    print(f"[pipeline] product_image_bytes len={len(product_image_bytes) if product_image_bytes else 0}")
    product_image_b64 = base64.b64encode(product_image_bytes).decode() if product_image_bytes else None

    st.write("// Searching for brand logo...")
    logo_url = fetch_brand_logo_url(request.brand_url)
    print(f"[pipeline] logo_url={logo_url!r}")
    logo_bytes: bytes | None = None
    if logo_url:
        logo_bytes = _download_image_bytes(logo_url)
        print(f"[pipeline] logo_bytes len={len(logo_bytes) if logo_bytes else 0}")
        brand_logo_b64 = base64.b64encode(logo_bytes).decode() if logo_bytes else None
        if brand_logo_b64:
            st.write("OK Brand logo found")
        else:
            st.write("WARN Brand logo URL found but could not download")
            brand_logo_b64 = None
    else:
        print("[pipeline] logo_bytes len=0 (no logo_url)")
        st.write("WARN No brand logo found — using initials")
        brand_logo_b64 = None

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
        platform=request.platform,
        campaign_goal=request.campaign_goal,
        offer=request.offer,
        landing_page_url=request.landing_page_url,
        brand_name=brand_name,
    )
    st.write("OK Ad copy written")

    st.write("// Generating image briefs (single request)...")
    briefs = _generate_all_image_briefs(client, product_intel, variations, brand_name)
    st.session_state.cached_image_briefs = briefs
    st.write("OK Image briefs ready")

    st.write("// Generating ad creatives with Nano Banana...")
    images_ok = _run_nb_image_loop(client, variations, briefs)

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
        product_image_url=product_image_url,
        product_image_b64=product_image_b64,
        brand_logo_b64=brand_logo_b64,
    )
