"""
Deterministic PIL compositor: creative_spec (JSON) + product bytes -> base64 JPEG.

Merges LLM partial specs with defaults from copy fields; no network calls.
"""

from __future__ import annotations

import base64
import copy
import logging
import math
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from asset_pipeline import prepare

log = logging.getLogger(__name__)

OPSZ_DEFAULT = 14

# ---------------------------------------------------------------------------
# Color & style helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = (s or "").strip()
    if not s.startswith("#"):
        return (40, 40, 50)
    s = s[1:]
    if len(s) == 3:
        return tuple(int(s[i] + s[i], 16) for i in range(3))
    if len(s) >= 6:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return (40, 40, 50)


def _parse_color(s: str | None) -> tuple[int, int, int, int]:
    """Returns RGBA; alpha 255 if omitted."""
    if not s or not isinstance(s, str):
        return (255, 255, 255, 255)
    s = s.strip()
    if s.startswith("#"):
        r, g, b = _hex_to_rgb(s)
        return (r, g, b, 255)
    m = re.match(
        r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)",
        s,
        re.I,
    )
    if m:
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        a = float(m.group(4)) if m.group(4) is not None else 1.0
        return (r, g, b, min(255, int(a * 255)) if a <= 1 else int(a))
    return (255, 255, 255, 255)


def _rgb_tuple_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _dominant_bg_color(rgb_image: Image.Image) -> tuple[int, int, int]:
    small = rgb_image.resize((48, 48), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.float64)
    flat = arr.reshape(-1, 3)
    lum = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
    mask = lum < 245
    if mask.sum() < 80:
        mask = np.ones(len(flat), dtype=bool)
    pixels = flat[mask]
    r, g, b = pixels.mean(axis=0)
    L = 0.299 * r + 0.587 * g + 0.114 * b
    factor = 0.22 if L > 170 else (0.55 if L < 45 else 0.38)
    return (
        max(0, min(255, int(r * factor))),
        max(0, min(255, int(g * factor))),
        max(0, min(255, int(b * factor))),
    )


def _accent_for_angle(angle: str) -> tuple[int, int, int]:
    if angle == "Aspiration":
        return (214, 165, 116)
    if angle == "Social Proof":
        return (42, 157, 143)
    return (230, 57, 70)


def _default_background(angle: str) -> dict[str, Any]:
    if angle == "Social Proof":
        return {"mode": "solid", "value": {"color": "#F5F5F0"}}
    if angle == "Aspiration":
        return {
            "mode": "gradient",
            "value": {
                "type": "linear",
                "angle_deg": 145,
                "stops": [
                    {"position": 0.0, "color": "#FFF8F0"},
                    {"position": 1.0, "color": "#E8DCC8"},
                ],
            },
        }
    return {
        "mode": "gradient",
        "value": {
            "type": "linear",
            "angle_deg": 160,
            "stops": [
                {"position": 0.0, "color": "#1a1a2e"},
                {"position": 1.0, "color": "#16213e"},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Creative spec merge (LLM partial -> complete spec)
# ---------------------------------------------------------------------------


def _clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def _deep_merge_dict(base: dict[str, Any], over: dict[str, Any]) -> None:
    for k, v in over.items():
        if k == "text_zones" and isinstance(v, list):
            base[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge_dict(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def _default_text_zones_from_item(item: dict[str, Any], accent_hex: str) -> list[dict[str, Any]]:
    hl = (item.get("creative_headline") or item.get("headline") or "").strip()
    st = (item.get("creative_subtext") or "").strip()
    cta = (item.get("creative_cta") or item.get("cta") or "Shop Now").strip()
    trust = (item.get("trust_element") or "").strip()
    zones: list[dict[str, Any]] = [
        {
            "id": "headline",
            "content": hl,
            "font_weight": "bold",
            "font_size_px": 52,
            "color": "#FFFFFF",
            "anchor": "top-center",
            "y_offset_pct": 6,
            "max_width_pct": 85,
            "alignment": "center",
        },
    ]
    if st:
        zones.append(
            {
                "id": "subtext",
                "content": st,
                "font_weight": "normal",
                "font_size_px": 24,
                "color": "rgba(255,255,255,0.75)",
                "anchor": "below-headline",
                "y_offset_pct": 2,
                "max_width_pct": 75,
                "alignment": "center",
            }
        )
    zones.append(
        {
            "id": "cta",
            "content": cta,
            "font_weight": "bold",
            "font_size_px": 22,
            "color": "#FFFFFF",
            "background_color": accent_hex,
            "corner_radius": 8,
            "padding": [12, 32, 12, 32],
            "anchor": "bottom-center",
            "y_offset_pct": 8,
        }
    )
    if trust:
        zones.append(
            {
                "id": "trust",
                "content": trust,
                "font_weight": "normal",
                "font_size_px": 18,
                "color": "rgba(255,255,255,0.6)",
                "anchor": "above-cta",
                "y_offset_pct": 2,
                "max_width_pct": 80,
                "alignment": "center",
            }
        )
    return zones[:4]


def _spec_from_item(item: dict[str, Any], brand_name: str) -> dict[str, Any]:
    angle = str(item.get("angle") or "Pain Point")
    accent = _accent_for_angle(angle)
    accent_hex = _rgb_tuple_to_hex(accent)
    fb = ((brand_name or "Brand").strip()[:40] or "Brand").upper()
    dom = _default_background(angle)
    dom_color = "#1a1a2e"
    if dom["mode"] == "solid":
        dom_color = dom["value"].get("color", dom_color)
    return {
        "canvas": {"width": 1080, "height": 1350, "aspect_ratio": "4:5"},
        "background": dom,
        "product_zone": {
            "anchor": "center",
            "y_offset_pct": 5,
            "max_width_pct": 65,
            "max_height_pct": 50,
            "crop_mode": "contain",
            "shadow": {
                "enabled": True,
                "offset_y": 12,
                "blur": 26,
                "color": "rgba(0,0,0,0.35)",
            },
        },
        "text_zones": _default_text_zones_from_item(item, accent_hex),
        "brand_badge": {
            "position": "top-left",
            "margin_px": 24,
            "use_logo": True,
            "fallback_text": fb,
            "max_height_px": 36,
        },
        "style": {
            "font_family": "sans-serif-bold",
            "mood": "",
            "dominant_color": dom_color,
            "accent_color": accent_hex,
        },
    }


def _sanitize_zone(z: dict[str, Any], default_color: str) -> None:
    fs = int(_clamp(float(z.get("font_size_px", 24)), 16, 64))
    z["font_size_px"] = fs
    if not z.get("color"):
        z["color"] = default_color
    if z.get("max_width_pct") is None:
        z["max_width_pct"] = 85
    z["max_width_pct"] = int(_clamp(float(z["max_width_pct"]), 20, 95))
    if z.get("y_offset_pct") is None:
        z["y_offset_pct"] = 0
    z["y_offset_pct"] = float(_clamp(float(z["y_offset_pct"]), 0, 25))


def merge_creative_spec_defaults(
    raw: dict[str, Any] | None,
    brand_name: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    """Start from copy-driven defaults; overlay any LLM `creative_spec` keys."""
    base = _spec_from_item(item, brand_name)
    if isinstance(raw, dict) and raw:
        _deep_merge_dict(base, raw)
    if not base.get("text_zones"):
        ang = str(item.get("angle") or "Pain Point")
        base["text_zones"] = _default_text_zones_from_item(
            item, _rgb_tuple_to_hex(_accent_for_angle(ang))
        )
    # Fill empty text from item if zone content missing
    by_id = {str(z.get("id")): z for z in base.get("text_zones") or []}
    if not (by_id.get("headline") or {}).get("content"):
        if by_id.get("headline") is not None:
            by_id["headline"]["content"] = (
                item.get("creative_headline") or item.get("headline") or ""
            ).strip()
    for z in base.get("text_zones") or []:
        _sanitize_zone(
            z,
            "#FFFFFF" if base.get("background", {}).get("mode") != "solid" else "#111111",
        )
    cw = int(base.get("canvas", {}).get("width") or 1080)
    ch = int(base.get("canvas", {}).get("height") or 1350)
    base["canvas"] = {"width": cw, "height": ch, "aspect_ratio": base.get("canvas", {}).get("aspect_ratio") or "4:5"}
    pz = base.get("product_zone") or {}
    pz.setdefault("anchor", "center")
    pz.setdefault("y_offset_pct", 5)
    pz.setdefault("max_width_pct", 65)
    pz.setdefault("max_height_pct", 50)
    pz.setdefault("crop_mode", "contain")
    pz["max_width_pct"] = int(_clamp(float(pz["max_width_pct"]), 40, 85))
    pz["max_height_pct"] = int(_clamp(float(pz["max_height_pct"]), 35, 60))
    base["product_zone"] = pz
    bb = base.get("brand_badge") or {}
    bb.setdefault("position", "top-left")
    bb.setdefault("margin_px", 24)
    bb.setdefault("use_logo", True)
    bb.setdefault("fallback_text", ((brand_name or "Brand").strip()[:32] or "BRAND").upper())
    bb.setdefault("max_height_px", 36)
    base["brand_badge"] = bb
    base["text_zones"] = (base.get("text_zones") or [])[:4]
    fb_final = ((brand_name or bb.get("fallback_text") or "Brand").strip()[:32] or "BRAND").upper()
    base["brand_badge"]["fallback_text"] = fb_final
    return base


# ---------------------------------------------------------------------------
# Fonts & text
# ---------------------------------------------------------------------------


def _project_root_fonts_dir(fonts_dir: str) -> Path:
    root = Path(__file__).resolve().parent
    p = Path(fonts_dir)
    return p if p.is_absolute() else (root / p)


def _wght_for_zone(z: dict[str, Any]) -> int:
    fw = str(z.get("font_weight", "normal")).lower()
    if fw == "bold":
        return 700
    if fw == "medium":
        return 500
    return 400


def _load_inter(fonts_dir: Path, size: int, wght: int) -> ImageFont.FreeTypeFont:
    var_path = fonts_dir / "InterVariable.ttf"
    for path in (var_path, fonts_dir / "Inter-Bold.ttf", fonts_dir / "Inter-Regular.ttf"):
        if path.is_file():
            try:
                font = ImageFont.truetype(str(path), size)
                if "Variable" in path.name:
                    font.set_variation_by_axes([OPSZ_DEFAULT, wght])
                return font
            except OSError:
                continue
    log.warning("No Inter font found under %s; using bitmap font", fonts_dir)
    return ImageFont.load_default()


def _wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        trial = (" ".join(current + [w])).strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current.append(w)
        else:
            lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


def _block_size(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    font: ImageFont.FreeTypeFont,
    line_gap: int,
) -> tuple[int, int]:
    if not lines:
        return 0, 0
    max_w = 0
    h = 0
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        max_w = max(max_w, bbox[2] - bbox[0])
        h += bbox[3] - bbox[1]
        if i < len(lines) - 1:
            h += line_gap
    return max_w, h


# ---------------------------------------------------------------------------
# Gradients & background
# ---------------------------------------------------------------------------


def _interpolate_stops(t: np.ndarray, stops: list[dict[str, Any]]) -> np.ndarray:
    stops = sorted(stops, key=lambda s: float(s.get("position", 0)))
    positions = [float(s["position"]) for s in stops]
    colors = [np.array(_parse_color(s.get("color", "#000"))[:3], dtype=np.float64) for s in stops]
    out = np.zeros(t.shape + (3,), dtype=np.float64)
    for c in range(3):
        channel = np.interp(t.flatten(), positions, [col[c] for col in colors])
        out[..., c] = channel.reshape(t.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _linear_gradient_rgba(size: tuple[int, int], spec: dict[str, Any]) -> Image.Image:
    w, h = size
    angle = float(spec.get("angle_deg", 160))
    stops = spec.get("stops") or [{"position": 0.0, "color": "#000"}, {"position": 1.0, "color": "#fff"}]
    rad = math.radians(angle)
    ux, uy = math.cos(rad), math.sin(rad)
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    proj = (X - cx) * ux + (Y - cy) * uy
    t = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
    rgb = _interpolate_stops(t, stops)
    return Image.fromarray(rgb, mode="RGB").convert("RGBA")


def _radial_gradient_rgba(size: tuple[int, int], spec: dict[str, Any]) -> Image.Image:
    w, h = size
    stops = spec.get("stops") or [{"position": 0.0, "color": "#222"}, {"position": 1.0, "color": "#000"}]
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = math.hypot(cx, cy) + 1e-9
    t = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / r
    t = np.clip(t, 0, 1)
    rgb = _interpolate_stops(t, stops)
    return Image.fromarray(rgb, mode="RGB").convert("RGBA")


def _render_background_layer(
    size: tuple[int, int],
    bg: dict[str, Any],
    product_bytes: bytes,
) -> Image.Image:
    w, h = size
    mode = str(bg.get("mode", "solid"))
    val = bg.get("value") or {}
    if mode == "solid":
        c = _parse_color(val.get("color", "#2a2a3a"))
        return Image.new("RGBA", (w, h), c[:3] + (255,))
    if mode == "gradient":
        gtype = str(val.get("type", "linear"))
        if gtype == "radial":
            return _radial_gradient_rgba((w, h), val)
        return _linear_gradient_rgba((w, h), val)
    if mode == "blur":
        try:
            asset = prepare(product_bytes)
            img = asset.image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
            r = int(_clamp(float(val.get("radius", 40)), 8, 80))
            img = img.filter(ImageFilter.GaussianBlur(r))
            base = img.convert("RGBA")
            tint = _parse_color(val.get("tint", "rgba(0,0,0,0.3)"))
            overlay = Image.new("RGBA", (w, h), tint)
            return Image.alpha_composite(base, overlay)
        except (ValueError, OSError) as exc:
            log.warning("Blur background failed, using solid: %s", exc)
            return Image.new("RGBA", (w, h), (42, 42, 58, 255))
    if mode == "ai_generated":
        log.warning("ai_generated background not implemented (Phase 4); using solid fill")
        return Image.new("RGBA", (w, h), (35, 35, 50, 255))
    return Image.new("RGBA", (w, h), (42, 42, 58, 255))


# ---------------------------------------------------------------------------
# Product placement
# ---------------------------------------------------------------------------


def _fit_product(
    prod: Image.Image,
    zone_w: int,
    zone_h: int,
    crop_mode: str,
) -> Image.Image:
    pw, ph = prod.size
    mode = (crop_mode or "contain").lower()
    if mode == "cover":
        scale = max(zone_w / pw, zone_h / ph)
        nw, nh = max(1, int(pw * scale)), max(1, int(ph * scale))
        resized = prod.resize((nw, nh), Image.Resampling.LANCZOS)
        l = (nw - zone_w) // 2
        t = (nh - zone_h) // 2
        return resized.crop((l, t, l + zone_w, t + zone_h))
    if mode == "original":
        scale = min(zone_w / pw, zone_h / ph, 1.0)
        nw, nh = max(1, int(pw * scale)), max(1, int(ph * scale))
        return prod.resize((nw, nh), Image.Resampling.LANCZOS)
    # contain
    scale = min(zone_w / pw, zone_h / ph)
    nw, nh = max(1, int(pw * scale)), max(1, int(ph * scale))
    resized = prod.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (zone_w, zone_h), (0, 0, 0, 0))
    ox = (zone_w - nw) // 2
    oy = (zone_h - nh) // 2
    canvas.alpha_composite(resized, (ox, oy))
    return canvas


def _shadow_layer(
    product_rgba: Image.Image,
    shadow_spec: dict[str, Any] | None,
) -> Image.Image:
    if not shadow_spec or not shadow_spec.get("enabled", True):
        return product_rgba
    off = int(shadow_spec.get("offset_y", 12))
    blur = int(_clamp(float(shadow_spec.get("blur", 24)), 4, 48))
    rgba = _parse_color(shadow_spec.get("color", "rgba(0,0,0,0.35)"))
    strength = rgba[3] / 255.0
    w, h = product_rgba.size
    pad = blur * 2 + abs(off) + 4
    out = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    alpha = product_rgba.getchannel("A")
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base_alpha = int(160 * strength)
    shadow.paste((0, 0, 0, base_alpha), mask=alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    sx = pad
    sy = pad + off
    out.alpha_composite(shadow, (sx, sy))
    out.alpha_composite(product_rgba, (pad, pad))
    return out


def _product_bbox_on_canvas(
    cw: int,
    ch: int,
    pz: dict[str, Any],
    layer_w: int,
    layer_h: int,
) -> tuple[int, int, int, int]:
    zone_w = int(cw * float(pz.get("max_width_pct", 65)) / 100)
    zone_h = int(ch * float(pz.get("max_height_pct", 50)) / 100)
    cx = cw // 2
    cy = ch // 2 + int(ch * float(pz.get("y_offset_pct", 0)) / 100)
    px = cx - layer_w // 2
    py = cy - layer_h // 2
    return px, py, px + layer_w, py + layer_h


# ---------------------------------------------------------------------------
# Text zones: measure, place, collision
# ---------------------------------------------------------------------------


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 4,
) -> bool:
    ax0, ay0, ax1, ay1 = a[0] - pad, a[1] - pad, a[2] + pad, a[3] + pad
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _measure_text_zone(
    z: dict[str, Any],
    draw_tmp: ImageDraw.ImageDraw,
    fonts_dir: Path,
    cw: int,
    ch: int,
    margin: int,
    headline_bottom: float,
    subtext_bottom: float,
    cta_rect: tuple[int, int, int, int] | None,
) -> dict[str, Any] | None:
    zid = str(z.get("id", ""))
    wght = _wght_for_zone(z)
    fs = int(z.get("font_size_px", 24))
    font = _load_inter(fonts_dir, fs, wght)
    max_w = int(cw * float(z.get("max_width_pct", 85)) / 100)
    content = str(z.get("content", ""))
    inner_pad = 56 if z.get("background_color") else 0
    lines = _wrap_lines(draw_tmp, content, font, max(80, max_w - inner_pad))
    if not lines:
        if zid == "cta" and not content.strip():
            lines = ["Shop Now"]
        elif zid != "cta":
            return None
    line_gap = 6
    tw, th = _block_size(draw_tmp, lines, font, line_gap)
    anchor = str(z.get("anchor", "top-center"))
    y_off = int(ch * float(z.get("y_offset_pct", 0)) / 100)
    pad_spec = z.get("padding") or [12, 28, 12, 28]
    if isinstance(pad_spec, list) and len(pad_spec) >= 4:
        pt, pr, pb, pl = int(pad_spec[0]), int(pad_spec[1]), int(pad_spec[2]), int(pad_spec[3])
    else:
        pt = pr = pb = pl = 12
    has_bg = bool(z.get("background_color"))
    if has_bg:
        box_w = tw + pl + pr
        box_h = th + pt + pb
    else:
        box_w, box_h = tw, th
    x0 = y0 = 0
    if anchor == "top-center":
        x0 = (cw - box_w) // 2
        y0 = margin + y_off
    elif anchor == "top-left":
        x0 = margin
        y0 = margin + y_off
    elif anchor == "top-right":
        x0 = cw - margin - box_w
        y0 = margin + y_off
    elif anchor == "bottom-center":
        x0 = (cw - box_w) // 2
        y0 = ch - margin - box_h - y_off
    elif anchor == "bottom-left":
        x0 = margin
        y0 = ch - margin - box_h - y_off
    elif anchor == "bottom-right":
        x0 = cw - margin - box_w
        y0 = ch - margin - box_h - y_off
    elif anchor == "below-headline":
        x0 = (cw - box_w) // 2
        y0 = int(headline_bottom) + y_off
    elif anchor == "below-subtext":
        x0 = (cw - box_w) // 2
        y0 = int(subtext_bottom) + y_off
    elif anchor == "above-cta" and cta_rect:
        x0 = (cw - box_w) // 2
        y0 = cta_rect[1] - box_h - 12 - y_off
    else:
        x0 = (cw - box_w) // 2
        y0 = margin + y_off

    fill = _parse_color(z.get("color", "#FFFFFF"))
    stroke_fill: tuple[int, int, int] | None = None
    if _luminance(fill[:3]) > 200:
        stroke_fill = (20, 20, 30)
    elif _luminance(fill[:3]) < 45:
        stroke_fill = (255, 255, 255)

    return {
        "zone": z,
        "zid": zid,
        "lines": lines,
        "font": font,
        "line_gap": line_gap,
        "x0": x0,
        "y0": y0,
        "box_w": box_w,
        "box_h": box_h,
        "tw": tw,
        "th": th,
        "pl": pl,
        "pt": pt,
        "fill": fill,
        "stroke_fill": stroke_fill,
        "has_bg": has_bg,
        "corner_radius": int(z.get("corner_radius", 8)),
        "bg_color": z.get("background_color"),
    }


def _resolve_text_layers(
    cw: int,
    ch: int,
    zones: list[dict[str, Any]],
    fonts_dir: Path,
    product_rect: tuple[int, int, int, int],
    margin: int,
) -> list[dict[str, Any]]:
    draw_tmp = ImageDraw.Draw(Image.new("RGBA", (cw, ch)))
    by_id: dict[str, dict[str, Any]] = {str(z.get("id")): z for z in zones}
    ordered_keys = ("headline", "subtext", "cta", "trust")
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for k in ordered_keys:
        if k in by_id:
            ordered.append(by_id[k])
            seen.add(k)
    for z in zones:
        zk = str(z.get("id"))
        if zk not in seen:
            ordered.append(z)
            seen.add(zk)

    headline_bottom = float(margin + 4)
    subtext_bottom = headline_bottom
    cta_rect: tuple[int, int, int, int] | None = None
    layers: list[dict[str, Any]] = []

    for z in ordered:
        info = _measure_text_zone(
            z,
            draw_tmp,
            fonts_dir,
            cw,
            ch,
            margin,
            headline_bottom,
            subtext_bottom,
            cta_rect,
        )
        if not info:
            continue
        zid = str(z.get("id"))
        if zid == "headline":
            headline_bottom = float(info["y0"] + info["box_h"])
        elif zid == "subtext":
            subtext_bottom = float(info["y0"] + info["box_h"])
        elif zid == "cta":
            cta_rect = (
                info["x0"],
                info["y0"],
                info["x0"] + info["box_w"],
                info["y0"] + info["box_h"],
            )
        layers.append(info)

    draw_order = {"headline": 0, "subtext": 1, "trust": 2, "cta": 3}
    layers.sort(key=lambda L: draw_order.get(L["zid"], 5))

    for L in layers:
        box = (L["x0"], L["y0"], L["x0"] + L["box_w"], L["y0"] + L["box_h"])
        if not _boxes_overlap(box, product_rect):
            continue
        dy = 0
        for _ in range(28):
            if not _boxes_overlap(
                (L["x0"], L["y0"] + dy, L["x0"] + L["box_w"], L["y0"] + L["box_h"] + dy),
                product_rect,
            ):
                break
            if L["y0"] < ch // 2:
                dy -= 12
            else:
                dy += 12
        else:
            log.warning("Text zone '%s' may overlap product after nudge", L["zid"])
        L["y0"] += dy

    return layers


def _draw_text_layers(canvas: Image.Image, layers: list[dict[str, Any]]) -> None:
    draw = ImageDraw.Draw(canvas)
    for L in layers:
        z = L["zone"]
        x0, y0 = L["x0"], L["y0"]
        font = L["font"]
        lines: list[str] = L["lines"]
        line_gap = L["line_gap"]
        fill = L["fill"]
        stroke = L["stroke_fill"]
        sw = 2 if stroke is not None else 0

        if L["has_bg"] and L.get("bg_color"):
            rr = L["corner_radius"]
            c = _parse_color(str(L["bg_color"]))[:3] + (255,)
            draw.rounded_rectangle(
                [x0, y0, x0 + L["box_w"], y0 + L["box_h"]],
                radius=rr,
                fill=c,
            )
            tx = x0 + L["pl"]
            ty = y0 + L["pt"]
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                kw: dict[str, Any] = {
                    "fill": fill[:3] + (fill[3],),
                    "font": font,
                }
                if sw and stroke:
                    kw["stroke_width"] = sw
                    kw["stroke_fill"] = stroke + (255,)
                draw.text((tx, ty), line, **kw)
                ty += bbox[3] - bbox[1] + line_gap
        else:
            align = str(z.get("alignment", "center"))
            tw, th = L["tw"], L["th"]
            ty = y0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                lw = bbox[2] - bbox[0]
                if align == "left":
                    tx = x0
                elif align == "right":
                    tx = x0 + tw - lw
                else:
                    tx = x0 + (tw - lw) // 2
                kw2: dict[str, Any] = {
                    "fill": fill[:3] + (fill[3],),
                    "font": font,
                }
                if sw and stroke:
                    kw2["stroke_width"] = sw
                    kw2["stroke_fill"] = stroke + (255,)
                draw.text((tx, ty), line, **kw2)
                ty += bbox[3] - bbox[1] + line_gap


def _paste_brand_badge(
    canvas: Image.Image,
    badge: dict[str, Any],
    logo_bytes: bytes | None,
    fonts_dir: Path,
) -> None:
    pos = str(badge.get("position", "top-left"))
    m = int(badge.get("margin_px", 24))
    max_h = int(badge.get("max_height_px", 36))
    use_logo = bool(badge.get("use_logo", True)) and bool(logo_bytes)
    x, y = m, m
    if use_logo:
        try:
            logo = Image.open(BytesIO(logo_bytes)).convert("RGBA")
        except OSError:
            use_logo = False
        else:
            lw, lh = logo.size
            if lh > 0:
                sc = max_h / lh
                nw, nh = max(1, int(lw * sc)), max(1, int(lh * sc))
                logo = logo.resize((nw, nh), Image.Resampling.LANCZOS)
                if pos == "top-right":
                    x = canvas.size[0] - m - nw
                canvas.paste(logo, (x, y), logo)
                return
    fb = str(badge.get("fallback_text", "BRAND"))[:32]
    if not fb:
        return
    font = _load_inter(fonts_dir, 20, 600)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), fb.upper(), font=font)
    tw = bbox[2] - bbox[0]
    if pos == "top-right":
        x = canvas.size[0] - m - tw
    draw.text((x, y), fb.upper(), fill=(255, 255, 255, 230), font=font)


# ---------------------------------------------------------------------------
# Phase-1 fallback composer (legacy dict)
# ---------------------------------------------------------------------------


def _compose_phase1_layout(
    spec: dict[str, Any],
    product_image: bytes,
    logo_image: bytes | None,
    fonts_dir: Path,
) -> str:
    """Minimal layout when spec uses _layout phase1 keys only."""
    item = {
        "creative_headline": spec.get("creative_headline"),
        "creative_subtext": spec.get("creative_subtext"),
        "creative_cta": spec.get("creative_cta"),
        "trust_element": spec.get("trust_element"),
        "angle": spec.get("angle"),
        "headline": spec.get("headline"),
        "cta": spec.get("cta"),
    }
    brand = spec.get("brand_fallback") or "BRAND"
    merged = merge_creative_spec_defaults({}, brand, item)
    return _compose_full_spec(merged, product_image, logo_image, fonts_dir)


def _compose_full_spec(
    spec: dict[str, Any],
    product_image: bytes,
    logo_image: bytes | None,
    fonts_dir: Path,
) -> str:
    canvas_cfg = spec["canvas"]
    cw = int(canvas_cfg["width"])
    ch = int(canvas_cfg["height"])
    asset = prepare(product_image)
    prod = asset.image

    bg_layer = _render_background_layer((cw, ch), spec["background"], product_image)
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    canvas.paste(bg_layer, (0, 0))

    pz = spec["product_zone"]
    zone_w = int(cw * float(pz["max_width_pct"]) / 100)
    zone_h = int(ch * float(pz["max_height_pct"]) / 100)
    fitted = _fit_product(prod, zone_w, zone_h, str(pz.get("crop_mode", "contain")))
    layered = _shadow_layer(fitted, pz.get("shadow"))
    lw, lh = layered.size
    cx = cw // 2
    cy = ch // 2 + int(ch * float(pz.get("y_offset_pct", 0)) / 100)
    px = int(cx - lw / 2)
    py = int(cy - lh / 2)
    canvas.alpha_composite(layered, (px, py))
    prod_rect = (px, py, px + lw, py + lh)

    margin = int(spec.get("brand_badge", {}).get("margin_px", 24)) + 28
    zones = spec.get("text_zones") or []
    layers = _resolve_text_layers(cw, ch, zones, fonts_dir, prod_rect, margin)
    _draw_text_layers(canvas, layers)

    badge = spec.get("brand_badge") or {}
    want_logo = badge.get("use_logo", True)
    _paste_brand_badge(
        canvas,
        badge,
        logo_image if want_logo else None,
        fonts_dir,
    )

    buf = BytesIO()
    flat = Image.new("RGB", (cw, ch), (255, 255, 255))
    flat.paste(canvas, mask=canvas.split()[3])
    flat.save(buf, format="JPEG", quality=92, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compose(
    spec: dict[str, Any],
    product_image: bytes,
    logo_image: bytes | None = None,
    fonts_dir: str = "assets/fonts",
) -> str:
    """Returns base64-encoded JPEG of the composed ad creative."""
    fd = _project_root_fonts_dir(fonts_dir)
    s = spec or {}
    if s.get("_layout") == "phase1":
        return _compose_phase1_layout(s, product_image, logo_image, fd)
    return _compose_full_spec(s, product_image, logo_image, fd)
