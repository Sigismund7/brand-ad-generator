"""Normalize and validate scraped product images before composition."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps

MIN_EDGE_PX = 200
MAX_PREPARE_DIM = 1080


@dataclass
class PreparedAsset:
    image: Image.Image
    original_size: tuple[int, int]
    has_transparency: bool


def detect_white_background(image: Image.Image, threshold: float = 0.85) -> bool:
    """True if corner pixels are predominantly light (e-commerce white backdrops)."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w < 2 or h < 2:
        return False
    corners = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((w - 1, 0)),
        rgb.getpixel((0, h - 1)),
        rgb.getpixel((w - 1, h - 1)),
    ]
    bright = 0
    for r, g, b in corners:
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        if lum >= threshold:
            bright += 1
    return bright >= 3


def prepare(image_bytes: bytes, remove_bg: bool = False) -> PreparedAsset:
    """
    Decode, validate, normalize to RGBA, resize to fit within MAX_PREPARE_DIM.
    remove_bg is reserved for Phase 3; ignored here.
    """
    _ = remove_bg
    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()
    except OSError as exc:
        raise ValueError(f"Invalid image bytes: {exc}") from exc

    original_size = img.size
    w, h = original_size
    if min(w, h) < MIN_EDGE_PX:
        raise ValueError(f"Product image too small: {w}x{h} (min edge {MIN_EDGE_PX}px)")

    img = ImageOps.exif_transpose(img)
    rgba = img.convert("RGBA")
    alpha = rgba.split()[3]
    has_transparency = alpha.getextrema() != (255, 255)

    rw, rh = rgba.size
    longest = max(rw, rh)
    if longest > MAX_PREPARE_DIM:
        scale = MAX_PREPARE_DIM / longest
        new_w = max(1, int(rw * scale))
        new_h = max(1, int(rh * scale))
        rgba = rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return PreparedAsset(
        image=rgba,
        original_size=original_size,
        has_transparency=has_transparency,
    )
