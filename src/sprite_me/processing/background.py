"""Background removal for sprites."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


def remove_background(image_data: bytes) -> bytes:
    """Remove background from a sprite image using rembg.

    Falls back to simple white-background removal if rembg is unavailable.
    """
    try:
        from rembg import remove
        output = remove(image_data)
        return output
    except ImportError:
        return _simple_bg_remove(image_data)


def _simple_bg_remove(
    image_data: bytes, threshold: int = 240
) -> bytes:
    """Fallback: remove near-white background pixels by making them transparent."""
    img = Image.open(BytesIO(image_data)).convert("RGBA")
    pixels = img.load()
    w, h = img.size

    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            if r > threshold and g > threshold and b > threshold:
                pixels[x, y] = (r, g, b, 0)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
