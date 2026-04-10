"""Smart crop for sprite images — detect bounds and trim whitespace."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


def smart_crop(
    image_data: bytes,
    mode: str = "tightest",
    margin: int = 2,
    bg_threshold: int = 240,
) -> bytes:
    """Crop a sprite image to its content bounds.

    Args:
        image_data: Raw PNG bytes.
        mode: "tightest" crops to exact bounds + margin,
              "padded" adds extra padding for animation safety.
        margin: Pixels of padding around the detected content.
        bg_threshold: Luminance above which a pixel is considered background.

    Returns:
        Cropped PNG bytes.
    """
    img = Image.open(BytesIO(image_data)).convert("RGBA")

    # Find bounding box of non-background pixels
    bbox = _find_content_bbox(img, bg_threshold)
    if bbox is None:
        return image_data  # nothing to crop

    x1, y1, x2, y2 = bbox

    if mode == "padded":
        margin *= 3

    # Apply margin
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img.width, x2 + margin)
    y2 = min(img.height, y2 + margin)

    cropped = img.crop((x1, y1, x2, y2))

    buf = BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def _find_content_bbox(
    img: Image.Image, bg_threshold: int
) -> tuple[int, int, int, int] | None:
    """Find the bounding box of non-background content."""
    pixels = img.load()
    w, h = img.size

    min_x, min_y = w, h
    max_x, max_y = 0, 0
    found = False

    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            # Skip transparent or near-white pixels
            if a < 10:
                continue
            if r > bg_threshold and g > bg_threshold and b > bg_threshold:
                continue
            found = True
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if not found:
        return None
    return (min_x, min_y, max_x + 1, max_y + 1)
