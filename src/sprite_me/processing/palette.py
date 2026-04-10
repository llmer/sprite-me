"""Palette reduction and cleanup for pixel art sprites."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


def reduce_palette(
    image_data: bytes,
    max_colors: int = 16,
    dither: bool = False,
) -> bytes:
    """Reduce the color palette of a sprite image.

    Args:
        image_data: Raw PNG bytes.
        max_colors: Maximum number of colors in the output palette.
        dither: Whether to apply dithering during quantization.

    Returns:
        PNG bytes with reduced palette.
    """
    img = Image.open(BytesIO(image_data)).convert("RGBA")

    # Separate alpha channel
    alpha = img.getchannel("A")

    # Quantize RGB
    rgb = img.convert("RGB")
    dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    quantized = rgb.quantize(colors=max_colors, dither=dither_mode)
    rgb_result = quantized.convert("RGB")

    # Recombine with alpha
    result = Image.merge("RGBA", (*rgb_result.split(), alpha))

    buf = BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def snap_to_grid(
    image_data: bytes,
    grid_size: int = 1,
) -> bytes:
    """Snap pixel art to a grid by downscaling then upscaling with nearest neighbor.

    Useful for cleaning up AI-generated pixel art that has sub-pixel anti-aliasing.

    Args:
        image_data: Raw PNG bytes.
        grid_size: The pixel grid size to snap to. E.g., 2 means each "pixel"
                   in the output represents a 2x2 block.

    Returns:
        PNG bytes with grid-snapped pixels.
    """
    if grid_size <= 1:
        return image_data

    img = Image.open(BytesIO(image_data)).convert("RGBA")
    w, h = img.size

    # Downscale
    small = img.resize(
        (w // grid_size, h // grid_size), Image.NEAREST
    )
    # Upscale back
    result = small.resize((w, h), Image.NEAREST)

    buf = BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()
