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


def pixelate(
    image_data: bytes,
    target_size: int = 64,
    palette_size: int = 16,
    upscale: bool = True,
    dither: bool = False,
) -> bytes:
    """Convert a smooth/high-res image into a sharp pixel-art sprite.

    Pipeline:
        1. Downsample to target_size x target_size using a high-quality filter
           (LANCZOS) so the smooth input collapses into distinct pixel blocks
        2. Quantize to palette_size colors (preserving alpha)
        3. Upsample back to the original resolution using NEAREST (no blur)
           so each tiny pixel becomes a crisp block — ready for game engines
           that expect rectangular pixel grids

    This is the fix for FLUX Kontext outputs looking "painterly" instead of
    "retro pixel art". Kontext renders at 1024x1024 smooth-shaded, which
    preserves character identity beautifully but isn't pixel-perfect. Run
    this over each frame and you get a proper pixel sprite that matches
    the palette aesthetic of classic game art.

    Args:
        image_data: PNG bytes of the input image (any size).
        target_size: The "pixel resolution" of the output. 64 gives a
            classic 64x64 sprite; 32 is Game Boy sized; 16 is tile-sized.
            Width and height are both set to this value; aspect ratio is
            preserved by padding the smaller side's crop (we use a square
            bounding box).
        palette_size: Maximum number of colors in the output. Classic game
            consoles used 16 or fewer. 32 gives a richer modern palette.
        upscale: If True (default), nearest-neighbor upsample the pixelated
            image back to the original dimensions. Useful when consumers
            expect the output to match input dimensions. If False, return
            the raw target_size x target_size image.
        dither: If True, apply Floyd-Steinberg dithering during quantization
            for a more retro/grainy look at low palette sizes. Default False.

    Returns:
        PNG bytes of the pixelated image.
    """
    img = Image.open(BytesIO(image_data)).convert("RGBA")
    orig_w, orig_h = img.size

    # Preserve aspect ratio: pick the larger side as the square canvas so
    # the content stays centered. For square inputs (common for sprites)
    # this is a no-op.
    if orig_w != orig_h:
        square_side = max(orig_w, orig_h)
        square = Image.new("RGBA", (square_side, square_side), (0, 0, 0, 0))
        off_x = (square_side - orig_w) // 2
        off_y = (square_side - orig_h) // 2
        square.paste(img, (off_x, off_y))
        img = square

    # Step 1: high-quality downsample to the target pixel resolution
    small = img.resize((target_size, target_size), Image.LANCZOS)

    # Step 2: quantize colors (preserves alpha)
    alpha = small.getchannel("A")
    rgb = small.convert("RGB")
    dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    quantized = rgb.quantize(colors=palette_size, dither=dither_mode).convert("RGB")
    pixelated = Image.merge("RGBA", (*quantized.split(), alpha))

    # Step 3: nearest-neighbor upscale back (unless the caller wants raw size)
    if upscale:
        # Scale up to the original longest side so tiny pixels become
        # crisp blocks at the caller's expected resolution
        target_upscale = max(orig_w, orig_h)
        pixelated = pixelated.resize(
            (target_upscale, target_upscale), Image.NEAREST
        )

    buf = BytesIO()
    pixelated.save(buf, format="PNG")
    return buf.getvalue()
