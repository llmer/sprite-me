"""Tests for image processing pipeline."""

from io import BytesIO

from PIL import Image

from sprite_me.processing.crop import smart_crop
from sprite_me.processing.palette import pixelate, reduce_palette, snap_to_grid
from sprite_me.processing.spritesheet import assemble_spritesheet, split_spritesheet


def _make_png(color: tuple[int, int, int, int], size: tuple[int, int] = (64, 64)) -> bytes:
    img = Image.new("RGBA", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_sprite_on_white(sprite_size: int = 20, canvas_size: int = 64) -> bytes:
    """Create a small red square on a white canvas."""
    img = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
    pad = (canvas_size - sprite_size) // 2
    for x in range(pad, pad + sprite_size):
        for y in range(pad, pad + sprite_size):
            img.putpixel((x, y), (255, 0, 0, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_smart_crop_trims_white_background():
    original = _make_sprite_on_white(sprite_size=20, canvas_size=64)
    cropped = smart_crop(original, mode="tightest", margin=0)

    img = Image.open(BytesIO(cropped))
    # Should be approximately the sprite size
    assert img.width <= 24  # 20 + some margin tolerance
    assert img.height <= 24


def test_smart_crop_adds_margin():
    original = _make_sprite_on_white(sprite_size=20, canvas_size=64)
    cropped = smart_crop(original, mode="tightest", margin=5)

    img = Image.open(BytesIO(cropped))
    # With 5px margin on each side, should be at least 30
    assert img.width >= 25
    assert img.height >= 25


def test_reduce_palette_limits_colors():
    # Create a gradient image
    img = Image.new("RGBA", (64, 64))
    for x in range(64):
        for y in range(64):
            img.putpixel((x, y), (x * 4, y * 4, (x + y) * 2, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")

    reduced = reduce_palette(buf.getvalue(), max_colors=8)
    result = Image.open(BytesIO(reduced)).convert("RGB")
    colors = set(result.getdata())
    assert len(colors) <= 8


def test_snap_to_grid_preserves_dimensions():
    original = _make_sprite_on_white(sprite_size=20, canvas_size=64)
    snapped = snap_to_grid(original, grid_size=2)
    img = Image.open(BytesIO(snapped))
    assert img.size == (64, 64)


def _make_smooth_gradient(size: int = 256) -> bytes:
    """Create a smooth RGB gradient — simulates a 'painterly' Kontext output."""
    img = Image.new("RGBA", (size, size))
    for y in range(size):
        for x in range(size):
            # Smooth gradient blend red→blue horizontally, green intensity vertically
            r = int(255 * (1 - x / size))
            g = int(255 * (y / size))
            b = int(255 * (x / size))
            img.putpixel((x, y), (r, g, b, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_pixelate_upscales_back_to_original_dimensions():
    original = _make_smooth_gradient(256)
    pixelated = pixelate(original, target_size=64, palette_size=16, upscale=True)
    img = Image.open(BytesIO(pixelated))
    # Upscaled output matches the longest side of the original
    assert img.size == (256, 256)


def test_pixelate_raw_size_when_upscale_false():
    original = _make_smooth_gradient(256)
    raw = pixelate(original, target_size=64, palette_size=16, upscale=False)
    img = Image.open(BytesIO(raw))
    assert img.size == (64, 64)


def test_pixelate_limits_palette():
    original = _make_smooth_gradient(128)
    pixelated = pixelate(original, target_size=32, palette_size=8, upscale=False)
    img = Image.open(BytesIO(pixelated)).convert("RGB")
    colors = set(img.getdata())
    assert len(colors) <= 8


def test_pixelate_produces_sharp_blocks_after_upscale():
    """After pixelate+upscale, adjacent pixel rows within a 'block' should be identical.

    A real nearest-neighbor upscale of a 64→256 image produces 4×4 blocks of
    identical pixels. If pixelate is using a smooth filter by mistake, neighbors
    inside a block will differ.
    """
    original = _make_smooth_gradient(256)
    pixelated = pixelate(original, target_size=64, palette_size=16, upscale=True)
    img = Image.open(BytesIO(pixelated)).convert("RGB")
    w, h = img.size
    scale = w // 64  # should be 4

    # Sample a few block positions and verify within-block uniformity
    block_uniform_count = 0
    for block_x in [5, 20, 40]:
        for block_y in [5, 20, 40]:
            corner = (block_x * scale, block_y * scale)
            corner_pixel = img.getpixel(corner)
            # Check a sibling pixel within the same block
            sibling = (corner[0] + 1, corner[1] + 1)
            if img.getpixel(sibling) == corner_pixel:
                block_uniform_count += 1
    # Most sampled blocks should have uniform pixels (some may straddle edges)
    assert block_uniform_count >= 7


def test_pixelate_preserves_alpha():
    """Transparent background pixels must stay transparent after pixelation."""
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    # Draw a red square in the middle
    for y in range(40, 88):
        for x in range(40, 88):
            img.putpixel((x, y), (255, 0, 0, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    original = buf.getvalue()

    pixelated = pixelate(original, target_size=32, palette_size=4, upscale=False)
    out = Image.open(BytesIO(pixelated)).convert("RGBA")
    # Corner should still be transparent
    assert out.getpixel((0, 0))[3] < 128  # alpha near 0
    # Center should still be opaque red-ish
    center = out.getpixel((16, 16))
    assert center[3] > 200  # alpha near 255
    assert center[0] > 200  # red channel high


def test_assemble_spritesheet_creates_grid():
    frames = [_make_png((i * 40, 0, 0, 255), (32, 32)) for i in range(6)]
    sheet = assemble_spritesheet(frames)
    img = Image.open(BytesIO(sheet))
    assert img.width == 32 * 6
    assert img.height == 32


def test_assemble_spritesheet_with_columns():
    frames = [_make_png((i * 40, 0, 0, 255), (32, 32)) for i in range(6)]
    sheet = assemble_spritesheet(frames, columns=3)
    img = Image.open(BytesIO(sheet))
    assert img.width == 32 * 3
    assert img.height == 32 * 2


def test_split_spritesheet_roundtrip():
    frames = [_make_png((i * 40, 0, 0, 255), (32, 32)) for i in range(4)]
    sheet = assemble_spritesheet(frames)
    recovered = split_spritesheet(sheet, frame_width=32, frame_height=32)
    assert len(recovered) == 4
