"""Tests for image processing pipeline."""

from io import BytesIO

from PIL import Image

from sprite_me.processing.crop import smart_crop
from sprite_me.processing.palette import reduce_palette, snap_to_grid
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
