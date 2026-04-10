"""Sprite sheet assembly and splitting."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


def assemble_spritesheet(
    frames: list[bytes],
    columns: int | None = None,
    padding: int = 0,
) -> bytes:
    """Assemble individual frame PNGs into a single sprite sheet.

    Args:
        frames: List of PNG bytes, one per frame.
        columns: Number of columns in the grid. Defaults to len(frames) (single row).
        padding: Pixels of padding between frames.

    Returns:
        PNG bytes of the assembled sprite sheet.
    """
    if not frames:
        raise ValueError("No frames to assemble")

    images = [Image.open(BytesIO(f)).convert("RGBA") for f in frames]

    # Use first frame dimensions as reference
    fw, fh = images[0].size
    cols = columns or len(images)
    rows = (len(images) + cols - 1) // cols

    sheet_w = cols * fw + (cols - 1) * padding
    sheet_h = rows * fh + (rows - 1) * padding
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

    for i, img in enumerate(images):
        col = i % cols
        row = i // cols
        x = col * (fw + padding)
        y = row * (fh + padding)
        # Resize if needed to match first frame
        if img.size != (fw, fh):
            img = img.resize((fw, fh), Image.NEAREST)
        sheet.paste(img, (x, y))

    buf = BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


def split_spritesheet(
    sheet_data: bytes,
    frame_width: int,
    frame_height: int,
    padding: int = 0,
) -> list[bytes]:
    """Split a sprite sheet into individual frame PNGs.

    Args:
        sheet_data: PNG bytes of the sprite sheet.
        frame_width: Width of each frame.
        frame_height: Height of each frame.
        padding: Pixels of padding between frames.

    Returns:
        List of PNG bytes, one per frame.
    """
    sheet = Image.open(BytesIO(sheet_data)).convert("RGBA")
    sw, sh = sheet.size

    cols = (sw + padding) // (frame_width + padding)
    rows = (sh + padding) // (frame_height + padding)

    frames: list[bytes] = []
    for row in range(rows):
        for col in range(cols):
            x = col * (frame_width + padding)
            y = row * (frame_height + padding)
            frame = sheet.crop((x, y, x + frame_width, y + frame_height))

            # Skip empty frames
            if frame.getbbox() is None:
                continue

            buf = BytesIO()
            frame.save(buf, format="PNG")
            frames.append(buf.getvalue())

    return frames
