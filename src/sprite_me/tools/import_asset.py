"""Import an existing image as a sprite-me asset."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from PIL import Image
from io import BytesIO

from sprite_me.storage.local import LocalStorage
from sprite_me.storage.manifest import Asset, AssetManifest


async def import_image(
    source: str,
    name: str = "",
    storage: LocalStorage | None = None,
    manifest: AssetManifest | None = None,
) -> dict[str, Any]:
    """Import a local file path or base64 data URL as a sprite-me asset.

    Args:
        source: File path or data URL (data:image/png;base64,...).
        name: Optional display name for the asset.

    Returns:
        Dict with asset_id, filename, path.
    """
    storage = storage or LocalStorage()
    manifest = manifest or AssetManifest()

    if source.startswith("data:"):
        # Parse data URL
        header, b64_data = source.split(",", 1)
        image_data = base64.b64decode(b64_data)
    else:
        # Local file path
        path = Path(source).expanduser().resolve()
        if not path.exists():
            return {"error": f"File not found: {source}"}
        image_data = path.read_bytes()

    # Get dimensions
    img = Image.open(BytesIO(image_data))
    w, h = img.size

    asset = Asset(
        name=name or Path(source).stem if not source.startswith("data:") else name,
        asset_type="imported",
        width=w,
        height=h,
    )
    asset.filename = f"{asset.asset_id}.png"

    # Convert to PNG if needed
    if img.format != "PNG":
        buf = BytesIO()
        img.save(buf, format="PNG")
        image_data = buf.getvalue()

    saved_path = storage.save(asset.filename, image_data)
    manifest.add(asset)

    return {
        "asset_id": asset.asset_id,
        "filename": asset.filename,
        "path": str(saved_path),
        "width": w,
        "height": h,
    }
