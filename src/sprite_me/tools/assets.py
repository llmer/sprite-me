"""Asset management tools — list, get, delete."""

from __future__ import annotations

from typing import Any

from sprite_me.storage.local import LocalStorage
from sprite_me.storage.manifest import AssetManifest


async def list_assets(
    manifest: AssetManifest | None = None,
) -> list[dict[str, Any]]:
    """List all sprite assets with metadata."""
    manifest = manifest or AssetManifest()
    return [a.model_dump() for a in manifest.list_assets()]


async def get_asset(
    asset_id: str,
    manifest: AssetManifest | None = None,
    storage: LocalStorage | None = None,
) -> dict[str, Any]:
    """Get details for a specific asset, including its file path."""
    manifest = manifest or AssetManifest()
    storage = storage or LocalStorage()

    asset = manifest.get(asset_id)
    if not asset:
        return {"error": f"Asset {asset_id} not found"}

    result = asset.model_dump()
    result["path"] = str(storage.get_path(asset.filename))
    result["exists"] = storage.exists(asset.filename)
    return result


async def delete_asset(
    asset_id: str,
    manifest: AssetManifest | None = None,
    storage: LocalStorage | None = None,
) -> dict[str, Any]:
    """Delete an asset from manifest and disk."""
    manifest = manifest or AssetManifest()
    storage = storage or LocalStorage()

    asset = manifest.get(asset_id)
    if not asset:
        return {"error": f"Asset {asset_id} not found"}

    storage.delete(asset.filename)
    manifest.delete(asset_id)

    return {"deleted": asset_id}
