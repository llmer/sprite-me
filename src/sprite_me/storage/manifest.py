"""Asset manifest — tracks generated sprites with metadata."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sprite_me.config import settings


class Asset(BaseModel):
    """A generated or imported sprite asset."""

    asset_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    prompt: str = ""
    filename: str = ""
    asset_type: str = "sprite"  # sprite | animation | imported
    width: int = 0
    height: int = 0
    seed: int = 0
    reference_asset_id: str | None = None
    frames: int = 1
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssetManifest:
    """JSON-file-backed asset manifest."""

    def __init__(self, path: Path | None = None):
        self.path = path or settings.manifest_path
        self._assets: dict[str, Asset] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for item in data.get("assets", []):
                asset = Asset(**item)
                self._assets[asset.asset_id] = asset

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"assets": [a.model_dump() for a in self._assets.values()]}
        self.path.write_text(json.dumps(data, indent=2))

    def add(self, asset: Asset) -> Asset:
        self._assets[asset.asset_id] = asset
        self._save()
        return asset

    def get(self, asset_id: str) -> Asset | None:
        return self._assets.get(asset_id)

    def list_assets(self) -> list[Asset]:
        return list(self._assets.values())

    def delete(self, asset_id: str) -> bool:
        if asset_id in self._assets:
            del self._assets[asset_id]
            self._save()
            return True
        return False

    def update(self, asset_id: str, **kwargs: Any) -> Asset | None:
        asset = self._assets.get(asset_id)
        if not asset:
            return None
        for key, value in kwargs.items():
            if hasattr(asset, key):
                setattr(asset, key, value)
        self._save()
        return asset
