"""Local filesystem storage for generated sprite assets."""

from __future__ import annotations

import base64
from pathlib import Path

from sprite_me.config import settings


class LocalStorage:
    """Store and retrieve sprite images from the local filesystem."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or settings.assets_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, data: bytes) -> Path:
        """Save image bytes to disk. Returns the full path."""
        path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def save_b64(self, filename: str, b64_data: str) -> Path:
        """Decode base64 and save to disk."""
        return self.save(filename, base64.b64decode(b64_data))

    def load(self, filename: str) -> bytes:
        """Load image bytes from disk."""
        path = self.base_dir / filename
        return path.read_bytes()

    def load_b64(self, filename: str) -> str:
        """Load image and return as base64 string."""
        return base64.b64encode(self.load(filename)).decode()

    def exists(self, filename: str) -> bool:
        return (self.base_dir / filename).exists()

    def delete(self, filename: str) -> bool:
        path = self.base_dir / filename
        if path.exists():
            path.unlink()
            return True
        return False

    def get_path(self, filename: str) -> Path:
        return self.base_dir / filename

    def list_files(self, pattern: str = "*.png") -> list[Path]:
        return sorted(self.base_dir.glob(pattern))
