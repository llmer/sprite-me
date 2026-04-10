"""MCP server for sprite-me.

Exposes sprite generation, animation, and asset management tools to AI agents
via the Model Context Protocol (stdio, SSE, or Streamable HTTP).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from sprite_me.config import settings
from sprite_me.inference.runpod_client import RunPodClient
from sprite_me.storage.local import LocalStorage
from sprite_me.storage.manifest import AssetManifest
from sprite_me.tools.animate import animate_sprite as _animate_sprite
from sprite_me.tools.assets import (
    delete_asset as _delete_asset,
    get_asset as _get_asset,
    list_assets as _list_assets,
)
from sprite_me.tools.generate import generate_sprite as _generate_sprite
from sprite_me.tools.import_asset import import_image as _import_image
from sprite_me.tools.status import check_job_status as _check_status

logger = logging.getLogger("sprite_me.server")

mcp = FastMCP(
    "sprite-me",
    instructions=(
        "Agent-first pixel art sprite generator. "
        "Use generate_sprite to create sprites from text prompts, "
        "animate_sprite to create sprite sheet animations, "
        "and import_image to bring in existing PNGs. "
        "Always generate a hero asset first, then use its asset_id as "
        "reference_asset_id for subsequent generations to keep a consistent style."
    ),
)

# Shared service instances — initialized lazily so stdio launch is fast
_runpod: RunPodClient | None = None
_storage: LocalStorage | None = None
_manifest: AssetManifest | None = None


def _services() -> tuple[RunPodClient, LocalStorage, AssetManifest]:
    global _runpod, _storage, _manifest
    if _runpod is None:
        _runpod = RunPodClient()
    if _storage is None:
        _storage = LocalStorage()
    if _manifest is None:
        _manifest = AssetManifest()
    return _runpod, _storage, _manifest


@mcp.tool()
async def generate_sprite(
    prompt: str,
    width: int = 512,
    height: int = 512,
    seed: int | None = None,
    steps: int = 30,
    cfg: float = 3.5,
    lora_strength: float = 0.85,
    smart_crop_mode: str = "tightest",
    remove_bg: bool = True,
    reference_asset_id: str | None = None,
) -> dict[str, Any]:
    """Generate a pixel art sprite from a text prompt.

    To keep a set of sprites visually consistent, generate one hero asset first,
    then pass its asset_id as reference_asset_id for subsequent calls.

    Args:
        prompt: What to generate, e.g. "warrior with sword and shield".
        width: Output width in pixels (default 512).
        height: Output height in pixels (default 512).
        seed: Random seed for reproducibility. None = random.
        steps: Inference steps (more = higher quality, slower).
        cfg: Classifier-free guidance scale.
        lora_strength: How strongly the game-asset LoRA influences output (0-1).
        smart_crop_mode: "tightest" or "padded".
        remove_bg: If true, output transparent background.
        reference_asset_id: Use this asset's prompt as a style reference.

    Returns:
        Dict with asset_id, filename, path, prompt, seed, width, height.
    """
    runpod, storage, manifest = _services()
    return await _generate_sprite(
        prompt=prompt,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        cfg=cfg,
        lora_strength=lora_strength,
        smart_crop_mode=smart_crop_mode,
        remove_bg=remove_bg,
        reference_asset_id=reference_asset_id,
        runpod=runpod,
        storage=storage,
        manifest=manifest,
    )


@mcp.tool()
async def animate_sprite(
    asset_id: str,
    animation: str = "idle",
    custom_prompt: str | None = None,
    frames: int = 6,
    edge_margin: int = 6,
    auto_enhance: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate an animated sprite sheet from an existing sprite asset.

    The source asset must already exist in the sprite-me manifest. If you have
    a local PNG, call import_image first to get an asset_id.

    Args:
        asset_id: The source sprite to animate.
        animation: Preset name — idle, walk, run, attack, jump, death, cast.
        custom_prompt: Override the preset with a custom motion description.
        frames: Number of animation frames (4-12 typical).
        edge_margin: Pixel margin around each frame (default 6).
        auto_enhance: Expand simple prompts into detailed motion descriptions.
        seed: Random seed for reproducibility.

    Returns:
        Dict with asset_id, filename, path, animation, frames, source_asset_id.
    """
    runpod, storage, manifest = _services()
    return await _animate_sprite(
        asset_id=asset_id,
        animation=animation,
        custom_prompt=custom_prompt,
        frames=frames,
        edge_margin=edge_margin,
        auto_enhance=auto_enhance,
        seed=seed,
        runpod=runpod,
        storage=storage,
        manifest=manifest,
    )


@mcp.tool()
async def import_image(source: str, name: str = "") -> dict[str, Any]:
    """Import a local PNG or data URL as a sprite-me asset.

    Required before animating an external image. Returns a new asset_id that
    can be passed to animate_sprite.

    Args:
        source: File path (e.g. "/path/to/sprite.png") or "data:image/png;base64,..." URL.
        name: Optional display name for the asset.

    Returns:
        Dict with asset_id, filename, path, width, height.
    """
    _, storage, manifest = _services()
    return await _import_image(source=source, name=name, storage=storage, manifest=manifest)


@mcp.tool()
async def check_status(job_id: str) -> dict[str, Any]:
    """Check the status of a running RunPod generation job.

    Most calls complete synchronously, but this is available for very long jobs.

    Args:
        job_id: The RunPod job ID from a previous submission.

    Returns:
        Dict with job_id, status, delay_time, execution_time, completed.
    """
    runpod, _, _ = _services()
    return await _check_status(job_id=job_id, runpod=runpod)


@mcp.tool()
async def list_assets() -> list[dict[str, Any]]:
    """List all generated and imported sprite assets with metadata."""
    _, _, manifest = _services()
    return await _list_assets(manifest=manifest)


@mcp.tool()
async def get_asset(asset_id: str) -> dict[str, Any]:
    """Get details for a specific asset, including file path.

    Args:
        asset_id: The asset to retrieve.

    Returns:
        Full asset metadata plus local file path.
    """
    _, storage, manifest = _services()
    return await _get_asset(asset_id=asset_id, manifest=manifest, storage=storage)


@mcp.tool()
async def delete_asset(asset_id: str) -> dict[str, Any]:
    """Delete an asset from both the manifest and disk.

    Args:
        asset_id: The asset to delete.
    """
    _, storage, manifest = _services()
    return await _delete_asset(asset_id=asset_id, manifest=manifest, storage=storage)


@mcp.resource("asset://{asset_id}")
async def get_asset_resource(asset_id: str) -> str:
    """Direct resource access to a sprite asset by ID. Returns the file path."""
    _, storage, manifest = _services()
    asset = manifest.get(asset_id)
    if not asset:
        return f"Asset {asset_id} not found"
    return str(storage.get_path(asset.filename))


def main() -> None:
    """Entry point for the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    transport = settings.mcp_transport
    logger.info("Starting sprite-me MCP server (transport=%s)", transport)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
