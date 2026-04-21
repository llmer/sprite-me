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
from sprite_me.loras import DEFAULT_LORA, LORAS
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
    guidance: float = 3.5,
    lora: str = DEFAULT_LORA,
    lora_strength: float | None = None,
    smart_crop_mode: str = "tightest",
    remove_bg: bool = True,
    pixelate: bool = False,
    pixel_size: int = 64,
    palette_size: int = 16,
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
        guidance: FluxGuidance scale (3.5 is a good default).
        lora: LoRA profile key. Picks the art style. Call list_loras()
            to get the full up-to-date list with empirical descriptions.
            Available profiles (see src/sprite_me/loras.py for details):
              - "cartoon-vector" (default): smooth cartoon/vector — heavy
                black outlines, flat fills, casual-game polish. NOT pixel
                art. [best for: cartoon, vector, heroes, casual-game]
              - "pixel-indie": clean modern pixel art with visible grid,
                vibrant colors, JRPG/indie-Steam feel. Works across humans
                and creatures. [best for: pixel-art, indie, jrpg]
              - "pixel-retro": retro 8/16-bit era small-canvas pixel art.
                Crunchy palette. Often includes a scene background — may
                need post-cropping. [best for: retro, 8-bit, 16-bit, nes]
              - "top-down": bird's-eye perspective scenes (not isolated
                sprites). [best for: top-down, tiles, environments]
        lora_strength: Override the profile's default strength (0-1). None =
            use the profile default.
        smart_crop_mode: "tightest" or "padded".
        remove_bg: If true, output transparent background.
        pixelate: If true, apply a retro pixelation pass after background removal.
            Turns the smooth FLUX output into a crisp classic-game pixel sprite.
        pixel_size: Target pixel resolution when pixelate=True (64 classic,
            32 Game Boy, 16 tile).
        palette_size: Max colors in the palette when pixelate=True (16 classic,
            8 Game Boy, 32 modern retro).
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
        guidance=guidance,
        lora=lora,
        lora_strength=lora_strength,
        smart_crop_mode=smart_crop_mode,
        remove_bg=remove_bg,
        pixelate=pixelate,
        pixel_size=pixel_size,
        palette_size=palette_size,
        reference_asset_id=reference_asset_id,
        runpod=runpod,
        storage=storage,
        manifest=manifest,
    )


@mcp.tool()
async def animate_sprite(
    asset_id: str,
    pose_prompts: list[str],
    seed: int | None = None,
    steps: int = 20,
    guidance: float = 2.5,
    denoise: float = 1.0,
    chain_frames: bool = False,
    edge_margin: int = 6,
    pixelate: bool = False,
    pixel_size: int = 64,
    palette_size: int = 16,
) -> dict[str, Any]:
    """Generate a sprite sheet by running one Kontext call per pose prompt.

    Before calling this, read the hero asset (via get_asset) to check its
    body topology and the LoRA that generated it. Compose pose_prompts
    that match the hero's shape — don't ask a chibi slime to walk or a
    top-down character to "turn to side view", Kontext will produce
    morphological garbage. See skills/sprite-me-animate.md for recipes.

    Args:
        asset_id: The source sprite to animate.
        pose_prompts: One pose-change instruction per output frame, in
            order. Each should describe what changes in this frame,
            not the character. Example: "Change the pose to walking
            forward, left foot lifted mid-stride, side view."
        seed: Base seed shared across frames (improves consistency).
        steps: Kontext sampling steps (20 is canonical default).
        guidance: FluxGuidance scale (2.5 is Kontext default).
        denoise: Pose-change magnitude per frame (0.0-1.0). Default 1.0
            is correct for action animations (walk, run, attack, jump).
            Drop to 0.5-0.6 only for idle/breathing loops where subtle
            motion is wanted.
        chain_frames: Experimental, off by default. Chains each frame's
            reference to the previous output instead of the hero. Causes
            cumulative VAE drift — colors desaturate, details wash out
            — because Kontext was designed for single-frame edits, not
            iterative sequences. Enable only for short 2-3 frame
            sequences where you consciously accept the drift. Leave off
            for normal sprite animation.
        edge_margin: Smart-crop margin per frame.
        pixelate: If true, apply retro pixelation to each frame.
        pixel_size: Target pixel resolution when pixelate=True.
        palette_size: Max colors per frame when pixelate=True.

    Returns:
        Dict with asset_id, filename, path, frames, source_asset_id, seed.
    """
    runpod, storage, manifest = _services()
    return await _animate_sprite(
        asset_id=asset_id,
        pose_prompts=pose_prompts,
        seed=seed,
        steps=steps,
        guidance=guidance,
        denoise=denoise,
        chain_frames=chain_frames,
        edge_margin=edge_margin,
        pixelate=pixelate,
        pixel_size=pixel_size,
        palette_size=palette_size,
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
async def list_loras() -> list[dict[str, Any]]:
    """List the LoRA style profiles available to generate_sprite.

    Returns one dict per profile with key, description, best_for tags,
    trigger word, and default strength. Use this to pick the right
    `lora` parameter for a generate_sprite call when the user describes
    a specific art style (e.g. "Pokemon trainer" → "trainer-sprites",
    "top-down dungeon tiles" → "top-down").
    """
    return [
        {
            "key": key,
            "description": profile.description,
            "best_for": list(profile.best_for),
            "trigger": profile.trigger,
            "default_strength": profile.default_strength,
        }
        for key, profile in LORAS.items()
    ]


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
