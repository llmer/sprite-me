"""Animate sprite tool — generate sprite sheet animation from an existing asset."""

from __future__ import annotations

import random
from typing import Any

from sprite_me.config import settings
from sprite_me.inference.runpod_client import RunPodClient
from sprite_me.inference.workflow_builder import build_animate_workflow
from sprite_me.processing.background import remove_background
from sprite_me.processing.crop import smart_crop
from sprite_me.processing.spritesheet import split_spritesheet, assemble_spritesheet
from sprite_me.storage.local import LocalStorage
from sprite_me.storage.manifest import Asset, AssetManifest

# Common animation presets with enhanced prompts
ANIMATION_PRESETS: dict[str, str] = {
    "idle": "idle breathing animation, subtle movement, standing still",
    "walk": "walk cycle animation, stepping forward, side view",
    "run": "running animation, fast movement, side view",
    "attack": "attack animation, sword slash, action pose",
    "jump": "jump animation, leaving ground and landing",
    "death": "death animation, falling down, collapse",
    "cast": "casting spell animation, magical energy, hands raised",
}


async def animate_sprite(
    asset_id: str,
    animation: str = "idle",
    custom_prompt: str | None = None,
    frames: int | None = None,
    edge_margin: int | None = None,
    auto_enhance: bool | None = None,
    seed: int | None = None,
    runpod: RunPodClient | None = None,
    storage: LocalStorage | None = None,
    manifest: AssetManifest | None = None,
) -> dict[str, Any]:
    """Generate an animated sprite sheet from an existing sprite asset.

    Returns a dict with the new animation asset_id, filename, frame count.
    """
    runpod = runpod or RunPodClient()
    storage = storage or LocalStorage()
    manifest = manifest or AssetManifest()

    # Look up the source asset
    source = manifest.get(asset_id)
    if not source:
        return {"error": f"Asset {asset_id} not found"}

    if not storage.exists(source.filename):
        return {"error": f"Asset file {source.filename} not found on disk"}

    # Build animation prompt
    should_enhance = auto_enhance if auto_enhance is not None else settings.auto_enhance_prompt
    if custom_prompt:
        anim_prompt = custom_prompt
    elif animation in ANIMATION_PRESETS and should_enhance:
        anim_prompt = ANIMATION_PRESETS[animation]
    else:
        anim_prompt = f"{animation} animation"

    # Include source character description for consistency
    if source.prompt:
        anim_prompt = f"{source.prompt}, {anim_prompt}"

    actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    actual_frames = frames or settings.default_animation_frames
    actual_margin = edge_margin if edge_margin is not None else settings.default_edge_margin

    # Load reference image
    ref_b64 = storage.load_b64(source.filename)

    workflow = build_animate_workflow(
        reference_image_b64=ref_b64,
        animation_prompt=anim_prompt,
        frames=actual_frames,
        width=source.width or settings.default_width,
        height=source.height or settings.default_height,
        seed=actual_seed,
        edge_margin=actual_margin,
    )

    # Submit to RunPod
    images = await runpod.generate(workflow)
    if not images:
        return {"error": "No animation frames generated"}

    # The workflow generates a wide image with all frames side by side
    # Split it into individual frames
    sheet_data = images[0]
    frame_w = source.width or settings.default_width
    frame_h = source.height or settings.default_height
    frame_images = split_spritesheet(sheet_data, frame_w, frame_h)

    if not frame_images:
        # If splitting failed, treat the whole image as the sheet
        frame_images = [sheet_data]

    # Post-process each frame
    processed_frames = []
    for frame in frame_images:
        frame = smart_crop(frame, mode="padded", margin=actual_margin)
        frame = remove_background(frame)
        processed_frames.append(frame)

    # Assemble into final sprite sheet
    final_sheet = assemble_spritesheet(processed_frames)

    # Save
    asset = Asset(
        prompt=anim_prompt,
        asset_type="animation",
        width=frame_w,
        height=frame_h,
        seed=actual_seed,
        reference_asset_id=asset_id,
        frames=len(processed_frames),
    )
    asset.filename = f"{asset.asset_id}_sheet.png"
    asset.name = f"{animation} animation of {source.name or asset_id}"

    path = storage.save(asset.filename, final_sheet)

    # Also save individual frames
    for i, frame_data in enumerate(processed_frames):
        frame_filename = f"{asset.asset_id}_frame{i:02d}.png"
        storage.save(frame_filename, frame_data)

    manifest.add(asset)

    return {
        "asset_id": asset.asset_id,
        "filename": asset.filename,
        "path": str(path),
        "animation": animation,
        "frames": len(processed_frames),
        "source_asset_id": asset_id,
        "seed": actual_seed,
    }
