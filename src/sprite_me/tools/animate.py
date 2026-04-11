"""Animate sprite tool — produce a pose-varied sprite sheet from a hero asset.

Pipeline:
    1. Look up the hero sprite in the manifest, load its PNG bytes
    2. Pick a list of per-frame pose prompts (from a preset or custom input)
    3. For each frame: submit a Kontext workflow to RunPod with the hero
       as the reference image + that frame's pose prompt. Same seed per
       frame anchors the RNG for character consistency.
    4. Post-process each returned frame: smart crop, background removal
    5. Assemble frames into a horizontal sprite sheet
    6. Save the sheet + individual frames, record in the manifest

Character identity is preserved by Kontext's ReferenceLatent mechanism —
the hero image carries color palette, armor design, and art style into
each frame. The Flux-2D-Game-Assets LoRA is NOT loaded during Kontext
inference; style flows from the reference instead.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
from typing import Any

from sprite_me.animation_presets import (
    ANIMATION_PRESETS,
    available_presets,
    preset_prompts,
)
from sprite_me.config import settings
from sprite_me.inference.runpod_client import RunPodClient, RunPodError
from sprite_me.inference.workflow_builder import build_animate_workflow
from sprite_me.processing.background import remove_background
from sprite_me.processing.crop import smart_crop
from sprite_me.processing.spritesheet import assemble_spritesheet
from sprite_me.storage.local import LocalStorage
from sprite_me.storage.manifest import Asset, AssetManifest

logger = logging.getLogger(__name__)

# Appended to custom_prompt when the user supplies a single text description
# instead of a list. Matches the suffix used in animation_presets.py so the
# character preservation instruction is always present.
_CUSTOM_PROMPT_SUFFIX = (
    "Keep the exact same character, clothing, armor, weapon, "
    "color palette, and art style unchanged. White background."
)


def _resolve_prompts(
    animation: str,
    custom_prompt: str | None,
    frames: int | None,
) -> list[str]:
    """Pick the list of per-frame pose prompts for this animation call."""
    if custom_prompt:
        # Treat as a single pose description; the preservation suffix gets
        # appended automatically.
        n = frames or 1
        return [f"{custom_prompt}. {_CUSTOM_PROMPT_SUFFIX}"] * n

    preset = preset_prompts(animation)
    if preset:
        # Truncate to the requested frame count if the caller asked for fewer
        # than the preset provides. Repeat-to-length is NOT done — we trust
        # the preset's canonical length.
        if frames and frames < len(preset):
            return preset[:frames]
        return preset

    # Unknown animation name, no custom prompt — fall back to a trivial
    # edit and log it so the caller sees what happened.
    logger.warning("Unknown animation preset %r; using literal prompt", animation)
    n = frames or 1
    return [f"Change the pose to the character performing a {animation} action. {_CUSTOM_PROMPT_SUFFIX}"] * n


async def _generate_frame(
    client: RunPodClient,
    pose_prompt: str,
    hero_b64: str,
    seed: int,
    steps: int,
    guidance: float,
) -> bytes | None:
    """Submit one Kontext frame generation. Returns PNG bytes or None on error."""
    workflow = build_animate_workflow(
        pose_prompt=pose_prompt,
        reference_image_name="hero.png",
        seed=seed,
        steps=steps,
        guidance=guidance,
    )
    # Kontext needs the hero image uploaded alongside the workflow so the
    # handler writes it to /comfyui/input/hero.png before the workflow runs.
    try:
        resp = await client.client.post(
            f"{client.base_url}/run",
            json={
                "input": {
                    "workflow": workflow,
                    "images": [{"name": "hero.png", "image": hero_b64}],
                }
            },
        )
        resp.raise_for_status()
        job_id = resp.json().get("id")
        if not job_id:
            logger.error("No job ID in submit response: %s", resp.text)
            return None
        payload = await client.wait_for_result(job_id)
        images = RunPodClient._extract_images(payload.get("output", {}))
        return images[0] if images else None
    except RunPodError as e:
        logger.warning("Frame generation failed: %s", e)
        return None
    except Exception as e:
        logger.warning("Unexpected error generating frame: %s", e)
        return None


async def animate_sprite(
    asset_id: str,
    animation: str = "idle",
    custom_prompt: str | None = None,
    frames: int | None = None,
    edge_margin: int | None = None,
    auto_enhance: bool | None = None,
    seed: int | None = None,
    steps: int = 20,
    guidance: float = 2.5,
    runpod: RunPodClient | None = None,
    storage: LocalStorage | None = None,
    manifest: AssetManifest | None = None,
) -> dict[str, Any]:
    """Generate a pose-varied sprite sheet from an existing hero asset.

    The public signature is kept compatible with previous sprite-me versions
    so MCP and REST callers don't need to change. `auto_enhance` is now a
    no-op (preset prompts are already fully fleshed out), `edge_margin` only
    controls the post-processing smart_crop margin, and `frames` optionally
    caps the preset length.

    Args:
        asset_id: Hero sprite ID in the manifest
        animation: Preset name (idle, walk, run, attack, jump, cast, death)
            or arbitrary label when custom_prompt is provided
        custom_prompt: Override preset with a single free-form pose prompt.
            The character-preservation suffix is appended automatically.
        frames: Optionally cap or expand the number of frames generated
        edge_margin: Pixels of padding in smart_crop for each frame (default 6)
        auto_enhance: No-op, kept for API compat
        seed: Base seed for the KSampler (same seed across frames for
            better consistency)
        steps: Kontext sampling steps, default 20
        guidance: FluxGuidance scale, default 2.5 (Kontext canonical value)

    Returns a dict with asset_id, filename, path, animation, frames,
    source_asset_id, seed. On failure: dict with "error" key.
    """
    runpod = runpod or RunPodClient()
    storage = storage or LocalStorage()
    manifest = manifest or AssetManifest()

    source = manifest.get(asset_id)
    if not source:
        return {"error": f"Asset {asset_id} not found"}
    if not storage.exists(source.filename):
        return {"error": f"Asset file {source.filename} not found on disk"}

    prompts = _resolve_prompts(animation, custom_prompt, frames)
    if not prompts:
        return {"error": f"No prompts resolved for animation {animation!r}"}

    actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    actual_margin = edge_margin if edge_margin is not None else settings.default_edge_margin

    hero_b64 = storage.load_b64(source.filename)
    logger.info(
        "animate_sprite: asset=%s animation=%s frames=%d seed=%d",
        asset_id, animation, len(prompts), actual_seed,
    )

    # Generate frames sequentially. Parallel submission is tempting but
    # risks throttling workers and makes debugging per-frame errors harder.
    # Each frame is ~20-40s warm, so 8 frames ≈ 3-5 min total.
    raw_frames: list[bytes] = []
    for i, pose_prompt in enumerate(prompts):
        logger.info("animate_sprite frame %d/%d", i + 1, len(prompts))
        frame_bytes = await _generate_frame(
            client=runpod,
            pose_prompt=pose_prompt,
            hero_b64=hero_b64,
            seed=actual_seed,
            steps=steps,
            guidance=guidance,
        )
        if frame_bytes is None:
            logger.warning("Skipping failed frame %d", i)
            continue
        raw_frames.append(frame_bytes)

    if not raw_frames:
        return {"error": "All frames failed to generate"}

    # Post-process: smart_crop to tighten bounds, then remove_background.
    # Padded mode for animation so arms/weapons don't get clipped when they
    # swing outside the static-frame bounds.
    processed_frames: list[bytes] = []
    for frame in raw_frames:
        try:
            frame = smart_crop(frame, mode="padded", margin=actual_margin)
            frame = remove_background(frame)
        except Exception as e:
            logger.warning("Post-processing failed on a frame: %s", e)
        processed_frames.append(frame)

    final_sheet = assemble_spritesheet(processed_frames)

    asset = Asset(
        prompt=custom_prompt or ANIMATION_PRESETS.get(animation, [animation])[0],
        asset_type="animation",
        width=source.width or settings.default_width,
        height=source.height or settings.default_height,
        seed=actual_seed,
        reference_asset_id=asset_id,
        frames=len(processed_frames),
    )
    asset.filename = f"{asset.asset_id}_sheet.png"
    asset.name = f"{animation} animation of {source.name or asset_id}"

    path = storage.save(asset.filename, final_sheet)
    for i, frame_data in enumerate(processed_frames):
        storage.save(f"{asset.asset_id}_frame{i:02d}.png", frame_data)
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
