"""Animate sprite tool — agent-composed pose sheet from a hero asset.

The agent supplies a list of pose-change prompts (one per frame); the tool
runs one Kontext call per prompt against the hero as reference, post-
processes each returned frame, assembles a horizontal sheet, and stores
the result.

There is deliberately no preset library here. The old `animation=\"walk\"`
API encoded side-view full-body humanoid assumptions; agents now compose
pose prompts that match their specific hero's body topology, using the
recipes in skills/sprite-me-animate.md.

Character identity is preserved by Kontext's ReferenceLatent mechanism —
the hero image carries color palette, armor design, and art style into
each frame. The generate-path LoRA is NOT loaded during Kontext inference;
style flows from the reference.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from sprite_me.config import settings
from sprite_me.inference.runpod_client import RunPodClient, RunPodError
from sprite_me.inference.workflow_builder import build_animate_workflow
from sprite_me.processing.background import remove_background
from sprite_me.processing.crop import smart_crop
from sprite_me.processing.palette import pixelate as pixelate_image
from sprite_me.processing.spritesheet import assemble_spritesheet
from sprite_me.storage.local import LocalStorage
from sprite_me.storage.manifest import Asset, AssetManifest

logger = logging.getLogger(__name__)

_PRESERVATION_SUFFIX = (
    "Keep the exact same character, clothing, armor, weapon, "
    "color palette, and art style unchanged. White background."
)

_REMOVED_PARAMS = ("animation", "custom_prompt", "frames", "auto_enhance")


def _apply_suffix(pose: str) -> str:
    """Append the character-preservation clause if the caller didn't already.

    Agents often forget to tack this on. It's cheap to check and adds
    materially to Kontext's identity preservation.
    """
    if "unchanged" in pose.lower():
        return pose
    return f"{pose.rstrip('. ')}. {_PRESERVATION_SUFFIX}"


async def _generate_frame(
    client: RunPodClient,
    pose_prompt: str,
    hero_b64: str,
    seed: int,
    steps: int,
    guidance: float,
) -> bytes | None:
    """Submit one Kontext frame. Returns PNG bytes or None on error."""
    workflow = build_animate_workflow(
        pose_prompt=pose_prompt,
        reference_image_name="hero.png",
        seed=seed,
        steps=steps,
        guidance=guidance,
    )
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
    pose_prompts: list[str],
    seed: int | None = None,
    steps: int = 20,
    guidance: float = 2.5,
    edge_margin: int | None = None,
    pixelate: bool = False,
    pixel_size: int = 64,
    palette_size: int = 16,
    runpod: RunPodClient | None = None,
    storage: LocalStorage | None = None,
    manifest: AssetManifest | None = None,
    **removed_params: Any,
) -> dict[str, Any]:
    """Generate a sprite sheet from a hero asset and a list of pose prompts.

    One Kontext call per entry in `pose_prompts`. Frames are generated
    sequentially (parallel submission risks worker throttling and muddies
    per-frame debugging).

    Args:
        asset_id: Hero sprite ID in the manifest.
        pose_prompts: One pose-change instruction per output frame. The
            character-preservation suffix is appended automatically if
            the prompt doesn't already include it. See
            skills/sprite-me-animate.md for composition recipes.
        seed: Base seed shared across frames (improves consistency).
        steps: Kontext sampling steps (default 20).
        guidance: FluxGuidance scale (default 2.5, Kontext canonical).
        edge_margin: Smart-crop margin per frame.
        pixelate, pixel_size, palette_size: Optional retro pixelation
            applied after background removal.
    """
    # v0.5.0 clean break: old callers passed animation/custom_prompt/frames.
    # Error loudly so the breakage is visible, don't silently translate.
    for name in _REMOVED_PARAMS:
        if name in removed_params:
            raise ValueError(
                f"animate_sprite no longer accepts {name!r}. Compose "
                f"pose_prompts directly — one entry per frame. "
                f"See skills/sprite-me-animate.md for pose recipes."
            )
    if removed_params:
        raise TypeError(
            f"Unknown kwargs: {sorted(removed_params)}. "
            f"See skills/sprite-me-animate.md."
        )

    if not pose_prompts:
        raise ValueError(
            "pose_prompts must contain at least one entry. "
            "See skills/sprite-me-animate.md for composition recipes."
        )

    runpod = runpod or RunPodClient()
    storage = storage or LocalStorage()
    manifest = manifest or AssetManifest()

    source = manifest.get(asset_id)
    if not source:
        return {"error": f"Asset {asset_id} not found"}
    if not storage.exists(source.filename):
        return {"error": f"Asset file {source.filename} not found on disk"}

    actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    actual_margin = edge_margin if edge_margin is not None else settings.default_edge_margin

    hero_b64 = storage.load_b64(source.filename)
    logger.info(
        "animate_sprite: asset=%s frames=%d seed=%d",
        asset_id, len(pose_prompts), actual_seed,
    )

    raw_frames: list[bytes] = []
    for i, raw_pose in enumerate(pose_prompts):
        pose = _apply_suffix(raw_pose)
        logger.info("animate_sprite frame %d/%d", i + 1, len(pose_prompts))
        frame_bytes = await _generate_frame(
            client=runpod,
            pose_prompt=pose,
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

    processed_frames: list[bytes] = []
    for frame in raw_frames:
        try:
            frame = smart_crop(frame, mode="padded", margin=actual_margin)
            frame = remove_background(frame)
            if pixelate:
                frame = pixelate_image(
                    frame,
                    target_size=pixel_size,
                    palette_size=palette_size,
                    upscale=True,
                )
        except Exception as e:
            logger.warning("Post-processing failed on a frame: %s", e)
        processed_frames.append(frame)

    final_sheet = assemble_spritesheet(processed_frames)

    asset = Asset(
        prompt=pose_prompts[0],
        asset_type="animation",
        width=source.width or settings.default_width,
        height=source.height or settings.default_height,
        seed=actual_seed,
        reference_asset_id=asset_id,
        frames=len(processed_frames),
        metadata={"source_lora": source.metadata.get("lora") if source.metadata else None},
    )
    asset.filename = f"{asset.asset_id}_sheet.png"
    asset.name = f"animation of {source.name or asset_id}"

    path = storage.save(asset.filename, final_sheet)
    for i, frame_data in enumerate(processed_frames):
        storage.save(f"{asset.asset_id}_frame{i:02d}.png", frame_data)
    manifest.add(asset)

    return {
        "asset_id": asset.asset_id,
        "filename": asset.filename,
        "path": str(path),
        "frames": len(processed_frames),
        "source_asset_id": asset_id,
        "seed": actual_seed,
    }
