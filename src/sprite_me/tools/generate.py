"""Generate sprite tool — text prompt to pixel art sprite."""

from __future__ import annotations

import random
from typing import Any

from sprite_me.config import settings
from sprite_me.inference.runpod_client import RunPodClient
from sprite_me.inference.workflow_builder import build_generate_workflow
from sprite_me.processing.background import remove_background
from sprite_me.processing.crop import smart_crop
from sprite_me.storage.local import LocalStorage
from sprite_me.storage.manifest import Asset, AssetManifest


async def generate_sprite(
    prompt: str,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    lora_strength: float | None = None,
    smart_crop_mode: str | None = None,
    remove_bg: bool = True,
    reference_asset_id: str | None = None,
    runpod: RunPodClient | None = None,
    storage: LocalStorage | None = None,
    manifest: AssetManifest | None = None,
) -> dict[str, Any]:
    """Generate a pixel art sprite from a text prompt.

    Returns a dict with asset_id, filename, path, and metadata.
    """
    runpod = runpod or RunPodClient()
    storage = storage or LocalStorage()
    manifest = manifest or AssetManifest()

    actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    # If reference asset provided, include its prompt info for consistency
    ref_prompt_suffix = ""
    if reference_asset_id:
        ref_asset = manifest.get(reference_asset_id)
        if ref_asset and ref_asset.prompt:
            ref_prompt_suffix = f", same style as {ref_asset.prompt}"

    workflow = build_generate_workflow(
        prompt=prompt + ref_prompt_suffix,
        width=width or settings.default_width,
        height=height or settings.default_height,
        seed=actual_seed,
        steps=steps or settings.default_steps,
        guidance=guidance or settings.default_guidance,
        lora_strength=lora_strength or settings.default_lora_strength,
    )

    # Submit to RunPod and wait for result
    images = await runpod.generate(workflow)

    if not images:
        return {"error": "No images generated"}

    image_data = images[0]

    # Post-processing
    crop_mode = smart_crop_mode or settings.default_smart_crop_mode
    image_data = smart_crop(image_data, mode=crop_mode)

    if remove_bg:
        image_data = remove_background(image_data)

    # Save to storage
    asset = Asset(
        prompt=prompt,
        asset_type="sprite",
        width=width or settings.default_width,
        height=height or settings.default_height,
        seed=actual_seed,
        reference_asset_id=reference_asset_id,
    )
    asset.filename = f"{asset.asset_id}.png"
    asset.name = prompt[:60]

    path = storage.save(asset.filename, image_data)
    manifest.add(asset)

    return {
        "asset_id": asset.asset_id,
        "filename": asset.filename,
        "path": str(path),
        "prompt": prompt,
        "seed": actual_seed,
        "width": asset.width,
        "height": asset.height,
    }
