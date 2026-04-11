"""Build ComfyUI workflow JSON dynamically from generation parameters.

The returned dict is the raw node graph that ComfyUI's /prompt API expects
— just a map of node_id -> node definition. The runpod/worker-comfyui
handler wraps it with {"prompt": ..., "client_id": ...} before submitting,
so we must NOT wrap it ourselves (otherwise ComfyUI rejects with 400).

FLUX-specific node graph shape:
    CheckpointLoaderSimple -> LoraLoader -> CLIPTextEncode (positive)
                                         -> CLIPTextEncode (negative)
    CLIPTextEncode(positive) -> FluxGuidance -> KSampler.positive
    CLIPTextEncode(negative)                 -> KSampler.negative
    EmptySD3LatentImage (NOT EmptyLatentImage) -> KSampler.latent_image
    KSampler -> VAEDecode -> SaveImage

KSampler MUST use cfg=1.0 and scheduler="simple" for FLUX — the real
guidance scale is on the FluxGuidance node. Using cfg=3.5 directly on
KSampler (the stable-diffusion pattern) produces 400 errors or blown-out
images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_WORKFLOW_DIR = Path(__file__).parent.parent.parent.parent / "docker" / "comfyui" / "workflows"


def _load_template(name: str) -> dict[str, Any]:
    path = _WORKFLOW_DIR / name
    with open(path) as f:
        return json.load(f)


def build_generate_workflow(
    prompt: str,
    negative_prompt: str = "blurry, low quality, watermark, text, realistic, photograph, 3d render",
    width: int = 512,
    height: int = 512,
    seed: int = 0,
    steps: int = 20,
    guidance: float = 3.5,
    lora_strength: float = 0.85,
    lora_trigger: str = "GRPZA",
) -> dict[str, Any]:
    """Build a FLUX+LoRA pixel art generation workflow from a text prompt.

    Returns the raw ComfyUI node graph (not wrapped in {"prompt": ...}).

    Args:
        prompt: User text describing the sprite.
        negative_prompt: What to avoid in the output.
        width, height: Output dimensions.
        seed: KSampler seed (0 = random).
        steps: Sampling steps (20 is a good FLUX default; 30 for extra quality).
        guidance: FluxGuidance scale (was called cfg in SD; 3.5 is the
            worker-comfyui test default).
        lora_strength: Strength of the Flux-2D-Game-Assets LoRA (0.0-1.0).
        lora_trigger: LoRA activation word (GRPZA for Flux-2D-Game-Assets).
    """
    nodes = _load_template("pixel_art_generate.json")

    # LoRA strength (node 2 = LoraLoader)
    nodes["2"]["inputs"]["strength_model"] = lora_strength
    nodes["2"]["inputs"]["strength_clip"] = lora_strength

    # Positive prompt format taken verbatim from the Flux-2D-Game-Assets-LoRA
    # model card "Usage" section:
    #   GRPZA, <<Your Prompt>>, white background, game asset
    # Matching this order preserves the trigger+suffix shape the LoRA was
    # trained on. Users can include "pixel art" etc. in their own prompt.
    full_prompt = f"{lora_trigger}, {prompt}, white background, game asset"
    nodes["3"]["inputs"]["text"] = full_prompt

    # Negative prompt (node 4 = CLIPTextEncode negative)
    nodes["4"]["inputs"]["text"] = negative_prompt

    # Latent dimensions (node 5 = EmptySD3LatentImage)
    nodes["5"]["inputs"]["width"] = width
    nodes["5"]["inputs"]["height"] = height

    # FluxGuidance scale (node 9 = FluxGuidance)
    nodes["9"]["inputs"]["guidance"] = guidance

    # Sampler (node 6 = KSampler). cfg and scheduler are hardcoded in the
    # template — don't override them.
    nodes["6"]["inputs"]["seed"] = seed
    nodes["6"]["inputs"]["steps"] = steps

    return nodes


def build_animate_workflow(
    reference_image_b64: str,
    animation_prompt: str,
    frames: int = 6,
    width: int = 512,
    height: int = 512,
    seed: int = 0,
    steps: int = 20,
    guidance: float = 3.5,
    lora_strength: float = 0.85,
    lora_trigger: str = "GRPZA",
    edge_margin: int = 6,
) -> dict[str, Any]:
    """Build a FLUX animation workflow that generates N frame candidates.

    EXPERIMENTAL — the current approach uses `batch_size=N` on a single
    `EmptySD3LatentImage`, which gives N independent same-prompt variations.
    Each candidate is a full centered character sprite (not a strip), so
    you can use them as idle-breathing variations or as a starting point
    for hand-selecting frames.

    This is NOT a proper animation cycle: there is no pose control and no
    inter-frame coherence. For real walk/attack cycles you'd want either:
      - IP-Adapter / reference_image conditioning + explicit per-frame
        pose prompts (would require additional custom nodes in the image)
      - Sprite Sheet Diffusion (arxiv 2412.03685) — needs a custom model
      - Per-frame generation with ControlNet OpenPose

    A previous attempt rendered frames side-by-side in a single wide
    latent (width * N × height). That produced visually interesting output
    but the characters drifted in position and weren't frame-aligned, so
    the sprite sheet couldn't be sliced cleanly. batch_size is strictly
    better for this use case.

    Args:
        reference_image_b64: Currently unused (reserved for future IP-Adapter
            integration). Pass "" for now.
        animation_prompt: Describes the desired animation / motion.
        frames: Number of candidate frames to generate in one batch (=N).
        width, height: Per-frame dimensions (each frame is a full image).
        seed: Base seed. All frames share this seed so their random
            variation comes from the batch index, not fresh RNG.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "flux-2d-game-assets.safetensors",
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
                "model": ["1", 0],
                "clip": ["1", 1],
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                # HF model card order: GRPZA, <prompt>, white background, game asset.
                "text": f"{lora_trigger}, {animation_prompt}, centered character, white background, game asset",
                "clip": ["2", 1],
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "blurry, low quality, watermark, text, realistic, photograph, 3d render",
                "clip": ["2", 1],
            },
        },
        "5": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": frames,
            },
        },
        "9": {
            "class_type": "FluxGuidance",
            "inputs": {
                "guidance": guidance,
                "conditioning": ["3", 0],
            },
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["9", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"images": ["7", 0], "filename_prefix": "sprite-me-anim"},
        },
    }


def build_remove_background_workflow(image_b64: str) -> dict[str, Any]:
    """Build a background removal workflow using rembg node."""
    return {
        "1": {
            "class_type": "LoadImageBase64",
            "inputs": {"image": image_b64},
        },
        "2": {
            "class_type": "Image Remove Background (rembg)",
            "inputs": {"image": ["1", 0]},
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {"images": ["2", 0], "filename_prefix": "sprite-me-nobg"},
        },
    }
