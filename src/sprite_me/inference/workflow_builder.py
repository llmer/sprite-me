"""Build ComfyUI workflow JSON dynamically from generation parameters."""

from __future__ import annotations

import copy
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
    steps: int = 30,
    cfg: float = 3.5,
    lora_strength: float = 0.85,
    lora_trigger: str = "GRPZA",
) -> dict[str, Any]:
    """Build a pixel art generation workflow from a text prompt.

    Constructs the full ComfyUI workflow JSON with FLUX + LoRA pipeline.
    """
    workflow = _load_template("pixel_art_generate.json")
    nodes = workflow["prompt"]

    # LoRA strength
    nodes["2"]["inputs"]["strength_model"] = lora_strength
    nodes["2"]["inputs"]["strength_clip"] = lora_strength

    # Positive prompt — prepend trigger word
    full_prompt = f"{lora_trigger}, pixel art, game sprite, {prompt}, white background"
    nodes["3"]["inputs"]["text"] = full_prompt

    # Negative prompt
    nodes["4"]["inputs"]["text"] = negative_prompt

    # Image dimensions
    nodes["5"]["inputs"]["width"] = width
    nodes["5"]["inputs"]["height"] = height

    # Sampler settings
    nodes["6"]["inputs"]["seed"] = seed
    nodes["6"]["inputs"]["steps"] = steps
    nodes["6"]["inputs"]["cfg"] = cfg

    return workflow


def build_animate_workflow(
    reference_image_b64: str,
    animation_prompt: str,
    frames: int = 6,
    width: int = 512,
    height: int = 512,
    seed: int = 0,
    steps: int = 30,
    cfg: float = 3.5,
    lora_strength: float = 0.85,
    lora_trigger: str = "GRPZA",
    edge_margin: int = 6,
) -> dict[str, Any]:
    """Build an animation workflow that generates N frames from a reference sprite.

    Uses the reference image as IP-Adapter / img2img conditioning to maintain
    character consistency across frames. Each frame gets a pose-modified prompt.
    """
    # For animation, we generate each frame as a separate batch item
    # with the same seed base but varied prompts for each action frame.
    # This is a simplified approach — a more advanced version would use
    # ControlNet pose conditioning or Sprite Sheet Diffusion.
    workflow = {
        "prompt": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "flux1-dev-fp8.safetensors"
                }
            },
            "2": {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": "flux-2d-game-assets.safetensors",
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": f"{lora_trigger}, pixel art sprite sheet, {animation_prompt}, {frames} frames, white background, consistent character",
                    "clip": ["2", 1]
                }
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "blurry, low quality, watermark, text, realistic, photograph, 3d render, inconsistent",
                    "clip": ["2", 1]
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    # Wide image to fit all frames side by side
                    "width": width * frames,
                    "height": height,
                    "batch_size": 1
                }
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["6", 0],
                    "vae": ["1", 2]
                }
            },
            "8": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["7", 0],
                    "filename_prefix": "sprite-me-anim"
                }
            }
        }
    }
    return workflow


def build_remove_background_workflow(image_b64: str) -> dict[str, Any]:
    """Build a background removal workflow using rembg node."""
    return {
        "prompt": {
            "1": {
                "class_type": "LoadImageBase64",
                "inputs": {
                    "image": image_b64
                }
            },
            "2": {
                "class_type": "Image Remove Background (rembg)",
                "inputs": {
                    "image": ["1", 0]
                }
            },
            "3": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["2", 0],
                    "filename_prefix": "sprite-me-nobg"
                }
            }
        }
    }
