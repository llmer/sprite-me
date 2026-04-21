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

from sprite_me.loras import DEFAULT_LORA, format_prompt, get_profile

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
    lora: str = DEFAULT_LORA,
    lora_strength: float | None = None,
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
        lora: Profile key from `sprite_me.loras.LORAS` — selects the
            filename, trigger word, and prompt template used below.
        lora_strength: Override the profile's default strength (0.0-1.0).
            None = use the profile default.
    """
    profile = get_profile(lora)
    strength = lora_strength if lora_strength is not None else profile.default_strength

    nodes = _load_template("pixel_art_generate.json")

    # LoRA file + strength (node 2 = LoraLoader). The template's default
    # filename is overwritten here so callers can swap LoRAs without
    # editing the JSON.
    nodes["2"]["inputs"]["lora_name"] = profile.name
    nodes["2"]["inputs"]["strength_model"] = strength
    nodes["2"]["inputs"]["strength_clip"] = strength

    # Each profile owns its own prompt shape — trigger-word LoRAs keep
    # the "GRPZA, ..., white background, game asset" form; BLIP-captioned
    # LoRAs use "a pixel art sprite of ...".
    nodes["3"]["inputs"]["text"] = format_prompt(profile, prompt)

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
    pose_prompt: str,
    reference_image_name: str = "hero.png",
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    steps: int = 20,
    guidance: float = 2.5,
    denoise: float = 1.0,
) -> dict[str, Any]:
    """Build a FLUX.1 Kontext single-frame animation workflow.

    Takes a reference image (hero sprite) + a pose-change prompt and returns
    a ComfyUI node graph that produces one frame showing the same character
    in the requested pose. Character identity, art style, and palette flow
    from the reference image via Kontext's ReferenceLatent mechanism — no
    LoRA load needed, no manual style prompting, no pose skeleton.

    The hero PNG must be uploaded as an "images" entry in the /run payload
    so the runpod handler writes it to /comfyui/input/ before ComfyUI starts
    the workflow.

    Args:
        pose_prompt: Kontext-style edit instruction. Best form:
            "Change the pose to <new pose>. Keep the exact same character,
             clothing, palette, and art style unchanged."
        reference_image_name: Filename the handler will write the hero to
            in /comfyui/input/.
        seed: KSampler seed. Same seed across frames improves consistency.
        steps: Sampling steps (default 20).
        guidance: FluxGuidance scale (default 2.5).
        denoise: KSampler denoise strength. 1.0 = full denoise from noise
            (maximum pose change, required for readable action animations).
            Lower values pull output toward the reference (more stable but
            less motion). Default 1.0 — chain_frames=True in animate_sprite
            provides frame-to-frame smoothness via the reference chain
            rather than via denoise reduction. Override to 0.5-0.7 for
            idle/breathing animations where only subtle motion is wanted.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux1-dev-kontext_fp8_scaled.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "flux",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"},
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image_name},
        },
        "5": {
            "class_type": "FluxKontextImageScale",
            "inputs": {"image": ["4", 0]},
        },
        "6": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["3", 0]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": pose_prompt, "clip": ["2", 0]},
        },
        "8": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["7", 0], "latent": ["6", 0]},
        },
        "9": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["8", 0], "guidance": guidance},
        },
        "10": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["7", 0]},
        },
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise,
                "model": ["1", 0],
                "positive": ["9", 0],
                "negative": ["10", 0],
                "latent_image": ["6", 0],
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
        },
        "13": {
            "class_type": "SaveImage",
            "inputs": {"images": ["12", 0], "filename_prefix": "sprite-me-kontext"},
        },
    }


def build_animate_workflow_animatediff(
    motion_prompt: str,
    reference_image_name: str = "hero.png",
    checkpoint: str = "toonyou_beta6.safetensors",
    motion_module: str = "v3_sd15_mm.ckpt",
    ipadapter_preset: str = "PLUS (high strength)",
    ipadapter_weight: float = 0.75,
    negative_prompt: str = (
        "blurry, low quality, watermark, text, photograph, realistic, "
        "extra limbs, deformed, bad anatomy, ugly"
    ),
    width: int = 512,
    height: int = 512,
    frames: int = 16,
    seed: int = 0,
    steps: int = 25,
    cfg: float = 7.5,
    sampler_name: str = "euler",
    scheduler: str = "normal",
) -> dict[str, Any]:
    """Build a SD1.5 + AnimateDiff v3 + IP-Adapter animation workflow.

    Replaces the single-frame Kontext path for `animate_sprite` as of v0.6.
    Character identity flows from the hero image via IP-Adapter's CLIP Vision
    conditioning (not from text-prompt engineering or ReferenceLatent).
    Temporal coherence is owned by the AnimateDiff motion module, which
    jointly denoises a 16-frame latent batch so poses morph smoothly between
    frames rather than being re-invented per call.

    Node graph:
        CheckpointLoaderSimple (SD1.5)
            └── IPAdapterUnifiedLoader (loads ipadapter + CLIP Vision)
                └── IPAdapterAdvanced (hero image -> model conditioning)
                    └── ADE_LoadAnimateDiffModel (motion module)
                        └── ADE_UseEvolvedSampling (wraps model for batched
                            temporal sampling)
                                └── KSampler (latent_image = empty 16-frame
                                    SD1.5 batch) -> VAEDecode -> SaveImage

    The hero PNG must be uploaded as an "images" entry in the /run payload
    so the runpod-worker-comfyui handler writes it to /comfyui/input/
    before the workflow runs (same mechanism as the Kontext path).

    Args:
        motion_prompt: Natural-language description of the whole animation
            (e.g. "a knight swinging a sword in a side-view attack, game
            sprite animation"). AnimateDiff takes ONE prompt for the whole
            sequence, not per-frame — motion emerges from the motion module,
            not from prompt variation.
        reference_image_name: Filename of the hero PNG in /comfyui/input/.
        checkpoint: SD1.5 base model on the volume under checkpoints/.
            Default toonyou_beta6 matches sprite-me's cartoon/indie aesthetic.
        motion_module: AnimateDiff motion module filename under
            animatediff_models/. v3_sd15_mm is the current v3 SD1.5 default.
        ipadapter_preset: IPAdapterUnifiedLoader preset name. "PLUS (high
            strength)" uses ip-adapter-plus_sd15 which gives strong identity
            preservation; drop to "STANDARD" for looser conditioning.
        ipadapter_weight: Identity-conditioning strength (0.0-1.0+). 0.75 is
            a good sprite default — strong enough to lock character, loose
            enough to let motion register.
        width, height: Frame dimensions. SD1.5 native is 512x512; going
            higher increases VRAM + time non-linearly.
        frames: Batch size for the motion module. 16 is the v3 training
            context. The caller can trim this down post-generation.
        seed: KSampler seed — same seed each call for reproducibility.
        steps: Denoising steps. 25 is a sane SD1.5 default; drop to 20 if
            time-bound.
        cfg: Guidance scale. 7.5 is SD1.5 canonical; IP-Adapter at weight
            0.75 will compete with text, so don't drop below 6.
        sampler_name, scheduler: KSampler config. euler/normal is the
            community default for AnimateDiff v3 and works without tuning.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "IPAdapterUnifiedLoader",
            "inputs": {
                "model": ["1", 0],
                "preset": ipadapter_preset,
            },
        },
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image_name},
        },
        "4": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["2", 0],
                "ipadapter": ["2", 1],
                "image": ["3", 0],
                "weight": ipadapter_weight,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only",
            },
        },
        "5": {
            "class_type": "ADE_LoadAnimateDiffModel",
            "inputs": {"model_name": motion_module},
        },
        "6": {
            "class_type": "ADE_UseEvolvedSampling",
            "inputs": {
                "model": ["4", 0],
                "m_models": ["5", 0],
                "beta_schedule": "autoselect",
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": motion_prompt, "clip": ["1", 1]},
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["1", 1]},
        },
        "9": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": frames},
        },
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["6", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["9", 0],
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["10", 0], "vae": ["1", 2]},
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {"images": ["11", 0], "filename_prefix": "sprite-me-animatediff"},
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
