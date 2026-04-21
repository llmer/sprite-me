"""Registry of the model assets needed by the AnimateDiff animation path.

Mirrors the LoRA registry pattern (`sprite_me.loras`): a small dataclass
per asset describing where it lives on the network volume and how to
re-download it if the volume is wiped. `scripts/bootstrap_animatediff.py`
reads this as the source of truth.

Unlike the LoRA registry, these are infrastructure deps — there's no
"profile picker" at call time. The workflow builder hardcodes the same
filenames listed here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnimateDiffAsset:
    name: str
    """Filename on the network volume (and in local cache)."""

    remote_prefix: str
    """Directory under /runpod-volume/models/ where this file lives.
    Must match a key in extra_model_paths.yaml."""

    local_subdir: str
    """Directory under ./models/ for the local cache. Mirrors remote_prefix."""

    source_url: str
    """Direct HTTPS download URL. All AnimateDiff assets are on HuggingFace
    or Civitai public URLs — none need auth tokens."""

    expected_size_mb: int
    """Rough expected size in MB. Used as a sanity check after download;
    exact byte size is whatever the server serves."""

    description: str
    """One-line summary for logs."""


ANIMATEDIFF_ASSETS: dict[str, AnimateDiffAsset] = {
    "toonyou": AnimateDiffAsset(
        name="toonyou_beta6.safetensors",
        remote_prefix="checkpoints",
        local_subdir="checkpoints",
        # ToonYou beta6 on HuggingFace mirror. Maintained by frankjoshua;
        # aligns with sprite-me's cartoon/indie LoRA aesthetics.
        source_url=(
            "https://huggingface.co/frankjoshua/toonyou_beta6/resolve/main/"
            "toonyou_beta6.safetensors"
        ),
        expected_size_mb=2000,
        description="SD1.5 base checkpoint — bold cartoon/anime style",
    ),
    "animatediff_v3_mm": AnimateDiffAsset(
        name="v3_sd15_mm.ckpt",
        remote_prefix="animatediff_models",
        local_subdir="animatediff_models",
        source_url=(
            "https://huggingface.co/guoyww/animatediff/resolve/main/"
            "v3_sd15_mm.ckpt"
        ),
        expected_size_mb=1670,
        description="AnimateDiff v3 motion module for SD1.5",
    ),
    "ipadapter_plus_sd15": AnimateDiffAsset(
        name="ip-adapter-plus_sd15.safetensors",
        remote_prefix="ipadapter",
        local_subdir="ipadapter",
        source_url=(
            "https://huggingface.co/h94/IP-Adapter/resolve/main/models/"
            "ip-adapter-plus_sd15.safetensors"
        ),
        expected_size_mb=100,
        description="IP-Adapter Plus SD1.5 — strong image-prompt conditioning",
    ),
    "clip_vision_h14": AnimateDiffAsset(
        name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        remote_prefix="clip_vision",
        local_subdir="clip_vision",
        # IP-Adapter bundles the image-encoder weights in its own repo.
        # Kept as the canonical upstream that cubiq's README points to.
        source_url=(
            "https://huggingface.co/h94/IP-Adapter/resolve/main/models/"
            "image_encoder/model.safetensors"
        ),
        expected_size_mb=2500,
        description="CLIP Vision H/14 — image encoder for IP-Adapter Plus SD1.5",
    ),
}
