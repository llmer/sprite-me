"""LoRA registry for the generate_sprite path.

Each profile owns a filename (on the network volume), an optional trigger
word, a prompt template, a default strength, and metadata that lets the
MCP tool (or a human) pick the right one. Callers select a profile by
key — no hardcoded filenames at the call site.

Adding a new LoRA:
    1. Download the .safetensors locally into ./models/loras/
    2. `uv run scripts/sync_models.py upload <path> --prefix models/loras`
    3. Add a profile entry below with `best_for` tags and a description
    4. Add a test in tests/test_workflow_builder.py that verifies the
       profile key + prompt shape
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoraProfile:
    name: str
    """Filename under /runpod-volume/models/loras/ on the network volume."""

    trigger: str
    """Trigger word or phrase. "" if the LoRA was trained without one
    (e.g. BLIP-captioned datasets)."""

    prompt_template: str
    """f-string-style template with {trigger} and {prompt} placeholders.
    The template owns the LoRA's preferred prompt shape — e.g. the
    game-assets LoRA wants "GRPZA, <subject>, white background, game asset"
    while the BLIP-captioned ones want "a pixel art sprite of <subject>"."""

    default_strength: float
    """LoRA loader strength (both model and clip). Overridable per call."""

    description: str
    """One-line summary of what this LoRA produces. Shown in the MCP tool
    docstring so agents can pick the right key."""

    best_for: tuple[str, ...]
    """Tags hinting at the best use cases: "characters", "trainers",
    "generic-assets", "top-down", etc. Non-authoritative — an agent can
    still use any LoRA for any prompt."""

    civitai_version_id: int | None = None
    """Civitai model version ID for `bootstrap_loras.py` to re-download.
    None for LoRAs not sourced from Civitai (use `source_url` instead)."""

    source_url: str | None = None
    """Direct download URL for non-Civitai LoRAs (e.g. HuggingFace
    /resolve/main/...). Used when civitai_version_id is None. None if
    the LoRA has no public source (manual re-upload required on rebuild)."""


LORAS: dict[str, LoraProfile] = {
    "cartoon-vector": LoraProfile(
        name="flux-2d-game-assets.safetensors",
        trigger="GRPZA",
        prompt_template="{trigger}, {prompt}, white background, game asset",
        default_strength=0.85,
        description="Smooth cartoon/vector illustration — heavy black outlines, "
                    "flat colors with gradient shading, polished casual-game "
                    "aesthetic. NOT pixel art. Good for heroic characters, "
                    "mascots, polished storybook looks.",
        best_for=("cartoon", "vector", "heroes", "casual-game", "polished"),
        source_url=(
            "https://huggingface.co/gokaygokay/Flux-2D-Game-Assets-LoRA/"
            "resolve/main/game_assets_v3.safetensors"
        ),
    ),
    "pixel-indie": LoraProfile(
        name="trainer-sprites-gen5.safetensors",
        trigger="A pixelart drawing",
        prompt_template="{trigger} of {prompt}",
        default_strength=1.0,
        description="Clean modern pixel art — visible pixel grid, vibrant "
                    "saturated colors, JRPG / indie-Steam aesthetic. Works "
                    "across humanoid and creature subjects despite being "
                    "trained on Pokemon-era trainer sprites. Best balance of "
                    "pixel crispness and subject flexibility.",
        best_for=("pixel-art", "indie", "jrpg", "characters", "vibrant"),
        civitai_version_id=788326,
    ),
    "pixel-retro": LoraProfile(
        name="pixel-game-assets-dever.safetensors",
        trigger="dvr-pixel-flux",
        prompt_template="{trigger}, {prompt}, illustration, cartoon",
        default_strength=1.0,
        description="Retro small-canvas pixel art — 8/16-bit era crunchy "
                    "palette, Zelda/NES/SNES feel. Outputs often include a "
                    "scene background rather than isolated characters, so "
                    "expect to post-crop. Best for authentic retro looks. "
                    "Avoid 'realistic' or 'photograph' in prompts.",
        best_for=("pixel-art", "retro", "8-bit", "16-bit", "nes", "snes"),
        civitai_version_id=1058316,
    ),
    "top-down": LoraProfile(
        name="top-down-pixel-art.safetensors",
        trigger="",
        prompt_template="top-down pixel art of {prompt}",
        default_strength=1.0,
        description="Bird's-eye / top-down perspective scenes, like Zelda, "
                    "Stardew Valley, or classic ARPGs. Produces full scene "
                    "framings with environment around the subject, NOT "
                    "isolated sprites. Use for tiles, environments, or when "
                    "the user explicitly wants a top-down view.",
        best_for=("top-down", "tiles", "environments", "scenes"),
        civitai_version_id=1298288,
    ),
}

DEFAULT_LORA = "cartoon-vector"


def get_profile(key: str) -> LoraProfile:
    if key not in LORAS:
        raise ValueError(
            f"Unknown LoRA profile {key!r}. Available: {sorted(LORAS)}"
        )
    return LORAS[key]


def format_prompt(profile: LoraProfile, prompt: str) -> str:
    text = profile.prompt_template.format(trigger=profile.trigger, prompt=prompt)
    if not profile.trigger:
        text = text.replace(", ,", ",").lstrip(", ")
    return text


def describe_all() -> str:
    """Human-readable multi-line list of LoRAs for use in docstrings."""
    lines = []
    for key, p in LORAS.items():
        tags = ", ".join(p.best_for)
        lines.append(f'  - "{key}": {p.description} [best for: {tags}]')
    return "\n".join(lines)
