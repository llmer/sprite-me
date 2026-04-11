"""Tests for tools/animate.py and animation_presets.py.

Covers:
    - Preset resolution: each preset returns the right number of pose prompts
    - Frame cap: frames=N truncates a preset of length M>N
    - Custom prompt: overrides preset entirely, returns N copies
    - Unknown animation: falls back to a literal prompt with warning
    - Preservation suffix: always appended (preset or custom)

Not covered here (too heavy for unit tests):
    - The full animate_sprite flow with mocked RunPodClient
    - Kontext workflow submission
These would need integration-style tests against a live endpoint.
"""

from __future__ import annotations

from sprite_me.animation_presets import (
    ANIMATION_PRESETS,
    available_presets,
    preset_prompts,
)
from sprite_me.tools.animate import _CUSTOM_PROMPT_SUFFIX, _resolve_prompts


# ---------- animation_presets.py ----------


def test_available_presets_returns_all_seven():
    presets = available_presets()
    assert set(presets) == {"idle", "walk", "run", "attack", "jump", "cast", "death"}


def test_preset_walk_has_eight_frames():
    assert len(preset_prompts("walk")) == 8


def test_preset_idle_has_four_frames():
    assert len(preset_prompts("idle")) == 4


def test_preset_attack_jump_cast_death_have_six_frames():
    for name in ("attack", "jump", "cast", "death"):
        assert len(preset_prompts(name)) == 6, f"{name} should have 6 frames"


def test_preset_run_has_eight_frames():
    assert len(preset_prompts("run")) == 8


def test_unknown_preset_returns_none():
    assert preset_prompts("nonexistent") is None


def test_every_preset_prompt_has_preservation_suffix():
    """All preset frame prompts must end with the 'Keep ... unchanged.' clause
    so Kontext preserves character identity regardless of which preset is used.
    """
    for name, prompts in ANIMATION_PRESETS.items():
        for i, p in enumerate(prompts):
            assert "Keep the exact same character" in p, \
                f"{name} frame {i} missing preservation suffix"
            assert "unchanged" in p, \
                f"{name} frame {i} missing 'unchanged' keyword"


def test_every_preset_prompt_starts_with_change_the_pose():
    """Kontext edit prompts should start with the 'Change X' instruction
    form to match Kontext's training distribution.
    """
    for name, prompts in ANIMATION_PRESETS.items():
        for i, p in enumerate(prompts):
            assert p.startswith("Change the pose"), \
                f"{name} frame {i} doesn't start with 'Change the pose': {p[:60]}"


# ---------- tools/animate.py::_resolve_prompts ----------


def test_resolve_prompts_returns_preset_by_default():
    prompts = _resolve_prompts(animation="walk", custom_prompt=None, frames=None)
    assert len(prompts) == 8
    # Every walk frame should mention "side view" (canonical view for walk cycles)
    assert all("side view" in p for p in prompts)
    # At least one frame should describe a forward stride
    assert any("forward" in p for p in prompts)


def test_resolve_prompts_frames_caps_preset_length():
    """frames=3 should trim the 8-frame walk cycle to its first 3 entries."""
    prompts = _resolve_prompts(animation="walk", custom_prompt=None, frames=3)
    assert len(prompts) == 3
    # Should be the first 3 of the canonical preset
    full = preset_prompts("walk")
    assert prompts == full[:3]


def test_resolve_prompts_frames_larger_than_preset_does_not_stretch():
    """Asking for more frames than the preset has should return the preset as-is."""
    prompts = _resolve_prompts(animation="idle", custom_prompt=None, frames=100)
    # Idle has 4 frames; frames=100 shouldn't stretch or repeat it.
    assert len(prompts) == 4


def test_resolve_prompts_custom_prompt_overrides_preset():
    """custom_prompt overrides the preset entirely, even if animation is a valid preset."""
    prompts = _resolve_prompts(
        animation="walk",
        custom_prompt="sitting on a bench, reading a book",
        frames=2,
    )
    assert len(prompts) == 2
    for p in prompts:
        assert "sitting on a bench" in p
        assert _CUSTOM_PROMPT_SUFFIX in p
        # The walk preset's prompt text should NOT appear
        assert "left foot forward" not in p


def test_resolve_prompts_custom_prompt_defaults_to_one_frame():
    """If no frames specified with custom_prompt, return exactly one prompt."""
    prompts = _resolve_prompts(
        animation="irrelevant",
        custom_prompt="sword raised in triumph",
        frames=None,
    )
    assert len(prompts) == 1


def test_resolve_prompts_unknown_animation_falls_back():
    """Unknown animation + no custom prompt should fall back to a literal prompt."""
    prompts = _resolve_prompts(
        animation="somersault",
        custom_prompt=None,
        frames=3,
    )
    assert len(prompts) == 3
    for p in prompts:
        assert "somersault" in p
        assert _CUSTOM_PROMPT_SUFFIX in p


def test_resolve_prompts_custom_suffix_always_appended():
    """The preservation suffix should be present in custom prompts so callers
    don't have to repeat the 'keep the character the same' clause.
    """
    prompts = _resolve_prompts(
        animation="x",
        custom_prompt="crouching",
        frames=1,
    )
    assert _CUSTOM_PROMPT_SUFFIX in prompts[0]
    assert prompts[0].startswith("crouching")
