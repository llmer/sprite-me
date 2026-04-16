"""Tests for tools/animate.py after the v0.5.0 clean-break refactor.

Covers:
    - Clean-break error: old `animation=` / `custom_prompt=` / `frames=` args
      raise ValueError with a skill-file pointer
    - Empty pose_prompts list is rejected
    - Preservation suffix is auto-appended to prompts that don't already
      contain "unchanged"
    - Prompts that DO contain "unchanged" are passed through as-is

Heavyweight concerns (the full Kontext round-trip, the RunPod submission
loop, and the spritesheet assembly) are not covered here — they need
integration tests against a live endpoint.
"""

from __future__ import annotations

import asyncio

import pytest

from sprite_me.tools.animate import _apply_suffix, animate_sprite


# ---------- _apply_suffix ----------


def test_apply_suffix_appends_when_missing():
    result = _apply_suffix("Change the pose to walking forward")
    assert "unchanged" in result.lower()
    assert result.startswith("Change the pose to walking forward")


def test_apply_suffix_idempotent_when_unchanged_present():
    already = "Change the pose to walking. Keep everything unchanged."
    result = _apply_suffix(already)
    assert result == already
    # Preservation clause appears exactly once
    assert result.lower().count("unchanged") == 1


def test_apply_suffix_strips_trailing_period_before_appending():
    result = _apply_suffix("Change the pose to standing still.")
    # No double period
    assert ".." not in result


# ---------- animate_sprite clean-break errors ----------


def test_animate_sprite_rejects_old_animation_param():
    """The old `animation` preset-name arg must hard-error with a skill
    file pointer, not silently translate.
    """
    with pytest.raises(ValueError, match="no longer accepts 'animation'"):
        asyncio.run(animate_sprite(
            asset_id="abc",
            pose_prompts=["Change pose to walking"],
            animation="walk",
        ))


def test_animate_sprite_rejects_old_custom_prompt_param():
    with pytest.raises(ValueError, match="no longer accepts 'custom_prompt'"):
        asyncio.run(animate_sprite(
            asset_id="abc",
            pose_prompts=["Change pose to walking"],
            custom_prompt="some old custom prompt",
        ))


def test_animate_sprite_rejects_old_frames_param():
    with pytest.raises(ValueError, match="no longer accepts 'frames'"):
        asyncio.run(animate_sprite(
            asset_id="abc",
            pose_prompts=["Change pose to walking"],
            frames=6,
        ))


def test_animate_sprite_rejects_old_auto_enhance_param():
    with pytest.raises(ValueError, match="no longer accepts 'auto_enhance'"):
        asyncio.run(animate_sprite(
            asset_id="abc",
            pose_prompts=["Change pose to walking"],
            auto_enhance=True,
        ))


def test_animate_sprite_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="Unknown kwargs"):
        asyncio.run(animate_sprite(
            asset_id="abc",
            pose_prompts=["Change pose to walking"],
            wobble=True,
        ))


def test_animate_sprite_rejects_empty_pose_prompts():
    with pytest.raises(ValueError, match="at least one entry"):
        asyncio.run(animate_sprite(
            asset_id="abc",
            pose_prompts=[],
        ))
