"""Tests for ComfyUI workflow construction."""

from sprite_me.inference.workflow_builder import (
    build_animate_workflow,
    build_generate_workflow,
    build_remove_background_workflow,
)


def test_generate_workflow_injects_prompt():
    wf = build_generate_workflow(prompt="warrior with sword", seed=42)
    nodes = wf["prompt"]

    # Prompt should include the trigger word and user text
    assert "GRPZA" in nodes["3"]["inputs"]["text"]
    assert "warrior with sword" in nodes["3"]["inputs"]["text"]
    assert "white background" in nodes["3"]["inputs"]["text"]

    # Seed should be set on the sampler
    assert nodes["6"]["inputs"]["seed"] == 42


def test_generate_workflow_custom_dimensions():
    wf = build_generate_workflow(prompt="icon", width=256, height=256)
    nodes = wf["prompt"]
    assert nodes["5"]["inputs"]["width"] == 256
    assert nodes["5"]["inputs"]["height"] == 256


def test_generate_workflow_lora_strength():
    wf = build_generate_workflow(prompt="slime", lora_strength=0.6)
    nodes = wf["prompt"]
    assert nodes["2"]["inputs"]["strength_model"] == 0.6
    assert nodes["2"]["inputs"]["strength_clip"] == 0.6


def test_animate_workflow_scales_width_by_frames():
    wf = build_animate_workflow(
        reference_image_b64="",
        animation_prompt="walk cycle",
        frames=6,
        width=512,
        height=512,
    )
    nodes = wf["prompt"]
    # Wide latent should be 512 * 6 = 3072
    assert nodes["5"]["inputs"]["width"] == 3072
    assert nodes["5"]["inputs"]["height"] == 512


def test_animate_workflow_includes_frame_count_in_prompt():
    wf = build_animate_workflow(
        reference_image_b64="",
        animation_prompt="attack",
        frames=8,
    )
    prompt_text = wf["prompt"]["3"]["inputs"]["text"]
    assert "8 frames" in prompt_text
    assert "attack" in prompt_text
    assert "sprite sheet" in prompt_text


def test_remove_background_workflow_shape():
    wf = build_remove_background_workflow("base64data")
    assert "prompt" in wf
    assert "1" in wf["prompt"]
    assert "2" in wf["prompt"]
    assert "3" in wf["prompt"]
