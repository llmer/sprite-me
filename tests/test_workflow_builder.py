"""Tests for ComfyUI workflow construction.

Builders return the raw ComfyUI node graph (no {"prompt": ...} wrapper)
because the runpod/worker-comfyui handler wraps it before submitting.
"""

from sprite_me.inference.workflow_builder import (
    build_animate_workflow,
    build_generate_workflow,
    build_remove_background_workflow,
)


def test_generate_workflow_injects_prompt():
    nodes = build_generate_workflow(prompt="warrior with sword", seed=42)

    # Prompt should include the trigger word and user text
    assert "GRPZA" in nodes["3"]["inputs"]["text"]
    assert "warrior with sword" in nodes["3"]["inputs"]["text"]
    assert "white background" in nodes["3"]["inputs"]["text"]

    # Seed should be set on the sampler
    assert nodes["6"]["inputs"]["seed"] == 42


def test_generate_workflow_returns_raw_node_dict():
    """Ensure the builder does NOT wrap nodes in {"prompt": ...}."""
    nodes = build_generate_workflow(prompt="test")
    assert "prompt" not in nodes  # no wrapper
    # All top-level keys should be node IDs (numeric strings)
    assert all(k.isdigit() for k in nodes.keys())
    # Each node has the expected shape
    for node in nodes.values():
        assert "class_type" in node
        assert "inputs" in node


def test_generate_workflow_custom_dimensions():
    nodes = build_generate_workflow(prompt="icon", width=256, height=256)
    assert nodes["5"]["inputs"]["width"] == 256
    assert nodes["5"]["inputs"]["height"] == 256


def test_generate_workflow_lora_strength():
    nodes = build_generate_workflow(prompt="slime", lora_strength=0.6)
    assert nodes["2"]["inputs"]["strength_model"] == 0.6
    assert nodes["2"]["inputs"]["strength_clip"] == 0.6


def test_animate_workflow_uses_batch_size():
    """Animation uses batch_size=N, not a wide side-by-side latent.

    Each frame is a full centered character at the per-frame dimensions.
    """
    nodes = build_animate_workflow(
        reference_image_b64="",
        animation_prompt="walk cycle",
        frames=6,
        width=512,
        height=512,
    )
    # Width stays at the per-frame size, batch_size encodes the frame count.
    assert nodes["5"]["inputs"]["width"] == 512
    assert nodes["5"]["inputs"]["height"] == 512
    assert nodes["5"]["inputs"]["batch_size"] == 6


def test_animate_workflow_embeds_animation_prompt():
    nodes = build_animate_workflow(
        reference_image_b64="",
        animation_prompt="attack swing",
        frames=4,
    )
    prompt_text = nodes["3"]["inputs"]["text"]
    assert "GRPZA" in prompt_text
    assert "attack swing" in prompt_text
    assert "game asset" in prompt_text
    assert "white background" in prompt_text
    # batch_size=N controls frame count, not the prompt text
    assert nodes["5"]["inputs"]["batch_size"] == 4


def test_animate_workflow_returns_raw_node_dict():
    nodes = build_animate_workflow(reference_image_b64="", animation_prompt="idle")
    assert "prompt" not in nodes
    assert all(k.isdigit() for k in nodes.keys())


def test_remove_background_workflow_shape():
    nodes = build_remove_background_workflow("base64data")
    assert "prompt" not in nodes
    assert "1" in nodes
    assert "2" in nodes
    assert "3" in nodes
    assert nodes["1"]["inputs"]["image"] == "base64data"
