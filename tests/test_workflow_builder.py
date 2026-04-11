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


# ---------- Animate workflow (Kontext) ----------


def test_animate_workflow_uses_unet_loader():
    """Kontext uses UNETLoader + DualCLIPLoader + VAELoader, NOT CheckpointLoaderSimple.

    CheckpointLoaderSimple was the generate-flow pattern (combined fp8 checkpoint
    bundles model+clip+vae). Kontext uses split files.
    """
    nodes = build_animate_workflow(pose_prompt="walking forward")
    assert nodes["1"]["class_type"] == "UNETLoader"
    assert "flux1-dev-kontext" in nodes["1"]["inputs"]["unet_name"]
    assert nodes["2"]["class_type"] == "DualCLIPLoader"
    assert nodes["3"]["class_type"] == "VAELoader"


def test_animate_workflow_has_reference_latent():
    """ReferenceLatent is the Kontext mechanism that carries character identity."""
    nodes = build_animate_workflow(pose_prompt="attack swing")
    reference_latent_nodes = [
        (k, v) for k, v in nodes.items() if v.get("class_type") == "ReferenceLatent"
    ]
    assert len(reference_latent_nodes) == 1
    _, ref_node = reference_latent_nodes[0]
    # ReferenceLatent takes conditioning (from CLIPTextEncode) + latent (from VAEEncode)
    assert "conditioning" in ref_node["inputs"]
    assert "latent" in ref_node["inputs"]


def test_animate_workflow_embeds_pose_prompt():
    """The pose prompt goes into the CLIPTextEncode node unchanged."""
    pose = "Change the pose to walking forward, left foot lifted. Keep everything else."
    nodes = build_animate_workflow(pose_prompt=pose)
    clip_text_nodes = [
        v for v in nodes.values() if v.get("class_type") == "CLIPTextEncode"
    ]
    assert len(clip_text_nodes) == 1
    assert clip_text_nodes[0]["inputs"]["text"] == pose


def test_animate_workflow_ksampler_kontext_defaults():
    """KSampler must use Kontext-specific sampler settings (cfg=1, scheduler=simple)."""
    nodes = build_animate_workflow(pose_prompt="idle", seed=42, steps=20, guidance=2.5)
    ksampler = next(v for v in nodes.values() if v.get("class_type") == "KSampler")
    assert ksampler["inputs"]["cfg"] == 1.0
    assert ksampler["inputs"]["scheduler"] == "simple"
    assert ksampler["inputs"]["denoise"] == 1.0
    assert ksampler["inputs"]["seed"] == 42
    assert ksampler["inputs"]["steps"] == 20

    # FluxGuidance holds the actual guidance scale
    flux_guidance = next(v for v in nodes.values() if v.get("class_type") == "FluxGuidance")
    assert flux_guidance["inputs"]["guidance"] == 2.5


def test_animate_workflow_load_image_uses_hero_png():
    """LoadImage reads hero.png from /comfyui/input/ (uploaded by the handler)."""
    nodes = build_animate_workflow(pose_prompt="walking")
    load_image = next(v for v in nodes.values() if v.get("class_type") == "LoadImage")
    assert load_image["inputs"]["image"] == "hero.png"


def test_animate_workflow_custom_reference_name():
    """reference_image_name parameter allows overriding the filename."""
    nodes = build_animate_workflow(pose_prompt="x", reference_image_name="my_char.png")
    load_image = next(v for v in nodes.values() if v.get("class_type") == "LoadImage")
    assert load_image["inputs"]["image"] == "my_char.png"


def test_animate_workflow_returns_raw_node_dict():
    nodes = build_animate_workflow(pose_prompt="idle")
    assert "prompt" not in nodes
    assert all(k.isdigit() for k in nodes.keys())


# ---------- Remove background workflow ----------


def test_remove_background_workflow_shape():
    nodes = build_remove_background_workflow("base64data")
    assert "prompt" not in nodes
    assert "1" in nodes
    assert "2" in nodes
    assert "3" in nodes
    assert nodes["1"]["inputs"]["image"] == "base64data"
