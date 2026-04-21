"""Smoke test for the sprite-me RunPod endpoint.

Two modes:
    - generate (default): text prompt → single pixel-art sprite
    - animate:            hero reference + preset → multi-frame sprite sheet via Kontext

Usage:
    # Generate
    uv run scripts/test_endpoint.py
    uv run scripts/test_endpoint.py --prompt "green slime"
    uv run scripts/test_endpoint.py --prompt "wizard" --seed 1 --out wizard.png

    # Animate (uses the generate + animate toolchain end-to-end)
    uv run scripts/test_endpoint.py --animate --reference-asset fresh_hero.png --preset walk
    uv run scripts/test_endpoint.py --animate --reference-asset hero.png --prompt "raising sword" --frames 3

Requires SPRITE_ME_RUNPOD_API_KEY and SPRITE_ME_RUNPOD_ENDPOINT_ID in .env.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sprite_me.inference.runpod_client import RunPodClient, RunPodError  # noqa: E402
from sprite_me.inference.workflow_builder import (  # noqa: E402
    build_animate_workflow,
    build_animate_workflow_animatediff,
    build_generate_workflow,
)
from sprite_me.loras import DEFAULT_LORA, LORAS  # noqa: E402
from sprite_me.processing.spritesheet import assemble_spritesheet  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for sprite-me RunPod endpoint")
    p.add_argument(
        "--prompt",
        default="knight with longsword",
        help="Generate mode: subject prompt. Animate mode: single pose description (overrides --preset)",
    )
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument(
        "--lora",
        default=DEFAULT_LORA,
        choices=sorted(LORAS.keys()),
        help="LoRA profile key (generate mode only)",
    )
    p.add_argument(
        "--lora-strength",
        type=float,
        default=None,
        help="Override profile default strength",
    )
    p.add_argument(
        "--out",
        default="test_output.png",
        help="Output filename (written to repo root)",
    )
    p.add_argument("--timeout", type=float, default=1800.0)

    # Animate-mode flags
    p.add_argument("--animate", action="store_true", help="Run Kontext animate flow")
    p.add_argument(
        "--reference-asset",
        help="[animate] Hero PNG path. Required when --animate is set.",
    )
    p.add_argument(
        "--pose-prompts",
        help="[animate] Path to a JSON file containing a list of pose prompt "
             "strings, one per frame. If omitted, uses --prompt as a single "
             "pose description.",
    )
    p.add_argument(
        "--denoise",
        type=float,
        default=1.0,
        help="[animate] Denoise strength 0-1. Default 1.0 (action anims). "
             "Drop to 0.5-0.6 for idle/breathing loops.",
    )
    p.add_argument(
        "--no-chain",
        action="store_true",
        help="[animate] Disable frame chaining (always reference the hero)",
    )
    p.add_argument(
        "--anchor-frame",
        action="store_true",
        help="[animate] Frame 0 references hero; frames 1..N reference "
             "frame 0's output. Bounds VAE drift to one round-trip. "
             "Overrides --no-chain.",
    )
    p.add_argument(
        "--seed-per-frame",
        action="store_true",
        help="[animate] Increment seed per frame (more RNG variation)",
    )
    # AnimateDiff-mode flags (v0.6 path)
    p.add_argument(
        "--animatediff",
        action="store_true",
        help="[animate] Use the SD1.5 + AnimateDiff v3 + IP-Adapter workflow "
             "instead of the Kontext path. Requires --reference-asset and "
             "--motion-description.",
    )
    p.add_argument(
        "--motion-description",
        help="[animatediff] Single motion prompt for the whole sequence, "
             "e.g. 'a knight swinging a sword in a side-view attack, game "
             "sprite animation'. Required when --animatediff is set.",
    )
    p.add_argument(
        "--frame-count",
        type=int,
        default=16,
        help="[animatediff] Number of frames to sample in one AnimateDiff "
             "batch. Default 16 (native v3 context length).",
    )
    p.add_argument(
        "--ipa-weight",
        type=float,
        default=0.0,
        help="[animatediff] IP-Adapter identity-conditioning weight (0.0-1.0+). "
             "Default 0.0 (disabled) — empirically any IPA weight >= 0.1 "
             "freezes the motion module. Raise at your own motion risk.",
    )
    p.add_argument(
        "--ipa-end-at",
        type=float,
        default=1.0,
        help="[animatediff] Sampling fraction at which to stop applying IP-Adapter "
             "conditioning (0.0-1.0). Dropping to ~0.6 lets AnimateDiff own the "
             "final denoising steps, unlocking motion while still anchoring "
             "identity early. Default 1.0 (full duration).",
    )
    p.add_argument(
        "--ad-checkpoint",
        default="toonyou_beta6.safetensors",
        help="[animatediff] SD1.5 checkpoint filename under checkpoints/.",
    )
    p.add_argument(
        "--ad-motion-module",
        default="mm_sd_v15_v2.ckpt",
        help="[animatediff] Motion module filename under animatediff_models/. "
             "mm_sd_v15_v2 is the default — empirically produces ~8x more "
             "inter-frame motion than v3_sd15_mm on this worker.",
    )
    p.add_argument(
        "--ad-loras",
        default=None,
        help="[animatediff] Comma-separated LoRA spec: "
             "name1:model_str:clip_str,name2:model_str:clip_str. "
             "Empty string disables all LoRAs. Omit to use the builder default.",
    )
    p.add_argument(
        "--pixelate",
        action="store_true",
        help="[animatediff] Run each returned frame through the pixelate "
             "pipeline (downsample -> palette quantize -> nearest-neighbor "
             "upscale) before writing to disk. Gives the pixel-art look.",
    )
    p.add_argument(
        "--pixel-size",
        type=int,
        default=96,
        help="[animatediff] Target pixel grid for --pixelate. 64 = chunky "
             "retro, 96 = detailed sprite, 128 = smooth-ish. Default 96.",
    )
    p.add_argument(
        "--palette-size",
        type=int,
        default=24,
        help="[animatediff] Palette quantization count for --pixelate. "
             "16 = classic NES-era, 24 = SNES-ish, 32 = GBA-ish. Default 24.",
    )
    return p.parse_args()


async def run_generate(args: argparse.Namespace, client: RunPodClient) -> list[bytes]:
    workflow = build_generate_workflow(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        lora=args.lora,
        lora_strength=args.lora_strength,
    )
    return await client.generate(workflow)


async def _generate_animate_frame(
    client: RunPodClient,
    pose_prompt: str,
    hero_b64: str,
    seed: int,
    steps: int,
    guidance: float,
    denoise: float = 0.7,
) -> bytes | None:
    """One Kontext frame submission with the hero uploaded."""
    workflow = build_animate_workflow(
        pose_prompt=pose_prompt,
        reference_image_name="hero.png",
        seed=seed,
        steps=steps,
        guidance=guidance,
        denoise=denoise,
    )
    resp = await client.client.post(
        f"{client.base_url}/run",
        json={
            "input": {
                "workflow": workflow,
                "images": [{"name": "hero.png", "image": hero_b64}],
            }
        },
    )
    resp.raise_for_status()
    job_id = resp.json()["id"]
    payload = await client.wait_for_result(job_id)
    images = RunPodClient._extract_images(payload.get("output", {}))
    return images[0] if images else None


async def run_animatediff(args: argparse.Namespace, client: RunPodClient) -> list[bytes]:
    """Single-shot AnimateDiff + IP-Adapter batch. Returns N frames in one submit."""
    if not args.reference_asset:
        raise ValueError("--animatediff requires --reference-asset <path-to-hero-png>")
    if not args.motion_description:
        raise ValueError("--animatediff requires --motion-description \"<prompt>\"")
    hero_path = _REPO_ROOT / args.reference_asset if not Path(args.reference_asset).is_absolute() else Path(args.reference_asset)
    if not hero_path.exists():
        raise FileNotFoundError(f"Hero PNG not found: {hero_path}")

    hero_bytes = hero_path.read_bytes()
    hero_b64 = base64.b64encode(hero_bytes).decode()
    print(f"Hero: {hero_path} ({len(hero_bytes)} bytes)")
    print(
        f"Motion: '{args.motion_description}' | frames: {args.frame_count} "
        f"| ipa_weight: {args.ipa_weight} | ckpt: {args.ad_checkpoint}"
    )

    # Parse --ad-loras into the workflow's expected list-of-tuples. Empty string
    # means "no LoRAs"; None means "use the builder's default stack".
    kwargs: dict = {}
    if args.ad_loras is not None:
        if args.ad_loras == "":
            kwargs["loras"] = ()
        else:
            spec = []
            for entry in args.ad_loras.split(","):
                parts = entry.strip().split(":")
                if len(parts) != 3:
                    raise ValueError(f"Bad --ad-loras entry: {entry!r}")
                spec.append((parts[0], float(parts[1]), float(parts[2])))
            kwargs["loras"] = tuple(spec)

    workflow = build_animate_workflow_animatediff(
        motion_prompt=args.motion_description,
        reference_image_name="hero.png",
        checkpoint=args.ad_checkpoint,
        motion_module=args.ad_motion_module,
        ipadapter_weight=args.ipa_weight,
        ipadapter_end_at=args.ipa_end_at,
        width=args.width,
        height=args.height,
        frames=args.frame_count,
        seed=args.seed,
        steps=args.steps,
        **kwargs,
    )
    t = time.time()
    resp = await client.client.post(
        f"{client.base_url}/run",
        json={
            "input": {
                "workflow": workflow,
                "images": [{"name": "hero.png", "image": hero_b64}],
            }
        },
    )
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"  submitted {job_id}, waiting...")
    payload = await client.wait_for_result(job_id)
    frames = RunPodClient._extract_images(payload.get("output", {}))
    print(f"  done in {time.time() - t:.1f}s, got {len(frames)} frame(s)")
    return frames


async def run_animate(args: argparse.Namespace, client: RunPodClient) -> list[bytes]:
    if not args.reference_asset:
        raise ValueError("--animate requires --reference-asset <path-to-hero-png>")
    hero_path = _REPO_ROOT / args.reference_asset if not Path(args.reference_asset).is_absolute() else Path(args.reference_asset)
    if not hero_path.exists():
        raise FileNotFoundError(f"Hero PNG not found: {hero_path}")

    hero_bytes = hero_path.read_bytes()
    hero_b64 = base64.b64encode(hero_bytes).decode()
    print(f"Hero: {hero_path} ({len(hero_bytes)} bytes)")

    # Resolve frame list: either a JSON file of prompts, or a single prompt
    # via --prompt that becomes one frame.
    if args.pose_prompts:
        import json as _json
        pose_prompts = _json.loads(Path(args.pose_prompts).read_text())
        if not isinstance(pose_prompts, list) or not all(isinstance(p, str) for p in pose_prompts):
            raise ValueError("--pose-prompts must be a JSON list of strings")
    else:
        pose_prompts = [args.prompt]
    chain = not args.no_chain and not args.anchor_frame
    mode = "anchor-frame" if args.anchor_frame else ("chain" if chain else "no-chain")
    print(f"Frames: {len(pose_prompts)} | denoise: {args.denoise} | mode: {mode} | seed-per-frame: {args.seed_per_frame}")

    current_ref_b64 = hero_b64
    anchor_ref_b64: str | None = None
    frames: list[bytes] = []
    _preservation = (
        "Keep the exact same character, clothing, armor, weapon, "
        "color palette, and art style unchanged. White background."
    )
    for i, pp in enumerate(pose_prompts):
        if "unchanged" not in pp.lower():
            pp = f"{pp.rstrip('. ')}. {_preservation}"
        t = time.time()
        frame_seed = args.seed + i if args.seed_per_frame else args.seed
        print(f"  frame {i+1}/{len(pose_prompts)}: seed={frame_seed} {pp[:50]}...")
        frame = await _generate_animate_frame(
            client, pp, current_ref_b64, seed=frame_seed, steps=args.steps,
            guidance=args.guidance, denoise=args.denoise,
        )
        if frame is None:
            print(f"    FAILED (skipping)")
            continue
        frames.append(frame)
        if chain:
            current_ref_b64 = base64.b64encode(frame).decode()
        elif args.anchor_frame and i == 0:
            anchor_ref_b64 = base64.b64encode(frame).decode()
            current_ref_b64 = anchor_ref_b64
        print(f"    done in {time.time()-t:.1f}s ({len(frame)} bytes)")
    return frames


async def main() -> int:
    args = _parse_args()
    client = RunPodClient(poll_interval=3.0, timeout_seconds=args.timeout)

    if not client.endpoint_id:
        print("ERROR: SPRITE_ME_RUNPOD_ENDPOINT_ID not set in .env", file=sys.stderr)
        return 1

    print(f"Endpoint: {client.endpoint_id}")
    if args.animatediff:
        mode = "animatediff"
    elif args.animate:
        mode = "animate"
    else:
        mode = "generate"
    print(
        f"Mode: {mode} | seed={args.seed} steps={args.steps} "
        f"guidance={args.guidance}"
    )
    t0 = time.time()
    try:
        if args.animatediff:
            images = await run_animatediff(args, client)
        elif args.animate:
            images = await run_animate(args, client)
        else:
            print(f"prompt='{args.prompt}' {args.width}x{args.height} lora_strength={args.lora_strength}")
            images = await run_generate(args, client)
    except (RunPodError, ValueError, FileNotFoundError) as e:
        print(f"Error after {time.time() - t0:.1f}s: {e}", file=sys.stderr)
        await client.close()
        return 2

    elapsed = time.time() - t0
    print(f"Got {len(images)} image(s) in {elapsed:.1f}s")

    if not images:
        print("ERROR: no images returned", file=sys.stderr)
        await client.close()
        return 3

    if args.animatediff and args.pixelate:
        from sprite_me.processing.palette import pixelate as _pixelate
        images = [
            _pixelate(img, target_size=args.pixel_size,
                      palette_size=args.palette_size, upscale=True)
            for img in images
        ]
        print(
            f"Pixelated {len(images)} frame(s) -> {args.pixel_size}px, "
            f"{args.palette_size}-color palette"
        )

    out = _REPO_ROOT / args.out
    multi_frame = (args.animate or args.animatediff) and len(images) > 1
    if multi_frame:
        stem = out.stem
        for i, frame_data in enumerate(images):
            (_REPO_ROOT / f"{stem}_frame{i:02d}.png").write_bytes(frame_data)
        sheet = assemble_spritesheet(images)
        out.write_bytes(sheet)
        print(f"Saved {len(images)} frame(s) to {stem}_frame*.png")
        print(f"Saved assembled sheet to {out} ({len(sheet)} bytes)")
    else:
        out.write_bytes(images[0])
        print(f"Saved {out} ({len(images[0])} bytes)")

    await client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
