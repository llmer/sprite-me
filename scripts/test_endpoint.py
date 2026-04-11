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

from sprite_me.animation_presets import available_presets, preset_prompts  # noqa: E402
from sprite_me.inference.runpod_client import RunPodClient, RunPodError  # noqa: E402
from sprite_me.inference.workflow_builder import (  # noqa: E402
    build_animate_workflow,
    build_generate_workflow,
)
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
    p.add_argument("--lora-strength", type=float, default=0.85)
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
        "--preset",
        default="walk",
        choices=available_presets(),
        help="[animate] Animation preset. Ignored if --prompt is provided.",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=None,
        help="[animate] Cap frame count (default: preset length, or 1 for --prompt)",
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
) -> bytes | None:
    """One Kontext frame submission with the hero uploaded."""
    workflow = build_animate_workflow(
        pose_prompt=pose_prompt,
        reference_image_name="hero.png",
        seed=seed,
        steps=steps,
        guidance=guidance,
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


async def run_animate(args: argparse.Namespace, client: RunPodClient) -> list[bytes]:
    if not args.reference_asset:
        raise ValueError("--animate requires --reference-asset <path-to-hero-png>")
    hero_path = _REPO_ROOT / args.reference_asset if not Path(args.reference_asset).is_absolute() else Path(args.reference_asset)
    if not hero_path.exists():
        raise FileNotFoundError(f"Hero PNG not found: {hero_path}")

    hero_bytes = hero_path.read_bytes()
    hero_b64 = base64.b64encode(hero_bytes).decode()
    print(f"Hero: {hero_path} ({len(hero_bytes)} bytes)")

    # Resolve frame list
    explicit_prompt = args.prompt and args.prompt != "knight with longsword"
    if explicit_prompt:
        n = args.frames or 1
        pose_prompts = [
            f"{args.prompt}. Keep the exact same character, clothing, armor, weapon, "
            f"color palette, and art style unchanged. White background."
        ] * n
    else:
        preset = preset_prompts(args.preset) or []
        if args.frames and args.frames < len(preset):
            pose_prompts = preset[:args.frames]
        else:
            pose_prompts = preset
    print(f"Preset: {args.preset if not explicit_prompt else '(custom)'}, frames: {len(pose_prompts)}")

    frames: list[bytes] = []
    for i, pp in enumerate(pose_prompts):
        t = time.time()
        print(f"  frame {i+1}/{len(pose_prompts)}: {pp[:60]}...")
        frame = await _generate_animate_frame(
            client, pp, hero_b64, seed=args.seed, steps=args.steps, guidance=args.guidance,
        )
        if frame is None:
            print(f"    FAILED (skipping)")
            continue
        frames.append(frame)
        print(f"    done in {time.time()-t:.1f}s ({len(frame)} bytes)")
    return frames


async def main() -> int:
    args = _parse_args()
    client = RunPodClient(poll_interval=3.0, timeout_seconds=args.timeout)

    if not client.endpoint_id:
        print("ERROR: SPRITE_ME_RUNPOD_ENDPOINT_ID not set in .env", file=sys.stderr)
        return 1

    print(f"Endpoint: {client.endpoint_id}")
    mode = "animate" if args.animate else "generate"
    print(
        f"Mode: {mode} | seed={args.seed} steps={args.steps} "
        f"guidance={args.guidance}"
    )
    t0 = time.time()
    try:
        if args.animate:
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

    out = _REPO_ROOT / args.out
    if args.animate and len(images) > 1:
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
