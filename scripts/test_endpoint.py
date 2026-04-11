"""Smoke test for the sprite-me RunPod endpoint.

Submits a job to the RunPod serverless endpoint via sprite-me's RunPodClient
and saves the PNG result. Used for end-to-end verification after the deploy
script creates a template + endpoint.

Usage:
    uv run scripts/test_endpoint.py
    uv run scripts/test_endpoint.py --prompt "green slime with a smile"
    uv run scripts/test_endpoint.py --prompt "warrior" --seed 1 --steps 20 --out knight.png
    uv run scripts/test_endpoint.py --animate --prompt "walk cycle" --frames 6

Requires SPRITE_ME_RUNPOD_API_KEY and SPRITE_ME_RUNPOD_ENDPOINT_ID in .env
(set by scripts/deploy_endpoint.sh).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root regardless of CWD
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")

# Make src/ importable when running the script directly
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sprite_me.inference.runpod_client import RunPodClient, RunPodError  # noqa: E402
from sprite_me.inference.workflow_builder import (  # noqa: E402
    build_animate_workflow,
    build_generate_workflow,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for sprite-me RunPod endpoint")
    p.add_argument(
        "--prompt",
        default="knight with longsword",
        help="Subject prompt (LoRA trigger + style tags are added automatically)",
    )
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument(
        "--lora-strength",
        type=float,
        default=0.85,
        help="Flux-2D-Game-Assets LoRA strength (0.0-1.0)",
    )
    p.add_argument(
        "--out",
        default="test_output.png",
        help="Output filename (written to repo root)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Total client-side timeout in seconds (default 30min for first cold pull)",
    )
    p.add_argument(
        "--animate",
        action="store_true",
        help="Test the animation workflow instead of single-image generation",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=6,
        help="Number of animation frames (only used with --animate)",
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


async def run_animate(args: argparse.Namespace, client: RunPodClient) -> list[bytes]:
    workflow = build_animate_workflow(
        reference_image_b64="",
        animation_prompt=args.prompt,
        frames=args.frames,
        width=args.width,
        height=args.height,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        lora_strength=args.lora_strength,
    )
    return await client.generate(workflow)


async def main() -> int:
    args = _parse_args()
    client = RunPodClient(poll_interval=3.0, timeout_seconds=args.timeout)

    if not client.endpoint_id:
        print("ERROR: SPRITE_ME_RUNPOD_ENDPOINT_ID not set in .env", file=sys.stderr)
        print("Run scripts/deploy_endpoint.sh first.", file=sys.stderr)
        return 1

    print(f"Endpoint: {client.endpoint_id}")
    mode = "animate" if args.animate else "generate"
    print(
        f"Mode: {mode} | prompt='{args.prompt}' "
        f"{args.width}x{args.height} seed={args.seed} steps={args.steps} "
        f"guidance={args.guidance} lora_strength={args.lora_strength}"
        + (f" frames={args.frames}" if args.animate else "")
    )
    print("Submitting workflow...")

    t0 = time.time()
    try:
        if args.animate:
            images = await run_animate(args, client)
        else:
            images = await run_generate(args, client)
    except RunPodError as e:
        print(f"RunPod error after {time.time() - t0:.1f}s: {e}", file=sys.stderr)
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
        # batch_size=N returns N separate PNGs (one per frame candidate).
        # Save each individually + also assemble a single-row sprite sheet.
        from sprite_me.processing.spritesheet import assemble_spritesheet

        stem = out.stem
        for i, frame_data in enumerate(images):
            frame_out = _REPO_ROOT / f"{stem}_frame{i:02d}.png"
            frame_out.write_bytes(frame_data)
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
