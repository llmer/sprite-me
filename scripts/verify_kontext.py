"""Phase 0 Kontext verification — submit a raw Kontext workflow against the
sprite-me endpoint and save the result.

Takes a hero reference PNG from the filesystem, base64-encodes it, submits
a FLUX.1 Kontext workflow (see docker/comfyui/workflows/kontext_verify.json
once it exists, or /tmp/sprite_me_kontext_verify.json during bring-up) with
the hero image in the handler's `images` input field. The handler writes
hero.png to /comfyui/input/ on the worker, ComfyUI's LoadImage node picks
it up, FluxKontextImageScale scales it, VAEEncode produces a latent, and
that latent feeds both ReferenceLatent (for conditioning) and KSampler
(as the starting latent).

Usage:
    uv run scripts/verify_kontext.py --hero test_alpha4.png --prompt "the same knight walking to the right"
    uv run scripts/verify_kontext.py --hero test_alpha4.png --prompt "sword raised overhead" --out kontext_attack.png
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sprite_me.inference.runpod_client import RunPodClient, RunPodError  # noqa: E402

_WORKFLOW_PATH = Path("/tmp/sprite_me_kontext_verify.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0 Kontext verification")
    p.add_argument("--hero", required=True, help="Hero reference PNG path")
    p.add_argument(
        "--prompt",
        default="the same knight walking to the right, left foot forward, mid-stride, side view",
        help="Pose prompt describing the desired change",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=2.5, help="FluxGuidance scale (Kontext defaults to 2.5)")
    p.add_argument("--out", default="kontext_verify.png")
    p.add_argument("--timeout", type=float, default=1500.0)
    p.add_argument(
        "--workflow",
        default=str(_WORKFLOW_PATH),
        help="Path to the Kontext workflow JSON template",
    )
    return p.parse_args()


def _load_workflow(path: str, args: argparse.Namespace) -> dict:
    """Load and parameterize the Kontext workflow JSON."""
    raw = json.loads(Path(path).read_text())
    # Strip any _comment keys before sending
    wf = {k: v for k, v in raw.items() if not k.startswith("_")}
    # Inject prompt-specific values
    wf["7"]["inputs"]["text"] = args.prompt
    wf["11"]["inputs"]["seed"] = args.seed
    wf["11"]["inputs"]["steps"] = args.steps
    wf["9"]["inputs"]["guidance"] = args.guidance
    return wf


async def main() -> int:
    args = _parse_args()
    hero_path = Path(args.hero)
    if not hero_path.exists():
        print(f"ERROR: hero image {hero_path} not found", file=sys.stderr)
        return 1

    hero_bytes = hero_path.read_bytes()
    hero_b64 = base64.b64encode(hero_bytes).decode()
    print(f"Hero: {hero_path} ({len(hero_bytes)} bytes)")

    wf = _load_workflow(args.workflow, args)
    print(f"Workflow loaded from {args.workflow}")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}, Steps: {args.steps}, Guidance: {args.guidance}")

    client = RunPodClient(poll_interval=5.0, timeout_seconds=args.timeout)
    if not client.endpoint_id:
        print("ERROR: SPRITE_ME_RUNPOD_ENDPOINT_ID not set", file=sys.stderr)
        return 1
    print(f"Endpoint: {client.endpoint_id}")

    # Build input payload — Kontext needs the hero image passed alongside
    # the workflow via the handler's `images` field.
    input_payload = {
        "workflow": wf,
        "images": [
            {"name": "hero.png", "image": hero_b64},
        ],
    }

    print("Submitting workflow...")
    t0 = time.time()
    try:
        # submit_workflow wraps in {"input": {"workflow": wf}} — we need to
        # override to include the images field too. Use the underlying
        # client directly.
        resp = await client.client.post(
            f"{client.base_url}/run",
            json={"input": input_payload},
        )
        resp.raise_for_status()
        job_id = resp.json()["id"]
        print(f"Job: {job_id}")

        payload = await client.wait_for_result(job_id)
        images = RunPodClient._extract_images(payload.get("output", {}))
    except RunPodError as e:
        print(f"RunPod error after {time.time() - t0:.1f}s: {e}", file=sys.stderr)
        await client.close()
        return 2
    except Exception as e:
        print(f"Error after {time.time() - t0:.1f}s: {e}", file=sys.stderr)
        await client.close()
        return 3

    elapsed = time.time() - t0
    print(f"Got {len(images)} image(s) in {elapsed:.1f}s")
    if not images:
        print("No images returned", file=sys.stderr)
        await client.close()
        return 4

    out = _REPO_ROOT / args.out
    out.write_bytes(images[0])
    print(f"Saved {out} ({len(images[0])} bytes)")
    await client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
