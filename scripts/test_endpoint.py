"""Smoke test for the sprite-me RunPod endpoint.

Submits a pixel art generation job end-to-end via the sprite-me RunPodClient
and saves the resulting PNG to ./test_output.png. First call usually takes
60-180s (cold start — RunPod has to pull the Docker image to the worker).

Run:
    uv run scripts/test_endpoint.py

Requires SPRITE_ME_RUNPOD_API_KEY and SPRITE_ME_RUNPOD_ENDPOINT_ID in .env
(set by scripts/deploy_endpoint.sh).
"""

from __future__ import annotations

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
from sprite_me.inference.workflow_builder import build_generate_workflow  # noqa: E402


async def main() -> int:
    client = RunPodClient(poll_interval=3.0, timeout_seconds=600.0)

    if not client.endpoint_id:
        print("ERROR: SPRITE_ME_RUNPOD_ENDPOINT_ID not set in .env", file=sys.stderr)
        print("Run scripts/deploy_endpoint.sh first.", file=sys.stderr)
        return 1

    print(f"Endpoint: {client.endpoint_id}")
    print("Building workflow: knight with longsword, 512x512, 20 steps, seed=42")

    workflow = build_generate_workflow(
        prompt="knight with longsword",
        width=512,
        height=512,
        seed=42,
        steps=20,
    )

    print("Submitting workflow...")
    t0 = time.time()
    try:
        images = await client.generate(workflow)
    except RunPodError as e:
        print(f"RunPod error: {e}", file=sys.stderr)
        await client.close()
        return 2
    finally:
        # Keep client open until after generate returns
        pass

    elapsed = time.time() - t0
    print(f"Got {len(images)} image(s) in {elapsed:.1f}s")

    if not images:
        print("ERROR: no images returned", file=sys.stderr)
        await client.close()
        return 3

    out = _REPO_ROOT / "test_output.png"
    out.write_bytes(images[0])
    print(f"Saved {out} ({len(images[0])} bytes)")

    await client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
