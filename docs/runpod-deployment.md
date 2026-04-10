# RunPod Deployment Guide

## Overview

sprite-me runs its ComfyUI worker as a RunPod serverless endpoint. This gives us auto-scaling (including scale-to-zero), pay-per-second billing, and workers that come up fast because the Docker image already has everything they need.

Reference: https://docs.runpod.io/tutorials/serverless/comfyui

## Architecture

```
sprite-me API / MCP Server
    │
    │ POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run
    │   body: { "input": { "workflow": <ComfyUI workflow JSON> } }
    │
    ▼
RunPod Serverless Platform
    │
    │ Schedules worker on any DC with RTX 5090 serverless capacity
    │ Pulls Docker image from Docker Hub (cached after first pull)
    │
    ▼
ComfyUI Worker Container  (bbbasddaaa/sprite-me-comfyui:latest)
    │
    │ Stock runpod/worker-comfyui:5.0.0-flux1-dev
    │   + Flux-2D-Game-Assets LoRA baked in at build time
    │
    │ Executes workflow:
    │   CheckpointLoaderSimple → LoraLoader → CLIPTextEncode →
    │   KSampler → VAEDecode → SaveImage
    │
    ▼
Returns: { "status": "COMPLETED", "output": { "images": [{ "image": "<base64>" }] } }
```

## Strategy: Bake the LoRA into the image

We extend the stock `runpod/worker-comfyui:5.0.0-flux1-dev` image with a single `RUN wget` step that downloads the Flux-2D-Game-Assets LoRA into `/comfyui/models/loras/` at Docker build time. No runtime downloads, no `docker-start-cmd` override, no network volume — the base image's `/start.sh` handles everything else.

This is deliberately the opposite of what many guides recommend. Our rationale:

- **Simple beats flexible**: we have one LoRA, it's 160 MB, and it changes rarely. The "clever" network-volume approach has many moving parts that can and did fail (see `docs/examples/volume-approach/README.md` for attempt-1 post-mortem).
- **No runtime coordination**: nothing to go wrong during worker boot.
- **No datacenter pinning**: no volume means RunPod can schedule the worker anywhere with RTX 5090 capacity.
- **Free**: no ongoing network volume storage cost.
- **Cold start is acceptable**: the first ever worker pull of the image takes 60–180s; subsequent workers on the same host are fast.

When to reconsider: you have many LoRAs, they change daily, or they're multi-gigabyte checkpoints. Then look at the archived volume approach.

## Dockerfile

`docker/comfyui/Dockerfile`:

```dockerfile
FROM runpod/worker-comfyui:5.0.0-flux1-dev

RUN mkdir -p /comfyui/models/loras && \
    wget -q -O /comfyui/models/loras/flux-2d-game-assets.safetensors \
      "https://huggingface.co/gokaygokay/Flux-2D-Game-Assets-LoRA/resolve/main/flux_2d_game_assets.safetensors" && \
    test -s /comfyui/models/loras/flux-2d-game-assets.safetensors
```

Files in the stock base image we rely on:
- `/start.sh` — upstream entrypoint, starts ComfyUI + the RunPod handler
- `/comfyui/models/checkpoints/flux1-dev-fp8.safetensors` — FLUX.1-dev fp8 checkpoint
- VAE, CLIP, text encoders under `/comfyui/models/`

Our workflow builder (`src/sprite_me/inference/workflow_builder.py`) references these filenames directly, so no path translation is needed.

## Build & push via GitHub Actions

Image builds run on GitHub Actions rather than locally — Actions runners have fat pipes to Docker Hub and don't tie up your home connection for 15–25 minutes. The workflow file is `.github/workflows/docker-image.yml`.

### One-time setup

1. **Create a GitHub repo** for sprite-me (public or private):
   ```bash
   cd /Users/jim/src/sprite-me
   git add -A && git commit -m "Initial commit"
   gh repo create sprite-me --public --source=. --push
   ```

2. **Add Docker Hub secrets** to the repo via Settings → Secrets and variables → Actions:
   - `DOCKERHUB_USERNAME` = `bbbasddaaa`
   - `DOCKERHUB_TOKEN` = your Docker Hub access token

### Trigger a build

Either push a git tag:
```bash
git tag docker-v0.1.0
git push origin docker-v0.1.0
```

Or manually dispatch from the Actions tab:
1. Open the repo on GitHub → Actions
2. Select "Build & push sprite-me ComfyUI image"
3. Click "Run workflow", pick a tag like `v0.1.0`, run

Expected build time: 5–10 minutes. The workflow frees disk space, sets up buildx, logs into Docker Hub, builds for `linux/amd64`, and pushes `bbbasddaaa/sprite-me-comfyui:{tag}` and `:latest`.

## Deploy the endpoint

Once the image is on Docker Hub, run:

```bash
cd /Users/jim/src/sprite-me
./scripts/deploy_endpoint.sh
```

What it does:
1. Creates a RunPod serverless template from `bbbasddaaa/sprite-me-comfyui:latest` via `runpodctl template create`
2. Creates a serverless endpoint using that template via `runpodctl serverless create`, on RTX 5090, with `workers-min=0`, `workers-max=1`, no network volume, no DC constraint
3. Writes the template ID and endpoint ID to `.env` (gitignored)

Environment variables the script reads:

| Variable | Default | Purpose |
|---|---|---|
| `RUNPOD_API_KEY` | (sourced from `/Users/jim/src/vid/.env`) | runpodctl auth |
| `IMAGE` | `bbbasddaaa/sprite-me-comfyui:latest` | Docker image to deploy |
| `GPU_ID` | `NVIDIA GeForce RTX 5090` | GPU type — change to `NVIDIA GeForce RTX 4090` for fallback |
| `CONTAINER_DISK_GB` | `30` | Worker container disk size |

## Smoke test

```bash
uv run scripts/test_endpoint.py
```

What it does:
1. Loads `.env` via `python-dotenv`
2. Builds a pixel art generation workflow via `build_generate_workflow("knight with longsword", ...)`
3. Submits via `RunPodClient.generate()` — polls until `COMPLETED`
4. Writes the first returned image to `./test_output.png`

Expected first-run output:
```
Endpoint: <id>
Submitting workflow...
Got 1 image(s) in 90.4s       # first cold start
Saved /Users/jim/src/sprite-me/test_output.png (24680 bytes)
```

Warm runs (same worker handling multiple jobs back-to-back) complete in 5–15 seconds.

## API reference

### Submit job

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run
Authorization: Bearer {API_KEY}
Content-Type: application/json

{
  "input": {
    "workflow": { <ComfyUI workflow JSON> }
  }
}

Response:
{ "id": "job-abc123", "status": "IN_QUEUE" }
```

### Check status

```
GET https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}
Authorization: Bearer {API_KEY}

Response:
{
  "id": "job-abc123",
  "status": "COMPLETED",       // IN_QUEUE | IN_PROGRESS | COMPLETED | FAILED | TIMED_OUT | CANCELLED
  "delayTime": 1234,           // ms in queue
  "executionTime": 5678,       // ms executing
  "output": {
    "images": [
      { "image": "<base64 PNG>" }
    ]
  }
}
```

### Synchronous run

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
```

Same body as `/run` but blocks until complete. 120s default timeout. Use for quick interactive requests; use `/run` + polling for anything that might take longer than 60s.

## Cost estimates

RTX 5090 runs ~$0.00042/s active. Scale-to-zero means $0 when idle.

| Operation | Time | Cost |
|---|---|---|
| Single sprite generation (20 steps, 512x512) | 5–10s warm, 60–180s cold | ~$0.002 warm, ~$0.05 cold |
| Animation sheet (6 frames) | 15–30s warm | ~$0.01 |
| Idle | — | $0 |

## Workflow customization

Our workflow builder (`src/sprite_me/inference/workflow_builder.py`) dynamically modifies `docker/comfyui/workflows/pixel_art_generate.json`:

- **Prompt injection**: `nodes["3"]["inputs"]["text"]` gets the user prompt + LoRA trigger word
- **Seed control**: `nodes["6"]["inputs"]["seed"]`
- **Dimensions**: `nodes["5"]["inputs"]["width"]/["height"]`
- **LoRA strength**: `nodes["2"]["inputs"]["strength_model"]/["strength_clip"]`
- **Steps/CFG**: `nodes["6"]["inputs"]["steps"]/["cfg"]`

To add a new workflow, design it in ComfyUI's web UI, export as API format JSON, save to `docker/comfyui/workflows/`, and add a builder function in `workflow_builder.py`.

## Troubleshooting

**Worker goes UNHEALTHY immediately**
- Check `runpodctl serverless get $SPRITE_ME_RUNPOD_ENDPOINT_ID --include-workers` for the last status and any error strings
- Verify the image exists on Docker Hub: `docker manifest inspect bbbasddaaa/sprite-me-comfyui:latest`
- Retry once — RunPod scheduling is occasionally flaky
- If persistent: temporarily deploy a second endpoint using the raw `runpod/worker-comfyui:5.0.0-flux1-dev` image to isolate whether the sprite-me layer is the problem

**Job fails with LoRA not found**
- Confirm the file is actually in the image: `docker run --rm --entrypoint ls bbbasddaaa/sprite-me-comfyui:latest /comfyui/models/loras/`
- The filename in the workflow JSON must match exactly: `flux-2d-game-assets.safetensors`

**"Max workers quota exceeded" on create**
- You have 5 worker slots across all endpoints. List other endpoints with `runpodctl serverless list` and either delete unused ones or set `workers-max=1` on the sprite-me endpoint

**RTX 5090 has no serverless capacity**
- Fall back to RTX 4090: `GPU_ID="NVIDIA GeForce RTX 4090" ./scripts/deploy_endpoint.sh`
- Delete the old endpoint first to free worker quota
