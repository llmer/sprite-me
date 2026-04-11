# RunPod Deployment Guide

## Overview

sprite-me runs its ComfyUI worker as a RunPod serverless endpoint. Architecture as of v0.2.0: **fat base image + RunPod network volume for model additions**. Workers pull a ~23 GB Docker image once per machine (cached after), and model updates (new LoRAs, new checkpoints like Kontext) are pushed to the network volume via S3 sync without rebuilding the image.

Reference: https://docs.runpod.io/tutorials/serverless/comfyui

## Architecture

```
sprite-me API / MCP Server
    │
    │ POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run
    │   body: { "input": { "workflow": <ComfyUI workflow JSON> } }
    │
    ▼
RunPod Serverless Platform  (scheduler picks a DC with volume access + 5090 capacity)
    │
    │ Pulls bbbasddaaa/sprite-me-comfyui:latest from Docker Hub
    │ Mounts network volume at /runpod-volume/
    │
    ▼
ComfyUI Worker Container  (RTX 5090, 30 GB container disk)
    │
    │ Base: runpod/worker-comfyui:5.8.5-flux1-dev-fp8
    │   - torch + CUDA + ComfyUI runtime
    │   - flux1-dev-fp8.safetensors baked into /comfyui/models/checkpoints/
    │   - VAE, text encoders baked in
    │   - rp_handler.py (RunPod handler)
    │
    │ + our extra_model_paths.yaml at /comfyui/extra_model_paths.yaml
    │   that tells ComfyUI to ALSO scan /runpod-volume/models/ at boot
    │
    ▼
Network Volume  (b690sw6p4j, 50 GB, EUR-NO-1)
    │
    │ /runpod-volume/models/
    │   checkpoints/flux1-dev-fp8.safetensors        (~17 GB, duplicate of baked, safe)
    │   loras/flux-2d-game-assets.safetensors        (~90 MB)
    │   diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors  (future: Kontext)
    │   (any additional text_encoders / vae / controlnet / etc.)
    │
    ▼
Returns: { "status": "COMPLETED", "output": { "images": [{ "image": "<base64>" }] } }
```

## Strategy: fat base + volume for additions

We extend the stock `runpod/worker-comfyui:5.8.5-flux1-dev-fp8` image (23 GB) with a single `COPY extra_model_paths.yaml` line. That YAML tells ComfyUI to scan `/runpod-volume/models/` in addition to the image-baked `/comfyui/models/`. The runtime behavior:

- **Models already in the image** (FLUX.1-dev fp8 checkpoint, VAE, text encoders) load directly from the image — fast, always available
- **Models on the volume** (additional LoRAs, Kontext checkpoint, custom controlnets) appear under their configured keys and are usable by workflows

The result: the Docker image only needs to be rebuilt when we change custom nodes or entrypoint logic. New LoRAs / checkpoints / model types get synced to the volume via `scripts/sync_models.py` or a one-shot CPU-pod wget, and appear at the next worker cold start — no image rebuild.

### Why this shape vs alternatives

| Alternative | Why not |
|---|---|
| Bake everything into the image (v0.1.x) | Any model change → 20-30 min iteration loop (rebuild + push + pull cycle) |
| Thin base image (`5.8.5-base-cuda12.8.1`, 16 GB) + install torch ourselves | Saves ~7 GB at the cost of a fragile manual Python dependency install. Tried this; bolting torch + ComfyUI requirements on top of the bare CUDA base isn't worth it |
| Custom entrypoint.sh to symlink or log volume contents | Tried this. Somehow broke ComfyUI startup in a way I haven't fully diagnosed (the upstream `/start.sh` expects something about its launch environment that my wrapper violated). The YAML alone gives us the iteration-speed benefit without the debugging headache |
| Network volume as the primary model store (no image-baked models) | Would save more disk but requires managing every model file manually on the volume, and v0.1.x's image-baked FLUX.1-dev is already a solid known-good baseline |

### Attempt-1 and attempt-2 post-mortems

- Attempt 1 (pure volume + runtime download): `docs/examples/volume-approach/README.md`
- Attempt 2 (thin base + yaml + custom entrypoint): the "custom entrypoint.sh to symlink" line above. The row above that ("thin base image") is what the alpha1 test revealed

## Dockerfile

`docker/comfyui/Dockerfile`:

```dockerfile
FROM runpod/worker-comfyui:5.8.5-flux1-dev-fp8

# Tell ComfyUI to ALSO read models from the mounted network volume.
# We inherit the upstream CMD ["/start.sh"] and WORKDIR /comfyui — no
# entrypoint override.
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
```

That's the whole thing.

`docker/comfyui/extra_model_paths.yaml`:

```yaml
sprite_me:
  base_path: /runpod-volume/models/
  checkpoints: checkpoints/
  clip_vision: clip_vision/
  configs: configs/
  controlnet: controlnet/
  diffusion_models: |
    diffusion_models/
    unet/
  embeddings: embeddings/
  loras: loras/
  text_encoders: |
    text_encoders/
    clip/
  upscale_models: upscale_models/
  vae: vae/
```

**Important**: the top-level keys must match the canonical ComfyUI model-type names. Do NOT add `clip:` or `unet:` as top-level keys — those were legacy directory names, and they're now subsumed under `text_encoders` and `diffusion_models` respectively via multiline string paths (the `|` block scalar). Invalid keys at the top level may cause ComfyUI to fail to start silently.

## Network volume setup

One-time: provision a 50 GB volume in a DC with verified RTX 5090 serverless capacity.

```bash
runpodctl network-volume create \
  --name sprite-me-thin \
  --size 50 \
  --data-center-id EUR-NO-1
# -> volume ID, e.g. b690sw6p4j
```

**DC selection**: run `runpodctl datacenter list` and look for DCs where the GPU you want has `stockStatus` High or Medium. As of this writing, EUR-NO-1 and EUR-IS-2 have consistently had RTX 5090 capacity. US DCs are more hit-or-miss for serverless.

## Seeding the volume

The fastest way is a one-shot CPU pod in the same DC as the volume. CPU pods have fast internal networking to HuggingFace (~1 Gbps) and cost ~$0.06/hr.

```bash
# Create pod via REST API (runpodctl pod create doesn't support --network-volume-id)
curl -sX POST "https://rest.runpod.io/v1/pods" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sprite-me-seeder",
    "imageName": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
    "computeType": "CPU",
    "containerDiskInGb": 10,
    "networkVolumeId": "b690sw6p4j",
    "volumeMountPath": "/runpod-volume",
    "ports": ["22/tcp"]
  }'
# -> pod ID

# Wait for SSH
runpodctl ssh info <pod-id>

# SSH in and wget
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> '
mkdir -p /runpod-volume/models/checkpoints /runpod-volume/models/loras
wget -q -O /runpod-volume/models/checkpoints/flux1-dev-fp8.safetensors \
  https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors
wget -q -O /runpod-volume/models/loras/flux-2d-game-assets.safetensors \
  https://huggingface.co/gokaygokay/Flux-2D-Game-Assets-LoRA/resolve/main/game_assets_v3.safetensors
ls -la /runpod-volume/models/checkpoints/ /runpod-volume/models/loras/
'

# Delete pod (models persist on the volume)
runpodctl pod delete <pod-id>
```

For incremental updates from your local machine, see `scripts/sync_models.py` which uses RunPod's S3-compatible API.

## Build & push via GitHub Actions

Image builds run on GitHub Actions. The workflow file is `.github/workflows/docker-image.yml`.

### One-time setup

1. Create a GitHub repo and push the initial commit.
2. Add Docker Hub secrets in Settings → Secrets and variables → Actions:
   - `DOCKERHUB_USERNAME` = `bbbasddaaa`
   - `DOCKERHUB_TOKEN` = your Docker Hub access token

### Trigger a build

```bash
git tag docker-v0.2.0
git push origin docker-v0.2.0
```

Or manually dispatch from the Actions tab.

GHA build takes ~5-15 minutes. Most of that is layer cache warming; actual work is small because the yaml is only a few KB.

## Deploy the endpoint

Once the image is on Docker Hub, run:

```bash
cd /Users/jim/src/sprite-me
VOLUME_IDS=b690sw6p4j \
IMAGE=bbbasddaaa/sprite-me-comfyui:v0.2.0 \
./scripts/deploy_endpoint.sh
```

What it does:
1. Creates a RunPod serverless template via `runpodctl template create`
2. Creates a serverless endpoint via `runpodctl serverless create` on RTX 5090, `workers-min=0, workers-max=5`
3. PATCHes the endpoint with `networkVolumeIds` to attach the volume (runpodctl doesn't expose this flag directly)
4. Writes template ID, endpoint ID, and volume ID to `.env`

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `RUNPOD_API_KEY` | sourced from `/Users/jim/src/vid/.env` | runpodctl auth |
| `IMAGE` | `bbbasddaaa/sprite-me-comfyui:latest` | Docker image to deploy |
| `VOLUME_IDS` | **required** | Comma-separated list of network volume IDs to attach |
| `GPU_ID` | `NVIDIA GeForce RTX 5090` | GPU type |
| `CONTAINER_DISK_GB` | `30` | Worker container disk size |

## Iteration cycle

| Change type | Work | Time |
|---|---|---|
| **Code / workflow JSON** (Python client, builder, skill files) | Nothing, pure client-side | instant |
| **New / updated LoRA** | `scripts/sync_models.py upload <file>.safetensors --prefix loras` | ~1 min for a 90 MB LoRA |
| **New checkpoint** (e.g. FLUX Kontext) | sync to volume via CPU pod wget or `sync_models.py` | ~5-10 min for a 12 GB checkpoint |
| **Custom node / yaml / entrypoint** | `git tag docker-v0.2.X + push`, GHA builds (~5 min), `runpodctl template update --image`, recreate endpoint to cycle workers, first worker cold pulls (~3-7 min first time on a machine, then cached) | ~10-15 min |

## Smoke test

```bash
uv run scripts/test_endpoint.py
```

Default: generates a knight with longsword at 512×512 with seed 42, saves to `test_output.png`.

First cold start on a fresh machine: ~3-7 minutes (image pull). Warm path: ~6-10 seconds (generation only).

```bash
# Custom prompt:
uv run scripts/test_endpoint.py --prompt "green slime with a smile" --out slime.png

# Different seed + size:
uv run scripts/test_endpoint.py --prompt "wizard staff" --seed 1 --width 256 --height 256
```

## API reference

### Submit job

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run
Authorization: Bearer {API_KEY}
Content-Type: application/json

{
  "input": {
    "workflow": { <raw ComfyUI node dict> }
  }
}

Response:
{ "id": "job-abc123", "status": "IN_QUEUE" }
```

**Important**: the `workflow` value must be the **raw** ComfyUI node dict (e.g. `{"1": {...}, "2": {...}, ...}`), NOT wrapped in `{"prompt": ...}`. The upstream `runpod/worker-comfyui` handler wraps it automatically before calling ComfyUI's `/prompt` endpoint. Double-wrapping causes a 400.

### Check status

```
GET https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}
Authorization: Bearer {API_KEY}

Response:
{
  "id": "job-abc123",
  "status": "COMPLETED",       // IN_QUEUE | IN_PROGRESS | COMPLETED | FAILED | TIMED_OUT | CANCELLED
  "delayTime": 1234,
  "executionTime": 5678,
  "output": { "images": [{ "image": "<base64 PNG>" }] }
}
```

### Synchronous run

`POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync` — same body, blocks up to 120s. Use for quick tests; for anything that might cold-start, prefer `/run` + polling.

## Cost estimates

RTX 5090 runs ~$0.99/hr active on serverless. Scale-to-zero means $0 when idle.

| Operation | Time | Cost |
|---|---|---|
| Cold start (first pull on a fresh machine) | 3-7 min | ~$0.05 (billed for cold-start time) |
| Warm sprite generation (20 steps, 512x512) | 6-10s | ~$0.002 |
| Network volume storage (50 GB, EUR-NO-1) | always-on | ~$3.60/month |
| Idle endpoint, no workers | — | $0 |

## Troubleshooting

**Worker goes UNHEALTHY immediately / "ComfyUI server not reachable after multiple retries"**

The upstream `/start.sh` launches ComfyUI in the background and the RunPod handler polls `127.0.0.1:8188`. If ComfyUI crashed at startup, the handler gives up. Known causes:

- `extra_model_paths.yaml` has invalid top-level keys (anything outside `checkpoints, text_encoders, clip_vision, configs, controlnet, diffusion_models, embeddings, loras, upscale_models, vae, audio_encoders, model_patches`)
- Custom entrypoint that `set -e`-exits or changes CWD in a way that breaks /start.sh's assumptions
- The base image is `-base-cuda12.8.1` (too bare — no torch installed, GPU check crashes)

Debug path: create a regular GPU pod via the REST API with `"dockerStartCmd": ["sleep","infinity"]` and the same image, attach the volume, SSH in, and run `/start.sh` manually to see the actual error.

**Worker goes "throttled"**

Single-machine pinning. Set `workers-max >= 3` so the scheduler can spread across machines. Default in `deploy_endpoint.sh` is 5.

**"Max workers quota exceeded"**

You have 5 worker slots across all endpoints. Delete unused endpoints or lower `workers-max` on the sprite-me one.

**RTX 5090 has no serverless capacity in the volume's DC**

Fall back to RTX 4090: `GPU_ID="NVIDIA GeForce RTX 4090" ./scripts/deploy_endpoint.sh`. Or move the volume to a different DC (create a new volume elsewhere, re-seed via CPU pod, delete old).

**FLUX workflow returns 400 from ComfyUI**

Check the workflow shape. FLUX requires `EmptySD3LatentImage` (not `EmptyLatentImage`), a `FluxGuidance` node between CLIPTextEncode and KSampler.positive, and `cfg=1 / scheduler="simple"` on KSampler. See `src/sprite_me/inference/workflow_builder.py` for a working reference.

## Workflow customization

Our workflow builder (`src/sprite_me/inference/workflow_builder.py`) dynamically modifies `docker/comfyui/workflows/pixel_art_generate.json`:

- **Prompt injection**: `nodes["3"]["inputs"]["text"]` gets the user prompt + LoRA trigger word
- **Seed control**: `nodes["6"]["inputs"]["seed"]`
- **Dimensions**: `nodes["5"]["inputs"]["width"]/["height"]`
- **LoRA strength**: `nodes["2"]["inputs"]["strength_model"]/["strength_clip"]`
- **Steps / guidance**: `nodes["6"]["inputs"]["steps"]` + `nodes["9"]["inputs"]["guidance"]` (FLUX-specific node 9)

To add a new workflow: design in ComfyUI's web UI, export as API-format JSON, save to `docker/comfyui/workflows/`, and add a builder function in `workflow_builder.py`.

To use a model that only exists on the volume (e.g. FLUX Kontext): just reference it by filename in the workflow JSON. ComfyUI's `extra_model_paths.yaml` scanner makes volume files appear under the same model-type key as image-baked files.
