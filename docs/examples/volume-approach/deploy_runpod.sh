#!/usr/bin/env bash
# Build and deploy the sprite-me ComfyUI image to RunPod.
#
# The image is slim (no models baked in). On first worker boot, models
# download to an attached network volume. Run sync_models.py to pre-populate
# the volume via the S3 API instead of waiting for the worker to download.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-sprite-me-comfyui}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"

info() { printf "\033[36m[sprite-me]\033[0m %s\n" "$*"; }
error() { printf "\033[31m[sprite-me]\033[0m %s\n" "$*" >&2; }

if [ -z "$REGISTRY" ]; then
  error "Set REGISTRY env var (e.g. REGISTRY=docker.io/yourusername)"
  exit 1
fi

FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

info "Building $FULL_IMAGE"
docker build \
  -t "$FULL_IMAGE" \
  -f "$REPO_DIR/docker/comfyui/Dockerfile" \
  "$REPO_DIR/docker/comfyui"

info "Pushing $FULL_IMAGE"
docker push "$FULL_IMAGE"

info "Image pushed. Next steps:"
cat <<EOF

1. Create network volumes (idempotent, safe to re-run):
     export SPRITE_ME_RUNPOD_API_KEY=...
     uv run sprite-me-volumes setup

2. Create serverless endpoint at:
     https://www.runpod.io/console/serverless

   Settings:
     Docker image: $FULL_IMAGE
     GPU: RTX 3090 or 4090 (24 GB VRAM)
     Min workers: 0
     Max workers: 3
     Idle timeout: 5 seconds
     Network volumes: attach the sprite-me volumes from step 1

3. Copy the endpoint ID and set:
     export SPRITE_ME_RUNPOD_ENDPOINT_ID=<id>

4. (Optional) Attach volumes to the new endpoint:
     uv run sprite-me-volumes setup

5. (Optional) Pre-populate LoRAs via S3 API (avoids first-boot download):
     mkdir -p models/loras
     curl -L -o models/loras/flux-2d-game-assets.safetensors \\
       https://huggingface.co/gokaygokay/Flux-2D-Game-Assets-LoRA/resolve/main/flux_2d_game_assets.safetensors
     uv run scripts/sync_models.py sync models/loras --parallel 3

6. Test:
     uv run sprite-me-api   # in one terminal
     curl -X POST http://localhost:8420/api/generate \\
       -H 'Content-Type: application/json' \\
       -d '{"prompt":"warrior with sword"}'
EOF
