#!/usr/bin/env bash
# Create (or update) the sprite-me RunPod serverless template + endpoint.
#
# Approach: bake the LoRA into a custom image (see docker/comfyui/Dockerfile)
# and let RunPod schedule workers on any available RTX 5090. No network volume,
# no datacenter constraint.
#
# Requires:
#   - runpodctl installed (https://cli.runpod.net)
#   - RUNPOD_API_KEY in env (or sourced from /Users/jim/src/vid/.env)
#   - Docker image pushed to Docker Hub (GitHub Actions handles this — see
#     .github/workflows/docker-image.yml)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-bbbasddaaa/sprite-me-comfyui:latest}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 5090}"
TEMPLATE_NAME="${TEMPLATE_NAME:-sprite-me}"
ENDPOINT_NAME="${ENDPOINT_NAME:-sprite-me}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-30}"

info() { printf "\033[36m[sprite-me]\033[0m %s\n" "$*"; }
error() { printf "\033[31m[sprite-me]\033[0m %s\n" "$*" >&2; }

# Source RUNPOD_API_KEY if not set
if [ -z "${RUNPOD_API_KEY:-}" ]; then
  if [ -f /Users/jim/src/vid/.env ]; then
    set -a && source /Users/jim/src/vid/.env && set +a
  fi
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
  error "RUNPOD_API_KEY not set and /Users/jim/src/vid/.env doesn't have it."
  error "Export RUNPOD_API_KEY=... or source your .env first."
  exit 1
fi

info "Image: $IMAGE"
info "GPU: $GPU_ID"

# 1. Create the serverless template
info "Creating serverless template '$TEMPLATE_NAME'..."
TEMPLATE_JSON=$(runpodctl template create \
  --name "$TEMPLATE_NAME" \
  --image "$IMAGE" \
  --serverless \
  --container-disk-in-gb "$CONTAINER_DISK_GB")
TEMPLATE_ID=$(echo "$TEMPLATE_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
info "Template ID: $TEMPLATE_ID"

# 2. Create the serverless endpoint (no volume, no DC constraint)
info "Creating serverless endpoint '$ENDPOINT_NAME'..."
ENDPOINT_JSON=$(runpodctl serverless create \
  --name "$ENDPOINT_NAME" \
  --template-id "$TEMPLATE_ID" \
  --gpu-id "$GPU_ID" \
  --workers-min 0 \
  --workers-max 1)
ENDPOINT_ID=$(echo "$ENDPOINT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
info "Endpoint ID: $ENDPOINT_ID"

# 3. Write IDs to .env (overwrites previous values)
info "Writing $REPO_DIR/.env"
cat > "$REPO_DIR/.env" <<EOF
# sprite-me runtime config (gitignored — do not commit)
SPRITE_ME_RUNPOD_API_KEY=$RUNPOD_API_KEY
SPRITE_ME_RUNPOD_ENDPOINT_ID=$ENDPOINT_ID
SPRITE_ME_RUNPOD_TEMPLATE_ID=$TEMPLATE_ID

# RunPod resources created $(date +%Y-%m-%d):
#   Image:     $IMAGE
#   Template:  $TEMPLATE_ID  ($TEMPLATE_NAME)
#   Endpoint:  $ENDPOINT_ID  ($ENDPOINT_NAME)
#   GPU:       $GPU_ID, max-workers=1, scale-to-zero
EOF

info ""
info "Done. Next step:"
info "  cd $REPO_DIR && uv run scripts/test_endpoint.py"
