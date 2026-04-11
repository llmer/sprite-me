#!/usr/bin/env bash
# Create a sprite-me RunPod serverless template + endpoint and attach
# network volumes. Models are NOT baked into the image — they live on
# the attached volume(s) and ComfyUI reads them via extra_model_paths.yaml.
# (See docker/comfyui/Dockerfile for the thin-image architecture.)
#
# Requires:
#   - runpodctl installed (https://cli.runpod.net)
#   - RUNPOD_API_KEY in env (or sourced from /Users/jim/src/vid/.env)
#   - Docker image pushed to Docker Hub (GitHub Actions handles this)
#   - One or more pre-seeded network volumes. Create via:
#       runpodctl network-volume create --name sprite-me-v2 \
#         --size 50 --data-center-id EU-RO-1
#     and seed via a CPU pod or scripts/sync_models.py.
#
# Usage (env-var driven):
#   IMAGE=bbbasddaaa/sprite-me-comfyui:v0.2.0 \
#   VOLUME_IDS=vol_abc123,vol_def456 \
#   GPU_ID="NVIDIA GeForce RTX 5090" \
#   ./scripts/deploy_endpoint.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-bbbasddaaa/sprite-me-comfyui:latest}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 5090}"
TEMPLATE_NAME="${TEMPLATE_NAME:-sprite-me}"
ENDPOINT_NAME="${ENDPOINT_NAME:-sprite-me}"
# Thin image is ~16 GB compressed. 30 GB container disk is plenty for
# extraction + ComfyUI temp files + outputs (models live on the volume).
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-30}"
# Comma-separated list of network volume IDs to attach to the endpoint.
# At least one is required — the endpoint needs a volume with the models
# for ComfyUI to load anything.
VOLUME_IDS="${VOLUME_IDS:-}"

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
  exit 1
fi

if [ -z "$VOLUME_IDS" ]; then
  error "VOLUME_IDS not set. Pass a comma-separated list of network volume IDs."
  error "List existing volumes with: runpodctl network-volume list"
  exit 1
fi

info "Image:     $IMAGE"
info "GPU:       $GPU_ID"
info "Volumes:   $VOLUME_IDS"

# 1. Create the serverless template
info "Creating serverless template '$TEMPLATE_NAME'..."
TEMPLATE_JSON=$(runpodctl template create \
  --name "$TEMPLATE_NAME" \
  --image "$IMAGE" \
  --serverless \
  --container-disk-in-gb "$CONTAINER_DISK_GB")
TEMPLATE_ID=$(echo "$TEMPLATE_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
info "Template ID: $TEMPLATE_ID"

# 2. Create the serverless endpoint.
# workers-max must be > 1 so the scheduler can spread across machines;
# max=5 avoids "throttled" state on a single pinned machine. See:
#   https://www.answeroverflow.com/m/1192648582847807539
info "Creating serverless endpoint '$ENDPOINT_NAME'..."
ENDPOINT_JSON=$(runpodctl serverless create \
  --name "$ENDPOINT_NAME" \
  --template-id "$TEMPLATE_ID" \
  --gpu-id "$GPU_ID" \
  --workers-min 0 \
  --workers-max 5)
ENDPOINT_ID=$(echo "$ENDPOINT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
info "Endpoint ID: $ENDPOINT_ID"

# 3. Attach network volumes. runpodctl doesn't support this flag on
# serverless create (as of 2.1.x), so we PATCH the REST API.
info "Attaching volume(s) to endpoint..."
VOLUME_JSON=$(python3 -c "
import json, sys
ids = '$VOLUME_IDS'.split(',')
print(json.dumps({'networkVolumeIds': [v.strip() for v in ids if v.strip()]}))
")
ATTACH_RESP=$(curl -sX PATCH "https://rest.runpod.io/v1/endpoints/$ENDPOINT_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$VOLUME_JSON")
ATTACHED=$(echo "$ATTACH_RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
ids = d.get('networkVolumeIds', [])
print(','.join(ids) if ids else 'NONE')
" 2>/dev/null || echo "PARSE_ERROR")
if [ "$ATTACHED" = "NONE" ] || [ "$ATTACHED" = "PARSE_ERROR" ]; then
  error "Volume attach failed. Response: $ATTACH_RESP"
  exit 1
fi
info "Attached: $ATTACHED"

# 4. Write IDs to .env (overwrites previous values)
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
#   Volumes:   $VOLUME_IDS
#   GPU:       $GPU_ID, max-workers=5, scale-to-zero
EOF

info ""
info "Done. Next step:"
info "  cd $REPO_DIR && uv run scripts/test_endpoint.py"
