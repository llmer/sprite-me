#!/usr/bin/env bash
# Swap the Docker image on an existing sprite-me RunPod serverless template.
#
# Uses the REST API PATCH /templates/{id} with a new imageName, which
# triggers a rolling release on every endpoint using that template —
# no new template, no new endpoint, no .env change. Workers cycle to
# the new image on their next scale-up.
#
# Reads RUNPOD_API_KEY and SPRITE_ME_RUNPOD_TEMPLATE_ID from .env (or
# sources /Users/jim/src/vid/.env as a fallback, matching deploy_endpoint.sh).
#
# Usage:
#   IMAGE=bbbasddaaa/sprite-me-comfyui:v0.3.0 ./scripts/update_endpoint_image.sh
#   TEMPLATE_ID=2ywwx1vimd IMAGE=bbbasddaaa/sprite-me-comfyui:v0.3.0 \
#     ./scripts/update_endpoint_image.sh   # override auto-resolved template

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-}"

info() { printf "\033[36m[sprite-me]\033[0m %s\n" "$*"; }
error() { printf "\033[31m[sprite-me]\033[0m %s\n" "$*" >&2; }

if [ -z "$IMAGE" ]; then
  error "IMAGE env var required (e.g. IMAGE=bbbasddaaa/sprite-me-comfyui:v0.3.0)"
  exit 1
fi

# Pull RUNPOD_API_KEY + SPRITE_ME_RUNPOD_TEMPLATE_ID from the project .env
# without emitting them to the transcript.
if [ -f "$REPO_DIR/.env" ]; then
  set -a && source "$REPO_DIR/.env" && set +a
fi
if [ -z "${RUNPOD_API_KEY:-}" ] && [ -f /Users/jim/src/vid/.env ]; then
  set -a && source /Users/jim/src/vid/.env && set +a
fi

API_KEY="${RUNPOD_API_KEY:-${SPRITE_ME_RUNPOD_API_KEY:-}}"
TEMPLATE_ID="${TEMPLATE_ID:-${SPRITE_ME_RUNPOD_TEMPLATE_ID:-}}"

if [ -z "$API_KEY" ]; then
  error "No API key found. Set RUNPOD_API_KEY or SPRITE_ME_RUNPOD_API_KEY."
  exit 1
fi

if [ -z "$TEMPLATE_ID" ]; then
  error "No template ID. Set TEMPLATE_ID or SPRITE_ME_RUNPOD_TEMPLATE_ID in .env."
  exit 1
fi

info "Template:  $TEMPLATE_ID"
info "New image: $IMAGE"

RESP=$(curl -sS -X PATCH "https://rest.runpod.io/v1/templates/$TEMPLATE_ID" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"imageName\": \"$IMAGE\"}")

if echo "$RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'imageName' in d and d['imageName'].endswith('${IMAGE##*:}'):
    print('OK')
    sys.exit(0)
print(json.dumps(d, indent=2))
sys.exit(1)
" 2>/dev/null; then
  info "Template updated. Endpoint will rolling-release on next worker spawn."
  info "Trigger a cold start to pick up the new image:"
  info "  runpodctl serverless get \$SPRITE_ME_RUNPOD_ENDPOINT_ID"
else
  error "PATCH failed or unexpected response:"
  echo "$RESP" >&2
  exit 1
fi
