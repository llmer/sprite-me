#!/bin/bash
# sprite-me ComfyUI worker entrypoint.
#
# Responsibilities:
#   1. Sanity-check that a RunPod network volume is mounted
#   2. Emit a log summary of what ComfyUI will see (helpful for debugging)
#   3. Hand off to the upstream /start.sh which runs ComfyUI + RunPod handler
#
# Models are NOT downloaded here. They must be pre-seeded on the network
# volume via scripts/sync_models.py (or a one-time wget in a CPU pod).
# extra_model_paths.yaml (baked into the image) tells ComfyUI to scan
# /runpod-volume/models/ in addition to /comfyui/models/.
#
# If the volume isn't attached or is empty, ComfyUI will still start but
# workflows that reference missing models will fail with clear errors.
# This is preferable to retrying downloads at boot (attempt 1 path) which
# caused silent health-check timeouts.

set -e

VOLUME_ROOT="/runpod-volume"
VOLUME_MODELS="$VOLUME_ROOT/models"

echo "[sprite-me] entrypoint starting"

if [ -d "$VOLUME_ROOT" ]; then
    echo "[sprite-me] /runpod-volume mounted"
else
    echo "[sprite-me] WARNING: /runpod-volume NOT mounted — workflows will fail" >&2
    echo "[sprite-me] WARNING: attach a network volume to this endpoint" >&2
fi

if [ -d "$VOLUME_MODELS" ]; then
    echo "[sprite-me] /runpod-volume/models contents:"
    find "$VOLUME_MODELS" -maxdepth 2 -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.ckpt" \) 2>/dev/null | sort | head -20
    count=$(find "$VOLUME_MODELS" -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.ckpt" \) 2>/dev/null | wc -l)
    echo "[sprite-me] $count model file(s) visible on volume"
else
    echo "[sprite-me] NOTE: /runpod-volume/models/ does not exist yet"
fi

if [ -f /comfyui/extra_model_paths.yaml ]; then
    echo "[sprite-me] /comfyui/extra_model_paths.yaml:"
    cat /comfyui/extra_model_paths.yaml | sed 's/^/  /'
fi

echo "[sprite-me] handing off to /start.sh"
exec /start.sh
