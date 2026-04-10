#!/bin/bash
# sprite-me ComfyUI worker entrypoint.
#
# Model provisioning strategy (pattern from vid/Infinitetalk_Runpod_hub):
#   1. Check for a mounted network volume at /runpod-volume
#   2. On first boot: download LoRA(s) to the volume, write sentinel file
#   3. On subsequent boots: sentinel present, skip download
#   4. Symlink volume's model subdirs into /comfyui/models for ComfyUI
#
# Keeps the Docker image small (~12 GB base) and lets you update LoRAs
# via sync_models.py without rebuilding the image.

set -e

MODEL_DIR="/comfyui/models"
VOLUME_DIR="/runpod-volume"
VOLUME_MODEL_DIR="$VOLUME_DIR/models"
SENTINEL=".sprite_me_models_ready"

download_models() {
    local dest="$1"
    echo "[sprite-me] Downloading models to $dest"

    mkdir -p "$dest/loras" "$dest/checkpoints"

    # Flux-2D-Game-Assets-LoRA — our primary pixel art LoRA
    if [ ! -s "$dest/loras/flux-2d-game-assets.safetensors" ]; then
        wget -q --show-progress \
            -O "$dest/loras/flux-2d-game-assets.safetensors" \
            "https://huggingface.co/gokaygokay/Flux-2D-Game-Assets-LoRA/resolve/main/flux_2d_game_assets.safetensors"
    fi

    # Verify
    [ -s "$dest/loras/flux-2d-game-assets.safetensors" ] || {
        echo "[sprite-me] FAILED: LoRA download missing or empty" >&2
        exit 1
    }

    touch "$dest/$SENTINEL"
    echo "[sprite-me] Model download complete."
}

if [ -d "$VOLUME_DIR" ]; then
    echo "[sprite-me] Network volume detected at $VOLUME_DIR"
    if [ -f "$VOLUME_MODEL_DIR/$SENTINEL" ]; then
        echo "[sprite-me] Models already present on network volume."
    else
        download_models "$VOLUME_MODEL_DIR"
    fi

    # Symlink model subdirs from volume into ComfyUI's expected layout
    for subdir in loras checkpoints; do
        if [ -d "$VOLUME_MODEL_DIR/$subdir" ]; then
            # Preserve any existing model dirs (like the base FLUX checkpoint
            # that's baked into the image) by copying their contents into the
            # volume on first boot, then symlinking.
            if [ -d "$MODEL_DIR/$subdir" ] && [ ! -L "$MODEL_DIR/$subdir" ]; then
                # Copy any baked-in files we don't already have on the volume
                for f in "$MODEL_DIR/$subdir"/*; do
                    [ -e "$f" ] || continue
                    target="$VOLUME_MODEL_DIR/$subdir/$(basename "$f")"
                    [ -e "$target" ] || cp -r "$f" "$target"
                done
                rm -rf "$MODEL_DIR/$subdir"
            fi
            ln -sf "$VOLUME_MODEL_DIR/$subdir" "$MODEL_DIR/$subdir"
            echo "[sprite-me] Linked $MODEL_DIR/$subdir -> $VOLUME_MODEL_DIR/$subdir"
        fi
    done
else
    echo "[sprite-me] WARNING: No network volume at $VOLUME_DIR"
    echo "[sprite-me] WARNING: Models will download to local disk on every cold start."
    echo "[sprite-me] WARNING: Attach a network volume for persistent model storage."
    if [ ! -f "$MODEL_DIR/$SENTINEL" ]; then
        download_models "$MODEL_DIR"
    fi
fi

# Hand off to the upstream runpod/worker-comfyui start script.
# The base image's default command starts ComfyUI and the RunPod handler.
echo "[sprite-me] Starting ComfyUI worker..."
exec /start.sh
