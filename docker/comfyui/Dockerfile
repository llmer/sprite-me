# sprite-me ComfyUI worker image.
#
# Thin layer on top of the stock runpod/worker-comfyui image that bakes
# the Flux-2D-Game-Assets LoRA at build time. No runtime downloads, no
# entrypoint override — the base image's /start.sh handles everything.
#
# The LoRA filename must match docker/comfyui/workflows/pixel_art_generate.json:
#   "lora_name": "flux-2d-game-assets.safetensors"
#
# Built by .github/workflows/docker-image.yml on tag push (docker-v*).

FROM runpod/worker-comfyui:5.0.0-flux1-dev

# Source filename on HF is game_assets_v3.safetensors (~90 MB). We rename
# on download so the local filename matches what workflow_builder.py and
# pixel_art_generate.json reference.
RUN mkdir -p /comfyui/models/loras && \
    wget -q -O /comfyui/models/loras/flux-2d-game-assets.safetensors \
      "https://huggingface.co/gokaygokay/Flux-2D-Game-Assets-LoRA/resolve/main/game_assets_v3.safetensors" && \
    test -s /comfyui/models/loras/flux-2d-game-assets.safetensors
