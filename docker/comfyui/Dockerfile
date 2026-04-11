# sprite-me ComfyUI worker image — fat base + volume for additions.
#
# We use the FLUX variant base (which already has torch + ComfyUI runtime
# + the FLUX.1-dev fp8 checkpoint baked in) NOT the -base-cuda12.8.1 variant.
# The "base" variant has CUDA but no Python ML deps installed, and bolting
# torch + ComfyUI requirements on top isn't worth the ~3.5 GB savings.
#
# What this image gives us:
#   - All FLUX.1-dev fp8 deps (torch, comfy, transformers, etc.) ready at boot
#   - flux1-dev-fp8.safetensors baked in /comfyui/models/checkpoints/ (default)
#   - Custom entrypoint.sh + extra_model_paths.yaml that ALSO read from
#     /runpod-volume/models/ — so we can ADD models (Kontext, new LoRAs)
#     via scripts/sync_models.py without rebuilding the image
#
# Iteration cycle benefits:
#   - Adding/updating LoRAs:         volume sync, no rebuild           (~1 min)
#   - Adding new checkpoints (e.g.   volume sync, no rebuild           (~5 min)
#     Kontext for animation):
#   - Code/workflow JSON changes:    no infrastructure touched         (instant)
#   - Custom node / entrypoint:      rebuild this image                (~10 min)
#
# Cold-start cost stays at ~7 min for the first pull on a fresh machine,
# then flashboot + cached layers make subsequent workers fast.
#
# Rationale and attempt-1 post-mortem: docs/examples/volume-approach/README.md.

FROM runpod/worker-comfyui:5.8.5-flux1-dev-fp8

# rembg custom node for background removal at inference time (also used
# client-side in sprite_me.processing.background as a fallback).
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/Jcd1230/rembg-comfyui-node.git && \
    cd rembg-comfyui-node && \
    pip install --no-cache-dir -r requirements.txt

# Tell ComfyUI to read models from the mounted network volume in addition
# to the image-local /comfyui/models/ tree.
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml

# Custom entrypoint: sanity-check the volume mount, then hand off to the
# upstream runpod/worker-comfyui start script. Baked into the image as a
# proper file (NOT a --docker-start-cmd one-liner — attempt 1 taught us
# that quoting through runpodctl/Docker/bash is unreliable).
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
