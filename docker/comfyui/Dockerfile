# sprite-me ComfyUI worker image — thin.
#
# Models are NOT baked into this image. They live on a RunPod network volume
# mounted at /runpod-volume/models/ and are exposed to ComfyUI via
# extra_model_paths.yaml. To update a model or add a LoRA/checkpoint, run
# scripts/sync_models.py instead of rebuilding this image.
#
# Iteration cycle compared to a fat image:
#   - Code changes (workflow_builder, tools, skills):     no image rebuild needed
#   - Model updates (new LoRA or checkpoint):              volume sync (~seconds-minutes)
#   - Custom node changes / entrypoint changes:            rebuild this image (~4-5 min)
#
# Rationale and attempt-1 post-mortem: docs/examples/volume-approach/README.md.

FROM runpod/worker-comfyui:5.8.5-base-cuda12.8.1

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
