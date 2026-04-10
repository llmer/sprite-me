# Custom ComfyUI image for sprite-me.
#
# Models are NOT baked into this image. On first boot the entrypoint
# downloads the LoRA to a RunPod network volume, then symlinks it into
# ComfyUI's models dir. Subsequent boots reuse the volume, keeping cold
# starts fast and allowing LoRA updates without rebuilding the image.
#
# Pattern mirrored from /Users/jim/src/vid/Infinitetalk_Runpod_hub/entrypoint.sh

FROM runpod/worker-comfyui:3.6.0-flux1-dev

# Install rembg custom node for background removal at inference time
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/Jcd1230/rembg-comfyui-node.git && \
    cd rembg-comfyui-node && \
    pip install --no-cache-dir -r requirements.txt

# Copy entrypoint and workflow templates
COPY entrypoint.sh /entrypoint.sh
COPY workflows/ /comfyui/workflows/sprite-me/
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
