# sprite-me ComfyUI worker image — fat base + volume for additions.
#
# We use the FLUX variant base (which already has torch + ComfyUI runtime
# + the FLUX.1-dev fp8 checkpoint baked in). Models on the network volume
# are exposed to ComfyUI via extra_model_paths.yaml — no runtime symlink
# or entrypoint override needed.
#
# Iteration cycle benefits:
#   - Adding/updating LoRAs:    volume sync, no rebuild   (~1 min)
#   - Adding new checkpoints:   volume sync, no rebuild   (~5 min)
#   - Code/workflow changes:    no infrastructure touched (instant)
#   - Custom node changes:      rebuild this image        (~10 min)
#
# Rationale and attempt-1 post-mortem: docs/examples/volume-approach/README.md.

FROM runpod/worker-comfyui:5.8.5-flux1-dev-fp8

# Custom nodes for the AnimateDiff + IP-Adapter animation path (v0.6.0).
#   - AnimateDiff-Evolved: the maintained ComfyUI port of AnimateDiff,
#     provides ADE_LoadAnimateDiffModel / ADE_UseEvolvedSampling / etc.
#   - IPAdapter-plus: cubiq's IP-Adapter node pack, provides
#     IPAdapterUnifiedLoader / IPAdapterAdvanced for reference-image
#     conditioning that preserves hero identity across AnimateDiff frames.
# TODO: pin to specific commits once the first successful build confirms
# node compatibility. For now, tracking main — reproducibility caveat.
RUN set -eux; \
    cd /comfyui/custom_nodes; \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git; \
    git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus.git; \
    for d in ComfyUI-AnimateDiff-Evolved ComfyUI_IPAdapter_plus; do \
      if [ -f "$d/requirements.txt" ]; then \
        pip install --no-cache-dir -r "$d/requirements.txt"; \
      fi; \
    done

# Tell ComfyUI to ALSO read models from the mounted network volume.
# We inherit the upstream CMD ["/start.sh"] and WORKDIR /comfyui — no
# entrypoint override (attempt with a custom entrypoint.sh broke ComfyUI
# startup in a way I haven't fully diagnosed; the yaml alone gives us
# the benefit we need).
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
