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

# Tell ComfyUI to ALSO read models from the mounted network volume.
# We inherit the upstream CMD ["/start.sh"] and WORKDIR /comfyui — no
# entrypoint override (attempt with a custom entrypoint.sh broke ComfyUI
# startup in a way I haven't fully diagnosed; the yaml alone gives us
# the benefit we need).
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
