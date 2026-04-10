# Volume-approach (attempt-1 artifacts)

This directory holds the Docker files from our first deployment attempt, where we tried to keep the image slim by storing the LoRA on a RunPod **network volume** and downloading it at worker startup. It didn't work and we replaced it with the simpler "bake the LoRA into the image" approach used by the current `docker/comfyui/Dockerfile`.

Keeping these files as a reference in case you ever want to revisit the volume approach for a valid reason — see *When would you still want this?* below.

## Files

- **`Dockerfile`** — extends `runpod/worker-comfyui:3.6.0-flux1-dev`, installs the `rembg-comfyui-node`, copies a custom `entrypoint.sh`, and overrides `ENTRYPOINT`.
- **`entrypoint.sh`** — at worker boot: detect `/runpod-volume`, download the LoRA if missing, write a sentinel file, symlink `/runpod-volume/models/{loras,checkpoints}` into `/comfyui/models/`, then `exec /start.sh` to hand off to the upstream RunPod worker handler.

## What went wrong (attempt 1, 2026-04-10)

We created a template + endpoint + 10 GB network volume in US-IL-1 and watched workers go `desiredStatus: EXITED` / UNHEALTHY immediately on every submission. We never got a single job to complete. Suspects, in order of likelihood:

1. **`--docker-start-cmd` bash one-liner was flaky.** The one-liner we passed to `runpodctl template create` was a multi-statement bash script with `wget`, square-bracketed `echo [sprite-me]` strings (bash tries to glob-expand brackets), and `&&` chains, all wedged into a single `-c` string that went through runpodctl → GraphQL → Docker → bash quoting. Any of those layers could have mangled it. Even if the quoting survived, the wget-then-symlink logic is doing three things the base image's `/start.sh` knows nothing about.
2. **Runtime LoRA download (~30s for 160 MB)** may have tripped RunPod's health check before `/start.sh` got a chance to run. The health check is lenient but the hand-off protocol is unclear — we couldn't confirm whether the worker is considered healthy during `entrypoint.sh` setup or only after `/start.sh` is up.
3. **Volume DC pinning**. Attaching a network volume to a serverless endpoint locks that endpoint's workers to the volume's datacenter. We chose US-IL-1 based on the pod-stock display, but pod capacity ≠ serverless capacity, and US-IL-1 may simply have had no 4090 serverless slots at that moment. We never got a clean signal either way.
4. **A bonus disaster**: a parallel agent spawned a GPU pod with a hallucinated Docker image tag (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-runtime-ubuntu22.04` — doesn't exist) to pre-seed the volume via SSH. The pod was billed at $0.99/hr while the container failed to pull, costing ~$0.44 before we killed it. Lesson: don't let sub-agents touch live billing without a narrow, verified tool list.

## What we did instead (attempt 2)

Bake the LoRA into a thin custom image at Docker build time. No network volume. No `docker-start-cmd` override. No runtime orchestration. The resulting Dockerfile is 15 lines. Build + push happens on GitHub Actions (fat network pipe to Docker Hub). See `docker/comfyui/Dockerfile` and `docs/runpod-deployment.md` in the repo root for the current approach.

## When would you still want this?

The volume approach has one real advantage: **fast LoRA swaps without rebuilding the image**. Keep it in mind if any of these become true:

- **You have many LoRAs** (dozens) and want to pick one at request time
- **Your LoRAs change frequently** (daily or hourly)
- **Your LoRAs are huge** (multi-GB checkpoints, not 160 MB adapters)
- **You want to share model storage** across multiple endpoints in the same datacenter

If you go back to this approach, the right way to do it is:

1. **Separate the model-seeding step from the worker start**. Don't put `wget` in the start command. Use a one-shot CPU pod (`ubuntu:22.04`, attached to the volume) to download models once, then delete the pod. The serverless worker only needs to symlink.
2. **Keep the start override trivial.** If you must override `docker-start-cmd`, make it a single `ln -sf ... && exec /start.sh` with no conditionals, no wget, no echoes with brackets.
3. **Confirm the target datacenter has serverless capacity for your GPU** before creating the volume. Check via the RunPod dashboard's serverless endpoint creation UI — it only offers DCs that can actually schedule the GPU you picked. CLI/API is less reliable at reporting this.
4. **Use the S3 API** (`https://s3api-{dc}.runpod.io/`) to pre-populate the volume from your local machine via `boto3` — see the original `scripts/sync_models.py` in git history.

Until one of those conditions applies, prefer baking the LoRA into the image.
