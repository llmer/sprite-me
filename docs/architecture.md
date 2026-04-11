# sprite-me Architecture

## Overview

sprite-me is an open-source, self-hosted pixel art sprite generator with an agent-first design. AI coding agents (Claude Code, Cursor, Copilot, etc.) connect via MCP to generate pixel art sprites and animations, with inference running on your own RunPod GPUs.

## System Architecture

```
┌─────────────────────────────────────────────────┐
│  AI Coding Agent                                │
│  (Claude Code / Cursor / VS Code Copilot / etc) │
└──────────────────┬──────────────────────────────┘
                   │ MCP Protocol (stdio or SSE)
                   │
┌──────────────────▼──────────────────────────────┐
│  sprite-me MCP Server (Python)                  │
│                                                  │
│  Tools:                                          │
│    generate_sprite    - text -> pixel art sprite │
│    animate_sprite     - asset -> sprite sheet    │
│    import_image       - local file -> asset      │
│    check_status       - poll job progress        │
│    list_assets        - browse generated assets  │
│    get_asset          - get asset details + path │
│                                                  │
│  Resources:                                      │
│    asset://{asset_id} - direct asset access      │
│                                                  │
│  Skills (teach agents best practices):           │
│    sprite-me-essentials                          │
│    sprite-me-generate                            │
│    sprite-me-animate                             │
└──────────────────┬──────────────────────────────┘
                   │ Internal REST API
                   │
┌──────────────────▼──────────────────────────────┐
│  sprite-me API Server (FastAPI)                  │
│                                                  │
│  POST /api/generate     - generate sprite        │
│  POST /api/animate      - animate sprite         │
│  POST /api/import       - import image           │
│  GET  /api/jobs/{id}    - job status             │
│  GET  /api/assets       - list assets            │
│  GET  /api/assets/{id}  - get asset              │
│  DELETE /api/assets/{id} - delete asset           │
│                                                  │
│  Post-processing pipeline:                       │
│    Smart crop -> Background removal ->           │
│    Palette cleanup -> Sprite sheet assembly      │
│                                                  │
│  Asset storage:                                  │
│    Local filesystem + JSON manifest              │
│    (optional S3/R2 for remote access)            │
└──────────────────┬──────────────────────────────┘
                   │ RunPod Serverless API
                   │ POST /run, GET /status/{id}
                   │
┌──────────────────▼──────────────────────────────┐
│  ComfyUI Workers on RunPod Serverless            │
│                                                  │
│  Docker image (~23 GB):                          │
│    runpod/worker-comfyui:5.8.5-flux1-dev-fp8 +   │
│    extra_model_paths.yaml (reads volume models)  │
│                                                  │
│  Auto-scaling: 0 -> 5 workers on demand          │
│  GPU: NVIDIA RTX 5090 (~$0.99/hr)                │
└──────┬──────────────────────────────────────────┘
       │ mounted at /runpod-volume
       ▼
┌─────────────────────────────────────────────────┐
│  RunPod Network Volume (EUR-NO-1, 50 GB)        │
│                                                  │
│  /runpod-volume/models/                          │
│    ── generate flow (FLUX.1-dev fp8) ──          │
│    checkpoints/flux1-dev-fp8.safetensors         │
│    loras/flux-2d-game-assets.safetensors         │
│                                                  │
│    ── animate flow (FLUX.1 Kontext fp8) ──       │
│    diffusion_models/flux1-dev-kontext_fp8.safet  │
│    text_encoders/clip_l.safetensors              │
│    text_encoders/t5xxl_fp8_e4m3fn_scaled.safet   │
│    vae/ae.safetensors                            │
│                                                  │
│  Updates via CPU pod + wget OR S3 API            │
│  (scripts/sync_models.py)                        │
└─────────────────────────────────────────────────┘
```

**Why network volumes + a fat base image?** Models are data, not code. Docker is optimized for small layered code deployments, not multi-gigabyte binary blobs. Storing model weights on a RunPod network volume lets us update models without rebuilding the image: code changes require zero infrastructure work, LoRA updates are a simple file sync, and only custom-node / entrypoint changes require a full rebuild. We keep the fat `runpod/worker-comfyui:5.8.5-flux1-dev-fp8` base (instead of the thin `5.8.5-base-cuda12.8.1`) because the base variant ships without `torch` or ComfyUI's Python dependencies — bolting them on manually wasn't worth the ~7 GB savings. The `extra_model_paths.yaml` baked into the image tells ComfyUI to scan `/runpod-volume/models/` alongside its default `/comfyui/models/` tree.

## Two-model architecture (generate + animate)

sprite-me runs **two separate FLUX checkpoints** for two different jobs:

| Tool | Checkpoint | Purpose |
|---|---|---|
| `generate_sprite` | `flux1-dev-fp8.safetensors` (combined) + `flux-2d-game-assets.safetensors` LoRA | Create a new stylized game-asset sprite from a text prompt |
| `animate_sprite` | `flux1-dev-kontext_fp8_scaled.safetensors` (split UNet) + separate text encoders + VAE, NO LoRA | Take an existing hero sprite and produce it in different poses |

Both checkpoints live on the same network volume. Workers load whichever one the current workflow references via `UNETLoader` or `CheckpointLoaderSimple`. See `docs/animation.md` for the Kontext pipeline in detail.

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Agent interface | MCP (Model Context Protocol) | Universal standard supported by Claude Code, Cursor, VS Code Copilot, Windsurf, Codex, etc. |
| Server language | Python (FastAPI) | Best ML ecosystem (diffusers, Pillow, rembg), matches MCP Python SDK |
| Inference backend | ComfyUI on RunPod Serverless | Workflow-based (easy to iterate), pre-built Docker images, auto-scaling, battle-tested |
| Base model | FLUX.1-dev | Best quality open image model, strong LoRA ecosystem, RunPod has pre-built images |
| Pixel art LoRA | Flux-2D-Game-Assets-LoRA | Specifically trained for game assets, clean white backgrounds, GRPZA trigger word |
| Animation approach | Pose-conditioned multi-frame generation | Start simple (per-frame with seed pinning), upgrade path to Sprite Sheet Diffusion |
| Storage | Local filesystem + JSON manifest | Simple, no external deps, easy to inspect and version control |
| Style consistency | Hero-asset + reference_asset_id pattern | Generate one asset, use it as style reference for the rest |
| Model distribution | RunPod network volumes, not Docker layers | Smaller images, faster cold starts, LoRA updates without image rebuilds (pattern from `vid/Infinitetalk_Runpod_hub`) |
| Polling | Deadline-based with monotonic clock | Terminal states handled, time spent in each poll doesn't inflate budget (pattern from `vid/avatar_llmer/runpod.py`) |

## Three-layer pattern

The overall shape is a three-layer architecture:

1. **MCP Server** — exposes generation and animation tools to AI agents
2. **Agent Skills** — markdown files that teach agents best practices (hero-asset-first, manifest tracking, crop defaults)
3. **Backend API** — handles inference, post-processing, asset storage

Built with open-source models and self-hosted inference on RunPod.
