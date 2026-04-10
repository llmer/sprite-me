# sprite-me Architecture

## Overview

sprite-me is an open-source, self-hosted alternative to [SpriteCook](https://www.spritecook.ai/) with an agent-first design. AI coding agents (Claude Code, Cursor, Copilot, etc.) connect via MCP to generate pixel art sprites and animations, with inference running on your own RunPod GPUs.

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
│  Slim Docker image (no models baked in):         │
│    runpod/worker-comfyui-flux1-dev +             │
│    rembg custom node +                           │
│    entrypoint.sh (symlinks volume models)        │
│                                                  │
│  Auto-scaling: 0 -> N workers on demand          │
│  GPU: NVIDIA 3090/4090 (~$0.22-0.40/hr)         │
└──────┬──────────────────────────────────────────┘
       │ mounted at /runpod-volume
       ▼
┌─────────────────────────────────────────────────┐
│  RunPod Network Volumes (one per datacenter)    │
│                                                  │
│  /runpod-volume/models/                          │
│    loras/flux-2d-game-assets.safetensors         │
│    checkpoints/... (optional custom checkpoints) │
│                                                  │
│  Updated via S3 API:                             │
│    scripts/sync_models.py sync ./models/loras    │
│                                                  │
│  Managed via REST API:                           │
│    sprite-me-volumes setup | list | teardown     │
└─────────────────────────────────────────────────┘
```

**Why network volumes instead of baking models into the image?** Same reasoning as the `vid` project's `Infinitetalk_Runpod_hub`: keeps the Docker image small (~12 GB vs 25 GB+), lets you update LoRAs via `sync_models.py` without rebuilding and pushing a new image, and cold starts are faster because workers don't redownload models on every boot. The entrypoint symlinks the volume's model subdirs into ComfyUI's expected paths on startup.

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
| Style consistency | Hero-asset + reference_asset_id pattern | Same approach as SpriteCook — generate one asset, use it as style reference for the rest |
| Model distribution | RunPod network volumes, not Docker layers | Smaller images, faster cold starts, LoRA updates without image rebuilds (pattern from `vid/Infinitetalk_Runpod_hub`) |
| Polling | Deadline-based with monotonic clock | Terminal states handled, time spent in each poll doesn't inflate budget (pattern from `vid/avatar_llmer/runpod.py`) |

## How SpriteCook Works (for reference)

SpriteCook's architecture (from their [skills repo](https://github.com/SpriteCook/skills) and [agent docs](https://www.spritecook.ai/agents)) splits into three layers:

1. **MCP Server** at `https://api.spritecook.ai/mcp/` — exposes generation and animation tools
2. **Agent Skills** — markdown files that teach agents best practices (hero-asset-first, manifest tracking, crop defaults)
3. **Backend API** — handles inference, post-processing, asset storage

Their skills define three capabilities:
- **spritecook-workflow-essentials** — core rules (check credits, use presigned URLs, maintain manifest, default `smart_crop_mode="tightest"`, default model `gemini-3.1-flash-image-preview`)
- **spritecook-generate-sprites** — generation guidance (reference_asset_id for consistency, edit_asset_id for modifications)
- **spritecook-animate-assets** — animation guidance (import first then animate by asset_id, edge_margin=6, auto_enhance_prompt=true)

We replicate this three-layer pattern with open-source models and self-hosted inference.
