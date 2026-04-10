# Project Structure

```
sprite-me/
├── docs/                                  # Design documentation (you are here)
│   ├── architecture.md                    # System architecture and design decisions
│   ├── models.md                          # Models, LoRAs, and inference pipeline
│   ├── mcp-server.md                      # MCP server tool/resource definitions
│   ├── runpod-deployment.md               # RunPod serverless deployment guide
│   ├── agent-skills.md                    # Agent skill design and patterns
│   ├── project-structure.md               # This file
│   └── references.md                      # All external references and prior art
│
├── pyproject.toml                         # Python project config (uv/pip)
│
├── docker/
│   └── comfyui/
│       ├── Dockerfile                     # Custom ComfyUI image with LoRAs
│       └── workflows/
│           ├── pixel_art_generate.json    # Text -> pixel art sprite workflow
│           ├── pixel_art_animate.json     # Sprite -> animation sheet workflow
│           └── remove_background.json     # Background removal workflow
│
├── src/
│   └── sprite_me/
│       ├── __init__.py
│       ├── config.py                      # Settings via env vars (pydantic-settings)
│       ├── server.py                      # MCP server entry point (stdio/SSE)
│       ├── api.py                         # FastAPI REST API server
│       │
│       ├── tools/                         # MCP tool implementations
│       │   ├── generate.py                # generate_sprite — text to pixel sprite
│       │   ├── animate.py                 # animate_sprite — asset to sprite sheet
│       │   ├── import_asset.py            # import_image — local file to asset
│       │   ├── status.py                  # check_status — poll job progress
│       │   └── assets.py                  # list/get/delete asset management
│       │
│       ├── inference/                     # RunPod/ComfyUI integration
│       │   ├── runpod_client.py           # Async RunPod serverless API client
│       │   └── workflow_builder.py        # Build ComfyUI workflow JSON dynamically
│       │
│       ├── processing/                    # Image post-processing pipeline
│       │   ├── crop.py                    # Smart crop (detect bounds, trim whitespace)
│       │   ├── background.py              # Background removal (rembg or threshold)
│       │   ├── spritesheet.py             # Assemble/split sprite sheets
│       │   └── palette.py                 # Palette reduction, grid snap
│       │
│       └── storage/                       # Asset persistence
│           ├── local.py                   # Local filesystem storage
│           └── manifest.py                # JSON manifest tracking asset metadata
│
├── skills/                                # Agent skill files (markdown)
│   ├── sprite-me-essentials.md            # Core workflow rules
│   ├── sprite-me-generate.md              # Sprite generation best practices
│   └── sprite-me-animate.md               # Animation workflow guidance
│
├── tests/
│   ├── test_workflow_builder.py           # Workflow JSON construction tests
│   ├── test_manifest.py                   # Asset manifest CRUD tests
│   ├── test_processing.py                 # Image processing tests
│   └── test_tools.py                      # Tool integration tests (mocked RunPod)
│
├── scripts/
│   ├── setup.sh                           # One-line MCP config installer
│   └── deploy_runpod.sh                   # Build + push Docker, create endpoint
│
└── assets/                                # Generated sprites (gitignored)
    └── .gitkeep
```

## Layer Responsibilities

| Layer | Purpose | Dependencies |
|---|---|---|
| **MCP Server** (`server.py`) | Expose tools to AI agents via MCP protocol | MCP Python SDK, tools layer |
| **API Server** (`api.py`) | REST API for direct HTTP access | FastAPI, tools layer |
| **Tools** (`tools/`) | Business logic for each operation | inference, processing, storage |
| **Inference** (`inference/`) | Submit workflows to RunPod, retrieve results | httpx, RunPod API |
| **Processing** (`processing/`) | Post-process generated images | Pillow, rembg |
| **Storage** (`storage/`) | Persist assets and metadata | filesystem, JSON |
| **Skills** (`skills/`) | Teach agents how to use tools | (plain markdown, no code) |

## Data Flow

```
Agent calls generate_sprite("warrior with sword")
    │
    ▼
tools/generate.py
    ├── Builds workflow via inference/workflow_builder.py
    ├── Submits to RunPod via inference/runpod_client.py
    ├── Waits for result (polling)
    ├── Post-processes via processing/crop.py, processing/background.py
    ├── Saves via storage/local.py
    ├── Records in storage/manifest.py
    └── Returns { asset_id, filename, path, ... }
```
