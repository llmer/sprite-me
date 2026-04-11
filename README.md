# sprite-me

Agent-first pixel art sprite generator. An open-source, self-hosted tool that lets AI coding agents (Claude Code, Cursor, Copilot, etc.) generate pixel art sprites and animations via MCP, with inference running on your own [RunPod](https://www.runpod.io/) GPUs.

## What It Does

Your AI agent can now:

- `generate_sprite("knight with longsword")` → a clean pixel-art PNG
- `animate_sprite(asset_id, "walk")` → a 6-frame walk cycle sprite sheet
- `import_image("/path/to/art.png")` → bring external art into the manifest
- Keep style consistent across a set using a "hero asset" reference pattern

All through standard [Model Context Protocol](https://modelcontextprotocol.io/) tools.

## Architecture

```
AI Agent -> MCP (stdio/SSE) -> sprite-me server -> RunPod ComfyUI workers
                                                    (FLUX + game-asset LoRA)
```

Full design: see [`docs/`](docs/).

## Models

- **Base**: [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (12B rectified flow transformer)
- **LoRA**: [Flux-2D-Game-Assets-LoRA](https://huggingface.co/gokaygokay/Flux-2D-Game-Assets-LoRA) (GRPZA trigger word, trained for game assets)
- **Background removal**: [rembg](https://github.com/danielgatis/rembg) (U2-Net)
- **Inference**: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on [RunPod Serverless](https://docs.runpod.io/tutorials/serverless/comfyui)

## Quick Start

### 1. Deploy the ComfyUI backend to RunPod

```bash
export REGISTRY=docker.io/yourusername
./scripts/deploy_runpod.sh
```

Then create a serverless endpoint from the pushed image. See [`docs/runpod-deployment.md`](docs/runpod-deployment.md) for details.

### 2. Install sprite-me and configure your editor

```bash
./scripts/setup.sh
```

This installs Python dependencies, prompts for your RunPod API key and endpoint ID, detects your editor, and writes the MCP config.

### 3. Use it from your AI agent

In Claude Code or Cursor, after restarting:

> "Generate a warrior character with a sword for my RPG, then make walk and attack animations."

The agent will call `generate_sprite`, then `animate_sprite` twice, saving all results to `./assets/`.

## Project Layout

```
docs/               # Full design documentation
src/sprite_me/      # Python package
  server.py         # MCP server (stdio/SSE)
  api.py            # FastAPI REST API
  tools/            # generate, animate, import, status, assets
  inference/        # RunPod client, ComfyUI workflow builder
  processing/       # Smart crop, background removal, sprite sheets, palette
  storage/          # Asset manifest and local file storage
docker/comfyui/     # Custom ComfyUI Docker image + workflow templates
skills/             # Agent skills (markdown) taught to coding agents
scripts/            # setup.sh, deploy_runpod.sh
tests/              # pytest unit tests
```

## Documentation

- [Architecture](docs/architecture.md)
- [Models & Inference Pipeline](docs/models.md)
- [MCP Server Design](docs/mcp-server.md)
- [RunPod Deployment Guide](docs/runpod-deployment.md)
- [Agent Skills](docs/agent-skills.md)
- [Project Structure](docs/project-structure.md)
- [References](docs/references.md)

## Configuration

Environment variables (all prefixed `SPRITE_ME_`):

| Variable | Default | Description |
|---|---|---|
| `SPRITE_ME_RUNPOD_API_KEY` | — | Your RunPod API key (required) |
| `SPRITE_ME_RUNPOD_ENDPOINT_ID` | — | Your ComfyUI endpoint ID (required) |
| `SPRITE_ME_ASSETS_DIR` | `./assets` | Where to store generated PNGs |
| `SPRITE_ME_MANIFEST_PATH` | `./sprite-me-assets.json` | Asset metadata manifest |
| `SPRITE_ME_MCP_TRANSPORT` | `stdio` | `stdio`, `sse`, or `streamable-http` |
| `SPRITE_ME_API_PORT` | `8420` | FastAPI server port |

## Development

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT
