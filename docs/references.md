# References & Prior Art

## Image Generation Models

### FLUX

- **FLUX.1-dev**: https://huggingface.co/black-forest-labs/FLUX.1-dev — 12B parameter rectified flow transformer by Black Forest Labs. Best quality open image model.
- **FLUX.1-schnell**: https://huggingface.co/black-forest-labs/FLUX.1-schnell — Faster variant, fewer steps needed.

### Game Asset LoRAs

- **Flux-2D-Game-Assets-LoRA**: https://huggingface.co/gokaygokay/Flux-2D-Game-Assets-LoRA — Trained with FAL Fast LoRA Trainer. Trigger word: `GRPZA`. Produces clean game assets on white backgrounds.
- **Flux-Game-Assets-LoRA-v2**: https://huggingface.co/gokaygokay/Flux-Game-Assets-LoRA-v2 — Updated version.
- **pixel-art-xl**: https://huggingface.co/nerijs/pixel-art-xl — SDXL-based pixel art LoRA.
- **Pixel Art Diffusion XL - Sprite Shaper**: https://civitai.com/models/277680/pixel-art-diffusion-xl — SDXL checkpoint for pixel art sprites.
- **flux-sprites (miike-ai)**: https://www.aimodels.fyi/models/replicate/flux-sprites-miike-ai — Replicate-hosted sprite generation model.

### Sprite Animation Research

- **Sprite Sheet Diffusion**: https://arxiv.org/abs/2412.03685 — "Generate Game Character for Animation" (March 2025). Uses Animate Anyone framework on SD v1.5 with ReferenceNet + Pose Guider + Motion Module.
  - Project page: https://chenganhsieh.github.io/spritesheet-diffusion/
- **Animyth**: https://github.com/ece1786-2023/Animyth — Combines GPT-4 for text processing with Stable Diffusion and ControlNet for sprite generation.
- **BitMapFlow**: https://github.com/Bauxitedev/bitmapflow — Open-source tool for generating inbetween frames for animated sprites, written in Godot-Rust.

## Background Removal

- **rembg**: https://github.com/danielgatis/rembg — Python library using U2-Net/SAM for background removal.
- **rembg ComfyUI node**: https://github.com/Jcd1230/rembg-comfyui-node — ComfyUI custom node wrapper.

## Post-Processing

- **PixelRefiner**: https://github.com/HappyOnigiri/PixelRefiner — Browser-based tool that cleans up AI-generated pixel art: removes anti-aliasing, detects grids, converts palettes, optimizes transparency.
- **Sprite-AI background removal**: https://www.sprite-ai.art/tools/remove-background
- **Pixel Refiner app**: https://www.pixel-refiner.app/

## RunPod & ComfyUI Deployment

- **RunPod serverless ComfyUI tutorial**: https://docs.runpod.io/tutorials/serverless/comfyui
- **RunPod ComfyUI worker**: https://github.com/runpod-workers/worker-comfyui — Docker images and serverless worker implementation.
- **RunPod ComfyUI + FLUX guide**: https://www.runpod.io/articles/guides/comfy-ui-flux
- **RunPod ComfyUI setup blog**: https://www.runpod.io/blog/stable-diffusion-comfyui-setup
- **Docker Hub image**: https://hub.docker.com/r/runpod/worker-comfyui
- **RunPod ComfyUI on serverless (DEV Community)**: https://dev.to/husnain/deploy-comfyui-with-runpod-serverless-1i25

## MCP (Model Context Protocol)

- **MCP Python SDK**: https://github.com/modelcontextprotocol/python-sdk — Official Python SDK. Decorator-based API with `FastMCP`/`MCPServer`. Supports stdio, SSE, and Streamable HTTP transports.
- **MCP Developer Guide**: https://particula.tech/blog/mcp-developer-guide
- **How to Build an MCP Server**: https://www.leanware.co/insights/how-to-build-mcp-server
- **MCP Server Architecture Guide (Stainless)**: https://www.stainless.com/mcp/api-mcp-server-architecture-guide
- **Building MCP servers for ChatGPT Apps (OpenAI)**: https://developers.openai.com/api/docs/mcp

## Internal Reference Project (vid)

Patterns we adopted from `/Users/jim/src/vid/`:

- **`vid/avatar_llmer/runpod.py::LiveRunpodPipeline`** — Async RunPod client with deadline-based polling via `asyncio.get_running_loop().time()`. Handles `{COMPLETED, FAILED, TIMED_OUT, CANCELLED}` terminal states and multi-format output extraction (`_extract_audio_bytes`, `_extract_video_bytes`). Our `runpod_client.py` mirrors this structure.
- **`vid/volumes.py`** — RunPod REST API wrapper for network volume CRUD: `list_volumes`, `create_volume`, `attach_volumes_to_endpoint`, `delete_volume`. Our `src/sprite_me/volumes.py` is a direct port with `sprite-me` naming.
- **`vid/sync_loras.py`** — S3-compatible LoRA uploader. Uses `boto3` against `https://s3api-{dc}.runpod.io/`, parallel `ThreadPoolExecutor`, `head_object` size-check before upload, multi-DC fan-out. Our `scripts/sync_models.py` is a direct port generalized to upload to any remote prefix.
- **`vid/Infinitetalk_Runpod_hub/entrypoint.sh`** — The critical pattern: on first boot download models to `/runpod-volume/models/`, write sentinel file, symlink subdirs into ComfyUI. On subsequent boots the sentinel triggers skip-download. Our `docker/comfyui/entrypoint.sh` follows the same flow but for the game-asset LoRA.
- **`vid/Infinitetalk_Runpod_hub/handler.py`** — Reference for custom handler patterns (idempotency via `request_key`, WebSocket progress from ComfyUI, multi-format input handling). We use the stock `runpod/worker-comfyui` handler rather than a custom one, but this is the upgrade path if we need finer control.

## Existing Open-Source Game Asset MCP Servers

- **game-asset-mcp**: https://github.com/MubarakHAlketbi/game-asset-mcp — Node.js MCP server for creating 2D/3D game assets using Hugging Face models. Uses `gokaygokay/Flux-2D-Game-Assets-LoRA` for 2D, InstantMesh/Hunyuan3D for 3D. Supports stdio and SSE transport.
  - Uses `@modelcontextprotocol/sdk`, `@gradio/client`, `@huggingface/inference`, `express`, `zod`, `sharp`
  - Tools: `generate_2d_asset`, `generate_3d_asset`
  - Resources: `asset://{type}/{filename}`

## Guides & Tutorials

- **Open-Source Image Generation Models 2026 (Pixazo)**: https://www.pixazo.ai/blog/top-open-source-image-generation-models
- **Open-Source Image Generation Models 2026 (BentoML)**: https://www.bentoml.com/blog/a-guide-to-open-source-image-generation-models
- **Z Image Turbo Pixel Art LoRA Guide**: https://apatero.com/blog/z-image-turbo-pixel-art-lora-complete-guide-2025
- **Training AI to Generate Sprites (OpenGameArt)**: https://opengameart.org/forumtopic/training-an-ai-to-generate-sprites
- **AI for Spritesheet Generation (OpenGameArt)**: https://opengameart.org/forumtopic/ai-for-spritesheet-generation
- **Create Pixel Art Sprite Sheets with AI (Toolify)**: https://www.toolify.ai/ai-news/create-pixel-art-sprite-sheets-with-ai-a-complete-guide-3579823
- **Game Asset Generator MCP Server guide**: https://skywork.ai/skypage/en/game-asset-generator-mcp-server/1980877040846041088
- **Replicate flux-2d-game-assets**: https://replicate.com/replicate/flux-2d-game-assets
