# Models & Inference Pipeline

## Models Required

### 1. Pixel Art Sprite Generation

**Base Model: FLUX.1-dev**
- Source: https://huggingface.co/black-forest-labs/FLUX.1-dev
- Architecture: Rectified flow transformer, 12B parameters
- Why: Best quality open image generation model as of 2025-2026. Strong LoRA support. RunPod has pre-built Docker images (`runpod/worker-comfyui:*-flux1-dev`).
- Alternative: FLUX.1-schnell (faster, fewer steps needed, but lower quality)

**LoRA: Flux-2D-Game-Assets-LoRA**
- Source: https://huggingface.co/gokaygokay/Flux-2D-Game-Assets-LoRA
- Trigger word: `GRPZA`
- Trained with: FAL Fast LoRA Trainer
- Output: Clean game-ready assets with consistent white backgrounds
- Strength: 0.85 recommended
- What it produces: Characters, items, tiles, UI elements, props — all isolated on white
- v2 available: https://huggingface.co/gokaygokay/Flux-Game-Assets-LoRA-v2

**Alternative LoRA: pixel-art-xl**
- Source: https://huggingface.co/nerijs/pixel-art-xl
- Base: SDXL (not FLUX) — would require switching base model
- Good for retro-style pixel art specifically

**Alternative LoRA: Pixel Art Diffusion XL - Sprite Shaper**
- Source: https://civitai.com/models/277680/pixel-art-diffusion-xl
- Base: SDXL checkpoint (full model, not just LoRA)
- Community-trained, specifically for sprite shapes

### 2. Sprite Animation

**Approach A: Per-frame generation with consistency (recommended to start)**
- Use FLUX + LoRA to generate a wide image containing all frames as a sprite sheet
- Prompt: `"GRPZA, pixel art sprite sheet, [character], [action] animation, N frames, white background, consistent character"`
- Split the output into individual frames
- Consistency via: same seed, same LoRA, reference prompt anchoring
- Pros: Simple, uses same model as generation, no extra models
- Cons: Consistency between frames is imperfect

**Approach B: Sprite Sheet Diffusion (advanced, future upgrade)**
- Paper: https://arxiv.org/abs/2412.03685
- Project page: https://chenganhsieh.github.io/spritesheet-diffusion/
- Based on Animate Anyone framework, built on SD v1.5
- Three components:
  - **ReferenceNet**: Encodes character appearance from a reference image using modified SD v1.5 with spatial-attention layers + CLIP image encoding
  - **Pose Guider**: 4 convolution layers that align pose images with noise latent resolution
  - **Motion Module**: Embedded in Res-Trans blocks after spatial/cross-attention layers, models smooth transitions between frames
- Generates full sprite sheets in one pass with character consistency
- Cons: Based on SD 1.5 (lower quality), research code may not be production-ready

**Approach C: Video diffusion models (experimental)**
- Use models like Seedance, Pixverse, or Stable Video Diffusion to generate short character animation clips
- Extract frames and convert to sprite sheet
- Tools like AutoSprite (https://www.autosprite.io/) and Segmind Pixelflow (https://www.segmind.com/pixelflows/ai-sprite-sheet-maker) do this commercially

### 3. Background Removal

**rembg**
- Source: https://github.com/danielgatis/rembg
- Python library, uses U2-Net or SAM models
- Reliable for removing white/solid backgrounds from sprites
- ComfyUI node available: https://github.com/Jcd1230/rembg-comfyui-node

**Simple fallback**: Threshold-based white background removal (no ML needed, works for clean white backgrounds from the LoRA)

### 4. Post-Processing (no ML, just image processing)

**PixelRefiner concepts** (https://github.com/HappyOnigiri/PixelRefiner):
- Remove anti-aliasing artifacts from AI-generated pixel art
- Auto-detect pixel grid size
- Snap to nearest palette colors
- These are Pillow-based operations, not separate models

### 5. Style Consistency (no extra model needed)

SpriteCook's pattern, which we replicate:
1. Generate one "hero asset" first (the primary character/item)
2. Use that asset's ID as `reference_asset_id` for all subsequent generations
3. The system prepends the hero asset's prompt to new generation prompts
4. Combined with seed pinning and the same LoRA settings, this produces visually consistent sets

Future upgrade: IP-Adapter for FLUX — use the hero asset image directly as visual conditioning (not just prompt-based). This would require adding IP-Adapter nodes to the ComfyUI workflow.

## Inference Pipeline

```
User prompt
    │
    ▼
Prompt construction:
  "{GRPZA}, pixel art, game sprite, {user_prompt}, white background"
    │
    ▼
ComfyUI Workflow execution on RunPod:
  CheckpointLoader (FLUX.1-dev-fp8)
    → LoraLoader (flux-2d-game-assets, strength=0.85)
    → CLIPTextEncode (positive prompt)
    → CLIPTextEncode (negative prompt)
    → EmptyLatentImage (512x512)
    → KSampler (euler, 30 steps, cfg=3.5)
    → VAEDecode
    → SaveImage
    │
    ▼
Post-processing (Python, on API server):
  1. Smart crop (detect content bounds, trim whitespace)
  2. Background removal (rembg or threshold)
  3. Palette reduction (optional, quantize to N colors)
  4. Grid snap (optional, remove sub-pixel anti-aliasing)
    │
    ▼
Storage:
  Save PNG to ./assets/{asset_id}.png
  Update sprite-me-assets.json manifest
```

## Model Size & GPU Requirements

| Model | Size | Min VRAM | Recommended GPU |
|---|---|---|---|
| FLUX.1-dev (fp8) | ~12 GB | 16 GB | RTX 3090 (24 GB) |
| FLUX.1-dev (fp16) | ~24 GB | 24 GB | RTX 4090 (24 GB) |
| FLUX.1-schnell (fp8) | ~12 GB | 16 GB | RTX 3090 (24 GB) |
| Flux-2D-Game-Assets-LoRA | ~150 MB | (loaded into base model) | — |
| rembg (U2-Net) | ~170 MB | CPU or any GPU | — |

RunPod pricing (approximate):
- RTX 3090 (24 GB): ~$0.22/hr community, ~$0.44/hr secure
- RTX 4090 (24 GB): ~$0.40/hr community, ~$0.69/hr secure
- Serverless: pay per second of compute, auto-scales to zero
