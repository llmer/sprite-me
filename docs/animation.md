# Animation: character-consistent sprite sheets via FLUX Kontext

## What `animate_sprite` actually does

Takes a hero sprite you already generated (or imported) and produces a
sequence of frames showing the **same character** in different poses,
assembled into a horizontal sprite sheet.

Character identity — armor design, cape color, weapon, face, art style —
is preserved **via Kontext's reference-image conditioning**, not via
repeated prompt engineering. You describe only the *pose delta*; the
character's visual signature flows from the reference image.

Example output (4-frame walk cycle from a generated knight hero):

```
hero → animate_sprite(asset_id=hero_id, animation="walk", frames=4)
     → 4 frames of the same knight: left-leg-planted, left-lifted, together, right-planted
     → assembled 4×1 sprite sheet
```

## How it works

Two-model architecture:

1. **`generate_sprite`** still runs on **FLUX.1-dev fp8** with the
   **Flux-2D-Game-Assets LoRA**. This is where the stylized pixel-art /
   game-asset look comes from.

2. **`animate_sprite`** runs on **FLUX.1 Kontext dev fp8** (separate
   checkpoint). Kontext is BFL's image-guided generation model — it
   takes a reference image + an edit instruction and produces a new
   image that preserves the reference's visual identity while applying
   the requested change.

Both models live on the same RunPod network volume. The Docker image
is a thin wrapper around `runpod/worker-comfyui:5.8.5-flux1-dev-fp8`
with an `extra_model_paths.yaml` that points ComfyUI at
`/runpod-volume/models/` for everything beyond what the base image
ships.

```
/runpod-volume/models/
  diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors   # Kontext UNet, 12 GB
  text_encoders/clip_l.safetensors                             # shared w/ FLUX
  text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors            # shared w/ FLUX
  vae/ae.safetensors                                           # shared w/ FLUX
  loras/flux-2d-game-assets.safetensors                        # generate flow only
  checkpoints/flux1-dev-fp8.safetensors                        # generate flow (combined)
```

### Pipeline per frame

```
hero.png (uploaded via handler's images field)
    │
    ▼
LoadImage("hero.png")
    │
    ▼
FluxKontextImageScale       (auto-scales to Kontext's preferred input size)
    │
    ▼
VAEEncode(ae.safetensors) → reference_latent
    │                         │
    │                         ├─→ ReferenceLatent ──→ FluxGuidance(2.5) ──→ KSampler.positive
    │                         │         ↑
    │                         │    CLIPTextEncode(pose prompt)
    │                         │         │
    │                         │         └─→ ConditioningZeroOut ──→ KSampler.negative
    │                         │
    │                         └─→ KSampler.latent_image
    │
    ▼
UNETLoader(flux1-dev-kontext_fp8_scaled) ──→ KSampler.model
    │
    ▼
KSampler (seed, steps=20, cfg=1, scheduler=simple, denoise=1)
    │
    ▼
VAEDecode → SaveImage
```

The critical trick: the VAE-encoded reference latent goes into BOTH
`ReferenceLatent` (so the text conditioning "knows about" the reference)
AND `KSampler.latent_image` (so the sampling process starts from a
noise-seeded copy of the reference). With `denoise=1` the KSampler
actually denoises from scratch, but the reference conditioning from
`ReferenceLatent` keeps pulling the output toward the character's
visual identity at every step.

### Prompt format

Kontext expects edit-instruction style, not description style. The
builder uses a two-part format:

```
<pose change sentence>. Keep the exact same character, clothing, armor,
weapon, color palette, and art style unchanged. White background.
```

The preservation suffix is appended automatically by `animate_sprite`
(or the preset library). Callers supply only the pose delta.

Good pose prompt: `"Change the pose to walking forward, left foot lifted off the ground, mid-stride, side view."`

Bad pose prompt: `"A knight with silver armor and red cape walking forward."` — this re-describes the character, which Kontext will interpret as "generate something new matching this description" and may lose consistency.

## Presets

Seven animation types, defined in `src/sprite_me/animation_presets.py`:

| Preset | Frames | Notes |
|---|---|---|
| `idle` | 4 | Subtle breathing/weight shifts. Good for idle loops. |
| `walk` | 8 | 8-frame side-view walk cycle, standard game-dev convention |
| `run` | 8 | Faster sprint, knees-up, both feet off ground at peak |
| `attack` | 6 | Windup → swing → recover → ready |
| `jump` | 6 | Crouch → launch → airborne → peak → fall → land |
| `cast` | 6 | Spell prep → channel → release → return |
| `death` | 6 | Hit → stagger → kneel → fall → down → defeated |

Each preset is a list of pose prompts with the preservation suffix already
appended. Frame counts are canonical — you can cap with `frames=N` but
the tool won't stretch a 4-frame idle to 8 frames automatically.

## API

### MCP tool signature (unchanged from v0.1.x — public contract)

```python
animate_sprite(
    asset_id: str,
    animation: str = "idle",        # preset name OR any label when custom_prompt is set
    custom_prompt: str | None = None,  # single pose description, overrides preset
    frames: int | None = None,      # cap or expand (default: preset length)
    edge_margin: int | None = None, # post-processing crop margin
    auto_enhance: bool | None = None,  # no-op, kept for back-compat
    seed: int | None = None,        # same seed across frames improves consistency
) -> dict
```

Additional internal parameters (available but not exposed over MCP):
- `steps: int = 20` (Kontext canonical default)
- `guidance: float = 2.5` (FluxGuidance scale, Kontext default)

Return shape:
```python
{
    "asset_id": "abc123",
    "filename": "abc123_sheet.png",
    "path": "/Users/.../assets/abc123_sheet.png",
    "animation": "walk",
    "frames": 8,
    "source_asset_id": "hero_id_here",
    "seed": 42,
}
```

## Performance

| Scenario | Timing |
|---|---|
| Cold start (first frame, no warm workers) | 2-7 min |
| Warm frame | ~20-40s |
| 4-frame walk cycle, cold start | ~8-9 min |
| 8-frame walk cycle, cold start | ~10-12 min |
| 8-frame walk cycle, all warm | ~3-5 min |

First-frame cold start is dominated by:
1. Worker image pull (~4-5 min, ~23 GB)
2. ComfyUI startup + UNETLoader loading Kontext from the volume (~30-60s)
3. Actual sampling (~15-20s)

Subsequent frames on the same warm worker reuse the loaded models
(only the reference image changes per frame, and that's small).

## Cost

- Cold start (one-time): ~$0.05
- Warm frame: ~$0.003 per frame on RTX 5090
- 8-frame walk cycle warm: ~$0.025
- 8-frame walk cycle cold: ~$0.08

Network volume storage adds a fixed ~$3.60/month regardless of usage.

## Known limitations

**Pose fidelity is approximate.** You ask for "left foot forward" and
you usually get something that reads as left-foot-forward but may not
be pixel-perfect. Kontext interprets pose instructions liberally.

**No temporal coherence.** Frames are generated independently. Frame 3
doesn't "know" about frames 2 and 4. You get keyframes, not smooth
animation. For a game loop you'll need a tween pass (in your engine
or via a dedicated inbetween tool).

**Pixel art → smooth art drift.** Kontext outputs 1024×1024 smooth-
shaded images. If your hero is crisp pixel art, Kontext will soften
it slightly during the reference-latent denoise. For strict pixel-art
consistency you'll need to re-pixelate each frame externally.

**Limb-count changes fail badly.** If your hero has two hands on a
sword and you ask for "one hand raised", Kontext produces weird hand
geometry. Animations work best when the limb topology stays constant.

**No multi-character scenes.** Kontext works on one subject at a time.
Two interacting characters would need a different approach.

**No pose control skeleton.** If you need exact bone positions, this
tool can't do that. Add FLUX ControlNet OpenPose to the image and
wire it into the workflow as an additional conditioning signal.
Possible future upgrade.

## Iteration cycle

Because models live on the network volume, upgrading Kontext (new
checkpoint release, better text encoder, etc.) does NOT require a
Docker image rebuild:

```bash
# Download the new Kontext fp8 locally or via CPU pod
# Upload to the volume via scripts/sync_models.py or direct wget
# Next worker cold-start picks it up automatically
```

Code changes (prompt presets, post-processing, workflow topology)
don't require any infrastructure changes — purely client-side Python.

Docker image changes (custom nodes, extra_model_paths.yaml edits,
entrypoint changes) require a GHA rebuild + template update.

## Further reading

- [FLUX.1 Kontext model card](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) — BFL's documentation
- [ComfyUI Kontext tutorial](https://docs.comfy.org/tutorials/flux/flux-1-kontext-dev) — canonical workflow
- [Comfy-Org/flux1-kontext-dev_ComfyUI](https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI) — fp8 checkpoint source
- `docs/runpod-deployment.md` — how the thin-image + network-volume architecture works
- `docs/architecture.md` — the two-model pipeline in sprite-me overall
- `src/sprite_me/animation_presets.py` — the per-frame pose prompt library
- `src/sprite_me/tools/animate.py` — the orchestration code
- `src/sprite_me/inference/workflow_builder.py` — `build_animate_workflow` — the Kontext node graph constructor
