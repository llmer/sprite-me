---
name: sprite-me-generate
description: How to generate game sprites with sprite-me. Covers LoRA style selection, prompt engineering, cold-start timing, consistency, and batch workflows.
---

# Generating Sprites with sprite-me

## Start here: pick a LoRA style

sprite-me has four LoRAs and the right choice depends on what kind of game the user is building. **Always pick explicitly — don't rely on the default.** Use `list_loras()` to get the up-to-date list with descriptions; here's the summary:

| Key | Art style | Example use |
|---|---|---|
| `cartoon-vector` | Smooth cartoon/vector illustration. Heavy black outlines, flat colors with gradient shading, storybook polish. **NOT pixel art.** | Casual mobile game, storybook hero, mascot, polished UI icon |
| `pixel-indie` | Clean modern pixel art with visible grid, vibrant saturated colors, JRPG/indie-Steam aesthetic. Works for humans and creatures. | Stardew Valley-style game, indie Steam roguelike, modern pixel RPG |
| `pixel-retro` | Retro small-canvas 8/16-bit pixel art. Crunchy palette, NES/SNES feel. Often includes a scene background. | Authentic retro game, NES-era homage, Zelda-like |
| `top-down` | Bird's-eye perspective scenes with environment. NOT standalone sprites. | Overhead tiles, top-down maps, Zelda/Stardew camera angles |

### Mapping user intent to a LoRA

| If the user says... | Use this LoRA |
|---|---|
| "pixel art X" / "make a pixel sprite" | `pixel-indie` (default pixel choice) |
| "retro" / "8-bit" / "NES" / "classic" | `pixel-retro` |
| "cartoon" / "casual game" / "mascot" / "storybook" | `cartoon-vector` |
| "top-down" / "overhead" / "Zelda-style map" | `top-down` |
| "pokemon style" / "trainer sprite" | `pixel-indie` (the LoRA is named "trainer-sprites" internally but works broadly) |
| nothing specific about style | Ask them what they want, OR fall back to `cartoon-vector` for polished "hero sprite" output or `pixel-indie` if context suggests a pixel game |

**Don't guess silently.** If the user says "make me a knight" without style hints, it's okay to ask *"Do you want pixel art or a smooth cartoon look?"* — the cost of picking wrong is a wasted cold-start generation.

### LoRA naming caveat

The internal file names (`trainer-sprites-gen5`, `pixel-game-assets-dever`, `flux-2d-game-assets`) don't match what the LoRAs actually produce. Don't rely on the filenames — use the `list_loras` output or the table above. The keys in the table are the authoritative names.

## Cold start behavior — this matters for UX

Every LoRA swap triggers a ComfyUI cold start on the RunPod worker: **~3–5 minutes** to unload the old weights and load the new ones into GPU memory. Repeated calls with the same LoRA take **~20–40 seconds** each once warm.

**Implications for call ordering:**

1. **Group same-LoRA calls together.** If the user wants a knight + sword + potion all in the same style, make all three calls with the same LoRA back-to-back before switching styles. Don't interleave.
2. **Warn the user before the first call of a new LoRA.** Say something like *"This will take ~5 minutes to start since I'm loading a new art style model. Follow-up generations in the same style will be ~30 seconds each."*
3. **Don't fire the generate in parallel.** RunPod's autoscaler will spin up multiple cold workers if you submit several jobs simultaneously — each one pays its own cold start. Sequential is faster.
4. **If a user asks for a multi-style set** (knight in pixel art AND cartoon), explain the ~5 min per-style cost upfront so they can reconsider.

## Prompt engineering

Each LoRA has its own trigger word and preferred prompt shape — the tool handles that automatically. You supply only the subject description.

**Good prompts (all LoRAs):**
- `"brave knight with longsword and red cape"`
- `"chibi pink slime with big smile"`
- `"wooden treasure chest with iron bindings"`
- `"elven archer with longbow drawn"`

**Bad prompts:**
- `"pixel art warrior"` — redundant, the LoRA adds its own style tokens
- `"warrior"` — too vague
- `"photograph of a warrior"` — contradicts every LoRA style
- `"GRPZA knight"` — don't include trigger words, the tool adds them

Include:
- **Subject type** (character, item, tile, prop)
- **Key visual features** (color, equipment, pose, facial expression)
- **Camera angle** for non-character assets (top-down, side, front-facing)

## Resolution guidance

| Asset type | Size |
|---|---|
| Main character | 512×512 |
| Large monster / boss | 512×512 or 768×768 |
| Inventory item / icon | 256×256 |
| Small prop | 256×256 |
| Single tile | 256×256 |
| Tileset (multi-tile) | 1024×512 |

FLUX handles square and near-square aspect ratios well. Avoid extreme ratios (>2:1).

## Consistency workflow (hero pattern)

For a cohesive set, generate the "hero" first, then reference its asset_id in follow-up calls. This keeps palette, outline style, and rendering technique consistent across the set.

```
1. hero = generate_sprite(prompt="brave knight with blue cape and silver longsword", lora="pixel-indie")
2. potion = generate_sprite(prompt="red healing potion in glass bottle", lora="pixel-indie", reference_asset_id=hero.asset_id)
3. sword = generate_sprite(prompt="silver longsword with golden hilt", lora="pixel-indie", reference_asset_id=hero.asset_id)
```

All three stay warm on the same worker (~30s each) AND share the hero's palette hints via the reference.

## Retro pixel-art post-processing

Even `pixel-indie` and `pixel-retro` produce smooth-ish 512×512 output by default. For **actually crisp pixel blocks**, set `pixelate=True`:

```
generate_sprite(
    prompt="knight with longsword",
    lora="pixel-indie",
    pixelate=True,
    pixel_size=64,      # 64 = classic sprite. 32 = Game Boy. 16 = tile.
    palette_size=16,    # 16 = NES/SNES. 8 = Game Boy. 32 = modern retro.
)
```

The pixelation runs AFTER background removal, so you still get a transparent sprite — just with hard pixel blocks instead of smooth edges. Character identity is preserved (it's a downsample, not a re-generation).

**When to enable `pixelate=True`:**
- User says "pixel art" / "retro" / "8-bit" / "16-bit" / "Game Boy"
- Project is explicitly a pixel game
- Even `pixel-indie` or `pixel-retro` outputs feel too smooth for the target aesthetic

## Seed control

- Pass `seed` to reproduce the exact same sprite deterministically
- Leave `seed=None` for variety
- For a set of variations (3 different swords same style), use different seeds with the same prompt + lora

## Lora strength adjustment

Each LoRA has a default strength baked into its profile. Override via `lora_strength=` if:
- **Lower (0.6-0.7)** — output too stylized/cartoonish; you want more base FLUX fidelity
- **Higher (0.9-1.0)** — output looks too generic-AI; you want more LoRA character

Default is set per-profile — trust it until you have a reason not to.

## What `generate_sprite` records for later

When a sprite is saved to the manifest, sprite-me writes `asset.metadata["lora"]` = the LoRA key used. This matters for `animate_sprite`: when you animate that asset later, you (the agent) can call `get_asset(hero_id)` and read `metadata.lora` to know what body-shape topology the reference image has. The skill file for animation explains how to use this to compose appropriate pose prompts.
