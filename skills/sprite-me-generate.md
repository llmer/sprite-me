---
name: sprite-me-generate
description: How to generate pixel art sprites effectively with sprite-me. Covers prompt engineering, resolution, consistency, and batch workflows.
---

# Generating Sprites with sprite-me

## Prompt Engineering

The tool automatically prepends the LoRA trigger word (`GRPZA`) and `"pixel art, game sprite"` to your prompt, and appends `"white background"`. You do **not** need to add these yourself.

**Good prompts:**
- `"warrior character with red cape and steel sword"`
- `"cute slime monster, green, eyes closed, smiling"`
- `"medieval wooden chest, closed, iron bindings"`
- `"grass tile with small flowers, top-down view"`

**Bad prompts:**
- `"pixel art"` (no subject)
- `"character"` (too vague, no details)
- `"warrior, pixel art, game sprite, white background"` (redundant — already added)
- `"photograph of a warrior"` (contradicts the LoRA style)

Include:
- **Subject type** (character, item, tile, prop)
- **Key visual features** (color, equipment, pose)
- **Camera angle** for non-character assets (top-down, side, front)

## Resolution Guidance

| Asset Type | Recommended Size |
|---|---|
| Main character | 512x512 |
| Large monster / boss | 512x512 or 768x768 |
| Inventory item / icon | 256x256 |
| Small prop | 256x256 |
| Tile (single) | 256x256 |
| Tileset (multi-tile) | 1024x512 or 1024x1024 |
| UI element (button, panel) | 512x256 or 512x512 |

## Consistency Workflow (Hero Pattern)

For a complete asset set:

```
1. Plan the assets needed (character + 5 items + 3 enemies)
2. generate_sprite(
     prompt="brave knight with blue cape and silver longsword, heroic stance"
   )
   -> hero_id = result.asset_id

3. For each remaining asset:
   generate_sprite(
     prompt="red healing potion in glass bottle",
     reference_asset_id=hero_id
   )
```

This keeps palette, outline style, and rendering technique consistent across the set.

## Batch Generation Pattern

Don't fire off 20 generations in parallel — RunPod serverless works best with sequential submissions. The endpoint auto-scales, but a flood of parallel requests causes thrashing.

```python
# Good: sequential
hero = await generate_sprite(prompt="knight", ...)
for item in inventory_items:
    await generate_sprite(prompt=item, reference_asset_id=hero["asset_id"])

# Bad: all parallel
await asyncio.gather(*[generate_sprite(...) for _ in range(20)])
```

## When to Adjust LoRA Strength

- `0.6-0.7` — more realistic, detailed rendering with less of the pixel-art-ish LoRA influence
- `0.8-0.9` (default 0.85) — balanced, clean game-ready assets
- `0.9-1.0` — maximum LoRA influence, strongest pixel art character, may look blocky

Lower strength if the output looks too stylized/blocky. Raise it if the output looks like a generic AI image instead of a game sprite.

## Seed Control

- Pass `seed` to reproduce the exact same sprite
- Leave `seed=None` for variety
- For a set of variations (3 different swords), use different seeds with the same prompt + reference asset

## Retro pixel-art mode

FLUX outputs smooth-shaded art by default. If you want **actual classic-game pixel sprites** (hard blocky pixels, 16-color palette, Game Boy aesthetic) set `pixelate=True` when calling `generate_sprite`.

```python
generate_sprite(
    prompt="warrior with longsword and shield",
    pixelate=True,
    pixel_size=64,       # 64x64 classic sprite. Try 32 for Game Boy, 16 for tiles.
    palette_size=16,     # 16 colors = classic NES/SNES. 8 = Game Boy. 32 = modern retro.
)
```

The pixelation runs after background removal, so the output is still a transparent-background sprite — just with crisp pixel blocks instead of smooth edges. Character identity and overall composition are preserved (it's a downsample, not a re-generation). Enable this when the user explicitly wants the retro aesthetic, the project is a pixel-art game, or the prompt includes words like "pixel art", "retro", "8-bit", "16-bit", "Game Boy".
