---
name: sprite-me-animate
description: How to generate pose-varied sprite sheet animations from an existing hero sprite using sprite-me's Kontext-backed animate_sprite tool.
---

# Animating Sprites with sprite-me

## The mental model

**`animate_sprite` takes a character you already have and produces that same character in different poses.** It does NOT generate a new character. The "hero sprite" you pass in is the ground truth for the character's look — armor, cape color, weapon, face, everything. Each frame of the resulting animation is the same character in a different stance.

Under the hood this uses FLUX.1 Kontext, a Black Forest Labs model purpose-built for image-guided editing. Character identity is preserved via reference-image conditioning, not via prompt engineering. You don't need to re-describe the character in your prompts.

## Prerequisite: an existing hero sprite

You must have a sprite in the sprite-me asset manifest before you can animate it. Two ways:

1. **Generate it**: `generate_sprite(prompt="knight with longsword and red cape")` returns an `asset_id`. Use that.
2. **Import it**: `import_image(source="/path/to/hero.png")` returns an `asset_id` for an existing PNG.

Never pass a file path or data URL directly to `animate_sprite` — it only takes `asset_id`.

## The basic flow

```python
# Step 1: generate the hero once, save the asset_id
hero = generate_sprite("brave knight with longsword and red cape, detailed armor")
# → {"asset_id": "abc123", "filename": "abc123.png", ...}

# Step 2: animate that hero into a walk cycle
anim = animate_sprite(asset_id="abc123", animation="walk")
# → {"asset_id": "def456", "filename": "def456_sheet.png", "frames": 8, ...}
```

You'll get back a single sprite-sheet PNG plus individual frame PNGs (`def456_frame00.png` through `def456_frame07.png`) in the assets directory. The sheet is assembled horizontally.

## Presets

Seven animation types are built in, each with a curated list of pose prompts:

| Preset | Frames | What it generates |
|---|---|---|
| `idle` | 4 | Standing still with subtle breathing/weight shifts |
| `walk` | 8 | Standard 8-frame walk cycle (side view) |
| `run` | 8 | Full sprint with high knees and ground clearance |
| `attack` | 6 | Weapon windup → swing → recover → ready |
| `jump` | 6 | Crouch → launch → airborne → peak → fall → land |
| `cast` | 6 | Spell prep → channel → release → recover |
| `death` | 6 | Hit → stagger → kneel → fall → down → defeated |

Call with `animate_sprite(asset_id=..., animation="walk")`. You can cap frame count with `frames=4` to trim a preset to the first N frames.

## Custom pose prompts

If you want a single specific pose that isn't in a preset, use `custom_prompt`:

```python
animate_sprite(
    asset_id="abc123",
    custom_prompt="crouching behind a stone shield, looking upward",
    frames=1,  # just one pose
)
```

**Custom prompt format**: describe the NEW pose only. Do NOT re-describe the character. The tool automatically appends a "keep the character, clothing, armor, weapon, palette, and art style unchanged" clause. Keep it short and specific — `"raising the sword overhead in a heroic pose"` is better than `"a knight with shiny silver armor and red cape raising a longsword"`.

## Quality expectations

This produces keyframes, not broadcast animation. Expect:

- **Strong character consistency** — same armor, same cape color, same weapon, same face. This is Kontext's main trained objective.
- **Decent pose fidelity** — the requested pose is interpreted loosely; you'll get something close to what you asked, not pixel-perfect.
- **Good style consistency** — the hero's art style (cartoon outlined, pixel art, whatever) carries through each frame. The LoRA that made the hero is not re-applied during animation — style flows from the reference image.
- **No temporal coherence between frames** — each frame is generated independently. For smooth 60fps animation you'll need a tween pass or manual cleanup.

These are usable as **keyframes for game engines** (Godot, Unity, Phaser) after light touch-up. They are not drop-in replacements for hand-animated sprites.

## Output dimensions

Kontext operates at 1024×1024 internally regardless of the hero's original size. Your 512×512 hero will be upscaled by `FluxKontextImageScale` before generation. If you need 512×512 output for your game engine, run each frame through sprite-me's post-processing (which does smart-crop + background removal) or resize externally.

## Retro pixel-art output

Kontext produces smooth painterly frames by default. For classic-game sprite aesthetics, set `pixelate=True` on `animate_sprite`:

```python
animate_sprite(
    asset_id=hero_id,
    animation="walk",
    pixelate=True,
    pixel_size=64,      # 64 classic, 32 Game Boy, 16 tile
    palette_size=16,    # 16 classic NES/SNES, 8 Game Boy, 32 modern retro
)
```

Each frame runs through a downsample → color-quantize → nearest-neighbor upsample pipeline after background removal. The result: crisp blocky pixels with a limited palette, ready for a retro game engine. Character identity is preserved (the pixel version is the same knight, just blocky). Use this whenever the hero was generated with `pixelate=True` OR the user's project is explicitly pixel-art style.

For best results the hero should be generated with `pixelate=True` too, so the hero and the animation share the same palette and grid resolution.

## Timing and cost

- First frame on a cold endpoint: ~2-5 minutes (cold start dominated)
- Subsequent frames on warm workers: ~20-40 seconds each
- 8-frame walk cycle warm path: ~3-5 minutes total, ~$0.05-0.10

Don't parallelize frame submissions. Sequential per-frame is more reliable and easier to debug than racing jobs against a warm-worker pool.

## Tips for better results

1. **Generate a clean, high-contrast hero**: busy or low-contrast hero sprites give Kontext less signal to lock onto. Detailed armor, bold cape color, clear silhouette = better preservation.

2. **Use the preset that matches your intent**: if you want an idle loop, `"idle"` gives 4 subtle variations that will tween well. If you ask for `"walk"` on an idle character that makes no sense (e.g., a character with no legs), you'll get unpredictable results.

3. **Pick ONE pose subject**: `custom_prompt` should describe ONE pose, not a sequence. For a sequence, use a preset or call multiple times.

4. **Seed matters**: same seed across frames helps consistency. Different seeds per frame give more variation. The tool uses the same seed for all frames in a single call; change `seed` between calls if you want variety.

5. **Animations won't magically have hands — Kontext can't add or remove limbs reliably**: if your hero has a two-handed sword, asking for "one hand raised" may produce weird hand geometry. Start with hero poses that have the limbs you'll need for the animation.

## Hero design rules (this matters more than anything else)

Kontext's character preservation is strong but not magical. It can change pose; it can't change body topology. The best animations come from heroes whose starting pose matches the limb usage of the target animation:

- **Walk/run cycle**: hero must have clearly visible legs in a neutral standing pose. If the hero is cropped at the waist or sitting, walk cycles will look broken.
- **Attack**: hero must have the weapon hand free and visible. A hero holding a weapon with both hands can still do an overhead swing but NOT a one-handed slash.
- **Jump**: hero needs both legs visible and not obstructed by a cape/robe flowing over them.
- **Cast**: hero needs at least one hand visible and free (not holding a weapon with both hands).
- **Death**: hero needs a full body — death animations show collapse, which requires showing the whole body falling. A bust/portrait hero can't do death.

**When in doubt, generate the hero with the animation in mind.** If the user asks for a walk cycle, prompt `generate_sprite` with `"standing knight facing forward, legs apart, longsword held in right hand"` rather than `"knight"`. The resulting hero will animate much better because its starting pose already supports the transformations Kontext needs to make.

**Limb-count rule**: Kontext can SHIFT existing limbs but struggles to ADD new ones. If the hero has two hands on a sword, a "one hand raised" pose will produce awkward hand geometry. Generate the hero in the hand-configuration you'll need most often in the animation.

**Weapon continuity**: the hero's weapon type must stay the same across frames (Kontext preserves it). You can't turn a longsword into a bow mid-animation.

## What animate_sprite does NOT do

- **Generate a new character from scratch** — use `generate_sprite` for that
- **Animate unrelated subjects together** (two characters interacting)
- **Produce frame-accurate walk cycles for pixel-perfect retro games** — the output is 1024×1024 smooth-shaded art that needs to be converted to pixels if you want a classic pixel-art look
- **Add or remove visual elements** — Kontext preserves what's there; it doesn't invent

If you need real-time generation, long-form character consistency across many scenes, or ControlNet-style skeletal pose control, those are separate pipelines. This tool is for the common case: "I have a hero sprite, give me 6-8 keyframes of them doing X."
