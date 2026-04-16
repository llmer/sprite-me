---
name: sprite-me-animate
description: How to animate a hero sprite into a multi-frame sheet using sprite-me's Kontext-backed animate_sprite. Teaches pose prompt composition per character topology.
---

# Animating Sprites with sprite-me

## Mental model

`animate_sprite` takes a hero you already have and produces that same character in different poses, assembled into a sprite sheet. It does NOT generate a new character — the hero image is the ground truth for visual identity (armor, cape, weapon, face). Each frame is the same character in a different stance.

Under the hood this uses FLUX.1 Kontext, an image-guided editing model. Character identity is preserved via reference-image conditioning, not prompt engineering — **you don't need to re-describe the character in your pose prompts.**

## The v0.5.0 API — agent-composed pose prompts

`animate_sprite` takes a **list of pose prompts**, one per frame. There are no preset names. You compose the list based on the specific hero's body topology and the user's intent.

```
animate_sprite(
    asset_id="abc123",
    pose_prompts=[
        "Change the pose to standing still, arms at sides.",
        "Change the pose to walking forward, left foot lifted mid-stride, side view.",
        "Change the pose to mid-step with both feet near the ground.",
        "Change the pose to walking forward, right foot lifted mid-stride, side view.",
    ],
    seed=42,
)
```

The tool automatically appends a character-preservation clause ("Keep the exact same character, clothing, armor...") if you didn't include one, so you can focus on describing the pose change.

**If you pass the old `animation="walk"` or `custom_prompt="..."` params, the tool will error out with a clear message pointing back at this skill file.** Those are gone.

## Step 1: Read the hero before animating

Before composing pose prompts, call `get_asset(hero_id)` and inspect:

- **`prompt`** — the original text that generated the hero. Tells you what it is.
- **`metadata.lora`** — which LoRA generated it (`cartoon-vector`, `pixel-indie`, `pixel-retro`, or `top-down`). Tells you what style to expect.

Then classify the hero's **body topology**, which drives everything else:

| Prompt contains... | Topology | Animations that work |
|---|---|---|
| knight, warrior, wizard, archer, hero, soldier | **Full-body side-view humanoid** | walk, run, attack, jump, cast, death, idle |
| slime, blob, ball, creature | **Non-humanoid amorphous** | idle pulse, bounce, stretch — NO walk/attack |
| chest, crate, barrel, sign, rock | **Inanimate object** | idle shimmer, lid-open (2-3 frames) — decline complex motion |
| was generated with `lora="top-down"` | **Top-down perspective** | shift/rotate limbs from above — NEVER flip to side view |

**If the topology doesn't match the user's request, say so.** A chibi slime can't attack with a sword. A treasure chest can't walk. Tell the user what WILL work instead. Don't waste a 5-minute cold start generating garbage.

## Step 2: Compose pose prompts that match the topology

These are **recipes**, not presets. Copy, adapt, and modify them to fit the specific hero. Each prompt describes a pose *change* — Kontext expects edit-instruction form ("Change the pose to X"), not description form ("A knight walking").

### Side-view humanoid: 8-frame walk cycle

```python
[
    "Change the pose to left leg forward planted, right leg back, mid-stride walk, side view.",
    "Change the pose to left leg fully forward, right leg lifted behind, arms swinging, side view.",
    "Change the pose to both legs near together passing mid-step, weight centered, side view.",
    "Change the pose to right leg forward planted, left leg back, mid-stride walk, side view.",
    "Change the pose to right leg fully forward, left leg lifted behind, arms swinging, side view.",
    "Change the pose to both legs near together passing mid-step, weight centered, side view.",
    "Change the pose to left leg forward planted, right leg back, mid-stride walk, side view.",
    "Change the pose to left leg fully forward, right leg lifted behind, arms swinging, side view.",
]
```

### Side-view humanoid: 6-frame attack

```python
[
    "Change the pose to winding up the weapon, pulled back behind the body, shoulders rotated.",
    "Change the pose to raising the weapon overhead, arms extending forward.",
    "Change the pose to mid-swing, weapon horizontal at shoulder height, body leaning in.",
    "Change the pose to completing the swing, weapon extended forward at hip height, weight on front foot.",
    "Change the pose to recovering, weapon lowered across the body, stepping back.",
    "Change the pose to ready stance, weapon held in front, knees slightly bent.",
]
```

### Side-view humanoid: 4-frame idle (breathing)

```python
[
    "Change the pose to a neutral standing idle, weight shifted to the right foot, chest slightly lifted.",
    "Change the pose to a neutral standing idle, weight centered, chest at rest.",
    "Change the pose to a neutral standing idle, weight shifted to the left foot, chest slightly lifted.",
    "Change the pose to a neutral standing idle, weight centered, chest at rest.",
]
```

### Non-humanoid (slime, blob): 4-frame idle pulse

```python
[
    "Change the shape to squished slightly wider and flatter at the base, as if breathing out.",
    "Change the shape back to the neutral round form.",
    "Change the shape to stretched slightly taller and narrower, as if breathing in.",
    "Change the shape back to the neutral round form.",
]
```

Do NOT use "pose" for slimes — they don't have poses, they have shape deformations. Use "shape".

### Inanimate object: simple 2-3 frame animation

Treasure chest opening:

```python
[
    "Change the pose to the chest lid beginning to lift upward, a small gap visible between lid and body.",
    "Change the pose to the chest lid fully open, tilted back, revealing the interior.",
]
```

Most inanimate objects should only get 2-3 frames, or the agent should decline and say "that doesn't need to be animated — a single sprite is enough."

### Top-down character: rotate limbs, don't change view

The top-down hero is viewed from above. **Never ask Kontext to flip to side view** — it will rewrite the entire character. Instead:

```python
[
    "Shift the legs to a mid-stride position, still viewed from directly above.",
    "Shift the legs back to a standing neutral position, still viewed from directly above.",
]
```

Or rotate the whole character:

```python
[
    "Change the facing direction to north, still viewed from directly above.",
    "Change the facing direction to east, still viewed from directly above.",
    "Change the facing direction to south, still viewed from directly above.",
    "Change the facing direction to west, still viewed from directly above.",
]
```

## Hard rules (Kontext limitations)

These are things Kontext genuinely cannot do reliably. Don't try them.

1. **Don't change the camera view.** Side → top-down, front → back, etc. Kontext interprets "view" changes as "generate a different character" and identity collapses.
2. **Don't add or remove limbs.** If the hero is holding a sword with both hands, don't ask for "one hand raised" — Kontext will produce broken hand geometry. Generate a new hero with free hands if you need that pose.
3. **Don't swap weapons.** The weapon type stays constant across frames. If you need the hero to switch from sword to bow, that's two different heroes, not an animation.
4. **Don't mix topologies across frames.** Every frame is the same character in the same view/style. Frame 1 humanoid + frame 3 top-down doesn't work.
5. **Don't animate limb count changes for amorphous subjects.** A slime with no arms can't suddenly grow arms in frame 3.

## When to decline

It's okay — and better — to refuse an animation request than to waste ~5 minutes generating garbage. Refusal cases:

- **Hero body doesn't match requested action**: slime asked to walk, chest asked to attack, portrait asked to jump (no legs visible).
- **Limb count changes required**: two-handed sword hero asked for a one-handed action where the other hand needs to be free.
- **Camera-view flip required**: top-down asset asked for a side-view animation.
- **Inanimate object asked for complex motion**: chest asked for an attack animation.

Give the user a constructive alternative. *"That chibi slime doesn't have legs to walk with. I can do an idle bounce/pulse animation instead, or generate a new hero with visible legs for the walk cycle. Which would you like?"*

## Frame count heuristics

| Animation | Frames | Why |
|---|---|---|
| Idle loop | 4 | Subtle variation, short loop |
| Attack | 6 | Windup → strike → recover |
| Jump | 6 | Crouch → launch → air → peak → fall → land |
| Walk cycle | 8 | Full gait cycle (2×4 phases) |
| Run cycle | 8 | Same as walk but more extreme |
| Cast/spell | 6 | Prep → channel → release → recover |
| Death | 6 | Hit → stagger → fall → down |

Don't ask for more than 8 frames unless the user explicitly wants cinematic — every frame is a ~30s warm generation, and the sprite sheet won't loop smoothly at high frame counts anyway.

## Timing expectations

- **Warm frame**: ~20–40s per frame. 8-frame walk cycle ≈ 3–5 minutes total.
- **Cold start** (first frame after a break): +3–5 minutes one-time cost on the first frame.
- **LoRA warm state doesn't matter for animate**: Kontext uses its own model, not the generate-path LoRA. A warm `pixel-indie` generate worker is NOT a warm animate worker.
- Tell the user the expected time upfront.

## Retro pixel-art output

Kontext produces smooth painterly 1024×1024 frames regardless of the hero's source style. If the hero was pixel art or the user wants retro output, enable pixelation:

```
animate_sprite(
    asset_id=hero_id,
    pose_prompts=[...],
    pixelate=True,
    pixel_size=64,
    palette_size=16,
)
```

Each frame runs through a downsample → color-quantize → nearest-neighbor upsample pipeline after background removal. Character identity is preserved across frames (same palette → same character).

**When to enable pixelate for animation:**
- The hero was generated with `pixelate=True`
- The hero was generated with `lora="pixel-indie"` or `lora="pixel-retro"`
- The user's project is explicitly pixel-art style

## Tips

- **Seed consistency**: use the same `seed` across all frames in a call (tool does this automatically). Different seeds produce stylistic drift between frames.
- **Start with a clear hero**: high-contrast heroes with bold silhouettes animate better. Muddy/low-contrast heroes give Kontext less signal to lock onto.
- **Describe the POSE DELTA, not the character**: "Change the pose to walking forward" beats "A knight walking forward".
- **Don't fear the preservation suffix**: the tool auto-appends it if you forget, so your pose prompts can be short and specific.

## What animate_sprite does NOT do

- Generate new characters — use `generate_sprite` for that
- Multi-character interactions
- Pixel-perfect walk cycles for strict retro games (the 1024×1024 smooth output needs pixelation)
- Add or remove visual elements to the hero
- Animate unrelated subjects in a single call (one hero per call)

For real-time generation, skeletal pose control, or long-form scene consistency, those are separate pipelines. This tool is for: *"I have a hero sprite, give me 4-8 keyframes of them doing X, where X is compatible with their body topology."*
