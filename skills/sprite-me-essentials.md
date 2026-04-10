---
name: sprite-me-essentials
description: Core workflow rules for using sprite-me to generate pixel art and animations. Apply these defaults on every sprite-me call.
---

# sprite-me Essentials

Foundational rules for every sprite-me operation. Apply these automatically — the user should not have to remind you.

## Hero-Asset-First Pattern

When generating a set of related sprites (character + items, full inventory, tileset, etc.):

1. Generate the primary "hero" asset first with a detailed prompt that establishes style
2. Remember its `asset_id` — this becomes the style anchor
3. Generate all subsequent assets with `reference_asset_id=<hero_id>`
4. The tool automatically appends the hero's prompt to keep style consistent

**Never** generate a set of assets without a reference asset — the style will drift and they won't look like they belong together.

## Manifest Tracking

Keep a local project manifest of important asset IDs so they survive across sessions:

```json
// sprite-me-project.json
{
  "hero_character": "a1b2c3d4",
  "hero_idle_anim": "e5f6g7h8",
  "inventory_items": ["i9j0k1l2", "m3n4o5p6"]
}
```

Write this file into the user's project root when generating a set. Reference it in future sessions to avoid regenerating.

## Default Settings

| Setting | Default | When to change |
|---|---|---|
| `smart_crop_mode` | `"tightest"` | Use `"padded"` only for animation source frames |
| `remove_bg` | `true` | Set `false` if the user wants the white background |
| `lora_strength` | `0.85` | Lower (0.65) for more realistic style, higher (0.95) for stronger pixel-art |
| `steps` | `30` | Rarely change — only raise for final hero assets |
| `width`, `height` | `512` | `256` for small items/icons, `1024` for tilesets |

## Return Format

- Always return the local file `path` to the user, not base64 data
- When showing a sprite in chat, reference the path so they can preview it in their editor

## Seed Recording

Record the seed returned by each generation in a comment alongside the asset usage. This allows exact reproduction:

```python
# sprite-me: asset=a1b2c3d4, seed=1234567890
HERO_SPRITE = "assets/a1b2c3d4.png"
```

## Error Handling

- If a generation returns `{"error": ...}`, read the error and fix before retrying
- Don't blindly retry — check the RunPod endpoint is configured if you get connection errors
- If an asset lookup fails, call `list_assets` to see what's actually in the manifest
