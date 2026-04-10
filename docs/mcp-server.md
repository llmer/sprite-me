# MCP Server Design

## Overview

The MCP server is the primary interface for AI coding agents. It exposes sprite generation and animation as MCP tools that agents can call directly from their IDE.

Built with the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) using the `FastMCP` / `MCPServer` decorator-based API.

## Transport Options

| Transport | Use Case | How to Connect |
|---|---|---|
| **stdio** (default) | Claude Code, Claude Desktop, local editors | Agent spawns the server process directly |
| **SSE** | Remote access, shared team server | Agent connects to `http://host:port/sse` |
| **Streamable HTTP** | Modern MCP clients | Agent connects to `http://host:port/mcp` |

## MCP Tools

### `generate_sprite`

Generate a pixel art sprite from a text prompt.

**Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | (required) | What to generate, e.g. "warrior character with sword" |
| `width` | int | 512 | Output width in pixels |
| `height` | int | 512 | Output height in pixels |
| `seed` | int | random | Seed for reproducibility |
| `smart_crop_mode` | string | "tightest" | "tightest" (tight bounds) or "padded" (extra margin) |
| `remove_bg` | bool | true | Remove white background, output transparent PNG |
| `reference_asset_id` | string | null | Use this asset's style as reference for consistency |
| `lora_strength` | float | 0.85 | LoRA influence strength (0.0-1.0) |
| `steps` | int | 30 | Inference steps (more = higher quality, slower) |

**Returns:** `{ asset_id, filename, path, prompt, seed, width, height }`

### `animate_sprite`

Generate an animated sprite sheet from an existing sprite asset.

**Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `asset_id` | string | (required) | Source sprite to animate |
| `animation` | string | "idle" | Preset: idle, walk, run, attack, jump, death, cast |
| `custom_prompt` | string | null | Override preset with custom animation description |
| `frames` | int | 6 | Number of animation frames |
| `edge_margin` | int | 6 | Pixel margin around each frame |
| `auto_enhance` | bool | true | Auto-enhance simple prompts with detailed descriptions |
| `seed` | int | random | Seed for reproducibility |

**Returns:** `{ asset_id, filename, path, animation, frames, source_asset_id, seed }`

### `import_image`

Import a local PNG or data URL as a sprite-me asset (required before animating external images).

**Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `source` | string | (required) | File path or `data:image/png;base64,...` URL |
| `name` | string | "" | Display name for the asset |

**Returns:** `{ asset_id, filename, path, width, height }`

### `check_status`

Check the status of a running generation job.

**Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `job_id` | string | (required) | RunPod job ID from a previous submission |

**Returns:** `{ job_id, status, delay_time, execution_time, completed }`

### `list_assets`

List all generated and imported sprite assets with metadata.

**Returns:** Array of asset objects with `{ asset_id, name, prompt, filename, asset_type, width, height, seed, reference_asset_id, frames, created_at }`

### `get_asset`

Get details for a specific asset including file path and download info.

**Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `asset_id` | string | (required) | The asset to retrieve |

**Returns:** Full asset object + `{ path, exists }`

## MCP Resources

| URI Pattern | Description |
|---|---|
| `asset://{asset_id}` | Direct access to asset image data |

## Agent Configuration

### Claude Code / Claude Desktop (stdio)

```json
{
  "mcpServers": {
    "sprite-me": {
      "command": "uv",
      "args": ["run", "sprite-me-server"],
      "env": {
        "SPRITE_ME_RUNPOD_API_KEY": "your-key",
        "SPRITE_ME_RUNPOD_ENDPOINT_ID": "your-endpoint-id"
      }
    }
  }
}
```

### Cursor (SSE / HTTP)

```json
{
  "sprite-me": {
    "url": "http://localhost:8420/mcp/",
    "headers": {}
  }
}
```

## Implementation Notes

The MCP Python SDK uses a decorator pattern:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sprite-me")

@mcp.tool()
async def generate_sprite(prompt: str, width: int = 512, ...) -> dict:
    """Generate a pixel art sprite from a text prompt."""
    ...

@mcp.resource("asset://{asset_id}")
async def get_asset_resource(asset_id: str) -> bytes:
    ...

if __name__ == "__main__":
    mcp.run(transport="stdio")  # or "sse" or "streamable-http"
```

Reference: https://github.com/modelcontextprotocol/python-sdk
