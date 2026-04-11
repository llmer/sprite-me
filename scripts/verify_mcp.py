"""Programmatic MCP round-trip verification.

Spawns the sprite-me MCP server over stdio, initializes a client session,
lists the advertised tools, and verifies that the critical ones
(generate_sprite, animate_sprite, import_image, list_assets, get_asset)
are all present with sensible input schemas.

This is a substitute for manually restarting Claude Code to confirm the
MCP integration works — it uses the same protocol Claude Code does, but
does it programmatically so you can catch regressions before shipping.

Usage:
    uv run scripts/verify_mcp.py
    uv run scripts/verify_mcp.py --call-generate   # also does a full round-trip
    uv run scripts/verify_mcp.py --call-generate --prompt "wizard with staff"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


# Tools the sprite-me MCP server must expose. Verified against server.py:
#   @mcp.tool() generate_sprite / animate_sprite / import_image /
#                check_status / list_assets / get_asset / delete_asset
_EXPECTED_TOOLS = {
    "generate_sprite": {
        "prompt",
        "pixelate",  # added in the recent pixelation work
    },
    "animate_sprite": {
        "asset_id",
        "animation",
        "pixelate",
    },
    "import_image": {"source"},
    "check_status": {"job_id"},
    "list_assets": set(),
    "get_asset": {"asset_id"},
    "delete_asset": {"asset_id"},
}


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}", file=sys.stderr)


async def verify_tools(session: ClientSession) -> list[str]:
    """List tools from the server and assert each expected tool is present
    with its expected input-schema params.

    Returns a list of failure messages (empty = pass).
    """
    print("\n[1/3] Listing tools from sprite-me MCP server...")
    tools = await session.list_tools()
    tool_names = sorted(t.name for t in tools.tools)
    print(f"  Server advertises {len(tool_names)} tool(s): {', '.join(tool_names)}")

    failures: list[str] = []

    for expected_name, expected_params in _EXPECTED_TOOLS.items():
        tool = next((t for t in tools.tools if t.name == expected_name), None)
        if tool is None:
            _fail(f"{expected_name!r} not registered")
            failures.append(f"missing tool: {expected_name}")
            continue

        # Input-schema params are under inputSchema.properties
        schema = tool.inputSchema or {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        actual_params = set(props.keys())

        missing = expected_params - actual_params
        if missing:
            _fail(f"{expected_name!r} missing expected params: {sorted(missing)}")
            failures.append(f"{expected_name} missing {sorted(missing)}")
        else:
            _ok(f"{expected_name}  ({len(actual_params)} params)")
    return failures


async def verify_resources(session: ClientSession) -> list[str]:
    """Sprite-me exposes asset://{asset_id} as a resource template."""
    print("\n[2/3] Listing resources from sprite-me MCP server...")
    failures: list[str] = []
    try:
        templates = await session.list_resource_templates()
        template_uris = [t.uriTemplate for t in templates.resourceTemplates]
        print(f"  Resource templates: {template_uris}")
        if not any("asset://" in t for t in template_uris):
            _fail("asset://{asset_id} template not registered")
            failures.append("no asset:// template")
        else:
            _ok("asset://{asset_id} template present")
    except Exception as e:
        # Resource templates are optional in MCP; treat as warning, not failure
        print(f"  (list_resource_templates not implemented: {e})")
    return failures


async def verify_call_generate(
    session: ClientSession,
    prompt: str,
) -> list[str]:
    """Actually call generate_sprite and verify the result shape.

    This does a real round-trip to RunPod. Requires .env with a working
    endpoint and will cost a few cents. Only runs when --call-generate
    is passed.
    """
    print(f"\n[3/3] Calling generate_sprite(prompt={prompt!r})...")
    failures: list[str] = []
    t0 = time.time()
    try:
        result = await session.call_tool(
            "generate_sprite",
            arguments={
                "prompt": prompt,
                "seed": 42,
                "steps": 20,
                "width": 512,
                "height": 512,
            },
        )
    except Exception as e:
        _fail(f"generate_sprite call failed: {e}")
        return [f"generate_sprite error: {e}"]

    elapsed = time.time() - t0
    print(f"  call completed in {elapsed:.1f}s")

    if not result.content:
        _fail("generate_sprite returned empty content")
        return ["empty result"]

    # FastMCP returns the dict as a TextContent with JSON
    first = result.content[0]
    if hasattr(first, "text"):
        print(f"  result text: {first.text[:200]}")

    structured = getattr(result, "structuredContent", None)
    if structured:
        print(f"  structured keys: {sorted(structured.keys())}")
        # Verify the expected shape
        for expected in ("asset_id", "filename", "path", "seed"):
            if expected not in structured:
                _fail(f"result missing {expected!r}")
                failures.append(f"missing {expected}")
            else:
                _ok(f"result has {expected}")
    return failures


async def main() -> int:
    parser = argparse.ArgumentParser(description="Programmatic MCP round-trip test")
    parser.add_argument(
        "--call-generate",
        action="store_true",
        help="Also perform a real generate_sprite call (costs ~$0.01)",
    )
    parser.add_argument(
        "--prompt",
        default="cute pixel art slime with a smile",
        help="Prompt to use for the real round-trip (only with --call-generate)",
    )
    args = parser.parse_args()

    print("sprite-me MCP verification")
    print(f"  repo: {_REPO_ROOT}")
    print("  spawning server: uv run --directory ... sprite-me-server")

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(_REPO_ROOT), "sprite-me-server"],
    )

    all_failures: list[str] = []
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("  ✓ MCP session initialized")

            all_failures.extend(await verify_tools(session))
            all_failures.extend(await verify_resources(session))

            if args.call_generate:
                all_failures.extend(await verify_call_generate(session, args.prompt))

    print()
    if all_failures:
        print(f"\033[31mFAIL\033[0m: {len(all_failures)} issue(s)")
        for f in all_failures:
            print(f"  - {f}")
        return 1
    else:
        print("\033[32mOK\033[0m: sprite-me MCP server is ready for Claude Code")
        print()
        print("Next: to use this from Claude Code, ensure ~/.claude/claude_desktop_config.json has:")
        print('  {')
        print('    "mcpServers": {')
        print('      "sprite-me": {')
        print('        "command": "uv",')
        print(f'        "args": ["run", "--directory", "{_REPO_ROOT}", "sprite-me-server"],')
        print('        "env": {')
        print('          "SPRITE_ME_RUNPOD_API_KEY": "...",')
        print('          "SPRITE_ME_RUNPOD_ENDPOINT_ID": "..."')
        print('        }')
        print('      }')
        print('    }')
        print('  }')
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
