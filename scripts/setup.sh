#!/usr/bin/env bash
# sprite-me MCP installer
# Detects your editor and installs the sprite-me MCP server configuration.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

info()  { printf "\033[36m[sprite-me]\033[0m %s\n" "$*"; }
warn()  { printf "\033[33m[sprite-me]\033[0m %s\n" "$*"; }
error() { printf "\033[31m[sprite-me]\033[0m %s\n" "$*" >&2; }

info "Installing sprite-me from $REPO_DIR"

# Check prerequisites
if ! command -v python3 >/dev/null 2>&1; then
  error "python3 not found. Please install Python 3.11+."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  warn "uv not found. Installing via pip..."
  python3 -m pip install --user uv
fi

# Install dependencies
info "Installing Python dependencies..."
cd "$REPO_DIR"
uv sync || python3 -m pip install -e .

# Prompt for RunPod credentials
if [ -z "${SPRITE_ME_RUNPOD_API_KEY:-}" ]; then
  read -r -p "RunPod API key: " RUNPOD_KEY
else
  RUNPOD_KEY="$SPRITE_ME_RUNPOD_API_KEY"
fi

if [ -z "${SPRITE_ME_RUNPOD_ENDPOINT_ID:-}" ]; then
  read -r -p "RunPod endpoint ID: " RUNPOD_EP
else
  RUNPOD_EP="$SPRITE_ME_RUNPOD_ENDPOINT_ID"
fi

# Detect editor and write MCP config
detect_and_install() {
  local cfg_path=""
  local editor=""

  if [ -d "$HOME/.claude" ]; then
    cfg_path="$HOME/.claude/claude_desktop_config.json"
    editor="Claude Code"
  elif [ -d "$HOME/Library/Application Support/Claude" ]; then
    cfg_path="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    editor="Claude Desktop"
  elif [ -d "$HOME/.cursor" ]; then
    cfg_path="$HOME/.cursor/mcp.json"
    editor="Cursor"
  else
    warn "No supported editor detected. Printing config for manual install."
  fi

  local config='{
  "mcpServers": {
    "sprite-me": {
      "command": "uv",
      "args": ["--directory", "'"$REPO_DIR"'", "run", "sprite-me-server"],
      "env": {
        "SPRITE_ME_RUNPOD_API_KEY": "'"$RUNPOD_KEY"'",
        "SPRITE_ME_RUNPOD_ENDPOINT_ID": "'"$RUNPOD_EP"'"
      }
    }
  }
}'

  if [ -n "$cfg_path" ]; then
    info "Writing MCP config to $cfg_path ($editor)"
    mkdir -p "$(dirname "$cfg_path")"
    # If file exists, warn before overwriting
    if [ -f "$cfg_path" ]; then
      warn "Config file already exists. Backing up to ${cfg_path}.bak"
      cp "$cfg_path" "${cfg_path}.bak"
    fi
    printf "%s\n" "$config" > "$cfg_path"
    info "Done. Restart $editor to load the server."
  else
    printf "\n%s\n\n" "$config"
    info "Copy the above into your editor's MCP config."
  fi
}

detect_and_install

# Install skills
info "Installing agent skills..."
if [ -d "$HOME/.claude/skills" ]; then
  mkdir -p "$HOME/.claude/skills/sprite-me"
  cp "$REPO_DIR"/skills/*.md "$HOME/.claude/skills/sprite-me/"
  info "Skills installed to ~/.claude/skills/sprite-me/"
fi

info "Setup complete."
