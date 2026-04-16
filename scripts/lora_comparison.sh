#!/usr/bin/env bash
# One-off driver for the Phase 1 LoRA comparison matrix.
# Groups calls by LoRA (not by prompt) to minimize cold starts.
#
# Usage: ./scripts/lora_comparison.sh
# Output: docs/lora-comparison/<lora>_<slug>.png
set -euo pipefail

cd "$(dirname "$0")/.."
set -a; source .env; set +a

PROMPTS=(
  "knight:brave knight with longsword and red cape"
  "wizard:young wizard with staff and blue robes"
  "warrior-f:female warrior with axe and leather armor"
  "goblin:goblin with dagger and loincloth"
  "slime:chibi pink slime with smile"
  "orc:orc warrior with greataxe"
  "archer:elven archer with longbow drawn"
  "spellcaster:robed spellcaster with hood up"
)

LORAS=(cartoon-vector pixel-indie pixel-retro)

for lora in "${LORAS[@]}"; do
  echo "=== LoRA: $lora ==="
  for pair in "${PROMPTS[@]}"; do
    slug="${pair%%:*}"
    prompt="${pair#*:}"
    out="docs/lora-comparison/${lora}_${slug}.png"
    if [[ -f "$out" ]]; then
      echo "  skip $slug (exists)"
      continue
    fi
    echo "  [$lora] $slug: $prompt"
    uv run scripts/test_endpoint.py \
      --lora "$lora" \
      --prompt "$prompt" \
      --seed 42 \
      --out "$out" 2>&1 | tail -3
  done
done

# Top-down perspective sanity check (not in the character matrix)
td_out="docs/lora-comparison/top-down_mage-walking.png"
if [[ ! -f "$td_out" ]]; then
  echo "=== LoRA: top-down (single test) ==="
  uv run scripts/test_endpoint.py \
    --lora top-down \
    --prompt "mage with staff walking through forest" \
    --seed 42 \
    --out "$td_out" 2>&1 | tail -3
fi

echo "Done. Outputs in docs/lora-comparison/"
ls -1 docs/lora-comparison/
