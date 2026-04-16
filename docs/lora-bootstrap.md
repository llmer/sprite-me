# LoRA bootstrap: re-creating volume state from scratch

If a sprite-me network volume gets destroyed, or you set up a new datacenter,
`scripts/bootstrap_loras.py` can re-populate the LoRA directory from the
registry in `src/sprite_me/loras.py` without any manual file juggling.

## What it does

1. Reads every profile in `LORAS` (src/sprite_me/loras.py).
2. For each profile:
   - If the file is missing from `./models/loras/`, downloads it using
     `civitai_version_id` (Civitai API) or `source_url` (direct HTTP GET).
   - For each sprite-me network volume, uploads the file to
     `models/loras/<name>` if it's missing or wrong-size.
3. Scans for **stray files** on the volume: files under `models/loras/`
   whose filename doesn't match any registered profile. By default these
   are reported only. Pass `--prune` to actually delete them.

Safe to re-run — the script skips anything that's already cached locally
and present at the correct size on the volume.

## Usage

```bash
# One-time first setup or after a volume teardown:
uv run scripts/bootstrap_loras.py

# See what it would do without touching anything:
uv run scripts/bootstrap_loras.py --dry-run

# Also delete stray files:
uv run scripts/bootstrap_loras.py --prune

# Only warm the local cache, don't touch volumes:
uv run scripts/bootstrap_loras.py --skip-upload
```

## Required .env keys

```
CIVITAI_API_TOKEN=<from https://civitai.com/user/account>
RUNPOD_S3_ACCESS_KEY=<from RunPod Settings > S3 API Keys>
RUNPOD_S3_SECRET_KEY=<ditto>
```

The Civitai token is only needed for profiles whose `civitai_version_id`
is set. HuggingFace downloads via `source_url` work without auth.

## Manual uploads

Some LoRAs may have `civitai_version_id=None` AND `source_url=None` — e.g.
a private LoRA you trained yourself, or one downloaded from a now-dead
source. The script will log:

```
! <name> has no civitai_version_id or source_url — cannot re-download.
  Manual upload required. See docs/lora-bootstrap.md.
```

For these, drop the `.safetensors` into `./models/loras/` yourself and
re-run the script — it'll pick up the local file and upload it to the
volume.

## Adding a new LoRA

1. Verify it's FLUX.1-dev compatible and download it locally into
   `./models/loras/<name>.safetensors`.
2. Add a `LoraProfile` entry to `src/sprite_me/loras.py`:
   - `name`: the exact filename
   - `trigger`: trigger word or phrase (empty string if BLIP-captioned)
   - `prompt_template`: f-string with `{trigger}` and `{prompt}`
   - `civitai_version_id`: the model version id from the Civitai URL
     (e.g. `https://civitai.com/api/download/models/1234567` → `1234567`),
     or `None` for non-Civitai sources
   - `source_url`: direct download URL for non-Civitai LoRAs, or `None`
   - `description` + `best_for`: for the MCP `list_loras` tool
3. Run `uv run scripts/bootstrap_loras.py` to upload to all volumes.
4. Add a test in `tests/test_workflow_builder.py` that verifies the
   profile key + prompt shape.
