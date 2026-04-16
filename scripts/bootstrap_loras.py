#!/usr/bin/env python3
"""Bootstrap LoRAs into the local cache and onto every sprite-me network volume.

Reads the `LORAS` registry from `sprite_me.loras` as the source of truth.
For each profile:

    1. If the file is missing locally (./models/loras/<name>), download it:
       - civitai_version_id set → Civitai API (needs CIVITAI_API_TOKEN)
       - source_url set → direct HTTP GET
       - neither → log as "manual-upload-required" and continue
    2. For each sprite-me volume, upload the file if it's missing there.

Idempotent — re-running skips files that already exist locally and remotely
(the remote check uses the same size comparison as sync_models.py).

Also detects **stray files** on the volume: anything under `models/loras/`
whose filename doesn't match a registered profile. By default, strays are
reported but left alone. Pass `--prune` to actually delete them.

Usage:
    uv run scripts/bootstrap_loras.py           # download + upload, report strays
    uv run scripts/bootstrap_loras.py --prune   # also delete reported strays
    uv run scripts/bootstrap_loras.py --dry-run # show plan, don't touch anything

Requires env vars (via .env):
    CIVITAI_API_TOKEN       — for Civitai downloads
    RUNPOD_S3_ACCESS_KEY    — for volume uploads
    RUNPOD_S3_SECRET_KEY    — ditto
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sprite_me.loras import LORAS, LoraProfile  # noqa: E402
from sprite_me.volumes import VOLUME_PREFIX, list_sprite_me_volumes_by_dc  # noqa: E402

# Reuse the S3 plumbing already in sync_models.py instead of duplicating it
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from sync_models import (  # noqa: E402
    S3_ENDPOINTS,
    _s3_client,
    _remote_file_exists,
    list_files,
    upload_file,
)

_LORAS_DIR = _REPO_ROOT / "models" / "loras"
_VOLUME_PREFIX = "models/loras"


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"  \033[33m!\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}", file=sys.stderr)


def _download_civitai(version_id: int, dest: Path, token: str) -> bool:
    url = f"https://civitai.com/api/download/models/{version_id}"
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=600) as r:
        if r.status_code != 200:
            _fail(f"Civitai download failed for version {version_id}: HTTP {r.status_code}")
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    return True


def _download_url(url: str, dest: Path) -> bool:
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        if r.status_code != 200:
            _fail(f"Direct download failed for {url}: HTTP {r.status_code}")
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    return True


def ensure_local(profile: LoraProfile, dry_run: bool) -> bool:
    """Make sure the LoRA exists under ./models/loras/. Return True if present
    (either already or after download). False on failure.
    """
    dest = _LORAS_DIR / profile.name
    if dest.exists() and dest.stat().st_size > 0:
        _ok(f"{profile.name} cached locally ({dest.stat().st_size // (1024*1024)} MB)")
        return True

    if dry_run:
        _warn(f"{profile.name} missing locally — would download")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    if profile.civitai_version_id is not None:
        token = os.environ.get("CIVITAI_API_TOKEN", "")
        if not token:
            _fail(f"{profile.name} needs CIVITAI_API_TOKEN in .env to re-download")
            return False
        print(f"  downloading {profile.name} from Civitai (version {profile.civitai_version_id})...")
        if not _download_civitai(profile.civitai_version_id, dest, token):
            dest.unlink(missing_ok=True)
            return False
        _ok(f"{profile.name} downloaded ({dest.stat().st_size // (1024*1024)} MB)")
        return True

    if profile.source_url:
        print(f"  downloading {profile.name} from {profile.source_url[:60]}...")
        if not _download_url(profile.source_url, dest):
            dest.unlink(missing_ok=True)
            return False
        _ok(f"{profile.name} downloaded ({dest.stat().st_size // (1024*1024)} MB)")
        return True

    _warn(
        f"{profile.name} has no civitai_version_id or source_url — cannot "
        f"re-download. Manual upload required. See docs/lora-bootstrap.md."
    )
    return False


def ensure_remote(
    profile: LoraProfile, volumes: dict[str, str], dry_run: bool
) -> None:
    """Upload the LoRA to every volume that's missing it."""
    local = _LORAS_DIR / profile.name
    if not local.exists():
        return
    local_size = local.stat().st_size
    remote_key = f"{_VOLUME_PREFIX}/{profile.name}"

    for dc, vol_id in sorted(volumes.items()):
        if dc not in S3_ENDPOINTS:
            _warn(f"{profile.name} skipped {dc} (no S3 endpoint)")
            continue
        client = _s3_client(dc)
        if _remote_file_exists(client, vol_id, remote_key, local_size):
            _ok(f"{profile.name} present on {dc}")
            continue

        if dry_run:
            _warn(f"{profile.name} would upload to {dc}")
            continue

        print(f"  uploading {profile.name} to {dc}...")
        try:
            upload_file(str(local), _VOLUME_PREFIX, vol_id, dc, progress_bar=None)
            _ok(f"{profile.name} uploaded to {dc}")
        except Exception as e:
            _fail(f"{profile.name} upload to {dc} failed: {e}")


def detect_strays(volumes: dict[str, str]) -> list[tuple[str, str, str]]:
    """List files on the volume that aren't in the registry.

    Returns [(dc, vol_id, key), ...]
    """
    known = {p.name for p in LORAS.values()}
    strays: list[tuple[str, str, str]] = []
    for dc, vol_id in sorted(volumes.items()):
        if dc not in S3_ENDPOINTS:
            continue
        client = _s3_client(dc)
        resp = client.list_objects_v2(Bucket=vol_id, Prefix=f"{_VOLUME_PREFIX}/")
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            filename = key.rsplit("/", 1)[-1]
            if not filename:
                continue
            if filename not in known:
                size_mb = obj["Size"] // (1024 * 1024)
                print(f"  stray: {key} ({size_mb} MB) on {dc}")
                strays.append((dc, vol_id, key))
    return strays


def prune_strays(strays: list[tuple[str, str, str]]) -> None:
    for dc, vol_id, key in strays:
        client = _s3_client(dc)
        try:
            client.delete_object(Bucket=vol_id, Key=key)
            _ok(f"deleted stray {key} from {dc}")
        except Exception as e:
            _fail(f"failed to delete stray {key} from {dc}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap LoRAs from registry")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded/uploaded/pruned, don't touch anything",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete files on the volume that aren't in the LORAS registry",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only ensure local cache; skip volume uploads",
    )
    args = parser.parse_args()

    print(f"sprite-me LoRA bootstrap (registry: {len(LORAS)} profiles)")
    if args.dry_run:
        print("  [DRY RUN — no changes]")

    print("\n[1/3] Ensuring local cache under ./models/loras/...")
    for key, profile in LORAS.items():
        print(f"{key}:")
        ensure_local(profile, dry_run=args.dry_run)

    volumes: dict[str, str] = {}
    if not args.skip_upload:
        volumes = list_sprite_me_volumes_by_dc()
        if not volumes:
            _fail(f"No {VOLUME_PREFIX} volumes found. Run sprite_me.volumes setup first.")
            return 1

    if not args.skip_upload:
        print(f"\n[2/3] Ensuring remote copies on {len(volumes)} volume(s)...")
        for key, profile in LORAS.items():
            print(f"{key}:")
            ensure_remote(profile, volumes, dry_run=args.dry_run)

        print("\n[3/3] Scanning for stray files on volumes...")
        strays = detect_strays(volumes)
        if not strays:
            _ok("no strays detected")
        elif args.prune and not args.dry_run:
            print(f"\nPruning {len(strays)} stray file(s) (--prune set)...")
            prune_strays(strays)
        elif strays:
            _warn(
                f"{len(strays)} stray file(s) present. "
                f"Re-run with --prune to delete them."
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
