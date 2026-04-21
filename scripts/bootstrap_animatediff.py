#!/usr/bin/env python3
"""Bootstrap AnimateDiff-path assets onto every sprite-me network volume.

Counterpart to bootstrap_loras.py for the SD1.5 + AnimateDiff v3 +
IP-Adapter pipeline. Reads `sprite_me.animatediff_assets.ANIMATEDIFF_ASSETS`
as the source of truth; for each asset:

    1. If the file is missing under ./models/<subdir>/<name>, download it
       (all upstreams are public HuggingFace URLs — no auth required).
    2. For each sprite-me volume, upload the file if it's missing there
       under models/<remote_prefix>/<name>.

Idempotent — existing files are skipped via size comparison, same as
sync_models.py / bootstrap_loras.py.

Usage:
    uv run scripts/bootstrap_animatediff.py           # download + upload
    uv run scripts/bootstrap_animatediff.py --dry-run # plan only
    uv run scripts/bootstrap_animatediff.py --skip-upload  # local cache only

Requires (via .env):
    RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_KEY — for volume uploads
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sprite_me.animatediff_assets import ANIMATEDIFF_ASSETS, AnimateDiffAsset  # noqa: E402
from sprite_me.volumes import VOLUME_PREFIX, list_sprite_me_volumes_by_dc  # noqa: E402

sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from sync_models import (  # noqa: E402
    S3_ENDPOINTS,
    _remote_file_exists,
    _s3_client,
    upload_file,
)

_MODELS_DIR = _REPO_ROOT / "models"


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"  \033[33m!\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}", file=sys.stderr)


def _download_url(url: str, dest: Path) -> bool:
    # 30-min timeout — these files are multi-GB on residential connections.
    with httpx.stream("GET", url, follow_redirects=True, timeout=1800) as r:
        if r.status_code != 200:
            _fail(f"Download failed for {url}: HTTP {r.status_code}")
            return False
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -1
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=4 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct and pct % 10 == 0:
                        print(f"    {pct}% ({downloaded // (1024*1024)} MB)")
                        last_pct = pct
    return True


def ensure_local(asset: AnimateDiffAsset, dry_run: bool) -> bool:
    dest_dir = _MODELS_DIR / asset.local_subdir
    dest = dest_dir / asset.name

    if dest.exists() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size // (1024 * 1024)
        _ok(f"{asset.name} cached locally ({size_mb} MB)")
        return True

    if dry_run:
        _warn(f"{asset.name} missing locally — would download ~{asset.expected_size_mb} MB")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {asset.name} (~{asset.expected_size_mb} MB) from {asset.source_url[:70]}...")
    if not _download_url(asset.source_url, dest):
        dest.unlink(missing_ok=True)
        return False
    size_mb = dest.stat().st_size // (1024 * 1024)
    _ok(f"{asset.name} downloaded ({size_mb} MB)")
    return True


def ensure_remote(
    asset: AnimateDiffAsset, volumes: dict[str, str], dry_run: bool
) -> None:
    local = _MODELS_DIR / asset.local_subdir / asset.name
    if not local.exists():
        return
    local_size = local.stat().st_size
    remote_key = f"models/{asset.remote_prefix}/{asset.name}"

    for dc, vol_id in sorted(volumes.items()):
        if dc not in S3_ENDPOINTS:
            _warn(f"{asset.name} skipped {dc} (no S3 endpoint)")
            continue
        client = _s3_client(dc)
        if _remote_file_exists(client, vol_id, remote_key, local_size):
            _ok(f"{asset.name} present on {dc}")
            continue

        if dry_run:
            _warn(f"{asset.name} would upload to {dc}")
            continue

        print(f"  uploading {asset.name} to {dc}...")
        try:
            upload_file(str(local), f"models/{asset.remote_prefix}", vol_id, dc, progress_bar=None)
            _ok(f"{asset.name} uploaded to {dc}")
        except Exception as e:
            _fail(f"{asset.name} upload to {dc} failed: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap AnimateDiff path assets")
    parser.add_argument("--dry-run", action="store_true", help="Plan, don't touch anything")
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only ensure local cache; skip volume uploads",
    )
    args = parser.parse_args()

    print(f"sprite-me AnimateDiff asset bootstrap ({len(ANIMATEDIFF_ASSETS)} assets)")
    if args.dry_run:
        print("  [DRY RUN — no changes]")

    print("\n[1/2] Ensuring local cache under ./models/...")
    for key, asset in ANIMATEDIFF_ASSETS.items():
        print(f"{key}:")
        ensure_local(asset, dry_run=args.dry_run)

    if args.skip_upload:
        print("\n[2/2] Skipping volume uploads (--skip-upload).")
        return 0

    volumes = list_sprite_me_volumes_by_dc()
    if not volumes:
        _fail(f"No {VOLUME_PREFIX} volumes found. Run sprite_me.volumes setup first.")
        return 1

    print(f"\n[2/2] Ensuring remote copies on {len(volumes)} volume(s)...")
    for key, asset in ANIMATEDIFF_ASSETS.items():
        print(f"{key}:")
        ensure_remote(asset, volumes, dry_run=args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
