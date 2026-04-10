#!/usr/bin/env python3
"""Upload and sync model files to RunPod network volumes via S3-compatible API.

Pattern mirrored from /Users/jim/src/vid/sync_loras.py.

Uploads LoRAs (and optionally checkpoints) to a standard layout on each
volume:

    /loras/*.safetensors
    /checkpoints/*.safetensors    (optional)

The container's entrypoint.sh symlinks these into ComfyUI's models dir.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from dotenv import load_dotenv
from tqdm import tqdm

# Make the sprite_me package importable when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sprite_me.volumes import VOLUME_PREFIX, list_sprite_me_volumes_by_dc  # noqa: E402

load_dotenv()

S3_ACCESS_KEY = os.environ.get("RUNPOD_S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("RUNPOD_S3_SECRET_KEY", "")

# S3 endpoint URLs per datacenter (from RunPod docs)
S3_ENDPOINTS = {
    "EUR-IS-1": "https://s3api-eur-is-1.runpod.io/",
    "EUR-NO-1": "https://s3api-eur-no-1.runpod.io/",
    "EU-RO-1": "https://s3api-eu-ro-1.runpod.io/",
    "EU-CZ-1": "https://s3api-eu-cz-1.runpod.io/",
    "US-CA-2": "https://s3api-us-ca-2.runpod.io/",
    "US-GA-2": "https://s3api-us-ga-2.runpod.io/",
    "US-KS-2": "https://s3api-us-ks-2.runpod.io/",
    "US-MD-1": "https://s3api-us-md-1.runpod.io/",
    "US-MO-2": "https://s3api-us-mo-2.runpod.io/",
    "US-NC-1": "https://s3api-us-nc-1.runpod.io/",
    "US-NC-2": "https://s3api-us-nc-2.runpod.io/",
}


def _s3_client(datacenter_id: str):
    endpoint_url = S3_ENDPOINTS.get(datacenter_id)
    if not endpoint_url:
        raise ValueError(
            f"No S3 API endpoint for {datacenter_id}. "
            f"Available: {', '.join(S3_ENDPOINTS.keys())}"
        )
    return boto3.client(
        "s3",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=datacenter_id,
        endpoint_url=endpoint_url,
    )


def _remote_file_exists(client, volume_id: str, object_key: str, local_size: int) -> bool:
    """Check if a file already exists on the volume with the same size."""
    try:
        resp = client.head_object(Bucket=volume_id, Key=object_key)
        return resp["ContentLength"] == local_size
    except client.exceptions.ClientError:
        return False


def upload_file(
    file_path: str,
    remote_prefix: str,
    volume_id: str,
    datacenter_id: str,
    progress_bar: tqdm | None = None,
) -> bool:
    """Upload a single file to {remote_prefix}/{filename} on a network volume."""
    client = _s3_client(datacenter_id)
    filename = Path(file_path).name
    object_key = f"{remote_prefix.rstrip('/')}/{filename}"
    local_size = Path(file_path).stat().st_size

    if _remote_file_exists(client, volume_id, object_key, local_size):
        if progress_bar:
            progress_bar.update(local_size)
        return False  # skipped

    callback = progress_bar.update if progress_bar else None
    client.upload_file(file_path, volume_id, object_key, Callback=callback)
    return True


def _upload_task(
    file_path: Path,
    remote_prefix: str,
    volume_id: str,
    datacenter_id: str,
    position: int,
) -> str:
    file_size = file_path.stat().st_size
    label = f"{file_path.name} -> {datacenter_id}"
    with tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc=label,
        position=position,
        leave=True,
    ) as pbar:
        uploaded = upload_file(str(file_path), remote_prefix, volume_id, datacenter_id, pbar)
        if not uploaded:
            pbar.set_description(f"{label} (skipped)")
    return label


def list_files(volume_id: str, datacenter_id: str, prefix: str = "loras/") -> list[str]:
    """List files on a network volume under the given prefix."""
    client = _s3_client(datacenter_id)
    resp = client.list_objects_v2(Bucket=volume_id, Prefix=prefix)
    files = []
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if key != prefix:
            files.append(key)
            size_mb = obj["Size"] / 1024 / 1024
            print(f"  {key} ({size_mb:.1f} MB)")
    if not files:
        print(f"  (no files in {prefix})")
    return files


def _resolve_targets(
    volumes: dict[str, str], files: list[Path], remote_prefix: str
) -> list[tuple[Path, str, str, str]]:
    """Build (file, remote_prefix, volume_id, dc) tasks."""
    targets = []
    skipped = []
    for dc, vol_id in sorted(volumes.items()):
        if dc not in S3_ENDPOINTS:
            skipped.append(dc)
            continue
        for f in files:
            targets.append((f, remote_prefix, vol_id, dc))
    for dc in skipped:
        print(f"[{dc}] SKIPPED (no S3 API for this datacenter)")
    return targets


def _run_uploads(targets: list[tuple[Path, str, str, str]], parallel: int = 1) -> None:
    workers = min(len(targets), parallel)
    errors = []

    if workers <= 1:
        for f, prefix, vol_id, dc in targets:
            try:
                _upload_task(f, prefix, vol_id, dc, position=0)
            except Exception as e:
                errors.append(f"  {f.name} -> {dc}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, (f, prefix, vol_id, dc) in enumerate(targets):
                fut = pool.submit(_upload_task, f, prefix, vol_id, dc, i)
                futures[fut] = (f.name, dc)
            for fut in as_completed(futures):
                name, dc = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append(f"  {name} -> {dc}: {e}")

    print()
    if errors:
        print("Errors:", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)


def cmd_sync(args) -> None:
    """Upload all .safetensors from a directory to all volumes."""
    lora_path = Path(args.dir)
    files = sorted(lora_path.glob("*.safetensors"))
    if not files:
        print(f"No .safetensors files found in {args.dir}")
        sys.exit(1)

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    volumes = list_sprite_me_volumes_by_dc()
    if not volumes:
        print(
            f"No {VOLUME_PREFIX} volumes found. Run: python -m sprite_me.volumes setup",
            file=sys.stderr,
        )
        sys.exit(1)

    targets = _resolve_targets(volumes, files, args.prefix)
    if not targets:
        print("No uploadable targets.", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nUploading {len(files)} file(s) to {len(volumes)} volume(s) "
        f"at {args.prefix}/ ({len(targets)} transfers)...\n"
    )
    _run_uploads(targets, parallel=args.parallel)
    print("Done.")


def cmd_upload(args) -> None:
    """Upload a single file to all volumes."""
    volumes = list_sprite_me_volumes_by_dc()
    if not volumes:
        print(f"No {VOLUME_PREFIX} volumes found.", file=sys.stderr)
        sys.exit(1)

    files = [Path(args.file)]
    targets = _resolve_targets(volumes, files, args.prefix)
    if not targets:
        sys.exit(1)

    print(f"\nUploading to {len(targets)} volume(s) at {args.prefix}/...\n")
    _run_uploads(targets, parallel=args.parallel)
    print("Done.")


def cmd_list(args) -> None:
    volumes = list_sprite_me_volumes_by_dc()
    if not volumes:
        print(f"No {VOLUME_PREFIX} volumes found.", file=sys.stderr)
        sys.exit(1)

    for dc, vol_id in sorted(volumes.items()):
        if dc not in S3_ENDPOINTS:
            print(f"\n[{dc}] volume {vol_id} — no S3 API")
            continue
        print(f"\n[{dc}] volume {vol_id}")
        list_files(vol_id, dc, prefix=args.prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync models to RunPod network volumes")
    sub = parser.add_subparsers(dest="command")

    sync_p = sub.add_parser("sync", help="Upload all .safetensors from a directory")
    sync_p.add_argument("dir", nargs="?", default="./models/loras", help="Local directory")
    sync_p.add_argument("--prefix", default="loras", help="Remote prefix on volume (default: loras)")
    sync_p.add_argument("--parallel", "-j", type=int, default=1)

    upload_p = sub.add_parser("upload", help="Upload a single file")
    upload_p.add_argument("file")
    upload_p.add_argument("--prefix", default="loras")
    upload_p.add_argument("--parallel", "-j", type=int, default=1)

    list_p = sub.add_parser("list", help="List files on all volumes")
    list_p.add_argument("--prefix", default="loras/")

    args = parser.parse_args()
    if args.command == "sync":
        cmd_sync(args)
    elif args.command == "upload":
        cmd_upload(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
