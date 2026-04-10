"""Network volume management for RunPod multi-region endpoints.

Network volumes let us store FLUX checkpoints and LoRAs once per datacenter,
then attach them to the serverless endpoint. This keeps the Docker image
small (LoRAs are loaded from /runpod-volume/loras/) and makes LoRA updates
fast — just sync to the volume, no image rebuild.

Pattern mirrored from /Users/jim/src/vid/volumes.py.
"""

from __future__ import annotations

import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("SPRITE_ME_RUNPOD_API_KEY", os.environ.get("RUNPOD_API_KEY", ""))
ENDPOINT_ID = os.environ.get("SPRITE_ME_RUNPOD_ENDPOINT_ID", "")
REST_BASE = "https://rest.runpod.io/v1"

# Datacenters to provision volumes in. Pick DCs where your serverless workers
# will run — RunPod routes jobs to whichever DC has the attached volume.
DEFAULT_DATACENTERS = ["US-GA-2", "US-KS-2", "US-NC-2"]
VOLUME_SIZE_GB = 20
VOLUME_PREFIX = "sprite-me"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def list_volumes() -> list[dict]:
    """List all network volumes on the account."""
    resp = httpx.get(f"{REST_BASE}/networkvolumes", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def create_volume(name: str, size_gb: int, datacenter_id: str) -> dict:
    """Create a network volume in the specified datacenter."""
    resp = httpx.post(
        f"{REST_BASE}/networkvolumes",
        headers=_headers(),
        json={"name": name, "size": size_gb, "dataCenterId": datacenter_id},
        timeout=30,
    )
    if not resp.is_success:
        error = resp.json().get("error", resp.text) if resp.content else resp.text
        print(f"  ERROR creating volume in {datacenter_id}: {error}", file=sys.stderr)
        resp.raise_for_status()
    return resp.json()


def delete_volume(volume_id: str) -> None:
    """Delete a network volume."""
    resp = httpx.delete(
        f"{REST_BASE}/networkvolumes/{volume_id}", headers=_headers(), timeout=30
    )
    if not resp.is_success:
        error = resp.json().get("error", resp.text) if resp.content else resp.text
        print(f"  ERROR deleting {volume_id}: {error}", file=sys.stderr)
        resp.raise_for_status()


def get_endpoint(endpoint_id: str) -> dict:
    """Get endpoint details."""
    resp = httpx.get(f"{REST_BASE}/endpoints/{endpoint_id}", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def attach_volumes_to_endpoint(endpoint_id: str, volume_ids: list[str]) -> dict:
    """Attach network volumes to a serverless endpoint."""
    resp = httpx.patch(
        f"{REST_BASE}/endpoints/{endpoint_id}",
        headers=_headers(),
        json={"networkVolumeIds": volume_ids},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def list_sprite_me_volumes_by_dc() -> dict[str, str]:
    """Get sprite-me volume IDs grouped by datacenter."""
    volumes = list_volumes()
    return {
        v["dataCenterId"]: v["id"]
        for v in volumes
        if v["name"].startswith(VOLUME_PREFIX)
    }


def setup(
    datacenters: list[str] | None = None,
    size_gb: int = VOLUME_SIZE_GB,
    endpoint_id: str | None = None,
) -> list[str]:
    """Create volumes in each datacenter and attach to the endpoint.

    Idempotent — existing volumes are reused.
    """
    dcs = datacenters or DEFAULT_DATACENTERS
    ep = endpoint_id or ENDPOINT_ID

    existing_by_dc = list_sprite_me_volumes_by_dc()

    volume_ids: list[str] = []
    for dc in dcs:
        if dc in existing_by_dc:
            vol_id = existing_by_dc[dc]
            print(f"  {dc}: using existing volume {vol_id}")
            volume_ids.append(vol_id)
        else:
            name = f"{VOLUME_PREFIX}-{dc.lower()}"
            print(f"  {dc}: creating {name} ({size_gb}GB)...")
            vol = create_volume(name, size_gb, dc)
            print(f"  {dc}: created {vol['id']}")
            volume_ids.append(vol["id"])

    if ep:
        print(f"\nAttaching {len(volume_ids)} volume(s) to endpoint {ep}...")
        attach_volumes_to_endpoint(ep, volume_ids)
        print("Done.")
    else:
        print("\nSkipping attach (no endpoint ID configured).")

    return volume_ids


def teardown(endpoint_id: str | None = None) -> None:
    """Delete all sprite-me volumes and detach from the endpoint."""
    ep = endpoint_id or ENDPOINT_ID
    to_delete = [v for v in list_volumes() if v["name"].startswith(VOLUME_PREFIX)]

    if not to_delete:
        print("No sprite-me volumes found.")
        return

    if ep:
        print(f"Detaching volumes from endpoint {ep}...")
        attach_volumes_to_endpoint(ep, [])

    for v in to_delete:
        print(f"  Deleting {v['id']} ({v['dataCenterId']}, {v['name']})...")
        delete_volume(v["id"])
    print("Done.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Manage RunPod network volumes for sprite-me")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all network volumes")

    setup_p = sub.add_parser("setup", help="Create volumes in all DCs and attach to endpoint")
    setup_p.add_argument(
        "--size", type=int, default=VOLUME_SIZE_GB, help=f"Volume size in GB (default: {VOLUME_SIZE_GB})"
    )
    setup_p.add_argument(
        "--dcs", nargs="+", default=DEFAULT_DATACENTERS, help="Datacenters to provision in"
    )

    sub.add_parser("endpoint", help="Show current endpoint config")
    sub.add_parser("teardown", help="Delete all sprite-me volumes and detach from endpoint")

    args = parser.parse_args()
    if args.command == "list":
        for v in list_volumes():
            print(
                f"  {v['id']}  {v.get('dataCenterId', '?'):10s}  "
                f"{v.get('size', '?'):>4}GB  {v['name']}"
            )
    elif args.command == "setup":
        setup(datacenters=args.dcs, size_gb=args.size)
    elif args.command == "endpoint":
        if not ENDPOINT_ID:
            print("SPRITE_ME_RUNPOD_ENDPOINT_ID not set", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(get_endpoint(ENDPOINT_ID), indent=2))
    elif args.command == "teardown":
        teardown()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
