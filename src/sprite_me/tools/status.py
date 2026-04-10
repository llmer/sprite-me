"""Check job status tool."""

from __future__ import annotations

from typing import Any

from sprite_me.inference.runpod_client import RunPodClient


async def check_job_status(
    job_id: str,
    runpod: RunPodClient | None = None,
) -> dict[str, Any]:
    """Check the status of a RunPod generation job.

    Returns status info including state, progress, and result if completed.
    """
    runpod = runpod or RunPodClient()
    status = await runpod.get_status(job_id)
    return {
        "job_id": job_id,
        "status": status.get("status", "UNKNOWN"),
        "delay_time": status.get("delayTime"),
        "execution_time": status.get("executionTime"),
        "completed": status.get("status") == "COMPLETED",
    }
