"""RunPod Serverless API client for submitting ComfyUI jobs.

Pattern reference: /Users/jim/src/vid/avatar_llmer/runpod.py (LiveRunpodPipeline)
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx

from sprite_me.config import settings

logger = logging.getLogger(__name__)

TERMINAL_STATES = {"COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"}


class RunPodError(Exception):
    """Raised when a RunPod API call fails or returns an unusable response."""


class RunPodClient:
    """Async client for RunPod's serverless ComfyUI endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint_id: str | None = None,
        poll_interval: float = 2.0,
        timeout_seconds: float = 600.0,
    ):
        self.api_key = api_key or settings.runpod_api_key
        self.endpoint_id = endpoint_id or settings.runpod_endpoint_id
        self.base_url = f"{settings.runpod_base_url}/{self.endpoint_id}"
        self.poll_interval = poll_interval
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self._headers(),
                timeout=httpx.Timeout(30.0, read=120.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def submit_workflow(self, workflow: dict[str, Any]) -> str:
        """Submit a ComfyUI workflow and return the job ID."""
        resp = await self.client.post(
            f"{self.base_url}/run",
            json={"input": {"workflow": workflow}},
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("id")
        if not job_id:
            raise RunPodError(f"No job ID in response: {data}")
        logger.info("Submitted job %s", job_id)
        return job_id

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """Fetch the current status of a job."""
        resp = await self.client.get(f"{self.base_url}/status/{job_id}")
        resp.raise_for_status()
        return resp.json()

    async def wait_for_result(self, job_id: str) -> dict[str, Any]:
        """Poll until the job reaches a terminal state or the deadline expires.

        Uses a monotonic deadline (not an elapsed counter) so time spent
        processing each poll response doesn't inflate the budget.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_seconds

        while True:
            payload = await self.get_status(job_id)
            status = payload.get("status")

            if status in TERMINAL_STATES:
                if status != "COMPLETED":
                    raise RunPodError(
                        f"Job {job_id} ended in state {status}: "
                        f"{payload.get('error', 'no error message')}"
                    )
                return payload

            if loop.time() > deadline:
                raise RunPodError(
                    f"Job {job_id} timed out after {self.timeout_seconds}s "
                    f"(last status: {status})"
                )

            await asyncio.sleep(self.poll_interval)

    async def generate(self, workflow: dict[str, Any]) -> list[bytes]:
        """Submit a workflow, wait for completion, return decoded images."""
        job_id = await self.submit_workflow(workflow)
        payload = await self.wait_for_result(job_id)
        output = payload.get("output")
        if output is None:
            raise RunPodError(f"Job {job_id} COMPLETED but returned no output")
        return self._extract_images(output)

    @staticmethod
    def _extract_images(output: Any) -> list[bytes]:
        """Extract image bytes from a ComfyUI worker output.

        ComfyUI workers return images in varying shapes depending on version
        and workflow. This tries several known formats:

        1. output = "<base64>"                                   (single string)
        2. output = ["<base64>", ...]                            (list of strings)
        3. output = {"images": ["<base64>", ...]}                (list of strings)
        4. output = {"images": [{"image": "<base64>"}, ...]}     (runpod-workers/worker-comfyui)
        5. output = {"image": "<base64>"}                        (single-image shorthand)

        Pattern mirrors vid/avatar_llmer/runpod.py::_extract_video_bytes.
        """
        candidates: list[str] = []

        if isinstance(output, str):
            candidates = [output]
        elif isinstance(output, list):
            for item in output:
                if isinstance(item, str):
                    candidates.append(item)
                elif isinstance(item, dict):
                    encoded = item.get("image") or item.get("data")
                    if encoded:
                        candidates.append(encoded)
        elif isinstance(output, dict):
            images = output.get("images")
            if isinstance(images, list):
                for item in images:
                    if isinstance(item, str):
                        candidates.append(item)
                    elif isinstance(item, dict):
                        encoded = item.get("image") or item.get("data")
                        if encoded:
                            candidates.append(encoded)
            elif isinstance(images, str):
                candidates.append(images)
            else:
                encoded = output.get("image") or output.get("result")
                if encoded:
                    candidates.append(encoded)

        if not candidates:
            raise RunPodError(
                f"Unreadable ComfyUI output shape: {type(output).__name__}"
            )

        return [base64.b64decode(c) for c in candidates]
