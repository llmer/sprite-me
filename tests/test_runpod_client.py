"""Tests for the RunPod API client (with mocked HTTP)."""

import base64

import httpx
import pytest

from sprite_me.inference.runpod_client import RunPodClient, RunPodError


def _make_client(handler, poll: float = 0.01, timeout: float = 5.0) -> RunPodClient:
    client = RunPodClient(
        api_key="test",
        endpoint_id="ep",
        poll_interval=poll,
        timeout_seconds=timeout,
    )
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers=client._headers(),
    )
    return client


@pytest.mark.asyncio
async def test_submit_workflow_returns_job_id():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/run")
        # Verify the workflow is wrapped in {"input": {"workflow": ...}}
        body = request.content.decode()
        assert "workflow" in body
        assert "input" in body
        return httpx.Response(200, json={"id": "job-123", "status": "IN_QUEUE"})

    client = _make_client(handler)
    job_id = await client.submit_workflow({"prompt": {}})
    assert job_id == "job-123"
    await client.close()


@pytest.mark.asyncio
async def test_wait_for_result_polls_until_completed():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(200, json={"status": "IN_PROGRESS"})
        return httpx.Response(
            200,
            json={
                "status": "COMPLETED",
                "output": {"images": [{"image": base64.b64encode(b"fakepng").decode()}]},
            },
        )

    client = _make_client(handler)
    payload = await client.wait_for_result("job-123")
    assert payload["status"] == "COMPLETED"
    assert call_count == 3
    await client.close()


@pytest.mark.asyncio
async def test_wait_for_result_raises_on_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "FAILED", "error": "out of memory"})

    client = _make_client(handler)
    with pytest.raises(RunPodError, match="out of memory"):
        await client.wait_for_result("job-123")
    await client.close()


@pytest.mark.asyncio
async def test_wait_for_result_raises_on_timed_out():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "TIMED_OUT"})

    client = _make_client(handler)
    with pytest.raises(RunPodError, match="TIMED_OUT"):
        await client.wait_for_result("job-123")
    await client.close()


@pytest.mark.asyncio
async def test_wait_for_result_respects_deadline():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "IN_PROGRESS"})

    client = _make_client(handler, poll=0.02, timeout=0.1)
    with pytest.raises(RunPodError, match="timed out"):
        await client.wait_for_result("job-123")
    await client.close()


class TestExtractImages:
    """Test the tolerant multi-format image extractor."""

    def test_list_of_dicts_with_image_key(self):
        output = {
            "images": [
                {"image": base64.b64encode(b"img1").decode()},
                {"image": base64.b64encode(b"img2").decode()},
            ]
        }
        assert RunPodClient._extract_images(output) == [b"img1", b"img2"]

    def test_list_of_base64_strings(self):
        output = {"images": [base64.b64encode(b"img1").decode()]}
        assert RunPodClient._extract_images(output) == [b"img1"]

    def test_single_string_output(self):
        output = base64.b64encode(b"img1").decode()
        assert RunPodClient._extract_images(output) == [b"img1"]

    def test_list_output_of_strings(self):
        output = [base64.b64encode(b"img1").decode(), base64.b64encode(b"img2").decode()]
        assert RunPodClient._extract_images(output) == [b"img1", b"img2"]

    def test_single_image_shorthand(self):
        output = {"image": base64.b64encode(b"img1").decode()}
        assert RunPodClient._extract_images(output) == [b"img1"]

    def test_unreadable_output_raises(self):
        with pytest.raises(RunPodError, match="Unreadable"):
            RunPodClient._extract_images(42)
        with pytest.raises(RunPodError, match="Unreadable"):
            RunPodClient._extract_images({"unknown": "key"})
