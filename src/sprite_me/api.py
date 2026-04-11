"""FastAPI REST API for sprite-me.

Provides direct HTTP access to sprite generation and management. The MCP
server uses these same underlying tool functions, so this is for non-MCP
integrations (e.g. web UIs, scripts, other services).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sprite_me.config import settings
from sprite_me.inference.runpod_client import RunPodClient
from sprite_me.storage.local import LocalStorage
from sprite_me.storage.manifest import AssetManifest
from sprite_me.tools.animate import animate_sprite
from sprite_me.tools.assets import delete_asset, get_asset, list_assets
from sprite_me.tools.generate import generate_sprite
from sprite_me.tools.import_asset import import_image
from sprite_me.tools.status import check_job_status

logger = logging.getLogger("sprite_me.api")


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    seed: int | None = None
    steps: int = 30
    guidance: float = 3.5
    lora_strength: float = 0.85
    smart_crop_mode: str = "tightest"
    remove_bg: bool = True
    reference_asset_id: str | None = None


class AnimateRequest(BaseModel):
    asset_id: str
    animation: str = "idle"
    custom_prompt: str | None = None
    frames: int = 6
    edge_margin: int = 6
    auto_enhance: bool = True
    seed: int | None = None


class ImportRequest(BaseModel):
    source: str = Field(description="File path or data URL")
    name: str = ""


# Module-level service instances shared across requests
_runpod: RunPodClient | None = None
_storage: LocalStorage | None = None
_manifest: AssetManifest | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runpod, _storage, _manifest
    _runpod = RunPodClient()
    _storage = LocalStorage()
    _manifest = AssetManifest()
    logger.info("sprite-me API ready")
    yield
    await _runpod.close()


app = FastAPI(
    title="sprite-me API",
    description="Agent-first pixel art sprite generator",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate")
async def api_generate(req: GenerateRequest) -> dict[str, Any]:
    return await generate_sprite(
        prompt=req.prompt,
        width=req.width,
        height=req.height,
        seed=req.seed,
        steps=req.steps,
        guidance=req.guidance,
        lora_strength=req.lora_strength,
        smart_crop_mode=req.smart_crop_mode,
        remove_bg=req.remove_bg,
        reference_asset_id=req.reference_asset_id,
        runpod=_runpod,
        storage=_storage,
        manifest=_manifest,
    )


@app.post("/api/animate")
async def api_animate(req: AnimateRequest) -> dict[str, Any]:
    return await animate_sprite(
        asset_id=req.asset_id,
        animation=req.animation,
        custom_prompt=req.custom_prompt,
        frames=req.frames,
        edge_margin=req.edge_margin,
        auto_enhance=req.auto_enhance,
        seed=req.seed,
        runpod=_runpod,
        storage=_storage,
        manifest=_manifest,
    )


@app.post("/api/import")
async def api_import(req: ImportRequest) -> dict[str, Any]:
    return await import_image(
        source=req.source,
        name=req.name,
        storage=_storage,
        manifest=_manifest,
    )


@app.get("/api/jobs/{job_id}")
async def api_job_status(job_id: str) -> dict[str, Any]:
    return await check_job_status(job_id=job_id, runpod=_runpod)


@app.get("/api/assets")
async def api_list_assets() -> list[dict[str, Any]]:
    return await list_assets(manifest=_manifest)


@app.get("/api/assets/{asset_id}")
async def api_get_asset(asset_id: str) -> dict[str, Any]:
    result = await get_asset(asset_id=asset_id, manifest=_manifest, storage=_storage)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/assets/{asset_id}/download")
async def api_download_asset(asset_id: str) -> FileResponse:
    asset = _manifest.get(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")
    path = _storage.get_path(asset.filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Asset file missing on disk")
    return FileResponse(path, media_type="image/png", filename=asset.filename)


@app.delete("/api/assets/{asset_id}")
async def api_delete_asset(asset_id: str) -> dict[str, Any]:
    result = await delete_asset(asset_id=asset_id, manifest=_manifest, storage=_storage)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    uvicorn.run(
        "sprite_me.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
