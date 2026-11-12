"""Import content-addressed assets for authored models."""

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.web.model_assets import import_mujoco_asset


router = APIRouter()
MAX_MUJOCO_ASSET_BYTES = 8 * 1024 * 1024


@router.post("/assets/mujoco", response_model=ContentObjectRef)
async def add_mujoco_asset(request: Request) -> ContentObjectRef:
    """Import self-contained MJCF for use in a spatial graph's asset binding."""
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > MAX_MUJOCO_ASSET_BYTES:
            raise HTTPException(status_code=413, detail="MuJoCo asset exceeds the 8 MiB limit")
        raw.extend(chunk)
    try:
        return await run_in_threadpool(import_mujoco_asset, bytes(raw))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
