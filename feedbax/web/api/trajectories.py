"""API router for trajectory dataset browsing and retrieval."""

from __future__ import annotations

import json

from fastapi import APIRouter, Query, Response
from typing import Optional

from feedbax.web.models.trajectory import DatasetInfo, FilterResult, TrajectoryMetadata
from feedbax.web.services.trajectory_service import TrajectoryService

router = APIRouter()
service = TrajectoryService()


@router.get('/datasets')
async def list_datasets() -> list[DatasetInfo]:
    return service.list_datasets()


@router.get('/{dataset}/metadata')
async def get_metadata(dataset: str) -> TrajectoryMetadata:
    return service.get_metadata(dataset)


@router.get('/{dataset}/filter')
async def filter_trajectories(
    dataset: str,
    body_idx: Optional[int] = Query(default=None),
    task_type: Optional[int] = Query(default=None),
) -> FilterResult:
    indices, count = service.filter_trajectories(dataset, body_idx, task_type)
    return FilterResult(indices=indices, count=count)


@router.get('/{dataset}/{index}')
async def get_trajectory(
    dataset: str,
    index: int,
    fields: Optional[str] = Query(default=None),
):
    """Return a single trajectory as JSON.

    Uses ``Response`` with ``json.dumps`` to bypass Pydantic serialisation for
    the large nested-list payload.
    """
    field_list: list[str] | None = None
    if fields is not None:
        field_list = [f.strip() for f in fields.split(',') if f.strip()]
    data = service.get_trajectory(dataset, index, field_list)
    return Response(content=json.dumps(data), media_type='application/json')
