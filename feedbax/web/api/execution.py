from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from feedbax.web.models.graph import GraphSpec

router = APIRouter()


class SimulationRequest(BaseModel):
    graph: GraphSpec
    n_steps: int
    inputs: Optional[Dict[str, Any]] = None


@router.post('/simulate')
async def simulate(_: SimulationRequest):
    raise HTTPException(status_code=501, detail='Simulation not implemented yet')
