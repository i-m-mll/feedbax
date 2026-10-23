from __future__ import annotations
from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.websocket('/simulation')
async def simulation_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        'type': 'simulation_state',
        'step': 0,
        'port_values': {},
        'node_states': {},
    })
    await websocket.close()
