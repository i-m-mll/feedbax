"""API endpoints for model inspection and Treescope visualization."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from feedbax.web.models.inspection import (
    CycleAnnotationModel,
    InspectionStatusResponse,
    TreescopeRequest,
    TreescopeResponse,
)
from feedbax.web.services.graph_service import GraphService

router = APIRouter()
graph_service = GraphService()


@router.get("/status")
async def get_inspection_status() -> InspectionStatusResponse:
    """Get the status of inspection capabilities."""
    from feedbax._treescope import get_treescope_status

    status = get_treescope_status()

    return InspectionStatusResponse(
        treescope_available=status.get("available", False),
        treescope_configured=status.get("configured", False),
        treescope_version=status.get("version"),
    )


@router.post("/treescope/{graph_id}")
async def render_treescope(
    graph_id: str,
    request: TreescopeRequest,
) -> TreescopeResponse:
    """Render a graph as Treescope HTML visualization.

    Args:
        graph_id: The ID of the graph to render.
        request: Rendering options.

    Returns:
        TreescopeResponse containing the HTML and cycle information.
    """
    from feedbax._treescope import (
        get_treescope_status,
        graph_to_tree,
        render_model_html,
    )

    # Check if treescope is available
    status = get_treescope_status()
    if not status.get("available", False):
        raise HTTPException(
            status_code=503,
            detail="Treescope is not installed. Install with: pip install treescope",
        )

    # Load the graph
    try:
        record = graph_service.get_graph(graph_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Graph not found")

    graph_spec = record.project.graph

    # For now, we render the GraphSpec as a PyTree (the spec itself is a PyTree)
    # In a full implementation, we would build an actual feedbax.graph.Graph
    # from the spec and render that.
    #
    # Since we don't have instantiated components, we render the spec directly.
    try:
        html = render_model_html(
            graph_spec,
            max_depth=request.max_depth,
            project_cycles=request.project_cycles,
            roundtrip_mode=request.roundtrip_mode,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to render Treescope visualization: {str(e)}",
        )

    # Build cycle information from the spec
    cycles: list[CycleAnnotationModel] = []
    execution_order: list[str] | None = None
    has_cycles = False

    if request.project_cycles:
        # Detect cycles in the graph spec
        from feedbax._graph import detect_cycles_and_sort

        adjacency: dict[str, set[str]] = {
            name: set() for name in graph_spec.nodes.keys()
        }
        for wire in graph_spec.wires:
            adjacency[wire.source_node].add(wire.target_node)

        order, back_edges = detect_cycles_and_sort(adjacency)
        execution_order = list(order)
        has_cycles = len(back_edges) > 0

        # Build cycle annotations
        for src, tgt in back_edges:
            for wire in graph_spec.wires:
                if wire.source_node == src and wire.target_node == tgt:
                    cycles.append(
                        CycleAnnotationModel(
                            source=f"{wire.source_node}.{wire.source_port}",
                            target=f"{wire.target_node}.{wire.target_port}",
                        )
                    )

    return TreescopeResponse(
        html=html,
        has_cycles=has_cycles,
        cycle_count=len(cycles),
        cycles=cycles,
        execution_order=execution_order,
    )


@router.post("/treescope")
async def render_treescope_inline(
    request: TreescopeRequest,
) -> TreescopeResponse:
    """Render an inline model as Treescope HTML visualization.

    This endpoint renders whatever model is currently active in the session.
    For now, it returns a placeholder.
    """
    from feedbax._treescope import get_treescope_status

    status = get_treescope_status()
    if not status.get("available", False):
        raise HTTPException(
            status_code=503,
            detail="Treescope is not installed. Install with: pip install treescope",
        )

    # Return a placeholder response - in production this would render
    # the currently active model from the training/execution context
    placeholder_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Treescope</title></head>
    <body style="font-family: system-ui; padding: 20px;">
        <p style="color: #64748b;">
            Select a saved graph to visualize, or start a training session
            to inspect the live model.
        </p>
    </body>
    </html>
    """

    return TreescopeResponse(
        html=placeholder_html,
        has_cycles=False,
        cycle_count=0,
        cycles=[],
        execution_order=None,
    )
