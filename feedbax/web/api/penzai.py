from __future__ import annotations

from fastapi import APIRouter, HTTPException

from feedbax.compiler.penzai_compiler import (
    compile_penzai_authoring_report,
    penzai_builder_options,
    render_penzai_treescope_html,
)
from feedbax.contracts.domain import DomainCompileReport
from feedbax.contracts.studio_api import (
    PenzaiBuilderInfo,
    PenzaiBuilderListPayload,
    PenzaiBuilderListResponse,
    PenzaiInspectorPayload,
    PenzaiInspectorResponse,
    PenzaiNodeRequest,
)

router = APIRouter()


@router.get("/builders", response_model=PenzaiBuilderListResponse)
async def list_builders() -> PenzaiBuilderListResponse:
    builders = [PenzaiBuilderInfo.model_validate(item) for item in penzai_builder_options()]
    return PenzaiBuilderListResponse(data=PenzaiBuilderListPayload(builders=builders))


@router.post("/compile", response_model=DomainCompileReport)
async def compile_node(payload: PenzaiNodeRequest) -> DomainCompileReport:
    return compile_penzai_authoring_report(
        builder_name=payload.builder_name,
        params=payload.params,
        input_port=payload.input_port,
        output_port=payload.output_port,
        node_path=payload.node_path,
    )


@router.post("/inspect", response_model=PenzaiInspectorResponse)
async def inspect_node(payload: PenzaiNodeRequest) -> PenzaiInspectorResponse:
    report = compile_penzai_authoring_report(
        builder_name=payload.builder_name,
        params=payload.params,
        input_port=payload.input_port,
        output_port=payload.output_port,
        node_path=payload.node_path,
    )
    if report.status == "error":
        return PenzaiInspectorResponse(data=PenzaiInspectorPayload(html="", report=report))
    try:
        html = render_penzai_treescope_html(
            builder_name=payload.builder_name,
            params=payload.params,
            input_port=payload.input_port,
            output_port=payload.output_port,
        )
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PenzaiInspectorResponse(data=PenzaiInspectorPayload(html=html, report=report))
