from __future__ import annotations

from typing import get_args, get_origin

import pytest
from fastapi.routing import APIRoute
from pydantic import BaseModel, ValidationError

from feedbax.contracts.studio_api import (
    AnalysisPackagesResponse,
    AnalysisPackagesPayload,
    GraphListResponse,
    GraphListPayload,
    STUDIO_API_TRANSPORT_SCHEMA_ID,
    STUDIO_API_TRANSPORT_SCHEMA_VERSION,
    StudioApiModel,
    TrainingProgressEvent,
    TrainingStartResponse,
    TrainingStartPayload,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.web.app import create_app
from scripts.generate_studio_contracts import CONTRACT_MODEL_NAMES, MODEL_TYPES, OUTPUT, generate


GENERATED_STUDIO_PREFIXES = (
    "/api/analyses",
    "/api/components",
    "/api/graphs",
    "/api/inspection",
    "/api/runs",
    "/api/training",
    "/api/trajectories",
)

NON_GENERATED_STUDIO_RESPONSE_ROUTES = {
    "/api/training/loss/validate",
}


def _response_model_members(response_model: object) -> list[type[BaseModel]]:
    origin = get_origin(response_model)
    if origin is list:
        response_model = get_args(response_model)[0]

    if isinstance(response_model, type) and issubclass(response_model, BaseModel):
        return [response_model]
    return []


def test_studio_api_openapi_uses_plural_analysis_jobs_route() -> None:
    app = create_app()
    paths = app.openapi()["paths"]

    assert "/api/analyses/jobs" in paths
    assert "/api/analyses/jobs/status/{request_id}" in paths
    assert "/api/analysis/generate" not in paths


def test_studio_api_envelopes_are_data_wrapped() -> None:
    graph_list = GraphListResponse(data=GraphListPayload(graphs=[])).model_dump()
    training_start = TrainingStartResponse(
        data=TrainingStartPayload(job_id="job-1")
    ).model_dump()
    analysis_packages = AnalysisPackagesResponse(
        data=AnalysisPackagesPayload(packages=[])
    ).model_dump()

    assert graph_list["data"]["graphs"] == []
    assert training_start["data"]["job_id"] == "job-1"
    assert analysis_packages["data"]["packages"] == []
    assert graph_list["schema_id"] == STUDIO_API_TRANSPORT_SCHEMA_ID
    assert graph_list["schema_version"] == STUDIO_API_TRANSPORT_SCHEMA_VERSION
    assert graph_list["data"]["schema_id"] == STUDIO_API_TRANSPORT_SCHEMA_ID


def test_training_progress_event_contract_accepts_worker_shape() -> None:
    event = TrainingProgressEvent.model_validate(
        {
            "type": "training_progress",
            "job_id": "job-1",
            "batch": 1,
            "total_batches": 10,
            "loss": 0.5,
            "loss_terms": {"position": 0.4},
            "grad_norm": 1.5,
            "step_time_ms": 12.0,
            "status": "running",
            "execution": "generic_graph",
        }
    )

    assert event.job_id == "job-1"
    assert event.loss_terms["position"] == 0.4
    assert event.schema_version == STUDIO_API_TRANSPORT_SCHEMA_VERSION


def test_studio_api_transport_models_declare_identity_and_reject_old_or_extra() -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        TrainingStartPayload.model_validate(
            {
                "schema_id": STUDIO_API_TRANSPORT_SCHEMA_ID,
                "schema_version": "feedbax.spec.studio.api_transport.v0",
                "job_id": "job-1",
            }
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainingStartPayload.model_validate({"job_id": "job-1", "unexpected": True})

    family = default_spec_registry.resolve("StudioApiTransport")
    assert family.identity == STUDIO_API_TRANSPORT_SCHEMA_ID
    assert family.current_version == STUDIO_API_TRANSPORT_SCHEMA_VERSION
    with pytest.raises(UnsupportedSpecVersion, match="api_transport.v0"):
        default_spec_registry.migrate(
            "StudioApiTransport",
            {"schema_version": "feedbax.spec.studio.api_transport.v0"},
        )

    assert issubclass(TrainingProgressEvent, StudioApiModel)


def test_generated_studio_contracts_cover_route_response_models() -> None:
    app = create_app()
    generated_model_names = {model.__name__ for model in MODEL_TYPES}
    generated_contract_names = set(CONTRACT_MODEL_NAMES)
    missing: list[str] = []

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if not route.path.startswith(GENERATED_STUDIO_PREFIXES):
            continue
        if route.path in NON_GENERATED_STUDIO_RESPONSE_ROUTES:
            continue
        if route.response_model is None:
            continue
        for model in _response_model_members(route.response_model):
            model_name = model.__name__
            if model_name not in generated_model_names:
                missing.append(f"{route.path} response_model={model_name}")
            elif (
                model_name.endswith(("Response", "Envelope", "Info"))
                and model_name not in generated_contract_names
            ):
                missing.append(f"{route.path} contractSchemas missing {model_name}")

    assert missing == []


def test_generated_studio_contracts_are_current() -> None:
    assert OUTPUT.read_text(encoding="utf-8") == generate()
