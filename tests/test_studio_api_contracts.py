from __future__ import annotations

from fastapi.routing import APIRoute
from pydantic import BaseModel

from feedbax.contracts.studio_api import (
    AnalysisPackagesResponse,
    AnalysisPackagesPayload,
    GraphListResponse,
    GraphListPayload,
    TrainingProgressEvent,
    TrainingStartResponse,
    TrainingStartPayload,
)
from feedbax.web.app import create_app
from scripts.generate_studio_contracts import CONTRACT_MODEL_NAMES, MODEL_TYPES, OUTPUT, generate


GENERATED_STUDIO_PREFIXES = (
    "/api/analyses",
    "/api/components",
    "/api/graphs",
    "/api/training",
)

NON_GENERATED_STUDIO_RESPONSE_ROUTES = {
    "/api/training/loss/validate",
}


def test_studio_api_openapi_uses_plural_analysis_jobs_route() -> None:
    app = create_app()
    paths = app.openapi()["paths"]

    assert "/api/analyses/jobs" in paths
    assert "/api/analyses/jobs/status/{request_id}" in paths
    assert "/api/analysis/generate" not in paths


def test_studio_api_envelopes_are_data_wrapped() -> None:
    assert GraphListResponse(data=GraphListPayload(graphs=[])).model_dump() == {
        "data": {"graphs": []}
    }
    assert TrainingStartResponse(data=TrainingStartPayload(job_id="job-1")).model_dump() == {
        "data": {"job_id": "job-1"}
    }
    assert AnalysisPackagesResponse(data=AnalysisPackagesPayload(packages=[])).model_dump() == {
        "data": {"packages": []}
    }


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
        if not isinstance(route.response_model, type) or not issubclass(
            route.response_model, BaseModel
        ):
            continue

        model_name = route.response_model.__name__
        if model_name not in generated_model_names:
            missing.append(f"{route.path} response_model={model_name}")
        elif model_name.endswith(("Response", "Envelope")) and model_name not in generated_contract_names:
            missing.append(f"{route.path} contractSchemas missing {model_name}")

    assert missing == []


def test_generated_studio_contracts_are_current() -> None:
    assert OUTPUT.read_text(encoding="utf-8") == generate()
