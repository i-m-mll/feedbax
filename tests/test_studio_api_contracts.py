from __future__ import annotations

from typing import get_args, get_origin

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.contracts.studio_api import (
    AnalysisBundleDryRunPayload,
    AnalysisBundleDryRunResponse,
    AnalysisPackagesResponse,
    AnalysisPackagesPayload,
    ComponentListResponse,
    GraphListResponse,
    GraphListPayload,
    STUDIO_API_TRANSPORT_SCHEMA_ID,
    STUDIO_API_TRANSPORT_SCHEMA_VERSION,
    StudioApiModel,
    TrainingErrorEvent,
    TrainingProgressEvent,
    TrainingResyncEvent,
    TrainingStartResponse,
    TrainingStartPayload,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.web.app import create_app
from feedbax.web.api import components as components_api
from scripts.generate_studio_contracts import CONTRACT_MODEL_NAMES, MODEL_TYPES, OUTPUT, generate


GENERATED_STUDIO_PREFIXES = (
    "/api/analyses",
    "/api/components",
    "/api/domains",
    "/api/graphs",
    "/api/inspection",
    "/api/penzai",
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
    analysis_dry_run = AnalysisBundleDryRunResponse(
        data=AnalysisBundleDryRunPayload(
            dry_run={
                "bundle_name": "bundle",
                "match_preview": {
                    "selection_spec": {
                        "mode": "explicit",
                        "manifest_kind": "EvaluationRunManifest",
                        "ids": ["eval-1"],
                    },
                    "match_count": 1,
                    "parent_refs": [
                        {
                            "kind": "EvaluationRunManifest",
                            "id": "eval-1",
                            "role": "evaluation_run",
                        }
                    ],
                },
                "matched_run_ids": ["eval-1"],
                "stages": [],
            }
        )
    ).model_dump()

    assert graph_list["data"]["graphs"] == []
    assert training_start["data"]["job_id"] == "job-1"
    assert analysis_packages["data"]["packages"] == []
    assert analysis_dry_run["data"]["dry_run"]["matched_run_ids"] == ["eval-1"]
    assert graph_list["schema_id"] == STUDIO_API_TRANSPORT_SCHEMA_ID
    assert graph_list["schema_version"] == STUDIO_API_TRANSPORT_SCHEMA_VERSION
    assert graph_list["data"]["schema_id"] == STUDIO_API_TRANSPORT_SCHEMA_ID


def test_analysis_bundle_dry_run_endpoint_returns_stage_status() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/analyses/bundles/dry-run",
        json={
            "bundle": {
                "schema_id": "feedbax.spec.analysis_bundle",
                "schema_version": "feedbax.spec.analysis_bundle.v2",
                "name": "dry-run-test",
                "predicate": {"manifest_kind": "EvaluationRunManifest"},
                "stages": [
                    {
                        "name": "disabled",
                        "kind": "analysis",
                        "skip_reason": "disabled for this bundle",
                    }
                ],
            },
            "preview_limit": 10,
        },
    )

    assert response.status_code == 200
    payload = response.json()["data"]["dry_run"]
    assert payload["bundle_name"] == "dry-run-test"
    assert payload["stages"][0]["status"] == "would_skip"
    assert payload["stages"][0]["reason"] == "disabled for this bundle"


def test_training_progress_event_contract_accepts_worker_shape() -> None:
    event = TrainingProgressEvent.model_validate(
        {
            "type": "training_progress",
            "job_id": "job-1",
            "seq": 3,
            "emitted_at_ms": 1783430000000,
            "worker_seq": 9,
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
    assert event.seq == 3
    assert event.worker_seq == 9
    assert event.loss_terms["position"] == 0.4
    assert event.schema_version == STUDIO_API_TRANSPORT_SCHEMA_VERSION


def test_training_error_and_resync_events_have_stable_coordinates() -> None:
    error = TrainingErrorEvent.model_validate(
        {
            "type": "training_error",
            "job_id": "job-1",
            "seq": 4,
            "emitted_at_ms": 1783430000100,
            "worker_seq": 10,
            "batch": 2,
            "error": "worker failed",
            "diagnostics": [
                {
                    "severity": "error",
                    "code": "graph.missing_subgraph",
                    "message": "Network node has no subgraph",
                    "node_ids": ["network"],
                }
            ],
        }
    )
    resync = TrainingResyncEvent.model_validate(
        {
            "type": "training_resync",
            "job_id": "job-1",
            "seq": 5,
            "emitted_at_ms": 1783430000200,
            "expected_worker_seq": 11,
            "observed_worker_seq": 14,
            "missed_events": 3,
            "reason": "gap",
            "message": "Training stream resumed after reconnect with 3 missed event(s).",
        }
    )

    assert error.schema_version == STUDIO_API_TRANSPORT_SCHEMA_VERSION
    assert error.diagnostics[0].code == "graph.missing_subgraph"
    assert error.diagnostics[0].node_ids == ["network"]
    assert error.job_id == resync.job_id
    assert resync.reason == "gap"
    assert resync.missed_events == 3


def test_component_api_serves_representation_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_component_type(
        "ApiRepresentedGain",
        lambda params: None,
        category="Test",
        description="API represented component fixture.",
        param_schema=[{"name": "gain", "type": "float", "default": 1.0}],
        input_ports=["input"],
        output_ports=["output"],
        representation={
            "anchors": [
                {
                    "id": "endpoint",
                    "semantic_role": "endpoint",
                    "interaction_roles": ["selectable"],
                    "binding": {"kind": "param_path", "path": "gain"},
                }
            ],
            "elements": [
                {
                    "id": "glyph",
                    "archetype": "marker",
                    "anchors": ["endpoint"],
                    "frame_provider": {"kind": "from_input_port", "input_port": "input"},
                }
            ],
        },
    )
    monkeypatch.setattr(components_api, "registry", registry)

    client = TestClient(create_app())
    response = client.get("/api/components")

    assert response.status_code == 200
    contract = ComponentListResponse.model_validate(response.json())
    represented = next(
        item for item in contract.data.components if item.name == "ApiRepresentedGain"
    )
    assert represented.representation is not None
    assert represented.representation.schema_version == "feedbax.spec.studio.representation.v3"
    assert represented.representation.elements[0].archetype == "marker"


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


def test_graph_validate_endpoint_returns_domain_diagnostics() -> None:
    client = TestClient(create_app())
    graph = GraphSpec(
        nodes={
            "network": ComponentSpec(
                type="Network",
                input_ports=["input"],
                output_ports=[],
            )
        }
    )

    response = client.post(
        "/api/graphs/graph-1/validate",
        json=graph.model_dump(mode="json", exclude_none=True),
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert isinstance(data, list)
    assert data[0]["severity"] == "error"
    assert data[0]["code"] == "graph.missing_input"
    assert data[0]["node_ids"] == ["network"]
    assert "valid" not in data[0]


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
