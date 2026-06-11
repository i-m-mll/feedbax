from __future__ import annotations

from feedbax.contracts.studio_api import (
    AnalysisPackagesResponse,
    GraphListResponse,
    TrainingProgressEvent,
    TrainingStartResponse,
)
from feedbax.web.app import create_app


def test_studio_api_openapi_uses_plural_analysis_jobs_route() -> None:
    app = create_app()
    paths = app.openapi()["paths"]

    assert "/api/analyses/jobs" in paths
    assert "/api/analyses/jobs/status/{request_id}" in paths
    assert "/api/analysis/generate" not in paths


def test_studio_api_envelopes_are_data_wrapped() -> None:
    assert GraphListResponse(data={"graphs": []}).model_dump() == {"data": {"graphs": []}}
    assert TrainingStartResponse(data={"job_id": "job-1"}).model_dump() == {
        "data": {"job_id": "job-1"}
    }
    assert AnalysisPackagesResponse(data={"packages": []}).model_dump() == {
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
