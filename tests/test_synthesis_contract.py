"""The Synthesize contract is discoverable without changing existing Train and Evaluate plans."""

from fastapi.testclient import TestClient

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.request_specs import SynthesisRequestSpec, TrialDocuments
from feedbax.contracts.synthesis import SynthesisStepPayload
from feedbax.planner.project_plan import compile_project
from feedbax.web.app import create_app
from tests.test_document_train_pair import authored_documents, document_project


def test_served_synthesis_catalog_keeps_existing_model_steps_runnable(tmp_path):
    authored = authored_documents()
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as client:
        response = client.get("/api/projects/authoring/catalog")
    assert response.status_code == 200, response.text
    catalog = response.json()
    assert set(catalog["step_payloads"]) == {"train", "synthesize", "evaluate", "analyze", "report"}
    assert set(catalog["step_payloads"]["synthesize"]["required"]) == {"model", "task", "objective", "method"}
    assert catalog["step_payloads"]["synthesize"]["properties"]["input_bindings"]["type"] == "object"
    method, = catalog["registries"]["synthesis_methods"]["entries"]
    assert method["key"] == "feedbax.constant_parameters"
    assert method["settings_schema"]["properties"]["value"]["default"] == 0
    for role in (
        catalog["step_inputs"]["train"]["checkpoint"], catalog["step_inputs"]["evaluate"]["subject"],
        catalog["analyze_inputs"]["model"],
    ):
        assert {producer["kind"]: producer["product"] for producer in role["producers"]} == {
            "train": "checkpoint", "synthesize": "model",
        }

    plan = compile_project(Project.model_validate(document_project(authored, evaluation=True)))
    assert not plan.diagnostics, plan.diagnostics
    assert {run.kind for run in plan.runs} == {"train", "evaluate"}
    assert not any(run.admission_problem for run in plan.runs)


def test_synthesis_request_hashes_resolved_documents_and_round_trips():
    _, model, task, objective = authored_documents()
    authored = SynthesisStepPayload.model_validate({
        "model": {"document": "reach-model"}, "task": {"document": "task"}, "objective": {"document": "objective"},
        "method": "feedbax.constant_parameters", "settings": {"value": 3.0},
    })
    request = SynthesisRequestSpec(
        trial_documents=TrialDocuments(model=model, task=task, objective=objective),
        method=authored.method, settings=authored.settings, target=authored.target,
    )
    encoded = encode_request(request, {})
    decoded = decode_request(encoded.data, encoded.documents.__getitem__)
    assert decoded.kind == "synthesize"
    assert decoded.id == encoded.id
    assert isinstance(decoded.spec, SynthesisRequestSpec)
    assert decoded.spec.target == ModelAddress.at_stored_path(())
    assert decoded.spec.numerical_settings.floating_point_bits == 64
    assert decoded.spec.settings == {"value": 3.0}
    assert encode_request(decoded.spec, {}).data == encoded.data
    changed = request.model_copy(update={"settings": {"value": 4.0}})
    assert encode_request(changed, {}).id != encoded.id
