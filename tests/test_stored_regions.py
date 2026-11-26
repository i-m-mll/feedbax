"""Registry defaults are omitted by authored writers and completed for execution identity."""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from feedbax.application.composition import load_application
from feedbax.contracts.project import RootDocument, Step
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, parse_text_project_files
from feedbax.contracts.result_address import ANALYSIS_SCHEMA
from feedbax.contracts.rollout import ROLLOUT_TYPE, authored_rollout_settings
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.stored_regions import StoredRegionRegistries
from feedbax.contracts.project_storage import write_project_file
from feedbax.contracts.task_document import PYTHON_TASK_SCHEMA_ID
from feedbax.planner.project_plan import RunKey, compile_project
from feedbax.web.app import create_app
from tests.support.managed_train_fixture import managed_evaluation_step, managed_train_project
from tests.support.python_task import REACHING_TASK, reaching_python_task  # noqa: F401


def _inline(kind, body):
    return Step.model_validate({"kind": kind, "label": "Fixture", "base": {"inline": body}})


def _document(schema, payload):
    return RootDocument(schema=schema, label="Fixture", payload=payload)


def test_registered_train_and_synthesis_settings_omit_defaults():
    training = load_application().bundle.training_methods.resolve("feedbax.supervised_gradient")
    settings = training.settings_model().model_dump(mode="json")
    written = write_project_file(_inline("train", {"method": training.key, "settings": settings}))
    assert written == write_project_file(_inline("train", {"method": training.key, "settings": {}}))
    assert json.loads(written)["base"]["inline"]["settings"] == {}

    synthesis = {"method": "feedbax.constant_parameters"}
    written = write_project_file(_inline("synthesize", synthesis | {"settings": {"value": 0}}))
    assert written == write_project_file(_inline("synthesize", synthesis))
    assert "settings" not in json.loads(written)["base"]["inline"]


def test_evaluate_inline_defaults_omit_without_changing_the_rollout():
    body = {"evaluation_type": ROLLOUT_TYPE}
    control = {"seed": 7, "trial_set": {"kind": "every_condition"}}
    authored = control | {"variation": [], "trial_policy": "paired"}
    written = write_project_file(_inline("evaluate", body | {"params": authored}))
    assert written == write_project_file(_inline("evaluate", body | {"params": control}))
    assert authored_rollout_settings(json.loads(written)["base"]["inline"]["params"]) == authored_rollout_settings(authored)


def test_rewritten_evaluate_defaults_keep_its_request_identity(tmp_path):
    project = managed_train_project()
    step = Step.model_validate(managed_evaluation_step("train"))
    step.base.inline["params"].update(retention="raw_states", input_bindings={})
    project.steps["check"] = step
    configuration = ProjectConfiguration(project="fixture", store="results")
    written = dump_text_project(project, configuration)
    reopened = parse_text_project_files(tmp_path, written).project
    params = reopened.steps["check"].base.inline["params"]
    assert not params.keys() & {"retention", "trial_policy", "variation", "input_bindings"}
    key = RunKey("default", "check", "default")
    original, canonical = (compile_project(source).row(key) for source in (project, reopened))
    assert original.request_id is not None and canonical.request_id == original.request_id


@pytest.mark.usefixtures("reaching_python_task")
def test_analysis_operation_and_python_task_use_the_installed_models():
    operation = {
        "operation": "feedbax.summarize_velocity",
        "label": "Velocity",
        "inputs": {"states": {"input": "states"}},
    }
    default_params = {"variable": "parallel_velocity", "bounds_scale": 1, "ddof": 0}
    explicit = _document(ANALYSIS_SCHEMA, {"operations": {"summary": operation | {"params": default_params}}})
    implicit = _document(ANALYSIS_SCHEMA, {"operations": {"summary": operation | {"params": {"ddof": 0}}}})
    assert write_project_file(explicit) == write_project_file(implicit)
    assert json.loads(write_project_file(explicit))["payload"]["operations"]["summary"]["params"] == {"ddof": 0}

    incomplete = _document(ANALYSIS_SCHEMA, {
        "operations": {"summary": operation | {"params": {"bounds_scale": 1}}},
    })
    assert json.loads(write_project_file(incomplete))["payload"]["operations"]["summary"]["params"] == {"bounds_scale": 1}

    registries = StoredRegionRegistries(python_tasks=load_application().bundle.python_tasks)
    explicit = _document(PYTHON_TASK_SCHEMA_ID, {"type": REACHING_TASK, "params": {"x": 0.15}})
    implicit = _document(PYTHON_TASK_SCHEMA_ID, {"type": REACHING_TASK})
    assert write_project_file(explicit, registries=registries) == write_project_file(implicit, registries=registries)
    assert "params" not in json.loads(write_project_file(explicit, registries=registries))["payload"]


def test_model_duration_is_written_in_the_authored_unit():
    tau = {"kind": "literal", "unit": "s", "value": 0.2}
    component = {"kind": "leaf", "type": "feedbax.VanillaRNN", "label": "Cell", "fields": {"tau": tau}}
    payload = {"step_seconds": 0.01, "ports": {}, "interior": {"components": {"cell": component}}}
    written = json.loads(write_project_file(_document(STORED_MODEL_DOCUMENT_SCHEMA_ID, payload)))
    assert written["payload"]["interior"]["components"]["cell"]["fields"]["tau"] == tau


def test_component_defaults_and_training_settings_have_one_request_identity(tmp_path):
    project = managed_train_project()
    model = next(doc.payload for doc in project.documents.values() if doc.schema_id == STORED_MODEL_DOCUMENT_SCHEMA_ID)
    model["interior"]["components"]["plant"]["fields"]["mass"] = 1
    step = project.steps["train"]
    settings = load_application().bundle.training_methods.resolve(
        "feedbax.supervised_gradient",
    ).settings_model().model_dump(mode="json")
    step.base.inline["settings"] = settings
    configuration = ProjectConfiguration(project="fixture", store="results")
    written = dump_text_project(project, configuration)
    reopened = parse_text_project_files(tmp_path, written).project
    assert dump_text_project(reopened, configuration) == written
    model = next(doc.payload for doc in reopened.documents.values() if doc.schema_id == STORED_MODEL_DOCUMENT_SCHEMA_ID)
    assert "mass" not in model["interior"]["components"]["plant"].get("fields", {})
    assert not reopened.steps["train"].base.inline["settings"]

    key = RunKey("default", "train", "default")
    original, canonical = (compile_project(source).row(key) for source in (project, reopened))
    assert original.request_id is not None and canonical.request_id == original.request_id


def test_missing_declaration_keeps_parameters_and_is_a_plan_problem():
    project = managed_train_project()
    inline = project.steps["train"].base.inline
    inline["method"] = "example.missing_method"
    inline["settings"] = {"opaque": {"numbers": [1.0, True, 1]}}
    raw = json.loads(write_project_file(project.steps["train"]))["base"]["inline"]
    assert raw["settings"] == inline["settings"]
    assert type(raw["settings"]["opaque"]["numbers"][0]) is float
    assert any(problem.missing_key == inline["method"] for problem in compile_project(project).diagnostics)


def test_missing_python_task_keeps_its_explicit_empty_parameters():
    task = _document(PYTHON_TASK_SCHEMA_ID, {"type": "example.missing_task", "params": {}})
    assert json.loads(write_project_file(task))["payload"]["params"] == {}


def test_installed_registry_rejects_invalid_authored_fields():
    request = _inline("train", {
        "method": "feedbax.supervised_gradient",
        "settings": {"grad_clip": "not-a-number"},
    })
    with pytest.raises(ValidationError, match="grad_clip"):
        write_project_file(request)


def _save_fixture(root, project):
    configuration = ProjectConfiguration(project="fixture", store="results")
    for relative, raw in dump_text_project(project, configuration).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def test_studio_can_repair_an_invalid_file_without_rewriting_other_files(tmp_path):
    root = tmp_path / "fixture"
    project = managed_train_project()
    _save_fixture(root, project)

    step_file = root / "steps/train.json"
    raw_step = json.loads(step_file.read_bytes())
    raw_step["base"]["inline"]["settings"]["grad_clip"] = "wrong"
    invalid = json.dumps(raw_step).encode()
    step_file.write_bytes(invalid)

    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders

        def edit(kind, member, patch):
            return client.patch("/api/projects/fixture", json={
                "revision": folders.read("fixture").revision,
                "files": [{"address": {"kind": kind, "id": member}, "patches": [patch]}],
            })

        objective = next(name for name in project.documents if name.endswith("objective"))
        updated = edit("document", objective, {"op": "replace", "path": "/label", "value": "Updated"})
        assert updated.status_code == 200, updated.text
        assert step_file.read_bytes() == invalid

        rejected = edit("step", "train", {
            "op": "replace", "path": "/base/inline/settings/grad_clip", "value": "still wrong",
        })
        assert rejected.status_code == 422, rejected.text
        assert step_file.read_bytes() == invalid

        repaired = edit("step", "train", {
            "op": "replace", "path": "/base/inline/settings/grad_clip", "value": 2.0,
        })
        assert repaired.status_code == 200, repaired.text
        assert json.loads(step_file.read_bytes())["base"]["inline"]["settings"]["grad_clip"] == 2


def test_studio_sweeps_a_train_setting_omitted_at_its_default(tmp_path):
    root = tmp_path / "fixture"
    project = managed_train_project()
    project.steps["train"].base.inline["settings"] = {}
    _save_fixture(root, project)
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders
        revision = folders.read("fixture").revision
        reply = client.post("/api/projects/fixture/authoring/train/train/fields", json={
            "revision": revision,
            "field": "/settings/optimizer/learning_rate/value",
            "edit": {"kind": "text", "text": "0.001, 0.003"},
        })
        assert reply.status_code == 200, reply.text
        response = reply.json()
        assert not response["problems"] and response["files"], response
        saved = client.patch("/api/projects/fixture", json={"revision": revision, "files": response["files"]})
        assert saved.status_code == 200, saved.text
        assert folders.read("fixture").project.steps["train"].base.inline["settings"] == {}
        plan = folders.read_plan("fixture", revision=None).plan
        assert not plan.diagnostics, plan.diagnostics
        assert len({row.request_id for row in plan.rows}) == 2
        assert all(row.request_id is not None for row in plan.rows)
        selected = folders.read_plan("fixture", revision=None, source_runs=tuple(row.key for row in plan.rows)).plan
        rate = "/payload/settings/optimizer/learning_rate/value"
        assert all(definition.sources[rate].kind == "sweep_value" for definition in selected.selected_definitions.values())
