"""Registry defaults are omitted by authored writers and completed for execution identity."""

import json

import pytest
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
from tests.support.managed_train_fixture import managed_train_project
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


@pytest.mark.usefixtures("reaching_python_task")
def test_analysis_operation_and_python_task_use_the_installed_models():
    operation = {
        "operation": "feedbax.summarize_velocity",
        "label": "Velocity",
        "inputs": {"states": {"input": "states"}},
    }
    default_params = {"variable": "parallel_velocity", "bounds_scale": 1}
    explicit = _document(ANALYSIS_SCHEMA, {"operations": {"summary": operation | {"params": default_params}}})
    implicit = _document(ANALYSIS_SCHEMA, {"operations": {"summary": operation}})
    assert write_project_file(explicit) == write_project_file(implicit)
    assert "params" not in json.loads(write_project_file(explicit))["payload"]["operations"]["summary"]

    registries = StoredRegionRegistries(python_tasks=load_application().bundle.python_tasks)
    explicit = _document(PYTHON_TASK_SCHEMA_ID, {"type": REACHING_TASK, "params": {"x": 0.15}})
    implicit = _document(PYTHON_TASK_SCHEMA_ID, {"type": REACHING_TASK})
    assert write_project_file(explicit, registries=registries) == write_project_file(implicit, registries=registries)
    assert "params" not in json.loads(write_project_file(explicit, registries=registries))["payload"]


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


def test_installed_registry_rejects_invalid_authored_fields():
    request = _inline("train", {
        "method": "feedbax.supervised_gradient",
        "settings": {"grad_clip": "not-a-number"},
    })
    with pytest.raises(ValidationError, match="grad_clip"):
        write_project_file(request)
