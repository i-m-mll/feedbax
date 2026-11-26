"""Registry defaults are omitted by authored writers and completed for execution identity."""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from feedbax.application.composition import load_application
from feedbax.application.project_writer import dump_text_project, write_project_file
from feedbax.application.stored_regions import StoredRegionRegistries
from feedbax.contracts.project import RootDocument, Step
from feedbax.contracts.project_io import ProjectConfiguration, parse_text_project_files
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.result_address import ANALYSIS_SCHEMA
from feedbax.contracts.rollout import ROLLOUT_TYPE, authored_rollout_settings
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.task_document import PYTHON_TASK_SCHEMA_ID
from feedbax.contracts.training_document import TRAINING_DOCUMENT_KEYS, TRAINING_DOCUMENT_SCHEMA_ID
from feedbax.planner.project_plan import PlanRequests, RunKey, compile_project
from feedbax.planner.project_plan_io import dump_plan_requests, load_plan_requests
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
    component = {"kind": "leaf", "type": "feedbax.VanillaRNN", "label": "Cell", "fields": {"tau": "200   ms"}}
    payload = {"step_seconds": "0.01 s", "ports": {}, "interior": {"components": {"cell": component}}}
    written = json.loads(write_project_file(_document(STORED_MODEL_DOCUMENT_SCHEMA_ID, payload)))
    assert written["payload"]["interior"]["components"]["cell"]["fields"]["tau"] == "200 ms"


def test_model_default_elision_compares_lowered_expressions():
    component = {"kind": "leaf", "type": "feedbax.PointMass", "label": "Plant", "fields": {"mass": "1.0"}}
    payload = {"step_seconds": "0.01 s", "ports": {}, "interior": {"components": {"plant": component}}}
    document = _document(STORED_MODEL_DOCUMENT_SCHEMA_ID, payload)
    assert "fields" not in json.loads(write_project_file(document))["payload"]["interior"]["components"]["plant"]
    protected = [("payload", "interior", "components", "plant", "fields", "mass")]
    written = json.loads(write_project_file(document, protected=protected))
    assert written["payload"]["interior"]["components"]["plant"]["fields"]["mass"] == 1


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


def test_studio_reads_effective_train_settings_and_saves_only_authored_values(tmp_path):
    root = tmp_path / "fixture"
    project = managed_train_project()
    project.steps["train"].base.inline["settings"] = {}
    _save_fixture(root, project)
    step_file = root / "steps/train.json"
    original = step_file.read_bytes()
    optimizer = "/settings/optimizer"
    schedule = optimizer + "/learning_rate"
    rate = schedule + "/value"
    url = "/api/projects/fixture"

    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders

        def prepare(*, endpoint="", **edit):
            response = client.post(f"{url}/authoring/train/train{endpoint}", json={
                "revision": folders.read("fixture").revision, **edit,
            })
            assert response.status_code == 200, response.text
            return response.json()

        def save(prepared):
            assert not prepared.get("problems") and prepared["files"], prepared
            response = client.patch(url, json={
                "revision": folders.read("fixture").revision, "files": prepared["files"],
            })
            assert response.status_code == 200, response.text
            return json.loads(step_file.read_bytes())["base"]["inline"]["settings"]

        shown = prepare()
        classes = {field["id"]: field["classes"] for field in shown["classes"]}
        assert classes[optimizer] == ["adam"]
        assert classes[schedule] == ["constant_schedule"]
        assert shown["settings"]["optimizer"]["learning_rate"] == {"type": "constant_schedule", "value": 0.001}
        assert next(field["value"] for field in shown["fields"] if field["id"] == rate) == 0.001
        assert not prepare(change={"field": optimizer, "value": "adam"})["files"]
        assert step_file.read_bytes() == original

        settings = save(prepare(endpoint="/fields", field=rate, edit={"kind": "text", "text": "0.003"}))
        assert settings == {"optimizer": {
            "type": "adam", "learning_rate": {"type": "constant_schedule", "value": 0.003},
        }}
        assert prepare()["settings"]["optimizer"]["learning_rate"]["value"] == 0.003
        assert save(prepare(endpoint="/fields", field=rate, edit={"kind": "text", "text": "0.001"})) == {}
        assert step_file.read_bytes() == original

        settings = save(prepare(change={"field": schedule, "value": "linear_schedule"}))
        assert settings == {"optimizer": {"type": "adam", "learning_rate": {"type": "linear_schedule"}}}
        assert save(prepare(change={"field": schedule, "value": "constant_schedule"})) == {}
        assert step_file.read_bytes() == original


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

        captured = compile_project(folders.read("fixture").project, capture_sources=True)
        reviewed, _ = load_plan_requests(dump_plan_requests(PlanRequests.from_plan(captured), ("default",)))
        for row in plan.rows:
            source = reviewed.source_index.source(row.key, tuple(rate[1:].split("/")))
            assert source == selected.selected_definitions[row.key].sources[rate]


def test_studio_plans_and_resets_a_based_optimizer_whose_adam_default_is_omitted(tmp_path):
    root = tmp_path / "fixture"
    project = managed_train_project()
    inline = project.steps["train"].base.inline
    inline["settings"] = {}
    training = {key: inline[key] for key in TRAINING_DOCUMENT_KEYS}
    _save_fixture(root, project)
    route = "/api/projects/fixture"
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders
        baseline = folders.read_plan("fixture", revision=None).plan
        (original,) = baseline.runs
        snapshot = client.get(route).json()

        def create(name, document):
            reply = client.post(f"{route}/files", json={"revision": snapshot["revision"], "file": {
                "kind": "document", "id": name,
                "document": {"schema": TRAINING_DOCUMENT_SCHEMA_ID, "label": name, **document},
            }})
            assert reply.status_code == 201, reply.text
            return reply.json()["snapshot"]

        snapshot = create("training", {"payload": training})
        base_file = root / "documents/training.json"
        base_bytes = base_file.read_bytes()
        assert json.loads(base_bytes)["payload"]["settings"] == {}
        difference = {"/settings/optimizer/type": {"value": "adabelief"}}
        snapshot = create("training-2", {"base": {"document": "training"}, "differences": difference})
        assert snapshot["composed"]["training-2"]["settings"] == {"optimizer": {"type": "adabelief"}}
        pointed = client.patch(route, json={"revision": snapshot["revision"], "files": [{
            "address": {"kind": "step", "id": "train"}, "patches": [
                {"op": "replace", "path": f"/base/inline/{key}",
                 "value": {"document": "training-2", "pointer": f"/{key}"}} for key in TRAINING_DOCUMENT_KEYS
            ],
        }]})
        assert pointed.status_code == 200, pointed.text
        revision = pointed.json()["revision"]
        changed = folders.read_plan("fixture", revision=revision, source_runs=(original.key,)).plan
        assert not changed.diagnostics, changed.diagnostics
        (run,) = changed.runs
        assert run.request_id != original.request_id
        request = decode_request(changed.requests[run.request_id], changed.documents.__getitem__)
        assert request.spec.settings["optimizer"]["type"] == "adabelief"
        source = changed.sources[run.key]["/payload/settings/optimizer/type"]
        assert (source.kind, source.name, source.path) == ("document", "training-2", "/settings/optimizer/type")
        derived_file = root / "documents/training-2.json"
        assert json.loads(derived_file.read_bytes())["differences"] == difference

        reset = client.post(f"{route}/authoring/train/train", json={
            "revision": revision, "reset": "/settings/optimizer/type",
        })
        assert reset.status_code == 200, reset.text
        assert reset.json()["files"]
        saved = client.patch(route, json={"revision": revision, "files": reset.json()["files"]})
        assert saved.status_code == 200, saved.text
        assert not json.loads(derived_file.read_bytes()).get("differences")
        restored = folders.read_plan("fixture", revision=saved.json()["revision"]).plan
        assert not restored.diagnostics, restored.diagnostics
        assert restored.runs[0].request_id == original.request_id
        assert base_file.read_bytes() == base_bytes

        def patch(revision, files):
            reply = client.patch(route, json={"revision": revision, "files": files})
            assert reply.status_code == 200, reply.text
            return reply.json()["revision"]

        revision = saved.json()["revision"]
        learning_rate = client.post(f"{route}/authoring/train/train/fields", json={
            "revision": revision, "field": "/settings/optimizer/learning_rate/value",
            "edit": {"kind": "text", "text": "0.002"},
        })
        assert learning_rate.status_code == 200, learning_rate.text
        assert not learning_rate.json().get("problems") and learning_rate.json()["files"]
        revision = patch(revision, learning_rate.json()["files"])
        rated = folders.read_plan("fixture", revision=revision).plan
        assert not rated.diagnostics and rated.runs[0].request_id != original.request_id
        spec = decode_request(rated.requests[rated.runs[0].request_id], rated.documents.__getitem__).spec
        assert spec.settings["optimizer"]["type"] == "adam"
        assert spec.settings["optimizer"]["learning_rate"] == {"type": "constant_schedule", "value": 0.002}
        assert base_file.read_bytes() == base_bytes
        revision = patch(revision, [{
            "address": {"kind": "document", "id": "training-2"},
            "patches": [{"op": "remove", "path": "/differences"}],
        }])

        pin = {"/settings/optimizer/type": {"value": "adam"}}
        revision = patch(revision, [{
            "address": {"kind": "document", "id": "training-2"},
            "patches": [{"op": "add", "path": "/differences", "value": pin}],
        }])
        batch_count = training["training_config"]["n_batches"] + 1
        edited = client.post(f"{route}/authoring/train/train", json={
            "revision": revision, "change": {"field": "/training_config/n_batches", "value": batch_count},
        })
        assert edited.status_code == 200, edited.text
        revision = patch(revision, edited.json()["files"])
        batches = {"/training_config/n_batches": {"value": batch_count}}
        assert json.loads(derived_file.read_bytes())["differences"] == pin | batches
        prior = folders.read_plan("fixture", revision=revision).plan

        reset_pin = client.post(f"{route}/authoring/train/train", json={
            "revision": revision, "reset": "/settings/optimizer/type",
        })
        assert reset_pin.status_code == 200, reset_pin.text
        revision = patch(revision, reset_pin.json()["files"])
        assert json.loads(derived_file.read_bytes())["differences"] == batches
        assert folders.read_plan("fixture", revision=revision).plan.runs[0].request_id == prior.runs[0].request_id
