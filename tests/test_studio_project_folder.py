"""Studio edits the same durable authored files the text reader opens."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
import json
from os import utime
from pathlib import Path
from threading import Barrier, Event

from fastapi.testclient import TestClient
from pydantic import Field, JsonValue
import pytest

from feedbax.contracts.project import json_pointer_parts
from feedbax._atomic import write_bytes_atomic
from feedbax import project_folder
from feedbax.contracts import request_identity
from feedbax.planner import project_plan
from feedbax.contracts.jobs import StudioJobEventHistory
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import InlineBase, Project, StepBase, member_name
from feedbax.builtins.training import StandardTrainingSettings, supervised_gradient
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, load_text_project
from feedbax.planner.project_plan import RunKey, compile_project
from feedbax.contracts.studio_place import definition_place, path_parts
from feedbax.contracts.project_resolution import ResolutionCode
from feedbax.contracts.studio_authoring import StudioAuthoringChange
from feedbax.contracts.task_document import BLANK_TASK
from feedbax.host import HostError
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.plugins.declarations import DistributionSource, callable_declaration
from feedbax.training.methods import TrainingMethodDeclaration, TrainingMethodRegistry
from feedbax.studio.authoring import project_train_fields
from feedbax.studio.selection import StoredEdit, TrainSelection
from feedbax.contracts.matrix_core import apply_override_patches
from feedbax.web.api import jobs
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderInvalid
from tests.support import external_method_witness
from tests.support.managed_train_fixture import managed_train_project
from tests.support.hang import HANG_SECONDS


_URL = "/api/projects/reach"
_MODEL = "point-mass"
_SETTINGS = "settings"
_VALUE = "/payload/value"
_MASS_ADDRESS = ModelAddress.model_validate({
    "composites": [],
    "element": {"kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"]},
})
_SEED = "seed"


@pytest.fixture
def study(tmp_path: Path):
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    fit = raw["steps"]["fit"] = raw["steps"].pop("train")
    fit["base"]["inline"]["seed"] = 0
    fit["sweep"] = {
        "axes": {"seed": {"values": {
            "10": {"patches": {_SEED: {"value": 10}}},
            "2": {"patches": {_SEED: {"value": 2}}},
        }}},
        "exceptions": {"z_first": {"where": {"rows": []}}, "a_second": {"where": {"rows": []}}},
    }
    raw["documents"][_SETTINGS] = {"schema": "example.settings", "payload": {"value": 1}}
    project = Project.model_validate(raw)
    root = tmp_path / "reach"
    for name, data in dump_text_project(
        project, ProjectConfiguration(project="reach", store="results")
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root, TestClient(create_app(projects_dir=tmp_path))


def _read(client: TestClient):
    response = client.get(_URL)
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    return response.json()


def _patch(
    client: TestClient,
    revision: str,
    value: JsonValue,
    *,
    kind="document",
    name=_SETTINGS,
    path=_VALUE,
):
    return client.patch(
        _URL,
        json={
            "revision": revision,
            "address": {"kind": kind, "id": name},
            "patches": [{"op": "replace", "path": path, "value": value}],
        },
    )


def _edit_mass(client: TestClient, url: str, snapshot, value: float):
    response = client.post(f"{url}/model/edit", json={
        "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
        "document": _MODEL,
        "command": {"operation": {"op": "set_field", "value": value, "address": _MASS_ADDRESS.model_dump(mode="json")}},
    })
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["status"] == "applied", result
    return result["snapshot"]


def _settings(root: Path):
    return load_text_project(root).project.documents[_SETTINGS].payload["value"]


def _mass(root: Path):
    return (
        load_text_project(root)
        .project.documents[_MODEL]
        .payload["interior"]["components"]["plant"]["fields"]["mass"]
    )


def _folder_state(root: Path):
    return {
        path.relative_to(root): (
            path.stat().st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        )
        for path in (root, *root.rglob("*"))
    }


def _read_plan(client, root, snapshot, *source_runs):
    before = _folder_state(root)
    result = client.app.state.project_folders.read_plan(
        snapshot["project_id"], revision=snapshot["revision"], source_runs=source_runs
    )
    assert _folder_state(root) == before
    assert (result.project_id, result.revision, result.semantic_root) == (
        snapshot["project_id"],
        snapshot["revision"],
        snapshot["semantic_root"],
    )
    assert result.configuration.store == "results"
    return result.plan


def test_plan_reads_an_incomplete_named_train_at_each_settled_revision(tmp_path):
    app = create_app(projects_dir=tmp_path)
    client = TestClient(app)
    response = client.post("/api/projects")
    assert response.status_code == 201
    document = response.json()
    snapshot = document["snapshot"]
    root = tmp_path / document["configuration"]["project"]
    url = f"/api/projects/{snapshot['project_id']}"
    example = managed_train_project()
    step = example.steps["train"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    trial = step["base"]["inline"]["trial_documents"]
    model, task, objective = [
        {"kind": "document", "id": name, "document": example.documents[name].model_dump(
            mode="json", by_alias=True, exclude_unset=True,
        )}
        for name in (trial[slot]["document"] for slot in ("model", "task", "objective"))
    ]
    train = {"kind": "step", "id": "train", "step": step}
    for entry in (
        model, task, objective,
        {"kind": "step", "id": "train", "step": {"kind": "train", "base": {"inline": {}}}},
    ):
        response = client.post(
            f"{url}/files", json={"revision": snapshot["revision"], "file": entry}
        )
        assert response.status_code == 201, response.text
        snapshot = response.json()["snapshot"]
    response = client.patch(
        url,
        json={
            "revision": snapshot["revision"],
            "address": {"kind": "step", "id": "train"},
            "patches": [
                {
                    "op": "replace",
                    "path": "/base/inline",
                    "value": {"trial_documents": trial},
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    snapshot = response.json()
    selected = RunKey("default", "train", "default")
    mass = "/payload/trial_documents/model/interior/components/plant/fields/mass"
    before = _read_plan(client, root, snapshot, selected)
    assert before.variants == ("default",)
    assert [row.key for row in before.rows] == [selected]
    assert not before.runs and not before.requests
    assert {item.path for item in before.diagnostics} == {
        "payload.method",
        "payload.settings",
        "payload.training_config",
    }
    assert {(item.code, item.variant, item.step, item.row) for item in before.diagnostics} == {
        (ResolutionCode.INVALID_PAYLOAD, selected.variant, selected.step, selected.row)
    }
    assert set(before.selected_definitions) == {selected}
    definition = before.selected_definitions[selected]
    assert definition.payload["trial_documents"]["model"] == model["document"]["payload"]
    assert definition.payload["trial_documents"]["task"] == task["document"]["payload"]
    assert (
        definition.sources[mass].kind,
        definition.sources[mass].name,
        definition.sources[mass].path,
    ) == ("document", _MODEL, "/interior/components/plant/fields/mass")
    assert definition.sources["/payload/trial_documents/task/id"].name == task["id"]
    assert not _read_plan(client, root, snapshot).selected_definitions

    with client:
        current = _edit_mass(client, url, snapshot, 2.0)
    assert current["revision"] != snapshot["revision"]
    plan = _read_plan(client, root, current, selected)
    assert plan.rows == before.rows and plan.diagnostics == before.diagnostics
    assert plan.sources[selected][mass] == before.sources[selected][mass]
    assert plan.selected_definitions[selected].document_ids == definition.document_ids
    assert (
        plan.selected_definitions[selected].payload["trial_documents"]["model"]["interior"]["components"]["plant"]["fields"]["mass"]
        == 2.0
    )
    assert definition.payload["trial_documents"]["model"]["interior"]["components"]["plant"]["fields"]["mass"] == 1.0
    files = _folder_state(root)
    with pytest.raises(project_folder.ProjectRevisionConflict) as conflict:
        app.state.project_folders.read_plan(snapshot["project_id"], revision=snapshot["revision"])
    assert (conflict.value.resource, conflict.value.current) == ("project", current["revision"])
    assert _folder_state(root) == files
    assert not (root / "results").exists()


def test_train_authoring_keeps_composed_steps_and_incompatible_documents_intact(study):
    root, client = study
    with client:
        snapshot = _read(client)
        original = _folder_state(root)
        composed = client.post(
            f"{_URL}/authoring/train/fit", json={"revision": snapshot["revision"]},
        )
        assert composed.status_code == 200, composed.text
        assert _folder_state(root) == original
        response = client.post(
            f"{_URL}/files",
            json={
                "revision": snapshot["revision"],
                "file": {
                    "kind": "step", "id": "train",
                    "step": {"kind": "train", "base": {"inline": {}}},
                },
            },
        )
        assert response.status_code == 201, response.text
        revision = response.json()["snapshot"]["revision"]
        original = _folder_state(root)
        url = f"{_URL}/authoring/train/train"
        projection = client.post(url, json={"revision": revision})
        assert projection.status_code == 200, projection.text
        assert projection.headers["Cache-Control"] == "no-store"
        fields = {field["id"]: field for field in projection.json()["fields"]}
        assert "feedbax.supervised_gradient" in {
            option["value"] for option in fields["/method"]["control"]["options"]
        }
        assert "/graph" not in fields and "/task" not in fields
        invalid = client.post(url, json={
            "revision": revision, "change": {"field": "/task", "value": "model"},
        })
        assert invalid.status_code == 422
        assert _folder_state(root) == original


_WHOLE_STEP = TrainSelection((), None, None, False, StoredEdit)
"""The whole step, storing nothing: an edit is its writes in the step's definition."""


def _written(project, writes):
    """The Train step's definition after ``writes``."""
    return apply_override_patches(project.steps["train"].base.inline, writes)


def _train_fields(project, registry, change=None):
    projection = project_train_fields(project, "train", registry, change, _WHOLE_STEP)
    return projection.fields, projection.settings, projection.stored.files, projection.documents


def test_an_invalid_training_setting_keeps_every_training_field_in_place():
    registry = new_application_registry_bundle().training_methods
    raw = managed_train_project(batches=25).model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["steps"]["train"]["base"]["inline"]["training_config"]["batch_size"] = -3
    fields, _, _, _ = _train_fields(Project.model_validate(raw), registry)
    values = {field.id: field.value for field in fields}
    assert values["/training_config/batch_size"] == -3
    assert values["/training_config/n_batches"] == 25


def test_initial_training_settings_retain_the_authored_method_rate_and_batch_count():
    registry = new_application_registry_bundle().training_methods
    project = managed_train_project(batches=25)
    _, _, patches, _ = _train_fields(
        project, registry,
        StudioAuthoringChange(field="/training_config/snapshot_interval", value=12),
    )
    patch, = patches
    assert patch.path == "training_config.snapshot_interval"
    assert patch.value == 12

    fields, _, patches, _ = _train_fields(project, registry)
    values = {field.id: field.value for field in fields}
    assert values["/training_config/n_batches"] == 25
    assert values["/settings/optimizer/learning_rate/value"] == (
        project.steps["train"].base.inline["settings"]["optimizer"]["learning_rate"]["value"]
    )
    assert (values["/settings/grad_clip"], values["/settings/optimizer/b1"]) == (1.0, 0.9)

    raw = project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["steps"]["train"]["base"]["inline"]["settings"] = {}
    _, _, (patch,), _ = _train_fields(Project.model_validate(raw), registry, StudioAuthoringChange(field="/settings/optimizer/b1", value=0.8),
    )
    assert patch.path == "settings.optimizer"
    assert StandardTrainingSettings.model_validate({"optimizer": patch.value}).optimizer.b1 == 0.8

    _, _, patches, _ = _train_fields(
        project, registry,
        StudioAuthoringChange(field="/seed", value=7.0),
    )
    seed, = patches
    assert seed.path == "seed"
    assert type(seed.value) is int and seed.value == 7


def test_a_swept_train_with_an_exception_projects_and_writes_its_settings():
    registry = new_application_registry_bundle().training_methods
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["steps"]["train"]["sweep"] = {
        "axes": {"seed": {"values": {"a": {"patches": {"seed": {"value": 1}}}, "b": {"patches": {"seed": {"value": 2}}}}}},
        "exceptions": {"wide": {
            "where": {"rule": "seed=b"},
            "patches": {"training_config.batch_size": {"value": 20}},
        }},
    }
    project = Project.model_validate(raw)
    fields, settings, _, _ = _train_fields(project, registry)
    assert {field.id for field in fields} >= {"/method", "/training_config/batch_size"}
    assert settings["optimizer"]["type"] == "adam"

    for field in ("/settings/optimizer", "/settings/optimizer/type"):
        _, _, patches, _ = _train_fields(project, registry, StudioAuthoringChange(field=field, value="sgd"))
        assert _written(project, patches)["settings"]["optimizer"]["type"] == "sgd"
        with pytest.raises(ValueError):
            _train_fields(project, registry, StudioAuthoringChange(field=field, value=3))


def test_a_class_change_is_written_even_when_it_lacks_a_swept_parameter():
    registry = new_application_registry_bundle().training_methods
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["steps"]["train"]["base"]["inline"]["settings"]["optimizer"]["learning_rate"] = {
        "type": "warmup_cosine_decay_schedule", "peak_value": 0.001,
    }
    peak = "settings.optimizer.learning_rate.peak_value"
    raw["steps"]["train"]["sweep"] = {"axes": {"peak": {"values": {
        "low": {"patches": {peak: {"value": 0.001}}}, "high": {"patches": {peak: {"value": 0.002}}},
    }}}}
    project = Project.model_validate(raw)

    def choose(schedule: str):
        _, _, patches, _ = _train_fields(project, registry, StudioAuthoringChange(
            field="/settings/optimizer/learning_rate", value=schedule,
        ))
        return patches

    changed = _written(project, choose("constant_schedule"))
    assert changed["settings"]["optimizer"]["learning_rate"]["type"] == "constant_schedule"
    assert _written(project, choose("warmup_constant_schedule"))["settings"]["optimizer"]["learning_rate"] == {
        "type": "warmup_constant_schedule", "init_value": 0.001, "peak_value": 0.001, "warmup_steps": 100,
    }


class _PenalizedSettings(StandardTrainingSettings):
    penalty: float = Field(default=0.1, gt=0)


def _penalized(state, ctx):
    return supervised_gradient(state, ctx)


def _three_methods() -> TrainingMethodRegistry:
    """The standard method, one that shares its optimizer, and one whose settings have no optimizer."""
    registry = TrainingMethodRegistry()
    source = DistributionSource("probe", "probe", "0")
    registry.register_declaration(
        supervised_gradient, callable_declaration(supervised_gradient), key="feedbax.supervised_gradient", source=source,
    )
    registry.register_declaration(
        _penalized, TrainingMethodDeclaration(settings_model=_PenalizedSettings, init=lambda *_: None, label="Penalized"),
        key="probe.penalized", source=source,
    )
    registry.register_declaration(
        external_method_witness.adversarial, callable_declaration(external_method_witness.adversarial),
        key="probe.adversarial", source=source,
    )
    return registry


def test_a_method_change_keeps_shared_settings_and_stores_a_method_without_a_swept_parameter():
    registry = _three_methods()
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    settings = raw["steps"]["train"]["base"]["inline"]["settings"]
    settings["optimizer"]["learning_rate"]["value"] = 0.003
    settings["grad_clip"] = 2.0

    def choose(project: Project, method: str):
        return _train_fields(project, registry, StudioAuthoringChange(field="/method", value=method))[2]

    written = _written(project := Project.model_validate(raw), choose(project, "probe.penalized"))
    assert written["method"] == "probe.penalized"
    assert written["settings"]["optimizer"] == settings["optimizer"]
    assert (written["settings"]["grad_clip"], written["settings"]["penalty"]) == (2.0, 0.1)

    rate = "settings.optimizer.learning_rate.value"
    raw["steps"]["train"]["sweep"] = {"axes": {"rate": {"values": {
        "slow": {"patches": {rate: {"value": 0.001}}}, "fast": {"patches": {rate: {"value": 0.01}}},
    }}}}
    swept = Project.model_validate(raw)
    fields, _, _, _ = _train_fields(swept, registry)
    assert next(field for field in fields if field.id == "/method").control is not None
    assert _written(swept, choose(swept, "probe.penalized"))["settings"]["optimizer"] == settings["optimizer"]
    changed = _written(swept, choose(swept, "probe.adversarial"))
    assert changed["method"] == "probe.adversarial" and "optimizer" not in changed["settings"]


def test_each_required_setting_of_a_method_saves_alone_and_the_plan_names_the_ones_still_missing():
    registry = _three_methods()
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)

    def edit(field, value):
        """The settings the plan names as problems after ``field`` is set to ``value``."""
        project = Project.model_validate(raw)
        writes = _train_fields(project, registry, StudioAuthoringChange(field=field, value=value))[2]
        raw["steps"]["train"]["base"]["inline"] = _written(project, writes)
        return {
            definition_place(path_parts(problem.path), lambda _slot: None).setting
            for problem in compile_project(Project.model_validate(raw), training_methods=registry).diagnostics
            if problem.step == "train"
        }

    assert edit("/method", "probe.adversarial") == {"learning_rate", "limit"}
    assert edit("/settings/limit", 0.5) == {"learning_rate"}
    with pytest.raises(ValueError, match="learning_rate: Input should be greater than 0"):
        edit("/settings/learning_rate", -1.0)
    assert edit("/settings/learning_rate", 0.01) == set()
    assert raw["steps"]["train"]["base"]["inline"]["settings"] == {"limit": 0.5, "learning_rate": 0.01}


def test_choosing_another_optimizer_keeps_the_learning_rate():
    registry = new_application_registry_bundle().training_methods
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["steps"]["train"]["base"]["inline"]["settings"]["optimizer"]["learning_rate"]["value"] = 0.003
    project = Project.model_validate(raw)
    _, _, patches, _ = _train_fields(project, registry, StudioAuthoringChange(field="/settings/optimizer", value="adamw"))
    optimizer = _written(project, patches)["settings"]["optimizer"]
    assert optimizer["type"] == "adamw"
    assert optimizer["learning_rate"] == {"type": "constant_schedule", "value": 0.003}


def test_checkpoint_carryover_uses_only_the_exact_paired_predecessors(tmp_path):
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    train = raw["steps"]["train"]
    carry_path = "checkpoint_initialization"
    rate_path = "settings.optimizer.learning_rate.value"
    train["sweep"] = {"axes": {"rate": {"values": {
        "slow": {"patches": {rate_path: {"value": 0.001}}},
        "fast": {"patches": {rate_path: {"value": 0.003}}},
        "long": {"patches": {"training_config.n_batches": {"value": 3}}},
    }}}}
    child = {
        "kind": "train",
        "base": {"step": "train"},
        "inputs": {"checkpoint": {"step": "train", "checkpoint": "last", "where": {"rule": "rate=slow,fast"}}},
        "deltas": [{"layer_id": "initialize", "patches": [
            {"path": carry_path, "op": "add", "value": "initialize_from"},
        ]}],
    }
    raw["steps"]["successor"] = child
    project = Project.model_validate(raw)
    root = tmp_path / "reach"
    for relative, data in dump_text_project(
        project, ProjectConfiguration(project="reach", store="results"),
    ).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    address = {"kind": "step", "id": "successor"}
    field_id = "/" + carry_path.replace(".", "/")
    url = f"{_URL}/authoring/train/successor"

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        before = _read(client)
        initial = _read_plan(client, root, before)
        assert not initial.diagnostics
        assert all(run.admission_problem is None for run in initial.runs)
        change = {"field": field_id, "value": "continue_from"}
        response = client.post(url, json={"revision": before["revision"], "change": change})
        assert response.status_code == 200, response.text
        reviewed = response.json()
        control = next(item for item in reviewed["fields"] if item["id"] == field_id)
        assert control["value"] == "initialize_from"
        [file] = reviewed["files"]
        [deltas] = file["patches"]
        assert deltas["path"] == "/deltas" and len(deltas["value"]) == 2
        policy, = deltas["value"][-1]["patches"]
        assert policy["path"] == carry_path
        assert policy["value"] == {
            "mode": "continue_from", "source_completed_updates": 2,
            "state": "preserve", "schedule": "continue", "rng": "carry",
        }
        saved = client.patch(_URL, json={"revision": before["revision"], "files": reviewed["files"]})
        assert saved.status_code == 200, saved.text
        after = saved.json()
        assert after["project"]["steps"]["train"] == before["project"]["steps"]["train"]
        stale = client.post(url, json={"revision": before["revision"], "change": change})
        assert stale.status_code == 409, stale.text
        final = _read_plan(client, root, after)
        assert not final.diagnostics
        prefix = {run.key: run.request_id for run in initial.runs if run.key.step == "train"}
        assert {run.key: run.request_id for run in final.runs if run.key.step == "train"} == prefix
        successors = {run.key: run.inputs["checkpoint"].producer for run in final.runs if run.key.step == "successor"}
        assert successors == {
            RunKey("default", "successor", f"rate={rate}"): RunKey("default", "train", f"rate={rate}")
            for rate in ("slow", "fast")
        }
        restored = client.post(url, json={
            "revision": after["revision"], "change": {**change, "value": "initialize_from"},
        })
        assert restored.status_code == 200, restored.text
        assert [patch["value"] for patch in restored.json()["files"][0]["patches"][0]["value"][-1]["patches"]] == [
            "initialize_from",
        ]

        widened = client.patch(_URL, json={
            "revision": after["revision"], "address": address,
            "patches": [{"op": "remove", "path": "/inputs/checkpoint/where"}],
        })
        assert widened.status_code == 200, widened.text
        refused = client.post(url, json={
            "revision": widened.json()["revision"], "change": {**change, "value": "initialize_from"},
        })
        assert refused.status_code == 422, refused.text
        assert "is read-only" in refused.text


def test_variant_model_authoring_preserves_sweeps_and_request_scope(study):
    root, client = study
    original = load_text_project(root)
    authored = original.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    authored["steps"]["other"] = authored["steps"]["fit"]

    existing = "steps.fit.base.inline.training_config.n_batches"
    authored["sweep"] = {
        "rows": {
            "base": {},
            "parent": {"from": "base"},
            "compare": {"from": "parent", "label": "Comparison", "patches": {existing: {
                "value": authored["steps"]["fit"]["base"]["inline"]["training_config"]["n_batches"],
            }}},
            "control": {"from": "base"},
        },
        "exceptions": {"keep": {"where": {"rows": []}}},
    }
    project = Project.model_validate(authored)
    for relative, data in dump_text_project(project, original.configuration).items():
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)

    before = _read(client)
    owner = {"composites": [], "element": {"kind": "component", "id": "plant"}}
    field = ModelAddress.model_validate(owner | {"element": {
        "kind": "field", "parent": owner["element"], "path": ["fields", "mass"],
    }})
    key = {"variant": "compare", "step": "fit", "row": "seed=10"}
    body = {"revision": before["revision"], "run": key, "document": _MODEL, "owner": owner}
    change = {
        "address": field.model_dump(mode="json"), "edit": {"kind": "text", "text": "3"},
        "chosen_steps": ["fit", "other"],
    }

    def review(**updates):
        return client.post(f"{_URL}/model/fields", json={**body, **updates})

    with client:
        inspection = review()
        assert inspection.status_code == 200, inspection.text
        mass = next(item for item in inspection.json()["fields"] if item["label"] == "Mass")
        assert ModelAddress.model_validate(mass["address"]) == field
        narrowed = review(**(change | {"chosen_steps": ["fit"]}))
        assert narrowed.status_code == 200, narrowed.text
        assert narrowed.json()["files"] == []
        choices = {item["kind"]: item for item in narrowed.json()["store_choices"]}
        assert set(choices["variant"]["steps"]) == {"fit", "other"}
        assert choices["selected_readers"]["steps"] == ["fit"]
        parent = review(**(change | {"run": {**key, "variant": "parent"}}))
        assert parent.status_code == 200, parent.text
        assert parent.json()["files"] == []
        assert {run["variant"] for run in parent.json()["store_choices"][0]["runs"]} == {"parent", "compare"}
        reviewed = review(**change)
        assert reviewed.status_code == 200, reviewed.text
        recipe, = reviewed.json()["files"]
        assert recipe["address"] == {"kind": "variant", "id": "compare"}
        target = f"documents.{_MODEL}.{field.to_text()}"
        assert recipe["patches"][0]["value"][target] == {"op": "replace", "value": 3}
        snapshot = client.patch(_URL, json={"revision": before["revision"], **recipe})
        assert snapshot.status_code == 200, snapshot.text
        after = snapshot.json()
        assert after["project"]["documents"] == before["project"]["documents"]
        assert after["project"]["steps"] == before["project"]["steps"]
        sweep = before["project"]["sweep"]
        sweep["rows"]["compare"]["patches"][target] = {"op": "replace", "value": 3}
        assert after["project"]["sweep"] == sweep
        stale = review(**change)
        assert stale.status_code == 409, stale.text
        again = review(revision=after["revision"], **change)
        assert again.status_code == 200, again.text
        inspected = review(revision=after["revision"])
        assert inspected.status_code == 200, inspected.text
        mass = next(item for item in inspected.json()["fields"] if item["label"] == "Mass")
        assert mass["text"] == "3"

        old_plan = compile_project(project)
        new_plan = _read_plan(client, root, after)
        old_requests = {run.key: run.request_id for run in old_plan.runs}
        new_requests = {run.key: run.request_id for run in new_plan.runs}
        assert new_requests.keys() == old_requests.keys()
        assert len(new_requests) == 16
        assert {key for key in new_requests if new_requests[key] != old_requests[key]} == {
            RunKey("compare", step, row) for step in ("fit", "other") for row in ("seed=10", "seed=2")
        }


def test_plan_preserves_full_variant_lineage_and_named_subset_after_edit(study):
    root, client = study
    original = load_text_project(root)
    authored = original.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    authored["documents"]["selected"] = {"schema": "subset", "payload": {"rows": [
        {"key": "selected-seed", "variant": "compare", "row": "seed=10"},
    ]}}
    authored["steps"]["follow"] = {
        "kind": "train",
        "base": {"step": "fit"},
        "sweep": {"axes": {"seed": None, "checkpoint": {"input": "checkpoint"}}},
        "inputs": {"checkpoint": {"step": "fit", "where": {"subset": "selected"}}},
        "deltas": [{"layer_id": "initialize", "patches": [
            {"path": "checkpoint_initialization", "op": "add", "value": "initialize_from"},
        ]}],
    }
    authored["sweep"] = {"rows": {"base": {}, "compare": {}}}
    project = Project.model_validate(authored)
    for relative, data in dump_text_project(project, original.configuration).items():
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)

    snapshot = _read(client)
    [selected] = [
        run.key for run in _read_plan(client, root, snapshot).runs
        if run.key.variant == "compare" and run.key.step == "follow"
    ]
    before = _read_plan(client, root, snapshot, selected)
    assert before.variants == ("base", "compare")
    assert len(before.runs) == len(before.rows) == 6
    assert not before.diagnostics
    assert len(before.requests) == 3
    for variant in before.variants:
        [follower] = [run for run in before.runs if run.key == RunKey(variant, "follow", selected.row)]
        producer = RunKey("compare", "fit", "seed=10")
        assert follower.inputs["checkpoint"].producer == producer
        assert follower.inputs["checkpoint"].member.key == "selected-seed"
        subset = before.subsets[variant, "follow", "checkpoint"]
        assert (subset.document, subset.candidates, subset.members) == ("selected", 2, (producer,))
        assert [(row.key, row.variant, row.row) for row in subset.terms.rows] == [
            ("selected-seed", "compare", "seed=10"),
        ]
        assert before.steps[variant, "follow"].inputs["checkpoint"].step == "fit"
    assert set(before.sources) == {selected}
    source = before.sources[selected]["/inputs/checkpoint"]
    assert (source.kind, source.variant, source.step, source.row) == (
        "upstream",
        "compare",
        "fit",
        "seed=10",
    )

    edited = _patch(
        client,
        snapshot["revision"],
        7,
        kind="step",
        name="fit",
        path=f"/sweep/axes/seed/values/10/patches/{_SEED}/value",
    )
    assert edited.status_code == 200, edited.text
    after = _read_plan(client, root, edited.json(), selected)
    before_requests = {run.key: run.request_id for run in before.runs}
    assert set(before_requests) == {run.key for run in after.runs}
    assert {run.key for run in after.runs if run.request_id != before_requests[run.key]} == {
        RunKey(variant, step, row)
        for variant in after.variants
        for step, row in (("fit", "seed=10"), ("follow", selected.row))
    }
    assert after.subsets == before.subsets and after.sources == before.sources

    broken = _patch(
        client,
        edited.json()["revision"],
        "missing",
        kind="step",
        name="follow",
        path="/inputs/checkpoint/step",
    )
    assert broken.status_code == 200, broken.text
    invalid = _read_plan(client, root, broken.json(), selected)
    assert {row.key.step for row in invalid.rows} == {"fit"}
    assert not invalid.sources
    assert {(item.variant, item.step, item.row, item.path) for item in invalid.diagnostics} == {
        (variant, "follow", None, "inputs.checkpoint") for variant in after.variants
    }
    assert invalid.steps["compare", "follow"].authored_inputs["checkpoint"].step == "missing"


def test_in_flight_plan_keeps_the_revision_it_loaded(study, monkeypatch):
    root, client = study
    opened = _read(client)
    service = client.app.state.project_folders
    selected = RunKey("default", "fit", "seed=10")
    entered = Event()
    release = Event()

    def pause_old_compile(project, **kwargs):
        if project.documents[_MODEL].payload["interior"]["components"]["plant"]["fields"]["mass"] == 1:
            entered.set()
            assert release.wait(HANG_SECONDS)
        return compile_project(project, **kwargs)

    monkeypatch.setattr(project_folder, "compile_project", pause_old_compile)
    with ThreadPoolExecutor(max_workers=1) as pool:
        previous = pool.submit(
            service.read_plan, "reach", revision=opened["revision"], source_runs=(selected,)
        )
        try:
            assert entered.wait(HANG_SECONDS)
            with client:
                changed = _edit_mass(client, _URL, opened, 4)
            current = _read_plan(client, root, changed, selected)
        finally:
            release.set()
        old = previous.result(timeout=HANG_SECONDS)

    assert (old.revision, old.semantic_root) == (opened["revision"], opened["semantic_root"])
    assert (
        old.plan.selected_definitions[selected].payload["trial_documents"]["model"]["interior"]["components"]["plant"]["fields"]["mass"]
        == 1
    )
    assert (
        current.selected_definitions[selected].payload["trial_documents"]["model"]["interior"]["components"]["plant"]["fields"]["mass"]
        == 4
    )
    assert _read_plan(client, root, changed, selected) == current


def _plan_dependencies(root: Path) -> None:
    authored = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    authored["steps"]["fit"] = authored["steps"].pop("train")
    fit = authored["steps"]["fit"]
    seed_path = "seed"
    fit["base"]["inline"]["seed"] = 0
    fit["sweep"] = {"axes": {"seed": {"values": {
        "10": {"label": "First", "patches": {seed_path: {"value": 10}}},
        "2": {"patches": {seed_path: {"value": 2}}},
    }}}}
    authored["documents"]["selected"] = {"schema": "subset", "payload": {"rows": [
        {"key": "seed=10", "variant": "default", "row": "seed=10"},
    ]}}
    authored["steps"]["follow"] = {
        "kind": "train",
        "base": {"step": "fit"},
        "sweep": {"axes": {"seed": None, "checkpoint": {"input": "checkpoint"}}},
        "inputs": {"checkpoint": {"step": "fit", "where": {"subset": "selected"}}},
        "deltas": [{"layer_id": "initialize", "patches": [
            {"path": "checkpoint_initialization", "op": "add", "value": "initialize_from"},
        ]}],
    }
    independent = deepcopy(fit["base"]["inline"])
    independent["seed"] = 88
    authored["steps"]["independent"] = {"kind": "train", "base": {"inline": independent}}
    project = Project.model_validate(authored)
    for relative, data in dump_text_project(
        project, ProjectConfiguration(project="reach", store="results"),
    ).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_incremental_plan_matches_fresh_compilation_and_skips_unchanged_work(tmp_path, monkeypatch, request):
    root = tmp_path / "reach"
    _plan_dependencies(root)
    lifespan = ExitStack()
    request.addfinalizer(lifespan.close)
    client = lifespan.enter_context(TestClient(create_app(projects_dir=tmp_path)))
    work = Counter()

    def count(owner, name):
        original = getattr(owner, name)

        @wraps(original)
        def counted(*args, **kwargs):
            work[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)

    count(project_plan, "compose_step")
    count(project_plan.StepTemplate, "resolve")
    count(request_identity, "encode_request")

    first = RunKey("default", "fit", "seed=10")
    second = RunKey("default", "fit", "seed=2")
    independent = RunKey("default", "independent", "default")
    snapshot = _read(client)

    def check(*keys):
        work.clear()
        plan = _read_plan(client, root, snapshot, *keys)
        observed = work.copy()
        fresh = compile_project(load_text_project(root).project, source_runs=keys)
        assert plan == fresh
        return plan, observed

    def patch(value, *, name="fit", path):
        response = _patch(client, snapshot["revision"], value, kind="step", name=name, path=path)
        assert response.status_code == 200, response.text
        return response.json()

    cold, counts = check(first)
    assert not cold.diagnostics
    assert all(run.admission_problem is None for run in cold.runs)
    assert counts["compose_step"] == 3 and counts["encode_request"] == 4
    [follow] = [run.key for run in cold.runs if run.key.step == "follow"]
    assert counts["resolve"] >= 4
    warm, counts = check(first)
    assert not counts
    warm.selected_definitions[first].payload["trial_documents"]["model"]["step_seconds"] = 999
    assert check(first)[0] == cold

    selected, counts = check(second)
    assert counts == {"resolve": 1}
    assert selected.requests == cold.requests and set(selected.sources) == {second}

    prefix = "/sweep/axes/seed/values/10"
    snapshot = patch("Renamed", path=f"{prefix}/label")
    relabelled, counts = check(first)
    assert counts["encode_request"] == 0
    assert relabelled.requests == cold.requests
    old_row = next(run for run in cold.runs if run.key == first)
    renamed_row = next(run for run in relabelled.runs if run.key == first)
    assert renamed_row.coordinates["seed"].label == "Renamed"
    assert renamed_row.label != old_row.label

    snapshot = patch(7, path=f"{prefix}/patches/seed/value")
    edited, counts = check(first)
    assert counts["encode_request"] == 2 and counts["compose_step"] == 2
    before_ids = {run.key: run.request_id for run in cold.runs}
    assert {run.key for run in edited.runs if run.request_id != before_ids[run.key]} == {first, follow}

    model = load_text_project(root).project.documents["point-mass"].payload
    snapshot = patch(model, path="/base/inline/trial_documents/model")
    inlined, counts = check(first)
    assert counts["encode_request"] == 0 and inlined.requests == edited.requests
    source_path = "/payload/trial_documents/model/step_seconds"
    assert inlined.sources[first][source_path].kind == "step"
    assert edited.sources[first][source_path].kind == "document"

    response = client.post(f"{_URL}/model/edit", json={
        "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
        "document": "point-mass",
        "command": {"operation": {
            "op": "set_field", "value": 2,
            "address": {"composites": [], "element": {
                "kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"],
            }},
        }},
    })
    assert response.status_code == 200, response.text
    assert response.json()["result"]["status"] == "applied", response.text
    snapshot = response.json()["result"]["snapshot"]
    document_edit, counts = check(first)
    assert counts["encode_request"] == 1 and counts["resolve"] == 1
    assert {
        run.key for run, old in zip(document_edit.runs, inlined.runs, strict=True)
        if run.request_id != old.request_id
    } == {independent}

    fit = load_text_project(root).project.steps["fit"].model_dump(mode="json", exclude_unset=True)
    values = fit["sweep"]["axes"]["seed"]["values"]
    snapshot = patch(
        {"2": values["2"], "repeat": values["2"], "10": values["10"]},
        path="/sweep/axes/seed/values",
    )
    reordered, counts = check(first)
    assert counts["encode_request"] == 1 and reordered.requests == document_edit.requests
    assert [run.key.row for run in reordered.runs if run.key.step == "fit"] == [
        "seed=2", "seed=repeat", "seed=10",
    ]
    assert reordered.subsets["default", "follow", "checkpoint"].candidates == 3

    snapshot = patch("follow", name="follow", path="/base/step")
    cyclic, _ = check(first)
    assert any(problem.code is ResolutionCode.STEP_CYCLE for problem in cyclic.diagnostics)
    snapshot = patch("fit", name="follow", path="/base/step")
    assert check(first)[0] == reordered

    count(project_plan, "source_shape")
    updated_authority, counts = check(first)
    assert counts["encode_request"] == 5 and counts["compose_step"] == 3
    assert updated_authority == reordered


def test_new_projects_are_durable_openable_and_distinct(tmp_path):
    projects = tmp_path / "projects"
    app = create_app(projects_dir=projects)

    def create():
        client = TestClient(app)
        try:
            response = client.post("/api/projects")
            assert response.status_code == 201, response.text
            assert response.headers["Cache-Control"] == "no-store"
            document = response.json()
            project_id = document["snapshot"]["project_id"]
            assert response.headers["Location"] == f"/api/projects/{project_id}/open"
            return document
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        creates = [pool.submit(create) for _ in range(2)]
        documents = [creation.result() for creation in creates]

    ids = {document["snapshot"]["project_id"] for document in documents}
    assert len(ids) == 2
    names = {document["configuration"]["project"] for document in documents}
    assert len(names) == 2
    assert {entry.name for entry in projects.iterdir()} == names
    reopened = TestClient(create_app(projects_dir=projects))
    for document in documents:
        snapshot = document["snapshot"]
        project_id = snapshot["project_id"]
        name = document["configuration"]["project"]
        folder = projects / name
        text = load_text_project(folder)
        assert text.project == Project()
        assert text.configuration.project == name
        assert text.store_path == folder / "results"
        assert text.revision == snapshot["revision"]
        assert json.loads((folder / "presentation.json").read_bytes()) == {}
        response = reopened.get(f"/api/projects/{project_id}/open")
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert response.json() == document
        assert reopened.get(f"/api/projects/{project_id}").json() == snapshot


def test_list_projects_reads_authored_folders_without_opening_or_creating_them(tmp_path):
    projects = tmp_path / "projects"
    client = TestClient(create_app(projects_dir=projects))
    assert client.get("/api/projects").json() == {"projects": []}
    assert not projects.exists()

    folder = projects / "original-folder-name"
    folder.mkdir(parents=True)
    for name, content in dump_text_project(
        Project(), ProjectConfiguration(project="current-project-name", store="results")
    ).items():
        declaration = folder / name
        declaration.write_bytes(content)
        utime(declaration, (1_000_000_000, 1_000_000_000))
    results = folder / "results"
    results.mkdir()
    (results / "finished").touch()
    (projects / "alias").symlink_to(folder, target_is_directory=True)
    (projects / "empty").mkdir()
    invalid = projects / "invalid"
    (invalid / "steps").mkdir(parents=True)
    (invalid / "invalid.feedbax").write_bytes(declaration.read_bytes())
    (invalid / "steps/step.json").write_text("invalid JSON")
    before = set(projects.rglob("*"))

    response = client.get("/api/projects")
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    listed, = response.json()["projects"]
    assert listed["name"] == "current-project-name"
    assert datetime.fromisoformat(listed["modified"]) == datetime(2001, 9, 9, 1, 46, 40, tzinfo=timezone.utc)
    assert set(projects.rglob("*")) == before

    url = f"/api/projects/{listed['project_id']}"
    opened = client.get(f"{url}/open")
    assert opened.status_code == 200, opened.text
    response = client.post(f"{url}/files", json={
        "revision": opened.json()["snapshot"]["revision"],
        "file": {"kind": "step", "id": "train", "step": {"kind": "train", "base": {"inline": {}}}},
    })
    assert response.status_code == 201, response.text
    edited, = client.get("/api/projects").json()["projects"]
    assert edited == {**listed, "modified": edited["modified"]}
    assert datetime.fromisoformat(edited["modified"]) > datetime.fromisoformat(listed["modified"])


def test_rename_changes_only_the_declaration_and_checks_authored_revision(study):
    root, client = study
    initial = client.get(f"{_URL}/open").json()
    presentation = {"documents": {_MODEL: {"camera": {"x": 1, "y": 2, "zoom": 1.5}}}}
    response = client.put(f"{_URL}/presentation", json={
        "revision": initial["snapshot"]["presentation_revision"],
        "presentation": presentation,
    })
    assert response.status_code == 200, response.text
    saved = response.json()
    store = root / "results"
    store.mkdir()
    (store / "retained").write_bytes(b"existing store contents")
    before = load_text_project(root)
    plan = compile_project(before.project)
    assert not plan.diagnostics
    declaration = root / before.declaration
    original_declaration = declaration.read_bytes()
    untouched = {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path != declaration
    }

    response = client.post(f"{_URL}/rename", json={"revision": saved["revision"], "name": "renamed"})
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    renamed = response.json()
    assert renamed["configuration"] == {**initial["configuration"], "project": "renamed"}
    current = renamed["snapshot"]
    assert current["revision"] != saved["revision"]
    assert current == {**saved, "revision": current["revision"]}
    assert not declaration.exists()
    assert (root / "renamed.feedbax").read_bytes() == original_declaration
    assert "project" not in json.loads(original_declaration)
    assert all(path.read_bytes() == data for path, data in untouched.items())
    after = load_text_project(root)
    assert after.store_path == before.store_path
    assert compile_project(after.project) == plan

    stale = client.post(f"{_URL}/rename", json={"revision": saved["revision"], "name": "stale"})
    assert stale.status_code == 409
    assert stale.json()["detail"]["current"] == current["revision"]
    assert _patch(client, saved["revision"], 9).status_code == 409
    assert client.post(
        f"{_URL}/rename", json={"revision": current["revision"], "name": "renamed"}
    ).json() == renamed
    reopened = TestClient(create_app(projects_dir=root.parent))
    assert reopened.get(f"{_URL}/open").json() == renamed
    assert reopened.get("/api/projects/renamed/open").status_code == 404


def test_open_uses_managed_folder_ids_and_refuses_directory_symlinks(tmp_path):
    projects = tmp_path / "projects"
    projects.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    for name, content in dump_text_project(
        Project(), ProjectConfiguration(project="external", store="results")
    ).items():
        (external / name).write_bytes(content)
    (projects / "alias").symlink_to(external, target_is_directory=True)
    client = TestClient(create_app(projects_dir=projects))
    assert client.get("/api/projects/alias/open").status_code == 422
    assert client.get("/api/projects/external/open").status_code == 404
    assert client.get("/api/projects/external.json/open").status_code == 422


def test_create_named_file_uses_project_cas_and_never_replaces_an_entry(tmp_path):
    projects = tmp_path / "projects"
    client = TestClient(create_app(projects_dir=projects))
    document = client.post("/api/projects").json()
    created = document["snapshot"]
    project_id = created["project_id"]
    url = f"/api/projects/{project_id}/files"
    train = {"kind": "step", "id": "train", "step": {"kind": "train", "base": {"inline": {}}}}
    response = client.post(url, json={"revision": created["revision"], "file": train})
    assert response.status_code == 201, response.text
    step_revision = response.json()["snapshot"]
    root = projects / document["configuration"]["project"]
    step_file = root / "steps/train.json"
    assert json.loads(step_file.read_bytes()) == train["step"]

    task = {"kind": "document", "id": "task", "document": {
        "schema": "feedbax.spec.task_document", "payload": BLANK_TASK.model_dump(mode="json"),
    }}
    rejected = client.post(url, json={"revision": created["revision"], "file": task})
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["current"] == step_revision["revision"]
    assert not (root / "documents/task.json").exists()

    response = client.post(url, json={"revision": step_revision["revision"], "file": task})
    assert response.status_code == 201, response.text
    document = response.json()
    assert document["snapshot"]["presentation_revision"] == created["presentation_revision"]
    duplicate = client.post(url, json={"revision": document["snapshot"]["revision"], "file": train})
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "authored_file_exists"
    assert json.loads(step_file.read_bytes()) == train["step"]
    duplicate_batch = client.post(url, json={
        "revision": document["snapshot"]["revision"], "files": [task | {"id": "new-task"}, train],
    })
    assert duplicate_batch.status_code == 409
    assert not (root / "documents/new-task.json").exists()
    assert TestClient(create_app(projects_dir=projects)).get(f"/api/projects/{project_id}/open").json() == document


def test_rename_and_patch_from_one_revision_have_one_winner(study):
    root, client = study
    original = _read(client)
    start = Barrier(2)

    def rename():
        start.wait(timeout=HANG_SECONDS)
        return client.post(f"{_URL}/rename", json={
            "revision": original["revision"], "name": "renamed",
        })

    def patch():
        start.wait(timeout=HANG_SECONDS)
        return _patch(client, original["revision"], 3)

    with ThreadPoolExecutor(max_workers=2) as pool:
        rename_request, patch_request = pool.submit(rename), pool.submit(patch)
        renamed, patched = rename_request.result(), patch_request.result()

    assert sorted([renamed.status_code, patched.status_code]) == [200, 409]
    opened = client.get(f"{_URL}/open").json()
    if renamed.is_success:
        assert opened == renamed.json()
        assert patched.json()["detail"]["current"] == opened["snapshot"]["revision"]
        assert opened["snapshot"]["semantic_root"] == original["semantic_root"]
    else:
        assert opened["snapshot"] == patched.json()
        assert renamed.json()["detail"]["current"] == opened["snapshot"]["revision"]
        assert opened["configuration"]["project"] == "reach"
        assert _settings(root) == 3


def test_named_document_edit_reopens_and_presentation_leaves_authored_identity(study):
    root, client = study
    before = _read(client)
    assert "note" not in before["project"]["steps"]["fit"]
    file_revisions = {entry["address"]["id"]: entry["revision"] for entry in before["file_revisions"]}
    authored_file = root / "documents/point-mass.json"
    authored_file.chmod(0o644)
    with client:
        saved = _edit_mass(client, _URL, before, 3)
    assert saved["revision"] != before["revision"]
    assert saved["semantic_root"] != before["semantic_root"]
    current_files = {entry["address"]["id"]: entry["revision"] for entry in saved["file_revisions"]}
    assert current_files["fit"] == file_revisions["fit"]
    assert current_files[_MODEL] != file_revisions[_MODEL]
    assert _mass(root) == 3
    assert authored_file.stat().st_mode & 0o777 == 0o644
    assert _read(TestClient(create_app(projects_dir=root.parent))) == saved
    plan_before = compile_project(load_text_project(root).project)
    assert len(plan_before.runs) == 2 and not plan_before.diagnostics
    authored_bytes = authored_file.read_bytes()

    presentation = {
        "documents": {
            _MODEL: {
                "positions": {"plant": {"x": 40, "y": 20}},
                "camera": {"x": 2, "y": 3, "zoom": 1.5},
            }
        },
        "steps": {"fit": {"sweep_order": ["seed"]}},
    }
    response = client.put(
        f"{_URL}/presentation",
        json={
            "revision": saved["presentation_revision"],
            "presentation": presentation,
        },
    )
    assert response.status_code == 200, response.text
    placed = response.json()
    assert placed["presentation"]["documents"] == presentation["documents"]
    assert placed["presentation"]["steps"] == presentation["steps"]
    assert placed["presentation_revision"] != saved["presentation_revision"]
    assert placed["revision"] == saved["revision"]
    assert placed["semantic_root"] == saved["semantic_root"]
    assert authored_file.read_bytes() == authored_bytes
    assert compile_project(load_text_project(root).project) == plan_before
    assert _read(TestClient(create_app(projects_dir=root.parent))) == placed

    presentation_bytes = (root / "presentation.json").read_bytes()
    invalid = client.put(
        f"{_URL}/presentation",
        json={
            "revision": placed["presentation_revision"],
            "presentation": {
                **presentation,
                "window": {"drawer_open": True},
                "playback_position": 0.3,
            },
        },
    )
    assert invalid.status_code == 422
    assert (root / "presentation.json").read_bytes() == presentation_bytes
    stale = client.put(
        f"{_URL}/presentation",
        json={
            "revision": saved["presentation_revision"],
            "presentation": {},
        },
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["current"] == placed["presentation_revision"]
    assert stale.json()["detail"]["resource"] == "presentation"


def test_named_sweep_edit_preserves_value_ids_order_and_other_request(study):
    root, client = study
    before = _read(client)
    previous = compile_project(load_text_project(root).project)
    response = _patch(
        client,
        before["revision"],
        7,
        kind="step",
        name="fit",
        path=f"/sweep/axes/seed/values/10/patches/{_SEED}/value",
    )
    assert response.status_code == 200, response.text
    reopened = load_text_project(root)
    sweep = reopened.project.steps["fit"].sweep
    assert list(sweep.axes["seed"].values) == ["10", "2"]
    assert list(sweep.exceptions) == ["z_first", "a_second"]
    current = compile_project(reopened.project)
    assert [run.key for run in current.runs] == [run.key for run in previous.runs]
    assert [run.coordinates["seed"].value_id for run in current.runs] == ["10", "2"]
    assert current.runs[0].request_id != previous.runs[0].request_id
    assert current.runs[1].request_id == previous.runs[1].request_id


def test_competing_writers_from_one_revision_have_one_winner(study):
    root, client = study
    revision = _read(client)["revision"]
    start = Barrier(2, timeout=HANG_SECONDS)

    def save(value):
        writer = TestClient(create_app(projects_dir=root.parent))
        start.wait()
        return value, _patch(writer, revision, value)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(save, [4, 5]))
    assert sorted(response.status_code for response in results.values()) == [200, 409]
    (winner,) = (value for value, response in results.items() if response.status_code == 200)
    (loser,) = (response for response in results.values() if response.status_code == 409)
    assert _settings(root) == winner
    assert loser.json()["detail"]["current"] == load_text_project(root).revision


def test_external_revision_and_invalid_batch_preserve_external_bytes(study):
    root, client = study
    baseline = _read(client)
    path = root / "documents/settings.json"
    path.write_bytes(path.read_bytes() + b"\n")
    reformatted = _read(client)
    assert reformatted["revision"] != baseline["revision"]
    assert reformatted["semantic_root"] == baseline["semantic_root"]
    assert _patch(client, baseline["revision"], 3).status_code == 409

    body = json.loads(path.read_bytes())
    body["payload"]["value"] = 9
    data = json.dumps(body, indent=4).encode()
    write_bytes_atomic(path, data)
    assert _settings(root) == 9
    assert _patch(client, reformatted["revision"], 3).status_code == 409
    latest = _read(client)
    assert latest["semantic_root"] != reformatted["semantic_root"]
    response = client.patch(
        _URL,
        json={
            "revision": latest["revision"],
            "address": {"kind": "document", "id": _SETTINGS},
            "patches": [
                {"op": "replace", "path": _VALUE, "value": 3},
                {"op": "remove", "path": "/schema"},
            ],
        },
    )
    assert response.status_code == 422
    assert path.read_bytes() == data
    assert load_text_project(root).revision == latest["revision"]


def test_external_change_while_staging_is_detected_before_replace(study, monkeypatch):
    root, client = study
    before = _read(client)
    path = root / "documents/settings.json"
    raw = path.read_bytes() + b"\n"

    def inject_edit(target, data, **kwargs):
        precondition = kwargs.pop("before_publish")

        def competing_edit():
            write_bytes_atomic(target, raw)
            precondition()

        return write_bytes_atomic(target, data, before_publish=competing_edit, **kwargs)

    monkeypatch.setattr(project_folder, "write_bytes_atomic", inject_edit)
    response = _patch(client, before["revision"], 3)
    assert response.status_code == 409
    assert path.read_bytes() == raw
    assert not list(path.parent.glob(".*.tmp"))


@pytest.mark.parametrize("placement", [False, True])
def test_multi_file_patch_recovers_one_revision_after_interruption(study, monkeypatch, placement):
    root, client = study
    before = _read(client)
    files = [
        {
            "address": {"kind": "document", "id": _SETTINGS},
            "patches": [{"op": "replace", "path": _VALUE, "value": 3}],
        },
        {
            "address": {"kind": "step", "id": "fit"},
            "patches": [{"op": "add", "path": "/note", "value": {"reason": "shared edit"}}],
        },
    ]
    body = {"revision": before["revision"], "files": files}
    if placement:
        files.append({
            "address": {"kind": "presentation", "id": "presentation"},
            "patches": [{
                "op": "add", "path": "/steps",
                "value": {"fit": {"model_positions": {_MODEL: {"plant": {"x": 1, "y": 2}}}}},
            }],
        })
        before_write = _folder_state(root)
        conflict = client.patch(_URL, json={**body, "presentation_revision": "0" * 64})
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["resource"] == "presentation"
        assert _folder_state(root) == before_write
        body["presentation_revision"] = before["presentation_revision"]
    fail = True
    interrupt_at = "presentation.json" if placement else "fit.json"

    def interrupt(target, data, **kwargs):
        nonlocal fail
        if target.name == interrupt_at and fail:
            fail = False
            raise OSError("simulated interruption after the first file")
        return write_bytes_atomic(target, data, **kwargs)

    monkeypatch.setattr(project_folder, "write_bytes_atomic", interrupt)
    interrupted = client.patch(_URL, json=body)
    assert interrupted.status_code == 503
    assert (root / ".studio-patch.json").is_file()
    after = _read(client)
    assert after["revision"] != before["revision"]
    assert _settings(root) == 3
    assert after["project"]["steps"]["fit"]["note"]["reason"] == "shared edit"
    if placement:
        assert after["presentation_revision"] != before["presentation_revision"]
        assert after["presentation"]["steps"]["fit"]["model_positions"][_MODEL] == {
            "plant": {"x": 1.0, "y": 2.0},
        }
    assert not (root / ".studio-patch.json").exists()
    stale = client.patch(_URL, json=body)
    assert stale.status_code == 409
    assert _read(client)["revision"] == after["revision"]


def test_unreadable_presentation_does_not_hide_the_authored_project(study):
    root, client = study
    before = _read(client)
    (root / "presentation.json").write_bytes(b"{")
    broken = _read(client)
    assert broken["presentation"] is None
    assert broken["presentation_error"] == "unreadable_presentation"
    assert broken["revision"] == before["revision"]
    assert broken["presentation_revision"] != before["presentation_revision"]
    assert _patch(client, broken["revision"], 3).status_code == 200
    repaired = client.put(
        f"{_URL}/presentation",
        json={
            "revision": broken["presentation_revision"],
            "presentation": {},
        },
    )
    assert repaired.status_code == 200
    assert repaired.json()["presentation_error"] is None
    assert _settings(root) == 3


def test_project_and_authored_paths_cannot_point_outside_the_root(study, tmp_path):
    root, client = study
    (tmp_path / "elsewhere").symlink_to(root, target_is_directory=True)
    assert client.get("/api/projects/elsewhere").status_code == 422
    original = root / "documents/point-mass.json"
    outside = tmp_path / "outside.json"
    original.rename(outside)
    original.symlink_to(outside)
    assert client.get(_URL).status_code == 422


@pytest.mark.parametrize("through_core", [False, True], ids=["raw-text", "core-edit"])
def test_existing_event_stream_observes_an_external_edit(study, monkeypatch, through_core):
    root, client = study
    before = load_text_project(root).revision
    monkeypatch.setattr(jobs, "_FOLDER_OBSERVATION_INTERVAL", 0)

    class JobsStream:
        calls = 0

        def events(self, *, after, projects, wait, machine):
            self.calls += 1
            if self.calls == 2:
                if through_core:
                    from feedbax.contracts.studio_project import StudioProjectPatch

                    project_folder.ProjectFolder(root.parent).patch(root.name, StudioProjectPatch.model_validate({
                        "revision": before,
                        "address": {"kind": "document", "id": _SETTINGS},
                        "patches": [{"op": "replace", "path": _VALUE, "value": 11}],
                    }))
                else:
                    path = root / "documents/settings.json"
                    document = json.loads(path.read_bytes())
                    document["payload"]["value"] = 11
                    write_bytes_atomic(path, json.dumps(document).encode())
            if self.calls > 2:
                raise HostError("unavailable", "Stream ended")
            return 0, StudioJobEventHistory(cursor=0, oldest_cursor=0, events=[])

    client.app.state.job_activity = JobsStream()
    response = client.get("/api/events?folder=reach")
    assert response.status_code == 200
    observations = [
        json.loads(frame.split("\ndata: ", 1)[1])
        for frame in response.text.split("\n\n")
        if frame.startswith("event: project_revision\n")
    ]
    assert [notice["revision"] for notice in observations] == [
        before,
        load_text_project(root).revision,
    ]
    assert observations[0]["revision"] != observations[1]["revision"]
    assert _settings(root) == 11


_BATCHES = "/training_config/n_batches"
_RATE = "/settings/optimizer/learning_rate/value"


def _shared_training(tmp_path, *, continuations=0, other=True, third=False, checkpoints=True, variants=()):
    """Two Train steps, ``train`` and ``other``, read their method, settings and run controls from ``training``.

    ``variants`` are the project's variant rows, none by default. ``third`` adds a third such step. ``continuations`` Train steps follow ``train`` in a chain, ``train-2`` based on
    ``train``, ``train-3`` on ``train-2``, each starting from its parent's checkpoint unless not ``checkpoints``.
    """
    from feedbax.contracts.training_document import TRAINING_DOCUMENT_KEYS, TRAINING_DOCUMENT_SCHEMA_ID

    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    inline = raw["steps"]["train"]["base"]["inline"]
    raw["documents"]["training"] = {
        "schema": TRAINING_DOCUMENT_SCHEMA_ID, "payload": {key: inline[key] for key in TRAINING_DOCUMENT_KEYS},
    }
    inline |= {key: {"document": "training", "pointer": f"/{key}"} for key in TRAINING_DOCUMENT_KEYS}
    if other:
        raw["steps"]["other"] = deepcopy(raw["steps"]["train"]) | {"note": None}
        raw["steps"]["other"]["base"]["inline"]["seed"] = 1
    if third:
        raw["steps"]["third"] = deepcopy(raw["steps"]["train"]) | {"note": None}
        raw["steps"]["third"]["base"]["inline"]["seed"] = 2
    if variants:
        raw["sweep"] = {"rows": {variant: {} for variant in variants}}
    parent = "train"
    for number in range(2, continuations + 2):
        raw["steps"][f"train-{number}"] = {
            "kind": "train", "base": {"step": parent},
            "inputs": {"checkpoint": {"step": parent, "checkpoint": "last"}},
            "deltas": [] if number > 2 else [{"layer_id": "initialize", "patches": [
                {"path": "checkpoint_initialization", "op": "add", "value": "initialize_from"},
            ]}],
        } if checkpoints else {"kind": "train", "base": {"step": parent}, "deltas": [{"layer_id": "seed", "patches": [
            {"path": "seed", "op": "add", "value": 10 + number},
        ]}]}
        parent = f"train-{number}"
    root = tmp_path / "reach"
    for name, data in dump_text_project(
        Project.model_validate(raw), ProjectConfiguration(project="reach", store="results"),
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root, TestClient(create_app(projects_dir=tmp_path))


class _Training:
    """Edits of the steps of a project through the Studio endpoints, each saved as it is made."""

    def __init__(self, client, root):
        self.client, self.root = client, root

    @property
    def revision(self):
        return _read(self.client)["revision"]

    def plan(self):
        folders = self.client.app.state.project_folders
        plan = folders.read_plan("reach", revision=self.revision).plan
        return folders.read_plan("reach", revision=self.revision, source_runs=tuple(row.key for row in plan.rows)).plan

    def requests(self, step):
        return {run.key: run.request_id for run in self.plan().runs if run.key.step == step}

    def save(self, response):
        assert response.status_code == 200, response.text
        body = response.json()
        assert not body.get("problems") and not body.get("refusal"), body
        for file in body.get("created", []):
            created = self.client.post(f"{_URL}/files", json={"revision": self.revision, "file": file})
            assert created.status_code == 201, created.text
        if body["files"]:
            saved = self.client.patch(_URL, json={"revision": self.revision, "files": body["files"]})
            assert saved.status_code == 200, saved.text
        return body

    def number(self, step, field, edit, chosen=(), variant=None):
        run = {"run": {"variant": variant, "step": step, "row": "default"}} if variant else {}
        return self.save(self.client.post(f"{_URL}/authoring/train/{step}/fields", json={
            "revision": self.revision, "field": field, "edit": edit, "chosen_steps": list(chosen), **run,
        }))

    def reset(self, step, field, chosen=()):
        return self.save(self.client.post(f"{_URL}/authoring/train/{step}", json={
            "revision": self.revision, "reset": field, "chosen_steps": list(chosen),
        }))

    def values(self, path=_BATCHES):
        return {key.step: _select(definition.payload, path)
                for key, definition in self.plan().selected_definitions.items()}

    def reads(self):
        return {key.step: definition.document_ids["payload.training_config"]
                for key, definition in self.plan().selected_definitions.items()}

    def documents(self):
        return load_text_project(self.root).project.documents

    def step_file(self, step):
        return (self.root / "steps" / f"{step}.json").read_bytes()

    def choose(self, step, document):
        row = min(row.key.row for row in self.plan().rows if row.key.step == step)
        return self.client.post(f"{_URL}/authoring/trials", json={
            "revision": self.revision, "run": {"variant": "default", "step": step, "row": row},
            "change": {"kind": "select_document", "role": "training", "document": document},
        })


def _changed(before, after):
    return {key for key in before if before[key] != after[key]}


def _select(payload, path):
    for part in json_pointer_parts(path):
        payload = payload[part]
    return payload


def _differences(train, name):
    return {path: difference.value for path, difference in train.documents()[name].differences.items()}


def test_a_narrowed_training_edit_is_a_document_based_on_the_shared_one(tmp_path):
    root, client = _shared_training(tmp_path)
    with client:
        train = _Training(client, root)
        shared = root / "documents" / "training.json"
        stored, other = shared.read_bytes(), train.requests("other")
        first = train.requests("train")
        train.number("train", _BATCHES, {"kind": "text", "text": "5"})
        assert shared.read_bytes() == stored
        assert train.documents()["training-2"].base.document == "training"
        assert _differences(train, "training-2") == {_BATCHES: 5}
        assert train.reads() == {"train": "training-2", "other": "training"}
        assert train.requests("other") == other
        assert _changed(first, train.requests("train")) == first.keys()

        before = {step: train.requests(step) for step in ("train", "other")}
        train.number("other", _RATE, {"kind": "text", "text": "0.002"}, chosen=("other", "train"))
        assert train.values(_RATE) == {"train": 0.002, "other": 0.002}
        assert all(_changed(before[step], train.requests(step)) == before[step].keys() for step in before)
        assert _differences(train, "training-2") == {_BATCHES: 5}

        before = {step: train.requests(step) for step in ("train", "other")}
        train.number("other", _BATCHES, {"kind": "text", "text": "6"})
        assert train.values() == {"train": 5, "other": 6}
        assert train.requests("train") == before["train"]
        assert _differences(train, "training-2") == {_BATCHES: 5}

        train.number("other", _BATCHES, {"kind": "text", "text": "7"}, chosen=("other", "train"))
        assert train.values() == {"train": 7, "other": 7}
        assert _differences(train, "training-2") == {}
        assert set(train.documents()) >= {"training", "training-2"} and "training-3" not in train.documents()


def test_a_variant_s_narrowed_training_edit_is_a_document_only_its_row_points_at(tmp_path):
    root, client = _shared_training(tmp_path, variants=("control", "treated"))
    with client:
        train = _Training(client, root)
        files = {path: path.read_bytes() for path in (root / "documents" / "training.json", *(root / "steps").iterdir())}
        before = {run.key: run.request_id for run in train.plan().runs}
        train.number("train", _BATCHES, {"kind": "text", "text": "5"}, variant="treated")
        assert {path: path.read_bytes() for path in files} == files
        assert train.documents()["training-2"].base.document == "training"
        assert _differences(train, "training-2") == {_BATCHES: 5}
        patches = load_text_project(root).project.sweep.rows["treated"].patches
        assert patches and all(
            path.startswith("steps.train.base.inline.") and write.value["document"] == "training-2"
            for path, write in patches.items()
        )
        assert not load_text_project(root).project.sweep.rows["control"].patches
        after = {run.key: run.request_id for run in train.plan().runs}
        assert _changed(before, after) == {key for key in before if (key.variant, key.step) == ("treated", "train")}
        values = {key: _select(definition.payload, _BATCHES) for key, definition in train.plan().selected_definitions.items()}
        assert values[RunKey("treated", "train", "default")] == 5
        assert values[RunKey("treated", "other", "default")] == values[RunKey("control", "train", "default")] != 5


def test_the_chosen_readers_of_a_document_share_one_document_based_on_it(tmp_path):
    root, client = _shared_training(tmp_path, third=True)
    with client:
        train = _Training(client, root)
        third = train.requests("third")
        train.number("train", _BATCHES, {"kind": "text", "text": "5"}, chosen=("train", "other"))
        assert train.reads() == {"train": "training-2", "other": "training-2", "third": "training"}
        assert train.values() == {"train": 5, "other": 5, "third": 2}
        assert train.requests("third") == third

        other = train.requests("other")
        train.number("train", _BATCHES, {"kind": "text", "text": "8"})
        documents = train.documents()
        assert documents["training-3"].base.document == "training-2"
        assert _differences(train, "training-3") == {_BATCHES: 8}
        assert train.reads() == {"train": "training-3", "other": "training-2", "third": "training"}
        assert train.requests("other") == other

        refused = client.post(f"{_URL}/authoring/train/train/fields", json={
            "revision": train.revision, "field": _BATCHES, "edit": {"kind": "text", "text": "9"},
            "chosen_steps": ["train", "third"],
        })
        assert refused.status_code == 422, refused.text
        assert member_name(documents, "training-3") in refused.json()["detail"]


def test_a_continuation_keeps_the_document_it_read_when_its_parent_moves(tmp_path):
    root, client = _shared_training(tmp_path, continuations=1, checkpoints=False)
    with client:
        train = _Training(client, root)
        child = train.requests("train-2")
        train.number("train", _BATCHES, {"kind": "text", "text": "5"})
        assert train.reads() == {"train": "training-2", "other": "training", "train-2": "training"}
        assert train.requests("train-2") == child
        references = load_text_project(root).project.steps["train-2"].deltas[-1].patches
        assert {(patch.path, patch.value["document"]) for patch in references} >= {("training_config", "training")}

    root, client = _shared_training(tmp_path / "both", continuations=1, checkpoints=False)
    with client:
        train = _Training(client, root)
        before = train.step_file("train-2")
        train.number("train", _BATCHES, {"kind": "text", "text": "5"}, chosen=("train", "train-2"))
        assert train.reads() == {"train": "training-2", "other": "training", "train-2": "training-2"}
        assert train.step_file("train-2") == before


def test_reset_gives_a_value_its_base_s(tmp_path):
    root, client = _shared_training(tmp_path)
    with client:
        train = _Training(client, root)
        on_training = train.requests("train")
        train.number("train", _BATCHES, {"kind": "text", "text": "5"})
        assert train.requests("train") != on_training
        train.reset("train", _BATCHES)
        assert _differences(train, "training-2") == {}
        assert train.reads()["train"] == "training-2"
        assert train.requests("train") == on_training


def test_a_continuation_s_edit_of_its_training_document_is_its_own(tmp_path):
    root, client = _shared_training(tmp_path, continuations=1)
    with client:
        train = _Training(client, root)
        parent, child = train.requests("train"), train.requests("train-2")
        assert child
        stored = (root / "documents" / "training.json").read_bytes()
        train.number("train-2", _BATCHES, {"kind": "text", "text": "4"})
        assert (root / "documents" / "training.json").read_bytes() == stored
        assert train.reads() == {"train": "training", "other": "training", "train-2": "training-2"}
        assert train.requests("train") == parent
        assert _changed(child, train.requests("train-2")) == child.keys()


def _treated_reads(train):
    return {key.step: definition.document_ids["payload.training_config"]
            for key, definition in train.plan().selected_definitions.items() if key.variant == "treated"}


def _ids(train):
    return {run.key: run.request_id for run in train.plan().runs}


def test_a_continuation_s_edit_in_a_variant_is_a_layer_its_row_appends(tmp_path):
    root, client = _shared_training(tmp_path, continuations=1, variants=("control", "treated"))
    with client:
        train = _Training(client, root)
        files = {path: path.read_bytes() for path in (root / "documents" / "training.json", *(root / "steps").iterdir())}
        before = _ids(train)
        train.number("train-2", _BATCHES, {"kind": "text", "text": "4"}, variant="treated")
        assert {path: path.read_bytes() for path in files} == files
        assert _differences(train, "training-2") == {_BATCHES: 4}
        assert _treated_reads(train) == {"train": "training", "other": "training", "train-2": "training-2"}
        assert _changed(before, _ids(train)) == {key for key in before if (key.variant, key.step) == ("treated", "train-2")}
        after = _ids(train)
        train.number("train-2", _BATCHES, {"kind": "text", "text": "6"}, variant="treated")
        assert _select(train.plan().selected_definitions[RunKey("treated", "train-2", "default")].payload, _BATCHES) == 6
        assert _changed(after, _ids(train)) == {key for key in after if (key.variant, key.step) == ("treated", "train-2")}


def test_a_variant_edit_back_to_the_default_value_drops_the_row_s_write(tmp_path):
    root, client = _shared_training(tmp_path, variants=("control", "treated"))
    with client:
        train = _Training(client, root)
        default = _ids(train)
        value = _select(train.plan().selected_definitions[RunKey("control", "train", "default")].payload, _BATCHES)
        write = {"documents.training.payload.training_config.n_batches": {"op": "replace", "value": value + 3}}
        saved = client.patch(_URL, json={"revision": train.revision, "files": [{
            "address": {"kind": "variant", "id": "treated"}, "patches": [{"op": "add", "path": "/patches", "value": write}],
        }]})
        assert saved.status_code == 200, saved.text
        assert _changed(default, _ids(train))
        train.number("train", _BATCHES, {"kind": "text", "text": str(value)}, chosen=("train", "other"), variant="treated")
        assert not load_text_project(root).project.sweep.rows["treated"].patches
        assert "training-2" not in train.documents()
        assert _ids(train) == default


def test_a_sweep_on_a_training_document_step_is_the_step_s_own(tmp_path):
    root, client = _shared_training(tmp_path)
    with client:
        train = _Training(client, root)
        document = root / "documents" / "training.json"
        stored = document.read_bytes()
        other = train.requests("other")
        train.number("train", _RATE, {"kind": "text", "text": "0.001, 0.003"})
        sweep = load_text_project(root).project.steps["train"].sweep
        assert list(sweep.axes) == ["learning-rate"]
        assert {path for value in sweep.axes["learning-rate"].values.values() for path in value.patches} == {
            "settings.optimizer.learning_rate.value",
        }
        assert document.read_bytes() == stored
        assert train.requests("other") == other
        assert len(set(train.requests("train").values())) == 2

        lacking = load_text_project(root).project.documents["training"].payload | {}
        lacking = deepcopy(lacking)
        lacking["settings"]["optimizer"]["learning_rate"] = {"type": "warmup_constant_schedule", "init_value": 0.0,
                                                             "peak_value": 0.01, "warmup_steps": 1}
        for name, payload in (("training-2", load_text_project(root).project.documents["training"].payload),
                              ("training-3", lacking)):
            created = client.post(f"{_URL}/files", json={"revision": train.revision, "file": {
                "kind": "document", "id": name, "document": {"schema": "feedbax.spec.training_document", "payload": payload},
            }})
            assert created.status_code == 201, created.text
        changed = train.choose("train", "training-3")
        assert changed.status_code == 200, changed.text
        recipe = changed.json()
        assert recipe["files"], recipe
        saved = client.patch(_URL, json={"revision": train.revision, "files": recipe["files"]})
        assert saved.status_code == 200, saved.text
        assert any(
            problem.step == "train" and problem.code == ResolutionCode.INVALID_SWEEP
            and "learning_rate" in (problem.path + problem.detail)
            for problem in train.plan().diagnostics
        )
        train.save(train.choose("train", "training-2"))
        plan = train.plan()
        definitions = {key: value for key, value in plan.selected_definitions.items() if key.step == "train"}
        assert len(definitions) == 2
        assert {value.document_ids["payload.settings"] for value in definitions.values()} == {"training-2"}
        assert {value.payload["settings"]["optimizer"]["learning_rate"]["value"] for value in definitions.values()} == {
            0.001, 0.003,
        }

        stored = (root / "documents" / "training.json").read_bytes()
        train.number("train", _RATE, {"kind": "to_value", "value_id": "value-2"})
        project = load_text_project(root).project
        assert not project.steps["train"].sweep.axes.get("learning-rate")
        assert (root / "documents" / "training.json").read_bytes() == stored
        assert project.documents["training-2"].payload["settings"]["optimizer"]["learning_rate"]["value"] == 0.003
        (definition,) = [value for key, value in train.plan().selected_definitions.items() if key.step == "train"]
        assert definition.payload["settings"]["optimizer"]["learning_rate"]["value"] == 0.003


def test_the_write_refuses_a_change_to_a_step_it_does_not_reach(tmp_path):
    from feedbax.web.services.trial_scope import refuse_unreached

    root, client = _shared_training(tmp_path)
    with client:
        folders = client.app.state.project_folders
        settled = folders.read_plan("reach", revision=None)
        project = load_text_project(root).project
        edited = project.model_copy(deep=True)
        edited.documents["training"].payload["training_config"]["n_batches"] = 3
        candidate = compile_project(edited, readers=settled.readers)
        reach = {row.key for row in settled.plan.rows if row.key.step == "train"}
        with pytest.raises(ProjectFolderInvalid, match="outside the selected scope"):
            refuse_unreached(settled.plan, candidate, reach)
        refuse_unreached(settled.plan, candidate, {row.key for row in settled.plan.rows})


def test_a_shared_training_document_keeps_a_sweep_s_value_as_the_step_s_difference(tmp_path):
    root, client = _shared_training(tmp_path)
    with client:
        train = _Training(client, root)
        stored = (root / "documents" / "training.json").read_bytes()
        train.number("train", _RATE, {"kind": "text", "text": "0.001, 0.003"})
        other = train.requests("other")
        train.number("train", _RATE, {"kind": "to_value", "value_id": "value-2"})
        assert (root / "documents" / "training.json").read_bytes() == stored
        assert train.requests("other") == other
        assert train.reads()["train"] == "training-2"
        assert _differences(train, "training-2") == {_RATE: 0.003}
        (definition,) = [value for key, value in train.plan().selected_definitions.items() if key.step == "train"]
        assert definition.payload["settings"]["optimizer"]["learning_rate"]["value"] == 0.003


def _start(train, step, source):
    return train.save(train.client.post(f"{_URL}/authoring/train/{step}/start", json={
        "revision": train.revision, "source": source,
    }))


def test_starting_a_continuation_from_its_model_detaches_it_and_back_continues(tmp_path):
    root, client = _shared_training(tmp_path, continuations=1, other=False)
    with client:
        train = _Training(client, root)
        train.number("train", _RATE, {"kind": "text", "text": "0.001, 0.003"})
        file = root / "steps" / "train-2.json"
        continuation = load_text_project(root).project.steps["train-2"]

        _start(train, "train-2", None)
        detached = load_text_project(root).project.steps["train-2"]
        assert isinstance(detached.base, InlineBase) and not detached.inputs and not detached.deltas
        assert detached.base.inline["settings"] == {"document": "training", "pointer": "/settings"}
        assert all(not run.inputs for run in train.plan().runs if run.key.step == "train-2")
        assert set(train.requests("train-2").values()) == set(train.requests("train").values())
        response = client.post(f"{_URL}/authoring/train/train-2", json={"revision": train.revision})
        assert response.status_code == 200, response.text

        _start(train, "train-2", {"step": "train", "checkpoint": "last"})
        restored = load_text_project(root).project.steps["train-2"]
        assert restored.base == continuation.base and restored.inputs == continuation.inputs
        assert restored.sweep is None and not restored.requires
        assert [patch.model_dump(exclude_unset=True) for layer in restored.deltas for patch in layer.patches] == [
            {"path": "checkpoint_initialization", "op": "add", "value": "initialize_from"},
        ]

        _start(train, "train-2", None)
        detached = train.requests("train-2")
        train.number("train", _RATE, {"kind": "text", "text": "0.002, 0.004"})
        assert train.requests("train-2") == detached
        stored = file.read_bytes()
        train.number("train-2", _BATCHES, {"kind": "text", "text": "5"}, chosen=("train", "train-2"))
        assert file.read_bytes() == stored
        assert _changed(detached, train.requests("train-2")) == detached.keys()


def test_a_step_based_on_a_step_without_its_checkpoint_opens_in_the_train_editor(tmp_path):
    root, client = _shared_training(tmp_path, continuations=1, other=False)
    with client:
        train = _Training(client, root)
        saved = client.patch(_URL, json={"revision": train.revision, "address": {"kind": "step", "id": "train-2"},
                                         "patches": [{"op": "remove", "path": "/deltas"}, {"op": "remove", "path": "/inputs"}]})
        assert saved.status_code == 200, saved.text
        step = load_text_project(root).project.steps["train-2"]
        assert step.base == StepBase(step="train") and not step.inputs and not step.deltas

        def fields(step):
            response = client.post(f"{_URL}/authoring/train/{step}", json={"revision": train.revision})
            assert response.status_code == 200, response.text
            return {field["id"]: field["value"] for field in response.json()["fields"]}

        parent = fields("train")
        assert fields("train-2") == parent
        train.number("train-2", "/seed", {"kind": "text", "text": "3"})
        assert fields("train-2")["/seed"] == 3 and fields("train") == parent


def test_choosing_a_document_and_then_the_one_it_replaced_leaves_the_step_as_it_was(tmp_path):
    root, client = _shared_training(tmp_path, continuations=1, other=False)
    with client:
        train = _Training(client, root)
        payload = load_text_project(root).project.documents["training"].payload
        created = client.post(f"{_URL}/files", json={"revision": train.revision, "file": {
            "kind": "document", "id": "training-2", "document": {"schema": "feedbax.spec.training_document", "payload": payload},
        }})
        assert created.status_code == 201, created.text
        file = root / "steps" / "train-2.json"
        stored = file.read_bytes()
        train.save(train.choose("train-2", "training-2"))
        assert file.read_bytes() != stored
        train.save(train.choose("train-2", "training"))
        assert file.read_bytes() == stored
