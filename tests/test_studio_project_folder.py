"""Studio edits the same durable authored files the text reader opens."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import wraps
import json
from pathlib import Path
from threading import Barrier, Event

from fastapi.testclient import TestClient
from pydantic import Field, JsonValue
import pytest

from feedbax._atomic import write_bytes_atomic
from feedbax.contracts import project_plan, request_identity
from feedbax.contracts.base import StrictModel
from feedbax.contracts.jobs import StudioJobEventHistory
from feedbax.contracts.project import Project
from feedbax.contracts.project_experiment import PROJECT_DECLARATION_FILENAME
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, load_text_project
from feedbax.contracts.project_plan import RunKey, compile_project
from feedbax.contracts.project_resolution import ResolutionCode
from feedbax.contracts.request_identity import REQUEST_KINDS
from feedbax.contracts.studio_authoring import StudioAuthoringChange
from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD
from feedbax.host import HostError
from feedbax.plugins.application import TRAINING_PROGRAMS, new_application_registry_bundle
from feedbax.plugins.bootstrap import BootstrapState, PluginProvenance
from feedbax.studio.authoring import authoring_surface, project_train_fields
from feedbax.web.api import jobs
from feedbax.web.app import create_app
from feedbax.web.services import project_folder
from web.e2e.studio.fixtures.reaching_project import reaching_project


_URL = "/api/projects/reach"
_GAIN = "/payload/nodes/gain/params/gain"


@pytest.fixture
def study(tmp_path: Path):
    corpus = json.loads(
        (Path(__file__).parent / "fixtures" / "request_identity_corpus.json").read_bytes()
    )
    spec = next(row["spec"] for row in corpus["requests"] if row["kind"] == "train")
    graph = spec.pop("graph")
    project = Project.model_validate(
        {
            "documents": {"model": {"schema": graph["schema_id"], "payload": graph}},
            "steps": {
                "fit": {
                    "kind": "train",
                    "base": {"inline": {"graph": {"document": "model"}, **spec}},
                    "sweep": {
                        "axes": {
                            "seed": {
                                "values": {
                                    "10": {"patches": {"seed": {"value": 10}}},
                                    "2": {"patches": {"seed": {"value": 2}}},
                                }
                            }
                        },
                        "exceptions": {
                            "z_first": {"where": {"rows": []}},
                            "a_second": {"where": {"rows": []}},
                        },
                    },
                }
            },
        }
    )
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
    name="model",
    path=_GAIN,
):
    return client.patch(
        _URL,
        json={
            "revision": revision,
            "address": {"kind": kind, "id": name},
            "patches": [{"op": "replace", "path": path, "value": value}],
        },
    )


def _gain(root: Path):
    return (
        load_text_project(root)
        .project.documents["model"]
        .payload["nodes"]["gain"]["params"]["gain"]
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
    snapshot = response.json()["snapshot"]
    root = tmp_path / snapshot["project_id"]
    url = f"/api/projects/{snapshot['project_id']}"
    fixture = reaching_project()
    model = fixture.graph.model_dump(mode="json", exclude_none=True)
    task = fixture.workspace.model_dump(mode="json", exclude_none=True)["scenarios"][
        "scenario:train"
    ]["task_spec"]
    for entry in (
        {"kind": "step", "id": "train", "step": {"kind": "train", "base": {"inline": {}}}},
        {
            "kind": "document",
            "id": "model",
            "document": {"schema": model["schema_id"], "payload": model},
        },
        {
            "kind": "document",
            "id": "task",
            "document": {"schema": "feedbax.spec.task", "payload": task},
        },
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
                    "value": {"graph": {"document": "model"}, "task": {"document": "task"}},
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    snapshot = response.json()
    selected = RunKey("default", "train", "default")
    mass = "/payload/graph/nodes/plant/params/mass"
    before = _read_plan(client, root, snapshot, selected)
    assert before.variants == ("default",)
    assert [row.key for row in before.rows] == [selected]
    assert not before.runs and not before.requests
    assert {item.path for item in before.diagnostics} == {
        "payload.objective",
        "payload.method_payload",
        "payload.method_ref",
        "payload.training_config",
    }
    assert {(item.code, item.variant, item.step, item.row) for item in before.diagnostics} == {
        (ResolutionCode.INVALID_PAYLOAD, selected.variant, selected.step, selected.row)
    }
    assert set(before.selected_definitions) == {selected}
    definition = before.selected_definitions[selected]
    assert definition.payload["graph"] == model
    assert definition.payload["task"] == task
    assert (
        definition.sources[mass].kind,
        definition.sources[mass].name,
        definition.sources[mass].path,
    ) == ("document", "model", "/nodes/plant/params/mass")
    assert definition.sources["/payload/task/params/n_steps"].name == "task"
    assert not _read_plan(client, root, snapshot).selected_definitions

    response = client.patch(
        url,
        json={
            "revision": snapshot["revision"],
            "address": {"kind": "document", "id": "model"},
            "patches": [
                {"op": "replace", "path": "/payload/nodes/plant/params/mass", "value": 2.0}
            ],
        },
    )
    assert response.status_code == 200, response.text
    current = response.json()
    assert current["revision"] != snapshot["revision"]
    plan = _read_plan(client, root, current, selected)
    assert plan.rows == before.rows and plan.diagnostics == before.diagnostics
    assert plan.sources == before.sources
    assert (
        plan.selected_definitions[selected].payload["graph"]["nodes"]["plant"]["params"]["mass"]
        == 2.0
    )
    assert definition.payload["graph"]["nodes"]["plant"]["params"]["mass"] == 1.0
    files = _folder_state(root)
    with pytest.raises(project_folder.ProjectRevisionConflict) as conflict:
        app.state.project_folders.read_plan(snapshot["project_id"], revision=snapshot["revision"])
    assert (conflict.value.resource, conflict.value.current) == ("project", current["revision"])
    assert _folder_state(root) == files
    assert not (root / "results").exists()


def test_authoring_surface_uses_bootstrapped_method_declarations():
    class PluginPayload(StrictModel):
        plateau_updates: int = Field(gt=0)

    bundle = new_application_registry_bundle()
    registry = bundle.training_programs
    builtin = registry.resolve(STUDIO_GRAPH_METHOD, path="/method_ref")
    method = replace(
        builtin,
        method_ref="example/authoring/v1",
        payload_model=PluginPayload,
        payload_schema_id="example.spec.authoring",
        payload_schema_version="example.spec.authoring.v1",
        owner="example.authoring",
        package="example",
    )
    registry.register(method)
    source = PluginProvenance(
        plugin_id="example.authoring",
        plugin_version="v1",
        downstream_protocol_version=1,
        distribution="example",
        distribution_version="1",
        fingerprint="test",
        entry_point_name="example",
        entry_point_value="example:register",
        family_protocols={TRAINING_PROGRAMS.family: TRAINING_PROGRAMS.protocol_version},
        registration_order=0,
        registered_keys={TRAINING_PROGRAMS.family: (method.method_ref,)},
    )
    surface = authoring_surface(BootstrapState(bundle, (source,)))

    assert surface.project_grammar == Project.model_json_schema()
    assert surface.step_payloads == {"train": REQUEST_KINDS["train"].spec_type.model_json_schema()}
    declaration = next(item for item in surface.training_methods if item.key == method.method_ref)
    assert declaration.plugin == source.plugin_id
    assert declaration.owner == method.owner
    assert declaration.payload_schema == PluginPayload.model_json_schema()


def test_train_authoring_keeps_composed_steps_and_incompatible_documents_intact(study):
    root, client = study
    with client:
        snapshot = _read(client)
        original = _folder_state(root)
        composed = client.post(
            f"{_URL}/authoring/train/fit", json={"revision": snapshot["revision"]},
        )
        assert composed.status_code == 422, composed.text
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
        assert fields["/graph"]["control"]["options"] == [{"label": "model", "value": "model"}]
        assert fields["/task"]["control"] is None
        invalid = client.post(url, json={
            "revision": revision, "change": {"field": "/task", "value": "model"},
        })
        assert invalid.status_code == 422
        assert _folder_state(root) == original


def test_initial_training_settings_retain_the_authored_method_rate_and_batch_count():
    registry = new_application_registry_bundle().training_programs
    method = registry.resolve(STUDIO_GRAPH_METHOD, path="/method_ref")
    project = Project.model_validate({
        "steps": {
            "train": {"kind": "train", "base": {"inline": {
                "task": {"type": "SimpleReaches", "params": {"n_steps": 80}},
                "method_ref": STUDIO_GRAPH_METHOD.model_dump(),
                "method_payload": {
                    "schema_id": method.payload_schema_id,
                    "schema_version": method.payload_schema_version,
                    "payload": {
                        "optimizer": {"type": "adam", "params": {"learning_rate": 0.02}},
                        "target_total": 25,
                    },
                },
            }}},
        },
    })
    _, patches, _ = project_train_fields(
        project, "train", registry,
        StudioAuthoringChange(field="/training_config/snapshot_interval", value=12),
    )
    patch, = patches
    assert patch.path == "/base/inline/training_config"
    assert patch.value == {
        "n_batches": 25, "learning_rate": 0.02, "batch_size": 1, "n_reach_steps": 80,
        "snapshot_interval": 12,
    }


    raw = project.model_dump()
    inline = raw["steps"]["train"]["base"]["inline"]
    inline["training_config"] = patch.value
    reference = {"document": "parameters"}
    inline["method_payload"]["payload"]["optimizer"]["params"] = reference
    raw["documents"] = {
        "parameters": {"schema_id": "example.optimizer", "payload": {"learning_rate": 0.02}},
    }
    referenced = Project.model_validate(raw, by_name=True)
    with pytest.raises(ValueError, match="has no supported edit"):
        project_train_fields(
            referenced, "train", registry,
            StudioAuthoringChange(field="/training_config/learning_rate", value=0.03),
        )
    assert inline["method_payload"]["payload"]["optimizer"]["params"] == reference

    _, patches, _ = project_train_fields(
        project, "train", registry,
        StudioAuthoringChange(field="/method_payload/payload/seed", value=7.0),
    )
    seed, = patches
    assert seed.path == "/base/inline/method_payload/payload/seed"
    assert type(seed.value) is int and seed.value == 7


def test_plan_preserves_full_variant_lineage_and_named_subset_after_edit(study):
    root, client = study
    original = load_text_project(root)
    authored = original.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    authored["documents"]["selected"] = {"schema": "subset", "payload": {"rows": ["seed=10"]}}
    authored["steps"]["follow"] = {
        "kind": "train",
        "base": {"step": "fit"},
        "sweep": {"axes": {"seed": None}},
        "inputs": {"checkpoint": {"step": "fit", "where": {"subset": "selected"}}},
    }
    authored["sweep"] = {"rows": {"base": {}, "compare": {}}}
    project = Project.model_validate(authored)
    for relative, data in dump_text_project(project, original.configuration).items():
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)

    snapshot = _read(client)
    selected = RunKey("compare", "follow", "default")
    before = _read_plan(client, root, snapshot, selected)
    assert before.variants == ("base", "compare")
    assert len(before.runs) == len(before.rows) == 6
    assert not before.diagnostics
    assert len(before.requests) == 3
    for variant in before.variants:
        [follower] = [run for run in before.runs if run.key == RunKey(variant, "follow", "default")]
        producer = RunKey(variant, "fit", "seed=10")
        assert follower.inputs["checkpoint"].producer == producer
        subset = before.subsets[variant, "follow", "checkpoint"]
        assert (subset.document, subset.candidates, subset.members) == ("selected", 2, (producer,))
        assert subset.terms.rows == ["seed=10"]
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
        path="/sweep/axes/seed/values/10/patches/seed/value",
    )
    assert edited.status_code == 200, edited.text
    after = _read_plan(client, root, edited.json(), selected)
    before_requests = {run.key: run.request_id for run in before.runs}
    assert set(before_requests) == {run.key for run in after.runs}
    assert {run.key for run in after.runs if run.request_id != before_requests[run.key]} == {
        RunKey(variant, step, row)
        for variant in after.variants
        for step, row in (("fit", "seed=10"), ("follow", "default"))
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
        if project.documents["model"].payload["nodes"]["gain"]["params"]["gain"] == 1:
            entered.set()
            assert release.wait(5)
        return compile_project(project, **kwargs)

    monkeypatch.setattr(project_folder, "compile_project", pause_old_compile)
    with ThreadPoolExecutor(max_workers=1) as pool:
        previous = pool.submit(
            service.read_plan, "reach", revision=opened["revision"], source_runs=(selected,)
        )
        try:
            assert entered.wait(5)
            changed = _patch(client, opened["revision"], 4)
            assert changed.status_code == 200, changed.text
            current = _read_plan(client, root, changed.json(), selected)
        finally:
            release.set()
        old = previous.result(timeout=5)

    assert (old.revision, old.semantic_root) == (opened["revision"], opened["semantic_root"])
    assert (
        old.plan.selected_definitions[selected].payload["graph"]["nodes"]["gain"]["params"]["gain"]
        == 1
    )
    assert (
        current.selected_definitions[selected].payload["graph"]["nodes"]["gain"]["params"]["gain"]
        == 4
    )
    assert _read_plan(client, root, changed.json(), selected) == current


def _plan_dependencies(root: Path) -> None:
    original = load_text_project(root)
    authored = original.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    fit = authored["steps"]["fit"]
    fit["sweep"]["axes"]["seed"]["values"]["10"]["label"] = "First"
    authored["documents"]["selected"] = {"schema": "subset", "payload": {"rows": ["seed=10"]}}
    authored["steps"]["follow"] = {
        "kind": "train",
        "base": {"step": "fit"},
        "sweep": {"axes": {"seed": None}},
        "inputs": {"checkpoint": {"step": "fit", "where": {"subset": "selected"}}},
    }
    independent = dict(fit["base"]["inline"], seed=88)
    authored["steps"]["independent"] = {"kind": "train", "base": {"inline": independent}}
    project = Project.model_validate(authored)
    for relative, data in dump_text_project(project, original.configuration).items():
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)


def test_incremental_plan_matches_fresh_compilation_and_skips_unchanged_work(study, monkeypatch):
    root, client = study
    _plan_dependencies(root)
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
    follow = RunKey("default", "follow", "default")
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
    assert counts["compose_step"] == 3 and counts["encode_request"] == 4
    assert counts["resolve"] >= 4
    warm, counts = check(first)
    assert not counts
    warm.selected_definitions[first].payload["graph"]["nodes"]["gain"]["params"]["gain"] = 999
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

    model = load_text_project(root).project.documents["model"].payload
    snapshot = patch(model, path="/base/inline/graph")
    inlined, counts = check(first)
    assert counts["encode_request"] == 0 and inlined.requests == edited.requests
    assert inlined.sources[first]["/payload/graph/schema_id"].kind == "step"
    assert edited.sources[first]["/payload/graph/schema_id"].kind == "document"

    response = _patch(client, snapshot["revision"], 4)
    assert response.status_code == 200, response.text
    snapshot = response.json()
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

    descriptor = request_identity.REQUEST_KINDS["train"]
    monkeypatch.setitem(
        request_identity.REQUEST_KINDS, "train",
        replace(descriptor, documents=descriptor.documents[:-1]),
    )
    updated_authority, counts = check(first)
    assert counts["encode_request"] == 5 and counts["compose_step"] == 3
    assert set(updated_authority.requests).isdisjoint(reordered.requests)


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
    assert {entry.name for entry in projects.iterdir()} == ids
    reopened = TestClient(create_app(projects_dir=projects))
    for document in documents:
        snapshot = document["snapshot"]
        project_id = snapshot["project_id"]
        folder = projects / project_id
        text = load_text_project(folder)
        assert text.project == Project()
        assert text.configuration.project == project_id
        assert text.store_path == folder / "results"
        assert text.revision == snapshot["revision"]
        assert json.loads((folder / "presentation.json").read_bytes()) == {}
        response = reopened.get(f"/api/projects/{project_id}/open")
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert response.json() == document
        assert reopened.get(f"/api/projects/{project_id}").json() == snapshot


def test_rename_changes_only_the_declaration_and_checks_authored_revision(study):
    root, client = study
    initial = client.get(f"{_URL}/open").json()
    presentation = {"documents": {"model": {"camera": {"x": 1, "y": 2, "zoom": 1.5}}}}
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
    declaration = root / PROJECT_DECLARATION_FILENAME
    original_declaration = json.loads(declaration.read_bytes())
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
    assert json.loads(declaration.read_bytes()) == {**original_declaration, "project": "renamed"}
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
    created = client.post("/api/projects").json()["snapshot"]
    project_id = created["project_id"]
    url = f"/api/projects/{project_id}/files"
    train = {"kind": "step", "id": "train", "step": {"kind": "train", "base": {"inline": {}}}}
    response = client.post(url, json={"revision": created["revision"], "file": train})
    assert response.status_code == 201, response.text
    step_revision = response.json()["snapshot"]
    root = projects / project_id
    step_file = root / "steps/train.json"
    assert json.loads(step_file.read_bytes()) == train["step"]

    task = {"kind": "document", "id": "task", "document": {
        "schema": "feedbax.spec.task", "payload": {"type": "SimpleReaches", "params": {"n_steps": 100}},
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
    assert TestClient(create_app(projects_dir=projects)).get(f"/api/projects/{project_id}/open").json() == document


def test_rename_and_patch_from_one_revision_have_one_winner(study):
    root, client = study
    original = _read(client)
    start = Barrier(2)

    def rename():
        start.wait(timeout=4)
        return client.post(f"{_URL}/rename", json={
            "revision": original["revision"], "name": "renamed",
        })

    def patch():
        start.wait(timeout=4)
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
        assert _gain(root) == 3


def test_named_document_edit_reopens_and_presentation_leaves_authored_identity(study):
    root, client = study
    before = _read(client)
    assert "note" not in before["project"]["steps"]["fit"]
    file_revisions = {entry["address"]["id"]: entry["revision"] for entry in before["file_revisions"]}
    authored_file = root / "documents/model.json"
    authored_file.chmod(0o644)
    response = _patch(client, before["revision"], 3)
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["revision"] != before["revision"]
    assert saved["semantic_root"] != before["semantic_root"]
    current_files = {entry["address"]["id"]: entry["revision"] for entry in saved["file_revisions"]}
    assert current_files["fit"] == file_revisions["fit"]
    assert current_files["model"] != file_revisions["model"]
    assert _gain(root) == 3
    assert authored_file.stat().st_mode & 0o777 == 0o644
    assert _read(TestClient(create_app(projects_dir=root.parent))) == saved
    plan_before = compile_project(load_text_project(root).project)
    assert len(plan_before.runs) == 2 and not plan_before.diagnostics
    authored_bytes = authored_file.read_bytes()

    presentation = {
        "documents": {
            "model": {
                "positions": {"gain": {"x": 40, "y": 20}},
                "camera": {"x": 2, "y": 3, "zoom": 1.5},
            }
        },
        "steps": {"fit": {"sweep_order": ["seed"], "folded_sweep_id": "seed"}},
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
        path="/sweep/axes/seed/values/10/patches/seed/value",
    )
    assert response.status_code == 200, response.text
    reopened = load_text_project(root)
    sweep = reopened.project.steps["fit"].sweep
    assert list(sweep.axes["seed"].values) == ["10", "2"]
    assert list(sweep.exceptions) == ["z_first", "a_second"]
    current = compile_project(reopened.project)
    assert [run.key for run in current.runs] == [run.key for run in previous.runs]
    assert [run.coordinates for run in current.runs] == [run.coordinates for run in previous.runs]
    assert current.runs[0].request_id != previous.runs[0].request_id
    assert current.runs[1].request_id == previous.runs[1].request_id


def test_competing_writers_from_one_revision_have_one_winner(study):
    root, client = study
    revision = _read(client)["revision"]
    start = Barrier(2, timeout=5)

    def save(value):
        writer = TestClient(create_app(projects_dir=root.parent))
        start.wait()
        return value, _patch(writer, revision, value)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(save, [4, 5]))
    assert sorted(response.status_code for response in results.values()) == [200, 409]
    (winner,) = (value for value, response in results.items() if response.status_code == 200)
    (loser,) = (response for response in results.values() if response.status_code == 409)
    assert _gain(root) == winner
    assert loser.json()["detail"]["current"] == load_text_project(root).revision


def test_external_revision_and_invalid_batch_preserve_external_bytes(study):
    root, client = study
    baseline = _read(client)
    path = root / "documents/model.json"
    path.write_bytes(path.read_bytes() + b"\n")
    reformatted = _read(client)
    assert reformatted["revision"] != baseline["revision"]
    assert reformatted["semantic_root"] == baseline["semantic_root"]
    assert _patch(client, baseline["revision"], 3).status_code == 409

    body = json.loads(path.read_bytes())
    body["payload"]["nodes"]["gain"]["params"]["gain"] = 9
    data = json.dumps(body, indent=4).encode()
    write_bytes_atomic(path, data)
    assert _gain(root) == 9
    assert _patch(client, reformatted["revision"], 3).status_code == 409
    latest = _read(client)
    assert latest["semantic_root"] != reformatted["semantic_root"]
    response = client.patch(
        _URL,
        json={
            "revision": latest["revision"],
            "address": {"kind": "document", "id": "model"},
            "patches": [
                {"op": "replace", "path": _GAIN, "value": 3},
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
    path = root / "documents/model.json"
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
    assert _gain(root) == 3


def test_project_and_authored_paths_cannot_point_outside_the_root(study, tmp_path):
    root, client = study
    (tmp_path / "elsewhere").symlink_to(root, target_is_directory=True)
    assert client.get("/api/projects/elsewhere").status_code == 422
    original = root / "documents/model.json"
    outside = tmp_path / "outside.json"
    original.rename(outside)
    original.symlink_to(outside)
    assert client.get(_URL).status_code == 422


def test_existing_event_stream_observes_an_actual_external_file_edit(study, monkeypatch):
    root, client = study
    before = load_text_project(root).revision
    monkeypatch.setattr(jobs, "_FOLDER_OBSERVATION_INTERVAL", 0)

    class JobsStream:
        calls = 0

        def events(self, *, after, projects, wait, machine):
            self.calls += 1
            if self.calls == 2:
                path = root / "documents/model.json"
                document = json.loads(path.read_bytes())
                document["payload"]["nodes"]["gain"]["params"]["gain"] = 11
                write_bytes_atomic(path, json.dumps(document).encode())
            if self.calls > 2:
                raise HostError("unavailable", "Stream ended")
            return 0, StudioJobEventHistory(cursor=0, oldest_cursor=0, events=[])

    client.app.state.job_service = JobsStream()
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
    assert _gain(root) == 11
