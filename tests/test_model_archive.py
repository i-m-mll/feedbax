"""Native model archives preserve the fitted graph and replica bytes."""

from io import BytesIO
import json
from shutil import rmtree
from typing import Annotated
from zipfile import ZipFile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import BaseModel

from feedbax.analysis.operations import DatasetOutput, OperationDeclaration
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.effective_success import BoundMember
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.loaded_model import (
    ImportedModelSource, ModelFitReport, ModelSlotFitProblem, decode_imported_model_document,
)
from feedbax.contracts.model_checkpoint import ModelReplicaManifest
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.model_input import InitialModelSource, document_input_object, encode_document_input
from feedbax.contracts.operation import DatasetPort, JaxOperation, ModelPort
from feedbax.contracts.stored_model import StoredModelDocument
from feedbax.contracts.task_document import StoredTaskDocument
from feedbax.dataset import Dataset, Variable
from feedbax.examples import load_shipped_projects
from feedbax.execution.trial_resolution import PreparedTrialExecution, compile_trial_execution
from feedbax.host.model_archive import UnreadableModelArchive, export_model_archive, stage_model_archive
from feedbax.host.model_checkpoint import ModelCheckpointError, SemanticModelSlot, SemanticModelWriter
from feedbax.host.model_input import ImportedModelWeights, ReconstructedModel, reconstruct_model
from feedbax.host.operation_runtime import run_operation
from feedbax.persistence.array_container import decode_array_container, encode_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.model_tree import MODEL_CHECKPOINT_STORE
from feedbax.plugins.declarations import DistributionSource
from feedbax.runtime.graph import Graph


@pytest.fixture(scope="module")
def native_archive(tmp_path_factory):
    project = load_shipped_projects()["point-mass-reaching"]
    model = StoredModelDocument.model_validate(project.documents["point-mass"].payload).lower()
    task = StoredTaskDocument.model_validate(project.documents["simple-reaches-task"].payload).lower()
    registry = new_application_registry_bundle()
    return model, registry.components, *_publish_archive(model, task, registry, tmp_path_factory.mktemp("native-train"))


def _publish_archive(model, task, registry, transaction):
    trial = compile_trial_execution(
        model, task, registry.components, registry.expression_functions, python_tasks=registry.python_tasks,
        input_bindings={"target_pos": "target_pos"},
    )
    assert isinstance(trial, PreparedTrialExecution), trial
    template = trial.executable.graph
    other = eqx.tree_at(lambda graph: graph.nodes["readout"].layer.weight, template,
                        jnp.full_like(template.nodes["readout"].layer.weight, 0.25))
    writer = SemanticModelWriter.for_graph(trial.executable.resolved.graph, None, registry.components)
    assert writer is not None
    checkpoint = writer.publish(transaction, (template, other), templates=(template, template), replica_ids=("a", "b"))
    store = ImmutableArtifactBlobProvider(transaction / MODEL_CHECKPOINT_STORE)
    archive = export_model_archive(store, checkpoint.tree, checkpoint.encoding)
    assert isinstance(archive, bytes), archive
    rmtree(transaction)
    return template, other, archive, checkpoint


def _delay_archive(root, *, x64):
    project = load_shipped_projects()["point-mass-reaching"]
    raw = project.documents["point-mass"].payload
    raw["interior"]["wires"]["command"]["attachments"] = {
        "lag": {"kind": "delay", "delay": {"value": 2, "unit": "st"}},
    }
    model = StoredModelDocument.model_validate(raw).lower()
    task = StoredTaskDocument.model_validate(project.documents["simple-reaches-task"].payload).lower()
    registry = new_application_registry_bundle()
    with jax.enable_x64(x64):
        return model, registry, *_publish_archive(model, task, registry, root)


@pytest.fixture(scope="module")
def float32_delay_archive(tmp_path_factory):
    return _delay_archive(tmp_path_factory.mktemp("float32-delay-archive"), x64=False)


class _ReadoutSettings(BaseModel):
    pass


def _readout(
    model: Annotated[Graph, ModelPort()], params: _ReadoutSettings,
) -> Annotated[Dataset, DatasetOutput("weights", DatasetPort(schema_ids=("test.readout.v1",)))]:
    assert model.input_arrays["target_pos"] == ((2,), "float64")
    assert any((wire.recurrent_initializer or {}).get("dtype") == "float64" for wire in model.wires)
    return Dataset.from_arrays({
        "kernel": Variable(np.asarray(model.nodes["readout"].layer.weight), ("output", "hidden")),
    }, schema_id="test.readout.v1")


def test_float32_archive_restores_exactly_in_an_x64_model_operation(float32_delay_archive, tmp_path):
    model, registry, first, second, raw, checkpoint = float32_delay_archive
    store = ImmutableArtifactBlobProvider(tmp_path / "loaded")
    assert first.input_arrays["target_pos"] == ((2,), "float32")
    graph = GraphSpec.model_validate_json(_read_files(raw)["graph.json"])
    assert any((wire.recurrent_initializer or {}).get("dtype") == "float32" for wire in graph.wires)
    registry.operations.register_declaration(
        _readout, OperationDeclaration(
            settings_model=_ReadoutSettings, numerical=JaxOperation(backend="cpu", floating_point_bits=64),
        ), key="test.readout", source=DistributionSource("test", "test", "1"),
    )
    with jax.enable_x64(True), stage_model_archive(raw, model, store, components=registry.components) as imported:
        assert isinstance(imported, ImportedModelSource), imported
        assert (imported.weights, imported.encoding) == (checkpoint.tree, checkpoint.encoding)
        content, reference = encode_document_input(imported)
        store.store_object(reference, expected=document_input_object(content))
        result = run_operation(
            registry.operations.resolve("test.readout"), _ReadoutSettings(),
            {"model": (BoundMember(key="model", content=content),)}, {}, registry=registry, artifacts=store,
        )
        expected = np.stack([first.nodes["readout"].layer.weight, second.nodes["readout"].layer.weight])
        assert expected.dtype == np.float32
        assert result.data("kernel").dtype == np.float64
        np.testing.assert_array_equal(result.data("kernel"), expected.astype(np.float64))
        assert export_model_archive(store, imported.weights, imported.encoding) == raw

        native = reconstruct_model(InitialModelSource(document=model), registry=registry, artifacts=store)
        assert isinstance(native, ReconstructedModel), native
        slot = SemanticModelSlot.open_store(store, imported.weights, imported.encoding, components=registry.components)
        with pytest.raises(ModelCheckpointError, match="dtype float64, found .* dtype float32"):
            slot.restore(native.graph)


@pytest.mark.parametrize("mismatch", ["topology", "slot-shape", "input-shape"])
def test_x64_import_validates_the_original_float32_archive(float32_delay_archive, tmp_path, mismatch):
    model, registry, _, _, raw, _ = float32_delay_archive
    files = _read_files(raw)
    manifest = ModelReplicaManifest.model_validate_json(files["model.json"])
    address = "/v1/model/value/readout/kernel"
    if mismatch == "topology":
        graph = GraphSpec.model_validate_json(files["graph.json"])
        graph.output_bindings["command"] = graph.output_bindings["activity"]
        files["graph.json"] = canonical_json_v2_bytes(graph.model_dump(mode="json"))
        address = "/graph/output_bindings/command"
    elif mismatch == "slot-shape":
        first = manifest.replicas[0]
        path = "replicas/0.safetensors"
        decoded = decode_array_container(files[path], expected_descriptor_sha256=first.array_descriptor_sha256)
        encoded = encode_array_container(
            dict(decoded.tensors) | {address: decoded.tensors[address].flatten()}, metadata=decoded.descriptor.metadata,
        )
        files[path] = encoded.data
        files["model.json"] = canonical_json_v2_bytes(manifest.model_copy(update={"replicas": (
            first.model_copy(update={"array_descriptor_sha256": encoded.descriptor_sha256}), manifest.replicas[1],
        )}).model_dump(mode="json"))
    else:
        input_arrays = dict(manifest.input_arrays)
        input_arrays["target_pos"] = input_arrays["target_pos"].model_copy(update={"shape": (9,)})
        files["model.json"] = canonical_json_v2_bytes(manifest.model_copy(update={
            "input_arrays": input_arrays,
        }).model_dump(mode="json"))
        address = "/model/input_arrays/target_pos"
    store = ImmutableArtifactBlobProvider(tmp_path / "rejected")
    with jax.enable_x64(True), stage_model_archive(
        _as_archive(files), model, store, components=registry.components,
    ) as rejected:
        assert isinstance(rejected, ModelFitReport), rejected
        assert any(problem.address == address for problem in rejected.problems), rejected
    assert not store.root.exists()


def test_float64_archive_cannot_narrow_to_float32(tmp_path):
    model, registry, _, _, raw, _ = _delay_archive(tmp_path / "float64-delay-archive", x64=True)
    store = ImmutableArtifactBlobProvider(tmp_path / "rejected")
    with jax.enable_x64(False), stage_model_archive(raw, model, store, components=registry.components) as rejected:
        assert isinstance(rejected, ModelFitReport), rejected
        assert any(
            problem.kind == "mismatched_slot" and "dtype float32" in problem.detail and "dtype float64" in problem.detail
            for problem in rejected.problems
        ), rejected
    assert not store.root.exists()


def _read_files(raw):
    with ZipFile(BytesIO(raw)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _as_archive(files):
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_native_archive_preserves_replicas_in_a_new_store(native_archive, tmp_path):
    model, components, template, other, raw, checkpoint = native_archive
    store = ImmutableArtifactBlobProvider(tmp_path / "loaded")
    with stage_model_archive(raw, model, store, components=components) as result:
        assert isinstance(result, ImportedModelSource), result
        assert result.weights == checkpoint.tree and result.encoding == checkpoint.encoding
        pinned = decode_imported_model_document(store.get_object(result.model))
        assert pinned.scientific_projection() == model.scientific_projection()
        store.register_root(ContentRootRecord(
            root_id="archive-test", root_kind="import",
            targets=(
                ContentRetentionTarget(ref_kind="object", object=result.model),
                ContentRetentionTarget(ref_kind="tree", tree=result.weights),
            ),
        ))
    slot = SemanticModelSlot.open_store(store, result.weights, result.encoding, components=components)
    assert slot.replica_ids == ("a", "b")
    first, second = slot.restore_all((template, template))
    assert eqx.tree_equal(first, template) and eqx.tree_equal(second, other)
    assert export_model_archive(store, result.weights, result.encoding) == raw


def test_archive_fits_changed_noise_without_changing_learned_weights(tmp_path):
    project = load_shipped_projects()["point-mass-reaching"]
    raw = project.documents["point-mass"].payload
    raw["interior"]["components"]["motor-noise"] = {
        "kind": "leaf", "label": "Motor noise", "type": "feedbax.Noise",
        "fields": {"std": 0.5, "shape": [2]},
    }
    task = StoredTaskDocument.model_validate(project.documents["simple-reaches-task"].payload).lower()
    registry = new_application_registry_bundle()
    trained = StoredModelDocument.model_validate(raw).lower()
    first, second, archive, checkpoint = _publish_archive(trained, task, registry, tmp_path / "trained")
    noise = raw["interior"]["components"]["motor-noise"]["fields"]
    noise["std"] = 0.0
    selected = StoredModelDocument.model_validate(raw).lower()

    store = ImmutableArtifactBlobProvider(tmp_path / "loaded")
    with stage_model_archive(archive, selected, store, components=registry.components) as imported:
        assert isinstance(imported, ImportedModelSource), imported
        assert (imported.weights, imported.encoding) == (checkpoint.tree, checkpoint.encoding)
        pinned = decode_imported_model_document(store.get_object(imported.model))
        model = reconstruct_model(
            ImportedModelWeights(
                initial=InitialModelSource(document=pinned), tree=imported.weights, encoding=imported.encoding,
            ),
            registry=registry, artifacts=store,
        )
        assert isinstance(model, ReconstructedModel), model
        for original, restored in zip((first, second), model.graphs, strict=True):
            assert restored.nodes["motor-noise"].std == 0.0
            np.testing.assert_array_equal(restored.nodes["readout"].layer.weight, original.nodes["readout"].layer.weight)
        assert export_model_archive(store, imported.weights, imported.encoding) == archive

    noise["shape"] = [3]
    with stage_model_archive(archive, StoredModelDocument.model_validate(raw).lower(), store, components=registry.components) as rejected:
        assert isinstance(rejected, ModelFitReport), rejected
        assert any(
            problem.kind == "graph_mismatch" and problem.address.startswith("/graph/nodes/motor-noise/params/shape")
            for problem in rejected.problems
        )


def test_a_loaded_models_replicas_are_planned_rendered_and_retained(native_archive, host, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from feedbax.contracts.project import Project
    from feedbax.web.app import create_app
    from feedbax.web.services.project_folder import ProjectFolderService
    from tests.support.host import job_row, token, wait_until
    from tests.test_canonical_evaluate_admission import _save_project
    from tests.test_project_report_plan import _report_project

    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    _, _, _, _, archive, _ = native_archive
    raw = _report_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    example = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    task = example["documents"]["simple-reaches-task"]
    task["payload"]["fixed_steps"] = 8
    task["payload"]["epochs"]["move"]["end"]["length"]["value"] = 8
    del task["payload"]["validation_trials"]
    figure = raw["documents"]["velocity-figure"]
    (layer,) = figure["payload"]["layers"]
    del layer["select"]
    layer.update(group=["replicate"], style={"label": "{replicate}"})
    raw["documents"] = {
        "point-mass": example["documents"]["point-mass"], "task": task,
        "velocity-analysis": raw["documents"]["velocity-analysis"], "velocity-figure": figure,
    }
    raw["steps"] = {
        "analyze": raw["steps"]["analyze"],
        "baseline": {
            "label": "Baseline",
            "kind": "evaluate",
            "base": {"inline": {"evaluation_type": "feedbax.rollout", "params": {
                "seed": 17, "trial_policy": "paired", "variation": [],
                "trial_set": {"kind": "sample", "count": 2},
            }}},
            "inputs": {
                "subject": {"kind": "loaded_model", "source": {"document": "ensemble"}},
                "task": {"kind": "task", "source": {"document": "task"}},
            },
        },
    }
    _save_project(ProjectFolderService(tmp_path), "replicated", Project.model_validate(raw))
    path = "/api/projects/replicated"

    def expect(response, status=200):
        assert response.status_code == status, response.text
        return response.json()

    def key(step):
        return {"variant": "default", "step": step, "row": "default"}

    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        revision = expect(client.get(path))["revision"]
        imported = expect(client.post(f"{path}/loaded-models/import", params={
            "revision": revision, "name": "ensemble", "model": "point-mass",
        }, content=archive, headers={"Content-Type": "application/octet-stream"}), 201)
        revision = imported["snapshot"]["revision"]
        catalog = expect(client.post(f"{path}/authoring/analyze", json={"revision": revision, "key": key("analyze")}))
        outputs = catalog["operation_outputs"]
        for name, role, variable, dims in (
            ("align", "velocity", "parallel_velocity", ["replicate", "condition", "trial", "state_time"]),
            ("summarize", "summary", "mean", ["replicate", "condition", "state_time"]),
        ):
            port = outputs[name][role]
            assert port["variables"][variable]["dims"] == dims
            assert port["coordinates"]["replicate"]["dims"] == ["replicate"]
            assert port["coordinates"]["replicate"]["inline_values"] == ["a", "b"]

        started = expect(client.post(f"{path}/train-runs", json={
            "revision": revision, "token": token("replicated-analysis"),
            "run_keys": [key("baseline"), key("analyze")],
        }), 202)["items"]
        assert len(started) == 4 and all(item["disposition"] == "admitted" for item in started), started
        for item in started:
            terminal = wait_until(lambda: (
                job if (job := job_row(host, item["job_id"])) and job["outcome"] is not None else None
            ))
            assert terminal["state"] == "succeeded", terminal["outcome"]

        window = expect(client.get(f"{path}/train-runs", params={
            "revision": revision, "window_map": True,
        }))["window_map"]
        assert not window["problems"], window["problems"]
        products = {element["name"]: element["requests"][0]["result"]["products"] for element in window["analysis_elements"]}
        summary = expect(client.get(products["summarize"]["summary"]))
        assert summary["variables"]["mean"]["dims"] == outputs["summarize"]["summary"]["variables"]["mean"]["dims"]
        assert summary["coords"]["replicate"]["inline_values"] == ["a", "b"]
        rendered = expect(client.get(products["profile"]["figure"]))
        assert [trace["name"] for trace in rendered["data"] if trace.get("showlegend", True)] == ["a", "b"]


def test_archive_import_creates_a_named_document_at_the_guarded_revision(native_archive, tmp_path):
    from fastapi.testclient import TestClient

    from feedbax.contracts.project import LoadedModelDocument
    from feedbax.web.app import create_app

    _, _, _, _, archive, checkpoint = native_archive
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as client:
        created = client.post("/api/projects", json={"name": "load", "example": "point-mass-reaching"})
        assert created.status_code == 201, created.text
        project = created.json()["snapshot"]["project_id"]
        folders = client.app.state.project_folders
        root = folders.read_authored_snapshot(project, revision=None).root
        model_file = root / "documents" / "point-mass.json"
        authored = json.loads(model_file.read_bytes())
        authored["payload"]["ports"]["inputs"]["target_pos"]["label"] = "Requested target"
        authored["payload"]["ports"]["outputs"]["effector"]["label"] = "Point mass state"
        model_file.write_text(json.dumps(authored))
        prior = client.get(f"/api/projects/{project}").json()
        url = f"/api/projects/{project}/loaded-models/import"
        query = {"name": "imported", "model": "point-mass", "revision": prior["revision"]}
        headers = {"Content-Type": "application/octet-stream"}
        rejected = client.post(url, params=query, content=b"not an archive", headers=headers)
        assert rejected.status_code == 422, rejected.text
        assert rejected.json()["status"] == "unreadable_archive"
        assert client.get(f"/api/projects/{prior['project_id']}").json()["revision"] == prior["revision"]
        imported = client.post(url, params=query, content=archive, headers=headers)
        assert imported.status_code == 201, imported.text
        saved = imported.json()
        assert saved["document"] == "imported"
        payload = LoadedModelDocument.model_validate(saved["snapshot"]["project"]["documents"]["imported"]["payload"])
        assert payload.lower().weights == checkpoint.tree
        assert payload.model.document == "point-mass"
        assert client.post(url, params=query, content=archive, headers=headers).status_code == 409


@pytest.mark.parametrize("unsupported_owner", [False, True])
def test_fit_reports_missing_extra_and_wrong_shape_across_replicas(native_archive, tmp_path, unsupported_owner):
    model, components, _, _, raw, _ = native_archive
    files = _read_files(raw)
    manifest = ModelReplicaManifest.model_validate_json(files["model.json"])
    descriptors = []
    missing = "/v1/model/value/readout/bias"
    mismatch = "/v1/model/value/readout/kernel"
    extra = "/v1/model/value/readout/extra"
    for index, replica in enumerate(manifest.replicas):
        path = f"replicas/{index}.safetensors"
        source = decode_array_container(files[path], expected_descriptor_sha256=replica.array_descriptor_sha256)
        arrays = dict(source.tensors)
        metadata = dict(source.descriptor.metadata)
        if index == 0:
            del arrays[missing]
            del metadata[missing]
            arrays[extra] = np.ones((1,), dtype=np.float32)
            metadata[extra] = metadata[mismatch]
        else:
            arrays[mismatch] = arrays[mismatch].flatten()
        encoded = encode_array_container(arrays, metadata=metadata)
        files[path] = encoded.data
        descriptors.append(replica.model_copy(update={"array_descriptor_sha256": encoded.descriptor_sha256}))
    files["model.json"] = canonical_json_v2_bytes(
        manifest.model_copy(update={"replicas": tuple(descriptors)}).model_dump(mode="json"),
    )
    if unsupported_owner:
        authored = model.model_dump(mode="json")
        authored["interior"]["components"]["constant"] = {
            "kind": "leaf", "type": "feedbax.Constant", "name": "Constant",
        }
        model = ModelDocument.model_validate(authored)
    store = ImmutableArtifactBlobProvider(tmp_path / "rejected")
    with stage_model_archive(_as_archive(files), model, store, components=components) as result:
        assert isinstance(result, ModelFitReport), result
        assert [
            (problem.kind, problem.replica, problem.address)
            for problem in result.problems if isinstance(problem, ModelSlotFitProblem)
        ] == [
            ("missing_slot", "a", missing),
            ("unexpected_slot", "a", extra),
            ("mismatched_slot", "b", mismatch),
        ]
        if unsupported_owner:
            assert [(problem.kind, problem.address) for problem in result.problems[:2]] == [
                ("unsupported_owner", "/graph/nodes/constant"),
                ("graph_mismatch", "/graph/nodes/constant"),
            ]
        else:
            assert len(result.problems) == 3
    assert not store.root.exists()


def test_fit_refuses_a_different_wiring_and_names_the_graph_field(native_archive, tmp_path):
    model, components, _, _, raw, _ = native_archive
    files = _read_files(raw)
    graph = GraphSpec.model_validate_json(files["graph.json"])
    files["graph.json"] = canonical_json_v2_bytes(graph.model_copy(update={
        "output_bindings": {**graph.output_bindings, "command": graph.output_bindings["activity"]},
    }).model_dump(mode="json"))
    store = ImmutableArtifactBlobProvider(tmp_path / "rejected")
    with stage_model_archive(_as_archive(files), model, store, components=components) as result:
        assert isinstance(result, ModelFitReport), result
        problem, = result.problems
        assert (problem.kind, problem.address) == ("graph_mismatch", "/graph/output_bindings/command")
    assert not store.root.exists()


def test_archive_refuses_a_changed_tensor_with_an_unchanged_descriptor(native_archive, tmp_path):
    model, components, _, _, raw, _ = native_archive
    files = _read_files(raw)
    path = "replicas/1.safetensors"
    files[path] = files[path][:-1] + bytes([files[path][-1] ^ 1])
    store = ImmutableArtifactBlobProvider(tmp_path / "rejected")
    with stage_model_archive(_as_archive(files), model, store, components=components) as result:
        assert isinstance(result, UnreadableModelArchive), result
        assert result.path == "/" + path
        assert "digest mismatch" in result.detail
    assert not store.root.exists()


def test_archive_refuses_a_graph_outside_native_json_identity(native_archive, tmp_path):
    model, components, _, _, raw, _ = native_archive
    files = _read_files(raw)
    graph = json.loads(files["graph.json"])
    graph["nodes"]["readout"]["params"]["output_size"] = 2**53
    files["graph.json"] = json.dumps(graph).encode()
    store = ImmutableArtifactBlobProvider(tmp_path / "rejected")
    with stage_model_archive(_as_archive(files), model, store, components=components) as result:
        assert isinstance(result, UnreadableModelArchive), result
        assert result.path == "/graph.json"
        assert "unsafe_integer" in result.detail
    assert not store.root.exists()
