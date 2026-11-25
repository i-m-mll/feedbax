"""Native model archives preserve the fitted graph and replica bytes."""

from io import BytesIO
import json
from shutil import rmtree
from zipfile import ZipFile

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.application.bundle import new_application_registry_bundle
from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.loaded_model import (
    ImportedModelSource, ModelFitReport, ModelSlotFitProblem, decode_imported_model_document,
)
from feedbax.contracts.model_checkpoint import ModelReplicaManifest
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.task_document import TaskDocument
from feedbax.examples import load_shipped_projects
from feedbax.execution.trial_resolution import PreparedTrialExecution, compile_trial_execution
from feedbax.host.model_archive import UnreadableModelArchive, export_model_archive, stage_model_archive
from feedbax.host.model_checkpoint import SemanticModelSlot, SemanticModelWriter
from feedbax.persistence.array_container import decode_array_container, encode_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.model_tree import MODEL_CHECKPOINT_STORE


@pytest.fixture(scope="module")
def native_archive(tmp_path_factory):
    project = load_shipped_projects()["point-mass-reaching"]
    model = ModelDocument.model_validate(project.documents["point-mass"].payload)
    task = TaskDocument.model_validate(project.documents["simple-reaches-task"].payload)
    registry = new_application_registry_bundle()
    trial = compile_trial_execution(
        model, task, registry.components, registry.expression_functions, python_tasks=registry.python_tasks,
    )
    assert isinstance(trial, PreparedTrialExecution), trial
    template = trial.executable.graph
    other = eqx.tree_at(lambda graph: graph.nodes["readout"].layer.weight, template,
                        jnp.full_like(template.nodes["readout"].layer.weight, 0.25))
    transaction = tmp_path_factory.mktemp("native-train")
    writer = SemanticModelWriter.for_graph(trial.executable.resolved.graph, None, registry.components)
    assert writer is not None
    checkpoint = writer.publish(transaction, (template, other), templates=(template, template), replica_ids=("a", "b"))
    store = ImmutableArtifactBlobProvider(transaction / MODEL_CHECKPOINT_STORE)
    archive = export_model_archive(store, checkpoint.tree, checkpoint.encoding)
    assert isinstance(archive, bytes), archive
    rmtree(transaction)
    return model, registry.components, template, other, archive, checkpoint


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


def test_archive_import_creates_a_named_document_at_the_guarded_revision(native_archive, tmp_path):
    from fastapi.testclient import TestClient

    from feedbax.contracts.project import LoadedModelDocument
    from feedbax.web.app import create_app

    _, _, _, _, archive, checkpoint = native_archive
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as client:
        created = client.post("/api/projects", json={"name": "load", "example": "point-mass-reaching"})
        assert created.status_code == 201, created.text
        prior = created.json()["snapshot"]
        url = f"/api/projects/{prior['project_id']}/loaded-models/import"
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
        assert payload.source.weights == checkpoint.tree
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
