"""A copied project carries its pinned Model and retains exact executable archive weights."""

from copy import deepcopy
from io import BytesIO
from shutil import copytree, ignore_patterns, rmtree
from zipfile import ZipFile

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.loaded_model import encode_imported_model_document
from feedbax.contracts.project import LoadedModelDocument, Project
from feedbax.contracts.request_identity import decode_request
from feedbax.examples import load_shipped_projects
from feedbax.host.model_archive import export_model_archive
from feedbax.host.model_checkpoint import SemanticModelWriter
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.content_retention import read_active_roots
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.model_tree import MODEL_CHECKPOINT_STORE
from feedbax.web.app import create_app
from tests.support.host import job_row, token, wait_until
from tests.support.managed_train_fixture import managed_evaluation_step
from tests.test_canonical_evaluate_admission import _save_project
from tests import test_model_archive

archive_bundle = test_model_archive.native_archive


def _seed(folders, name):
    raw = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    train = raw["steps"]["simple-reaches"]
    subject = {"kind": "loaded_model", "source": {"document": "portable"}}
    train["inputs"] = {"checkpoint": subject}
    payload = train["base"]["inline"]
    payload["checkpoint_initialization"] = "initialize_from"
    payload["training_config"] = {"n_batches": 1, "batch_size": 2, "replicates": 2}
    task = raw["documents"]["simple-reaches-task"]["payload"]
    task["fixed_steps"] = 8
    task["epochs"][0]["end"]["length"]["value"] = 8
    task["validation_trials"] = task["validation_trials"][:1]
    evaluate = managed_evaluation_step("unused")
    evaluate["inputs"] = {
        "subject": subject,
        "task": {"kind": "task", "source": payload["trial_documents"]["task"]},
    }
    _save_project(folders, name, Project.model_validate({"documents": raw["documents"], "steps": {
        "evaluate": evaluate, "train": train,
    }}))


def _expect(response, status=200):
    assert response.status_code == status, response.text
    return response.json()


def _load(client, project, raw, *, name="portable", revision=None):
    url = f"/api/projects/{project}"
    revision = revision or _expect(client.get(url))["revision"]
    return client.post(f"{url}/loaded-models/import", params={
        "revision": revision, "name": name, "model": "point-mass",
    }, content=raw, headers={"Content-Type": "application/octet-stream"})


def _loaded(imported, name="portable"):
    return LoadedModelDocument.model_validate(imported["snapshot"]["project"]["documents"][name]["payload"])


def _clone(projects, name):
    return copytree(projects / "original", projects / name, ignore=ignore_patterns("results", ".studio-write.lock"))


def _check_plan(folders, name):
    settled = folders.read_plan(name, revision=None)
    plan = settled.plan
    assert not plan.diagnostics, plan.diagnostics
    assert {row.key.step for row in plan.runs} == {"evaluate", "train"}
    assert all(not row.problems for row in plan.runs), [(row.key, row.problems) for row in plan.runs]
    return settled


def _root_id(loaded):
    source = loaded.source
    return f"loaded-model-import:{source.model.sha256}:{source.weights.descriptor.sha256}"


def _different_archive(transaction, archive, graph, components):
    with ZipFile(BytesIO(archive)) as zip_file:
        spec = GraphSpec.model_validate_json(zip_file.read("graph.json"))
    writer = SemanticModelWriter.for_graph(spec, None, components)
    assert writer is not None
    checkpoint = writer.publish(transaction, (graph,), templates=(graph,), replica_ids=("a",))
    changed = export_model_archive(
        ImmutableArtifactBlobProvider(transaction / MODEL_CHECKPOINT_STORE), checkpoint.tree, checkpoint.encoding,
    )
    assert isinstance(changed, bytes) and changed != archive
    return changed


def test_copied_project_runs_loaded_evaluate_and_train_from_exact_portable_assets(host, tmp_path, monkeypatch, archive_bundle):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1, light=1)
    model, _, _, _, archive, checkpoint = archive_bundle
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders
        _seed(folders, "original")
        loaded = _loaded(_expect(_load(client, "original", archive), 201))
        assert loaded.archive == "models/portable.feedbax-model"
        assert loaded.source.weights == checkpoint.tree and loaded.source.encoding == checkpoint.encoding
        projection = f"models/{loaded.source.model.sha256}.model.json"
        original = tmp_path / "original"
        assert (original / loaded.archive).read_bytes() == archive
        assert (original / projection).read_bytes() == encode_imported_model_document(model)
        warm = _check_plan(folders, "original")

        cold = _clone(tmp_path, "clone")
        rmtree(original)
        assert not (cold / "results").exists()
        rebuilt = _check_plan(folders, "clone")
        assert warm.plan.requests == rebuilt.plan.requests and warm.plan.documents == rebuilt.plan.documents
        store = ImmutableArtifactBlobProvider(rebuilt.store_path)
        assert store.get_object(loaded.source.model) == (cold / projection).read_bytes()
        assert export_model_archive(store, loaded.source.weights, loaded.source.encoding) == archive
        assert len(read_active_roots(store, (_root_id(loaded),))) == 1

        decoded = {row.key.step: decode_request(rebuilt.plan.requests[row.request_id], rebuilt.plan.documents.__getitem__)
                   for row in rebuilt.plan.runs}
        assert decoded["evaluate"].inputs["subject"] == decoded["train"].inputs["checkpoint"]
        started = _expect(client.post("/api/projects/clone/train-runs", json={
            "revision": rebuilt.revision,
            "token": token("portable-fresh-clone"),
            "run_keys": [{"variant": "default", "step": step, "row": "default"} for step in ("evaluate", "train")],
        }), 202)["items"]
        assert [item["disposition"] for item in started] == ["admitted", "admitted"]
        for item in started:
            def complete():
                job = job_row(host, item["job_id"])
                return job if job is not None and job["outcome"] is not None else None

            job = wait_until(complete)
            assert job["state"] == "succeeded", job["outcome"]
            outcome = ContentObjectRef.model_validate(job["outcome"]["store_success"])
            receipt = SuccessView(store).read_success(outcome).receipt
            assert receipt.request.id in rebuilt.plan.requests
        (cold / loaded.archive).unlink()
        (cold / projection).unlink()
        assert _check_plan(folders, "clone").plan.requests == warm.plan.requests


def test_same_name_load_reuses_pins_and_repairs_missing_assets_at_the_checked_revision(tmp_path, archive_bundle):
    model, components, first, _, archive, _ = archive_bundle
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders
        _seed(folders, "original")
        imported = _expect(_load(client, "original", archive), 201)
        loaded = _loaded(imported)
        revision = imported["snapshot"]["revision"]
        assert _load(client, "original", archive, revision=revision).json() == imported
        other = _loaded(_expect(_load(client, "original", archive, name="equal"), 201), "equal")
        assert other.source == loaded.source and other.archive != loaded.archive
        root = tmp_path / "original"
        assert len(list((root / "models").glob("*.model.json"))) == 1
        assert _load(client, "original", archive, revision=revision).status_code == 409
        current = _expect(client.get("/api/projects/original"))["revision"]
        projection = next((root / "models").glob("*.model.json"))
        projection.unlink()
        rmtree(root / "results")
        restored = _expect(_load(client, "original", archive, revision=current), 201)
        assert _loaded(restored) == loaded and projection.read_bytes() == encode_imported_model_document(model)

        authored = deepcopy(folders.read("original").project)
        authored.documents["point-mass"].payload["step_seconds"] = 0.02
        _save_project(folders, "original", authored)
        latest = _expect(client.get("/api/projects/original"))["revision"]
        store = ImmutableArtifactBlobProvider(root / "results")
        changed = _different_archive(tmp_path / "other", archive, first, components)
        assert _load(client, "original", changed, revision=latest).status_code == 409

        asset = root / loaded.archive
        asset.write_bytes(b"different user bytes")
        assert _load(client, "original", archive, revision=latest).status_code == 409
        assert asset.read_bytes() == b"different user bytes"
        asset.unlink()
        rmtree(store.root)
        repaired = _expect(_load(client, "original", archive, revision=latest), 201)
        assert repaired["snapshot"]["revision"] == latest
        assert _loaded(repaired) == loaded and asset.read_bytes() == archive
        assert store.get_object(loaded.source.model) == encode_imported_model_document(model)
        assert len(read_active_roots(store, (_root_id(loaded),))) == 1


def test_cold_plan_checks_assets_before_publishing_any_store_bytes(tmp_path, archive_bundle):
    _, components, first, _, archive, _ = archive_bundle
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders
        _seed(folders, "original")
        loaded = _loaded(_expect(_load(client, "original", archive), 201))
        warm = _check_plan(folders, "original")
        projection = f"models/{loaded.source.model.sha256}.model.json"
        with pytest.raises(ValidationError):
            LoadedModelDocument.model_validate(loaded.model_dump(mode="json") | {"archive": "models/../portable.feedbax-model"})

        def refused(name, phrase):
            plan = folders.read_plan(name, revision=None).plan
            assert not plan.runs
            assert {item.path for item in plan.diagnostics} == {"inputs.subject", "inputs.checkpoint"}
            assert all(phrase in item.detail for item in plan.diagnostics), plan.diagnostics
            assert not (tmp_path / name / "results").exists()

        swapped = _clone(tmp_path, "swapped")
        (swapped / loaded.archive).write_bytes(_different_archive(tmp_path / "other", archive, first, components))
        refused("swapped", "pinned weights or encoding")
        (swapped / loaded.archive).write_bytes(archive)
        assert _check_plan(folders, "swapped").plan.requests == warm.plan.requests

        edited = _clone(tmp_path, "edited")
        (edited / projection).write_bytes(b"{}")
        refused("edited", "pinned Model revision")

        linked = _clone(tmp_path, "linked")
        (linked / loaded.archive).unlink()
        (linked / loaded.archive).symlink_to(tmp_path / "original" / loaded.archive)
        refused("linked", "symlink")

        linked_dir = _clone(tmp_path, "linked-dir")
        rmtree(linked_dir / "models")
        (linked_dir / "models").symlink_to(tmp_path / "original" / "models", target_is_directory=True)
        refused("linked-dir", "symlink")


def test_locator_survives_rename_and_legacy_reads_only_matching_navigation(tmp_path, archive_bundle):
    _, _, _, _, archive, _ = archive_bundle
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders
        _seed(folders, "original")
        imported = _expect(_load(client, "original", archive), 201)
        loaded = _loaded(imported)
        requests = _check_plan(folders, "original").plan.requests
        authored = deepcopy(folders.read("original").project)
        _clone(tmp_path, "renamed")
        renamed = authored.model_dump(mode="json", by_alias=True, exclude_unset=True)
        renamed["documents"]["different-name"] = renamed["documents"].pop("portable")
        for role, step in (("subject", "evaluate"), ("checkpoint", "train")):
            renamed["steps"][step]["inputs"][role]["source"]["document"] = "different-name"
        (tmp_path / "renamed" / "documents" / "portable.json").unlink()
        # Keep the declaration's name while writing the edited project.
        (tmp_path / "renamed" / "original.feedbax").unlink()
        _save_project(folders, "renamed", Project.model_validate(renamed))
        assert _check_plan(folders, "renamed").plan.requests == requests
        assert (tmp_path / "renamed" / loaded.archive).is_file()

        legacy = _clone(tmp_path, "legacy")
        (legacy / f"models/{loaded.source.model.sha256}.model.json").unlink()
        (legacy / "original.feedbax").unlink()
        legacy_raw = authored.model_dump(mode="json", by_alias=True, exclude_unset=True)
        legacy_raw["documents"]["portable"]["payload"].pop("archive")
        _save_project(folders, "legacy", Project.model_validate(legacy_raw))
        assert _check_plan(folders, "legacy").plan.requests == requests

        legacy_raw["documents"]["point-mass"]["payload"]["step_seconds"] = 0.02
        _save_project(folders, "legacy", Project.model_validate(legacy_raw))
        rmtree(legacy / "results")
        plan = folders.read_plan("legacy", revision=None).plan
        assert not plan.runs and any("pinned Model revision" in item.detail for item in plan.diagnostics)
        assert not (legacy / "results").exists()
