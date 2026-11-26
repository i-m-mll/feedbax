"""A copied project carries its pinned Model and retains exact executable archive weights."""

from copy import deepcopy
from io import BytesIO
import json
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
    task["epochs"]["move"]["end"]["length"]["value"] = 8
    task["validation_trials"] = dict(list(task["validation_trials"].items())[:1])
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
    source = loaded.lower()
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
        source = loaded.lower()
        assert loaded.archive is None
        assert source.weights == checkpoint.tree and source.encoding == checkpoint.encoding
        asset = loaded.archive_path("portable")
        projection = f"models/{source.model.sha256}.model.json"
        original = tmp_path / "original"
        assert (original / asset).read_bytes() == archive
        authored = (original / "documents/portable.json").read_bytes()
        assert len(authored) <= 600
        assert set(json.loads(authored)["payload"]) == {"model", "model_revision", "weights", "encoding_sha256"}
        for pin in (source.model.sha256, source.weights.descriptor.sha256, source.encoding.descriptor_sha256):
            assert authored.count(pin.encode()) == 1
        assert (original / projection).read_bytes() == encode_imported_model_document(model)
        warm = _check_plan(folders, "original")

        cold = _clone(tmp_path, "clone")
        rmtree(original)
        assert not (cold / "results").exists()
        rebuilt = _check_plan(folders, "clone")
        assert warm.plan.requests == rebuilt.plan.requests and warm.plan.documents == rebuilt.plan.documents
        store = ImmutableArtifactBlobProvider(rebuilt.store_path)
        assert store.get_object(loaded.lower().model) == (cold / projection).read_bytes()
        assert export_model_archive(store, source.weights, source.encoding) == archive
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
        (cold / asset).unlink()
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
        assert other.lower() == loaded.lower()
        assert other.archive is None and other.archive_path("equal") != loaded.archive_path("portable")
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

        asset = root / loaded.archive_path("portable")
        asset.write_bytes(b"different user bytes")
        assert _load(client, "original", archive, revision=latest).status_code == 409
        assert asset.read_bytes() == b"different user bytes"
        asset.unlink()
        rmtree(store.root)
        repaired = _expect(_load(client, "original", archive, revision=latest), 201)
        assert repaired["snapshot"]["revision"] == latest
        assert _loaded(repaired) == loaded and asset.read_bytes() == archive
        assert store.get_object(loaded.lower().model) == encode_imported_model_document(model)
        assert len(read_active_roots(store, (_root_id(loaded),))) == 1


def test_cold_plan_checks_assets_before_publishing_any_store_bytes(tmp_path, archive_bundle):
    _, components, first, _, archive, _ = archive_bundle
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders
        _seed(folders, "original")
        loaded = _loaded(_expect(_load(client, "original", archive), 201))
        warm = _check_plan(folders, "original")
        asset = loaded.archive_path("portable")
        projection = f"models/{loaded.model_revision.sha256}.model.json"
        with pytest.raises(ValidationError):
            LoadedModelDocument.model_validate(loaded.model_dump(mode="json") | {"archive": "models/../portable.feedbax-model"})

        def refused(name, phrase):
            plan = folders.read_plan(name, revision=None).plan
            assert not plan.runs
            assert {item.path for item in plan.diagnostics} == {"inputs.subject", "inputs.checkpoint"}
            assert all(phrase in item.detail for item in plan.diagnostics), plan.diagnostics
            assert not (tmp_path / name / "results").exists()

        swapped = _clone(tmp_path, "swapped")
        (swapped / asset).write_bytes(_different_archive(tmp_path / "other", archive, first, components))
        refused("swapped", "pinned weights or encoding")
        (swapped / asset).write_bytes(archive)
        assert _check_plan(folders, "swapped").plan.requests == warm.plan.requests

        edited = _clone(tmp_path, "edited")
        (edited / projection).write_bytes(b"{}")
        refused("edited", "pinned Model revision")

        linked = _clone(tmp_path, "linked")
        (linked / asset).unlink()
        (linked / asset).symlink_to(tmp_path / "original" / asset)
        refused("linked", "symlink")

        linked_dir = _clone(tmp_path, "linked-dir")
        rmtree(linked_dir / "models")
        (linked_dir / "models").symlink_to(tmp_path / "original" / "models", target_is_directory=True)
        refused("linked-dir", "symlink")


def test_loaded_model_locator_and_request_identity_survive_rekey_and_respell(tmp_path, archive_bundle):
    _, _, _, _, archive, _ = archive_bundle
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders
        _seed(folders, "original")
        imported = _expect(_load(client, "original", archive), 201)
        loaded = _loaded(imported)
        asset = loaded.archive_path("portable")
        requests = _check_plan(folders, "original").plan.requests

        renamed = _clone(tmp_path, "renamed")
        based = {
            "label": "Other fit", "schema": imported["snapshot"]["project"]["documents"]["portable"]["schema"],
            "base": {"document": "portable"},
        }
        (renamed / "documents/based-fit.json").write_text(json.dumps(based))
        (renamed / "models/based-fit.feedbax-model").write_bytes(archive)
        current = _expect(client.get("/api/projects/renamed"))
        changed = _expect(client.post("/api/projects/renamed/members/command", json={
            "revision": current["revision"], "presentation_revision": current["presentation_revision"],
            "command": {"kind": "change_key", "family": "documents", "member": "portable", "key": "different-name"},
        }))
        documents = changed["snapshot"]["project"]["documents"]
        assert documents["different-name"]["payload"]["archive"] == asset
        assert documents["based-fit"]["differences"] == {"/archive": {"value": "models/based-fit.feedbax-model"}}
        assert _check_plan(folders, "renamed").plan.requests == requests
        assert (renamed / asset).read_bytes() == archive

        respelled = _clone(tmp_path, "respelled")
        file = respelled / "documents/portable.json"
        document = json.loads(file.read_bytes())
        document["payload"]["archive"] = asset
        document["payload"]["model"]["pointer"] = None
        file.write_text(json.dumps(document, sort_keys=True))
        assert _check_plan(folders, "respelled").plan.requests == requests

        missing = _clone(tmp_path, "missing-projection")
        (missing / f"models/{loaded.model_revision.sha256}.model.json").unlink()
        failed = folders.read_plan("missing-projection", revision=None).plan
        assert not failed.runs and all("needs its project assets" in item.detail for item in failed.diagnostics)
        assert not (missing / "results").exists()
