from dataclasses import asdict
from hashlib import sha256
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.effective_success import INDEX_REF, read_success_object
from feedbax.contracts.execution_target import LOCAL_TARGET, CloseWhenIdle, SessionTarget, SessionTargetProfile
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import RunKey
from feedbax.host import HostError
from feedbax.orchestration.session import KeepOpen, SessionRequest
from feedbax.orchestration.targets import TargetRegistry
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.store_sync import collect_success
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.session_venue import ENVIRONMENT, LocalVenueProvider
from tests.test_canonical_evaluate_admission import _save_project, managed_evaluation_definition
from tests.test_local_job_host import host as host_fixture, wait_until

host = host_fixture


def test_unprofiled_session_request_keeps_its_saved_identity():
    saved = {
        "id": str(uuid4()), "target": "local-rehearsal", "resources": {"compute": 1},
        "environment": ENVIRONMENT, "lifecycle": {"kind": "keep-open"}, "maximum_microusd": 10,
    }
    original_bytes = canonical_json_v2_bytes(saved)
    restored = SessionRequest.model_validate_json(original_bytes)
    assert canonical_json_v2_bytes(restored.model_dump(mode="json")) == original_bytes
    assert restored.digest == sha256(original_bytes).hexdigest()
    named = restored.model_copy(update={"profile": "demo"})
    assert named.digest != restored.digest and named.placement.profile == "demo"


def test_target_catalog_and_default_saved_submission_share_one_contract(tmp_path, host, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    host.policy(compute=1)
    root = tmp_path / "projects"
    folders = ProjectFolderService(root)
    _save_project(folders, "reaching", Project.model_validate(managed_evaluation_definition()))
    provider = LocalVenueProvider(tmp_path / "venue")
    ready = SessionTargetProfile(
        kind="session", id="demo", label="Demo venue", driver=provider.target, resources={"compute": 1},
        lifecycle=KeepOpen(), maximum_microusd=1000,
    )
    idle = ready.model_copy(update={"id": "idle", "lifecycle": CloseWhenIdle(seconds=30)})
    missing = ready.model_copy(update={"id": "missing", "driver": "unregistered"})
    registry = TargetRegistry((ready, idle, missing), (provider,))
    catalog = registry.catalog(kinds=("train",), session_submission=True)
    assert catalog.targets[1].availability.state == "ready"
    assert registry.catalog(kinds=("report",), session_submission=True).targets[1].availability.state == "unavailable"

    key = RunKey("default", "train", "default")
    before = folders.read_plan("reaching", revision=None)
    original = next(run.request_id for run in before.plan.runs if run.key == key)

    with TestClient(create_app(projects_dir=root, targets=registry)) as client:
        catalog_response = client.get("/api/orchestration/targets", params={"kind": "train"})
        assert catalog_response.status_code == 200, catalog_response.text
        catalog_value = catalog_response.json()
        assert catalog_value["default"] == "local"
        options = {entry["profile"]["id"]: entry for entry in catalog_value["targets"]}
        assert options["local"]["profile"]["label"] == "This Mac"
        assert options["local"]["availability"] == {"state": "ready"}
        assert "not configured" in options["demo"]["availability"]["reason"]
        assert "close-when-idle" in options["idle"]["availability"]["reason"]
        assert "not registered" in options["missing"]["availability"]["reason"]
        assert client.get("/api/orchestration/targets", params={"kind": "unknown"}).status_code == 422

        url = "/api/projects/reaching/train-runs"
        request = {"revision": before.revision, "run_keys": [asdict(key)], "token": "placement-test"}
        denied = client.post(url, json=request | {"target": "missing"})
        assert denied.status_code == 409 and "not registered" in denied.text
        assert host.jobs(project="reaching") == []
        response = client.post(url, json=request)
        assert response.status_code == 202, response.text
        submission = response.json()
        assert submission["target"] == LOCAL_TARGET.model_dump(mode="json")
        assert client.post(url, json=request | {"target": "local"}).json() == submission
        job_id = submission["items"][0]["job_id"]

        service = client.app.state.job_service
        remote = SessionTarget(kind="session", profile="demo", driver=provider.target, session_id=uuid4())
        with pytest.raises(HostError, match="authenticated Host session"):
            service.admit_plan(folders, before, run_keys=(key,), token="wrong-host", placement=remote)

        def finished():
            (job,) = host.jobs(project="reaching", job_id=job_id)
            return job if job["outcome"] is not None else None

        job = wait_until(finished, timeout=25)
        assert job["state"] == "succeeded", job["outcome"]
        assert job["request"] == original and job["data"]["target"] == submission["target"]
        activity = client.get("/api/jobs", params={"project": "reaching"}).json()["jobs"]
        assert activity[0]["target"] == submission["target"]

    store = ImmutableArtifactBlobProvider(before.store_path)
    ref = ContentObjectRef.model_validate(job["outcome"]["store_success"])
    receipt = SuccessView(store).read_receipt(ref)
    assert receipt.target == LOCAL_TARGET and receipt.request.id == original
    assert receipt.schema_version == "feedbax.proof.effective-success.v2"
    other = ImmutableArtifactBlobProvider(tmp_path / "wrong-collection")
    with pytest.raises(ValueError, match="expected execution target"):
        collect_success(store, other, ref, attempt=job_id, request_id=original, expected_target=remote)
    assert other.read_ref(INDEX_REF) is None
    old = receipt.model_dump(mode="json")
    old["schema_version"] = "feedbax.proof.effective-success.v1"
    del old["target"]
    legacy_ref = store.store_object(canonical_json_v2_bytes(old))
    with pytest.raises(ValueError, match="feedbax.proof.effective-success.v2"):
        read_success_object(legacy_ref, store.get_object)
