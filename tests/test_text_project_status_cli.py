"""Read the same saved receipt evidence through Studio and the text CLI."""

from hashlib import sha256
import json

from feedbax.cli import main
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.jobs import JobStatus
from feedbax.contracts.manifest import TrainingRunManifest, training_run_manifest_id, write_manifest
from feedbax.contracts.project import InlineBase
from feedbax.contracts.project_io import dump_text_project, read_text_project_snapshot
from feedbax.contracts.project_plan import RunKey, compile_project
from feedbax.host import HostClient
from feedbax.host.plan_history import advance_plan, plan_ref_key, plan_run_key
from feedbax.job_completion import training_binding_path
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput, bind_canonical_train
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.support.managed_train_fixture import managed_train_project
from tests.test_text_project_plan_cli import _edit, _files, _write_walkthrough


def _status_table(output):
    header, *rows = output.splitlines()
    assert header.split("\t") == ["ADDRESS", "ROW", "STATE", "REASON"]
    result = {}
    for line in rows:
        address, label, state, reason = line.split("\t", 3)
        result[address] = (label, state, reason)
    return result


def _retain_train(root, project, registries):
    job_id = "saved-attempt"
    plan = compile_project(project)
    (train,) = (run for run in plan.runs if run.kind == "train")
    store = ImmutableArtifactBlobProvider(root / "results")
    spec, binding = bind_canonical_train(
        train.request_id,
        plan.requests[train.request_id],
        plan.documents,
        registries.training_programs,
        store,
        registries=registries,
    )
    bound = (
        BoundStudioTrainingJobInput(
            spec=spec,
            project_id=root.name,
            request_id=train.request_id,
            manifest_root=store.root,
            canonical_request=binding,
        )
        .model_dump_json()
        .encode()
    )
    binding_path = training_binding_path(store.root, job_id)
    binding_path.parent.mkdir(parents=True)
    binding_path.write_bytes(bound)
    manifest_id = training_run_manifest_id(job_id)
    manifest = write_manifest(
        TrainingRunManifest(id=manifest_id, job_id=job_id, status="completed"),
        root=store.root,
        index=False,
    )
    receipt = {
        "manifest_id": manifest_id,
        "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
    }
    root_name = str(store.root)
    plan_key = plan_ref_key(root.name, root_name)
    request_map = {plan_run_key("default", "train", "default"): train.request_id}
    ref = store.store_object(canonical_json_v2_bytes({"requests": request_map}))
    snapshot = {
        "cursor": 9,
        "oldest_cursor": 0,
        "jobs": [
            {
                "id": job_id,
                "project": root.name,
                "request": train.request_id,
                "state": "succeeded",
                "data": {"manifest_root": root_name},
                "edges": [],
                "continues": None,
                "outcome": {
                    "receipt": receipt,
                    "completion": {
                        "project_id": root.name,
                        "manifest_root": root_name,
                        "producer_job_id": job_id,
                        "request_id": train.request_id,
                        "binding_sha256": sha256(bound).hexdigest(),
                        "receipt": receipt,
                    },
                },
            }
        ],
        "records": {plan_key: advance_plan(ref.model_dump(mode="json"), request_map, None, {})},
    }
    return snapshot, plan_key, manifest


def test_cli_and_studio_verify_saved_status_and_authored_history(tmp_path, monkeypatch, capsys):
    root = (tmp_path / "saved").resolve()
    project, _ = _write_walkthrough(root)
    managed = managed_train_project()
    project.steps["train"] = managed.steps["train"]
    project.documents.update(managed.documents)
    baseline = project.steps["baseline"].base
    assert isinstance(baseline, InlineBase)
    params = baseline.inline["params"]
    del params["trials"]
    params["trial_set"] = {"kind": "every_condition"}
    configuration = read_text_project_snapshot(root).configuration
    for relative, data in dump_text_project(project, configuration).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    registries = new_application_registry_bundle()
    registries.seal()
    snapshot, plan_key, manifest = _retain_train(root, project, registries)
    receipt = manifest.read_bytes()
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(tmp_path / "host"))

    def snapshot_rpc(self, *, project, start_host, record_keys, include_launch):
        assert (project, start_host, record_keys, include_launch) == (
            root.name,
            False,
            (plan_key,),
            False,
        )
        return snapshot

    monkeypatch.setattr(HostClient, "jobs_snapshot", snapshot_rpc)
    monkeypatch.chdir(root / "steps")
    service = ProjectPlanService(
        ProjectFolderService(root.parent, operations=registries.operations), JobService(HostClient(), registries)
    )
    key = RunKey("default", "train", "default")

    def compare(state):
        before = _files(root)
        assert main(["status", "--variant", "default"]) == 0
        table = _status_table(capsys.readouterr().out)
        assert _files(root) == before
        assert not (tmp_path / "host").exists()
        saved = read_text_project_snapshot(root)
        studio = service.read(root.name, revision=saved.revision)
        history = studio.history[key]
        assert history.status is state
        assert table["default/train/default"] == ("default", state.value, history.reason or "")
        evaluation = studio.evaluations[RunKey("default", "baseline", "default")]
        assert evaluation.reason == "/evaluation_type: Evaluate activation is held."
        assert table["default/baseline/default"][1:] == (evaluation.status.value, evaluation.reason)
        mounted = studio.window_map()
        mounted_rows = {
            f"{item.key.variant}/{item.key.step}/{item.key.row}": item.status
            for item in mounted.rows if item.status is not None
        }
        assert all(table[address][1] == status for address, status in mounted_rows.items())
        counts = {item.step_id: item.status_counts for item in mounted.steps}
        assert counts["train"] == {history.status.value: 1}
        assert counts["baseline"] == {evaluation.status.value: 1}
        (profile,) = (item for item in mounted.analysis_elements if item.name == "profile")
        assert table["default/analyze/default/profile"][1] == profile.status == "not_run"
        assert mounted_rows["default/report/default"] == "not_run"
        return history

    assert compare(JobStatus.SUCCEEDED).result is not None
    manifest.write_bytes(receipt + b" ")
    assert compare(JobStatus.CORRUPT).result is None
    manifest.write_bytes(receipt)

    attempts = snapshot["jobs"]
    attempts.append(
        attempts[0]
        | {
            "id": "failed-attempt",
            "state": "failed",
            "outcome": {"error": {"message": "Training stopped at the loss guard."}},
        }
    )
    after_failure = compare(JobStatus.SUCCEEDED)
    assert after_failure.result is not None
    assert after_failure.reason == "Training stopped at the loss guard."
    attempts.pop()

    step_path = root / "steps/train.json"
    step = json.loads(step_path.read_bytes())
    method = step["base"]["inline"]["method_payload"]["payload"]
    method["seed"] = method.get("seed", 0) + 1
    step_path.write_text(json.dumps(step), encoding="utf-8")
    stale = compare(JobStatus.STALE)
    assert stale.change.path == "payload.method_payload.payload.seed"
    assert stale.change.path in stale.reason
    assert stale.previous is not None


def test_status_displays_failed_element_with_valid_siblings(tmp_path, monkeypatch, capsys):
    root = tmp_path / "saved"
    _, files = _write_walkthrough(root)
    invalid = json.loads(files["documents/velocity-figure.json"])
    invalid["payload"]["layers"][0]["style"]["band_alpha"] = -1
    name = "documents/bad-figure.json"
    files[name] = json.dumps(invalid).encode()
    (root / name).write_bytes(files[name])

    def invalid_siblings(document):
        document["operations"]["unused"] = document["operations"]["align"] | {"operation": "feedbax.unknown"}
        document["views"]["bad"] = document["views"]["profile"] | {"figure": {"document": "bad-figure"}}

    _edit(root, files, "velocity-analysis", invalid_siblings)
    monkeypatch.setattr(
        HostClient,
        "jobs_snapshot",
        lambda *args, **kwargs: {"jobs": [], "cursor": 0},
    )
    assert main(["status", str(root)]) == 2
    output = capsys.readouterr()
    table = _status_table(output.out)
    assert table["default/analyze/default"][1] == "cannot_run"
    assert table["default/analyze/default/profile"][1] == "not_run"
    assert table["default/analyze/default/bad"][1] == "cannot_run"
    _, state, reason = table["default/analyze/default/unused"]
    assert state == "cannot_run" and "payload.operations.unused.operation" in reason
    assert table["default/report/default"][1] == "not_run"
    assert "steps/analyze.json" in output.err
    assert "documents/bad-figure.json:payload.layers.0.style.band_alpha" in output.err
    assert _files(root) == files

    registries = new_application_registry_bundle()
    registries.seal()
    service = ProjectPlanService(
        ProjectFolderService(root.parent, operations=registries.operations), JobService(HostClient(), registries)
    )
    mounted = service.read(root.name, revision=read_text_project_snapshot(root).revision).window_map()
    (analyze,) = (row for row in mounted.rows if row.key.step == "analyze")
    assert analyze.status == table["default/analyze/default"][1]
    (step,) = (step for step in mounted.steps if step.step_id == "analyze")
    assert step.status_counts == {"cannot_run": 1}
    assert any(
        (problem.element, problem.document, problem.document_path)
        == ("bad", "bad-figure", "payload.layers.0.style.band_alpha")
        for problem in mounted.problems
    )
    assert any(element.name == "profile" and element.status == "not_run" for element in mounted.analysis_elements)


def test_status_does_not_start_a_missing_host(tmp_path, monkeypatch, capsys):
    root = tmp_path / "saved"
    _, files = _write_walkthrough(root)
    host_root = tmp_path / "host"
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host_root))
    assert main(["status", str(root)]) == 2
    output = capsys.readouterr()
    assert "saved result history is unavailable" in output.err and not output.out
    assert not host_root.exists()
    assert _files(root) == files
