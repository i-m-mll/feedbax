"""Read the same saved receipt evidence through Studio and the text CLI."""

import json

from fastapi.testclient import TestClient

from feedbax.cli import main
from feedbax.contracts.jobs import JobStatus, StudioJobSubmission
from feedbax.contracts.manifest import canonical_manifest_path
from feedbax.application.project_writer import dump_text_project
from feedbax.contracts.project_io import read_text_project_snapshot
from feedbax.planner.project_plan import RunKey
from feedbax.planner.project_plan_io import load_plan_requests
from feedbax.contracts.studio_place import StudioAnalysisPlace, StudioStepPlace
from feedbax.host import HostClient
from feedbax.host.plan_history import plan_ref_key
from feedbax.jobs import JobService, plan_attempt_key
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.support.managed_train_fixture import managed_train_project
from tests.support.host import job_row, token, wait_until
from tests.test_portable_loaded_model import _check_plan, _expect, _load, _loaded, _seed, archive_bundle as archive_bundle
from tests.test_text_project_plan_cli import _edit, _files, _write_walkthrough


def _status_table(output):
    header, *rows = output.splitlines()
    assert header.split("\t") == ["ADDRESS", "ROW", "STATE", "REASON"]
    result = {}
    for line in rows:
        address, label, state, reason = line.split("\t", 3)
        result[address] = (label, state, reason)
    return result


def _retain_train(host, folders, reviewed, registries, key):
    submission = JobService(host, registries).admit_plan(
        folders, reviewed, run_keys=(key,), token=token("saved-train"),
    )
    (attempt,) = submission.items
    finished = wait_until(
        lambda: row if (row := job_row(host, attempt.job_id))["outcome"] else None,
    )
    assert finished["state"] == "succeeded", finished
    store = reviewed.store_path
    receipt = finished["outcome"]["receipt"]
    manifest = canonical_manifest_path("TrainingRunManifest", receipt["manifest_id"], root=store)
    plan_key = plan_ref_key(reviewed.project_id, str(store))
    snapshot = host.jobs_snapshot(
        project=reviewed.project_id, start_host=False, record_keys=(plan_key,), include_launch=True,
    )
    return snapshot, plan_key, manifest


def test_cli_reads_and_plans_a_portable_loaded_model(tmp_path, monkeypatch, capsys, archive_bundle):
    _, _, _, _, archive, _ = archive_bundle
    host_root = tmp_path / "host"
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host_root))
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders
        _seed(folders, "loaded")
        loaded = _loaded(_expect(_load(client, "loaded", archive), 201))
        expected = _check_plan(folders, "loaded")
    root = tmp_path / "loaded"
    before = _files(root)

    def jobs_snapshot(self, *, project, start_host, checkpoint_progress_root, **options):
        assert project == expected.project_id and not start_host
        assert checkpoint_progress_root == str(expected.store_path)
        return {"jobs": [], "cursor": 0}

    monkeypatch.setattr(HostClient, "jobs_snapshot", jobs_snapshot)
    assert main(["status", str(root / "steps")]) == 0
    status = capsys.readouterr()
    assert not status.err
    assert {name: state for name, (_, state, _) in _status_table(status.out).items()} == {
        "evaluate": "not_run", "train": "not_run",
    }

    review = tmp_path / "review.json"
    assert main(["plan", str(root), "--out", str(review)]) == 0
    assert not capsys.readouterr().err
    exported, _ = load_plan_requests(review.read_bytes())
    assert exported.requests == expected.plan.requests
    asset = root / loaded.archive_path("portable")
    assert main(["plan", str(root), "--out", str(asset)]) == 2
    assert "overlaps an authored project file" in capsys.readouterr().err

    def review_admission(self, folders, reviewed, *, run_keys, token, actor, **options):
        assert reviewed.store_path == expected.store_path
        assert reviewed.plan.requests == expected.plan.requests
        assert run_keys == (RunKey("default", "evaluate", "default"),)
        return StudioJobSubmission(id="checked", token=token, project_id=reviewed.project_id, items=[])

    monkeypatch.setattr(JobService, "admit_plan", review_admission)
    assert main(["run", str(root), "evaluate", "--json"]) == 0
    submitted = capsys.readouterr()
    assert not submitted.err and StudioJobSubmission.model_validate_json(submitted.out).id == "checked"
    assert _files(root) == before and not host_root.exists()


def test_cli_and_studio_verify_saved_status_and_authored_history(host, tmp_path, monkeypatch, capsys):
    root = (tmp_path / "saved").resolve()
    project, _ = _write_walkthrough(root)
    managed = managed_train_project()
    project.steps["train"] = managed.steps["train"]
    project.documents.update(managed.documents)
    configuration = read_text_project_snapshot(root).configuration
    for relative, data in dump_text_project(project, configuration).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    registries = new_application_registry_bundle()
    registries.seal()
    folders = ProjectFolderService(root.parent, operations=registries.operations)
    reviewed = folders.read_plan(root.name, revision=None)
    key = RunKey("default", "train", "default")
    snapshot, plan_key, manifest = _retain_train(host, folders, reviewed, registries, key)
    receipt = manifest.read_bytes()
    absent_host = tmp_path / "absent-host"
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(absent_host))

    def snapshot_rpc(
        self, *, project, start_host, record_keys, include_launch, checkpoint_progress_root, checkpoint_admission,
        attempt_addresses,
    ):
        assert project == reviewed.project_id
        assert not start_host and record_keys == (plan_key,)
        assert include_launch and checkpoint_progress_root == str(root / "results")
        scope, addresses = attempt_addresses
        assert scope == checkpoint_progress_root
        assert plan_attempt_key(key) in addresses
        return snapshot

    monkeypatch.setattr(HostClient, "jobs_snapshot", snapshot_rpc)
    monkeypatch.chdir(root / "steps")
    service = ProjectPlanService(folders, JobService(HostClient(), registries))

    def compare(state):
        before = _files(root)
        assert main(["status", "--variant", "default"]) == 0
        table = _status_table(capsys.readouterr().out)
        assert _files(root) == before
        assert not absent_host.exists()
        saved = read_text_project_snapshot(root)
        studio = service.read(root.name, revision=saved.revision)
        history = studio.history[key]
        assert history.status is state
        mounted = studio.window_map()

        def named(step):
            return " | ".join(problem.detail for problem in mounted.problems if (problem.step, problem.row) == (step, "default"))

        changed = f"Definition changed at {history.change.path}." if history.change and state is JobStatus.STALE else None
        assert table["train"] == ("default", state.value, history.reason or changed or named("train"))
        evaluation = studio.evaluations[RunKey("default", "baseline", "default")]
        assert evaluation.result is None
        assert table["baseline"][1:] == (evaluation.status.value, evaluation.reason or named("baseline"))
        mounted_rows = {
            item.key.step: item.status
            for item in mounted.rows if item.status is not None and item.key.row == item.key.variant == "default"
        }
        assert all(table[address][1] == status for address, status in mounted_rows.items())
        counts = {item.step_id: item.status_counts for item in mounted.steps}
        assert counts["train"] == {history.status.value: 1}
        assert counts["baseline"] == {evaluation.status.value: 1}
        (profile,) = (item for item in mounted.analysis_elements if item.name == "profile")
        dependent_status = "stale" if state is JobStatus.STALE else "waiting"
        assert table["analyze/profile"][1] == profile.status == dependent_status
        assert mounted_rows["report"] == dependent_status
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
    payload = step["base"]["inline"]
    payload["seed"] = payload.get("seed", 0) + 1
    step_path.write_text(json.dumps(step), encoding="utf-8")
    stale = compare(JobStatus.STALE)
    assert stale.change.path == "payload.seed"
    assert stale.change.place == StudioStepPlace(part="step", field="seed")
    assert stale.reason is None
    assert stale.previous is not None


def test_status_displays_failed_element_with_valid_siblings(tmp_path, monkeypatch, capsys):
    root = tmp_path / "saved"
    _, files = _write_walkthrough(root)
    invalid = json.loads(files["documents/velocity-figure.json"])
    invalid["payload"]["layers"]["bounds"]["style"] = {"band_alpha": -1}
    name = "documents/bad-figure.json"
    files[name] = json.dumps(invalid).encode()
    (root / name).write_bytes(files[name])

    def invalid_siblings(document):
        document["operations"]["unused"] = document["operations"]["align"] | {"operation": "feedbax.unknown"}
        document["views"]["bad"] = document["views"]["profile"] | {
            "label": "Saved comparison", "figure": {"document": "bad-figure"},
        }

    _edit(root, files, "velocity-analysis", invalid_siblings)
    monkeypatch.setattr(
        HostClient,
        "jobs_snapshot",
        lambda *args, **kwargs: {"jobs": [], "cursor": 0},
    )
    assert main(["status", str(root)]) == 2
    output = capsys.readouterr()
    table = _status_table(output.out)
    assert table["analyze"][1] == "cannot_run"
    assert table["analyze/profile"][1] == "waiting"
    assert table["analyze/bad"][1] == "cannot_run"
    _, state, reason = table["analyze/unused"]
    assert state == "cannot_run" and "feedbax.unknown" in reason
    assert table["report"][1] == "waiting"
    assert "steps/analyze.json" in output.err
    assert "documents/bad-figure.json:payload.layers.bounds.style.band_alpha" in output.err
    assert _files(root) == files

    registries = new_application_registry_bundle()
    registries.seal()
    service = ProjectPlanService(
        ProjectFolderService(root.parent, operations=registries.operations), JobService(HostClient(), registries)
    )
    mounted = service.read(root.name, revision=read_text_project_snapshot(root).revision).window_map()
    (analyze,) = (row for row in mounted.rows if row.key.step == "analyze")
    assert analyze.status == table["analyze"][1]
    (step,) = (step for step in mounted.steps if step.step_id == "analyze")
    assert step.status_counts == {"cannot_run": 1}
    assert any(
        (problem.place, problem.document_path)
        == (StudioAnalysisPlace(part="analysis", element="bad", name="Saved comparison"),
            "payload.layers.bounds.style.band_alpha")
        for problem in mounted.problems
    )
    assert any(element.name == "profile" and element.status == "waiting" for element in mounted.analysis_elements)


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
