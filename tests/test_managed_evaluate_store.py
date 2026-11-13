"""Accepted saved Evaluate terminal evidence commits once through Host replay."""

import json
import os
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from feedbax.contracts.artifact_custody import ArtifactBlobIntegrityError, ContentObjectRef
from feedbax.contracts.project import Project
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef
from feedbax.contracts.store_check import StoreCheckResult
from feedbax.cli import main as cli_main
from feedbax.host import HostError
from feedbax.jobs import JobService
from feedbax.persistence.array_container import ArrayContainerError, ArrayContainerReader
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.persistence.store_sync import sync_store
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_evaluate import EvaluationResult, states_ref
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_local_job_host import admit, host as host_fixture, job_row, restart, run_job, wait_until
from tests.test_saved_update_series import definition
from tests.support.store_damage import writable_store_object

host = host_fixture


@pytest.mark.parametrize("damage", [False, True], ids=["replay", "damaged"])
def test_accepted_saved_evaluate_recovers_committed_store_attempt(host, tmp_path, monkeypatch, capsys, damage):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    jobs = JobService(host, new_application_registry_bundle(), enable_evaluation_candidate=True)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "evaluate-store"
    _save_project(folders, project, Project.model_validate(definition("last")))
    plan = folders.read_plan(project, revision=None)
    runs = {run.kind: run for run in plan.plan.runs}
    (train,) = jobs.admit_plan(folders, plan, run_keys=(runs["train"].key,), token="train").items
    wait_until(lambda: job_row(host, train.job_id, "succeeded"), timeout=22)
    host.policy(compute=1)
    release = tmp_path / "release"
    gate = run_job(
        "gate",
        f"from pathlib import Path\nimport time\nwhile not Path({str(release)!r}).exists(): time.sleep(.01)",
        project=project,
    )
    admit(host, "gate", gate)
    wait_until(lambda: job_row(host, gate.id, "running"))

    def submit(token):
        return jobs.admit_plan(folders, plan, run_keys=(runs["evaluate"].key,), token=token)

    first = submit("prior")
    release.touch()
    prior = wait_until(lambda: job_row(host, first.items[0].job_id, "succeeded"), timeout=22)
    release.unlink()
    gate = run_job("retry-gate", gate.launch.argv[2], project=project)
    admit(host, "retry-gate", gate)
    wait_until(lambda: job_row(host, gate.id, "running"))
    store = ImmutableArtifactBlobProvider(plan.store_path)
    from feedbax.contracts.effective_success import INDEX_REF

    previous_index = store.read_ref(INDEX_REF)
    (stopped,) = submit("stop-before-publication").items
    assert jobs.stop((stopped.job_id,), "checkpoint")[0].job.status == "stopped"
    assert store.read_ref(INDEX_REF) == previous_index
    submission = submit("evaluate")
    (evaluate,) = submission.items
    assert evaluate.disposition == "admitted"
    attempt = evaluate.job_id
    probe = tmp_path / "probe"
    probe.mkdir()
    marker = probe / "committed"
    (probe / "sitecustomize.py").write_text(dedent(f"""\
        import errno
        import os
        from pathlib import Path

        from feedbax.host.store import Store
        import feedbax.host.evaluate_success as publisher

        finished = Store.finished
        publish = publisher.publish_evaluate_success
        marker = Path({str(marker)!r})
        pending = marker.with_suffix(".pending")

        def publish_once(**kwargs):
            if kwargs["attempt"] == {attempt!r} and not pending.exists():
                pending.touch()
                raise OSError(errno.EIO, "injected transient Store failure")
            return publish(**kwargs)

        def interrupt(self, row, outcome):
            if row["id"] == {attempt!r} and "store_success" in outcome and not marker.exists():
                marker.write_text(outcome["store_success"]["sha256"])
                os._exit(81)
            return finished(self, row, outcome)

        publisher.publish_evaluate_success = publish_once
        Store.finished = interrupt
    """))
    with monkeypatch.context() as env:
        env.setenv("PYTHONPATH", os.pathsep.join((str(probe), str(Path.cwd()), os.environ.get("PYTHONPATH", ""))))
        restart(host)
        release.touch()
        wait_until(marker.exists, timeout=26)
        view = SuccessView(store)
        committed = view.read_attempt(attempt)
        assert committed.ref.sha256 == marker.read_text()
        assert committed.receipt.supersedes.model_dump(mode="json") == prior["outcome"]["store_success"]
        if damage:
            # Equal Evaluate results share dataset bytes. Damage unique attempt evidence instead.
            target = store.object_path(committed.receipt.evaluation_manifest)
            target.unlink()

        def recovered():
            try:
                return job_row(host, attempt, "failed" if damage else "succeeded")
            except HostError as error:
                if error.code != "disconnected":
                    raise
                return None

        result = wait_until(recovered, timeout=22)
    assert result["delivery"] == 1
    assert submit("evaluate") == submission
    assert marker.with_suffix(".pending").exists()
    assert SuccessView(store).attempt_ref(attempt) == committed.ref
    if damage:
        assert result["outcome"]["error"]["code"] == "publication_failed"
        assert "store_success" not in result["outcome"]
        with jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=prior["id"]).open_states() as dataset:
            assert dataset.data("checkpoint").tolist() == [4]
        return

    assert ContentObjectRef.model_validate(result["outcome"]["store_success"]) == committed.ref
    assert SuccessView(store).authored(runs["evaluate"].request_id) == committed
    assert committed.receipt.inputs["subject"][0].evaluation_subject.checkpoints[0].occurrence.run_id == train.job_id
    original = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=attempt)
    (states,) = committed.receipt.products["states"]
    assert states.content.sha256 == original._manifest().artifacts[0].sha256
    manifest = original._manifest()
    states_binding = states_ref(manifest)
    train_success = SuccessView(store).read_attempt(train.job_id)
    assert train_success is not None and train_success.receipt.continuation is not None
    train_slot = train_success.receipt.continuation.prng.content
    train_path = store.object_path(train_slot)
    train_raw = train_path.read_bytes()
    project_root = folders.root / project
    check = ["check", str(project_root), "--json"]
    before = store.read_ref(INDEX_REF)

    def checked(args, status):
        capsys.readouterr()
        assert cli_main([*check, *args]) == status
        output = capsys.readouterr()
        report = StoreCheckResult.model_validate_json(output.out)
        return report, [json.loads(line) for line in output.err.splitlines()]

    for selection in ([], ["--attempt", train.job_id], ["--attempt", attempt], ["--attempt", prior["id"]]):
        report, diagnostics = checked(selection, 0)
        assert report.verified_objects and report.verified_bytes
        assert report.findings == () and diagnostics == []
    producer_request = store.object_path(train_success.receipt.request.object)
    parked_request = producer_request.with_suffix(".held")
    producer_request.rename(parked_request)
    try:
        report, diagnostics = checked(["--attempt", attempt], 1)
        assert report.findings[0].address == train_success.receipt.request.object.object_id
        assert diagnostics[0]["path"] == str(producer_request)
        assert any(finding.code == "check_incomplete" for finding in report.findings)
    finally:
        parked_request.rename(producer_request)
    with writable_store_object(train_path):
        damaged_train = train_raw[:-1] + bytes([train_raw[-1] ^ 1])
        try:
            train_path.write_bytes(damaged_train)
            report, diagnostics = checked(["--attempt", train.job_id], 1)
            assert [entry.address for entry in report.findings] == [train_slot.object_id]
            assert [entry["path"] for entry in diagnostics] == [str(train_path)]
            assert train_path.read_bytes() == damaged_train
            assert store.read_ref(INDEX_REF) == before
        finally:
            train_path.write_bytes(train_raw)
    with original.open_states(verify=True) as dataset:
        position = dataset.descriptor.attrs["scene_position"]
        trial = dataset.data("trial")[1]
        expected = dataset.data(position, (0, 0, 1))
        expected_axis = dataset.data("state_time").tolist()

    read_slice = ArrayContainerReader.read_unverified_slice
    reads = []

    def observe_slice(reader, name, index):
        reads.append((name, index))
        return read_slice(reader, name, index)

    def unexpected_tensor(*args):
        pytest.fail("The default Store-owned reader verified a whole tensor")

    with monkeypatch.context() as probe:
        probe.setattr(ArrayContainerReader, "read_unverified_slice", observe_slice)
        probe.setattr(ArrayContainerReader, "read_tensor", unexpected_tensor)
        with original.open_states() as dataset:
            assert not reads
            np.testing.assert_array_equal(dataset.data(position, (0, 0, 1)), expected)
            assert reads == [(position, (0, 0, 1))]
        reads.clear()
        replayed = original.replay_trial(trial, anchor_id="effector")
        assert reads == [("state_time", ()), (position, (0, 0, 1))]
        assert replayed.time.values == expected_axis
        assert replayed.tracks[0].samples == expected.tolist()

    unpinned = EvaluationResult(plan.store_path, attempt, runs["evaluate"].request_id, original.receipt)
    path = store.object_path(states_binding.object_ref)
    raw = path.read_bytes()
    header_end = 8 + int.from_bytes(raw[:8], "little")
    offsets = json.loads(raw[8:header_end])[position]["data_offsets"]
    offset = header_end + offsets[0] + expected.nbytes
    altered = raw[:offset] + bytes([raw[offset] ^ 1]) + raw[offset + 1:]
    closure = StoreClosureRef(roots=(ReceiptRoot(receipt=committed.ref),), include_ancestry=True)
    destination = ImmutableArtifactBlobProvider(tmp_path / "copied")
    mode = path.stat().st_mode & 0o777
    path.chmod(mode | 0o200)
    try:
        with pytest.raises(ArtifactBlobIntegrityError, match="changed during read"):
            with original.open_states() as dataset:
                np.testing.assert_array_equal(dataset.data(position, (0, 0, 1)), expected)
                path.write_bytes(altered)
        with original.open_states() as dataset:
            assert dataset.data(position, (0, 0, 1)).tobytes() != expected.tobytes()
        for reader in (original.open_states(verify=True), unpinned.open_states()):
            with pytest.raises(ArrayContainerError, match="digest mismatch"):
                with reader as dataset:
                    dataset.data(position, (0, 0, 1))
        with pytest.raises(ArtifactBlobIntegrityError):
            store.verify_object(states_binding.object_ref)
        for selection in ([], ["--attempt", attempt], ["--attempt", prior["id"]]):
            report, diagnostics = checked(selection, 1)
            assert [entry.address for entry in report.findings] == [states_binding.object_ref.object_id]
            assert [entry["path"] for entry in diagnostics] == [str(path)]
        assert path.read_bytes() == altered and store.read_ref(INDEX_REF) == before
        with pytest.raises(ArtifactBlobIntegrityError):
            sync_store(store, destination, closure, destination_ref="proof/evaluate-closure")
        assert destination.read_ref("proof/evaluate-closure") is None
    finally:
        path.write_bytes(raw)
        path.chmod(mode)
    sync_store(store, destination, closure, destination_ref="proof/evaluate-closure")
    refs = ReceiptClosure(destination).walk(closure)
    assert committed.receipt.evaluation_manifest in refs
    assert committed.receipt.inputs["subject"][0].evaluation_subject.producer.object in refs
    from feedbax.contracts.dataset import DatasetProductRef
    from feedbax.persistence.dataset import open_stored_dataset

    with open_stored_dataset(destination, DatasetProductRef(
        content=states.content, descriptor_sha256=states.descriptor["descriptor_sha256"],
    )) as copied:
        assert copied.data("checkpoint").tolist() == [4]
