"""Accepted saved Evaluate terminal evidence commits once through Host replay."""

import os
from pathlib import Path
from textwrap import dedent

import pytest

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.project import Project
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef
from feedbax.host import HostError
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.persistence.store_sync import sync_store
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_local_job_host import admit, host as host_fixture, job_row, restart, run_job, wait_until
from tests.test_saved_update_series import definition

host = host_fixture


@pytest.mark.parametrize("damage", [False, True], ids=["replay", "damaged"])
def test_accepted_saved_evaluate_recovers_committed_store_attempt(host, tmp_path, monkeypatch, damage):
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
    closure = StoreClosureRef(roots=(ReceiptRoot(receipt=committed.ref),), include_ancestry=True)
    destination = ImmutableArtifactBlobProvider(tmp_path / "copied")
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
