"""Rehearse exact Train collection and local Resume through two real Hosts."""

from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import signal

import pytest

from feedbax.contracts.artifact_custody import ArtifactBlobCustodyError, ContentObjectRef
from feedbax.contracts.checkpoint_projection import CheckpointOccurrenceRef
from feedbax.contracts.method_checkpoint import MethodCheckpointManifest, method_checkpoint_extra_path
from feedbax.contracts.jobs import JobStatus
from feedbax.contracts.project_io import dump_text_project, read_text_project_snapshot
from feedbax.host import HostClient
from feedbax.jobs import JobConflict, JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.store_sync import collect_success
from feedbax.persistence.train_job import collect_train_job, read_train_job
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.support.finite_train import install_finite_gate
from tests.support.store_damage import writable_store_object
from tests.support.store_transport import loopback_store
from tests.test_canonical_train_admission import _saved_chain
from tests.support.host import private_host as host_fixture, restart, wait_until

host = host_fixture


@pytest.mark.round(
    "Real Train results move between two Hosts and resume after the first is removed."
)
def test_collect_train_and_resume_a_stopped_successor_after_venue_removal(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    registries = new_application_registry_bundle()
    venue = HostClient(tmp_path / "venue-host")
    folders = ProjectFolderService(tmp_path / "projects")
    reviewed, prefix, child = _saved_chain(folders)
    project = reviewed.project_id
    _, location = folders.open_native_id(project)
    saved = read_text_project_snapshot(location.path)
    source = ImmutableArtifactBlobProvider(reviewed.store_path)
    destination = ImmutableArtifactBlobProvider(tmp_path / "collected-results")
    venue_jobs = JobService(venue, registries)
    local_jobs = JobService(host, registries)

    def launch(jobs, folders, plan, key, token):
        (item,) = jobs.admit_plan(folders, plan, run_keys=(key,), token=token).items
        assert item.disposition == "admitted", item.reason
        return item.job_id

    def terminal(client, job, expected):
        def read():
            (row,) = client.jobs(job_id=job, include_launch=True)
            return row if row["outcome"] is not None else None

        result = wait_until(read)
        assert result["state"] == expected, result["outcome"]
        return result

    def close_venue():
        if venue.address.exists():
            os.kill(venue.ping()["pid"], signal.SIGTERM)
            wait_until(lambda: not venue.address.exists())
        venue.prohibit_start()

    def observed_success(row, provider):
        ref = ContentObjectRef.model_validate(row["outcome"]["store_success"])
        return SuccessView(provider).read_owned_success(ref)

    with ExitStack() as cleanup:
        venue.policy(compute=1)
        cleanup.callback(close_venue)
        producer_id = launch(venue_jobs, folders, reviewed, prefix, "venue-producer")
        producer = terminal(venue, producer_id, "succeeded")
        original_producer = observed_success(producer, source)
        control_id = launch(venue_jobs, folders, reviewed, child, "venue-control")
        control = observed_success(terminal(venue, control_id, "succeeded"), source).receipt

        gated = JobService(venue, registries)
        gate = tmp_path / "checkpoint-gate"
        gate.mkdir()
        install_finite_gate(gated, gate, stop_at=3, resume_at=8)
        stopped_id = launch(gated, folders, reviewed, child, "venue-stop")
        wait_until(lambda: (gate / "before-stop").exists())
        (stop,) = gated.stop((stopped_id,), "checkpoint")
        assert stop.error is None
        stopped = terminal(venue, stopped_id, "stopped")
        assert stopped["outcome"].get("store_publication_error") is None, stopped["outcome"]
        assert "store_success" not in stopped["outcome"]
        observed_entry = ContentObjectRef.model_validate(stopped["outcome"]["store_job_entry"])
        exact = read_train_job(source, stopped_id)
        assert exact.object_ref == observed_entry
        assert exact.recovery.pointer.completed_updates == 3
        (selected,) = exact.inputs["checkpoint"]
        assert selected.selection.receipt == original_producer.ref
        close_venue()

        remote = loopback_store(source.root, tmp_path / "connection")
        transported = ImmutableArtifactBlobProvider(tmp_path / "transported")
        with pytest.raises(OSError, match="observed Store reference"):
            collect_train_job(
                remote.endpoint, transported, attempt=stopped_id,
                request_id=exact.request.id, pointer=exact.recovery.pointer, expected=original_producer.ref,
            )
        assert not remote.rsync_calls()
        assert transported.read_ref(f"jobs/{stopped_id}") is None
        transported_entry = collect_train_job(
            remote.endpoint, transported, attempt=stopped_id,
            request_id=exact.request.id, pointer=exact.recovery.pointer, expected=observed_entry,
        )
        assert transported_entry == exact
        assert transported_entry.object_ref == exact.object_ref
        assert transported.read_ref(f"jobs/{stopped_id}")["roots"][0] == exact.object_ref.model_dump(mode="json")
        received = SuccessView(transported)
        assert received.read_attempt(producer_id) == original_producer
        assert received.authored(original_producer.receipt.request.id) is None
        assert received.attempt_ref(stopped_id) is None
        assert read_train_job(transported, stopped_id) == exact
        assert collect_success(
            remote.endpoint, transported, original_producer.ref,
            attempt=producer_id, request_id=original_producer.receipt.request.id,
        ) == original_producer
        assert SuccessView(transported).authored(original_producer.receipt.request.id) == original_producer
        assert len(remote.rsync_calls()) == 1

        configuration = saved.configuration.model_copy(update={"store": str(destination.root)})
        (location.path / saved.declaration).write_bytes(
            dump_text_project(saved.project, configuration)[saved.declaration],
        )
        local_plan = folders.read_plan(project, revision=None)
        assert local_plan.plan.requests == reviewed.plan.requests
        destination_project = {
            "project": local_plan.project_id, "store": local_plan.store_path, "project_root": local_plan.project_root,
        }
        collected = local_jobs.collect_train_attempt(source, producer, **destination_project)
        assert collected.status == JobStatus.SUCCEEDED and collected.job_id == producer_id
        assert observed_success(host.jobs(job_id=producer_id)[0], destination) == original_producer
        assert not (Path(destination.root) / "checkpoints").exists()
        plan_service = ProjectPlanService(folders, local_jobs)
        projected = plan_service.read(project, revision=local_plan.revision).history[prefix]
        assert projected.status == JobStatus.SUCCEEDED
        assert projected.result.receipt.manifest_sha256 == original_producer.ref.sha256

        advanced_id = launch(local_jobs, folders, local_plan, prefix, "local-advance")
        advanced = observed_success(terminal(host, advanced_id, "succeeded"), destination)
        assert advanced.ref != original_producer.ref
        assert advanced.receipt.supersedes == original_producer.ref
        retry = local_jobs.collect_session_attempt(source, producer, **destination_project)
        assert retry.activity.job_id == producer_id
        assert retry.evidence_sha256 == original_producer.ref.sha256
        assert SuccessView(destination).authored(original_producer.receipt.request.id) == advanced

        wrong = stopped | {"outcome": stopped["outcome"] | {"store_job_entry": producer["outcome"]["store_success"]}}
        with pytest.raises(ValueError, match="observed Store reference"):
            local_jobs.collect_session_attempt(source, wrong, **destination_project)
        assert not host.jobs(job_id=stopped_id)

        recovery_tree = exact.recovery.content.tree
        clone = ImmutableArtifactBlobProvider._clone_object_from
        with monkeypatch.context() as interrupted:
            def interrupt_after_checkpoint_descriptor(provider, other, ref):
                copied = clone(provider, other, ref)
                if ref == recovery_tree.descriptor:
                    raise InterruptedError("collection connection interrupted")
                return copied

            interrupted.setattr(ImmutableArtifactBlobProvider, "_clone_object_from", interrupt_after_checkpoint_descriptor)
            with pytest.raises(InterruptedError, match="connection interrupted"):
                local_jobs.collect_train_attempt(source, stopped, **destination_project)
        assert not host.jobs(job_id=stopped_id)
        with pytest.raises(FileNotFoundError):
            read_train_job(destination, stopped_id)

        members = {file.relative_path: file.object for file in source.get_tree(recovery_tree).files}
        manifest_path = Path(exact.recovery.pointer.manifest_relative_path)
        manifest = MethodCheckpointManifest.model_validate_json(source.get_object(members[manifest_path.as_posix()]))
        damaged = members[(manifest_path.parent / method_checkpoint_extra_path(manifest.extra)).as_posix()]
        with writable_store_object(source.object_path(damaged)) as path:
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with pytest.raises((ArtifactBlobCustodyError, OSError, ValueError)):
                local_jobs.collect_train_attempt(source, stopped, **destination_project)
            path.write_bytes(raw)
            source.verify_object(damaged)
        assert not host.jobs(job_id=stopped_id)

        with monkeypatch.context() as disconnected:
            def lose_controller(*args, **kwargs):
                raise ConnectionError("controller disconnected before acknowledgement")

            disconnected.setattr(host, "mirror_collected_train", lose_controller)
            with pytest.raises(ConnectionError, match="before acknowledgement"):
                local_jobs.collect_train_attempt(source, stopped, **destination_project)
        assert read_train_job(destination, stopped_id) == exact
        assert not host.jobs(job_id=stopped_id)

        mirrored = local_jobs.collect_train_attempt(source, stopped, **destination_project)
        assert mirrored.status == JobStatus.STOPPED and mirrored.can_resume and mirrored.outcome.receipt is None
        assert read_train_job(destination, stopped_id) == exact
        observed_cursor = host.jobs_snapshot()["cursor"]
        other_root = tmp_path / "other-project"
        other_root.mkdir()
        with pytest.raises(JobConflict):
            local_jobs.collect_session_attempt(source, stopped, **(destination_project | {"project_root": other_root}))
        retry = local_jobs.collect_session_attempt(source, stopped, **destination_project)
        assert retry.activity == mirrored
        assert retry.evidence_sha256 == stopped["outcome"]["store_job_entry"]["sha256"]
        assert host.jobs_snapshot()["cursor"] == observed_cursor
        assert SuccessView(destination).authored(original_producer.receipt.request.id) == advanced
        assert SuccessView(destination).attempt_ref(stopped_id) is None
        shutil.rmtree(source.root)
        shutil.rmtree(venue.root)

        restart(host)
        resumed = local_jobs.resume(stopped_id, "local-resume")
        assert resumed.job_id != stopped_id and resumed.continues == stopped_id
        assert resumed.request_id == exact.request.id
        result = observed_success(terminal(host, resumed.job_id, "succeeded"), destination).receipt
        assert result.trace is not None and result.provenance is not None
        assert result.request == exact.request and result.inputs == exact.inputs
        recovered = exact.recovery.pointer
        assert result.recovery == CheckpointOccurrenceRef(
            run_id=recovered.run_id, transaction_id=recovered.transaction_id, manifest_sha256=recovered.manifest_sha256,
        )
        assert result.products == control.products
        assert result.continuation.clock == control.continuation.clock
        assert result.continuation.state == control.continuation.state
