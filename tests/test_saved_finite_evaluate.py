"""Real saved finite selection through Plan, Host, checkpoint custody and datasets."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from feedbax.contracts.project import Project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import BatchSetMember, CheckpointCadence
from feedbax.jobs import JobService
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_evaluate import CannotRun
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored_documents, document_project
from tests.test_local_job_host import host as host_fixture, job_row, restart, wait_until
from tests.test_private_checkpoint_evaluation import install_finite_gate

host = host_fixture


def _train(step, batches):
    step = deepcopy(step)
    payload = step["base"]["inline"]
    payload["training_config"].update(n_batches=batches, snapshot_interval=2)
    payload["method_payload"]["payload"]["target_total"] = batches
    return step


def _fixture(*, late=False):
    definition = document_project(authored_documents(), evaluation=True)
    baseline = definition["steps"]["baseline"]

    def evaluate(rule, source="train"):
        step = deepcopy(baseline)
        step["inputs"]["subject"] = {"step": source, "checkpoint": rule}
        return step

    steps = {
        "train": _train(definition["steps"]["train"], 6),
        "short": _train(definition["steps"]["train"], 2),
        "ordered": evaluate({"batches": [4, 2]}),
        "cadence": evaluate({"every_k": 2, "through": 4}),
        "ascending": evaluate({"batches": [2, 4]}),
        "missing": evaluate({"batches": [1, 2]}, source="short"),
    }
    if late:
        steps["late"] = evaluate({"batches": [2, 6]})
    return Project.model_validate(definition | {"steps": steps})


@pytest.mark.parametrize("rule", [
    {"batches": []}, {"batches": [0, 2]}, {"batches": [2, 2]},
    {"every_k": 2}, {"every_k": 2, "through": 1}, {"every_k": 1, "through": 65},
])
def test_invalid_finite_rule_is_rejected_during_project_validation(rule):
    definition = _fixture().model_dump(mode="json", exclude_unset=True)
    definition["steps"]["ordered"]["inputs"]["subject"]["checkpoint"] = rule
    with pytest.raises(ValidationError):
        Project.model_validate(definition)


def test_saved_finite_selection_releases_across_resume_and_retains_exact_checkpoints(
    host, tmp_path, monkeypatch,
):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    jobs = JobService(host, new_application_registry_bundle(), enable_evaluation_candidate=True)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "finite"
    _save_project(folders, project, _fixture())
    plan = folders.read_plan(project, revision=None)
    assert not plan.plan.diagnostics and not plan.store_path.exists()
    by_step = {run.key.step: run for run in plan.plan.runs}
    for name, selector in (
        ("ordered", BatchSetMember(batches=(4, 2))),
        ("ascending", BatchSetMember(batches=(2, 4))),
        ("cadence", BatchSetMember(batches=(2, 4), cadence=CheckpointCadence(every_k=2, through=4))),
    ):
        run = by_step[name]
        encoded = decode_request(plan.plan.requests[run.request_id], plan.plan.documents.__getitem__)
        (member,) = encoded.inputs["subject"]
        assert member.reference.select == selector
        assert member.reference.request_id == by_step["train"].request_id
        assert member == run.inputs["subject"].member
    assert len({by_step[name].request_id for name in ("ordered", "ascending", "cadence")}) == 3

    host.policy(compute=1)
    install_finite_gate(jobs, tmp_path, stop_at=3, resume_at=4, managed=True)

    def launch(token, *steps):
        return jobs.admit_plan(folders, plan, run_keys=tuple(by_step[name].key for name in steps), token=token)

    def terminal(job_id, state):
        row = wait_until(lambda: (
            row if (row := job_row(host, job_id)) and row["outcome"] is not None else None
        ), timeout=22)
        assert row["state"] == state, row.get("outcome")
        return row

    (producer,) = launch("producer", "train").items
    source = host.jobs(job_id=producer.job_id, include_launch=True)[0]
    assert source["launch"]["payload"]["checkpoint_catalog"] == [2, 4]
    admitted = launch("consumers", "ordered", "cadence")
    assert all(item.disposition == "admitted" for item in admitted.items)
    wait_until(lambda: (tmp_path / "before-stop").exists(), timeout=16)
    jobs.stop([producer.job_id], "checkpoint")
    stopped = terminal(producer.job_id, "stopped")
    assert stopped["outcome"]["checkpoint"]["completed_updates"] == 3
    restart(host)
    assert launch("consumers", "ordered", "cadence") == admitted
    for item in admitted.items:
        row = job_row(host, item.job_id, "queued")
        assert row["delivery"] == 0 and not row["edges"][0]["satisfied"]

    resumed = jobs.resume(producer.job_id, token="resume")
    wait_until(lambda: (tmp_path / "resumed-interior").exists(), timeout=16)
    host.policy(compute=2)
    try:
        rows = [terminal(item.job_id, "succeeded") for item in admitted.items]
        assert job_row(host, resumed.job_id, "running") is not None
    finally:
        (tmp_path / "finish-train").touch()
    terminal(resumed.job_id, "succeeded")
    restart(host)
    assert launch("consumers", "ordered", "cadence") == admitted
    events = host.events(projects=(project,))["events"]

    position = "/v1/model/state/plant/effector~1position"
    target = "/v1/task/output/target_pos"
    service = ProjectPlanService(folders, jobs)
    settled = service.read(project, revision=plan.revision)
    trajectories = []
    for name, row, coordinates, attempts in zip(
        ("ordered", "cadence"), rows, ([4, 2], [2, 4]),
        ([resumed.job_id, producer.job_id], [producer.job_id, resumed.job_id]), strict=True,
    ):
        binding = row["edges"][0]["binding"]["checkpoint_selection"]
        assert binding["requested"] == coordinates
        assert binding["lineage"] == [producer.job_id, resumed.job_id]
        assert [item["producer_job_id"] for item in binding["members"]] == attempts
        assert row["delivery"] == 1
        assert [
            event["data"] for event in events
            if event["job_id"] == row["id"] and event["kind"] == "dependency_bound"
        ] == [row["edges"][0]["binding"]]
        result = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=row["id"])
        receipt = result._manifest()
        assert receipt.input_training_runs == []
        assert receipt.provenance.metadata["checkpoint_selection"]["lineage_tip"] == resumed.job_id
        assert [item.id for item in receipt.provenance.parents] == [
            item["checkpoint"]["pointer"]["transaction_id"] for item in binding["members"]
        ]
        assert [item.metadata["publication_cursor"] for item in receipt.provenance.parents] == [
            item["publication_cursor"] for item in binding["members"]
        ]
        assert [item.metadata["manifest_sha256"] for item in receipt.provenance.parents] == [
            item["checkpoint"]["pointer"]["manifest_sha256"] for item in binding["members"]
        ]
        planned = settled.evaluations[by_step[name].key].result
        assert planned.receipt == result.receipt
        assert planned.checkpoint_source is None
        assert [item.completed_updates for item in planned.checkpoint_sources] == coordinates
        with result.open_states() as dataset:
            assert dataset.descriptor.dimensions["checkpoint"] == 2
            assert dataset.descriptor.dimensions["state_time"] == 81
            assert dataset.data("checkpoint").tolist() == coordinates
            np.testing.assert_array_equal(dataset.data(target)[0], dataset.data(target)[1])
            trajectories.append(dataset.data(position))
            np.testing.assert_array_equal(
                dataset.select(position, labels={"checkpoint": 4, "trial": settled.evaluations[by_step[name].key].trials[0]})[0],
                trajectories[-1][coordinates.index(4), 0, 0],
            )
        trials = settled.evaluations[by_step[name].key].trials
        with pytest.raises(CannotRun, match="choose a retained checkpoint"):
            result.replay_trial(trials[0], anchor_id="effector")
        with pytest.raises(CannotRun, match="absent from the retained set"):
            result.replay_trial(trials[0], anchor_id="effector", checkpoint=3)
        reads = []
        original = ArrayContainerReader.read_tensor

        def observed(reader, field):
            reads.append(field)
            return original(reader, field)

        with monkeypatch.context() as context:
            context.setattr(ArrayContainerReader, "read_tensor", observed)
            replay = result.replay_trial(trials[1], anchor_id="effector", checkpoint=4)
        assert reads == ["checkpoint", "state_time", position]
        assert replay.identity.stable_id == trials[1] and replay.identity.index == 1
        np.testing.assert_allclose(replay.tracks[0].samples, trajectories[-1][coordinates.index(4), 0, 1])
    assert not np.array_equal(trajectories[0][0], trajectories[0][1])
    np.testing.assert_array_equal(trajectories[0], trajectories[1][::-1])

    (short,) = launch("short-producer", "short").items
    (missing,) = launch("missing-consumer", "missing").items
    assert missing.disposition == "admitted"
    terminal(short.job_id, "succeeded")
    refused = terminal(missing.job_id, "blocked")
    assert refused["delivery"] == 0 and not refused["edges"][0]["satisfied"]
    assert refused["outcome"]["error"] == {
        "code": "input_unavailable", "path": "/inputs/subject/checkpoint",
        "message": "Producer closed without checkpoint coordinates [1].",
    }

    _save_project(folders, project, _fixture(late=True))
    edited = folders.read_plan(project, revision=None)
    by_edited_step = {run.key.step: run for run in edited.plan.runs}
    assert by_edited_step["train"].request_id == by_step["train"].request_id
    late = by_edited_step["late"]
    assert not service.read(project, revision=edited.revision).evaluations[late.key].can_run
    (late_item,) = jobs.admit_plan(folders, edited, run_keys=(late.key,), token="late").items
    assert late_item.disposition == "rejected"
    assert late_item.reason == "/inputs/subject/checkpoint: admitted Train catalogue omits requested batches"
    (stale_train,) = launch("stale-producer", "train").items
    assert stale_train.disposition == "rejected"
    assert stale_train.reason == "/checkpoint_catalog: the reviewed publication demand changed."
    train_attempts = [
        row for row in host.jobs(project=project, include_launch=True)
        if row["request"] == by_step["train"].request_id
    ]
    assert {row["id"] for row in train_attempts} == {producer.job_id, resumed.job_id}
    assert all(row["launch"]["payload"]["checkpoint_catalog"] == [2, 4] for row in train_attempts)
