"""Real saved checkpoint selection through Plan, Host, custody and datasets."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from feedbax.contracts.effective_success import EvaluationSubjectValue
from feedbax.contracts.jobs import JobStatus
from feedbax.contracts.project import Project
from feedbax.contracts.request_identity import decode_request, effective_binding_id
from feedbax.contracts.request_refs import BatchMember, BatchSetMember, CheckpointCadence
from feedbax.contracts.rollout import ResolvedRolloutBinding
from feedbax.host.protocol import AdmissionItem, Job, Launch
from feedbax.jobs import JobService
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.evaluation_subject import evaluation_subject_objects
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.contracts.rollout import CannotRun
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored_documents, document_project
from tests.support.host import admit, held_until_released, job_row, restart, run_job, token, wait_until
from tests.support.finite_train import damaged_checkpoint_model, install_finite_gate



def _train(step, batches):
    step = deepcopy(step)
    payload = step["base"]["inline"]
    payload["training_config"].update(n_batches=batches, snapshot_interval=2)
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
        "short-selection": evaluate({"batches": [1, 2]}, source="short"),
    }
    if late:
        steps["late"] = evaluate({"batches": [2, 6]})
    return Project.model_validate(definition | {"steps": steps})


@pytest.mark.parametrize("rule", [
    {"batches": []}, {"batches": [0, 2]}, {"batches": [2, 2]},
    {"every_k": 2}, {"every_k": 2, "through": 1}, {"every_k": 1, "through": 65},
])
def test_invalid_finite_rule_is_rejected_during_project_validation(rule):
    definition = _fixture().model_dump(mode="json", by_alias=True, exclude_unset=True)
    definition["steps"]["ordered"]["inputs"]["subject"]["checkpoint"] = rule
    with pytest.raises(ValidationError):
        Project.model_validate(definition)


def test_saved_finite_selection_releases_across_resume_and_retains_exact_checkpoints(
    host, tmp_path, monkeypatch,
):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    jobs = JobService(host, new_application_registry_bundle())
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
    install_finite_gate(jobs, tmp_path, stop_at=3, resume_at=4)

    def launch(name, *steps):
        return jobs.admit_plan(folders, plan, run_keys=tuple(by_step[step].key for step in steps), token=token(name))

    def terminal(job_id, state):
        row = wait_until(lambda: (
            row if (row := job_row(host, job_id)) and row["outcome"] is not None else None
        ))
        assert row["state"] == state, row.get("outcome")
        return row

    (producer,) = launch("producer", "train").items
    source = host.jobs(job_id=producer.job_id, include_launch=True)[0]
    assert source["launch"]["payload"]["checkpoint_catalog"] == [2, 4]
    admitted = launch("consumers", "ordered", "cadence")
    assert all(item.disposition == "admitted" for item in admitted.items)
    wait_until(lambda: (tmp_path / "before-stop").exists())
    jobs.stop([producer.job_id], "checkpoint")
    stopped = terminal(producer.job_id, "stopped")
    assert stopped["outcome"]["checkpoint"]["completed_updates"] == 3
    restart(host)
    assert launch("consumers", "ordered", "cadence") == admitted
    for item in admitted.items:
        row = job_row(host, item.job_id, "queued")
        assert row["delivery"] == 0 and not row["edges"][0]["satisfied"]

    resumed = jobs.resume(producer.job_id, token=token("resume"))
    wait_until(lambda: (tmp_path / "resumed-interior").exists())
    host.policy(compute=2)
    try:
        rows = [terminal(item.job_id, "succeeded") for item in admitted.items]
        assert job_row(host, resumed.job_id, "running") is not None
    finally:
        (tmp_path / "finish-train").touch()
    terminal(resumed.job_id, "succeeded")
    restart(host)
    assert launch("consumers", "ordered", "cadence") == admitted
    events = host.events(projects=(plan.project_id,))["events"]

    position = "/v1/model/state/plant/effector~1position"
    target = "/v1/task/output/target_pos"
    service = ProjectPlanService(folders, jobs)
    settled = service.read(project, revision=plan.revision)
    trajectories, effective_bindings = [], []
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
        result = jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=row["id"])
        receipt = result._manifest()
        effective = ResolvedRolloutBinding.model_validate(receipt.metadata["effective_binding"])
        assert [member.coordinate for member in effective.members] == coordinates
        assert receipt.metadata["effective_binding_sha256"] == effective.digest()
        effective_bindings.append(effective)
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
            result.replay(trials[0])
        with pytest.raises(CannotRun, match="absent from the retained set"):
            result.replay(trials[0], checkpoint=3)
        reads = []
        read_slice = ArrayContainerReader.read_unverified_slice

        def observed(reader, field, selection):
            reads.append((field, selection))
            return read_slice(reader, field, selection)

        with monkeypatch.context() as context:
            context.setattr(ArrayContainerReader, "read_unverified_slice", observed)
            replay = result.replay(trials[1], checkpoint=4)
        assert [field for field, _ in reads] == ["checkpoint", "state_time", position]
        assert reads[-1] == (position, (coordinates.index(4), 0, 1))
        assert replay.drawn.trials[0].id == trials[1]
        np.testing.assert_allclose(replay.drawn.trials[0].states[replay.position].samples, trajectories[-1][coordinates.index(4), 0, 1])
    assert not np.array_equal(trajectories[0][0], trajectories[0][1])
    np.testing.assert_array_equal(trajectories[0], trajectories[1][::-1])
    first, second = effective_bindings
    assert first.members == second.members[::-1]
    assert len({member.model_tree for member in first.members}) == 2
    assert first.digest() != second.digest()

    store = ImmutableArtifactBlobProvider(plan.store_path)
    view = SuccessView(store)
    reversed_result, cadence_result = (view.read_attempt(row["id"]) for row in rows)
    assert reversed_result is not None and cadence_result is not None
    assert reversed_result.receipt.effective != cadence_result.receipt.effective
    fixed = decode_request(plan.plan.requests[by_step["ascending"].request_id], plan.plan.documents.__getitem__)
    (fixed_member,) = fixed.inputs["subject"]
    (bound,) = cadence_result.receipt.inputs["subject"]
    value = EvaluationSubjectValue.model_validate(bound.descriptor)
    assert value.resolved == second
    evaluation_subject_objects(store, fixed, fixed_member, bound)
    assert effective_binding_id(fixed, cadence_result.receipt.inputs) == cadence_result.receipt.effective
    (reversed_bound,) = reversed_result.receipt.inputs["subject"]
    with pytest.raises(ValueError, match="coverage differs from the authored selector"):
        evaluation_subject_objects(store, fixed, fixed_member, reversed_bound)

    source = bound.evaluation_subject
    swapped_models = bound.model_copy(update={"evaluation_subject": source.model_copy(update={
        "checkpoints": source.checkpoints[::-1],
    })})
    with pytest.raises(ValueError, match="checkpoint differs from its resolved model and coordinate"):
        evaluation_subject_objects(store, fixed, fixed_member, swapped_models)

    first_member, second_member = second.members
    wrong_trials = first_member.model_copy(update={"trials": first_member.trials[::-1]})
    wrong_binding = second.model_copy(update={"members": (wrong_trials, second_member)})
    wrong_value = value.model_copy(update={"resolved": wrong_binding})
    with pytest.raises(ValueError, match="trial binding differs from its authored parameters"):
        evaluation_subject_objects(store, fixed, fixed_member, bound.model_copy(update={
            "descriptor": wrong_value.model_dump(mode="json"),
        }))

    (short,) = launch("short-producer", "short").items
    (short_evaluation,) = launch("short-consumer", "short-selection").items
    assert short_evaluation.disposition == "admitted"
    terminal(short.job_id, "succeeded")
    short_binding = terminal(short_evaluation.job_id, "succeeded")["edges"][0]["binding"]["checkpoint_selection"]
    assert [member["checkpoint"]["completed_updates"] for member in short_binding["members"]] == [1, 2]

    _save_project(folders, project, _fixture(late=True))
    edited = folders.read_plan(project, revision=None)
    by_edited_step = {run.key.step: run for run in edited.plan.runs}
    assert by_edited_step["train"].request_id == by_step["train"].request_id
    late = by_edited_step["late"]
    assert not service.read(project, revision=edited.revision).evaluations[late.key].can_run
    (late_item,) = jobs.admit_plan(folders, edited, run_keys=(late.key,), token=token("late")).items
    assert late_item.disposition == "rejected"
    assert late_item.reason == "/inputs/subject/checkpoint: admitted Train catalogue omits requested batches"
    (stale_train,) = launch("stale-producer", "train").items
    assert stale_train.disposition == "rejected"
    assert stale_train.reason == "/checkpoint_catalog: the reviewed publication demand changed."
    train_attempts = [
        row for row in host.jobs(project=plan.project_id, include_launch=True)
        if row["request"] == by_step["train"].request_id
    ]
    assert {row["id"] for row in train_attempts} == {producer.job_id, resumed.job_id}
    assert all(row["launch"]["payload"]["checkpoint_catalog"] == [2, 4] for row in train_attempts)


def test_saved_exact_stopped_checkpoint_is_distinct_from_last(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    registries = new_application_registry_bundle()
    jobs = JobService(host, registries)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "stopped"
    definition = document_project(authored_documents(), evaluation=True)
    baseline = definition["steps"]["baseline"]

    def evaluate(rule, source="train"):
        return baseline | {"inputs": {"subject": {"step": source, "checkpoint": rule}}}

    # Last demands no checkpoint, so the Train launches with no catalogue.
    steps = {
        "train": _train(definition["steps"]["train"], 6),
        "short": _train(definition["steps"]["train"], 4),
        "last": evaluate("last"),
    }

    def save():
        _save_project(folders, project, Project.model_validate(definition | {"steps": steps}))
        saved = folders.read_plan(project, revision=None)
        assert not saved.plan.diagnostics
        return saved, {run.key.step: run for run in saved.plan.runs}

    plan, by_step = save()

    def launch(name, *names, service=jobs):
        return service.admit_plan(
            folders, plan, run_keys=tuple(by_step[step].key for step in names), token=token(name),
        ).items

    def terminal(job_id, state):
        row = wait_until(lambda: (
            row if (row := job_row(host, job_id)) and row["outcome"] is not None else None
        ))
        assert row["state"] == state, row.get("outcome")
        return row

    def pending(item):
        row = job_row(host, item.job_id, "queued")
        assert row["delivery"] == 0 and not row["edges"][0]["satisfied"]

    (held,) = launch("held", "last", service=JobService(host, registries))
    assert held.disposition == "rejected" and "/inputs/subject" in held.reason

    host.policy(compute=1)
    install_finite_gate(jobs, tmp_path, stop_at=3, resume_at=4)
    (producer,) = launch("producer", "train")
    # Authored after the Train's admission, so its catalogue holds none of them; 1 and 3 are off its cadence.
    steps |= {"exact": evaluate({"at": 3}), "past": evaluate({"at": 1}), "future": evaluate({"at": 4})}
    plan, by_step = save()
    subject = by_step["exact"].inputs["subject"].member.reference
    assert subject.select == BatchMember(batch=3)
    assert subject.request_id == by_step["train"].request_id
    last, exact = launch("consumers", "last", "exact")
    assert {last.disposition, exact.disposition} == {"admitted"}
    wait_until(lambda: (tmp_path / "before-stop").exists())
    pending(last)
    pending(exact)
    jobs.stop([producer.job_id], "checkpoint")
    stopped = terminal(producer.job_id, "stopped")
    checkpoint = stopped["outcome"]["checkpoint"]
    assert checkpoint["completed_updates"] == 3
    assert host.stopped_checkpoint(producer.job_id, project=plan.project_id, manifest_root=str(plan.store_path)) == 3
    assert "checkpoint_catalog" not in host.jobs(job_id=producer.job_id, include_launch=True)[0]["launch"]["payload"]
    evaluated = terminal(exact.job_id, "succeeded")
    binding = evaluated["edges"][0]["binding"]
    selection = binding["checkpoint_selection"]
    (member,) = selection["members"]
    assert selection["requested"] == [3] and selection["lineage"] == [producer.job_id]
    assert member["checkpoint"] == checkpoint
    assert member["producer_job_id"] == producer.job_id
    assert evaluated["delivery"] == 1
    restart(host)
    pending(last)
    assert launch("consumers", "last", "exact") == [last, exact]
    past, future = launch("around-stop", "past", "future")
    absent = terminal(past.job_id, "blocked")
    assert absent["outcome"]["error"]["code"] == "input_unavailable"
    assert absent["delivery"] == 0 and not absent["edges"][0]["satisfied"]
    pending(future)

    # A coordinate authored after admission uses the retained publication in the same lineage.
    steps["late"] = deepcopy(evaluate({"at": 3}))
    steps["late"]["base"]["inline"]["params"]["seed"] += 1
    plan, by_step = save()
    service = ProjectPlanService(folders, jobs)
    assert service.read(project, revision=plan.revision).evaluations[by_step["late"].key].can_run
    resumed = jobs.resume(producer.job_id, token=token("resume"))
    wait_until(lambda: (tmp_path / "resumed-interior").exists())
    pending(last)
    try:
        host.policy(compute=2)
        released = terminal(future.job_id, "succeeded")["edges"][0]["binding"]["checkpoint_selection"]
        assert job_row(host, resumed.job_id, "running") is not None
    finally:
        (tmp_path / "finish-train").touch()
    completed = terminal(resumed.job_id, "succeeded")
    final = terminal(last.job_id, "succeeded")
    bound_terminal = final["edges"][0]["binding"]["terminal"]
    assert bound_terminal["receipt"] == completed["outcome"]["store_success"]
    last_result = jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=last.job_id)
    with last_result.open_states() as states:
        assert states.data("checkpoint").tolist() == [6]
    assert released["requested"] == [4]
    assert released["lineage"] == [producer.job_id, resumed.job_id]
    (resumed_member,) = released["members"]
    assert resumed_member["producer_job_id"] == resumed.job_id
    restart(host)
    assert job_row(host, exact.job_id)["edges"][0]["binding"] == binding
    assert job_row(host, last.job_id)["edges"][0]["binding"]["terminal"] == bound_terminal

    result = jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=exact.job_id)
    receipt = result._manifest()
    assert receipt.input_training_runs == []
    (parent,) = receipt.provenance.parents
    assert parent.id == checkpoint["pointer"]["transaction_id"]
    assert parent.metadata["publication_cursor"] == member["publication_cursor"]
    assert parent.metadata["transaction_root_sha256"] == checkpoint["pointer"]["transaction_root_sha256"]
    settled = service.read(project, revision=plan.revision)
    planned = settled.evaluations[by_step["exact"].key].result
    assert planned.receipt == result.receipt
    (source,) = planned.checkpoint_sources
    assert source.job_id == producer.job_id and source.completed_updates == 3
    with result.open_states() as dataset:
        assert dataset.data("checkpoint").tolist() == [3]
    trial = settled.evaluations[by_step["exact"].key].trials[0]
    assert result.replay(trial).drawn.trials[0].id == trial

    # A Train that finished without update 3: it ran outside the finite gate, which would hold it at 3, before
    # anything read it, so 3 was neither demanded nor on its cadence of 2.
    (short,) = launch("short", "short", service=JobService(host, registries))
    terminal(short.job_id, "succeeded")
    steps["missing"] = evaluate({"at": 3}, source="short")
    plan, by_step = save()
    (missing,) = launch("missing", "missing")
    unavailable = terminal(missing.job_id, "blocked")
    assert unavailable["delivery"] == 0 and not unavailable["edges"][0]["satisfied"]
    assert unavailable["outcome"]["error"] == {
        "code": "input_unavailable", "path": "/inputs/subject/checkpoint",
        "message": "Producer closed without checkpoint coordinates [3].",
    }

    # Native Host continuations can branch even though Studio Resume offers a single child.
    (tmp_path / "resume-model-factor").write_text("1.25")
    [first_resume] = host.jobs(job_id=resumed.job_id, include_launch=True)
    sibling = Job(
        id=token("current-sibling"), project=plan.project_id, request=first_resume["request"],
        launch=Launch(**first_resume["launch"]), continues=producer.job_id,
        data={key: value for key, value in first_resume["data"].items() if key != "run_key"},
    )
    host.admit(
        token("resume-again"), [AdmissionItem("sibling", sibling)],
        record={"project": sibling.project, "target": first_resume["data"]["target"]},
    )
    terminal(sibling.id, "succeeded")
    future_run = by_step["future"]
    earlier = settled.evaluations[future_run.key].result
    now = service.read(project, revision=plan.revision).evaluations[future_run.key]
    assert now.status is JobStatus.STALE and now.can_run
    assert now.result.receipt == earlier.receipt
    assert {reason.kind for reason in now.result.validity.reasons} == {"input"}

    (fresh,) = jobs.admit_plan(
        folders, plan, run_keys=(future_run.key,), token=token("evaluate-current-sibling"), reuse_completed=True,
    ).items
    assert fresh.disposition == "admitted"
    future_selection = terminal(fresh.job_id, "succeeded")["edges"][0]["binding"]["checkpoint_selection"]
    assert future_selection["lineage"] == [producer.job_id, sibling.id]
    assert future_selection["members"][0]["producer_job_id"] == sibling.id
    assert service.read(project, revision=plan.revision).evaluations[future_run.key].result.validity.state == "current"

    # Check both new and already pinned consumers after valid reuse. Observed Store damage is durable.
    host.policy(compute=1)
    release = tmp_path / "release-pinned"
    gate = run_job(
        "hold-pinned", held_until_released(release),
        project=plan.project_id,
    )
    admit(host, token("hold-pinned"), gate)
    try:
        wait_until(lambda: job_row(host, gate.id, "running"))
        (late,) = launch("late", "late")
        assert late.disposition == "admitted"
        pinned = wait_until(lambda: (
            row["edges"][0]["binding"] if (row := job_row(host, late.job_id, "queued")) else None
        ))
        assert pinned == binding
        with damaged_checkpoint_model(checkpoint):
            assert host.stopped_checkpoint(producer.job_id, project=plan.project_id, manifest_root=str(plan.store_path)) is None
            (corrupt,) = launch("corrupt", "exact")
            rejected = terminal(corrupt.job_id, "blocked")
            assert not rejected["edges"][0]["satisfied"]
            release.touch()
            refused = terminal(late.job_id, "blocked")
            assert refused["edges"][0]["binding"] == binding
            for row in (rejected, refused):
                assert row["outcome"]["error"]["code"] == "dependency_corrupt"
                assert row["delivery"] == 0 and "receipt" not in row["outcome"]
    finally:
        release.touch()


def test_saved_fixed_evaluate_reuses_only_equal_current_train_model(host, tmp_path, monkeypatch):
    from dataclasses import replace

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1)
    jobs = JobService(host, new_application_registry_bundle())
    folders = ProjectFolderService(tmp_path / "projects")
    project = "fixed-current"
    raw = document_project(authored_documents(), evaluation=True)
    raw["steps"]["baseline"]["inputs"]["subject"]["checkpoint"] = {"at": 2}
    _save_project(folders, project, Project.model_validate(raw))
    saved = folders.read_plan(project, revision=None)
    by_step = {run.key.step: run for run in saved.plan.runs}
    train, evaluate = by_step["train"], by_step["baseline"]
    service = ProjectPlanService(folders, jobs)

    def submit(name, *runs, reuse=False):
        return jobs.admit_plan(
            folders, saved, run_keys=tuple(run.key for run in runs), token=token(name), reuse_completed=reuse,
        ).items

    def succeed(item):
        assert item.disposition == "admitted", item
        row = wait_until(lambda: (
            row if (row := job_row(host, item.job_id)) and row["outcome"] is not None else None
        ))
        assert row["state"] == "succeeded", row.get("outcome")
        return row

    initial_train, initial_evaluation = submit("initial", train, evaluate)
    succeed(initial_train)
    succeed(initial_evaluation)
    initial = service.read(project, revision=saved.revision).evaluations[evaluate.key]
    assert initial.status is JobStatus.SUCCEEDED
    assert initial.result.validity.state == "current"

    (identical,) = submit("identical", train)
    succeed(identical)
    same = service.read(project, revision=saved.revision).evaluations[evaluate.key]
    assert same.can_run and same.result.validity.state == "current"
    assert same.result.receipt == initial.result.receipt
    (reused,) = submit("reuse", evaluate, reuse=True)
    assert reused.disposition == "reused"

    (tmp_path / "changed_model.py").write_text("""
import equinox as eqx
import jax
from feedbax.training.method_runtime import TrainingLoop

run_block = TrainingLoop.run_block

def revised(self, state, *, completed, count):
    result = run_block(self, state, completed=completed, count=count)
    model = jax.tree.map(
        lambda value: value * 1.25 if eqx.is_inexact_array(value) else value,
        result.state.model,
    )
    return result._replace(state=result.state._replace(model=model))

TrainingLoop.run_block = revised
""")
    worker = jobs._training_worker
    jobs._training_worker = replace(
        worker, preload=(*worker.preload, "changed_model"),
        env=worker.env | {"PYTHONPATH": f"{tmp_path}:{Path.cwd()}"},
    )

    (changed,) = submit("changed", train)
    succeed(changed)
    stale = service.read(project, revision=saved.revision).evaluations[evaluate.key]
    assert stale.status is JobStatus.STALE and stale.can_run
    assert stale.result.receipt == initial.result.receipt
    assert {(reason.kind, reason.role) for reason in stale.result.validity.reasons} == {("input", "subject")}

    (replacement,) = submit("refresh", evaluate, reuse=True)
    rerun = succeed(replacement)
    [edge] = rerun["edges"]
    assert edge["binding"]["checkpoint_selection"]["members"][0]["producer_job_id"] == changed.job_id
    assert service.read(project, revision=saved.revision).evaluations[evaluate.key].result.validity.state == "current"
