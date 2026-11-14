"""Saved Best evaluates exact ranked Train results through Plan, Host and Store."""

import pytest

from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.manifest import canonical_manifest_path
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import PlanRequests, compile_project
from feedbax.contracts.project_plan_io import dump_plan_requests, load_plan_requests
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import BestRank, ProductRef
from feedbax.contracts.studio_project import StudioProjectPatchBatch
from feedbax.contracts.studio_subset import StudioSubsetToRowsRequest
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef
from feedbax.host.best_selection import rank_train_metric
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.evaluation_subject import evaluation_subject_objects
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_local_job_host import admit, host as host_fixture, job_row, restart, run_job, wait_until
from tests.test_saved_update_series import definition

host = host_fixture


def authored_best():
    raw = definition("last")
    path = "method_payload.payload.optimizer.params.learning_rate"
    raw["steps"]["train"]["sweep"] = {"axes": {"rate": {"values": {
        name: {"value": value, "patches": {path: {"value": value}}}
        for name, value in (("a", 0.003), ("b", 0.002), ("c", 0.001))
    }}}}
    subject = raw["steps"]["baseline"]["inputs"]["subject"]
    subject["where"] = {"best": {"by": "train_loss", "k": 2, "order": "min"}}
    raw["steps"]["baseline"]["sweep"] = {"axes": {"subject": {"input": "subject"}}}
    return raw


def test_best_plan_identity_changes_with_candidates_and_refuses_unsupported_rules():
    raw = authored_best()
    before = compile_project(Project.model_validate(raw))
    assert not before.diagnostics
    plan_requests = PlanRequests.from_plan(before)
    decoded, _ = load_plan_requests(dump_plan_requests(plan_requests, None))
    assert decoded == plan_requests
    leaf = raw["steps"]["train"]["sweep"]["axes"]["rate"]["values"]
    edit = "method_payload.payload.optimizer.params.learning_rate"
    leaf["d"] = {"patches": {edit: {"value": .004}}}
    after = compile_project(Project.model_validate(raw))
    assert not after.diagnostics
    old = {row.key: row.request_id for row in before.runs}
    new = {row.key: row.request_id for row in after.runs}
    assert all((old[run.key] == new[run.key]) == (run.kind == "train") for run in before.runs)
    subject = raw["steps"]["baseline"]["inputs"]["subject"]
    subject["where"]["best"]["per"] = ["rate"]
    refused = compile_project(Project.model_validate(raw))
    assert [item.path for item in refused.diagnostics] == ["inputs.subject.where.best"]
    subject["where"] = {"when": {"metric": "train_loss", "op": "lt", "value": 1}}
    refused = compile_project(Project.model_validate(raw))
    assert [item.path for item in refused.diagnostics] == ["inputs.subject.where.when"]


def test_saved_best_binds_and_freezes_ordered_ranks(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1)
    project = "saved-best"
    folders = ProjectFolderService(tmp_path / "projects")
    jobs = JobService(host, new_application_registry_bundle())
    service = ProjectPlanService(folders, jobs)
    raw = authored_best()
    _save_project(folders, project, Project.model_validate(raw))
    saved = folders.read_plan(project, revision=None)
    plan = saved.plan
    assert not plan.diagnostics, plan.diagnostics
    trains = {run.key.row: run for run in plan.runs if run.kind == "train"}
    rankings = [run for run in plan.runs if run.kind == "evaluate"]
    assert list(trains) == ["rate=a", "rate=b", "rate=c"]
    assert [run.inputs["subject"].member.key for run in rankings] == ["rank-1", "rank-2"]
    request_objects = [
        decode_request(plan.requests[run.request_id], plan.documents.__getitem__) for run in rankings
    ]
    rule, second = (request.inputs["subject"][0].reference for request in request_objects)
    assert isinstance(rule, BestRank) and isinstance(second, BestRank)
    assert rule.rank == 1 and second.rank == 2
    assert rule.winner([1, 1, 1]) == 0 and second.winner([1, 1, 1]) == 1
    assert [item.reference.request_id for item in rule.candidates] == [run.request_id for run in trains.values()]
    assert all(run.inputs["subject"].producer is None for run in rankings)

    release = tmp_path / "release"
    gate = run_job(
        "gate", f"from pathlib import Path\nimport time\nwhile not Path({str(release)!r}).exists(): time.sleep(.01)",
        project=project,
    )
    admit(host, "gate", gate)
    wait_until(lambda: job_row(host, gate.id, "running"))
    keys = tuple(run.key for run in (*trains.values(), *rankings))
    admitted = jobs.admit_plan(folders, saved, run_keys=keys, token="train-and-rank")
    assert [item.disposition for item in admitted.items] == ["admitted"] * 5
    attempts = [item.job_id for item in admitted.items]
    for job_id in attempts[-2:]:
        row = job_row(host, job_id, "queued")
        assert row is not None and len(row["edges"]) == 3
        assert all(not edge["satisfied"] for edge in row["edges"])
    release.touch()
    for job_id in attempts:
        row = wait_until(lambda: (
            current if (current := job_row(host, job_id))["outcome"] is not None else None
        ), timeout=22)
        assert row["state"] == "succeeded", row["outcome"]

    scores = {}
    for row_name, job_id in zip(trains, attempts[:3], strict=True):
        receipt = job_row(host, job_id)["outcome"]["receipt"]
        path = canonical_manifest_path("TrainingRunManifest", receipt["manifest_id"], root=saved.store_path)
        scores[row_name] = rank_train_metric(path.read_bytes(), JobReceipt.model_validate(receipt), job_id, rule.metric)
    ranked = sorted(scores, key=scores.__getitem__)[:2]
    assert ranked != list(trains)[:2]
    view = SuccessView(ImmutableArtifactBlobProvider(saved.store_path))
    successes = []
    for run, decoded, job_id, winner in zip(rankings, request_objects, attempts[-2:], ranked, strict=True):
        row = job_row(host, job_id)
        binding, = [edge["binding"] for edge in row["edges"] if edge.get("binding")]
        assert binding["completion"]["request_id"] == trains[winner].request_id
        assert len(binding["best_rank"]["candidates"]) == 3
        success = view.read_attempt(job_id)
        bound, = success.receipt.inputs["subject"]
        assert bound.key == decoded.inputs["subject"][0].key
        assert len(bound.evaluation_subject.ranked_candidates) == 3
        assert bound.evaluation_subject.producer.id == trains[winner].request_id
        ranked_source = bound.evaluation_subject
        closure = ReceiptClosure(view.store).walk(StoreClosureRef(
            roots=(ReceiptRoot(receipt=success.ref),), include_ancestry=True,
        ))
        for candidate in ranked_source.ranked_candidates:
            assert candidate.receipt in closure and candidate.checkpoint.train_binding in closure
            assert candidate.producer.object in closure and candidate.checkpoint.transaction in closure
        primary, unrelated, third = ranked_source.ranked_candidates
        swapped = primary.model_copy(update={"receipt": unrelated.receipt, "attempt": unrelated.attempt})
        changed = ranked_source.model_copy(update={"ranked_candidates": (swapped, unrelated, third)})
        with pytest.raises(ValueError, match="exact checkpoint"):
            evaluation_subject_objects(
                view.store, decoded, decoded.inputs["subject"][0], bound.model_copy(update={"evaluation_subject": changed}),
            )
        with jobs.read_evaluation_result(project=project, store=saved.store_path, job_id=job_id).open_states() as states:
            assert states.data("trial").tolist() == decoded.spec.params["trials"]
        successes.append(success)

    restart(host)
    response = service.response(service.read(project, revision=saved.revision), include_window_map=True)
    card = next(step for step in response.window_map.steps if step.step_id == "baseline")
    planned_subset = card.inputs[0].subset
    assert (planned_subset.count, planned_subset.total) == (2, 3)
    assert planned_subset.ranks == ["rank-1", "rank-2"]
    by_key = {item.key.row: item for item in response.window_map.evaluations}
    best_results = [by_key[run.key.row].result for run in rankings]
    assert [item.bound_subject.row for item in best_results] == ranked
    assert len({item.bound_subject.selection_sha256 for item in best_results}) == 1
    assert [item.job_id for item in best_results] == attempts[-2:]
    assert [item.bound_subject.consumer_job_id for item in best_results] == attempts[-2:]

    with pytest.raises(ProjectFolderInvalid, match="complete Best selection"):
        service.subset_to_rows(project, StudioSubsetToRowsRequest(
            revision=saved.revision, variant="default", step="baseline", attempts=[attempts[-1]] * 2,
        ))
    edit = service.subset_to_rows(project, StudioSubsetToRowsRequest(
        revision=saved.revision, variant="default", step="baseline", attempts=attempts[-2:][::-1],
    ))
    snapshot = folders.patch(project, StudioProjectPatchBatch(revision=edit.revision, files=edit.files))
    selected = snapshot.project.steps["baseline"].inputs["subject"].where
    assert [(item.key, item.variant, item.row) for item in selected.rows] == [
        (f"rank-{rank}", "default", winner) for rank, winner in enumerate(ranked, start=1)
    ]
    frozen = folders.read_plan(project, revision=None).plan
    assert not frozen.diagnostics, frozen.diagnostics
    fixed = [run for run in frozen.runs if run.kind == "evaluate"]
    assert [run.key for run in fixed] == [run.key for run in rankings]
    assert [run.inputs["subject"].producer.row for run in fixed] == ranked
    assert all(isinstance(run.inputs["subject"].member.reference, ProductRef) for run in fixed)
    assert all(fixed_run.request_id != measured.request_id
               for fixed_run, measured in zip(fixed, rankings, strict=True))
    after_freeze = jobs.admit_plan(
        folders, folders.read_plan(project, revision=snapshot.revision),
        run_keys=(fixed[0].key,), token="frozen",
    ).items[0]
    assert after_freeze.disposition == "admitted" and after_freeze.job_id != attempts[-2]
    wait_until(lambda: job_row(host, after_freeze.job_id, "succeeded"), timeout=22)
    updated = SuccessView(ImmutableArtifactBlobProvider(saved.store_path))
    assert updated.read_attempt(after_freeze.job_id).receipt.effective == successes[0].receipt.effective
