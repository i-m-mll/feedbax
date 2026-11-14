"""Rank each known group independently in saved Evaluate."""

from copy import deepcopy
import json

import pytest

from feedbax.cli import main
from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.manifest import canonical_manifest_path
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import ElementChange, PlanRequests, compile_project, diff_plan_requests
from feedbax.contracts.project_plan_io import dump_plan_requests, load_plan_requests
from feedbax.contracts.project_sources import ValueSource
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.request_refs import BestRank, ProductRef
from feedbax.contracts.studio_project import StudioProjectPatchBatch
from feedbax.contracts.studio_subset import StudioSubsetToRowsRequest
from feedbax.host.best_selection import rank_train_metric
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_local_job_host import admit, host as host_fixture, job_row, restart, run_job, wait_until
from tests.test_saved_best_evaluate import authored_best

host = host_fixture


def grouped_project():
    raw = authored_best()
    train = raw["steps"]["train"]
    train["base"]["inline"]["method_payload"]["payload"]["seed"] = 0
    axes = train["sweep"]["axes"]
    del axes["rate"]["values"]["b"]
    axes["seed"] = {"values": {
        name: {"value": seed, "patches": {"method_payload.payload.seed": {"value": seed}}}
        for name, seed in (("one", 1), ("two", 2))
    }}
    raw["steps"]["baseline"]["inputs"]["subject"]["where"]["best"]["per"] = ["seed"]
    return raw


def evaluation_ranks(plan):
    return {
        run.inputs["subject"].member.key: run for run in plan.runs if run.kind == "evaluate"
    }


def _rank_review(root, target, plan, capsys):
    assert main(["plan", str(root), "--out", str(target)]) == 0
    capsys.readouterr()
    data = target.read_bytes()
    review, variants = load_plan_requests(data)
    expected = PlanRequests.from_plan(plan)
    assert review == expected
    assert tuple(review.entries) == tuple(expected.entries)
    assert dump_plan_requests(review, variants) == data
    return review


def test_grouped_rank_plan_preserves_group_keys_and_diagnoses_invalid_groups():
    raw = grouped_project()
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    ranks = evaluation_ranks(plan)
    subset = plan.subsets["default", "baseline", "subject"]
    assert [dict(group.coordinates) for group in subset.rank_groups] == [{"seed": "one"}, {"seed": "two"}]
    assert tuple(ranks) == subset.ranks
    assert len(ranks) == 4
    for group in subset.rank_groups:
        for position, key in enumerate(group.ranks, start=1):
            run = ranks[key]
            assert {axis: coordinate.value_id for axis, coordinate in run.coordinates.items()} == group.coordinates
            request = decode_request(plan.requests[run.request_id], plan.documents.__getitem__)
            rank = request.inputs["subject"][0].reference
            assert isinstance(rank, BestRank) and rank.group == group.coordinates and rank.rank == position
            assert encode_request(request.spec, request.inputs).data == plan.requests[run.request_id]
            assert len(rank.candidates) == 2
            assert {item.row.split("seed=")[1] for item in rank.candidates} == {group.coordinates["seed"]}
    named = deepcopy(raw)
    subject = named["steps"]["baseline"]["inputs"]["subject"]
    named["documents"]["ranked"] = {"schema": "subset", "payload": subject["where"]}
    subject["where"] = {"subset": "ranked"}
    named_plan = compile_project(Project.model_validate(named), source_runs=(next(iter(ranks.values())).key,))
    assert named_plan.requests == plan.requests
    assert named_plan.subsets["default", "baseline", "subject"].source.origin == ValueSource(
        "document", "ranked", "/best",
    )
    sources, = named_plan.sources.values()
    assert sources["/inputs/subject/where"] == ValueSource("document", "ranked", "/best")
    assert "/inputs/subject" not in sources

    named["sweep"] = {"rows": {
        "default": {},
        "reverse": {"patches": {"documents.ranked.payload.best.order": {"value": "max"}}},
    }}
    varied = Project.model_validate(named)
    unnamed_plan = compile_project(varied)
    named_key = next(run.key for run in unnamed_plan.runs if (run.key.variant, run.kind) == ("reverse", "evaluate"))
    projected = compile_project(varied, source_runs=(named_key,))
    origin = ValueSource("variant", "reverse", "/documents/ranked/payload/best/order", variant="reverse")
    assert projected.sources[named_key]["/inputs/subject/where"] == origin
    review = PlanRequests.from_plan(unnamed_plan)
    restored, scope = load_plan_requests(dump_plan_requests(review, ("reverse",)))
    assert scope == ("reverse",)
    assert list(restored.subset_sources) == [("reverse", "baseline", "subject")]
    assert restored.subset_sources["reverse", "baseline", "subject"].origin == origin

    added = deepcopy(raw)
    axes = added["steps"]["train"]["sweep"]["axes"]
    axes["seed"]["values"]["three"] = {
        "value": 3, "patches": {"method_payload.payload.seed": {"value": 3}},
    }
    larger = evaluation_ranks(compile_project(Project.model_validate(added)))
    assert len(larger) == 6
    assert all(larger[key].request_id == run.request_id for key, run in ranks.items())

    changed = deepcopy(raw)
    changed["steps"]["train"]["sweep"]["exceptions"] = {
        "rate-for-one": {"where": {"axes": {"seed": ["one"]}}, "patches": {
            "training_config.n_reach_steps": {"value": 81},
        }},
    }
    revised = compile_project(Project.model_validate(changed))
    assert not revised.diagnostics, revised.diagnostics
    current = evaluation_ranks(revised)
    assert set(current) == set(ranks)
    assert all((run.request_id != current[key].request_id) == (run.coordinates["seed"].value_id == "one")
               for key, run in ranks.items())

    best = raw["steps"]["baseline"]["inputs"]["subject"]["where"]["best"]
    for per in (["unknown"], ["seed", "seed"]):
        best["per"] = per
        invalid = compile_project(Project.model_validate(raw))
        assert [problem.path for problem in invalid.diagnostics] == ["inputs.subject.where.best.per"]
    best["per"] = ["seed", "rate"]
    invalid = compile_project(Project.model_validate(raw))
    diagnostic, = invalid.diagnostics
    assert diagnostic.path == "inputs.subject.where.best" and "1 candidates in group" in diagnostic.detail
    best["k"] = 1
    multiple_axes = compile_project(Project.model_validate(raw))
    assert not multiple_axes.diagnostics
    assert len(evaluation_ranks(multiple_axes)) == 4
    assert all(set(run.coordinates) == {"seed", "rate"} for run in evaluation_ranks(multiple_axes).values())
    best["per"].reverse()
    reordered_axes = compile_project(Project.model_validate(raw))
    assert reordered_axes.requests == multiple_axes.requests

    best["per"] = ["seed"]
    del raw["steps"]["baseline"]["sweep"]
    invalid = compile_project(Project.model_validate(raw))
    diagnostic, = invalid.diagnostics
    assert diagnostic.path == "inputs.subject.where.best" and "subject input axis" in diagnostic.detail


def test_saved_grouped_best_binds_and_freezes_each_group(host, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1)
    folders = ProjectFolderService(tmp_path / "projects")
    jobs = JobService(host, new_application_registry_bundle())
    service = ProjectPlanService(folders, jobs)
    project = "grouped-best"
    _save_project(folders, project, Project.model_validate(grouped_project()))
    saved = folders.read_plan(project, revision=None)
    assert not saved.plan.diagnostics, saved.plan.diagnostics
    trains = [run for run in saved.plan.runs if run.kind == "train"]
    ranks = evaluation_ranks(saved.plan)
    groups = saved.plan.subsets["default", "baseline", "subject"].rank_groups
    source_key = ("default", "baseline", "subject")
    before_origin = ValueSource("step", "baseline", "/inputs/subject/where/best", step="baseline")
    after_origin = ValueSource("step", "baseline", "/inputs/subject/where/rows", step="baseline")
    root = folders.root / project
    baseline = tmp_path / "best.json"
    before_review = _rank_review(root, baseline, saved.plan, capsys)
    assert before_review.subset_sources[source_key].origin == before_origin
    assert before_review.subset_sources[source_key].producer_step == "train"
    assert not before_review.subset_sources[source_key].rows
    named_keys = tuple(run.key for run in ranks.values())
    named_plan = service.read(project, revision=saved.revision, source_runs=named_keys).settled.plan
    assert set(named_plan.sources) == set(named_keys)
    for run in ranks.values():
        input_ = run.inputs["subject"]
        expected = [
            (key, next(train.request_id for train in trains if train.key == key))
            for key in input_.candidates
        ]
        binding, = decode_request(
            before_review.requests[run.request_id], before_review.documents.__getitem__,
        ).inputs["subject"]
        reference = binding.reference
        assert isinstance(reference, BestRank) and binding.key == input_.member.key
        assert [(item.variant, item.row, item.reference.request_id) for item in reference.candidates] == [
            (key.variant, key.row, request_id) for key, request_id in expected
        ]
        origins = named_plan.sources[run.key]
        assert origins["/inputs/subject/where"] == before_origin
        assert "/inputs/subject" not in origins

    release = tmp_path / "release"
    gate = run_job("gate", f"from pathlib import Path\nimport time\nwhile not Path({str(release)!r}).exists(): time.sleep(.01)",
                   project=project)
    admit(host, "gate", gate)
    wait_until(lambda: job_row(host, gate.id, "running"))
    runs = [*trains, *ranks.values()]
    submitted = jobs.admit_plan(folders, saved, run_keys=tuple(run.key for run in runs), token="grouped")
    assert [item.disposition for item in submitted.items] == ["admitted"] * len(runs)
    ids = {run.key: item.job_id for run, item in zip(runs, submitted.items, strict=True)}
    for run in ranks.values():
        row = job_row(host, ids[run.key], "queued")
        sources = [ids[key] for key in run.inputs["subject"].candidates]
        assert {edge["job_id"] for edge in row["edges"]} == set(sources)
        assert len(sources) == 2 and all(not edge["satisfied"] for edge in row["edges"])
    release.touch()
    for run in runs:
        row = wait_until(lambda: (
            current if (current := job_row(host, ids[run.key]))["outcome"] is not None else None
        ), timeout=28)
        assert row["state"] == "succeeded", row["outcome"]
    scores = {}
    for run in trains:
        attempt = ids[run.key]
        receipt = JobReceipt.model_validate(job_row(host, attempt)["outcome"]["receipt"])
        path = canonical_manifest_path("TrainingRunManifest", receipt.manifest_id, root=saved.store_path)
        scores[run.key] = rank_train_metric(path.read_bytes(), receipt, attempt, "train_loss")

    restart(host)
    response = service.response(service.read(project, revision=saved.revision), include_window_map=True)
    subset = next(step for step in response.window_map.steps if step.step_id == "baseline").inputs[0].subset
    assert tuple(subset.ranks) == tuple(ranks)
    assert [(item.coordinates, tuple(item.ranks)) for item in subset.rank_groups] == [
        (group.coordinates, group.ranks) for group in groups
    ]
    bindings = {row.key.row: row.bound_subject for row in response.window_map.evaluations}
    view = SuccessView(ImmutableArtifactBlobProvider(saved.store_path))
    expected = {}
    snapshots = []
    for group in groups:
        candidates = ranks[group.ranks[0]].inputs["subject"].candidates
        winners = sorted(candidates, key=scores.__getitem__)
        digests = set()
        for key, winner in zip(group.ranks, winners, strict=True):
            run = ranks[key]
            bound_subject = bindings[run.key.row]
            assert bound_subject.key == key and bound_subject.row == winner.row
            assert bound_subject.producer_job_id == ids[winner] and bound_subject.completed_updates == 4
            proof, = view.read_attempt(ids[run.key]).receipt.inputs["subject"]
            assert len(proof.evaluation_subject.ranked_candidates) == 2
            assert [claim.attempt for claim in proof.evaluation_subject.ranked_candidates] == [
                ids[candidate] for candidate in candidates
            ]
            digests.add(bound_subject.selection_sha256)
            expected[key] = winner
        digest, = digests
        snapshots.append(digest)
    assert len(set(snapshots)) == 2

    attempts = [ids[run.key] for run in reversed(tuple(ranks.values()))]
    body = StudioSubsetToRowsRequest(revision=saved.revision, variant="default", step="baseline", attempts=attempts)
    with pytest.raises(ProjectFolderInvalid, match="every Best rank"):
        service.subset_to_rows(project, body.model_copy(update={"attempts": attempts[1:]}))
    recipe = service.subset_to_rows(project, body)
    snapshot = folders.patch(project, StudioProjectPatchBatch(revision=recipe.revision, files=recipe.files))
    rows = snapshot.project.steps["baseline"].inputs["subject"].where.rows
    assert [(row.key, row.variant, row.row) for row in rows] == [
        (key, winner.variant, winner.row) for key, winner in expected.items()
    ]
    frozen = folders.read_plan(project, revision=snapshot.revision)
    fixed = evaluation_ranks(frozen.plan)
    assert tuple(fixed) == tuple(ranks)
    after_review = _rank_review(root, tmp_path / "rows.json", frozen.plan, capsys)
    frozen_source = after_review.subset_sources[source_key]
    assert (frozen_source.origin, frozen_source.rows) == (after_origin, tuple(rows))
    tampered = json.loads((tmp_path / "rows.json").read_bytes())
    (subset,) = tampered["subset_sources"]
    first, second, *_ = subset["source"]["rows"]
    first["row"] = second["row"]
    with pytest.raises(ValueError, match="Invalid subset source for"):
        load_plan_requests(json.dumps(tampered).encode())
    changes = diff_plan_requests(before_review, after_review)
    assert changes == diff_plan_requests(saved.plan, frozen.plan)
    named_plan = service.read(project, revision=snapshot.revision, source_runs=named_keys).settled.plan
    assert set(named_plan.sources) == set(named_keys)
    assert main(["plan", str(root), "--check", str(baseline), "--json"]) == 1
    reviewed = json.loads(capsys.readouterr().out)["review"]
    assert len([item for item in reviewed if item["state"] == "changed"]) == len(fixed)
    for key, run in fixed.items():
        assert run.key == ranks[key].key and run.request_id != ranks[key].request_id
        assert run.inputs["subject"].producer == expected[key]
        assert run.coordinates["seed"] == ranks[key].coordinates["seed"]
        binding, = decode_request(
            after_review.requests[run.request_id], after_review.documents.__getitem__,
        ).inputs["subject"]
        assert isinstance(binding.reference, ProductRef) and binding.key == key
        assert binding.reference.request_id == next(train.request_id for train in trains if train.key == expected[key])
        assert changes[run.key] == ElementChange(
            "changed", "inputs.subject", before_source=before_origin, after_source=after_origin,
        )
        entry = next(item for item in reviewed if item["request_id"] == run.request_id)
        assert entry["path"] == "inputs.subject"
        assert entry["before_source"]["path"] == before_origin.path
        assert entry["after_source"]["path"] == after_origin.path
        sources = named_plan.sources[run.key]
        assert sources["/inputs/subject/where"] == after_origin
        winner = expected[key]
        assert sources["/inputs/subject"] == ValueSource(
            "upstream", winner.step, step=winner.step, variant=winner.variant, row=winner.row,
        )
    key = groups[-1].ranks[0]
    evaluated, = jobs.admit_plan(folders, frozen, run_keys=(fixed[key].key,), token="frozen").items
    assert evaluated.disposition == "admitted"
    wait_until(lambda: job_row(host, evaluated.job_id, "succeeded"), timeout=22)
    fresh = SuccessView(view.store).read_attempt(evaluated.job_id)
    assert fresh.receipt.effective == view.read_attempt(ids[ranks[key].key]).receipt.effective
