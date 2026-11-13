"""Saved update rules retain request identity and account for missing checkpoints."""

from copy import deepcopy

from feedbax.contracts.effective_success import EvaluationSubjectValue, success_content_ref
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.persistence.store_sync import sync_store
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef
from feedbax.contracts.manifest import canonical_manifest_path
from feedbax.contracts.project import Project
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.request_refs import UpdateSeriesMember
from feedbax.jobs import JobService
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored_documents, document_project
from tests.test_local_job_host import host as host_fixture, job_row, restart, wait_until
from tests.test_private_checkpoint_evaluation import install_finite_gate

host = host_fixture


def definition(rule):
    raw = document_project(authored_documents(), evaluation=True)
    raw["steps"]["train"]["base"]["inline"]["training_config"].update(n_batches=4, snapshot_interval=2)
    raw["steps"]["train"]["base"]["inline"]["method_payload"]["payload"]["target_total"] = 4
    raw["steps"]["baseline"]["inputs"]["subject"]["checkpoint"] = rule
    return raw


def test_update_series_expands_per_train_definition_without_store(tmp_path):
    folders = ProjectFolderService(tmp_path / "projects")
    raw = definition({"every": 2})
    raw["steps"]["explicit"] = deepcopy(raw["steps"]["baseline"])
    raw["steps"]["explicit"]["inputs"]["subject"]["checkpoint"] = [{"at": 4}, {"at": 1}]
    raw["steps"]["bounded"] = deepcopy(raw["steps"]["baseline"])
    raw["steps"]["bounded"]["inputs"]["subject"]["checkpoint"] = {"every": 2, "from": 3}
    raw["steps"]["empty"] = deepcopy(raw["steps"]["baseline"])
    raw["steps"]["empty"]["inputs"]["subject"]["checkpoint"] = {"every": 5}
    raw["steps"]["duplicate"] = deepcopy(raw["steps"]["baseline"])
    raw["steps"]["duplicate"]["inputs"]["subject"]["checkpoint"] = [{"at": 2}, {"at": 2}]
    _save_project(folders, "series", Project.model_validate(raw))
    plan = folders.read_plan("series", revision=None)
    assert not plan.store_path.exists()
    stored = (folders.root / "series/steps/bounded.json").read_text()
    assert '"from": 3' in stored and '"from_"' not in stored
    rows = {run.key.step: run for run in plan.plan.runs}
    for step, updates in (("baseline", (2, 4)), ("explicit", (4, 1)), ("bounded", (4,))):
        run = rows[step]
        payload = plan.plan.requests[run.request_id]
        decoded = decode_request(payload, plan.plan.documents.__getitem__)
        selector = decoded.inputs["subject"][0].reference.select
        assert isinstance(selector, UpdateSeriesMember)
        assert selector.updates == updates
        assert selector.rule == Project.model_validate(raw).steps[step].inputs["subject"].checkpoint
        assert decoded.inputs["subject"][0].reference.request_id == rows["train"].request_id
        assert encode_request(decoded.spec, decoded.inputs).data == payload
    assert {problem.step for problem in plan.plan.diagnostics} == {"empty", "duplicate"}
    assert all(problem.path == "inputs.subject.checkpoint" for problem in plan.plan.diagnostics)


def test_saved_series_settles_late_partial_and_all_gap_results(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    registry = new_application_registry_bundle()
    folders = ProjectFolderService(tmp_path / "projects")
    project = "series"
    raw = definition({"every": 2})
    _save_project(folders, project, Project.model_validate(raw))
    first = folders.read_plan(project, revision=None)
    train = next(run for run in first.plan.runs if run.kind == "train")
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
    (producer,) = jobs.admit_plan(folders, first, run_keys=(train.key,), token="train").items
    assert producer.disposition == "admitted"
    finished = wait_until(lambda: job_row(host, producer.job_id, "succeeded"), timeout=20)
    assert "checkpoint_catalog" not in host.jobs(job_id=producer.job_id, include_launch=True)[0]["launch"]["payload"]

    raw["steps"]["baseline"]["inputs"]["subject"]["checkpoint"] = [{"at": 4}, {"at": 1}, {"at": 2}, {"at": 6}]
    raw["steps"]["empty"] = deepcopy(raw["steps"]["baseline"])
    raw["steps"]["empty"]["inputs"]["subject"]["checkpoint"] = [{"at": 1}, {"at": 5}]
    _save_project(folders, project, Project.model_validate(raw))
    plan = folders.read_plan(project, revision=None)
    assert not plan.plan.diagnostics
    rows = {run.key.step: run for run in plan.plan.runs}
    assert rows["train"].request_id == train.request_id
    assert next(run for run in first.plan.runs if run.kind == "evaluate").request_id != rows["baseline"].request_id

    gated = JobService(host, registry)
    (disabled,) = gated.admit_plan(folders, plan, run_keys=(rows["baseline"].key,), token="disabled").items
    assert disabled.disposition == "rejected" and "activation is held" in disabled.reason
    run_keys = (rows["baseline"].key, rows["empty"].key)
    submission = jobs.admit_plan(folders, plan, run_keys=run_keys, token="evaluation")
    for item, expected, observed in zip(submission.items, ([4, 1, 2, 6], [1, 5]), ([4, 2], []), strict=True):
        assert item.disposition == "admitted"
        row = wait_until(lambda: (
            row if (row := job_row(host, item.job_id))["outcome"] is not None else None
        ), timeout=20)
        assert row["state"] == "succeeded", row["outcome"]
        selection = row["edges"][0]["binding"]["checkpoint_selection"]
        assert selection["requested"] == expected
        gaps = [{"update": update, "reason": "checkpoint_unavailable"} for update in expected if update not in observed]
        assert selection["gaps"] == gaps
        assert selection["closure"] == {"kind": "completion", "product": finished["outcome"]["completion"]}
        result = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=item.job_id)
        manifest = result._manifest()
        assert manifest.provenance.metadata["checkpoint_selection"]["gaps"] == gaps
        assert manifest.metadata["placement"] == "ordinary"
        store = ImmutableArtifactBlobProvider(plan.store_path)
        success = SuccessView(store).read_attempt(item.job_id)
        assert success.ref.model_dump(mode="json") == row["outcome"]["store_success"]
        bound, = success.receipt.inputs["subject"]
        value = EvaluationSubjectValue.model_validate(bound.descriptor)
        assert value.requested == tuple(expected)
        assert [gap.model_dump(mode="json") for gap in value.gaps] == gaps
        tree = store.get_tree(success_content_ref(bound.content))
        if observed:
            assert [member.coordinate for member in value.resolved.members] == observed
            assert len([file for file in tree.files if file.relative_path.endswith("/model.safetensors")]) == len(observed)
        else:
            assert value.resolved is None and bound.evaluation_subject.checkpoints == ()
            assert [file.relative_path for file in tree.files] == ["binding.json"]
        assert bound.evaluation_subject.closure.receipt.sha256 == finished["outcome"]["receipt"]["manifest_sha256"]
        shallow = StoreClosureRef(roots=(ReceiptRoot(receipt=success.ref),))
        destination = ImmutableArtifactBlobProvider(tmp_path / f"shallow-{item.job_id}")
        sync_store(store, destination, shallow, destination_ref="proof/evaluate")
        assert bound.evaluation_subject.closure.checkpoint.transaction in ReceiptClosure(destination).walk(shallow)
        assert [parent.id for parent in manifest.input_training_runs] == [finished["outcome"]["receipt"]["manifest_id"]]
        with result.open_states() as dataset:
            assert dataset.data("update").tolist() == observed
            if not observed:
                assert dataset.descriptor.variables == {}
    restart(host)
    assert jobs.admit_plan(folders, plan, run_keys=run_keys, token="evaluation") == submission
    for item in submission.items:
        jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=item.job_id)

    raw["steps"]["empty"]["base"]["inline"]["params"]["seed"] += 1
    _save_project(folders, project, Project.model_validate(raw))
    new_plan = folders.read_plan(project, revision=None)
    empty = next(run for run in new_plan.plan.runs if run.key.step == "empty")
    closure_manifest = canonical_manifest_path(
        "TrainingRunManifest", finished["outcome"]["receipt"]["manifest_id"], root=plan.store_path,
    )
    original = closure_manifest.read_bytes()
    try:
        closure_manifest.write_bytes(original + b" ")
        (corrupt,) = jobs.admit_plan(folders, new_plan, run_keys=(empty.key,), token="corrupt-closure").items
        assert corrupt.disposition == "admitted"
        rejected = wait_until(lambda: job_row(host, corrupt.job_id, "blocked"))
        assert rejected["outcome"]["error"]["code"] == "dependency_corrupt"
        assert rejected["delivery"] == 0
    finally:
        closure_manifest.write_bytes(original)


def test_completed_resume_settles_gaps_while_its_parent_remains_resumable(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    jobs = JobService(host, new_application_registry_bundle(), enable_evaluation_candidate=True)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "resumed-series"
    _save_project(folders, project, Project.model_validate(definition([{"at": 2}, {"at": 5}])))
    plan = folders.read_plan(project, revision=None)
    rows = {run.kind: run for run in plan.plan.runs}
    host.policy(compute=1)
    install_finite_gate(jobs, tmp_path, stop_at=3, resume_at=4, managed=True)
    (producer,) = jobs.admit_plan(folders, plan, run_keys=(rows["train"].key,), token="train").items
    assert producer.disposition == "admitted"
    wait_until(lambda: (tmp_path / "before-stop").exists(), timeout=16)
    jobs.stop([producer.job_id], "checkpoint")
    wait_until(lambda: job_row(host, producer.job_id, "stopped"))
    (evaluate,) = jobs.admit_plan(folders, plan, run_keys=(rows["evaluate"].key,), token="evaluate").items
    assert evaluate.disposition == "admitted"
    assert job_row(host, evaluate.job_id, "queued") is not None
    resumed = jobs.resume(producer.job_id, "resume")
    (tmp_path / "finish-train").touch()
    wait_until(lambda: job_row(host, resumed.job_id, "succeeded"), timeout=20)
    result = wait_until(lambda: job_row(host, evaluate.job_id, "succeeded"), timeout=20)
    selection = result["edges"][0]["binding"]["checkpoint_selection"]
    assert selection["lineage"] == [producer.job_id, resumed.job_id]
    assert selection["closure"]["product"]["producer_job_id"] == resumed.job_id
    assert selection["gaps"] == [{"update": 5, "reason": "checkpoint_unavailable"}]
    assert [member["producer_job_id"] for member in selection["members"]] == [producer.job_id]
