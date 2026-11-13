"""Exercise saved Model attachment differences through document Train and Evaluate."""

from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.model_edit import EvaluateAttachmentRequest
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.rollout import RolloutParameters
from feedbax.contracts.studio_project import StudioProjectPatchBatch
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.jobs import JobService
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.training.checkpoint_custody import open_published_model_slot
from feedbax.training.evaluation_variant import resolve_evaluation_variant
from feedbax.web.services.evaluate_attachment import (
    EvaluateAttachmentRefused,
    prepare_evaluate_attachment,
    read_evaluate_attachments,
)
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.managed_import_probe import check_managed_worker, probe_managed_workers
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored, document_project  # noqa: F401
from tests.test_local_job_host import host, job_row, wait_until  # noqa: F401
from tests.test_model_document_plan import (
    _add_attachment,
    _attachment_address,
    _attachment_owner,
    _model_controller,
    _scale_attachment,
)


def test_saved_addressed_condition_restores_document_train_model(host, tmp_path, authored):  # noqa: F811
    registry = new_application_registry_bundle()
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
    imports = tmp_path / "worker-imports"
    probe_managed_workers(jobs, imports)
    folders = ProjectFolderService(
        tmp_path / "projects",
        components=registry.components,
        functions=registry.expression_functions,
    )
    project = "addressed-condition"
    raw = document_project(authored, evaluation=True)
    raw["steps"]["condition"] = deepcopy(raw["steps"]["baseline"])
    _model_controller(raw)["wires"]["read"]["attachments"] = [
        _scale_attachment("retain", 0.8),
        _scale_attachment("remove", 0.5),
    ]
    _save_project(folders, project, Project.model_validate(raw))
    original = folders.read_plan(project, revision=None)
    previous = {run.key: run.request_id for run in original.plan.runs}
    train = RunKey("default", "train", "default")
    baseline = RunKey("default", "baseline", "default")
    condition = RunKey("default", "condition", "default")
    assert previous[baseline] == previous[condition]

    owner = _attachment_owner("read")
    selection = {
        "run": {"step": "condition", "row": "default", "variant": "default"},
        "scope": {"kind": "step"},
    }
    invalid = EvaluateAttachmentRequest.model_validate(
        selection
        | {
            "revision": original.revision,
            "difference": _add_attachment(owner, "invalid", "missing"),
        }
    )
    with pytest.raises(
        EvaluateAttachmentRefused, match="anchor 'missing' no longer exists"
    ) as refused:
        prepare_evaluate_attachment(folders, project, invalid, registry.components)
    assert refused.value.problem.address == invalid.difference.address

    differences = (
        _add_attachment(owner, "add", "retain", 0.4),
        {
            "op": "change",
            "address": _attachment_address(owner, "retain"),
            "attachment": _scale_attachment("retain", 0.6),
        },
        {"op": "remove", "address": _attachment_address(owner, "remove")},
    )
    plan = original
    for difference in differences:
        request = EvaluateAttachmentRequest.model_validate(
            selection
            | {
                "revision": plan.revision,
                "difference": difference,
            }
        )
        edit = prepare_evaluate_attachment(folders, project, request, registry.components)
        assert edit.changed == request.difference.address
        folders.patch(project, StudioProjectPatchBatch(revision=plan.revision, files=[edit.file]))
        plan = folders.read_plan(project, revision=None)

    saved = read_evaluate_attachments(
        folders, project, request.model_copy(update={"revision": plan.revision})
    )
    assert saved.records == saved.effective
    assert [item.op for item in saved.records] == ["add", "change", "remove"]
    assert folders.read(project).project.documents == Project.model_validate(raw).documents
    current = {run.key: run.request_id for run in plan.plan.runs}
    assert current.keys() == previous.keys()
    assert {key for key in current if current[key] != previous[key]} == {condition}

    jobs_by_run = {}
    for key in (train, baseline, condition):
        (admitted,) = jobs.admit_plan(folders, plan, run_keys=(key,), token=key.step).items
        assert admitted.disposition == "admitted", admitted
        jobs_by_run[key] = admitted.job_id

    completions = {}
    for run, job_id in jobs_by_run.items():
        terminal = wait_until(
            lambda: (job_row(host, job_id, "succeeded") or job_row(host, job_id, "failed")),
            timeout=26,
        )
        assert terminal["state"] == "succeeded", terminal.get("outcome")
        completions[run] = terminal
        check_managed_worker(
            imports,
            job_id,
            required=(
                "feedbax.training.managed_studio" if run == train
                else "feedbax.training.evaluate_worker"
            ),
        )
    checkpoint = completions[train]["outcome"]["completion"]
    for run in (baseline, condition):
        (edge,) = completions[run]["edges"]
        assert edge["binding"]["completion"] == checkpoint
    published = PublishedTrainingCheckpoint.model_validate(checkpoint["checkpoint"])
    slot = open_published_model_slot(published.checkpoint_root, published.pointer)
    assert slot is not None
    source_hash = sha256(canonical_json_v2_bytes(slot.graph.model_dump(mode="json"))).hexdigest()
    train_request = decode_request(
        plan.plan.requests[current[train]],
        plan.plan.documents.__getitem__,
    ).spec
    condition_request = decode_request(
        plan.plan.requests[current[condition]],
        plan.plan.documents.__getitem__,
    ).spec
    params = RolloutParameters.model_validate(condition_request.params)
    variation_hash = canonical_sha256(params.model_dump(mode="json")["variation"])
    documents = train_request.trial_documents
    assert documents is not None
    variant = resolve_evaluation_variant(slot.graph, documents.task, params.variation)
    task_hash = sha256(canonical_json_v2_bytes(variant.task.model_dump(mode="json"))).hexdigest()
    paired = compile_trial_execution(
        documents.model_name,
        documents.model,
        variant.task,
        registry.components,
        registry.expression_functions,
        input_overrides=documents.input_overrides,
        graph_override=variant.graph,
    )
    assert isinstance(paired, PreparedTrialExecution), paired
    effective_hash = canonical_sha256(
        {
            "resolved_graph": paired.executable.resolved.resolved_sha256,
            "checkpoint_values": slot.arrays.sha256,
            "source_overrides": {},
        }
    )

    def open_states(run):
        return jobs.read_evaluation_result(
            project=project,
            store=plan.store_path,
            job_id=jobs_by_run[run],
        ).open_states()

    with open_states(baseline) as control, open_states(condition) as treated:
        for states in (control, treated):
            assert states.data("checkpoint").item() == 2
            assert states.data("objective/coverage/effort").item() == 28
            assert states.data("state_time").shape == (81,)
            assert states.descriptor.attrs["source_graph_sha256"] == source_hash
        assert "effective_graph_sha256" not in control.descriptor.attrs
        attrs = treated.descriptor.attrs
        assert attrs["effective_graph_sha256"] == effective_hash
        assert attrs["effective_task_sha256"] == task_hash
        assert attrs["variation_sha256"] == treated.data("condition").item() == variation_hash
        assert control.data("condition").item() != treated.data("condition").item()
        control_loss = control.data("objective/total").item()
        assert np.isfinite(control_loss) and control_loss > 0
        assert treated.data("objective/total").item() == pytest.approx(
            control_loss * ((0.6 * 0.4) / (0.8 * 0.5)) ** 2,
            rel=1e-5,
        )
        carry = "/v1/model/carry/wrapper/controller/carry/carry"
        np.testing.assert_array_equal(control.data(carry), treated.data(carry))
