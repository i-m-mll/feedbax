"""Saved addressed differences flow through Plan and managed jobs."""

from copy import deepcopy

import pytest

from feedbax.compiler.model_attachment import resolve_effective_model
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_attachment import MODEL_DIFFERENCE_BATCHES
from feedbax.contracts.project import Project
from feedbax.planner.project_plan import ProjectPlanMemo, RunKey, compile_project
from feedbax.contracts.project_sweep import Sweep, SweepExpansionError, axis_to_list, expand_sweep, model_field_path
from feedbax.contracts.request_identity import decode_request
from feedbax.jobs import JobService
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored, document_project
from tests.support.host import job_row, token, wait_until
from tests.support.model_documents import read_changed_model
from tests.test_model_document_plan import _add_attachment, _attachment_address, _attachment_owner, _model_controller


__all__ = ("authored",)


def _field(parent, *path):
    return ModelAddress.model_validate(parent | {
        "element": {"kind": "field", "parent": parent["element"], "path": path},
    })


def _wire_field(attachment, *path):
    return _field(_attachment_address(_attachment_owner("read"), attachment), *path)


def _controller(model):
    return model.interior.components["wrapper"].interior.components["controller"].interior


def _decode(plan, key):
    run = next(item for item in plan.runs if item.key == key)
    return decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec


_TRAINED = "trained-model"


def _managed_project(authored):
    raw = document_project(authored, evaluation=True)
    raw["steps"] = {name: raw["steps"][name] for name in ("train", "baseline")}
    read_changed_model(
        raw, "train", [_add_attachment(_attachment_owner("read"), "train-added", None, 0.8)], name=_TRAINED,
    )
    return raw


def test_saved_managed_train_clock_range_preserves_equal_request_identity(tmp_path, authored):
    raw = document_project(authored)
    train = raw["steps"]["train"]
    raw["documents"]["reach-model"]["payload"]["step_seconds"] = "0.02 s"
    baseline = compile_project(Project.model_validate(raw))
    assert not baseline.diagnostics, baseline.diagnostics
    (original,) = baseline.runs
    assert original.admission_problem is None
    assert _decode(baseline, original.key).trial_documents.model.step_seconds == 0.02

    clock = _field({"composites": [], "element": {"kind": "graph"}}, "step_seconds")
    train["sweep"] = {"axes": {"clock": {
        "path": model_field_path(clock), "range": {"start": 0.01, "stop": 0.03, "count": 3},
    }}}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "clock-range", Project.model_validate(raw))
    plan = folders.read_plan("clock-range", revision=None).plan
    assert not plan.diagnostics, plan.diagnostics
    assert all(run.admission_problem is None for run in plan.runs)
    assert [_decode(plan, run.key).trial_documents.model.step_seconds for run in plan.runs] == [0.01, 0.02, 0.03]
    assert len(plan.requests) == 3
    assert plan.runs[1].request_id == original.request_id


def test_saved_train_and_evaluate_use_the_same_attachment_id(host, tmp_path, authored):
    registry = new_application_registry_bundle()
    folders = ProjectFolderService(
        tmp_path, components=registry.components, functions=registry.expression_functions,
    )
    jobs = JobService(host, registry)
    raw = _managed_project(authored)
    target = model_field_path(_wire_field("train-added", "operands", "factor", "value"))
    raw["steps"]["train"]["sweep"] = {"axes": {"factor": {"path": target, "values": [0.5]}}}
    raw["steps"]["baseline"]["sweep"] = {
        "axes": {"trained": {"input": "subject"}},
        "exceptions": {"condition": {"where": {"rule": ""}, "patches": {target: {"value": 0.25}}}},
    }
    _save_project(folders, "addressed", Project.model_validate(raw))
    saved = folders.read("addressed").project
    assert saved.steps["train"].sweep.axes["factor"].path == target
    assert not _model_controller(saved.model_dump(mode="json", by_alias=True))["wires"]["read"].get("attachments")
    snapshot = folders.read_plan("addressed", revision=None)
    plan = snapshot.plan
    assert not plan.diagnostics
    train = next(run for run in plan.runs if run.kind == "train")
    evaluate = next(run for run in plan.runs if run.kind == "evaluate")
    assert evaluate.inputs["subject"].producer == train.key
    assert evaluate.coordinates["factor"] == train.coordinates["factor"]
    document = _decode(plan, train.key).trial_documents.model
    (attachment,) = _controller(document).wires["read"].attachments
    assert attachment.id == "train-added"
    assert attachment.operands["factor"].value == 0.5
    variation = _decode(plan, evaluate.key).params["variation"]
    (patch,) = variation
    assert patch["value"]["params"]["scale"] == 0.25

    train_job, evaluate_job = [
        jobs.admit_plan(folders, snapshot, run_keys=(key,), token=token(key.step)).items[0]
        for key in (train.key, evaluate.key)
    ]
    for job in (train_job, evaluate_job):
        assert job.disposition == "admitted", job
    trained, evaluated = [
        wait_until(
            lambda: job_row(host, job.job_id, "succeeded") or job_row(host, job.job_id, "failed"),
        )
        for job in (train_job, evaluate_job)
    ]
    assert trained["state"] == "succeeded", trained["outcome"]
    assert evaluated["state"] == "succeeded", evaluated["outcome"]
    assert evaluated["edges"][0]["binding"]["terminal"]["receipt"] == trained["outcome"]["store_success"]


def test_addressed_rules_list_conversion_and_scoped_variant(authored):
    raw = _managed_project(authored)
    target = model_field_path(_wire_field("train-added", "operands", "factor", "value"))
    variant_target = f"documents.{_TRAINED}.{target}"
    raw["documents"]["alternate"] = deepcopy(raw["documents"]["reach-model"])
    other = deepcopy(raw["steps"]["train"])
    other["base"]["inline"]["trial_documents"]["model"] = {"document": "alternate"}
    raw["steps"]["other"] = other
    mass = model_field_path(_field({"composites": [], "element": {"kind": "component", "id": "plant"}}, "fields", "mass"))
    raw["sweep"] = {"axes": {
        "condition": {"path": variant_target, "values": [0.6, 0.9]},
        "step-mass": {"path": f"documents.{_TRAINED}.{mass}", "values": [2]},
        "base-mass": {"path": f"documents.alternate.{mass}", "values": [4]},
    }}
    raw["steps"]["baseline"]["sweep"] = {
        "axes": {"trained": {"input": "subject"}, "factor": {
            "path": target,
            "sampler": {"distribution": "saved", "count": 2, "seed": 4},
            "draws": {"first": 0.4, "second": 0.5},
        }},
    }
    memo = ProjectPlanMemo()

    def requests():
        plan = compile_project(Project.model_validate(raw), memo=memo)
        assert not plan.diagnostics
        trains = [run for run in plan.runs if run.key.step == "train"]
        assert {
            _controller(_decode(plan, run.key).trial_documents.model).wires["read"].attachments[0].operands["factor"].value
            for run in trains
        } == {0.6, 0.9}
        assert {
            (run.key.step, _decode(plan, run.key).trial_documents.model.interior.components["plant"].fields["mass"])
            for run in plan.runs if run.kind == "train"
        } == {("train", 2), ("other", 4)}
        return [(run.key, run.coordinates, run.request_id) for run in plan.runs]

    before = requests()
    authored_axis = Project.model_validate(raw).steps["baseline"].sweep.axes["factor"]
    raw["steps"]["baseline"]["sweep"]["axes"]["factor"] = axis_to_list(authored_axis).model_dump(
        mode="json", exclude_unset=True,
    )
    assert requests() == before

    variant_axis = Project.model_validate(raw).sweep.axes["condition"]
    raw["sweep"]["axes"]["condition"] = axis_to_list(variant_axis).model_dump(
        mode="json", exclude_unset=True,
    )
    assert requests() == before

    missing = deepcopy(raw)
    missing["documents"][_TRAINED]["differences"] = {}
    plan = compile_project(Project.model_validate(missing), memo=memo)
    assert len(plan.diagnostics) == 2
    assert all(
        (problem.document, model_field_path(problem.model_address)) == (_TRAINED, target) for problem in plan.diagnostics
    )
    assert not plan.runs


def test_vector_operand_member_edit_does_not_reset_attachment_anchors(authored):
    _, model, _, _ = authored
    owner = _attachment_owner("read")
    first = _add_attachment(owner, "first", None)
    first["attachment"]["operands"]["factor"]["value"] = "vector(0.1, 0.8)"
    address = _wire_field("first", "operands", "factor", "value", 0)
    records = [
        first, _add_attachment(owner, "second", "first"),
        {"op": "set_field", "address": address.model_dump(mode="json"), "value": 0.2},
        _add_attachment(owner, "third", "first"),
    ]
    differences = tuple(MODEL_DIFFERENCE_BATCHES.validate_python({"batch": records})["batch"])
    effective = resolve_effective_model(model, ComponentRegistry(), differences)
    chain = _controller(effective).wires["read"].attachments
    assert [attachment.id for attachment in chain] == ["first", "second", "third"]
    assert chain[0].operands["factor"].value == (0.2, 0.8)

    whole = _wire_field("first", "operands", "factor", "value")
    sweep = Sweep.model_validate({
        "axes": {"draw": {"path": model_field_path(whole), "values": [first["attachment"]["operands"]["factor"]["value"]]}},
        "exceptions": {"part": {"where": {"rule": ""}, "patches": {model_field_path(address): {"value": 0.2}}}},
    })
    with pytest.raises(SweepExpansionError, match="already writes"):
        expand_sweep(sweep)


def test_document_variant_only_changes_the_addressed_base_model(authored):
    raw = document_project(authored, evaluation=False)
    address = model_field_path(_wire_field("base", "operands", "factor", "value"))
    _model_controller(raw)["wires"]["read"]["attachments"] = {
        "base": _add_attachment(_attachment_owner("read"), "base", None, 0.9)["attachment"],
    }
    raw["sweep"] = {"rows": {
        "original": {},
        "variant": {"patches": {f"documents.reach-model.{address}": {"value": 0.6}}},
    }}
    keys = tuple(RunKey(variant, "train", "default") for variant in ("original", "variant"))
    plan = compile_project(Project.model_validate(raw), source_runs=keys)
    assert not plan.diagnostics
    assert [
        _controller(_decode(plan, key).trial_documents.model).wires["read"].attachments[0].operands["factor"].value
        for key in keys
    ] == [0.9, 0.6]
    provenance = "/payload/trial_documents/model/interior/components/wrapper/interior/components/controller"
    origin = plan.selected_definitions[keys[1]].sources[provenance + "/interior/wires/read/attachments/base/operands/factor/value"]
    assert origin.kind == "variant"
    assert plan.selected_definitions[keys[1]].sources[provenance + "/label"].kind == "document"


def test_integer_range_uses_the_site_and_preserves_whole_request_ids(tmp_path, authored):
    raw = document_project(authored)
    raw["steps"]["train"]["sweep"] = {
        "axes": {"batches": {
            "path": "training_config.n_batches", "range": {"start": 2, "stop": 4, "count": 3},
        }},
    }
    project = Project.model_validate(raw)
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "integers", project)
    saved = folders.read("integers").project
    rule = saved.steps["train"].sweep.axes["batches"].range
    assert rule.model_dump(exclude_unset=True) == {"start": 2, "stop": 4, "count": 3}
    plan = folders.read_plan("integers", revision=None).plan
    assert not plan.diagnostics
    assert [run.coordinates["batches"].value for run in plan.runs] == [2, 3, 4]
    assert all(type(run.coordinates["batches"].value) is int for run in plan.runs)
    assert [_decode(plan, run.key).training_config.n_batches for run in plan.runs] == [2, 3, 4]

    def request_ids(candidate):
        return [(run.key, run.request_id) for run in compile_project(candidate).runs]

    axis = saved.steps["train"].sweep.axes["batches"]
    saved.steps["train"].sweep.axes["batches"] = axis_to_list(axis)
    assert request_ids(saved) == [(run.key, run.request_id) for run in plan.runs]
    saved.steps["train"].sweep.axes["batches"] = axis_to_list(axis, integer_site=lambda _: True)
    assert request_ids(saved) == [(run.key, run.request_id) for run in plan.runs]

    axis.range = axis.range.model_copy(update={"start": 16, "stop": 256, "count": 5, "scale": "log"})
    assert [item.value for item in axis_to_list(axis, integer_site=lambda _: True).values.values()] == [
        16, 32, 64, 128, 256,
    ]
    address = _field({
        "composites": [], "element": {"kind": "component", "id": "wrapper"},
    }, "substeps")
    raw["steps"]["train"]["sweep"]["axes"]["batches"] = {
        "path": model_field_path(address), "range": {"start": 1, "stop": 2, "count": 4},
    }
    invalid = compile_project(Project.model_validate(raw))
    (problem,) = invalid.diagnostics
    assert problem.model_address == address
    assert all(member in problem.detail for member in ("4/3", "5/3"))


def test_exact_integer_range_bounds_above_the_float_precision_limit():
    sweep = Sweep.model_validate({
        "axes": {"size": {"path": "size", "range": {
            "start": 9007199254740993, "stop": 9007199254740995, "count": 3,
        }}},
    })
    assert [row.coordinates["size"].value for row in expand_sweep(sweep, integer_site=lambda _: True)] == [
        9007199254740993, 9007199254740994, 9007199254740995,
    ]
