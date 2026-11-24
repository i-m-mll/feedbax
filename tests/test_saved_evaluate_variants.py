"""Qualify saved task and connection variants on one real Train checkpoint."""

from copy import deepcopy

import numpy as np
import pytest

from feedbax.compiler.model_document import lower_model_document
from feedbax.contracts.expression_values import CallValue
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.jobs import StudioProjectWindowMapProblem
from feedbax.contracts.project import Project
from feedbax.planner.project_plan import RunKey, compile_project
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.rollout import RolloutParameters
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.contracts.task_document import PythonTaskSelection
from feedbax.host import HostClient
from feedbax.jobs import JobService, PlanJobReader
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.persistence.canonical_evaluate import RAW_STATES_SCHEMA, decode_rollout_request
from feedbax.contracts.rollout import CannotRun
from feedbax.examples import load_shipped_projects
from tests.support.plugins.reaching_task import REACHING_TASK
from feedbax.training.evaluation_variant import resolve_evaluation_variant
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project, managed_evaluation_definition
from tests.support.model_documents import read_changed_model
from tests.support.host import job_row, token, wait_until
from tests.test_model_document_plan import _add_attachment, _attachment_owner



def _patch(path, value, *, op="replace"):
    return {"path": path, "value": value, "op": op}


def _saved_variants():
    definition = managed_evaluation_definition()
    steps = definition["steps"]
    baseline = steps["baseline"]
    task = deepcopy(definition["documents"]["task"])
    signal = next(signal for signal in task["payload"]["signals"] if signal["id"] == "target_pos")
    signal["cells"][0]["value"] = {
        "kind": "call",
        "function": "vector",
        "args": [{"kind": "literal", "value": coordinate, "unit": "m"} for coordinate in (-0.05, 0.10)],
    }
    definition["documents"]["replacement-task"] = task

    def step(variation):
        result = deepcopy(baseline)
        result["base"]["inline"]["params"]["variation"] = variation
        return result

    selection = _patch("task", {"document": "replacement-task"})
    replacement = step([selection])
    perturbed = step([selection])
    incompatible = deepcopy(task["payload"])
    incompatible["signals"] = [signal for signal in incompatible["signals"] if signal["id"] != "target_pos"]
    dtype = step([])
    dtype_change = [{
        "op": "set_field",
        "address": {
            "composites": ["wrapper", "controller"],
            "element": {"kind": "field", "parent": {"kind": "component", "id": "readout"}, "path": ["fields", "dtype"]},
        },
        "value": "float16",
    }]
    definition["steps"] = {
        "train": steps["train"], "baseline": baseline, "replacement": replacement, "perturbed": perturbed,
        "different-dtype": dtype,
        "incompatible": step([_patch("task", incompatible)]),
    }
    trained = steps["train"]["base"]["inline"]["trial_documents"]["model"]["document"]
    components = new_application_registry_bundle().components
    read_changed_model(
        definition, "perturbed", [_add_attachment(_attachment_owner("read"), "scale", None, 0.5)],
        base=trained, components=components,
    )
    read_changed_model(definition, "different-dtype", dtype_change, base=trained, components=components)
    return Project.model_validate(definition)


def test_two_crossed_wire_insertions_compose_through_saved_plan(tmp_path):
    definition = managed_evaluation_definition()
    registries = new_application_registry_bundle()
    from feedbax.contracts.model_document import ModelDocument
    from feedbax.contracts.task_document import TaskDocument

    model = ModelDocument.model_validate(definition["documents"]["reach-model"]["payload"])
    task = TaskDocument.model_validate(definition["documents"]["task"]["payload"])
    graph = lower_model_document(model, registries.components).graph
    first = next(wire for wire in graph.wires if wire.target_node == "plant")
    second = next(wire for wire in graph.wires if wire.target_node == "wrapper")
    baseline = deepcopy(definition["steps"]["baseline"])
    baseline["base"]["inline"]["trial_documents"] = {"model": {"document": "reach-model"}}
    axes = {}
    for name, wire in (("first", first), ("second", second)):
        identity = wire.authored_id()
        authored = definition["documents"]["reach-model"]["payload"]["interior"]["wires"][identity]
        attachment = _add_attachment(_attachment_owner(identity), name, None)["attachment"]
        axes[name] = {"values": {"inserted": {"patches": {
            f"documents.reach-model.payload.interior.wires.{identity}.attachments": {
                "op": "replace" if "attachments" in authored else "add",
                "value": [attachment, *authored.get("attachments", [])],
            },
        }}}}
    baseline["sweep"] = {"axes": axes}
    project = Project.model_validate(definition | {"steps": {
        "train": definition["steps"]["train"], "crossed": baseline,
    }})
    folders = ProjectFolderService(tmp_path / "projects")
    _save_project(folders, "crossed", project)
    plan = folders.read_plan("crossed", revision=None).plan
    assert not plan.diagnostics
    (run,) = (run for run in plan.runs if run.key.step == "crossed")
    request = decode_request(plan.requests[run.request_id], plan.documents.__getitem__)
    variation = RolloutParameters.model_validate(request.spec.params).variation
    assert isinstance(variation, tuple) and all(edit.path.startswith("graph.") for edit in variation)
    variant = resolve_evaluation_variant(graph, task, variation, components=registries.components)
    unchanged = [wire for wire in graph.wires if wire not in (first, second)]
    assert all(wire in variant.graph.wires for wire in unchanged)
    inserted = variant.graph.nodes.keys() - graph.nodes.keys()
    assert len(inserted) == 2
    for original in (first, second):
        (before,) = (
            wire for wire in variant.graph.wires if wire.target_node in inserted
            and (wire.source_node, wire.source_port) == (original.source_node, original.source_port)
        )
        (after,) = (wire for wire in variant.graph.wires if wire.source_node == before.target_node)
        assert (before.target_port, after.source_port) == ("input", "output")
        assert (after.target_node, after.target_port) == (original.target_node, original.target_port)
        scale = variant.graph.nodes[before.target_node]
        assert scale.type == "feedbax.Scale" and scale.params["scale"] == 0.5
    reader = PlanJobReader(HostClient())
    reader.evaluation_parameters(plan, run)


@pytest.mark.parametrize(("scope", "site", "stored_path"), [
    ("inline", "bad~edit", "/base/inline/params/variation/bad~0edit/path"),
    ("axis", "0", "/sweep/axes/mass/values/heavy/patches/params.variation/value/0/path"),
    ("exception", "bad-edit", "/sweep/exceptions/heavy/patches/params.variation.bad-edit/value/path"),
])
def test_saved_raw_graph_edit_refuses_at_its_exact_author(tmp_path, scope, site, stored_path):
    raw = managed_evaluation_definition()
    raw["steps"] = {name: raw["steps"][name] for name in ("train", "baseline")}
    step = raw["steps"]["baseline"]
    edit = _patch("graph.nodes.plant.params.mass", 2.0)
    if scope == "inline":
        step["base"]["inline"]["params"]["variation"] = {site: edit}
    elif scope == "axis":
        step["sweep"] = {"axes": {"mass": {"values": {"heavy": {
            "patches": {"params.variation": {"value": [edit]}},
        }}}}}
    else:
        step["base"]["inline"]["params"]["variation"] = {}
        step["sweep"] = {"exceptions": {"heavy": {
            "where": {"rule": ""},
            "patches": {f"params.variation.{site}": {"op": "add", "value": edit}},
        }}}
    folders = ProjectFolderService(tmp_path / "projects")
    _save_project(folders, "raw", Project.model_validate(raw))
    target = folders.root / "raw" / "steps" / "baseline.json"
    saved = target.read_bytes()
    settled = folders.read_plan("raw", revision=None)
    (problem,) = settled.plan.diagnostics
    published = StudioProjectWindowMapProblem.from_diagnostic(problem, declaration="raw.feedbax")
    assert published.code == "authored_graph_variation"
    assert published.path == f"/params/variation/{site.replace('~', '~0')}/path"
    assert (published.address, published.authored_path) == ("steps/baseline.json", stored_path)
    assert {run.kind for run in settled.plan.runs} == {"train"}
    (key,) = (row.key for row in settled.plan.rows if row.kind == "evaluate")
    selected = folders.read_plan("raw", revision=settled.revision, source_runs=(key,)).plan
    definition = selected.selected_definitions[key]
    variation = definition.payload["params"]["variation"]
    assert (variation[int(site)] if isinstance(variation, list) else variation[site]) == edit
    assert definition.sources["/payload" + problem.path].path == stored_path
    assert selected.diagnostics == settled.plan.diagnostics
    assert target.read_bytes() == saved


def test_encoded_rollout_still_decodes_named_graph_variation():
    from feedbax.contracts.request_refs import InputMember, LastMember, ProductRef
    from feedbax.contracts.request_specs import EvaluationRequestSpec
    from feedbax.contracts.rollout import ROLLOUT_TYPE

    edits = {"heavy": _patch("graph.nodes.plant.params.mass", 2.0)}
    spec = EvaluationRequestSpec(evaluation_type=ROLLOUT_TYPE, params={
        "seed": 731, "trial_policy": "paired", "trials": ["alpha"], "variation": edits,
    })
    checkpoint = ProductRef(request_id="train:" + "a" * 64, role="checkpoint", select=LastMember())
    inputs = {"subject": (InputMember(key="default", reference=checkpoint),)}
    stored = encode_request(spec, inputs)
    decoded, params, subject = decode_rollout_request(stored.id, stored.data)
    assert subject == checkpoint
    assert params.variation["heavy"].path == edits["heavy"]["path"]
    assert encode_request(decoded, inputs) == stored


def _short_reaches(steps):
    """The shipped simple reaches, ``steps`` long, with a target the step's fields set and no fixed cohort."""
    task = load_shipped_projects()["point-mass-reaching"].documents["simple-reaches-task"].model_dump(
        mode="json", by_alias=True, exclude_unset=True,
    )
    payload = task["payload"]
    payload["fixed_steps"] = steps
    payload["epochs"][0]["end"]["length"]["value"] = steps
    del payload["validation_trials"]
    (target,) = (variable for variable in payload["variables"] if variable["id"] == "target")
    target["value"] = {"kind": "call", "function": "vector", "args": [
        {"kind": "literal", "value": 0.1, "unit": "m"}, {"kind": "literal", "value": 0.0, "unit": "m"},
    ]}
    return task


def _selected_task_variants():
    raw = managed_evaluation_definition()
    train = raw["steps"]["train"]
    raw["documents"]["task"] = _short_reaches(5)
    raw["documents"]["objective"]["payload"]["terms"][0].pop("epochs")
    train["sweep"] = {"axes": {"target": {"values": {
        name: {"patches": {"trial_documents.task.variables.target.value.args.1.value": {"value": y}}}
        for name, y in (("low", -0.2), ("high", 0.3))
    }}}}
    evaluate = raw["steps"]["baseline"]
    evaluate["sweep"] = {"axes": {"trained": {"input": "subject"}}}
    params = evaluate["base"]["inline"]["params"]
    params["trial_set"] = {"kind": "sample", "count": 1}
    params["variation"] = {"target-x": _patch("task.variables.target.value.args.0.value", 0.4)}
    refused_step = deepcopy(evaluate)
    refused_step["base"]["inline"]["params"]["variation"] = {
        "bad-length": _patch("task.epochs.move.end.length.value", -1),
    }
    raw["steps"] = {"train": train, "baseline": evaluate, "refused": refused_step}
    return Project.model_validate(raw)


def test_saved_task_field_preserves_each_selected_train_task(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    folders = ProjectFolderService(tmp_path / "projects")
    project = "task-fields"
    _save_project(folders, project, _selected_task_variants())
    plan = folders.read_plan(project, revision=None)
    producers = {run.key: run for run in plan.plan.runs if run.kind == "train"}
    selected = [run for run in plan.plan.runs if run.kind == "evaluate"]
    assert len(producers) == len(selected) == 2
    assert len(plan.plan.diagnostics) == 2
    assert all(problem.step == "refused" and problem.path == "/params/variation/bad-length"
               for problem in plan.plan.diagnostics)
    jobs = JobService(host, new_application_registry_bundle())
    refused = jobs.admit_plan(
        folders, plan, run_keys=tuple(row.key for row in plan.plan.rows if row.key.step == "refused"),
        token=token("refused-task-patch"),
    ).items
    assert len(refused) == 2 and all(item.disposition == "rejected" and item.job_id is None for item in refused)
    assert not host.jobs(project=plan.project_id)

    started = jobs.admit_plan(
        folders, plan, run_keys=(*producers, *(run.key for run in selected)), token=token("selected-task-patch"),
    ).items
    assert all(item.disposition == "admitted" for item in started)
    targets = []
    for run, item in zip(selected, started[len(producers):], strict=True):
        settled = wait_until(lambda: (
            job_row(host, item.job_id, "succeeded") or job_row(host, item.job_id, "failed")
            or job_row(host, item.job_id, "blocked")
        ))
        assert settled["state"] == "succeeded", settled["outcome"]
        producer = producers[run.inputs["subject"].producer]
        inherited = decode_request(
            plan.plan.requests[producer.request_id], plan.plan.documents.__getitem__,
        ).spec.trial_documents.task
        x, y = (arg.value for arg in next(item for item in inherited.variables if item.id == "target").value.args)
        assert x == 0.1
        result = jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=item.job_id)
        with result.open_states() as states:
            actual = states.data("/v1/task/output/target_pos")
            np.testing.assert_allclose(actual[..., 0], 0.4)
            np.testing.assert_allclose(actual[..., 1], y)
        targets.append(y)
    assert set(targets) == {-0.2, 0.3}


def test_python_task_field_patch_respects_existing_variation_order_and_overlap():
    original = PythonTaskSelection(type=REACHING_TASK, params={"x": 0.1, "y": -0.2})
    selected = PythonTaskSelection(type=REACHING_TASK, params={"x": 0.4, "y": 0.3})
    replace_task = _patch("task", selected.model_dump(mode="json"))
    edit_field = _patch("task.params.x", 0.7)
    with pytest.raises(ValueError, match="named edits overlap"):
        RolloutParameters(seed=0, trial_policy="paired", trials=("0",),
                          variation={"select": replace_task, "field": edit_field})
    edits = tuple(OverridePatch.model_validate(patch) for patch in (replace_task, edit_field))
    components = new_application_registry_bundle().components
    assert resolve_evaluation_variant(GraphSpec(), original, edits, components=components).task.params == {"x": 0.7, "y": 0.3}
    assert resolve_evaluation_variant(GraphSpec(), original, edits[::-1], components=components).task == selected
    absent_field = OverridePatch.model_validate(_patch("task.params.z", 0.7))
    with pytest.raises(CannotRun, match="missing key/index") as refusal:
        resolve_evaluation_variant(GraphSpec(), original, {"absent": absent_field}, components=components)
    assert refusal.value.path == "/params/variation/absent"


def test_saved_task_replacement_and_scale_use_one_exact_trained_checkpoint(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    registry = new_application_registry_bundle()
    jobs = JobService(host, registry)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "compatibility"
    _save_project(folders, project, _saved_variants())
    reviewed = folders.read_plan(project, revision=None)
    (problem,) = reviewed.plan.diagnostics
    assert (problem.step, problem.path) == ("incompatible", "/params/variation/0")
    assert "target_pos" in problem.detail
    planned = {run.key.step: run for run in reviewed.plan.runs}
    cases = ["baseline", "replacement", "perturbed", "different-dtype"]
    assert len({planned[name].request_id for name in cases}) == len(cases)
    for name in cases:
        assert planned[name].inputs["subject"].producer == planned["train"].key
    decoded = decode_request(
        reviewed.plan.requests[planned["replacement"].request_id],
        reviewed.plan.documents.__getitem__,
    )
    replacement_task = decoded.spec.params["variation"][0]["value"]
    signal = next(signal for signal in replacement_task["signals"] if signal["id"] == "target_pos")
    assert signal["cells"][0]["value"]["args"] == [
        {"kind": "literal", "value": coordinate, "unit": "m"} for coordinate in (-0.05, 0.10)
    ]

    (train,) = jobs.admit_plan(
        folders, reviewed, run_keys=(RunKey("default", "train", "default"),), token=token("train"),
    ).items
    (refused,) = jobs.admit_plan(
        folders, reviewed, run_keys=(RunKey("default", "incompatible", "default"),), token=token("incompatible"),
    ).items
    assert refused.disposition == "rejected" and refused.job_id is None
    started = jobs.admit_plan(
        folders, reviewed, run_keys=tuple(planned[name].key for name in cases), token=token("variants"),
    )
    assert [item.disposition for item in started.items] == ["admitted"] * len(cases)

    def settled(job_id):
        return wait_until(lambda: (
            job_row(host, job_id, "succeeded") or job_row(host, job_id, "failed")
        ))

    trained = settled(train.job_id)
    assert trained["state"] == "succeeded", trained["outcome"]
    completion = trained["outcome"]["completion"]
    checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
    position = "/v1/model/state/plant/effector~1position"
    velocity = "/v1/model/state/plant/effector~1velocity"
    carry = "/v1/model/carry/wrapper/controller/carry/carry"
    hidden = "/v1/model/state/wrapper/controller/cell/hidden"
    target = "/v1/task/output/target_pos"
    data, descriptors, results = {}, {}, {}
    for name, item in zip(cases, started.items, strict=True):
        row = settled(item.job_id)
        assert row["edges"][0]["binding"]["terminal"]["receipt"] == trained["outcome"]["store_success"]
        if name == "different-dtype":
            assert row["state"] == "failed"
            error = row["outcome"]["error"]
            assert error["code"] == "cannot_run"
            assert error["path"].startswith("/inputs/subject/checkpoint/v1/model/value/")
            assert "dtype" in error["message"]
            assert "receipt" not in row["outcome"]
            continue
        assert row["state"] == "succeeded", row["outcome"]
        result = jobs.read_evaluation_result(project=reviewed.project_id, store=reviewed.store_path, job_id=item.job_id)
        results[name] = result
        assert result._manifest().provenance.parents[0].id == checkpoint.pointer.transaction_id
        with result.open_states() as states:
            descriptors[name] = states.descriptor
            assert states.descriptor.schema_id == RAW_STATES_SCHEMA
            assert len(states.data("trial")) == 28
            assert states.data("objective/coverage/effort").item() == 28
            data[name] = {
                key: states.data(key)[0, 0] for key in (
                    position, velocity, carry, hidden, target, "objective/total",
                )
            }
            data[name]["condition"] = states.data("condition")[0]
    baseline, replacement, perturbed = (data[name] for name in cases[:3])
    assert len({item["condition"] for item in data.values()}) == 3
    assert all(d.variables == descriptors["baseline"].variables for d in descriptors.values())
    assert len({d.attrs["source_graph_sha256"] for d in descriptors.values()}) == 1
    b, c = (descriptors[name].attrs for name in ("replacement", "perturbed"))
    assert b["effective_task_sha256"] == c["effective_task_sha256"]
    assert b["effective_graph_sha256"] != c["effective_graph_sha256"]
    assert b["variation_sha256"] != c["variation_sha256"]

    visible = np.any(replacement[target] != 0.0, axis=-1)
    assert len({tuple(mask) for mask in visible}) > 1
    np.testing.assert_array_equal(
        replacement[target][visible],
        np.broadcast_to(np.array([-0.05, 0.10], dtype=np.float32), replacement[target][visible].shape),
    )
    assert not np.array_equal(replacement[target], baseline[target])
    for key in (target, hidden, carry):
        np.testing.assert_array_equal(perturbed[key], replacement[key])
    displacement = replacement[position] - replacement[position][:, :1]
    assert np.max(np.abs(displacement)) > 0
    np.testing.assert_allclose(
        perturbed[position] - perturbed[position][:, :1], 0.5 * displacement, rtol=3e-5, atol=1e-7,
    )
    np.testing.assert_allclose(perturbed[velocity], 0.5 * replacement[velocity], rtol=3e-5, atol=1e-7)
    assert replacement["objective/total"] > 0
    np.testing.assert_allclose(perturbed["objective/total"], 0.25 * replacement["objective/total"], rtol=2e-5)

    read_slice = ArrayContainerReader.read_unverified_slice
    reads = []
    trial_index = 1

    def observe(reader, name, selection):
        reads.append((name, selection))
        return read_slice(reader, name, selection)

    def refuse_full_read(_reader, name):
        pytest.fail(f"Store trial selection verified the whole tensor {name!r}")

    with monkeypatch.context() as context:
        context.setattr(ArrayContainerReader, "read_unverified_slice", observe)
        context.setattr(ArrayContainerReader, "read_tensor", refuse_full_read)
        with results["perturbed"].open_states() as states:
            trial = states.data("trial")[trial_index]
            position_samples = states.select(position, labels={"trial": trial})[0, 0]
    assert reads == [(position, (slice(None), slice(None), trial_index))]
    np.testing.assert_array_equal(position_samples, perturbed[position][trial_index])


def test_an_evaluate_variation_edits_a_task_document_field_by_id():
    """An Evaluate changes one field of its subject's Task document, addressed by ids; a variation naming a member
    the document lacks is refused at that variation."""
    raw = managed_evaluation_definition()
    target = {"kind": "call", "function": "vector", "args": [
        {"kind": "literal", "value": coordinate, "unit": "m"} for coordinate in (0.25, -0.1)
    ]}
    evaluate = raw["steps"]["baseline"]
    evaluate["base"]["inline"]["params"]["variation"] = {"far": _patch("task.variables.target.value", target)}
    missing = deepcopy(evaluate)
    missing["base"]["inline"]["params"]["variation"] = {"absent": _patch("task.variables.goal.value", target)}
    raw["steps"] = {"train": raw["steps"]["train"], "baseline": evaluate, "missing": missing}
    plan = compile_project(Project.model_validate(raw))
    assert [(problem.step, problem.path) for problem in plan.diagnostics] == [("missing", "/params/variation/absent")]
    runs = {run.key: run for run in plan.runs}
    (evaluated,) = (run for run in plan.runs if run.key.step == "baseline")
    request = decode_request(plan.requests[evaluated.request_id], plan.documents.__getitem__).spec
    subject = decode_request(
        plan.requests[runs[evaluated.inputs["subject"].producer].request_id], plan.documents.__getitem__,
    ).spec.trial_documents.task
    variant = resolve_evaluation_variant(
        GraphSpec(), subject, RolloutParameters.model_validate(request.params).variation,
        components=new_application_registry_bundle().components,
    )
    (variable,) = (item for item in variant.task.variables if item.id == "target")
    assert variable.value == CallValue.model_validate(target)
