"""Qualify saved task and connection variants on one real Train checkpoint."""

from copy import deepcopy

import numpy as np
import pytest

from feedbax.compiler.model_document import lower_model_document
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.jobs import StudioProjectWindowMapProblem
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.rollout import RolloutParameters
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.contracts.task_document import PythonTaskSelection
from feedbax.host import HostClient
from feedbax.jobs import JobService, PlanJobReader
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_evaluate import CannotRun, RAW_STATES_SCHEMA, decode_rollout_request
from feedbax.tasks.reaching import REACHING_TASK
from feedbax.training.evaluation_variant import resolve_evaluation_variant
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project, managed_evaluation_definition
from tests.test_local_job_host import host as host_fixture, job_row, wait_until
from tests.test_model_document_plan import _add_attachment, _attachment_owner

host = host_fixture


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
    perturbed["model_attachments"] = {"wire": [_add_attachment(_attachment_owner("read"), "scale", None, 0.5)]}
    incompatible = deepcopy(task["payload"])
    incompatible["signals"] = [signal for signal in incompatible["signals"] if signal["id"] != "target_pos"]
    dtype = step([])
    dtype["model_attachments"] = {"dtype": [{
        "op": "set_field",
        "address": {
            "document": "reach-model", "composites": ["wrapper", "controller"],
            "element": {"kind": "field", "parent": {"kind": "component", "id": "readout"}, "path": ["fields", "dtype"]},
        },
        "value": "float16",
    }]}
    definition["steps"] = {
        "train": steps["train"], "baseline": baseline, "replacement": replacement, "perturbed": perturbed,
        "different-dtype": dtype,
        "incompatible": step([_patch("task", incompatible)]),
    }
    return Project.model_validate(definition)


def test_two_crossed_wire_insertions_compose_through_saved_plan(tmp_path):
    definition = managed_evaluation_definition()
    registries = new_application_registry_bundle()
    from feedbax.contracts.model_document import ModelDocument
    from feedbax.contracts.task_document import TaskDocument

    model = ModelDocument.model_validate(definition["documents"]["reach-model"]["payload"])
    task = TaskDocument.model_validate(definition["documents"]["task"]["payload"])
    graph = lower_model_document("reach-model", model, registries.components).graph
    first = next(wire for wire in graph.wires if wire.target_node == "plant")
    second = next(wire for wire in graph.wires if wire.target_node == "wrapper")
    baseline = deepcopy(definition["steps"]["baseline"])
    axes = {}
    for name, wire in (("first", first), ("second", second)):
        owner = _attachment_owner(wire.authored_id()) | {"composites": []}
        attachment = _add_attachment(owner, name, None)
        axes[name] = {"values": {"inserted": {"patches": {
            f"model_attachments.{name}": {"op": "add", "value": [attachment]},
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
    variant = resolve_evaluation_variant(graph, task, variation)
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
            "where": {"axes": {}},
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


def _selected_task_variants():
    raw = managed_evaluation_definition()
    train = raw["steps"]["train"]
    inline = train["base"]["inline"]
    inline["training_config"].pop("n_reach_steps")
    inline["trial_documents"].update(task=None, python_task={"document": "task"}, input_overrides={})
    raw["documents"]["task"] = {
        "schema": "feedbax.spec.python_task",
        "payload": {"type": REACHING_TASK, "params": {"x": 0.1, "n_steps": 5}},
    }
    raw["documents"]["objective"]["payload"]["terms"][0].pop("epochs")
    train["sweep"] = {"axes": {"target": {"values": {
        name: {"patches": {"trial_documents.python_task.params.y": {"op": "add", "value": y}}}
        for name, y in (("low", -0.2), ("high", 0.3))
    }}}}
    evaluate = raw["steps"]["baseline"]
    evaluate["sweep"] = {"axes": {"trained": {"input": "subject"}}}
    params = evaluate["base"]["inline"]["params"]
    params["trial_set"] = {"kind": "sample", "count": 1}
    params["variation"] = {"target-x": _patch("task.params.x", 0.4)}
    refused_step = deepcopy(evaluate)
    refused_step["base"]["inline"]["params"]["variation"] = {"bad-length": _patch("task.params.n_steps", 0)}
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
        token="refused-task-patch",
    ).items
    assert len(refused) == 2 and all(item.disposition == "rejected" and item.job_id is None for item in refused)
    assert not host.jobs()

    started = jobs.admit_plan(
        folders, plan, run_keys=(*producers, *(run.key for run in selected)), token="selected-task-patch",
    ).items
    assert all(item.disposition == "admitted" for item in started)
    targets = []
    for run, item in zip(selected, started[len(producers):], strict=True):
        settled = wait_until(lambda: (
            job_row(host, item.job_id, "succeeded") or job_row(host, item.job_id, "failed")
            or job_row(host, item.job_id, "blocked")
        ), timeout=30)
        assert settled["state"] == "succeeded", settled["outcome"]
        producer = producers[run.inputs["subject"].producer]
        inherited = decode_request(
            plan.plan.requests[producer.request_id], plan.plan.documents.__getitem__,
        ).spec.trial_documents.python_task
        assert inherited.params["x"] == 0.1
        result = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=item.job_id)
        with result.open_states() as states:
            actual = states.data("/v1/task/output/reach_target")
            np.testing.assert_allclose(actual[..., 0], 0.4)
            np.testing.assert_allclose(actual[..., 1], inherited.params["y"])
        targets.append(inherited.params["y"])
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
    assert resolve_evaluation_variant(GraphSpec(), original, edits).task.params == {"x": 0.7, "y": 0.3}
    assert resolve_evaluation_variant(GraphSpec(), original, edits[::-1]).task == selected
    absent_field = OverridePatch.model_validate(_patch("task.params.z", 0.7))
    with pytest.raises(CannotRun, match="missing key/index") as refusal:
        resolve_evaluation_variant(GraphSpec(), original, {"absent": absent_field})
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
        folders, reviewed, run_keys=(RunKey("default", "train", "default"),), token="train",
    ).items
    (refused,) = jobs.admit_plan(
        folders, reviewed, run_keys=(RunKey("default", "incompatible", "default"),), token="incompatible",
    ).items
    assert refused.disposition == "rejected" and refused.job_id is None
    started = jobs.admit_plan(
        folders, reviewed, run_keys=tuple(planned[name].key for name in cases), token="variants",
    )
    assert [item.disposition for item in started.items] == ["admitted"] * len(cases)

    def settled(job_id):
        return wait_until(lambda: (
            job_row(host, job_id, "succeeded") or job_row(host, job_id, "failed")
        ), timeout=25)

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
        assert row["edges"][0]["binding"]["completion"] == completion
        if name == "different-dtype":
            assert row["state"] == "failed"
            error = row["outcome"]["error"]
            assert error["code"] == "cannot_run"
            assert error["path"].startswith("/inputs/subject/checkpoint/model/")
            assert "dtype" in error["message"]
            assert "receipt" not in row["outcome"]
            continue
        assert row["state"] == "succeeded", row["outcome"]
        result = jobs.read_evaluation_result(project=project, store=reviewed.store_path, job_id=item.job_id)
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
