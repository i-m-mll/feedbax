"""Map an operation over the label dimensions its ports take one element of, and stack listed outputs flat."""

from copy import deepcopy

import numpy as np

from feedbax.cli import main
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.project import Project
from feedbax.planner.project_plan import PlannedAnalyzeRun, compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.host.light_jobs import dataset_member
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.persistence.effective_success import SuccessView
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.plugins.declarations import DistributionSource, callable_declaration
from feedbax.contracts.studio_project import StudioMemberName
from feedbax.project_members import merge_cards, split_card
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support import map_probe
from tests.support.host import wait_until
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_list_inputs import _two_model_definition

MASSES = {"nominal": ("light", 0.5), "heavy": ("heavier", 3)}


def _operations():
    bundle = new_application_registry_bundle()
    source = DistributionSource("map_probe", "map-probe", "1.0")
    for function in (
        map_probe.align_each, map_probe.align_over, map_probe.summarize_each, map_probe.pair, map_probe.pair_left,
    ):
        bundle.operations.register_declaration(
            function, callable_declaration(function), key=f"map_probe.{function.__name__}", source=source,
        )
    bundle.seal()
    return bundle.operations


OPERATIONS = _operations()


def _compile(definition):
    return compile_project(Project.model_validate(definition), operations=OPERATIONS)


def _plan(definition):
    plan = _compile(definition)
    assert not plan.diagnostics, plan.diagnostics
    return plan


def _card(plan, step="analyze"):
    (row,) = (row for row in plan.rows if isinstance(row, PlannedAnalyzeRun) and row.key.step == step)
    return row


def _operations_of(definition):
    return definition["documents"]["velocity-analysis"]["payload"]["operations"]


def _named(element, label, value_id):
    return element.names[label][value_id].key


def test_a_port_taking_one_member_plans_a_request_per_member_and_a_model_edit_changes_only_its_requests():
    base = _two_model_definition(MASSES)
    assert len(_card(_plan(base)).requests("align")) == 1
    operations = _operations_of(base)
    operations["align"]["operation"] = "map_probe.align_each"
    operations["summarize"]["operation"] = "map_probe.summarize_each"
    plan = _plan(base)
    card = _card(plan)
    aligned = card.requests("align")
    assert len(aligned) == len(card.requests("summarize")) == 8
    assert all(
        len(decode_request(plan.requests[element.request_id], plan.documents.__getitem__).inputs["states"]) == 1
        for element in aligned
    )
    (figure,) = card.requests("profile")
    members = figure.inputs["summary"]
    assert len(members) == 8
    assert sorted(
        (_named(figure, "source", labels["source"].value_id), _named(figure, "model", labels["model"].value_id),
         labels["mass"].value)
        for labels in (binding.member.labels for binding in members)
    ) == sorted(
        (source, model, mass)
        for source, masses in (("conditions", (1.0, 2.0)), ("conditions_b", (0.5, 3)))
        for model in ("train", "train_b") for mass in masses
    )

    edited = deepcopy(base)
    edited["steps"]["train"]["base"]["inline"]["training_config"]["n_batches"] += 1
    after = {element.request_id for element in _card(_plan(edited)).requests("align")}
    kept = {element.request_id for element in aligned if element.request_id in after}
    assert kept == {
        element.request_id for element in aligned
        if _named(element, "model", element.mapped["model"].value_id) == "train_b"
    }
    assert len(kept) == 4


def test_a_dimensions_parameter_plans_a_request_per_other_label_and_a_figure_splits_the_stack():
    base = _two_model_definition(MASSES)
    align = _operations_of(base)["align"]
    align["operation"] = "map_probe.align_over"
    align["params"]["whole"] = ["model"]
    plan = _plan(base)
    card = _card(plan)
    aligned = card.requests("align")
    assert len(aligned) == 4
    assert {tuple(sorted(element.mapped)) for element in aligned} == {("mass", "source")}
    assert len({(element.mapped["source"].value_id, element.mapped["mass"].value_id) for element in aligned}) == 4
    for element in aligned:
        (states,) = element.inputs.values()
        assert {_named(element, "model", binding.member.labels["model"].value_id) for binding in states} == {
            "train", "train_b",
        }
        assert all("source" not in binding.member.labels for binding in states)
    (summary,) = card.requests("summarize")
    assert {label for entry in summary.entries for label in entry.coordinates} >= {"source", "model", "mass"}
    assert len(summary.entries) == 8

    figure = base["documents"]["velocity-figure"]["payload"]["layers"][0]
    figure["group"] = ["source", "model", "nothing"]
    (problem,) = (problem for problem in _compile(base).diagnostics if problem.step == "analyze")
    assert "nothing" in problem.detail

    align["params"]["whole"] = ["nothing"]
    figure["group"] = ["source", "model", "mass"]
    (problem,) = (problem for problem in _compile(base).diagnostics if problem.element == "align")
    assert problem.path == "payload.operations.align.params.whole"


def _two_cards():
    """One card aligns each Evaluate step's states; the second card's role lists both operations' outputs."""
    definition = _two_model_definition(MASSES)
    documents, steps = definition["documents"], definition["steps"]
    analysis = documents["velocity-analysis"]["payload"]
    align = analysis["operations"].pop("align")
    documents["pair-analysis"] = {"label": "Pair analysis", "schema": documents["velocity-analysis"]["schema"], "payload": {
        "schema_version": analysis["schema_version"],
        "inputs": {"left": {"kind": "dataset"}, "right": {"kind": "dataset"}},
        "operations": {
            f"align_{side}": deepcopy(align) | {"inputs": {"states": {"input": role}}}
            for side, role in (("a", "left"), ("b", "right"))
        },
    }}
    steps["pair"] = {"label": "Pair", "kind": "analyze", "base": {"document": "pair-analysis"}, "inputs": {
        "left": {"step": "conditions"}, "right": {"step": "conditions_b"},
    }}
    analysis["operations"]["summarize"]["inputs"]["velocity"] = {"input": "states"}
    steps["analyze"]["inputs"]["states"] = [
        {"step": "pair", "element": f"align_{side}", "product": "velocity"} for side in ("a", "b")
    ]
    return definition


def _listed(definition):
    plan = _plan(definition)
    (summary,) = _card(plan).requests("summarize")
    labels = {
        label: {_named(summary, label, entry.coordinates[label].value_id) for entry in summary.entries}
        for label in ("source", "element")
    }
    return labels, sorted(plan.requests)


def test_outputs_of_two_cards_listed_on_a_role_stack_flat_and_split_and_merge_change_no_request():
    definition = _two_cards()
    labels, requests = _listed(definition)
    assert labels == {"source": {"conditions", "conditions_b"}, "element": {"align_a", "align_b"}}

    split, presentation = split_card(
        definition, {}, "pair", {"align_b"},
        StudioMemberName(id="pair-b-analysis", label="Pair analysis 2"),
        StudioMemberName(id="pair_b", label="Pair 2"),
    )
    assert split["steps"]["analyze"]["inputs"]["states"][1]["step"] == "pair_b"
    assert _listed(split) == (labels, requests)
    merged, _ = merge_cards(split, presentation, "pair", ["pair_b"])
    assert _listed(merged) == (labels, requests)


def test_listing_outputs_that_already_stack_listed_outputs_is_refused():
    definition = _two_cards()
    documents, steps = definition["documents"], definition["steps"]
    documents["copy-analysis"] = deepcopy(documents["velocity-analysis"])
    documents["copy-analysis"]["payload"]["operations"]["summarize"]["params"]["ddof"] = 0
    steps["copy"] = deepcopy(steps["analyze"]) | {"base": {"document": "copy-analysis"}}
    documents["again-analysis"] = deepcopy(documents["velocity-analysis"])
    del documents["again-analysis"]["payload"]["views"]
    del documents["again-analysis"]["payload"]["operations"]["summarize"]
    documents["again-analysis"]["payload"]["operations"]["summarize_again"] = deepcopy(
        documents["velocity-analysis"]["payload"]["operations"]["summarize"]
    ) | {"operation": "map_probe.summarize_each"}
    steps["again"] = {"label": "Again", "kind": "analyze", "base": {"document": "again-analysis"}, "inputs": {"states": [
        {"step": card, "element": "summarize", "product": "summary"} for card in ("analyze", "copy")
    ]}}
    problems = [problem for problem in _compile(definition).diagnostics if problem.step == "again"]
    assert problems and all("one level" in problem.detail for problem in problems), problems


def test_mapped_and_listed_outputs_stack_by_their_labels_when_run(request, probe_operations, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host = request.getfixturevalue("private_host")
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    host.policy(compute=2, light=2)
    definition = _two_cards()
    pair = definition["documents"]["pair-analysis"]["payload"]["operations"]
    summarize = definition["documents"]["velocity-analysis"]["payload"]["operations"]["summarize"]
    pair["align_over"] = deepcopy(pair["align_a"]) | {"operation": "map_probe.align_over"}
    pair["align_over"]["params"]["whole"] = ["model"]
    pair["align_each"] = deepcopy(pair["align_b"]) | {"operation": "map_probe.align_each"}
    for name in ("over", "each"):
        pair[f"summarize_{name}"] = deepcopy(summarize) | {"inputs": {"velocity": {
            "element": f"align_{name}", "product": "velocity",
        }}}
    del definition["steps"]["report"]
    projects = tmp_path / "projects"
    folders = ProjectFolderService(projects, operations=probe_operations)
    _save_project(folders, "mapped", Project.model_validate(definition))
    review = folders.read_plan("mapped", revision=None)
    plan = review.plan
    assert not plan.diagnostics, plan.diagnostics

    assert main(["run", str(projects / "mapped")]) == 0
    jobs = wait_until(lambda: (
        jobs if (jobs := host.jobs(project=review.project_id)) and all(job["outcome"] for job in jobs) else None
    ))
    assert all(job["state"] == "succeeded" for job in jobs), [
        job["outcome"] for job in jobs if job["state"] != "succeeded"
    ]
    store = ImmutableArtifactBlobProvider(projects / "mapped" / "results")
    by_request = {job["request"]: job for job in jobs}

    def summary(step, name):
        (element,) = _card(plan, step).requests(name)
        job = by_request[element.request_id]
        receipt = SuccessView(store).read_success(ContentObjectRef.model_validate(job["outcome"]["store_success"])).receipt
        with open_stored_dataset(store, dataset_member(receipt.products["summary"][0])) as data:
            coords = data.descriptor.coords
            stacking = coords["member"].dims
            shape = tuple(data.descriptor.dimensions[dim] for dim in stacking)

            def along(label):
                values = np.asarray(data.data(label))
                axes = [stacking.index(dim) for dim in coords[label].dims]
                spread = values.reshape([shape[axis] if axis in axes else 1 for axis in range(len(shape))])
                return [
                    element.names[label][value].key if label in element.names else value
                    for value in np.broadcast_to(spread, shape).ravel().tolist()
                ]

            return data.descriptor.variables["mean"].dims[:len(stacking)], sorted(zip(*(
                along(label) for label in ("source", "element", "model", "mass") if label in coords
            ), strict=True))

    pairs = [(model, mass) for model in ("train", "train_b") for mass in ("nominal", "heavy")]
    dims, labels = summary("pair", "summarize_over")
    assert labels == sorted(("conditions", *pair) for pair in pairs), dims
    dims, labels = summary("pair", "summarize_each")
    assert labels == sorted(("conditions_b", model, mass) for model in ("train", "train_b") for mass in ("light", "heavier")), dims
    dims, labels = summary("analyze", "summarize")
    assert dims == ("member",) and labels == sorted(
        [("conditions", "align_a", *pair) for pair in pairs]
        + [("conditions_b", "align_b", model, mass) for model in ("train", "train_b") for mass in ("light", "heavier")]
    )
