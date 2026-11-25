"""Gather list inputs from several steps and stack their runs by label."""

from copy import deepcopy
from itertools import product

from fastapi.testclient import TestClient

from feedbax.cli import main
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.project import Project
from feedbax.planner.project_plan import ElementKey, PlannedAnalyzeRun, compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.host.light_jobs import dataset_member
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.persistence.effective_success import SuccessView
from feedbax.project_members import change_key
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.host import wait_until
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_project_analyze_plan import _operations
from tests.test_saved_light_run import _saved_light_definition


def _two_model_definition(masses=None):
    """Two Evaluate steps over a mass axis, each taking the Train step and one chosen row of a second.

    ``masses`` renames and revalues the second Evaluate's masses, so that no two Evaluate rows are one request.
    """
    definition = _saved_light_definition(paired=True)
    steps = definition["steps"]
    del steps["baseline"]
    steps["train_b"] = deepcopy(steps["train"]) | {"sweep": {"axes": {"rate": {
        "values": {
            key: {"value": rate, "patches": {"settings.optimizer.learning_rate.value": {"value": rate}}}
            for key, rate in (("low", 0.001), ("high", 0.003))
        },
    }}}}
    conditions = steps["conditions"]
    conditions["inputs"]["subject"] = [
        {"step": "train", "checkpoint": "last"},
        {"step": "train_b", "checkpoint": "last", "where": {"axes": {"rate": [{"id": "high", "label": "0.003"}]}}},
    ]
    conditions["sweep"]["axes"] = {"model": {"input": "subject"}, **conditions["sweep"]["axes"]}
    steps["conditions_b"] = deepcopy(conditions)
    values = steps["conditions_b"]["sweep"]["axes"]["mass"]["values"]
    for old, (new, mass) in (masses or {}).items():
        value = values.pop(old)
        value["value"] = mass
        for patch in value["patches"].values():
            patch["value"] = mass
        values[new] = value
    steps["analyze"]["inputs"]["states"] = [{"step": "conditions"}, {"step": "conditions_b"}]
    bounds = definition["documents"]["velocity-figure"]["payload"]["layers"][0]
    bounds["group"] = ["source", "model", "mass"]
    bounds["style"] = {"label": "{source} {model} {mass}"}
    return definition


def test_one_analyze_stacks_two_evaluates_of_two_train_steps_by_label(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=2, light=1)
    projects = tmp_path / "projects"
    folders = ProjectFolderService(projects, operations=_operations())
    definition = _two_model_definition({"nominal": ("light", 0.5), "heavy": ("heavier", 3)})
    _save_project(folders, "models", Project.model_validate(definition))
    review = folders.read_plan("models", revision=None)
    plan = review.plan
    assert not plan.diagnostics
    assert sum(run.key.step in {"conditions", "conditions_b"} for run in plan.runs) == 8

    assert main(["run", str(projects / "models")]) == 0
    jobs = wait_until(lambda: (
        jobs if (jobs := host.jobs(project=review.project_id)) and all(job["outcome"] for job in jobs) else None
    ))
    assert all(job["state"] == "succeeded" for job in jobs), [
        job["outcome"] for job in jobs if job["state"] != "succeeded"
    ]
    (analyze,) = (run for run in plan.rows if isinstance(run, PlannedAnalyzeRun))
    align = plan.elements[ElementKey(analyze.key, "align")]
    decoded = decode_request(plan.requests[align.request_id], plan.documents.__getitem__)
    assert len(decoded.inputs["states"]) == 8
    (job,) = (job for job in jobs if job["request"] == align.request_id)
    store = ImmutableArtifactBlobProvider(projects / "models" / "results")
    receipt = SuccessView(store).read_success(ContentObjectRef.model_validate(job["outcome"]["store_success"])).receipt
    with open_stored_dataset(store, dataset_member(receipt.products["velocity"][0])) as data:
        stacked = data.descriptor.variables["parallel_velocity"].dims
        names = align.names
        labels = list(zip(
            *(
                [names[name][value].key for value in data.data(name).tolist()] if name in names
                else data.data(name).tolist()
                for name in ("source", "model", "mass")
            ),
            strict=True,
        ))
        rates = data.data("rate").tolist()
        assert data.descriptor.coords["trial"].dims == ("trial",)
    assert stacked == ("member", "checkpoint", "condition", "trial", "state_time")
    assert sorted(labels) == sorted(
        (source, model, mass)
        for source, masses in (("conditions", ("nominal", "heavy")), ("conditions_b", ("light", "heavier")))
        for model, mass in product(("train", "train_b"), masses)
    )
    assert [rate is None for rate in rates] == [model == "train" for _, model, _ in labels]
    (profile,) = (job for job in jobs if job["request"] == plan.elements[ElementKey(analyze.key, "profile")].request_id)
    assert profile["state"] == "succeeded"

    with TestClient(create_app(projects_dir=projects)) as client:
        response = client.post("/api/projects/models/authoring/analyze", json={
            "revision": review.revision,
            "key": {"variant": analyze.key.variant, "step": analyze.key.step, "row": analyze.key.row},
        })
    assert response.status_code == 200, response.text
    assert "/v1/model/state/plant/effector~1velocity" in response.json()["states"]["variables"]


def test_zip_pairs_two_inputs_by_label_and_refuses_a_partial_pairing():
    definition = _two_model_definition()
    analysis = definition["documents"]["velocity-analysis"]["payload"]
    analysis["inputs"] = {"left": {"kind": "dataset"}, "right": {"kind": "dataset"}}
    analysis["operations"]["align"]["inputs"]["states"] = {"input": "left"}
    del analysis["views"]
    reordered = [{"id": value, "label": value} for value in ("heavy", "nominal")]
    analyze = definition["steps"]["analyze"]
    analyze["inputs"] = {"left": {"step": "conditions"}, "right": {"step": "conditions_b", "where": {"axes": {"mass": reordered}}}}
    analyze["sweep"] = {"axes": {"a": {"input": "left"}, "b": {"input": "right"}}, "combination": "zip"}
    del definition["steps"]["report"]
    plan = compile_project(Project.model_validate(definition), operations=_operations())
    assert not plan.diagnostics
    pairs = [run for run in plan.rows if isinstance(run, PlannedAnalyzeRun)]
    assert len(pairs) == 4
    for run in pairs:
        (left,), (right,) = run.inputs["left"], run.inputs["right"]
        assert left.producer.row == right.producer.row

    analyze["inputs"]["right"]["where"]["axes"]["mass"] = reordered[:1]
    (problem,) = compile_project(Project.model_validate(definition), operations=_operations()).diagnostics
    assert (problem.step, problem.path) == ("analyze", "sweep.combination")
    assert "one to one" in problem.detail


def _request_ids(definition):
    plan = compile_project(Project.model_validate(definition), operations=_operations())
    assert not plan.diagnostics, plan.diagnostics
    return plan, {element.key.name: element.request_id for element in plan.elements.values()}


def _figures_by(definition):
    """Three figures of the summary, splitting by source and model, by model, and by neither."""
    documents = definition["documents"]
    views = documents["velocity-analysis"]["payload"]["views"]
    for name, group in (("models", ["model", "mass"]), ("masses", ["mass"])):
        figure = deepcopy(documents["velocity-figure"])
        figure["payload"]["layers"][0] |= {"group": group, "style": {"label": " ".join(f"{{{g}}}" for g in group)}}
        documents[f"{name}-figure"] = figure
        views[name] = deepcopy(views["profile"]) | {"figure": {"document": f"{name}-figure"}}
    return definition


def test_change_key_and_rename_recompute_no_operation_and_redraw_the_figures_that_print_the_name():
    base = _figures_by(_two_model_definition({"nominal": ("light", 0.5), "heavy": ("heavier", 3)}))
    plan, before = _request_ids(base)
    operations = {name for name, request in before.items() if request.startswith("analyze:")}
    assert operations == {"align", "summarize"}
    for step, printing in (("train", {"profile", "models"}), ("conditions", {"profile"})):
        renamed = deepcopy(base)
        renamed["steps"][step]["label"] = "Renamed"
        rekeyed, _ = change_key(deepcopy(base), {}, "steps", step, f"{step}_x")
        for edited, printed in ((renamed, "Renamed"), (rekeyed, f"{step}_x")):
            plan, after = _request_ids(edited)
            assert {name for name in before if before[name] != after[name]} == printing
            label = "model" if step == "train" else "source"
            (profile,) = (element for element in plan.elements.values() if element.key.name == "profile")
            spec = decode_request(plan.requests[profile.request_id], plan.documents.__getitem__).spec
            assert printed in spec.names[label].values()


def test_a_key_two_selections_share_is_qualified_by_its_step_and_kept_across_their_edits():
    def keys(definition):
        plan = compile_project(Project.model_validate(definition), operations=_operations())
        assert not plan.diagnostics, plan.diagnostics
        (run,) = (run for run in plan.rows if isinstance(run, PlannedAnalyzeRun))
        return sorted(binding.member.key for binding in run.inputs["states"])

    base = _two_model_definition()
    before = keys(base)
    assert {key.split("/")[0] for key in before} == {"conditions", "conditions_b"}
    edited = deepcopy(base)
    edited["steps"]["conditions_b"]["base"]["inline"]["params"]["seed"] += 1
    assert keys(edited) == before


def test_a_duplicate_step_read_in_place_of_the_original_plans_the_same_operations():
    base = _two_model_definition({"nominal": ("light", 0.5), "heavy": ("heavier", 3)})
    _, before = _request_ids(base)
    duplicated = deepcopy(base)
    duplicated["steps"]["conditions_copy"] = deepcopy(duplicated["steps"]["conditions"]) | {"label": "Copy"}
    duplicated["steps"]["analyze"]["inputs"]["states"][0] = {"step": "conditions_copy"}
    _, after = _request_ids(duplicated)
    assert {name: after[name] for name in ("align", "summarize")} == {name: before[name] for name in ("align", "summarize")}


def test_two_roles_over_the_same_members_agree_on_label_ids():
    definition = _two_model_definition({"nominal": ("light", 0.5), "heavy": ("heavier", 3)})
    definition["documents"]["velocity-analysis"]["payload"]["inputs"]["again"] = {"kind": "dataset"}
    analyze = definition["steps"]["analyze"]["inputs"]
    analyze["again"] = deepcopy(analyze["states"])
    plan = compile_project(Project.model_validate(definition), operations=_operations())
    (run,) = (run for run in plan.rows if isinstance(run, PlannedAnalyzeRun))
    states, again = ([binding.member.labels for binding in run.inputs[role]] for role in ("states", "again"))
    assert states == again
    assert len({labels["source"].value_id for labels in states}) == 2
