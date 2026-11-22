"""A document based on another reads as its base's composed content with its differences applied."""

import json
from copy import deepcopy

from hypothesis import given, strategies as st

from feedbax.contracts.project import Project, apply_differences, diff
from feedbax.planner.project_plan import ProjectPlanMemo, compile_project
from feedbax.contracts.project_resolution import ResolutionCode
from tests.support.managed_train_fixture import managed_train_project

MASS = "/interior/components/plant/fields/mass"
SLOT = "/payload/trial_documents/model"

_leaves = st.none() | st.booleans() | st.integers(-3, 3) | st.floats(-2, 2, allow_nan=False) | st.text("ab", max_size=2)
_values = st.recursive(
    _leaves,
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.sampled_from("abcd"), inner, max_size=4),
    max_leaves=12,
)
_payloads = st.dictionaries(st.sampled_from("abcd"), _values, max_size=4)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True)


@given(_payloads, _payloads)
def test_applying_the_difference_of_two_payloads_gives_the_second(base, content):
    assert _json(apply_differences(base, diff(base, content))) == _json(content)
    assert diff(content, content) == {}


def _data():
    data = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    data["documents"]["model"] = data["documents"].pop("point-mass")
    fit = data["steps"].pop("train")
    fit["base"]["inline"]["seed"] = 0
    fit["base"]["inline"]["trial_documents"]["model"] = {"document": "model"}
    other = deepcopy(fit)
    other["base"]["inline"]["trial_documents"]["model"] = {"document": "model-2"}
    data["steps"] = {"fit": fit, "other": other}
    data["documents"]["model-2"] = {"schema": data["documents"]["model"]["schema"], "base": {"document": "model"}}
    return data


def _fields(data, component: str, document: str = "model") -> dict:
    return data["documents"][document]["payload"]["interior"]["components"][component]["fields"]


def _requests(plan) -> dict[str, str]:
    return {run.key.step: run.request_id for run in plan.runs}


def test_a_based_document_plans_as_a_root_document_with_its_composed_content():
    data = _data()
    plain = _requests(compile_project(Project.model_validate(data)))
    assert plain["other"] == plain["fit"]

    data["documents"]["model-2"]["differences"] = {MASS: {"value": 3.0}}
    based = _requests(compile_project(Project.model_validate(data)))
    assert based["other"] != based["fit"]
    composed = deepcopy(data)
    composed["documents"]["model-2"] = deepcopy(data["documents"]["model"])
    _fields(composed, "plant", "model-2")["mass"] = 3.0
    assert _requests(compile_project(Project.model_validate(composed)))["other"] == based["other"]


def test_an_incremental_plan_follows_the_base_chain():
    data = _data()
    data["documents"]["model-2"]["differences"] = {MASS: {"value": 3.0}}
    memo = ProjectPlanMemo()
    first = _requests(compile_project(Project.model_validate(data), memo=memo))

    _fields(data, "cell")["hidden_size"] = _fields(data, "readout")["input_size"] = 5
    project = Project.model_validate(data)
    changed = _requests(compile_project(project, memo=memo))
    assert changed == _requests(compile_project(project))
    assert changed["other"] != first["other"] and changed["fit"] != first["fit"]

    _fields(data, "plant")["mass"] = 5.0
    project = Project.model_validate(data)
    overridden = _requests(compile_project(project, memo=memo))
    assert overridden == _requests(compile_project(project))
    assert overridden["other"] == changed["other"] and overridden["fit"] != changed["fit"]


def test_a_value_names_the_document_that_sets_it():
    data = _data()
    data["documents"]["model-2"]["differences"] = {MASS: {"value": 3.0}}
    project = Project.model_validate(data)
    other = next(run.key for run in compile_project(project).runs if run.key.step == "other")
    sources = compile_project(project, source_runs=[other]).sources[other]
    assert (sources[SLOT + MASS].kind, sources[SLOT + MASS].name, sources[SLOT + MASS].path) == (
        "document", "model-2", MASS,
    )
    inherited = next(path for path in sources if path.startswith(SLOT + "/interior/wires"))
    assert (sources[inherited].kind, sources[inherited].name) == ("document", "model")


def test_a_broken_base_is_a_problem_of_the_based_document_alone():
    for base, code in (("model-3", ResolutionCode.MISSING_DOCUMENT), ("model-2", ResolutionCode.DOCUMENT_CYCLE)):
        data = _data()
        data["documents"]["model-2"]["base"] = {"document": base}
        plan = compile_project(Project.model_validate(data))
        assert [(problem.code, problem.document, problem.step) for problem in plan.diagnostics] == [
            (code, "model-2", "other"),
        ]
        assert [run.key.step for run in plan.runs] == ["fit"]


BATCHES = "/training_config/n_batches"


def _training_data(overridden: bool) -> dict:
    """``fit`` reads ``training``; ``other`` reads ``training-2``, based on it, which sets ``n_batches`` when
    ``overridden``."""
    from feedbax.contracts.training_document import TRAINING_DOCUMENT_KEYS, TRAINING_DOCUMENT_SCHEMA_ID

    data = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    fit = data["steps"].pop("train")
    inline = fit["base"]["inline"]
    data["documents"]["training"] = {
        "schema": TRAINING_DOCUMENT_SCHEMA_ID, "payload": {key: inline[key] for key in TRAINING_DOCUMENT_KEYS},
    }
    inline |= {key: {"document": "training", "pointer": f"/{key}"} for key in TRAINING_DOCUMENT_KEYS}
    other = deepcopy(fit)
    other["base"]["inline"] |= {key: {"document": "training-2", "pointer": f"/{key}"} for key in TRAINING_DOCUMENT_KEYS}
    data["steps"] = {"fit": fit, "other": other}
    data["documents"]["training-2"] = {
        "schema": TRAINING_DOCUMENT_SCHEMA_ID, "base": {"document": "training"},
        "differences": {BATCHES: {"value": 4}} if overridden else {},
    }
    return data


def _variant_requests(data, writes: dict[str, dict]) -> dict[tuple[str, str], str]:
    data = deepcopy(data)
    data["sweep"] = {"rows": {"default": {}} | {
        name: {"patches": {f"documents.{document}.payload.training_config.n_batches": {"value": 7}}}
        for name, document in writes.items()
    }}
    plan = compile_project(Project.model_validate(data))
    assert not plan.diagnostics, plan.diagnostics
    return {(run.key.variant, run.key.step): run.request_id for run in plan.runs}


def test_a_variant_writes_a_based_document_s_composed_content():
    for overridden in (False, True):
        data = _training_data(overridden)
        alone = _requests(compile_project(Project.model_validate(data)))
        ids = _variant_requests(data, {"root": "training", "based": "training-2"})
        assert {step: ids["default", step] for step in ("fit", "other")} == alone
        assert ids["root", "fit"] != alone["fit"]
        assert (ids["root", "other"] == alone["other"]) is overridden
        assert ids["based", "fit"] == alone["fit"] and ids["based", "other"] != alone["other"]

    data = _training_data(False)
    data["sweep"] = {"rows": {"default": {}, "based": {"patches": {
        "documents.training-2.payload.training_config.n_batches": {"value": 7},
    }}}}
    project = Project.model_validate(data)
    other = next(run.key for run in compile_project(project).runs if (run.key.variant, run.key.step) == ("based", "other"))
    source = compile_project(project, source_runs=[other]).sources[other]["/payload" + BATCHES]
    assert (source.kind, source.variant) == ("variant", "based")
