"""A continuation whose Model cannot take its checkpoint's values reads Can't run when it is planned."""

from copy import deepcopy

import jax
import pytest

from feedbax.contracts.project import Project, diff
from feedbax.planner.project_plan import compile_project
from tests.support.managed_train_fixture import managed_train_project


def _data(**inline):
    data = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    data["documents"]["model"] = data["documents"].pop("point-mass")
    fee = data["steps"].pop("train")
    fee["base"]["inline"]["trial_documents"]["model"] = {"document": "model"}
    cont = deepcopy(fee)
    cont["base"]["inline"]["checkpoint_initialization"] = "initialize_from"
    cont["base"]["inline"]["trial_documents"]["model"] = {"document": "model-2"}
    for name, value in inline.items():
        cont["base"]["inline"][name] = value
    cont["inputs"] = {"checkpoint": {"step": "fee", "checkpoint": "last"}}
    data["steps"] = {"fee": fee, "cont": cont}
    return data


def _variant(data, edit):
    content = deepcopy(data["documents"]["model"]["payload"])
    edit(content["interior"]["components"])
    data["documents"]["model-2"] = {
        "label": "Model 2",
        "schema": data["documents"]["model"]["schema"], "base": {"document": "model"},
        "differences": {
            path: value.model_dump(mode="json")
            for path, value in diff(data["documents"]["model"]["payload"], content).items()
        },
    }
    return data


def _problem(data):
    plan = compile_project(Project.model_validate(data))
    assert not plan.diagnostics
    return next(run for run in plan.runs if run.key.step == "cont").admission_problem


def _hidden(components, size=8):
    components["cell"]["fields"]["hidden_size"] = size


def test_a_changed_layer_size_reads_cant_run_on_the_component():
    problem = _problem(_variant(_data(), _hidden))
    assert problem.model_address.element.id == "cell"
    assert problem.path == "/trial_documents/model/interior/components/cell"
    assert problem.detail == "hidden_size is 4 in the checkpoint and 8 here."


@pytest.mark.parametrize("edit", [
    lambda components: components["plant"]["fields"].update(mass=2.0),
    lambda components: components["readout"]["fields"].update(initialization="zeros"),
], ids=["source value", "learned owner, same shapes"])
def test_a_change_no_learned_shape_depends_on_plans(edit):
    assert _problem(_variant(_data(), edit)) is None


def test_a_differing_replicate_count_reads_on_its_field():
    data = _variant(_data(), lambda components: None)
    data["steps"]["cont"]["base"]["inline"]["training_config"]["replicates"] = 2
    problem = _problem(data)
    assert problem.path == "/training_config/replicates"


def test_a_build_that_raises_under_the_trace_is_the_reason(monkeypatch):
    from feedbax.compiler import builders

    original = builders.build_component

    def build(name, node_type, params, **context):
        if params.get("hidden_size") == 6:
            raise ValueError("no GRU of size 6")
        return original(name, node_type, params, **context)

    monkeypatch.setattr(builders, "build_component", build)
    problem = _problem(_variant(_data(), lambda components: _hidden(components, 6)))
    assert (problem.model_address.element.id, problem.detail) == ("cell", "no GRU of size 6")


def test_a_plan_without_a_changed_learned_owner_makes_no_jax_call(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("traced")

    monkeypatch.setattr(jax, "eval_shape", forbidden)
    assert _problem(_variant(_data(), lambda components: components["plant"]["fields"].update(mass=2.0))) is None
