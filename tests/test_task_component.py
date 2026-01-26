import jax
import jax.numpy as jnp

import equinox as eqx

from feedbax._mapping import WhereDict
from feedbax.graph import init_state_from_component
from feedbax.loss import AbstractLoss
from feedbax.task import (
    AbstractTask,
    TaskComponent,
    TaskInterventionSpecs,
    TaskTrialSpec,
    TrialSpecDependency,
)


class DummyLoss(AbstractLoss):
    label: str = "dummy"

    def term(self, states, trial_specs, model):
        return jnp.array(0.0)


class DummyTask(AbstractTask):
    loss_func: AbstractLoss = DummyLoss()
    n_steps: int = 3
    seed_validation: int = 0
    intervention_specs: TaskInterventionSpecs = TaskInterventionSpecs()
    input_dependencies: dict[str, TrialSpecDependency] = eqx.field(default_factory=dict)

    def get_train_trial(self, key, batch_info=None):
        return TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict(),
            inputs=jnp.zeros((self.n_steps, 1)),
        )

    def get_validation_trials(self, key):
        return self.get_train_trial(key)

    @property
    def n_validation_trials(self) -> int:
        return 1


def test_task_component_open_loop_steps():
    task = DummyTask()
    inputs = jnp.array([[1.0], [2.0], [3.0]])
    intervene = {"foo": jnp.array([10.0, 20.0, 30.0])}
    trial_spec = TaskTrialSpec(
        inits=WhereDict(),
        targets=WhereDict(),
        inputs=inputs,
        intervene=intervene,
    )

    component = TaskComponent(task=task, trial_spec=trial_spec)
    state = init_state_from_component(component)

    out1, state = component({}, state, key=jax.random.PRNGKey(0))
    out2, state = component({}, state, key=jax.random.PRNGKey(1))
    out3, _ = component({}, state, key=jax.random.PRNGKey(2))

    assert (out1["target"] == inputs[0]).all()
    assert (out2["target"] == inputs[1]).all()
    assert (out3["target"] == inputs[2]).all()
    assert out1["intervention_params"]["foo"] == intervene["foo"][0]
    assert out2["intervention_params"]["foo"] == intervene["foo"][1]
    assert out3["intervention_params"]["foo"] == intervene["foo"][2]
