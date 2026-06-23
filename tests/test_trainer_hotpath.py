from __future__ import annotations

from contextlib import contextmanager

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from feedbax.config.mapping import WhereDict
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.contracts.graphs.serialization import spec_to_graph
from feedbax.objectives.loss import AbstractLoss
from feedbax.tasks import AbstractTask, TaskInterventionSpecs, TaskTrialSpec, TrialSpecDependency
from feedbax.training import trainer as trainer_module
from feedbax.training.trainer import TaskTrainer


class _WeightSumLoss(AbstractLoss):
    label: str = "weight_sum"

    def term(self, states, trial_specs, model):
        return jnp.ones((1,)) * jnp.sum(model.nodes["readout"].layer.weight)


class _TinyTask(AbstractTask):
    loss_func: AbstractLoss = _WeightSumLoss()
    n_steps: int = 2
    seed_validation: int = 0
    intervention_specs: TaskInterventionSpecs = TaskInterventionSpecs()
    input_dependencies: dict[str, TrialSpecDependency] = eqx.field(default_factory=dict)

    def get_train_trial(self, key, batch_info=None):
        return TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict(),
            inputs=jnp.ones((self.n_steps, 1)),
        )

    def get_validation_trials(self, key):
        return TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict(),
            inputs=jnp.ones((self.n_validation_trials, self.n_steps, 1)),
        )

    def validation_plots(self, states, trial_specs=None):
        return {}

    @property
    def n_validation_trials(self) -> int:
        return 1


class _ProgressUpdateRecorder:
    def __init__(self):
        self.subdescriptions: list[str] = []

    def subdescription(self, value: str) -> None:
        self.subdescriptions.append(value)


def _linear_model():
    return spec_to_graph(
        GraphSpec(
            nodes={
                "readout": ComponentSpec(
                    type="Linear",
                    params={"input_size": 1, "output_size": 1, "activation": "identity"},
                    input_ports=["input"],
                    output_ports=["output"],
                )
            },
            input_ports=["input"],
            output_ports=["output"],
            input_bindings={"input": ("readout", "input")},
            output_bindings={"output": ("readout", "output")},
        )
    )


def _train(*, disable_progress: bool, n_batches: int = 3, log_step: int = 2):
    task = _TinyTask()
    model = _linear_model()
    trainer = TaskTrainer(optimizer=optax.sgd(0.1), checkpointing=False)
    return trainer(
        task,
        model,
        n_batches=n_batches,
        batch_size=1,
        where_train=lambda graph: graph.nodes["readout"],
        disable_progress=disable_progress,
        log_step=log_step,
        verbose_progress=False,
        key=jax.random.PRNGKey(0),
    )


def test_task_trainer_disabled_progress_does_not_itemize_batch_loss(monkeypatch) -> None:
    array_cls = type(jnp.array(0.0))
    original_item = array_cls.item

    def fail_item(self, *args, **kwargs):
        raise AssertionError("disabled progress should not call JAX array .item()")

    monkeypatch.setattr(array_cls, "item", fail_item)

    _train(disable_progress=True, n_batches=3, log_step=2)

    monkeypatch.setattr(array_cls, "item", original_item)


def test_task_trainer_progress_subdescription_uses_log_cadence(monkeypatch) -> None:
    recorder = _ProgressUpdateRecorder()

    @contextmanager
    def fake_progress(iterable, **kwargs):
        yield iterable, recorder

    monkeypatch.setattr(trainer_module, "progress_piter", fake_progress)

    _train(disable_progress=False, n_batches=5, log_step=2)

    assert len(recorder.subdescriptions) == 3
