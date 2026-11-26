"""Small external scientific operations for the managed Analyze witness."""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated, ClassVar, NamedTuple

from equinox.nn import StateIndex
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field

from feedbax.analysis.eig import eig
from feedbax.analysis.fp_finder import FixedPointFinder, fp_adam_optimizer
from feedbax.analysis.operations import DatasetOutput, operation
from feedbax.builtins.runtime.components import GRU, Linear
from feedbax.contracts.component import PortType
from feedbax.contracts.expression_values import CallValue, LiteralValue, ValueType
from feedbax.contracts.model_values import ModelFacts, StateValue
from feedbax.contracts.operation import DatasetPort, JaxOperation, ModelPort, TaskPort
from feedbax.contracts.portable import ComponentOptions, Ports
from feedbax.contracts.rollout import RAW_STATES_SCHEMA
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.task_document import Cell, Epoch, Rest, Signal, TaskDocument
from feedbax.dataset import Dataset, Variable
from feedbax.runtime.graph import Component, Graph
from feedbax.runtime.prototypes import input_passthrough_output_prototype
from feedbax.tasks.document import compile_task
from feedbax.tasks.registry import PythonTask, PythonTaskInstance, PythonTaskTemplate


PACKAGE = "feedbax_witness"
FIXED_POINTS = f"{PACKAGE}.fixed_points.v1"
LINEARISED = f"{PACKAGE}.linearised.v1"
ASSEMBLED = f"{PACKAGE}.assembled.v1"
GAIN = f"{PACKAGE}.gain.v1"


class PinnedGRU(GRU):
    """GRU with the construction key used to capture the reference products."""

    @classmethod
    def from_config(cls, config, **context):
        context["key"] = jax.random.split(jax.random.key(7), 3)[0]
        return super().from_config(config, **context)


class PinnedReadout(Linear):
    """Readout with the construction key used to capture the reference products."""

    @classmethod
    def from_config(cls, config, **context):
        context["key"] = jax.random.split(jax.random.key(7), 3)[1]
        return super().from_config(config, **context)


class NoSettings(BaseModel, extra="forbid", frozen=True):
    pass


class _FeedbackState(NamedTuple):
    output: jax.Array


class QuadraticFeedback(Component):
    """One-step recurrence x' = x² / 2 + u with a zero reset."""

    config: ClassVar = NoSettings
    ports: ClassVar = Ports(
        inputs={"input": PortType(dtype="vector", shape=(1,))},
        outputs={"output": PortType(dtype="vector", shape=(1,))},
        required_inputs=("input",),
    )
    component_options: ClassVar = ComponentOptions(
        output_prototype_fn=input_passthrough_output_prototype,
    )
    model_facts: ClassVar = ModelFacts(states=(StateValue("value", ("output",), "1", "unit", ("output",)),))
    state_index: StateIndex

    def __init__(self):
        self.state_index = StateIndex(_FeedbackState(jnp.zeros(1, dtype=jnp.float32)))

    def __call__(self, inputs, state, *, key):
        previous = state.get(self.state_index).output
        value = 0.5 * previous**2 + inputs["input"]
        return {"output": value}, state.set(self.state_index, _FeedbackState(value))


class DriveSettings(BaseModel, extra="forbid", frozen=True):
    nominal: float = 1.0
    disturbance_matrix: tuple[tuple[float]] = ((1.0,),)


class DriveTask(PythonTask[DriveSettings]):
    settings_model = DriveSettings

    @staticmethod
    def template(settings: DriveSettings) -> PythonTaskTemplate:
        vector = ValueType(base="real", shape=(1,))
        document = TaskDocument(
            id="drive", variables=(), epochs=(Epoch(id="rollout", name="Rollout", end=Rest()),), fixed_steps=2,
            signals=(Signal(
                id="drive", name="drive", type=vector,
                cells=(Cell(
                    id="rollout", first_epoch="rollout", last_epoch="rollout",
                    value=CallValue(function="vector", args=(LiteralValue(value=settings.nominal),)),
                ),),
            ),),
        )

        def prepare(model, functions):
            return compile_task(document, functions, model=replace(model, inputs=()))

        return PythonTaskTemplate(signals={"drive": vector}, prepare=prepare)


class FixedPointSettings(BaseModel, extra="forbid", frozen=True):
    loss_tol: float = 1e-6
    unique_tol: float = 0.025
    outlier_tol: float = 1.0


def _spectrum(values, accepted, period):
    magnitude = jnp.abs(values)
    angle = jnp.abs(jnp.angle(values))
    stable = magnitude < 1.0
    decaying = stable & (magnitude > 0.0)
    rotational = angle > 0.0
    decay_time = jnp.where(decaying, -1.0 / jnp.log(magnitude), 0.0)
    rotation_time = jnp.where(rotational, 2.0 * jnp.pi / angle, 0.0)
    metrics = {
        "eigenvalue_real": values.real,
        "eigenvalue_imag": values.imag,
        "magnitude": magnitude,
        "angle_rad": angle,
        "stability_margin": 1.0 - magnitude,
        "stable_mode": stable,
        "decaying_mode": decaying,
        "rotational_mode": rotational,
        "decay_time_steps": decay_time,
        "decay_time_seconds": decay_time * period,
        "rotational_period_steps": rotation_time,
        "rotational_period_seconds": rotation_time * period,
    }
    return {name: jnp.where(accepted[:, None], value, 0) for name, value in metrics.items()}


def _fixed_point_products(model: Graph, levels: tuple[float, ...], params: FixedPointSettings):
    key = jax.random.PRNGKey(0)

    def step(feedback, hidden):
        state = model.init_state(key=key)
        outputs, _, _ = model.step({"feedback": feedback, "memory": hidden}, state, key=key)
        return outputs

    feedback = tuple(jnp.asarray([0.0, 0.0, level], dtype=jnp.float32) for level in levels)
    candidates = jnp.stack([step(value, jnp.zeros(4, dtype=value.dtype))["start"] for value in feedback])
    finder = FixedPointFinder(fp_adam_optimizer())
    roots = []
    spectra = []
    for condition in feedback:
        def advance(hidden):
            return step(condition, hidden)["hidden"]

        fixed = finder.find_and_filter(
            advance, candidates, loss_tol=jnp.asarray(params.loss_tol, dtype=jnp.result_type(float)),
            unique_tol=params.unique_tol, outlier_tol=params.outlier_tol, key=key,
        )
        accepted = fixed.masks["meets_all_criteria"]
        commands = jax.vmap(lambda hidden: step(condition, hidden)["command"])(fixed.fps)
        roots.append({
            "fixed_points": jnp.where(accepted[:, None], fixed.fps, 0),
            "losses": jnp.where(accepted, fixed.losses, 0),
            "accepted": accepted,
            "accepted_count": fixed.counts["meets_all_criteria"],
            "command_norm": jnp.where(accepted, jnp.linalg.norm(commands, axis=-1), 0),
        })
        jacobians = jax.vmap(jax.jacrev(advance))(fixed.fps)
        eigenvalues = eig(
            jacobians, compute_left_eigenvectors=False, compute_right_eigenvectors=False,
        ).vals
        spectra.append(_spectrum(eigenvalues, accepted, model.period_seconds))
    return (
        jax.tree.map(lambda *rows: jnp.stack(rows), *roots),
        jax.tree.map(lambda *rows: jnp.stack(rows), *spectra),
    )


@operation(settings=FixedPointSettings, numerical=JaxOperation(backend="cpu", floating_point_bits=32))
def fixed_points(
    model: Annotated[Graph, ModelPort()],
    task: Annotated[TaskDocument, TaskPort()],
    params: FixedPointSettings,
) -> Annotated[Dataset, DatasetOutput("roots", DatasetPort(schema_ids=(FIXED_POINTS,)))]:
    """Find and linearise roots on each condition's hidden-state manifold."""
    levels = tuple(float(row.variables["conditioning"].value) for row in task.validation_trials)
    roots, spectrum = _fixed_point_products(model, levels, params)
    accepted = roots["accepted"]
    spectral_radius = jnp.max(spectrum["magnitude"], axis=(-1, -2))
    replica_summary = {
        "spectral_radius": spectral_radius,
        "stability_margin": 1.0 - spectral_radius,
        "longest_decay_time_seconds": jnp.max(spectrum["decay_time_seconds"], axis=(-1, -2)),
        "rotational_mode_count": jnp.sum(spectrum["rotational_mode"], axis=(-1, -2)),
    }
    root_dims = ("conditioning", "candidate")
    fields = {
        "fixed_points": Variable(roots["fixed_points"], (*root_dims, "unit")),
        "accepted_count": Variable(roots["accepted_count"], ("conditioning",)),
        "candidate_count": Variable(np.asarray(len(levels)), ()),
    }
    for name in ("losses", "accepted", "command_norm"):
        fields[name] = Variable(roots[name], root_dims)
    fields.update({
        f"spectrum.{name}": Variable(value, (*root_dims, "mode")) for name, value in spectrum.items()
    })
    for name, value in replica_summary.items():
        fields[f"replica_summary.{name}"] = Variable(value, ("conditioning",))
        valid = (
            jnp.any(spectrum["decaying_mode"], axis=(-1, -2))
            if name == "longest_decay_time_seconds" else jnp.any(accepted, axis=-1)
        )
        fields[f"summary.{name}.valid_replica_count"] = Variable(valid.astype(jnp.int32), ("conditioning",))
        fields[f"summary.{name}.mean"] = Variable(jnp.where(valid, value, 0), ("conditioning",))
        fields[f"summary.{name}.sample_std_ddof_1"] = Variable(jnp.zeros_like(value), ("conditioning",))
    return Dataset.from_arrays(
        fields, coords={"conditioning_level": Variable(np.asarray(levels), ("conditioning",))},
        attrs={"candidate_strategy": "conditioning_initial_state_bank"}, schema_id=FIXED_POINTS,
    )


@operation(settings=NoSettings, numerical=JaxOperation(backend="cpu", floating_point_bits=32))
def linearise(
    model: Annotated[Graph, ModelPort()],
    states: Annotated[Dataset, DatasetPort(schema_ids=(RAW_STATES_SCHEMA,))],
    params: NoSettings,
) -> Annotated[Dataset, DatasetOutput("jacobians", DatasetPort(schema_ids=(LINEARISED,)))]:
    """Differentiate one model step at the pre-transition recorded states and inputs."""
    state_field = semantic_key("model", "state", ("feedback",), "value")
    input_field = semantic_key("task", "output", (), "drive")
    points = jnp.asarray(states.data(state_field)).reshape(-1)[:-1, None]
    inputs = jnp.asarray(states.data(input_field)).reshape(-1, 1)
    key = jax.random.PRNGKey(0)
    state_index = model.nodes["feedback"].state_index

    def transition(value, drive):
        state = model.init_state(key=key)
        operating_state = state.set(state_index, _FeedbackState(value))
        output, _, _ = model.step({"drive": drive}, operating_state, key=key)
        return output["value"][0]

    feedback, actuation = jax.vmap(jax.jacrev(transition, argnums=(0, 1)))(points, inputs)
    return Dataset.from_arrays({
        "feedback": Variable(feedback[:, 0], ("transition",)),
        "actuation": Variable(actuation[:, 0], ("transition",)),
    }, schema_id=LINEARISED)


@operation(settings=NoSettings)
def assemble(
    jacobians: Annotated[Dataset, DatasetPort(schema_ids=(LINEARISED,))],
    task: Annotated[PythonTaskInstance[DriveSettings], TaskPort(task_key=f"{PACKAGE}.DriveTask")],
    params: NoSettings,
) -> Annotated[Dataset, DatasetOutput("system", DatasetPort(schema_ids=(ASSEMBLED,)))]:
    """Assemble the scalar time-varying system with the task plant's actuation coefficient."""
    ((actuation,),) = task.parameters.disturbance_matrix
    return Dataset.from_arrays({
        "A": Variable(jacobians.data("feedback"), ("transition",)),
        "B": Variable(jacobians.data("actuation") * actuation, ("transition",)),
    }, schema_id=ASSEMBLED)


class GainSettings(BaseModel, extra="forbid", frozen=True):
    iterations: int = Field(default=32, gt=0)


@operation(settings=GainSettings, numerical=JaxOperation(backend="cpu", floating_point_bits=64))
def power_iteration(
    system: Annotated[Dataset, DatasetPort(schema_ids=(ASSEMBLED,))],
    params: GainSettings,
) -> Annotated[Dataset, DatasetOutput("gain", DatasetPort(schema_ids=(GAIN,)))]:
    """Estimate the induced norm of the finite-horizon system using forward and adjoint sweeps."""
    state = jnp.asarray(system.data("A"), dtype=jnp.float64)
    actuation = jnp.asarray(system.data("B"), dtype=jnp.float64)
    initial = jnp.ones_like(state)

    def forward(disturbance):
        def step(value, row):
            a, b, w = row
            next_value = a * value + b * w
            return next_value, next_value

        return jax.lax.scan(step, jnp.zeros((), dtype=state.dtype), (state, actuation, disturbance))[1]

    _, adjoint = jax.vjp(forward, jnp.zeros_like(initial))

    def iterate(_index, vector):
        (next_vector,) = adjoint(forward(vector))
        return next_vector / jnp.linalg.norm(next_vector)

    unit_vector = jax.lax.fori_loop(0, params.iterations, iterate, initial)
    return Dataset.from_arrays({"gain": Variable(jnp.linalg.norm(forward(unit_vector)), ())}, schema_id=GAIN)
