"""Built-in operations over saved Evaluate dataset products."""

import numpy as np
import pytest
from pydantic import ValidationError

from feedbax.builtins.operations import (
    AlignedVelocityParameters,
    VelocitySummaryParameters,
    aligned_velocity,
    summarize_velocity,
)
from feedbax.contracts.dataset import TrialFigureTime
from feedbax.dataset import Dataset, Variable
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_dataset, publish_dataset
from feedbax.application.composition import load_application
from feedbax.persistence.canonical_evaluate import RAW_STATES_SCHEMA, state_key


_POSITION = state_key("plant", "effector/position")
_VELOCITY = state_key("plant", "effector/velocity")
_TARGET = "/v1/task/output/reach_target"
_PARAMETERS = AlignedVelocityParameters(position=_POSITION, velocity=_VELOCITY, target=_TARGET)


def _states(*, changing_target=False):
    members = ("checkpoint", "condition", "trial")
    state_axes = (*members, "state_time", "xy")
    transition_axes = (*members, "transition", "xy")
    position = np.full((2, 2, 3, 3, 2), 0.5, dtype=np.float32)
    position[..., 1:, :] += 1.0
    velocities = np.arange(position.size, dtype=np.float32).reshape(position.shape) / 4
    directions = np.broadcast_to(np.array([[1, 0], [0, -1], [-1, 0]], np.float32), (2, 2, 3, 2))
    target = np.repeat((position[..., 0, :] + directions)[..., None, :], 2, axis=-2)
    if changing_target:
        target[..., 0, :] = position[..., 0, :]
    time = np.array([0, 0.01, 0.02], dtype=np.float64)
    dataset = Dataset.from_arrays(
        {
            _POSITION: Variable(position, state_axes, units="m", frame="world"),
            _VELOCITY: Variable(velocities, state_axes, units="m/s", frame="world"),
            _TARGET: Variable(target, transition_axes, units="m", frame="world"),
        },
        coords={
            "checkpoint": Variable(np.array([10, 20]), ("checkpoint",)),
            "condition": Variable(np.array(["control", "perturbed"]), ("condition",)),
            "trial": Variable(np.array(["alpha", "beta", "gamma"]), ("trial",)),
            "state_time": Variable(time, ("state_time",), units="s"),
            "transition_start": Variable(time[:-1], ("transition",), units="s"),
            "transition_end": Variable(time[1:], ("transition",), units="s"),
        },
        schema_id=RAW_STATES_SCHEMA,
    )
    parallel = np.einsum("...ti,...i->...t", velocities, directions)
    lateral = directions[..., 0, None] * velocities[..., 1] - directions[..., 1, None] * velocities[..., 0]
    return dataset, parallel, lateral


def test_registered_operations_publish_typed_products_from_saved_states(tmp_path):
    states, parallel, lateral = _states()
    store = ImmutableArtifactBlobProvider(tmp_path)
    states_ref = publish_dataset(store, states)
    operations = load_application().bundle.operations
    run = operations.resolve("feedbax.aligned_velocity")
    with (
        store.open_owned_object(states_ref.object_ref) as path,
        open_dataset(path, expected_descriptor_sha256=states_ref.descriptor_sha256) as saved,
    ):
        velocity = run.defining_object(saved, run.value.parameter_model.model_validate(_PARAMETERS.model_dump()))
    np.testing.assert_allclose(velocity.data("parallel_velocity"), parallel, atol=1e-6)
    np.testing.assert_allclose(velocity.data("lateral_velocity"), lateral, atol=1e-6)
    run = operations.resolve("feedbax.summarize_velocity")
    for ddof, scale in ((0, 0.5), (1, 2.0)):
        product = run.defining_object(velocity, run.value.parameter_model.model_validate({"ddof": ddof, "bounds_scale": scale}))
        result_ref = publish_dataset(store, product)
        with (
            store.open_owned_object(result_ref.object_ref) as path,
            open_dataset(path, expected_descriptor_sha256=result_ref.descriptor_sha256) as reopened,
        ):
            mean, sd = parallel.mean(axis=2), parallel.std(axis=2, ddof=ddof)
            np.testing.assert_allclose(reopened.data("mean"), mean, atol=1e-6)
            np.testing.assert_allclose(reopened.data("standard_deviation"), sd, atol=1e-6)
            np.testing.assert_allclose(reopened.data("lower"), mean - scale * sd, atol=2e-6)
            np.testing.assert_allclose(reopened.data("upper"), mean + scale * sd, atol=2e-6)
            np.testing.assert_array_equal(reopened.data("count"), np.full(mean.shape, 3))
            for name in ("checkpoint", "condition", "state_time"):
                np.testing.assert_array_equal(reopened.data(name), states.data(name))


def test_velocity_uses_the_authored_reference_transition():
    states, parallel, _ = _states(changing_target=True)
    np.testing.assert_allclose(aligned_velocity(states, _PARAMETERS).data("parallel_velocity"), parallel)
    first = AlignedVelocityParameters(position=_POSITION, velocity=_VELOCITY, target=_TARGET, target_transition=0)
    with pytest.raises(ValueError, match="finite nonzero length"):
        aligned_velocity(states, first)


def test_peak_velocity_summary_discards_time_and_trial_but_keeps_member_groups():
    states, parallel, _ = _states()
    velocity = aligned_velocity(states, _PARAMETERS)
    descriptor = velocity.descriptor
    parallel[..., 0, 1] = parallel[..., 0, 2]
    stacked_parallel = np.stack([parallel, 2 * parallel])
    # The first trial peaks at samples 1 and 2; the other two peak at sample 0.
    # Different clocks make the two members report different mean peak times.
    clocks = np.array([[0.0, 0.009, 0.021], [0.1, 0.12, 0.18]])
    stacked = Dataset.from_arrays(
        {
            name: Variable.from_field(
                stacked_parallel if name == "parallel_velocity" else np.stack([velocity.data(name)] * 2),
                field,
                ("member", *field.dims),
            )
            for name, field in descriptor.variables.items()
        },
        coords={
            **{
                name: Variable.from_field(velocity.data(name), field, field.dims)
                for name, field in descriptor.coords.items()
                if name != "state_time"
            },
            "state_time": Variable(
                clocks, ("member", "state_time"), units="s", trial_time=TrialFigureTime(zero_epoch="go"),
            ),
            "member": Variable(np.asarray(["base", "double"]), ("member",)),
        },
        schema_id=descriptor.schema_id,
    )
    operation = load_application().bundle.operations.resolve("feedbax.summarize_peak_velocity")
    params = operation.value.parameter_model.model_validate({"ddof": 1, "bounds_scale": 1.5})
    summary = operation.defining_object(stacked, params)

    port = operation.value.definition.outputs["summary"]
    assert port.variables["mean"].dims == ("checkpoint", "condition")
    assert (port.variables["peak_time"].dims, port.variables["peak_time"].units) == (("checkpoint", "condition"), "s")
    assert port.variables["peak_time"].trial_time == TrialFigureTime()
    peaks = stacked_parallel.max(axis=-1)
    mean = peaks.mean(axis=-1)
    spread = 1.5 * peaks.std(axis=-1, ddof=1)
    assert set(summary.descriptor.coords) == {"member", "checkpoint", "condition"}
    for name in summary.descriptor.coords:
        np.testing.assert_array_equal(summary.data(name), stacked.data(name))
    for name, expected in (("mean", mean), ("lower", mean - spread), ("upper", mean + spread)):
        field = summary.descriptor.variables[name]
        assert (field.dims, field.units, field.frame) == (("member", "checkpoint", "condition"), "m/s", "reach-aligned")
        np.testing.assert_allclose(summary.data(name), expected)
    peak_time = summary.descriptor.variables["peak_time"]
    assert (peak_time.dims, peak_time.units) == (("member", "checkpoint", "condition"), "s")
    assert peak_time.trial_time == stacked.descriptor.coords["state_time"].trial_time
    expected_time = (2 * clocks[:, 0] + clocks[:, 1]) / 3
    np.testing.assert_allclose(summary.data("peak_time"), np.broadcast_to(expected_time[:, None, None], mean.shape))


def test_summary_refuses_invalid_parameters_and_sample_count():
    with pytest.raises(ValidationError, match="ddof"):
        VelocitySummaryParameters.model_validate({"ddof": -1})
    with pytest.raises(ValidationError, match="bounds_scale"):
        VelocitySummaryParameters.model_validate({"ddof": 1, "bounds_scale": float("nan")})
    states, _, _ = _states()
    velocity = aligned_velocity(states, _PARAMETERS)
    with pytest.raises(ValueError, match="Trial count must exceed ddof"):
        summarize_velocity(velocity, VelocitySummaryParameters(ddof=3))
