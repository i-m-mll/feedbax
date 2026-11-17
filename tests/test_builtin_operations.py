"""The two M3 built-ins operate on saved Evaluate dataset products."""

import numpy as np
import pytest
from pydantic import ValidationError

from feedbax.analysis.builtin_operations import (
    AlignedVelocityParameters,
    VelocitySummaryParameters,
    aligned_velocity,
    summarize_velocity,
)
from feedbax.dataset import Dataset, Variable
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_dataset, publish_dataset
from feedbax.plugins.application import OPERATIONS, new_application_registry_bundle
from feedbax.studio.canonical_evaluate import RAW_STATES_SCHEMA, state_key


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
    bundle = new_application_registry_bundle()
    bundle.seal()
    assert OPERATIONS.registered_keys(bundle.operations) == (
        "feedbax.aligned_velocity", "feedbax.summarize_velocity",
    )
    align, summarize = bundle.operations.definitions()
    assert align.inputs["states"].selected_variables["velocity"].units == "m/s"
    assert align.outputs["velocity"] == summarize.inputs["velocity"]
    assert summarize.outputs["summary"].variables["mean"].dims == ("checkpoint", "condition", "state_time")
    assert summarize.static_parameters == ("variable",)
    assert summarize.parameter_json_schema["properties"]["ddof"]["minimum"] == 0
    assert summarize.parameter_json_schema["required"] == ["ddof"]
    with pytest.raises(RuntimeError, match="sealed"):
        bundle.operations.register("demo.velocity", aligned_velocity, owner="demo", label="Velocity")
    run = bundle.operations.resolve(align.key)
    with (
        store.open_owned_object(states_ref.object_ref) as path,
        open_dataset(path, expected_descriptor_sha256=states_ref.descriptor_sha256) as saved,
    ):
        velocity = run.function(saved, run.parameter_model.model_validate(_PARAMETERS.model_dump()))
    np.testing.assert_allclose(velocity.data("parallel_velocity"), parallel, atol=1e-6)
    np.testing.assert_allclose(velocity.data("lateral_velocity"), lateral, atol=1e-6)
    run = bundle.operations.resolve(summarize.key)
    for ddof, scale in ((0, 0.5), (1, 2.0)):
        product = run.function(velocity, run.parameter_model.model_validate({"ddof": ddof, "bounds_scale": scale}))
        assert set(product.descriptor.variables) == set(summarize.outputs["summary"].variables)
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


def test_summary_refuses_invalid_parameters_and_sample_count():
    with pytest.raises(ValidationError, match="ddof"):
        VelocitySummaryParameters.model_validate({"ddof": -1})
    with pytest.raises(ValidationError, match="bounds_scale"):
        VelocitySummaryParameters.model_validate({"ddof": 1, "bounds_scale": float("nan")})
    states, _, _ = _states()
    velocity = aligned_velocity(states, _PARAMETERS)
    with pytest.raises(ValueError, match="Trial count must exceed ddof"):
        summarize_velocity(velocity, VelocitySummaryParameters(ddof=3))
