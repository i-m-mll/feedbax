from functools import partial

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from feedbax._vendor.jax_cookbook import LDict
from feedbax.analysis.aligned import DEFAULT_VARSET, VAR_LEVEL_LABEL, AlignedVars, Measure
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.execution import run_analyses_with_context
from feedbax.analysis.func import ApplyFns
from feedbax.analysis.types import AnalysisInputData, Direction, ResponseVar
from feedbax.config.namespace import TreeNamespace
from feedbax.contracts.manifest import AnalysisRunSpec


def _velocity(values):
    return LDict.of(VAR_LEVEL_LABEL)({ResponseVar.VELOCITY: jnp.asarray(values)})


@pytest.mark.parametrize(
    ("reducer", "expected"),
    [(jnp.mean, [[3.0, 10.0]]), (jnp.max, [[5.0, 20.0]])],
)
def test_measure_reduces_time_and_preserves_velocity_components(reducer, expected):
    response = _velocity([[[1.0, 20.0], [3.0, 4.0], [5.0, 6.0]]])

    result = Measure(ResponseVar.VELOCITY, agg_fn=reducer)(response)

    np.testing.assert_allclose(result, expected)


def test_measure_slices_and_reduces_scalar_series():
    measure = Measure(
        ResponseVar.VELOCITY,
        time_axis=-1,
        timesteps=lambda *, start: slice(start, None),
        agg_fn=jnp.mean,
        normalizer=lambda *, divisor: divisor,
    )
    response = _velocity([[50.0, 4.0, 8.0], [90.0, 2.0, 6.0]])

    np.testing.assert_allclose(measure(response, start=1, divisor=2), [3.0, 2.0])
    np.testing.assert_allclose(
        Measure(ResponseVar.VELOCITY, timesteps=slice(1, None), agg_fn=jnp.mean)(
            _velocity([50.0, 4.0, 8.0])
        ),
        6.0,
    )


@pytest.mark.parametrize(
    ("transform", "transformed_time_axis", "expected"),
    [
        (partial(jnp.linalg.norm, axis=-1), None, [[5.0, 15.0]]),
        (partial(jnp.diff, axis=-2), None, [[[6.0, 8.0], [6.0, 8.0]]]),
        (partial(jnp.moveaxis, source=-2, destination=0), 0, [[[3.0, 4.0], [9.0, 12.0]]]),
    ],
)
def test_measure_reduces_time_after_whole_array_transform(
    transform, transformed_time_axis, expected
):
    response = _velocity([[[[0.0, 0.0], [6.0, 8.0]], [[6.0, 8.0], [12.0, 16.0]]]])
    measure = Measure(
        ResponseVar.VELOCITY,
        transform_fn=transform,
        transformed_time_axis=transformed_time_axis,
        agg_fn=jnp.mean,
    )

    np.testing.assert_allclose(measure(response), expected)


class _Task(eqx.Module):
    directions: Array


class _States(eqx.Module):
    mechanics: TreeNamespace


def test_apply_fns_measures_aligned_velocity(tmp_path):
    velocity = jnp.asarray([[[1.0, 20.0], [3.0, 4.0], [5.0, 6.0]]])
    data = AnalysisInputData(
        models={},
        tasks=_Task(jnp.asarray([[0.0, 1.0]])),
        states=_States(TreeNamespace(effector=TreeNamespace(vel=velocity))),
        hps=TreeNamespace(),
        extras={},
    )
    aligned = AlignedVars(
        varset=LDict.of(VAR_LEVEL_LABEL)(
            {ResponseVar.VELOCITY: DEFAULT_VARSET[ResponseVar.VELOCITY]}
        ),
        directions_fn=lambda task, _: task.directions,
    )
    analysis = ApplyFns(
        inputs=ApplyFns.Ports(input=aligned),
        fns={
            "mean": Measure(ResponseVar.VELOCITY, agg_fn=jnp.mean),
            "first_selected_parallel": Measure(
                ResponseVar.VELOCITY,
                timesteps=slice(1, 3),
                direction=Direction.PARALLEL,
                agg_fn=lambda values, *, axis: jnp.take(values, 0, axis=axis),
            ),
        },
        is_leaf=lambda leaf: isinstance(leaf, LDict) and leaf.label == VAR_LEVEL_LABEL,
    )
    context = AnalysisRunContext(spec=AnalysisRunSpec(analysis_type="measure"), root=tmp_path)

    _, results, _ = run_analyses_with_context({"measure": analysis}, data, context)

    np.testing.assert_allclose(results["measure"]["mean"], [[10.0, -3.0]])
    np.testing.assert_allclose(results["measure"]["first_selected_parallel"], [4.0])
