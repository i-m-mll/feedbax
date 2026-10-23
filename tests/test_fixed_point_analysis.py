import jax.numpy as jnp
import jax.random as jr

import feedbax.analysis.fps as fps_module
from feedbax.analysis.fp_finder import (
    FixedPointFinder,
    exclude_points,
    fp_adam_optimizer,
    get_fp_loss_fn,
)
from feedbax.analysis.fps import FixedPoints, FixedPointsPorts


def test_fixed_points_surface_keeps_generic_analysis_entrypoint() -> None:
    analysis = FixedPoints(inputs=FixedPointsPorts(fns="fns", candidates="candidates"))

    assert FixedPoints.Ports is FixedPointsPorts
    assert isinstance(analysis.inputs, FixedPointsPorts)
    assert analysis.inputs.fns == "fns"
    assert analysis.inputs.candidates == "candidates"
    assert callable(analysis.fpf_fn)


def test_fixed_points_surface_does_not_export_simple_reach_helpers() -> None:
    removed_names = {
        "get_endpoint_positions",
        "get_initial_network_inputs",
        "get_simple_reach_endpoint_fps",
        "get_fp_trajs",
        "get_simple_reach_fps",
        "get_simple_reach_first_fps",
    }

    assert removed_names.isdisjoint(vars(fps_module))


def test_fixed_point_finder_converges_on_linear_fixed_point() -> None:
    finder = FixedPointFinder(fp_adam_optimizer(learning_rate=0.05))
    candidates = jnp.array([[2.0], [-3.0], [0.25]], dtype=jnp.float32)

    _n_batches, fixed_points, losses = finder(
        lambda state: 0.5 * state,
        candidates,
        jnp.array(1e-4, dtype=jnp.float32),
        n_batches=400,
        n_batches_per_iter=20,
        key=jr.PRNGKey(0),
    )

    assert float(losses[0]) < 1e-4
    assert float(jnp.abs(fixed_points[0, 0])) < 0.05


def test_fixed_point_loss_and_filters_are_available_as_low_level_utilities() -> None:
    loss_fn = get_fp_loss_fn(lambda state: 0.5 * state)
    losses = loss_fn(jnp.array([[2.0], [4.0]], dtype=jnp.float32))

    masks, counts = exclude_points(
        points=jnp.array([[0.0], [0.01], [5.0]], dtype=jnp.float32),
        values=jnp.array([1e-8, 1e-8, 1.0], dtype=jnp.float32),
        value_tol=jnp.array(1e-6, dtype=jnp.float32),
        outlier_tol=0.1,
        unique_tol=0.02,
    )

    assert jnp.allclose(losses, jnp.array([1.0, 4.0], dtype=jnp.float32))
    assert counts["meets_tolerance"] == 2
    assert counts["meets_all_criteria"] == 1
