import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import feedbax.analysis.fps as fps_module
from feedbax.analysis.fp_finder import (
    FixedPointFinder,
    exclude_points,
    fp_adam_optimizer,
)
from feedbax.analysis.fps import FixedPoints, FixedPointsPorts


def test_fixed_points_surface_keeps_generic_analysis_entrypoint() -> None:
    analysis = FixedPoints(inputs=FixedPointsPorts(fns="fns", candidates="candidates"))

    assert FixedPoints.Ports is FixedPointsPorts
    assert isinstance(analysis.inputs, FixedPointsPorts)
    assert analysis.inputs.fns == "fns"
    assert analysis.inputs.candidates == "candidates"
    assert callable(analysis.fpf_fn)


def test_fixed_points_reuses_trace_for_new_array_argument_values() -> None:
    analysis = FixedPoints(inputs=FixedPointsPorts(fns="fns", candidates="candidates"))
    candidates = jnp.array([[0.0], [2.0]])
    traces = 0

    def dynamics(params, state):
        nonlocal traces
        traces += 1
        return params["gain"] * state + params["offset"]

    def solve(offset):
        return analysis.compute(
            {},
            fns=dynamics,
            candidates=candidates,
            fn_args=({"gain": float("0.5"), "offset": jnp.array(offset)},),
        )

    first = solve(0.0)
    first.fps.block_until_ready()
    initial_traces = traces
    assert initial_traces > 0
    second = solve(0.5)
    second.fps.block_until_ready()

    assert traces == initial_traces
    assert first.fps[:, 0].tolist() == pytest.approx([0.0, 0.0], abs=1e-3)
    assert second.fps[:, 0].tolist() == pytest.approx([1.0, 1.0], abs=1e-3)


def test_fixed_points_can_run_eagerly_after_a_jitted_solve() -> None:
    analysis = FixedPoints(
        inputs=FixedPointsPorts(fns="fns", candidates="candidates"),
        fp_optimizer=lambda: optax.sgd(jnp.array(0.2)),
    )
    candidates = jnp.array([[0.0], [2.0]])

    def dynamics(target, state):
        return 0.5 * (state + target)

    def solve(target):
        return analysis.compute({}, fns=dynamics, candidates=candidates, fn_args=(target,)).fps

    target = jnp.array(1.0)
    traced = jax.jit(solve)(target).block_until_ready()
    eager = solve(target)

    assert eager[:, 0].tolist() == pytest.approx([1.0, 1.0], abs=1e-3)
    assert jnp.array_equal(traced, eager)


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


def test_fixed_point_finder_isolates_invalid_residuals_and_gradients() -> None:
    finder = FixedPointFinder(optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(0.2)))

    def dynamics(state):
        return 0.5 * jnp.sqrt(state) + 0.5

    options = dict(loss_tol=jnp.array(1e-7), n_batches=128, n_batches_per_iter=4, key=jr.key(0))
    alone = finder.find_and_filter(dynamics, jnp.array([[2.0]]), **options)
    mixed = finder.find_and_filter(dynamics, jnp.array([[-1.0], [0.0], [2.0]]), **options)

    # At -1 the residual is NaN. At 0 the residual is finite but its gradient is not.
    assert mixed.masks["meets_all_criteria"].tolist() == [True, False, False]
    assert float(mixed.fps[0, 0]) == pytest.approx(1.0, abs=5e-4)
    assert jnp.array_equal(mixed.fps[:1], alone.fps)

    n_updates, _, losses = finder(dynamics, jnp.array([[-1.0], [0.0]]), **options)
    assert int(n_updates) == 0
    assert not jnp.any(jnp.isfinite(losses))


def test_fixed_point_finder_isolates_candidates_that_leave_the_domain() -> None:
    finder = FixedPointFinder(optax.sgd(learning_rate=3.0))
    # The first step sends 1 outside the square root's domain and keeps 0.5 inside.
    # The exact root at 0 has a singular gradient and should remain accepted.
    results = finder.find_and_filter(
        lambda state: 0.5 * jnp.sqrt(state),
        jnp.array([[0.0], [0.5], [1.0]]),
        loss_tol=jnp.array(1e-7),
        n_batches=128,
        n_batches_per_iter=4,
        key=jr.key(0),
    )

    assert results.masks["meets_all_criteria"].tolist() == [True, True, False]
    assert results.fps[:2, 0].tolist() == pytest.approx([0.0, 0.25], abs=7e-4)


@pytest.mark.parametrize(
    ("budget", "tolerance", "expected_updates"),
    [
        pytest.param(3, 0.0, 3, id="partial-final-chunk"),
        pytest.param(6, 0.2, 2, id="current-chunk-loss-meets-tolerance"),
        pytest.param(6, 0.7, 2, id="check-only-between-chunks"),
        pytest.param(6, 2.5, 0, id="initial-loss-meets-tolerance"),
    ],
)
def test_fixed_point_finder_update_boundaries(budget, tolerance, expected_updates) -> None:
    finder = FixedPointFinder(optax.sgd(learning_rate=0.5))
    n_updates, points, losses = finder(
        jnp.zeros_like,
        jnp.array([[2.0], [1.0]]),
        jnp.array(tolerance),
        n_batches=budget,
        n_batches_per_iter=2,
        key=jr.PRNGKey(0),
    )

    # With two scalar candidates the mean squared loss gives a gradient equal
    # to each candidate's value, so an SGD step halves both values.
    expected_points = jnp.array([[1.0], [2.0]]) * 0.5**expected_updates
    assert int(n_updates) == expected_updates
    assert jnp.array_equal(points, expected_points)
    assert jnp.array_equal(losses, expected_points[:, 0] ** 2)


@pytest.mark.parametrize("middle_loss", [1.0, 0.25])
def test_duplicate_chain_keeps_both_separated_roots(middle_loss: float) -> None:
    masks, _ = exclude_points(
        points=jnp.array([[0.0], [0.9], [1.8]]),
        values=jnp.array([0.0, middle_loss, 0.0]),
        value_tol=jnp.array(0.5),
        unique_tol=1.0,
    )

    assert masks["meets_all_criteria"].tolist() == [True, False, True]


def test_outlier_needs_an_eligible_neighbor_and_best_duplicate_survives() -> None:
    masks, counts = exclude_points(
        points=jnp.array([[0.0], [0.125], [1.75], [1.875]]),
        values=jnp.array([0.25, 0.0, 0.0, 1.0]),
        value_tol=jnp.array(0.5),
        outlier_tol=0.125,
        unique_tol=0.125,
    )

    assert masks["not_outlier"].tolist() == [True, True, False, False]
    assert masks["meets_all_criteria"].tolist() == [False, True, False, False]
    assert counts == {
        "meets_tolerance": 3,
        "not_outlier": 2,
        "unique": 1,
        "meets_all_criteria": 1,
    }
