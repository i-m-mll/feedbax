"""Focused contract tests for raw-array and state PCA."""

from io import BytesIO
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import pytest
from jaxtyping import Array

from feedbax.analysis.dimred import pca
from feedbax.analysis.pca import (
    PCACentering,
    PCA_RESULT_SCHEMA_VERSION,
    PCAResults,
    PCAStateSelector,
    StatesPCA,
)
from feedbax.analysis.types import AnalysisInputData


class _NetworkState(eqx.Module):
    hidden: Array


class _FeedbackState(eqx.Module):
    net: _NetworkState


def _data(hidden: Array) -> AnalysisInputData:
    return AnalysisInputData(
        models={},
        tasks={},
        states=_FeedbackState(net=_NetworkState(hidden=hidden)),
        hps={},
        extras={},
    )


def _data_with_states(states: object) -> AnalysisInputData:
    return AnalysisInputData(models={}, tasks={}, states=states, hps={}, extras={})


@pytest.mark.parametrize(("shape", "rank"), [((2, 5), 2), ((5, 2), 2), ((2, 2, 7), 4)])
def test_raw_pca_retains_sample_axes_and_reconstructs_in_the_state_pca_basis(
    shape: tuple[int, ...], rank: int
) -> None:
    values = jnp.asarray(np.random.default_rng(42).normal(size=shape), dtype=jnp.float32)
    raw = jax.jit(pca)(values)
    fitted = StatesPCA().compute(_data(values))

    assert raw.components.shape == (*shape[:-1], rank)
    assert raw.singular_values.shape == (rank,)
    assert raw.components.dtype == values.dtype

    scores = raw.components.reshape(-1, rank)
    state_scores = fitted.states_pc.reshape(-1, rank)
    alignment = jnp.where(jnp.sum(scores * state_scores, axis=0) < 0, -1, 1)
    np.testing.assert_allclose(
        (raw.components * alignment) @ fitted.basis + fitted.mean,
        values,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("samples", "hermitian"),
    [
        pytest.param(
            np.array([[1e8], [1e8], [1e8 + 8]], dtype=np.float32),
            False,
            id="rounded-mean",
        ),
        pytest.param(
            np.array(
                [[3, -1 + 1j, -2 - 1j], [-1 - 1j, 2, -1 + 1j], [-2 + 1j, -1 - 1j, 3]],
                dtype=np.complex64,
            ),
            True,
            id="complex-hermitian",
        ),
    ],
)
def test_raw_pca_keeps_the_svd_score_orientation(samples: np.ndarray, hermitian: bool) -> None:
    values = jnp.asarray(samples)
    centered = values - values.mean(axis=0)
    left, singular_values, _ = jnp.linalg.svd(centered, full_matrices=False, hermitian=hermitian)

    raw = pca(values, hermitian=hermitian)

    np.testing.assert_allclose(raw.components, left * singular_values, rtol=1e-6)
    np.testing.assert_allclose(raw.singular_values, singular_values, rtol=1e-6)


def test_states_pca_matches_covariance_and_has_deterministic_component_signs() -> None:
    hidden = jnp.array(
        [
            [-3.0, -1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [1.0, 2.0, 0.0],
            [3.0, -1.0, -1.0],
        ]
    )

    first = StatesPCA(n_components=3).compute(_data(hidden))
    second = StatesPCA(n_components=3).compute(_data(hidden))

    covariance_eigenvalues = jnp.linalg.eigvalsh(jnp.cov(hidden, rowvar=False))[::-1]
    np.testing.assert_allclose(first.explained_variance, covariance_eigenvalues, rtol=1e-5)
    np.testing.assert_allclose(first.basis, second.basis)
    np.testing.assert_allclose(first.states_pc, second.states_pc)
    np.testing.assert_allclose(
        first.states_pc @ first.basis + first.mean,
        hidden,
        rtol=1e-5,
        atol=1e-5,
    )

    pivot_indices = jnp.argmax(jnp.abs(first.basis), axis=1)
    pivot_loadings = jnp.take_along_axis(first.basis, pivot_indices[:, None], axis=1)[:, 0]
    assert bool(jnp.all(pivot_loadings >= 0))


def test_states_pca_records_preprocessing_and_applies_it_to_new_batches() -> None:
    hidden = jnp.array(
        [
            [1.0, 10.0],
            [2.0, 13.0],
            [4.0, 14.0],
            [8.0, 20.0],
        ]
    )
    result = StatesPCA(
        n_components=2,
        normalization="standard_deviation",
    ).compute(_data(hidden))

    expected_mean = jnp.mean(hidden, axis=0)
    expected_scale = jnp.sqrt(jnp.mean(jnp.square(hidden - expected_mean), axis=0))
    np.testing.assert_allclose(result.mean, expected_mean)
    np.testing.assert_allclose(result.scale, expected_scale)
    np.testing.assert_allclose(
        result.states_pc,
        ((hidden - expected_mean) / expected_scale) @ result.basis.T,
    )

    batches = {
        "first": hidden[:2],
        "second": hidden[2:],
    }
    transformed = result.batch_transform(batches)
    assert jt.structure(transformed) == jt.structure(batches)
    np.testing.assert_allclose(transformed["first"], result.transform(hidden[:2]))


def test_states_pca_supports_explicit_no_centering() -> None:
    hidden = jnp.array([[1.0, 2.0], [2.0, 5.0], [4.0, 3.0]])
    result = StatesPCA(
        n_components=1,
        centering="none",
        normalization="standard_deviation",
    ).compute(_data(hidden))

    np.testing.assert_array_equal(result.mean, jnp.zeros(2))
    np.testing.assert_allclose(result.scale, jnp.std(hidden, axis=0))
    np.testing.assert_allclose(result.states_pc, (hidden / result.scale) @ result.basis.T)


@pytest.mark.parametrize(
    ("centering", "basis", "variance", "ratio"),
    [
        ("none", [[1, 0, 0, 0, 0]], 4 / 3, 1 / 5),
        ("mean", [[0, 1, 0, 0, 0]], 16 / 3, 4 / 5),
    ],
)
def test_truncated_pca_reports_sample_variance_relative_to_the_full_spectrum(
    centering: PCACentering, basis: list[list[int]], variance: float, ratio: float
) -> None:
    hidden = jnp.array(
        [
            [999_999.0, -2.0, 0.0, 0.0, 0.0],
            [1_000_001.0, -2.0, 0.0, 0.0, 0.0],
            [999_999.0, 2.0, 0.0, 0.0, 0.0],
            [1_000_001.0, 2.0, 0.0, 0.0, 0.0],
        ]
    )
    result = StatesPCA(n_components=1, centering=centering, return_data=False).compute(
        _data(hidden)
    )

    np.testing.assert_allclose(result.basis, basis, atol=1e-6)
    np.testing.assert_allclose(result.explained_variance, [variance], rtol=1e-6)
    np.testing.assert_allclose(result.explained_variance_ratio, [ratio], rtol=1e-6)
    assert result.states_pc is None


def test_centered_variance_accounts_for_a_mean_that_rounds_at_large_offset() -> None:
    hidden = jnp.array([[1e8], [1e8], [1e8 + 8]], dtype=jnp.float32)
    result = StatesPCA().compute(_data(hidden))

    np.testing.assert_allclose(result.explained_variance, [64 / 3], rtol=1e-6)
    np.testing.assert_array_equal(result.states_pc, [[0.0], [0.0], [8.0]])


def test_complex_variance_uses_the_public_score_projection() -> None:
    hidden = jnp.array([[1, 2j, 1 + 2j, 3], [-1, -2j, -1 - 2j, -3]])
    full = StatesPCA().compute(_data(hidden))
    truncated = StatesPCA(n_components=1, return_data=False).compute(_data(hidden))

    expected_variance = np.var(full.states_pc, axis=0, ddof=1)
    np.testing.assert_allclose(full.explained_variance, expected_variance, rtol=1e-6)
    np.testing.assert_allclose(truncated.explained_variance, expected_variance[:1], rtol=1e-6)
    np.testing.assert_allclose(
        truncated.explained_variance_ratio,
        expected_variance[:1] / expected_variance.sum(),
        rtol=1e-6,
    )


def test_states_pca_fit_is_jit_trace_safe() -> None:
    hidden = jnp.array(
        [
            [-2.0, 0.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [2.0, 0.0],
        ]
    )
    analysis = StatesPCA(n_components=2, normalization="standard_deviation")

    def fit(values: Array) -> tuple[Array, Array]:
        result = analysis.compute(_data(values))
        return result.basis, result.explained_variance

    basis, explained_variance = jax.jit(fit)(hidden)
    eager = analysis.compute(_data(hidden))

    np.testing.assert_allclose(basis, eager.basis)
    np.testing.assert_allclose(explained_variance, eager.explained_variance)


def test_authored_state_selector_is_typed_serializable_and_preserves_public_key() -> None:
    authored = StatesPCA(where_states="states.net.hidden")

    assert isinstance(authored.where_states, PCAStateSelector)
    assert authored.where_states == "net.hidden"
    assert json.loads(json.dumps(authored.where_states)) == "net.hidden"
    with pytest.raises(TypeError, match="string path"):
        StatesPCA(where_states=lambda state: state.net.hidden)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="did not resolve"):
        StatesPCA(where_states="states.missing").compute(_data(jnp.ones((3, 2))))


def test_sampling_contract_is_explicit_and_validated() -> None:
    hidden = jnp.arange(24.0).reshape(2, 3, 4)
    flattened = StatesPCA(n_components=2, sampling="flatten_leading").compute(_data(hidden))

    assert flattened.sample_count == 6
    assert flattened.feature_count == 4
    assert flattened.states_pc is not None
    assert flattened.states_pc.shape == (2, 3, 2)

    with pytest.raises(ValueError, match="requires selected state shape"):
        StatesPCA(sampling="rows").compute(_data(hidden))
    with pytest.raises(ValueError, match="one of"):
        StatesPCA(sampling="columns")  # type: ignore[arg-type]


def test_explicit_selector_supports_authored_aggregation_without_mixing_state_leaves() -> None:
    states = {
        "first": _FeedbackState(net=_NetworkState(hidden=jnp.ones((2, 3)))),
        "second": _FeedbackState(net=_NetworkState(hidden=2 * jnp.ones((4, 3)))),
    }

    result = StatesPCA(
        n_components=1,
        aggregate_over_labels="all",
    ).compute(_data_with_states(states))

    assert result.sample_count == 6
    assert result.states_pc is not None
    assert result.states_pc["first"].shape == (2, 1)
    assert result.states_pc["second"].shape == (4, 1)

    incompatible = dict(states)
    incompatible["second"] = _FeedbackState(net=_NetworkState(hidden=jnp.ones((4, 2))))
    with pytest.raises(ValueError, match="inconsistent feature counts"):
        StatesPCA(aggregate_over_labels="all").compute(_data_with_states(incompatible))


def test_pca_result_round_trips_with_equinox_leaf_serialization() -> None:
    hidden = jnp.array([[1.0, 4.0], [2.0, 3.0], [4.0, 1.0]])
    result = StatesPCA(n_components=2).compute(_data(hidden))
    stream = BytesIO()

    eqx.tree_serialise_leaves(stream, result)
    stream.seek(0)
    restored = eqx.tree_deserialise_leaves(stream, result)

    assert isinstance(restored, PCAResults)
    assert restored.schema_version == PCA_RESULT_SCHEMA_VERSION
    assert restored.state_selector == PCAStateSelector("net.hidden")
    assert restored.sampling == "flatten_leading"
    assert restored.centering == "mean"
    assert restored.normalization == "none"
    for field in (
        "basis",
        "explained_variance",
        "explained_variance_ratio",
        "mean",
        "scale",
        "states_pc",
    ):
        np.testing.assert_array_equal(getattr(restored, field), getattr(result, field))
    np.testing.assert_allclose(restored.batch_transform(hidden), result.states_pc)


@pytest.mark.parametrize(
    ("analysis", "message"),
    [
        (StatesPCA(n_components=3), "exceeds the fitted rank bound"),
        (
            StatesPCA(normalization="standard_deviation"),
            "undefined for constant features",
        ),
    ],
)
def test_states_pca_rejects_invalid_fit_contracts(
    analysis: StatesPCA,
    message: str,
) -> None:
    hidden = jnp.array([[1.0, 2.0], [1.0, 3.0]])

    with pytest.raises(ValueError, match=message):
        analysis.compute(_data(hidden))


@pytest.mark.parametrize("invalid", [jnp.nan, jnp.inf, -jnp.inf])
def test_states_pca_rejects_concrete_nonfinite_data(invalid: float) -> None:
    with pytest.raises(ValueError, match="only finite values"):
        StatesPCA().compute(_data(jnp.array([[1.0, 2.0], [3.0, invalid]])))


@pytest.mark.parametrize("centering", ["mean", "none"])
def test_states_pca_rejects_constant_data(centering: PCACentering) -> None:
    with pytest.raises(ValueError, match="positive finite total variance"):
        StatesPCA(centering=centering).compute(_data(jnp.full((7, 2), 0.1)))


def test_result_retains_public_batch_transform_and_optional_states_pc_surface() -> None:
    hidden = jnp.array([[1.0, 3.0], [2.0, 1.0], [4.0, 2.0]])
    result = StatesPCA(n_components=1, return_data=False).compute(_data(hidden))

    assert isinstance(result, PCAResults)
    assert result.states_pc is None
    assert result.batch_transform(hidden).shape == (3, 1)
    with pytest.raises(ValueError, match="features; expected"):
        result.batch_transform(jnp.ones((2, 3)))
