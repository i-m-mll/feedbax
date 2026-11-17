"""Numerical contracts for fitting and projecting sample matrices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.pca import fit_pca_samples


def test_fit_matches_sample_covariance_and_reconstructs_the_samples() -> None:
    samples = jnp.array(
        [[-3.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [1.0, 2.0, 0.0], [3.0, -1.0, -1.0]]
    )

    fitted = jax.jit(
        lambda values: fit_pca_samples(
            values, n_components=3, centering="mean", normalization="none"
        )
    )(samples)

    covariance = np.cov(np.asarray(samples), rowvar=False)
    np.testing.assert_allclose(
        fitted.explained_variance, np.linalg.eigvalsh(covariance)[::-1], rtol=1e-5
    )
    np.testing.assert_allclose(
        fitted.transform(samples) @ fitted.basis + fitted.mean, samples, rtol=1e-5, atol=1e-5
    )
    basis = np.asarray(fitted.basis)
    assert np.all(basis[np.arange(3), np.abs(basis).argmax(axis=1)] >= 0)


@pytest.mark.parametrize(
    ("centering", "basis", "variance", "ratio"),
    [
        ("none", [[1, 0, 0, 0, 0]], 4 / 3, 1 / 5),
        ("mean", [[0, 1, 0, 0, 0]], 16 / 3, 4 / 5),
    ],
)
def test_truncated_fit_reports_variance_relative_to_the_full_sample_spectrum(
    centering, basis, variance, ratio
) -> None:
    samples = jnp.array(
        [
            [999_999.0, -2.0, 0.0, 0.0, 0.0],
            [1_000_001.0, -2.0, 0.0, 0.0, 0.0],
            [999_999.0, 2.0, 0.0, 0.0, 0.0],
            [1_000_001.0, 2.0, 0.0, 0.0, 0.0],
        ]
    )

    fitted = fit_pca_samples(
        samples, n_components=1, centering=centering, normalization="none"
    )

    np.testing.assert_allclose(fitted.basis, basis, atol=1e-6)
    np.testing.assert_allclose(fitted.explained_variance, [variance], rtol=1e-6)
    np.testing.assert_allclose(fitted.explained_variance_ratio, [ratio], rtol=1e-6)


def test_normalized_fit_projects_a_held_out_sample() -> None:
    samples = jnp.array([[1.0, 10.0], [3.0, 14.0], [5.0, 10.0], [7.0, 14.0]])

    fitted = fit_pca_samples(
        samples, n_components=2, centering="mean", normalization="standard_deviation"
    )

    np.testing.assert_allclose(fitted.mean, [4.0, 12.0])
    np.testing.assert_allclose(fitted.scale, [np.sqrt(5), 2.0])
    np.testing.assert_allclose(fitted.transform(jnp.array([4.0 + np.sqrt(5), 14.0])), [np.sqrt(2), 0], atol=1e-6)


def test_fit_accounts_for_a_sample_mean_that_rounds_at_large_offset() -> None:
    samples = jnp.array([[1e8], [1e8], [1e8 + 8]], dtype=jnp.float32)

    fitted = fit_pca_samples(
        samples, n_components=None, centering="mean", normalization="none"
    )

    np.testing.assert_allclose(fitted.explained_variance, [64 / 3], rtol=1e-6)
    np.testing.assert_array_equal(fitted.transform(samples), [[0.0], [0.0], [8.0]])


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_fit_refuses_nonfinite_samples(value) -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_pca_samples(
            jnp.array([[0.0], [value]]),
            n_components=None,
            centering="mean",
            normalization="none",
        )
