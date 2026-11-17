"""Numerical PCA on explicit sample and feature arrays."""

from math import sqrt
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Array

PCACentering = Literal["mean", "none"]
PCANormalization = Literal["none", "standard_deviation"]


def _validate_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        choices_text = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {choices_text}; got {value!r}")


@jax.jit
def _all_finite(samples: Array) -> Array:
    if jnp.iscomplexobj(samples):
        return jnp.all(jnp.isfinite(samples))
    # Real extrema propagate NaNs and detect infinities without allocating a mask.
    return jnp.isfinite(jnp.min(samples)) & jnp.isfinite(jnp.max(samples))


def _subtract_mean(samples: Array) -> tuple[Array, Array]:
    mean = jnp.mean(samples, axis=0)
    return samples - mean, mean


@jax.jit
def _center_and_scale(samples: Array, scale: Array | None) -> tuple[Array, Array]:
    centered, mean = _subtract_mean(samples)
    # Remove the residual left when the feature mean rounds at a large offset.
    centered = centered - jnp.mean(centered, axis=0)
    return (centered if scale is None else centered / scale), mean


@jax.jit
def _projected_variance(samples: Array, basis: Array) -> Array:
    # Reduce each score column before projecting onto the next axis.
    return jax.lax.map(lambda axis: jnp.var(samples @ axis, ddof=1), basis)


@jax.jit
def _uncentered_variance(samples: Array, scale: Array | None, basis: Array) -> tuple[Array, Array]:
    centered, _ = _center_and_scale(samples, scale)
    # The complete reduced real basis spans the row space, including after centering.
    total_variance = jnp.sum(jnp.var(centered, axis=0, ddof=1))
    return _projected_variance(centered, basis), total_variance


def _fit_pca(
    samples: Array,
    *,
    centering: PCACentering = "mean",
    scale: Array | None = None,
    recenter: bool = False,
    svd_options: dict[str, Any] | None = None,
) -> tuple[Array, Array, tuple[Array, Array, Array]]:
    """Preprocess sample rows and fit a reduced SVD."""
    if recenter and centering == "mean":
        preprocessed, mean = _center_and_scale(samples, scale)
    else:
        if centering == "mean":
            preprocessed, mean = _subtract_mean(samples)
        else:
            mean = jnp.zeros((samples.shape[1],), dtype=samples.dtype)
            preprocessed = samples if jnp.issubdtype(samples.dtype, jnp.inexact) else samples - mean
        if scale is not None:
            preprocessed = preprocessed / scale

    left, singular_values, basis = jnp.linalg.svd(
        preprocessed, full_matrices=False, **(svd_options or {})
    )
    return preprocessed, mean, (left, singular_values, basis)


def _transform_pca(array: Array, basis: Array, mean: Array, scale: Array) -> Array:
    if array.ndim < 1:
        raise ValueError("PCA transform input must have a feature axis")
    if array.shape[-1] != basis.shape[-1]:
        raise ValueError(
            f"PCA transform input has {array.shape[-1]} features; expected {basis.shape[-1]}"
        )
    return ((array - mean) / scale) @ basis.T


class PCAFit(NamedTuple):
    """Numeric fit of a PCA basis on an explicit sample matrix."""

    basis: Array
    explained_variance: Array
    explained_variance_ratio: Array
    mean: Array
    scale: Array
    sample_count: int
    feature_count: int

    def transform(self, array: Array) -> Array:
        """Project an array whose last axis is the fitted feature axis."""
        return _transform_pca(array, self.basis, self.mean, self.scale)


def fit_pca_samples(
    samples: Array,
    *,
    n_components: int | None,
    centering: PCACentering,
    normalization: PCANormalization,
) -> PCAFit:
    """Fit the signed PCA basis of a (sample, feature) array."""
    if samples.ndim != 2:
        raise ValueError("PCA samples must have shape (samples, features)")
    if n_components is not None and n_components < 1:
        raise ValueError("n_components must be positive or None")
    _validate_choice("centering", centering, ("mean", "none"))
    _validate_choice("normalization", normalization, ("none", "standard_deviation"))
    if samples.shape[0] < 2:
        raise ValueError("PCA requires at least two samples")
    if samples.shape[1] < 1:
        raise ValueError("PCA requires at least one feature")
    if not isinstance(samples, Tracer) and not bool(_all_finite(samples)):
        raise ValueError("Selected PCA samples must contain only finite values")
    n_samples, n_features = samples.shape
    max_components = min(n_samples, n_features)
    n_components = n_components or max_components
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds the fitted rank bound "
            f"min(samples, features)={max_components}"
        )

    scale = None
    if normalization == "standard_deviation":
        scale = jnp.std(samples, axis=0)
        if not isinstance(scale, Tracer) and bool(jnp.any(scale == 0)):
            raise ValueError(
                "normalization='standard_deviation' is undefined for constant features"
            )
    complex_samples = jnp.iscomplexobj(samples)
    preprocessed, mean, (_, singular_values, basis) = _fit_pca(
        samples, centering=centering, scale=scale, recenter=not complex_samples
    )
    if complex_samples:
        # The plain transpose used by transform makes complex score
        # variance differ from the singular spectrum.
        spectrum = _projected_variance(preprocessed, basis)
    elif centering == "mean":
        spectrum = jnp.square(singular_values / sqrt(n_samples - 1))
    else:
        spectrum = None
    basis = basis[:n_components]
    if spectrum is None:
        explained_variance, total_variance = _uncentered_variance(samples, scale, basis)
    else:
        explained_variance = spectrum[:n_components]
        total_variance = jnp.sum(spectrum)

    if not isinstance(total_variance, Tracer) and not bool(
        jnp.isfinite(total_variance) & (total_variance > 0)
    ):
        raise ValueError("PCA requires positive finite total variance")

    pivot_indices = jnp.argmax(jnp.abs(basis), axis=1)
    pivot_loadings = jnp.take_along_axis(basis, pivot_indices[:, None], axis=1)[:, 0]
    signs = jnp.where(pivot_loadings < 0, -1, 1)
    basis = basis * signs[:, None]

    return PCAFit(
        basis=basis,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance / total_variance,
        mean=mean,
        scale=scale if scale is not None else jnp.ones((n_features,), dtype=samples.dtype),
        sample_count=n_samples,
        feature_count=n_features,
    )
