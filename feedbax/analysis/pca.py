"""Principal-component analysis of evaluated model states."""

from collections.abc import Sequence
from typing import Any, Literal, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from jax.core import Tracer
from jax_cookbook import LDict, is_module, is_type
from jaxtyping import Array, PyTree
import numpy as np

from feedbax.analysis.analysis import AbstractAnalysis, NoPorts
from feedbax.analysis.types import AnalysisInputData
from feedbax.runtime.where_selectors import attr_str_tree_to_where_func
from feedbax.config.tree import rearrange_ldict_levels


PCA_RESULT_SCHEMA_VERSION = 1
PCASampling = Literal["flatten_leading", "rows"]
PCACentering = Literal["mean", "none"]
PCANormalization = Literal["none", "standard_deviation"]


class PCAStateSelector(str):
    """Serializable authored path selecting one array from an evaluated state.

    Paths use Feedbax's compact attribute syntax, for example ``"net.hidden"``.
    An authored ``"states."`` prefix is normalized away because
    :class:`StatesPCA` already receives the evaluated state subtree. Selecting
    exactly one state array makes the PCA feature space explicit and prevents
    unrelated heterogeneous state leaves from being concatenated.
    """

    def __new__(cls, path: str) -> "PCAStateSelector":
        if not isinstance(path, str):
            raise TypeError("PCA state selector must be a string path")
        if not path or path.strip() != path:
            raise ValueError("PCA state selector must be a nonempty, trimmed path")
        if path.startswith("states."):
            path = path.removeprefix("states.")
        if not path:
            raise ValueError("PCA state selector must name a path below states")
        return super().__new__(cls, path)

    def select(self, state: Any) -> Array:
        """Select the authored array from ``state``."""
        try:
            value = attr_str_tree_to_where_func(str(self))(state)
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"PCA state selector {self!r} did not resolve") from exc
        if not eqx.is_array(value):
            raise TypeError(
                f"PCA state selector {self!r} resolved to {type(value).__name__}, not a JAX array"
            )
        return value


class PCAResults(eqx.Module):
    """Fitted, serializable PCA state and optional transformed input data.

    Attributes:
        basis: Principal axes with shape ``(components, features)``.
        explained_variance: Variance explained by each retained component.
        explained_variance_ratio: Fraction of total variance explained by each
            retained component.
        mean: Per-feature mean subtracted during preprocessing. This is zero
            when ``centering="none"``.
        scale: Per-feature divisor used during preprocessing. This is one when
            ``normalization="none"``.
        states_pc: Selected input state transformed into principal-component
            coordinates, or ``None`` when ``return_data=False``.
    """

    basis: Array
    explained_variance: Array
    explained_variance_ratio: Array
    mean: Array
    scale: Array
    states_pc: Optional[PyTree[Array]]
    state_selector: PCAStateSelector
    sampling: PCASampling
    centering: PCACentering
    normalization: PCANormalization
    sample_count: int
    feature_count: int
    schema_version: int = PCA_RESULT_SCHEMA_VERSION

    def transform(self, array: Array) -> Array:
        """Project an array whose last axis is the fitted feature axis."""
        if array.ndim < 1:
            raise ValueError("PCA transform input must have a feature axis")
        if array.shape[-1] != self.feature_count:
            raise ValueError(
                f"PCA transform input has {array.shape[-1]} features; expected {self.feature_count}"
            )
        return ((array - self.mean) / self.scale) @ self.basis.T

    def batch_transform(self, tree: PyTree[Array]) -> PyTree[Array]:
        """Transform array leaves, retaining every leading batch axis."""
        return jt.map(self.transform, tree)


def _validate_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        choices_text = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {choices_text}; got {value!r}")


def _as_concrete_numpy(array: Array) -> Optional[np.ndarray]:
    """Return a NumPy view for eager validation, deferring dynamic tracers."""
    if isinstance(array, Tracer):
        return None
    return np.asarray(array)


class StatesPCA(AbstractAnalysis[NoPorts]):
    """Fit report-grade PCA to one explicitly selected evaluated state array.

    The selected array's final axis is always the feature axis. With
    ``sampling="flatten_leading"``, every leading-axis coordinate is one sample.
    With ``sampling="rows"``, the selected array must already have shape
    ``(samples, features)``.

    ``centering="mean"`` subtracts the per-feature sample mean.
    ``normalization="standard_deviation"`` divides by each feature's population
    standard deviation, independently of the centering choice; constant features
    are rejected. PCA uses a JAX-native SVD. Each component's sign is made
    deterministic by forcing its largest-absolute loading (first on ties) to be
    nonnegative.
    """

    n_components: Optional[int] = None
    where_states: PCAStateSelector = eqx.field(
        default=PCAStateSelector("net.hidden"),
        converter=PCAStateSelector,
    )
    sampling: PCASampling = "flatten_leading"
    centering: PCACentering = "mean"
    normalization: PCANormalization = "none"
    return_data: bool = True
    aggregate_over_labels: Sequence[str] | Literal["all"] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_components is not None and self.n_components < 1:
            raise ValueError("n_components must be positive or None")
        _validate_choice("sampling", self.sampling, ("flatten_leading", "rows"))
        _validate_choice("centering", self.centering, ("mean", "none"))
        _validate_choice(
            "normalization",
            self.normalization,
            ("none", "standard_deviation"),
        )

    def _samples_from_states(self, states: Any) -> tuple[PyTree[Array], Array]:
        selected = jt.map(self.where_states.select, states, is_leaf=is_module)
        selected_leaves = jt.leaves(selected)
        if not selected_leaves:
            raise ValueError("PCA state selection produced no arrays")

        sample_arrays = []
        feature_count = None
        for array in selected_leaves:
            if array.ndim < 1:
                raise ValueError("Selected PCA state must have a feature axis")
            if feature_count is None:
                feature_count = array.shape[-1]
            elif array.shape[-1] != feature_count:
                raise ValueError(
                    "Selected PCA states have inconsistent feature counts: "
                    f"expected {feature_count}, got {array.shape[-1]}"
                )
            if self.sampling == "rows":
                if array.ndim != 2:
                    raise ValueError(
                        "sampling='rows' requires selected state shape (samples, features)"
                    )
                sample_arrays.append(array)
            else:
                sample_arrays.append(array.reshape(-1, array.shape[-1]))

        samples = jnp.concatenate(sample_arrays, axis=0)
        if samples.shape[0] < 2:
            raise ValueError("PCA requires at least two samples")
        if samples.shape[1] < 1:
            raise ValueError("PCA requires at least one feature")
        concrete_samples = _as_concrete_numpy(samples)
        if concrete_samples is not None and not np.all(np.isfinite(concrete_samples)):
            raise ValueError("Selected PCA samples must contain only finite values")
        return selected, samples

    def _fit_states(self, states: Any) -> PCAResults:
        selected, samples = self._samples_from_states(states)
        n_samples, n_features = samples.shape
        max_components = min(n_samples, n_features)
        n_components = self.n_components or max_components
        if n_components > max_components:
            raise ValueError(
                f"n_components={n_components} exceeds the fitted rank bound "
                f"min(samples, features)={max_components}"
            )

        if self.centering == "mean":
            mean = jnp.mean(samples, axis=0)
        else:
            mean = jnp.zeros((n_features,), dtype=samples.dtype)
        centered = samples - mean

        if self.normalization == "standard_deviation":
            scale = jnp.std(samples, axis=0)
            concrete_scale = _as_concrete_numpy(scale)
            if concrete_scale is not None and np.any(concrete_scale == 0):
                raise ValueError(
                    "normalization='standard_deviation' is undefined for constant features"
                )
        else:
            scale = jnp.ones((n_features,), dtype=samples.dtype)
        preprocessed = centered / scale

        _, _, basis = jnp.linalg.svd(preprocessed, full_matrices=False)
        pivot_indices = jnp.argmax(jnp.abs(basis), axis=1)
        pivot_loadings = jnp.take_along_axis(basis, pivot_indices[:, None], axis=1)[:, 0]
        signs = jnp.where(pivot_loadings < 0, -1, 1)
        basis = basis * signs[:, None]

        projected_scores = preprocessed @ basis.T
        all_explained_variance = jnp.var(projected_scores, axis=0, ddof=1)
        total_variance = jnp.sum(all_explained_variance)
        concrete_total_variance = _as_concrete_numpy(total_variance)
        if concrete_total_variance is not None and (
            not np.isfinite(concrete_total_variance) or concrete_total_variance <= 0
        ):
            raise ValueError("PCA requires positive finite total variance")

        basis = basis[:n_components]
        explained_variance = all_explained_variance[:n_components]
        explained_variance_ratio = explained_variance / total_variance

        result = PCAResults(
            basis=basis,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            mean=mean,
            scale=scale,
            states_pc=None,
            state_selector=self.where_states,
            sampling=self.sampling,
            centering=self.centering,
            normalization=self.normalization,
            sample_count=n_samples,
            feature_count=n_features,
        )
        states_pc = result.batch_transform(selected) if self.return_data else None
        return eqx.tree_at(lambda fitted: fitted.states_pc, result, states_pc)

    def compute(
        self,
        data: AnalysisInputData,
        **kwargs: Any,
    ) -> PyTree[PCAResults]:
        states = data.states
        if self.aggregate_over_labels == "all":
            is_leaf = is_type(type(states))
        else:
            if self.aggregate_over_labels:
                states = rearrange_ldict_levels(
                    states,
                    [...] + list(self.aggregate_over_labels),
                    is_leaf=is_module,
                )
                is_leaf = LDict.is_of(self.aggregate_over_labels[0])
            else:
                is_leaf = is_module

        return jt.map(self._fit_states, states, is_leaf=is_leaf)
