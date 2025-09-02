"""Facilities for linear regression given regressor-structured PyTrees."""

import itertools
from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import NamedTuple, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from feedbax.loss import nan_safe_mse
from feedbax.train import SimpleTrainer, grad_wrap_simple_loss_func
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax_experiments.analysis.aligned import AlignedVars
from feedbax_experiments.analysis.analysis import AbstractAnalysis, AbstractAnalysisPorts, InputOf
from feedbax_experiments.tree_utils import ldict_level_keys, tree_level_labels
from feedbax_experiments.types import AnalysisInputData, LDictTree


def prepare_interaction_indices(
    regressor_labels: Sequence[str], interactions: Sequence[Tuple[str, str]]
):
    """Convert interaction label pairs to index pairs."""
    interaction_indices = []
    for label1, label2 in interactions:
        idx1 = regressor_labels.index(label1)
        idx2 = regressor_labels.index(label2)
        interaction_indices.append((idx1, idx2))
    return interaction_indices


def build_feature_names(regressor_labels: Sequence[str], interactions: Sequence[Tuple[str, str]]):
    """Build feature names for the design matrix."""
    feature_names = ["intercept"] + list(regressor_labels)
    for label1, label2 in interactions:
        feature_names.append(f"{label1}*{label2}")
    return feature_names


def prepare_regression_data(
    tree: LDictTree[Array],
    interaction_indices: Sequence[tuple[int, int]],
    n_features: int,
    center_and_scale: bool = True,
):
    """Build design matrix from a PyTree where tree structure defines regressors.

    Converts a nested LDict tree into a regression design matrix where each level
    of the tree hierarchy becomes a regressor variable. The keys at each level
    are used directly as regressor values (can be numeric or categorical).

    Args:
        tree: Nested LDict structure where paths encode regressor values
            and leaves contain response variable arrays
        interaction_indices: List of (idx1, idx2) tuples specifying which pairs
            of tree levels should have interaction terms
        n_features: Total number of features (1 + n_levels + n_interactions)

    Returns:
        X: Design matrix of shape (n_total_obs, n_features) with intercept,
            main effects, and interaction terms
        y_data: Flattened response vector of shape (n_total_obs,)
    """
    # Extract paths and leaves
    leaves_with_paths = jax.tree.leaves_with_path(tree)
    paths_and_leaves = [
        ([key_obj.key for key_obj in path], leaf) for path, leaf in leaves_with_paths
    ]

    # Get dimensions
    n_combinations = len(paths_and_leaves)
    sample_leaf = paths_and_leaves[0][1]
    n_obs_per_combination = int(jnp.prod(jnp.array(sample_leaf.shape)))
    total_obs = n_combinations * n_obs_per_combination

    # Build regressor value matrix for all combinations
    n_regressors = len(paths_and_leaves[0][0])  # number of PyTree levels
    regressor_matrix = jnp.array(
        [[path[i] for i in range(n_regressors)] for path, _ in paths_and_leaves]
    )  # Shape: (n_combinations, n_regressors)

    # Repeat for observations within each combination
    regressor_expanded = jnp.repeat(regressor_matrix, n_obs_per_combination, axis=0)

    # Pre-allocate design matrix
    X = jnp.zeros((total_obs, n_features))

    # Fill design matrix
    X = X.at[:, 0].set(1.0)  # intercept
    X = X.at[:, 1 : 1 + n_regressors].set(regressor_expanded)  # main effects

    # Add interactions using pre-computed indices
    for i, (idx1, idx2) in enumerate(interaction_indices):
        interaction_col = regressor_expanded[:, idx1] * regressor_expanded[:, idx2]
        X = X.at[:, 1 + n_regressors + i].set(interaction_col)

    # Build y_data
    y_parts = [leaf.flatten() for path, leaf in paths_and_leaves]
    y_data = jnp.concatenate(y_parts)

    if center_and_scale:
        # Do not scale intercept
        for j in range(1, X.shape[1]):
            col = X[:, j]
            X = X.at[:, j].set((col - jnp.mean(col)) / (jnp.std(col) + 1e-8))

        # Scale response
        y_data = (y_data - jnp.mean(y_data)) / (jnp.std(y_data) + 1e-8)

    return X, y_data


def fit_single_regression(
    tree: LDictTree[Array],
    interaction_indices: Sequence[tuple[int, int]],
    n_features: int,
    key: PRNGKeyArray,
    n_iter: int = 50,
) -> eqx.nn.Linear:
    """Fit a single regression on completely flattened data."""
    X, y_data = prepare_regression_data(tree, interaction_indices, n_features)

    # Create and fit model
    lin_model: eqx.nn.Linear = jt.map(
        jnp.zeros_like,
        eqx.nn.Linear(X.shape[-1], 1, use_bias=False, key=key),
    )

    trainer = SimpleTrainer(
        loss_func=grad_wrap_simple_loss_func(nan_safe_mse, nan_safe=True),
    )

    model = trainer(lin_model, X.T, y_data[:, None].T, n_iter=n_iter, progress_bar=False)
    return model


class RegressionResults(NamedTuple):
    model: eqx.nn.Linear
    feature_names: Sequence[str]


def fit_regression_from_pytree_vmap(
    tree: LDictTree[Array],
    interactions: Sequence[Tuple[str, str]] = (),
    parallel_axis: Optional[int] = None,
    n_iter: int = 50,
    *,
    key: PRNGKeyArray,
):
    """Fit linear regression(s) using tree structure to define regressors.

    Performs regression where regressors are derived from the hierarchical structure
    of a PyTree of nested LDicts. Each tree level (identified by its LDict label)
    becomes a regressor, with its keys used directly as regressor values. Keys can
    be numeric (e.g., amplitudes, parameters) or categorical. Optionally includes
    interaction terms between specified pairs of tree levels.

    Args:
        tree: Nested LDict structure where each level represents a regressor
            variable and leaves contain data arrays. Keys at each level become
            the regressor values for that variable.
        interactions: Pairs of tree level labels (e.g., [("amplitude", "frequency")])
            for which to include interaction terms (product of main effects)
        parallel_axis: If None, performs single regression on all data. If specified,
            vmaps over this axis of leaf arrays to run multiple independent regressions
        n_iter: Number of training iterations for each regression
        key: PRNG key for model initialization

    Returns:
        model: Fitted linear model(s). Single model if parallel_axis=None,
            otherwise vmapped models with one per slice along parallel_axis
        feature_names: List of feature names ["intercept", <main_effects>, <interactions>]

    Example:
        tree = LDict("amplitude", {
            0.5: LDict("frequency", {1.0: data1, 2.0: data2}),
            1.0: LDict("frequency", {1.0: data3, 2.0: data4})
        })
        # Creates regressors for amplitude (0.5 or 1.0), frequency (1.0 or 2.0),
        # and optionally their interaction (amplitude * frequency)
    """
    # Pre-compute regressor structure (independent of vmapping)
    regressor_vals = {label: ldict_level_keys(tree, label) for label in tree_level_labels(tree)}
    regressor_labels = list(regressor_vals.keys())

    # Convert interactions to indices
    interaction_indices = prepare_interaction_indices(regressor_labels, interactions)

    # Build feature names
    feature_names = build_feature_names(regressor_labels, interactions)
    n_features = len(feature_names)

    if parallel_axis is None:
        # Single regression case
        model = fit_single_regression(tree, interaction_indices, n_features, key, n_iter=n_iter)
        return RegressionResults(model, feature_names)
    else:
        # Parallel case - vmap over the specified axis

        # Get number of parallel regressions
        sample_leaf = jax.tree.leaves(tree)[0]
        n_parallel = sample_leaf.shape[parallel_axis]
        keys = jax.random.split(key, n_parallel)

        # Vmap the fitting function
        vmapped_fit = eqx.filter_vmap(
            partial(fit_single_regression, n_iter=n_iter),
            in_axes=(
                parallel_axis,
                None,
                None,
                0,
            ),  # tree, interaction_indices, n_features, n_iter, keys
        )

        models = vmapped_fit(tree, interaction_indices, n_features, n_iter, keys)

        return RegressionResults(models, feature_names)


class RegressionPorts(AbstractAnalysisPorts):
    """Input ports for Regression analysis."""

    regressor_tree: InputOf[LDictTree[Array]]


class Regression(AbstractAnalysis[RegressionPorts]):
    Ports = RegressionPorts
    inputs: RegressionPorts = eqx.field(
        default_factory=RegressionPorts, converter=RegressionPorts.converter
    )

    fig_params: Mapping = MappingProxyType(
        dict(
            mode="std",  # or 'curves'
            n_std_plot=1,
            layout_kws=dict(
                width=600,
                height=400,
                legend_tracegroupgap=1,
            ),
        )
    )

    interactions: Sequence[Tuple[str, str]] = ()
    key: PRNGKeyArray = eqx.field(default_factory=lambda: jr.PRNGKey(0))

    def compute(self, data: AnalysisInputData, *, regressor_tree, **kwargs) -> RegressionResults:
        # Regression
        # independents: SISU and curl field amplitude are in PyTree structure
        # dependents: computed from `aligned_vars` as in `transform_profile_vars`

        return fit_regression_from_pytree_vmap(
            regressor_tree,
            key=self.key,
            interactions=self.interactions,
        )
