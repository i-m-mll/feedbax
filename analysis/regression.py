"""Facilities for linear regression given regressor-structured PyTrees."""

from collections.abc import Mapping
from functools import partial
import itertools
from types import MappingProxyType
from typing import Sequence, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import PRNGKeyArray

from feedbax.train import SimpleTrainer, grad_wrap_simple_loss_func
from feedbax.loss import nan_safe_mse

from rlrmp.analysis.aligned import AlignedVars
from rlrmp.analysis.analysis import AbstractAnalysis, AbstractAnalysisPorts, InputOf
from rlrmp.tree_utils import ldict_level_keys, tree_level_labels
from rlrmp.types import AnalysisInputData


def prepare_interaction_indices(regressor_labels, interactions):
    """Convert interaction label pairs to index pairs."""
    interaction_indices = []
    for label1, label2 in interactions:
        idx1 = regressor_labels.index(label1)
        idx2 = regressor_labels.index(label2)
        interaction_indices.append((idx1, idx2))
    return interaction_indices


def build_feature_names(regressor_labels, interactions):
    """Build feature names for the design matrix."""
    feature_names = ['intercept'] + regressor_labels[:]
    for label1, label2 in interactions:
        feature_names.append(f'{label1}*{label2}')
    return feature_names


def prepare_regression_data(tree, interaction_indices, n_features):
    """Build design matrix with pre-computed interaction indices."""
    # Extract paths and leaves
    leaves_with_paths = jax.tree.leaves_with_path(tree)
    paths_and_leaves = [
        ([key_obj.key for key_obj in path], leaf) 
        for path, leaf in leaves_with_paths
    ]
    
    # Get dimensions
    n_combinations = len(paths_and_leaves)
    sample_leaf = paths_and_leaves[0][1]
    n_obs_per_combination = int(jnp.prod(jnp.array(sample_leaf.shape)))
    total_obs = n_combinations * n_obs_per_combination
    
    # Build regressor value matrix for all combinations
    n_regressors = len(paths_and_leaves[0][0])  # number of PyTree levels
    regressor_matrix = jnp.array([
        [path[i] for i in range(n_regressors)]
        for path, _ in paths_and_leaves
    ])  # Shape: (n_combinations, n_regressors)
    
    # Repeat for observations within each combination
    regressor_expanded = jnp.repeat(regressor_matrix, n_obs_per_combination, axis=0)
    
    # Pre-allocate design matrix
    X = jnp.zeros((total_obs, n_features))
    
    # Fill design matrix
    X = X.at[:, 0].set(1.0)  # intercept
    X = X.at[:, 1:1+n_regressors].set(regressor_expanded)  # main effects
    
    # Add interactions using pre-computed indices
    for i, (idx1, idx2) in enumerate(interaction_indices):
        interaction_col = regressor_expanded[:, idx1] * regressor_expanded[:, idx2]
        X = X.at[:, 1 + n_regressors + i].set(interaction_col)
    
    # Build y_data
    y_parts = [leaf.flatten() for path, leaf in paths_and_leaves]
    y_data = jnp.concatenate(y_parts)
    
    return X, y_data


def fit_single_regression(tree, interaction_indices, n_features, key, n_iter=50):
    """Fit a single regression on completely flattened data."""
    X, y_data = prepare_regression_data(tree, interaction_indices, n_features)
    
    # Create and fit model
    lin_model = jt.map(
        jnp.zeros_like,
        eqx.nn.Linear(X.shape[-1], 1, use_bias=False, key=key),
    )
    
    trainer = SimpleTrainer(
        loss_func=grad_wrap_simple_loss_func(nan_safe_mse, nan_safe=True),
    )
    
    model = trainer(lin_model, X.T, y_data[:, None].T, n_iter=n_iter, progress_bar=False)
    return model


def fit_regression_from_pytree_vmap(
    tree,
    interactions: Sequence[Tuple[str, str]] = (),
    parallel_axis: Optional[int] = None,
    n_iter: int = 50,
    *,
    key
):
    """
    Fit regressions using vmap for clean separation of concerns.
    """
    # Pre-compute regressor structure (independent of vmapping)
    regressor_vals = {
        label: ldict_level_keys(tree, label)
        for label in tree_level_labels(tree)
    }
    regressor_labels = list(regressor_vals.keys())
    
    # Convert interactions to indices
    interaction_indices = prepare_interaction_indices(regressor_labels, interactions)
    
    # Build feature names
    feature_names = build_feature_names(regressor_labels, interactions)
    n_features = len(feature_names)
    
    if parallel_axis is None:
        # Single regression case
        model = fit_single_regression(tree, interaction_indices, n_features, key, n_iter=n_iter)
        return model, feature_names
    else:
        # Parallel case - vmap over the specified axis
        
        # Get number of parallel regressions
        sample_leaf = jax.tree.leaves(tree)[0]
        n_parallel = sample_leaf.shape[parallel_axis]
        keys = jax.random.split(key, n_parallel)
        
        # Vmap the fitting function
        vmapped_fit = eqx.filter_vmap(
            partial(fit_single_regression, n_iter=n_iter),
            in_axes=(parallel_axis, None, None, 0)  # tree, interaction_indices, n_features, n_iter, keys
        )
        
        models = vmapped_fit(tree, interaction_indices, n_features, n_iter, keys)
        
        return models, feature_names
    
    
class RegressionPorts(AbstractAnalysisPorts):
    """Input ports for Regression analysis."""
    regressor_tree: InputOf[AlignedVars]


class Regression(AbstractAnalysis[RegressionPorts]):
    Ports = RegressionPorts
    inputs: RegressionPorts = eqx.field(default_factory=RegressionPorts, converter=RegressionPorts.converter)
    
    variant: Optional[str] = "full"
    fig_params: Mapping = MappingProxyType(dict(
        mode='std', # or 'curves'
        n_std_plot=1,
        layout_kws=dict(
            width=600,
            height=400,
            legend_tracegroupgap=1,
        )
    ))
    key: PRNGKeyArray = eqx.field(default_factory=lambda: jr.PRNGKey(0))

    def compute(self, data: AnalysisInputData, *, regressor_tree, **kwargs):
        # Regression 
        # independents: SISU and curl field amplitude are in PyTree structure 
        # dependents: computed from `aligned_vars` as in `transform_profile_vars`
        
        models, feature_names = fit_regression_from_pytree_vmap(regressor_tree[self.variant], key=self.key)
        return models, feature_names