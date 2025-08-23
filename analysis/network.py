from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from feedbax.train import SimpleTrainer
from jaxtyping import PyTree, PRNGKeyArray
import numpy as np

from feedbax.train import grad_wrap_simple_loss_func
from feedbax.loss import nan_safe_mse
from jax_cookbook import is_module

from rlrmp.analysis.analysis import AbstractAnalysis, NoPorts
from rlrmp.analysis.state_utils import output_corr
from rlrmp.misc import center_and_rescale, ravel_except_last
from rlrmp.plot import get_violins
from rlrmp.types import AnalysisInputData, LDict, TreeNamespace


class OutputWeightCorrelation(AbstractAnalysis[NoPorts]):
    variant: Optional[str] = "full"
    
    def compute(
        self, 
        data: AnalysisInputData,
        **kwargs,
    ):
        activities = jt.map(
            lambda states: states.net.hidden,
            data.states[self.variant],
            is_leaf=is_module,
        )

        output_weights = jt.map(
            lambda models: models.step.net.readout.weight,
            data.models,
            is_leaf=is_module,
        )
        
        #! TODO: Generalize
        output_corrs = jt.map(
            lambda activities: LDict.of("train__pert__std")({
                train_std: output_corr(
                    activities[train_std], 
                    output_weights[train_std],
                )
                for train_std in activities
            }),
            activities,
            is_leaf=LDict.is_of("train__pert__std"),
        )
        
        return output_corrs
        
    def make_figs(
        self, 
        data: AnalysisInputData,
        *, 
        result, 
        colors, 
        **kwargs,
    ):
        #! TODO: Generalize
        assert result is not None
        fig = get_violins(
            result, 
            yaxis_title="Output correlation", 
            xaxis_title="Train field std.",
            colors=colors['pert__amp'].dark,
        )
        return fig

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape)),
            measure="output_correlation",
        )


def fit_linear(X, y, n_iter=50, *, key):
    lin_model = jt.map(
        jnp.zeros_like,
        eqx.nn.Linear(X.shape[-1], 1, key=key),
    )
    
    trainer = SimpleTrainer(
        #! Use nanmean loss to avoid training on excluded data.
        loss_func=grad_wrap_simple_loss_func(nan_safe_mse, nan_safe=True),
    )
    return trainer(lin_model, X.T, y, n_iter=n_iter, progress_bar=False)


class UnitPreferences(AbstractAnalysis[NoPorts]):
    variant: Optional[str] = "full"
    n_iter_fit: int = 50
    feature_fn: Callable = lambda task, states: task.validation_trials.targets["mechanics.effector.pos"].value
    key: PRNGKeyArray = eqx.field(default_factory=lambda: jr.PRNGKey(0))  # For linear fit -- not very important.

    def compute(
            self,
            data: AnalysisInputData,
            **kwargs,
    ):
        return jt.map(
            lambda task, states_by_task: jt.map(
                lambda states: self.get_prefs(task, states, self.key),
                states_by_task,
                is_leaf=is_module,
            ),
            data.tasks,
            data.states,
            is_leaf=is_module,
        )

    # We could also pass `model` and `hps` here, but I don't see why we'd ever be
    # treating them as features -- they don't have a time dimension.
    def get_prefs(self, task, states, key):
        activities = states.net.hidden
        features = self.feature_fn(task, states)
        # Generally, `activities` may have more axes than `features`, e.g. when 
        # the features are from the task, and we are evaluating the same conditions 
        # multiple times. However, any batch axes that are present in `features`
        # must be broadcastable with any that are present in `activities`.
        # We explicitly broadcast `features` here, because the trainer works with 
        # aggregated data.
        features_broadcast = jnp.broadcast_to(
            features,
            activities.shape[:-1] + (features.shape[-1],),
        )
        features_flat = center_and_rescale(ravel_except_last(features_broadcast))
        activities_flat = ravel_except_last(activities)
        return jnp.squeeze(self._batch_fit_linear(key=key)(
            features_flat, activities_flat
        ).weight)

    def _batch_fit_linear(self, key):
        return jax.vmap(
            partial(fit_linear, n_iter=self.n_iter_fit, key=key),
            in_axes=(None, 1),
        )
        
    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        result,
        colors,
        **kwargs,
    ):
        ...
            