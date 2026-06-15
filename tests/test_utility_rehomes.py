from __future__ import annotations

import warnings

import jax.numpy as jnp
import jax.random as jr
import matplotlib.figure as mplfig

from feedbax.analysis.dimred import PCAResults, pca
from feedbax.config.defaults import get_iterations_to_save_model_parameters
from feedbax.config.hyperparams import flat_key_to_where_fn, flatten_hps
from feedbax.config.logging import BacktickPathHighlighter, enable_logging_handlers
from feedbax.config.warnings import enable_warning_dedup
from feedbax.intervene.perturbations import feedback_impulse
from feedbax.plot.utils import get_label_str, savefig
from feedbax.training.environment import EnvironmentProtocol, EnvironmentStep
from feedbax.types import TreeNamespace


def test_plot_utils_label_and_matplotlib_save(tmp_path):
    assert get_label_str("train__pert__std") == "Train pert. std."

    fig = mplfig.Figure()
    savefig(fig, "figure", tmp_path, ["svg"])

    assert (tmp_path / "figure.svg").exists()


def test_hyperparams_and_defaults_import_from_config_home():
    hps = TreeNamespace(train=TreeNamespace(batch_size=32), model=TreeNamespace(n_replicates=2))

    flat = flatten_hps(hps)
    where = flat_key_to_where_fn("train__batch_size")

    assert flat.train__batch_size == 32
    assert where(hps) == 32
    assert get_iterations_to_save_model_parameters(25).tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        20,
    ]


def test_training_environment_protocol_home():
    class DummyEnvironment:
        def init_env_state(self, trial_spec, key):
            return None

        def step_env(self, env_state, action, key, t, trial_spec):
            return EnvironmentStep(
                obs={"t": t},
                target=None,
                intervene=None,
                reward=None,
                done=None,
            )

    env = DummyEnvironment()

    assert isinstance(env, EnvironmentProtocol)
    assert env.step_env(None, None, jr.PRNGKey(0), 1, None).obs == {"t": 1}


def test_intervention_perturbation_constructor_home():
    perturbation = feedback_impulse(
        n_steps=5,
        amplitude=0.25,
        duration=2,
        feedback_var=0,
        start_timestep=1,
        feedback_dim=1,
    )

    assert perturbation._initial_state.scale == 0.25
    assert perturbation._initial_state.active.tolist() == [False, True, True, False]


def test_analysis_dimred_home():
    result = pca(jnp.arange(12.0).reshape(2, 3, 2))

    assert isinstance(result, PCAResults)
    assert result.components.shape == (2, 3, 2)
    assert result.singular_values.shape == (2,)


def test_cli_logging_and_warning_helpers_import_from_config_home():
    previous_showwarning = enable_warning_dedup()
    try:
        assert warnings.showwarning is not previous_showwarning
    finally:
        warnings.showwarning = previous_showwarning

    assert issubclass(BacktickPathHighlighter, object)
    assert callable(enable_logging_handlers)
