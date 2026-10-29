from __future__ import annotations


import jax.random as jr
import matplotlib.figure as mplfig

from feedbax.config.defaults import get_iterations_to_save_model_parameters
from feedbax.config.hyperparams import flat_key_to_where_fn, flatten_hps
from feedbax.config.namespace import TreeNamespace
from feedbax.intervene.perturbations import feedback_impulse
from feedbax.plot.utils import get_label_str, savefig
from feedbax.training.environment import EnvironmentProtocol, EnvironmentStep


def test_plot_utils_label_and_matplotlib_save(tmp_path):
    assert get_label_str("train__pert__std") == "Train pert. std."

    fig = mplfig.Figure()
    savefig(fig, "figure", tmp_path, ["svg"])

    assert (tmp_path / "figure.svg").exists()


def test_hyperparameter_selectors_and_default_save_iterations():
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


def test_training_environment_protocol_accepts_structural_implementation():
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


def test_feedback_impulse_activation_window():
    perturbation = feedback_impulse(
        n_steps=5,
        amplitude=0.25,
        duration=2,
        feedback_var=0,
        start_timestep=1,
        feedback_dim=1,
    )

    assert perturbation.params_index.init.scale == 0.25
    assert perturbation.params_index.init.active.tolist() == [False, True, True, False]
