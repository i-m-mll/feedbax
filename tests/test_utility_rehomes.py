from __future__ import annotations


import matplotlib.figure as mplfig

from feedbax.config.defaults import get_iterations_to_save_model_parameters
from feedbax.config.hyperparams import flat_key_to_where_fn, flatten_hps
from feedbax.config.namespace import TreeNamespace
from feedbax.plot.utils import get_label_str, savefig
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
