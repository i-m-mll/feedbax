"""Backward compatibility shim - hyperparams is now in feedbax.hyperparams."""
from feedbax.hyperparams import *
from feedbax.hyperparams import (
    set_dependent_hps,
    cast_hps,
    load_hps,
    config_to_hps,
    promote_hps,
    flatten_hps,
    update_hps_given_tree_path,
    fill_out_hps,
    take_train_histories_hps,
    flat_key_to_where_fn,
    use_train_hps_when_none,
)
