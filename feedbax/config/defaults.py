from typing import Any

import jax.numpy as jnp
import numpy as np

## Task parameters

EVAL_REACH_LENGTH = 0.5

TASK_EVAL_PARAMS: dict[str, dict[str, Any]] = dict(
    full=dict(
        eval_grid_n=2,
        eval_n_directions=24,
        eval_reach_length=EVAL_REACH_LENGTH,
    ),
    small=dict(
        eval_grid_n=1,
        eval_n_directions=7,
        eval_reach_length=EVAL_REACH_LENGTH,
    ),
)
# Once effector positions are center-subtracted and aligned to reach direction,
# all the effector-relative endpoints are the same and only depend on the reach length,
# which we have defined once and for all, above.
POS_ENDPOINTS_ALIGNED = {
    k: jnp.array([[0.0, 0.0], [params["eval_reach_length"], 0.0]])
    for k, params in TASK_EVAL_PARAMS.items()
}
# POS_ENDPOINTS_ALIGNED = jnp.array([
#     [0., 0.], [EVAL_REACH_LENGTH, 0.]
# ])


# Criterion by which to exclude model replicates from analysis
REPLICATE_CRITERION = "best_total_loss"

"""
Define the training iterations on which to retain the model weights:
Every iteration until iteration 10, then every 10 until 100, every 100 until 1000, etc.
"""


def get_iterations_to_save_model_parameters(n_batches):
    save_iterations = jnp.concatenate(
        [jnp.array([0])]
        + [jnp.arange(10**i, 10 ** (i + 1), 10**i) for i in range(0, int(np.log10(n_batches)) + 1)]
    )
    return save_iterations[save_iterations < n_batches]
