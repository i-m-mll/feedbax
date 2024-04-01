
import jax.random as jr
import optax

from feedbax import get_ensemble, save
from feedbax.task import SimpleReaches
from feedbax.train import TaskTrainer
from feedbax.xabdeef.losses import simple_reach_loss
from feedbax.xabdeef.models import point_mass_nn


hyperparameters = dict(
    workspace=((-1., -1.),  # ((x_min, y_min),
               (1., 1.)),   #  (x_max, y_max))
    n_steps=100,
    n_replicates = 5,
    dt=0.05,
)

key = jr.PRNGKey(0)

key_init, key_train, key_eval = jr.split(key, 3)

def setup(
    workspace, n_steps, n_replicates, dt, *, key
):

    task = SimpleReaches(
        loss_func=simple_reach_loss(),
        workspace=workspace,
        n_steps=n_steps,
    )

    models = get_ensemble(
        point_mass_nn,
        task,  # We need to pass anything we would need to pass to `point_mass_nn`
        dt=dt,  # We can also pass keyword arguments that `point_mass_nn` accepts
        n_ensemble=n_replicates,
        key=key,
    )

    return task, models


if __name__ == '__main__':
    task, models = setup(
        **hyperparameters,
        key=jr.PRNGKey(0),
    )

    trainer = TaskTrainer(
        optimizer=optax.adam(learning_rate=1e-2),
        checkpointing=True,
    )

    models, train_history = trainer(
        task=task,
        model=models,
        ensembled=True,
        n_batches=1000,
        batch_size=250,
        log_step=200,
        where_train=lambda model: model.step.net,
        key=key_train,
    )


    save_path = "dash_demo_data.eqx"

    save(
        save_path,
        (task, models),
        hyperparameters=hyperparameters,
    )