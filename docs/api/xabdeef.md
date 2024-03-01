# Pre-built models

Feedbax provides models, loss functions, and model-task pairings which can be constructed immediately, without needing to build them out of core components.

## Model-task pairings

!!! Example
    All pairings are provided as [`TrainingContext`][feedbax.xabdeef.TrainingContext] objects, and can be
    trained immediately.

    For example:

    ```python
    import jax.random as jr
    from feedbax.xabdeef import point_mass_nn_simple_reaches

    context = point_mass_nn_simple_reaches(key=jr.PRNGKey(0))
    model_trained, train_history = context.train()
    ```

::: feedbax.xabdeef.point_mass_nn_simple_reaches
    options:
        separate_signature: true

---

::: feedbax.xabdeef.TrainingContext
    options:
        members: ['train']


## Model only

::: feedbax.xabdeef.models.point_mass_nn
    options:
        separate_signature: true

## Loss functions

::: feedbax.xabdeef.losses.simple_reach_loss