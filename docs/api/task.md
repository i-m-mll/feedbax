# Tasks

!!! Note ""
    Feedbax tasks are objects that group together:

    1. A loss function that is used to evaluated performance on a task;
    2. Per-trial data to:
        1. Initialize the state of a model prior to evaluation on task trials;
        2. Specify the parameters of task trials to the model and to the loss
           function.

---

## Reaching

### Simple reaching

::: feedbax.tasks.SimpleReachTaskInputs

<!-- ::: feedbax.tasks.SimpleReachTrialSpec -->

::: feedbax.tasks.SimpleReaches

### Delayed (cued) reaching

::: feedbax.tasks.DelayedReachTaskInputs

<!-- ::: feedbax.tasks.DelayedReachTrialSpec -->

::: feedbax.tasks.DelayedReaches

## Task trial specifications

::: feedbax.tasks.TaskTrialSpec

Custom training and streaming-evaluation loops can prepare a trial without
running the model through `AbstractTask.eval_trials` by calling
`feedbax.tasks.prepare_trial`. The returned `PreparedTrial` contains the
trial-specific initial state, normalized model inputs, merged time-varying
intervention inputs, and inferred step count.

Lower-level helpers are public for loops that need to compose only part of the
preparation path: `prepare_inputs`, `merge_intervene_inputs`,
`extract_timeseries_params`, `infer_n_steps`, `where_key_to_path`,
`set_state_by_path`, and `set_state_matching_dtypes`. `safe_state_set` is an
alias for `set_state_matching_dtypes`.

When a loop needs trainable submodules from a named graph node, use
`Graph.get_node_attrs("node", "attr", ...)` rather than reaching into
`Graph.nodes` directly.

::: feedbax.tasks.PreparedTrial

::: feedbax.tasks.prepare_trial

::: feedbax.tasks.prepare_inputs

::: feedbax.tasks.merge_intervene_inputs

::: feedbax.tasks.extract_timeseries_params

::: feedbax.tasks.infer_n_steps

::: feedbax.tasks.where_key_to_path

::: feedbax.tasks.set_state_by_path

::: feedbax.tasks.set_state_matching_dtypes

::: feedbax.tasks.safe_state_set

## Abstract base classes

<!-- ::: feedbax.tasks.AbstractTaskInputs -->

::: feedbax.tasks.AbstractTask

::: feedbax.tasks.TaskComponent

## Useful functions for building tasks

::: feedbax.tasks.internal_grid_points

## Using lambda functions as dictionary keys

::: feedbax.tasks.WhereDict
    options:
        members: []
        show_bases: false
