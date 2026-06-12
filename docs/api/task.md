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

::: feedbax.task.SimpleReachTaskInputs

<!-- ::: feedbax.task.SimpleReachTrialSpec -->

::: feedbax.task.SimpleReaches

### Delayed (cued) reaching

::: feedbax.task.DelayedReachTaskInputs

<!-- ::: feedbax.task.DelayedReachTrialSpec -->

::: feedbax.task.DelayedReaches

## Task trial specifications

::: feedbax.task.TaskTrialSpec

Custom training and streaming-evaluation loops can prepare a trial without
running the model through `AbstractTask.eval_trials` by calling
`feedbax.task.prepare_trial`. The returned `PreparedTrial` contains the
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

::: feedbax.task.PreparedTrial

::: feedbax.task.prepare_trial

::: feedbax.task.prepare_inputs

::: feedbax.task.merge_intervene_inputs

::: feedbax.task.extract_timeseries_params

::: feedbax.task.infer_n_steps

::: feedbax.task.where_key_to_path

::: feedbax.task.set_state_by_path

::: feedbax.task.set_state_matching_dtypes

::: feedbax.task.safe_state_set

## Abstract base classes

<!-- ::: feedbax.task.AbstractTaskInputs -->

::: feedbax.task.AbstractTask

::: feedbax.task.TaskComponent

## Useful functions for building tasks

::: feedbax.task.internal_grid_points

## Using lambda functions as dictionary keys

::: feedbax.task.WhereDict
    options:
        members: []
        show_bases: false
