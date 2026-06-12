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
