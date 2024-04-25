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

::: feedbax.task.SimpleReachTrialSpec

::: feedbax.task.SimpleReaches

### Delayed (cued) reaching

::: feedbax.task.DelayedReachTaskInputs

::: feedbax.task.DelayedReachTrialSpec

::: feedbax.task.DelayedReaches

## Abstract base classes

<!-- ::: feedbax.task.AbstractTaskInputs -->

::: feedbax.task.AbstractTaskTrialSpec

::: feedbax.task.AbstractTask

## Useful functions for building tasks

::: feedbax.task.internal_grid_points

## Using lambda functions as dictionary keys

::: feedbax.task.WhereDict
    options:
        members: []
        show_bases: false