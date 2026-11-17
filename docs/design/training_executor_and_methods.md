# Training executor and methods

A Train request binds a Model, a Task, an Objective and one installed training method. The method supplies validated settings, initialization and an update for one replica. Feedbax owns admission, replica mapping, the update clock, randomness, state adoption, metrics, checkpoint publication and recovery.

## Declare a method

Expose the package that contains the decorated function through a `feedbax.plugins` entry point. The function's default key is its top-level package and name. Feedbax's supervised gradient method uses this same discovery path. The application catalog derives its label and settings schema from the declaration.

```python
from pydantic import BaseModel

from feedbax.training.methods import TrainingState, training_method


class Settings(BaseModel):
    step_size: float = 0.001


def init(settings: Settings, model, key) -> TrainingState:
    return TrainingState(model, extra=())


@training_method(settings=Settings, init=init)
def update(state, ctx):
    # Compute one candidate update and its metrics using JAX.
    return state, {}
```

The resolved request contains the canonical method key and validated settings with defaults. The attempt records the installed distribution and version separately for provenance. The settings model owns method-specific values such as optimizer configuration. Train owns its seed, batch size, replica count, update count and checkpoint interval.

`TrainingState.model` holds the graph. `TrainingState.extra` holds an arbitrary persistent array PyTree with fixed structure. It may hold an optimizer state, an adaptive price, an inner optimization variable or a running average. Array shapes, dtypes and static PyTree metadata stay fixed during execution. Feedbax validates that the state can be represented in a checkpoint.

## Compute one update

`ctx.sample_trials()` returns a trial batch and its keys. Pass the same batch and keys to `ctx.rollout` several times to compare matched trials. Supply `input_overrides` to replace selected graph input values for a rollout. `ctx.objective` scores those outputs using the paired Objective. Methods can construct an optimizer with `ctx.optimizer` and apply a gradient update to learned model parameters with `ctx.optimizer_step`.

`ctx.step` gives the zero-based update position, `ctx.start_step` gives the schedule origin and `ctx.key` gives this replica's update key. Feedbax derives update keys from the seed, replica ID and logical position. The checkpoint stores the effective root seed and clocks. A method can derive extra keys from its update key or store its own key arrays as method state.

An update may run arbitrary JAX computations, including an inner optimization loop with a disturbance input. Its result is a candidate state and a mapping of named numerical metrics. Metric leaves may be scalars or arrays with fixed shapes. Feedbax maps the update over replicas, compiles bounded blocks and adopts a logical update only when every replica's candidate state is finite. A rejected candidate leaves the preceding committed state and update position intact.

The Host visits the run between blocks. It sends progress and retains each committed update's metric row, including the ordered replica axis for multi-replica runs. It publishes checkpoints at scheduled boundaries, at selected updates and when stopping after a committed update. Checkpoint bytes are durable before the publication event. A separate Evaluate request handles independent validation; a measurement that influences future training belongs in the method state.

## Start and recover

A fresh Train initializes the Model and calls the selected method's `init`. A Train with one predecessor checkpoint and `checkpoint_initialization: initialize_from` reads only the predecessor's semantic model product. It calls the target method's `init` with the restored model and begins a new update clock. This supports switching methods without needing the predecessor's method code or extra state.

A Train with `checkpoint_initialization.mode: continue_from` starts at its authenticated predecessor's declared completed update. It selects whether to preserve or reset method state, whether to carry or reset the random seed, and whether to continue or restart the schedule origin. Preserving extra state requires the same method and a compatible state layout. The effective input identity includes the model, completed clock and only the predecessor state, schedule origin and seed the policy consumes.

Resume continues the same request from its own authenticated checkpoint. The original authored input binding stays intact. The recovery binding verifies the method and state ABI, restores the exact model and extra arrays, and restores the completed update, segment start, schedule origin and random seed. The numerical loop starts at the next update. A deterministic backend with the same declarations reproduces the committed state from an uninterrupted run.

The checkpoint publishes the semantic model as its own product with ordered replicas. Evaluate and model-consuming operations restore that product through the shared model reconstruction service. The method-specific arrays remain separately addressable for exact recovery and compatible continuation.
