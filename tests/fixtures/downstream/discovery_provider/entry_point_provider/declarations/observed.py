from dataclasses import replace

from equinox.nn import StateIndex
import jax.numpy as jnp

from feedbax.builtins.runtime.components import Linear
from feedbax.contracts.model_values import ModelFacts, ModelValue, StateValue


class ObservedReadout(Linear):
    component_options = replace(Linear.component_options, instantaneous=False)
    model_facts = ModelFacts(
        values=(
            ModelValue("transform", "layer.weight", "parameter"),
            ModelValue("offset", "layer.bias", "parameter"),
        ),
        states=(StateValue("last", (), "1", "unit", ("output",)),),
    )

    state_index: StateIndex

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_index = StateIndex(jnp.zeros(self.output_size, dtype=self.dtype))

    def __call__(self, inputs, state, *, key):
        outputs, state = super().__call__(inputs, state, key=key)
        return outputs, state.set(self.state_index, outputs["output"])

    def initial_outputs(self, value):
        return {"output": value}
