import equinox as eqx
import jax.numpy as jnp
import pytest
from equinox.nn import State, StateIndex

from feedbax.intervene import FixedField, FixedFieldParams
from feedbax.training.trainer import _state_set_matching_dtypes


class _WeakStateComponent(eqx.Module):
    state_index: StateIndex

    def __init__(self):
        self.state_index = StateIndex(1.0)


def test_intervenor_defaults_are_strong_typed_for_public_state_set():
    component = FixedField(params=FixedFieldParams(scale=1.0))
    state = State(component)
    current = state.get(component.params_index)

    assert getattr(current.scale, "weak_type", None) is False

    replacement = FixedFieldParams(scale=jnp.asarray(2.0, dtype=jnp.float32))
    new_state = _state_set_matching_dtypes(state, component.params_index, replacement)

    assert new_state.get(component.params_index).scale == jnp.asarray(2.0, dtype=jnp.float32)


def test_state_set_matching_dtypes_rejects_weak_state_defaults():
    component = _WeakStateComponent()
    state = State(component)

    with pytest.raises(ValueError, match="weakly typed StateIndex leaf"):
        _state_set_matching_dtypes(
            state,
            component.state_index,
            jnp.asarray(2.0, dtype=jnp.float32),
        )
