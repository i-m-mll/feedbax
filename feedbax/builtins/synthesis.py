"""Builtin synthesis methods."""

from pydantic import FiniteFloat

from feedbax.contracts.base import StrictModel
from feedbax.contracts.operation import JaxOperation
from feedbax.synthesis import Synthesis, SynthesisProblem, synthesis_method


class ConstantParametersSettings(StrictModel):
    value: FiniteFloat = 0


@synthesis_method(
    settings=ConstantParametersSettings,
    label="Constant parameters",
    numerical=JaxOperation(backend="cpu", floating_point_bits=64),
)
def constant_parameters(problem: SynthesisProblem, settings: ConstantParametersSettings) -> Synthesis:
    """Set each addressed parameter to the supplied constant."""
    import jax.numpy as jnp

    return Synthesis({
        address: jnp.full(slot.shape, settings.value, dtype=slot.dtype)
        for address, slot in problem.slots.items()
    })
