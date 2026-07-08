"""Mechanics-domain acausal leaf elements."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from feedbax.acausal.base import AcausalElement, AcausalEquation, AcausalPort, Domain


def _fqn(element: str, port: str, slot: str) -> str:
    return f"{element}.{port}.{slot}"


@dataclass
class RigidTendonHillMuscle(AcausalElement):
    """Preformed rigid-tendon Hill muscle leaf for mechanics interiors.

    The current acausal vocabulary has no nonlinear force-length/velocity law,
    activation state element, or articulated muscle path vocabulary. This leaf is
    therefore intentionally honest: a single conserving translational port and a
    causal activation signal input. C2 can replace it with a decomposed interior
    once multibody links, joints, and path geometry exist.
    """

    def __init__(
        self,
        name: str,
        max_isometric_force: float = 1.0,
        direction: float = 1.0,
    ):
        super().__init__(name=name)
        self.ports["flange"] = AcausalPort(
            name="flange",
            domain=Domain.TRANSLATIONAL,
            across_vars=("pos", "vel"),
            through_var="force",
        )
        self.signal_inputs = ("activation",)
        self.params[f"{name}.max_isometric_force"] = max_isometric_force
        self.params[f"{name}.direction"] = direction

        activation = f"{name}.activation"
        force = _fqn(name, "flange", "force")
        fmax = f"{name}.max_isometric_force"
        sign = f"{name}.direction"
        self.equations.append(
            AcausalEquation(
                lhs_var=force,
                rhs_fn=lambda vals, _a=activation, _f=fmax, _s=sign: (
                    vals[_s] * vals[_f] * jnp.clip(vals[_a], 0.0, 1.0)
                ),
                depends_on=(activation, fmax, sign),
                is_through_def=True,
            )
        )
