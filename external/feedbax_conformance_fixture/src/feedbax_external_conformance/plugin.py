"""Typed registrations exported through the single ``feedbax.plugins`` group."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from feedbax import Component
from feedbax.plugins import (
    COMPONENTS,
    FamilyRequirement,
    PluginDeclaration,
    PluginDependency,
    PluginRegistration,
    RegistrationContext,
)

from .family import EXTERNAL_DYNAMIC_COMPONENT, FIXTURE_RECORDS


_FOUNDATION_PLUGIN_ID = "feedbax_external_conformance.foundation"
_DEPENDENT_PLUGIN_ID = "feedbax_external_conformance.dependent"


class VariableFanIn(Component):
    """Small external runtime component with policy-derived input ports."""

    output_ports = ("output",)

    n_inputs: int = eqx.field(static=True)
    input_ports: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, n_inputs: int) -> None:
        self.n_inputs = n_inputs
        self.input_ports = tuple(f"source_{index}" for index in range(n_inputs))

    def __call__(self, inputs, state, *, key):
        del key
        values = tuple(jnp.atleast_1d(inputs[name]) for name in self.input_ports)
        return {"output": jnp.concatenate(values)}, state


def _build_variable_fan_in(params) -> VariableFanIn:
    return VariableFanIn(len(params["channels"]))


def _register_foundation(context: RegistrationContext) -> None:
    context.registry(FIXTURE_RECORDS).register("foundation")
    context.registry(COMPONENTS).register_component_type(
        EXTERNAL_DYNAMIC_COMPONENT,
        _build_variable_fan_in,
        category="External conformance",
        description="External dynamic fan-in bootstrap proof.",
        param_schema=[
            {"name": "channels", "type": "array", "default": ["left", "right"]},
        ],
        input_ports=["source_0", "source_1"],
        output_ports=["output"],
        port_types={
            "inputs": {
                "source_0": {"dtype": "vector"},
                "source_1": {"dtype": "vector"},
            },
            "outputs": {"output": {"dtype": "vector"}},
        },
        dynamic_port_policy={
            "count_param": "channels",
            "count_mode": "sequence_length",
            "direction": "input",
            "fixed_output_ports": ["output"],
            "generated_name_template": "source_{index}",
            "dynamic_port_type": {"dtype": "vector"},
        },
        owner="feedbax-external-conformance",
        provenance="package:feedbax-external-conformance",
    )


def _register_dependent(context: RegistrationContext) -> None:
    context.registry(FIXTURE_RECORDS).register("dependent")


FOUNDATION_PLUGIN_REGISTRATION = PluginRegistration(
    declaration=PluginDeclaration(
        plugin_id=_FOUNDATION_PLUGIN_ID,
        version="1",
        families=(
            FamilyRequirement(COMPONENTS.family),
            FamilyRequirement(FIXTURE_RECORDS.family),
        ),
    ),
    register=_register_foundation,
)

DEPENDENT_PLUGIN_REGISTRATION = PluginRegistration(
    declaration=PluginDeclaration(
        plugin_id=_DEPENDENT_PLUGIN_ID,
        version="1",
        dependencies=(PluginDependency(_FOUNDATION_PLUGIN_ID, "1"),),
        families=(FamilyRequirement(FIXTURE_RECORDS.family),),
    ),
    register=_register_dependent,
)


__all__ = [
    "DEPENDENT_PLUGIN_REGISTRATION",
    "EXTERNAL_DYNAMIC_COMPONENT",
    "FOUNDATION_PLUGIN_REGISTRATION",
]
