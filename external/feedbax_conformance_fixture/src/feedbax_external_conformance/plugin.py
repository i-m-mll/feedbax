"""Typed registrations exported through the single ``feedbax.plugins`` group."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from feedbax import Component
from feedbax.plugins import (
    COMPONENTS,
    DRIVERS,
    FamilyRequirement,
    PluginDeclaration,
    PluginDependency,
    PluginRegistration,
    RegistrationContext,
)
from feedbax.orchestration.drivers import (
    AcquisitionSemantics,
    AuthorizationSemantics,
    CustodySemantics,
    DriverCapabilityEnvelope,
    DriverCapabilityFacts,
    DriverRegistration,
    DriverVenue,
    EnvironmentSemantics,
    MonitoringSemantics,
    RecoverySemantics,
    ResourceSemantics,
    RetrySemantics,
    SpendSemantics,
    TeardownSemantics,
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


class FixtureOrchestrationDriver:
    """Minimal external driver proving registry construction without CLI edits."""

    poll_interval_seconds = 0.05
    capability_envelope = DriverCapabilityEnvelope.single(
        "fixture:driver",
        DriverCapabilityFacts(
            variant_id="fixture",
            venue=DriverVenue.LOCAL_PROCESS,
            resources=ResourceSemantics.EXTERNALLY_MANAGED,
            spend=SpendSemantics.NONE,
            authorization=AuthorizationSemantics.NONE,
            environment=EnvironmentSemantics.OPAQUE_DRIVER_IDENTITY,
            monitoring=MonitoringSemantics.ROW_POLL,
            recovery=RecoverySemantics.NONE,
            retry=RetrySemantics.NONE,
            acquisition=AcquisitionSemantics.EXTERNALLY_PROVIDED,
            teardown=TeardownSemantics.RESOURCES_PRESERVED,
            custody=CustodySemantics.EXTERNAL_SERVICE,
        ),
    )

    def __init__(self, realized) -> None:
        self.realized_capabilities = realized

    def provision(self, *_args):
        return {}

    def realize_env(self, *_args):
        return "fixture"

    def stage_inputs(self, *_args):
        return {}

    def launch_row(self, *_args):
        return {}

    def probe(self, *_args):
        return object()

    def stop_row(self, *_args):
        return {}

    def collect(self, *_args):
        return {}

    def teardown(self, *_args):
        return {}


def _fixture_driver_registration() -> DriverRegistration:
    envelope = FixtureOrchestrationDriver.capability_envelope
    return DriverRegistration(
        name="fixture:driver",
        supported_capabilities=envelope,
        resolve_capabilities=lambda _context: envelope.realize("fixture"),
        factory=lambda _context, realized: FixtureOrchestrationDriver(realized),
    )


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
    context.registry(DRIVERS).register(_fixture_driver_registration())


def _register_dependent(context: RegistrationContext) -> None:
    context.registry(FIXTURE_RECORDS).register("dependent")


FOUNDATION_PLUGIN_REGISTRATION = PluginRegistration(
    declaration=PluginDeclaration(
        plugin_id=_FOUNDATION_PLUGIN_ID,
        version="1",
        families=(
            FamilyRequirement(COMPONENTS.family),
            FamilyRequirement(FIXTURE_RECORDS.family),
            FamilyRequirement(DRIVERS.family),
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
