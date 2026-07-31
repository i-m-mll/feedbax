"""Typed registrations exported through the single ``feedbax.plugins`` group."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from dataclasses import replace

from feedbax import Component
from feedbax.plugins import (
    ANALYSIS_RECIPES,
    COMPONENTS,
    DOWNSTREAM_PROTOCOL_CURRENT,
    DRIVERS,
    EVALUATION_BATCH_CONSUMERS,
    EVALUATION_PRODUCT_UNION_FINALIZERS,
    EVALUATION_RECIPES,
    EXECUTION_PREPARATIONS,
    FamilyRequirement,
    PluginDeclaration,
    PluginDependency,
    PluginRegistration,
    RegistrationContext,
    ROW_LOWERERS,
    TRAINING_METHODS,
    EvaluationBatchFragment,
    ExecutionPreparationRegistration,
    TrainingRowLowererRegistration,
)
from feedbax.contracts.training import (
    TrainingMethodDescriptor,
    standard_supervised_method_contract,
    standard_supervised_method_descriptor,
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


_FIXTURE_METHOD_REF = "feedbax_external_conformance/training/v1"
_FIXTURE_ANALYSIS_TYPE = "feedbax_external_conformance.analysis"
_FIXTURE_EVALUATION_TYPE = "feedbax_external_conformance.evaluation"
_FIXTURE_CONSUMER_ID = "feedbax_external_conformance.consumer"
_FIXTURE_CONSUMER_VERSION = "v1"


def _fixture_method_descriptor() -> TrainingMethodDescriptor:
    baseline = standard_supervised_method_descriptor()
    return replace(
        baseline,
        method_ref=_FIXTURE_METHOD_REF,
        contract_compiler=lambda _payload: standard_supervised_method_contract().model_copy(
            update={"method_ref": _FIXTURE_METHOD_REF}
        ),
        owner="feedbax-external-conformance",
        package="feedbax-external-conformance",
    )


def _fixture_lowerer(_row, _context):
    return {"fixture": "lowered"}


def _fixture_preparation(_request):
    return None


def _fixture_analysis(_run_spec, _root, _inputs, _execution_context):
    return None


def _fixture_evaluation(_run_spec, _root, _states_path, _execution_context):
    return None


def _fixture_evaluation_batch(_items, _execution_context):
    return ()


def _fixture_compact(_input):
    return EvaluationBatchFragment({}, "fixture.batch", "v1", "compact")


def _fixture_merge(_input):
    from feedbax.plugins import EvaluationBatchMergeState

    return EvaluationBatchMergeState({}, "fixture.batch", "v1")


def _fixture_finalize(_input):
    return EvaluationBatchFragment({}, "fixture.batch", "v1", "final")


def _fixture_union(_input):
    return EvaluationBatchFragment({}, "fixture.union", "v1", "union")


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
    context.registry(TRAINING_METHODS).register_descriptor(_fixture_method_descriptor())
    context.registry(ROW_LOWERERS).register(
        TrainingRowLowererRegistration(
            authored_schema_id="feedbax_external_conformance.training",
            authored_schema_version="v1",
            lowerer_id="feedbax_external_conformance.lowerer",
            lowerer_version="v1",
            implementation_sha256="0" * 64,
            lower=_fixture_lowerer,
            owner="feedbax-external-conformance",
        )
    )
    context.registry(EXECUTION_PREPARATIONS).register(
        ExecutionPreparationRegistration(
            method_ref=_FIXTURE_METHOD_REF,
            provider=_fixture_preparation,
            owner="feedbax-external-conformance",
        )
    )
    context.registry(ANALYSIS_RECIPES).register(_FIXTURE_ANALYSIS_TYPE, _fixture_analysis)
    context.registry(EVALUATION_RECIPES).register(
        _FIXTURE_EVALUATION_TYPE,
        _fixture_evaluation,
        batch_recipe=_fixture_evaluation_batch,
    )
    context.registry(EVALUATION_BATCH_CONSUMERS).register(
        _FIXTURE_CONSUMER_ID,
        _FIXTURE_CONSUMER_VERSION,
        compact=_fixture_compact,
        merge=_fixture_merge,
        finalize=_fixture_finalize,
    )
    context.registry(EVALUATION_PRODUCT_UNION_FINALIZERS).register(
        _FIXTURE_CONSUMER_ID,
        _FIXTURE_CONSUMER_VERSION,
        _fixture_union,
    )


def _register_dependent(context: RegistrationContext) -> None:
    context.registry(FIXTURE_RECORDS).register("dependent")


FOUNDATION_PLUGIN_REGISTRATION = PluginRegistration(
    declaration=PluginDeclaration(
        plugin_id=_FOUNDATION_PLUGIN_ID,
        version="1",
        downstream_protocol_version=DOWNSTREAM_PROTOCOL_CURRENT,
        families=(
            FamilyRequirement(COMPONENTS.family),
            FamilyRequirement(FIXTURE_RECORDS.family),
            FamilyRequirement(DRIVERS.family),
            FamilyRequirement(TRAINING_METHODS.family),
            FamilyRequirement(ROW_LOWERERS.family),
            FamilyRequirement(EXECUTION_PREPARATIONS.family),
            FamilyRequirement(ANALYSIS_RECIPES.family),
            FamilyRequirement(EVALUATION_RECIPES.family),
            FamilyRequirement(EVALUATION_BATCH_CONSUMERS.family),
            FamilyRequirement(EVALUATION_PRODUCT_UNION_FINALIZERS.family),
        ),
    ),
    register=_register_foundation,
)

DEPENDENT_PLUGIN_REGISTRATION = PluginRegistration(
    declaration=PluginDeclaration(
        plugin_id=_DEPENDENT_PLUGIN_ID,
        version="1",
        downstream_protocol_version=DOWNSTREAM_PROTOCOL_CURRENT,
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
