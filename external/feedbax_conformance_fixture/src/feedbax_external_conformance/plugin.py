"""Typed registrations exported through the single ``feedbax.plugins`` group."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict
from typing import Literal, NamedTuple

from feedbax import Component
from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.plugins import (
    ANALYSIS_RECIPES,
    COMPONENTS,
    DOWNSTREAM_PROTOCOL_CURRENT,
    DRIVERS,
    EVALUATION_BATCH_CONSUMERS,
    EVALUATION_PRODUCT_UNION_FINALIZERS,
    EVALUATION_RECIPES,
    FamilyRequirement,
    PluginDeclaration,
    PluginDependency,
    PluginRegistration,
    RegistrationContext,
    RegistryFamilyRegistration,
    TRAINING_PROGRAMS,
    EvaluationBatchFragment,
    EvaluationAuthoringSchema,
)
from feedbax.contracts.training import (
    MethodExtensionsSpec,
    TrainingConfig,
    TrainingMethodAuthoringContribution,
    TrainingMethodAuthoringHook,
    DeclaredTrainingProgram,
    declare_training_program,
    standard_supervised_method_contract,
    standard_supervised_training_program,
)
from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.analysis.specs import AnalysisRecipeResult
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace
from feedbax.contracts.run_matrix import RowLowererIdentity, TrainingRowLoweringResult
from feedbax.training.preparation import ExecutionPreparationResult
from feedbax.training.row_lowering import (
    TrainingRowLowererRegistration,
    training_row_lowerer_implementation_sha256,
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

from .family import EXTERNAL_DYNAMIC_COMPONENT, FIXTURE_RECORDS, FixtureRecordRegistry


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


class FixtureTrainingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    gain: int
    states_custody: Literal["cache", "durable"] = "cache"


class FixtureEvaluationStates(NamedTuple):
    value: object


class FixtureAnalysis(AbstractAnalysis):
    """Small installed-package analysis used by the public lifecycle case."""

    def compute(self, data: AnalysisInputData, **_kwargs):
        return {"total": float(jnp.asarray(data.states.value).sum())}

    def make_figs(self, _data: AnalysisInputData, *, result, **_kwargs):
        return None


def _fixture_training_program() -> DeclaredTrainingProgram:
    baseline = standard_supervised_training_program()
    return declare_training_program(
        method_ref=_FIXTURE_METHOD_REF,
        payload_schema_id="feedbax_external_conformance.training",
        payload_schema_version="feedbax_external_conformance.training.v1",
        payload_model=FixtureTrainingPayload,
        contract_compiler=lambda _payload: standard_supervised_method_contract().model_copy(
            update={
                "method_ref": _FIXTURE_METHOD_REF,
                "method_payload_schema_version": "feedbax_external_conformance.training.v1",
            }
        ),
        owner="feedbax-external-conformance",
        package="feedbax-external-conformance",
        update_kernels_factory=baseline.update_kernels_factory,
        guard_predicates_factory=baseline.guard_predicates_factory,
        preparation_provider=_fixture_preparation,
        row_lowerers=(TrainingRowLowererRegistration(
            authored_schema_id="feedbax_external_conformance.training",
            authored_schema_version="v1",
            lowerer_id="feedbax_external_conformance.lowerer",
            lowerer_version="v1",
            implementation_sha256=FIXTURE_LOWERER_IMPLEMENTATION_SHA256,
            lower=_fixture_lowerer,
            owner="feedbax-external-conformance",
        ),),
        authoring_hook=TrainingMethodAuthoringHook(
            lowerer_id="feedbax_external_conformance.authoring",
            lowerer_version="v1",
            compile=lambda payload: TrainingMethodAuthoringContribution(
                training_config=TrainingConfig(n_batches=2, batch_size=1, hidden_dim=1),
                checkpoint_interval=1,
                progress_interval=1,
                method_extensions=MethodExtensionsSpec(metadata={"gain": payload.gain}),
            ),
            graph=lambda _payload: {
                "inline": {
                    "nodes": {"gain": {"type": "Gain", "params": {"gain": 1.0}}},
                    "wires": [],
                    "input_ports": ["input"],
                    "output_ports": ["output"],
                    "input_bindings": {"input": ("gain", "input")},
                    "output_bindings": {"output": ("gain", "output")},
                }
            },
            task=lambda _payload: {"type": "fixture", "params": {"n_steps": 1}},
            objective=lambda _payload: {
                "loss": {"type": "target_state", "label": "fixture", "selector": "port:gain.output"}
            },
            domain=lambda payload: {"fixture_gain": payload.gain},
        ),
    )


def _fixture_lowerer(row, _context):
    return TrainingRowLoweringResult(
        execution_payload={"fixture_lowered_gain": row.payload["gain"]},
        lowerer_identities=[
            RowLowererIdentity(
                lowerer_id="feedbax_external_conformance.lowerer", lowerer_version="v1"
            )
        ],
    )


FIXTURE_LOWERER_IMPLEMENTATION_SHA256 = training_row_lowerer_implementation_sha256(_fixture_lowerer)


def _fixture_preparation(_request):
    return ExecutionPreparationResult(initial_slots={}, kernel_context={"fixture": True})


def _fixture_analysis(_run_spec, _root, inputs, _execution_context):
    return AnalysisRecipeResult(
        analyses={"fixture": FixtureAnalysis(variant="fixture")},
        data=AnalysisInputData(
            models={},
            tasks={},
            states=inputs[0].states,
            hps={"fixture": TreeNamespace(task=TreeNamespace(eval_n=1))},
            extras={},
        ),
    )


def _fixture_evaluation(run_spec, _root, _states_path, _execution_context):
    gain = int(run_spec.params["gain"])
    states = FixtureEvaluationStates(jnp.asarray([gain, gain + 1], dtype=jnp.int32))
    return EvaluationRecipeResult(
        states=states,
        summary_metrics={"fixture": float(states.value.sum())},
        metadata={"states_schema": "feedbax_external_conformance.states.v1"},
    )


def _fixture_evaluation_batch(items, _execution_context):
    return tuple(_fixture_evaluation(item.spec, None, None, None) for item in items)


def _fixture_evaluation_structure(_manifest):
    return jax.tree.structure(FixtureEvaluationStates(jnp.asarray([0, 0], dtype=jnp.int32)))


def _fixture_compact(value):
    return EvaluationBatchFragment(
        {"rows": list(value.batch.ordered_row_ids)},
        "feedbax_external_conformance.batch",
        "feedbax_external_conformance.batch.v1",
        "compact",
    )


def _fixture_merge(value):
    from feedbax.plugins import EvaluationBatchMergeState

    prior = [] if value.prior_merge_state is None else value.prior_merge_state["rows"]
    return EvaluationBatchMergeState(
        {"rows": [*prior, *value.fragment["rows"]]},
        "feedbax_external_conformance.merge",
        "feedbax_external_conformance.merge.v1",
    )


def _fixture_finalize(value):
    return EvaluationBatchFragment(
        value.terminal_merge_state,
        "feedbax_external_conformance.batch",
        "feedbax_external_conformance.batch.v1",
        "compact",
    )


def _fixture_union(value):
    return EvaluationBatchFragment(
        {
            "cohorts": [
                {
                    "cohort_key": source.cohort_key,
                    "matrix_intent_hash": source.matrix_intent_hash,
                    "rows": source.payload["rows"],
                }
                for source in value.sources
            ]
        },
        value.declaration.output_schema_id,
        value.declaration.output_schema_version,
        value.declaration.output_role,
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
    context.registry(TRAINING_PROGRAMS).register_program(_fixture_training_program())
    context.registry(ANALYSIS_RECIPES).register(
        _FIXTURE_ANALYSIS_TYPE,
        _fixture_analysis,
        evaluation_states_structure=_fixture_evaluation_structure,
    )
    context.registry(EVALUATION_RECIPES).register(
        _FIXTURE_EVALUATION_TYPE,
        _fixture_evaluation,
        batch_recipe=_fixture_evaluation_batch,
    )
    context.registry(EVALUATION_RECIPES).register_authoring_schema(
        _FIXTURE_EVALUATION_TYPE,
        EvaluationAuthoringSchema(
            schema_id="feedbax_external_conformance.evaluation_params",
            schema_version="feedbax_external_conformance.evaluation_params.v1",
            params_model=FixtureTrainingPayload,
            axis_profiles=(
                {"fixture": ("one",)},
                {"fixture": ("left", "right")},
            ),
        ),
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
            FamilyRequirement(TRAINING_PROGRAMS.family),
            FamilyRequirement(ANALYSIS_RECIPES.family),
            FamilyRequirement(EVALUATION_RECIPES.family),
            FamilyRequirement(EVALUATION_BATCH_CONSUMERS.family),
            FamilyRequirement(EVALUATION_PRODUCT_UNION_FINALIZERS.family),
        ),
    ),
    register=_register_foundation,
    registry_families=(
        RegistryFamilyRegistration(
            key=FIXTURE_RECORDS,
            factory=FixtureRecordRegistry,
            seal=lambda registry: registry.seal(),
        ),
    ),
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
    "FIXTURE_LOWERER_IMPLEMENTATION_SHA256",
]
