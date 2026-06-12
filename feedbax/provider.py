"""Feedbax provider capability, registry, and validation contract."""

from __future__ import annotations

import pkgutil
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

import feedbax.analysis as analysis_pkg
from feedbax.artifact_schema import ArrayRecord, ArrayStorePayload
from feedbax.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArrayStoreRef,
    ArtifactRef,
    ArtifactMigrationRecord,
    ArtifactValidationRecord,
    EvaluationRunManifest,
    EvaluationRunSpec,
    GraphSpecManifest,
    ModelArtifactManifest,
    PROVIDER_VERSION,
    ReportManifest,
    ReportSpec,
    SCHEMA_VERSION,
    TrainingRunManifest,
    TrainingRunSetManifest,
    feedbax_version,
    utc_now,
)
from feedbax.objective_spec import objective_schema_models
from feedbax.execution import ExecutionPlan, ExecutionSpec, LocalExecutionResult
from feedbax.studio_protocol import parse_positive_n_steps, task_n_steps_values
from feedbax.studio_execution import (
    StudioPipelineMaterializationRequest,
    StudioPipelineMaterializationResult,
    StudioTrainingLocalRunRequest,
    StudioTrainingLocalRunResult,
    StudioTrainingExecutionPreparation,
    StudioTrainingExecutionRequest,
)
from feedbax.studio_schema import (
    PortSchema,
    SchemaValidationIssue,
    SelectorTargetSchema,
    RuntimeIntrospectionOptions,
    RuntimeIntrospectionResult,
    RuntimeSampleLeafSchema,
    StudioSchemaEnumerationRequest,
    StudioSchemaRegistry,
    TaskDataSchema,
    ValueSchema,
    validate_graph_connection_schema,
)
from feedbax.contracts.graph import AnalysisInputRequirement, GraphSpec
from feedbax.contracts.training import LossTermSpec, TaskSpec, TrainingSpec

TASK_COMPONENT_TYPES = {"ReachingTask", "SimpleReaches", "DelayedReaches", "Stabilization"}


class ProviderModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderHealth(ProviderModel):
    status: Literal["ok", "degraded", "error"] = "ok"
    provider: str = "feedbax"
    provider_version: str = PROVIDER_VERSION
    feedbax_version: str = Field(default_factory=feedbax_version)
    checked_at: datetime = Field(default_factory=utc_now)


class CapabilitySpec(ProviderModel):
    input_schema: Optional[str] = None
    output_schema: Optional[str] = None
    requires_review: bool = False
    description: str = ""
    transports: list[str] = Field(default_factory=lambda: ["python", "cli", "http"])
    action: Optional[
        Literal["open", "validate", "execute", "import", "export", "publish", "inspect", "handoff"]
    ] = None
    compatibility_predicates: list[str] = Field(default_factory=list)
    mutates_state: bool = False
    may_launch_compute: bool = False
    artifact_roles: list[str] = Field(default_factory=list)
    selected_node_kinds: list[str] = Field(default_factory=list)
    custody_expectations: list[str] = Field(default_factory=list)


class MandibleArtifactMapping(ProviderModel):
    source_field: str
    role: str
    mandible_artifact_kind: str = "artifact"
    preserves_local_uri: bool = True
    optional_artifact_id: bool = True
    custody_hint: str = "feedbax-local-with-optional-mandible-enrichment"
    description: str = ""


class MandibleManifestMapping(ProviderModel):
    manifest_kind: str
    subject_node_type: str
    subject_id_field: str = "id"
    title_fields: list[str] = Field(default_factory=list)
    status_field: str = "status"
    artifact_fields: list[MandibleArtifactMapping] = Field(default_factory=list)
    spec_fields: list[str] = Field(default_factory=list)
    parent_ref_fields: list[str] = Field(default_factory=list)
    issue_provenance_field: str = "provenance.issues"
    source_provenance_fields: list[str] = Field(
        default_factory=lambda: [
            "provenance.source_repo",
            "provenance.source_branch",
            "provenance.source_commit",
            "provenance.dirty",
            "provenance.entrypoint",
        ]
    )
    opaque_domain_fields: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    related_issue_refs: list[str] = Field(default_factory=list)
    description: str = ""


class ProviderManifest(ProviderModel):
    provider: str = "feedbax"
    kind: Literal["experiment_provider"] = "experiment_provider"
    version: str = PROVIDER_VERSION
    feedbax_version: str = Field(default_factory=feedbax_version)
    schema_version: str = SCHEMA_VERSION
    health: dict[str, str] = Field(
        default_factory=lambda: {"check": "python:feedbax.provider.health"}
    )
    capabilities: dict[str, CapabilitySpec]
    artifact_roles: list[str]
    schemas: dict[str, dict[str, Any]]
    mandible_manifest_mappings: dict[str, MandibleManifestMapping] = Field(default_factory=dict)
    entry_points: dict[str, str] = Field(default_factory=dict)


class RegistryEntry(ProviderModel):
    type_id: str
    name: str
    category: str
    description: str = ""
    provider: str = "feedbax"
    package: Optional[str] = None
    version: Optional[str] = None
    import_path: Optional[str] = None
    input_ports: list[str] = Field(default_factory=list)
    output_ports: list[str] = Field(default_factory=list)
    parameter_schema: Any = Field(default_factory=list)
    artifact_roles: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegistrySnapshot(ProviderModel):
    kind: str
    schema_version: str = SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=utc_now)
    entries: list[RegistryEntry]


class ValidationIssue(ProviderModel):
    type: str
    message: str
    location: Optional[dict[str, str]] = None


class ProviderValidationResult(ProviderModel):
    valid: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)


def health() -> ProviderHealth:
    return ProviderHealth()


def _schema_models() -> dict[str, type[BaseModel]]:
    return {
        "GraphSpec": GraphSpec,
        "AnalysisInputRequirement": AnalysisInputRequirement,
        "TrainingSpec": TrainingSpec,
        "TaskSpec": TaskSpec,
        "LossTermSpec": LossTermSpec,
        "EvaluationRunSpec": EvaluationRunSpec,
        "AnalysisRunSpec": AnalysisRunSpec,
        "ReportSpec": ReportSpec,
        "ArtifactRef": ArtifactRef,
        "ArrayRecord": ArrayRecord,
        "ArrayStorePayload": ArrayStorePayload,
        "ArrayStoreRef": ArrayStoreRef,
        "ArtifactValidationRecord": ArtifactValidationRecord,
        "ArtifactMigrationRecord": ArtifactMigrationRecord,
        "ExecutionSpec": ExecutionSpec,
        "ExecutionPlan": ExecutionPlan,
        "LocalExecutionResult": LocalExecutionResult,
        "StudioTrainingExecutionRequest": StudioTrainingExecutionRequest,
        "StudioTrainingExecutionPreparation": StudioTrainingExecutionPreparation,
        "StudioTrainingLocalRunRequest": StudioTrainingLocalRunRequest,
        "StudioTrainingLocalRunResult": StudioTrainingLocalRunResult,
        "StudioPipelineMaterializationRequest": StudioPipelineMaterializationRequest,
        "StudioPipelineMaterializationResult": StudioPipelineMaterializationResult,
        "ValueSchema": ValueSchema,
        "PortSchema": PortSchema,
        "TaskDataSchema": TaskDataSchema,
        "SelectorTargetSchema": SelectorTargetSchema,
        "SchemaValidationIssue": SchemaValidationIssue,
        "RuntimeIntrospectionOptions": RuntimeIntrospectionOptions,
        "RuntimeSampleLeafSchema": RuntimeSampleLeafSchema,
        "RuntimeIntrospectionResult": RuntimeIntrospectionResult,
        "StudioSchemaRegistry": StudioSchemaRegistry,
        "StudioSchemaEnumerationRequest": StudioSchemaEnumerationRequest,
        "GraphSpecManifest": GraphSpecManifest,
        "ModelArtifactManifest": ModelArtifactManifest,
        "TrainingRunSetManifest": TrainingRunSetManifest,
        "TrainingRunManifest": TrainingRunManifest,
        "EvaluationRunManifest": EvaluationRunManifest,
        "AnalysisRunManifest": AnalysisRunManifest,
        "ReportManifest": ReportManifest,
        "CapabilitySpec": CapabilitySpec,
        "MandibleArtifactMapping": MandibleArtifactMapping,
        "MandibleManifestMapping": MandibleManifestMapping,
        "ProviderManifest": ProviderManifest,
        **objective_schema_models(),
    }


def _schemas() -> dict[str, dict[str, Any]]:
    return {name: model.model_json_schema() for name, model in _schema_models().items()}


def _mandible_manifest_mappings() -> dict[str, MandibleManifestMapping]:
    common_actions = ["inspect_manifest", "export_manifest", "handoff_artifacts"]
    run_actions = [
        "inspect_manifest",
        "open_in_feedbax_studio",
        "export_manifest",
        "handoff_artifacts",
    ]
    return {
        "GraphSpecManifest": MandibleManifestMapping(
            manifest_kind="GraphSpecManifest",
            subject_node_type="feedbax.graph_spec",
            title_fields=["graph_spec.ref", "id"],
            spec_fields=["graph_spec"],
            parent_ref_fields=["provenance.parents"],
            opaque_domain_fields=["graph_spec.inline", "metadata"],
            actions=common_actions + ["validate_graph_spec"],
            related_issue_refs=["51832b9", "c6c6da0", "f68cf66"],
            description=(
                "Mandible should treat the graph payload as Feedbax-owned domain "
                "detail and use Feedbax validation for semantics."
            ),
        ),
        "ModelArtifactManifest": MandibleManifestMapping(
            manifest_kind="ModelArtifactManifest",
            subject_node_type="feedbax.model_artifact",
            title_fields=["graph_spec.ref", "id"],
            artifact_fields=[
                MandibleArtifactMapping(
                    source_field="parameter_store",
                    role="model_parameters",
                    mandible_artifact_kind="array_store",
                    description="Role-addressed trainable/non-structural parameter arrays.",
                ),
                MandibleArtifactMapping(
                    source_field="state_store",
                    role="model_state",
                    mandible_artifact_kind="array_store",
                    description="Optional role-addressed state arrays.",
                ),
                MandibleArtifactMapping(
                    source_field="optimizer_store",
                    role="optimizer_state",
                    mandible_artifact_kind="array_store",
                    description="Optional optimizer/checkpoint state arrays.",
                ),
                MandibleArtifactMapping(
                    source_field="artifacts[]",
                    role="manifest_artifact",
                    description="Additional files referenced by the model artifact manifest.",
                ),
            ],
            spec_fields=["graph_spec"],
            parent_ref_fields=["graph_spec", "provenance.parents"],
            opaque_domain_fields=[
                "parameter_store.roles",
                "state_store.roles",
                "optimizer_store.roles",
                "validation_records",
                "migration_records",
                "metadata",
            ],
            actions=common_actions + ["materialize_model_artifact"],
            related_issue_refs=["51832b9", "63c798f", "mandible/e967b9", "mandible/2322726"],
            description=(
                "Mandible may index array stores and custody hints, but Feedbax owns "
                "role semantics and materialization validation."
            ),
        ),
        "TrainingRunSetManifest": MandibleManifestMapping(
            manifest_kind="TrainingRunSetManifest",
            subject_node_type="feedbax.training_run_set",
            title_fields=["name", "id"],
            spec_fields=["graph_spec"],
            parent_ref_fields=["graph_spec", "run_ids", "provenance.parents"],
            opaque_domain_fields=["tags", "metadata"],
            actions=run_actions,
            related_issue_refs=["51832b9", "e33f487"],
        ),
        "TrainingRunManifest": MandibleManifestMapping(
            manifest_kind="TrainingRunManifest",
            subject_node_type="feedbax.training_run",
            title_fields=["job_id", "id"],
            artifact_fields=[
                MandibleArtifactMapping(
                    source_field="artifacts[]",
                    role="training_artifact",
                    description=(
                        "Checkpoint, history, retained-observable, log, or local "
                        "execution artifacts."
                    ),
                ),
            ],
            spec_fields=["graph_spec", "training_spec", "task_spec", "task_binding_spec"],
            parent_ref_fields=["graph_spec", "run_set_id", "provenance.parents"],
            opaque_domain_fields=["overrides", "summary_metrics", "metadata"],
            actions=run_actions + ["start_evaluation_run", "run_analysis"],
            related_issue_refs=["51832b9", "e33f487", "63c798f"],
            description=(
                "Mandible should link issue/source provenance and artifacts without "
                "interpreting Feedbax training/task specs directly."
            ),
        ),
        "EvaluationRunManifest": MandibleManifestMapping(
            manifest_kind="EvaluationRunManifest",
            subject_node_type="feedbax.evaluation_run",
            title_fields=["evaluation_spec.inline.evaluation_type", "id"],
            artifact_fields=[
                MandibleArtifactMapping(
                    source_field="artifacts[]",
                    role="evaluation_artifact",
                ),
            ],
            spec_fields=["evaluation_spec"],
            parent_ref_fields=["input_training_runs", "provenance.parents"],
            opaque_domain_fields=["evaluation_spec.inline", "summary_metrics", "metadata"],
            actions=run_actions + ["run_analysis"],
            related_issue_refs=["51832b9", "63c798f"],
        ),
        "AnalysisRunManifest": MandibleManifestMapping(
            manifest_kind="AnalysisRunManifest",
            subject_node_type="feedbax.analysis_run",
            title_fields=["analysis_spec.inline.analysis_type", "id"],
            artifact_fields=[
                MandibleArtifactMapping(
                    source_field="artifacts[]",
                    role="analysis_artifact",
                ),
            ],
            spec_fields=["analysis_spec"],
            parent_ref_fields=["inputs", "provenance.parents"],
            opaque_domain_fields=["analysis_spec.inline", "summary_metrics", "metadata"],
            actions=run_actions + ["publish_report"],
            related_issue_refs=["51832b9", "63c798f"],
        ),
        "ReportManifest": MandibleManifestMapping(
            manifest_kind="ReportManifest",
            subject_node_type="feedbax.report",
            title_fields=["report_spec.inline.report_type", "id"],
            artifact_fields=[
                MandibleArtifactMapping(
                    source_field="artifacts[]",
                    role="report_artifact",
                ),
            ],
            spec_fields=["report_spec"],
            parent_ref_fields=["inputs", "provenance.parents"],
            opaque_domain_fields=["report_spec.inline", "metadata"],
            actions=run_actions + ["publish_report"],
            related_issue_refs=["51832b9", "63c798f"],
        ),
    }


def provider_manifest() -> ProviderManifest:
    capabilities = {
        "health": CapabilitySpec(output_schema="ProviderHealth", action="inspect"),
        "provider_manifest": CapabilitySpec(output_schema="ProviderManifest", action="inspect"),
        "validate_graph_spec": CapabilitySpec(
            input_schema="GraphSpec",
            output_schema="ProviderValidationResult",
            action="validate",
            compatibility_predicates=["selected node payload is GraphSpec-compatible"],
            selected_node_kinds=["feedbax.graph_spec"],
        ),
        "validate_training_spec": CapabilitySpec(
            input_schema="TrainingSpec",
            output_schema="ProviderValidationResult",
            action="validate",
            compatibility_predicates=["selected node has Feedbax training spec payload"],
            selected_node_kinds=["feedbax.training_run", "feedbax.training_run_set"],
        ),
        "validate_task_spec": CapabilitySpec(
            input_schema="TaskSpec",
            output_schema="ProviderValidationResult",
            action="validate",
            compatibility_predicates=["selected node has Feedbax task spec payload"],
        ),
        "validate_evaluation_spec": CapabilitySpec(
            input_schema="EvaluationRunSpec",
            output_schema="ProviderValidationResult",
            action="validate",
            compatibility_predicates=["selected node has Feedbax evaluation spec payload"],
            selected_node_kinds=["feedbax.evaluation_run"],
        ),
        "validate_analysis_spec": CapabilitySpec(
            input_schema="AnalysisRunSpec",
            output_schema="ProviderValidationResult",
            action="validate",
            compatibility_predicates=["selected node has Feedbax analysis spec payload"],
            selected_node_kinds=["feedbax.analysis_run"],
        ),
        "start_training_run": CapabilitySpec(
            input_schema="TrainingSpec",
            output_schema="TrainingRunManifest",
            requires_review=True,
            description="Start a local or configured worker training run.",
            action="execute",
            compatibility_predicates=["graph and training specs validate through Feedbax"],
            mutates_state=True,
            may_launch_compute=True,
            artifact_roles=["training_checkpoint", "training_history", "execution_log"],
            selected_node_kinds=["feedbax.graph_spec", "feedbax.training_run_set"],
            custody_expectations=[
                "Feedbax writes local manifests/artifacts first.",
                "Mandible artifact IDs are optional enrichment after handoff.",
            ],
        ),
        "prepare_execution_plan": CapabilitySpec(
            input_schema="ExecutionSpec",
            output_schema="ExecutionPlan",
            description="Prepare a deterministic local, SSH, RunPod, or Modal execution plan.",
            action="validate",
            compatibility_predicates=["execution spec is provider-owned and backend-supported"],
            artifact_roles=["execution_plan"],
        ),
        "run_local_execution": CapabilitySpec(
            input_schema="ExecutionSpec",
            output_schema="LocalExecutionResult",
            requires_review=True,
            description="Run an explicitly local execution and emit a durable manifest.",
            action="execute",
            compatibility_predicates=["execution spec backend is local"],
            mutates_state=True,
            may_launch_compute=True,
            artifact_roles=["manifest", "execution_log"],
            custody_expectations=[
                "Local outputs remain usable without Mandible.",
                "Mandible may ingest emitted manifest and artifacts later.",
            ],
        ),
        "prepare_studio_training_execution": CapabilitySpec(
            input_schema="StudioTrainingExecutionRequest",
            output_schema="StudioTrainingExecutionPreparation",
            description="Lower a Studio train-stage scenario into a provider execution plan.",
            transports=["python", "http"],
            action="validate",
            compatibility_predicates=["selected Studio train stage has graph and training specs"],
            artifact_roles=["execution_plan"],
            selected_node_kinds=["feedbax.studio_stage.train"],
        ),
        "run_studio_training_local_execution": CapabilitySpec(
            input_schema="StudioTrainingLocalRunRequest",
            output_schema="StudioTrainingLocalRunResult",
            requires_review=True,
            description=(
                "Run a Studio train-stage scenario through the local provider execution "
                "boundary and return updated workspace lineage refs."
            ),
            transports=["python", "http"],
            action="execute",
            compatibility_predicates=["prepared Studio train stage validates through Feedbax"],
            mutates_state=True,
            may_launch_compute=True,
            artifact_roles=["training_checkpoint", "training_history", "manifest"],
            selected_node_kinds=["feedbax.studio_stage.train"],
            custody_expectations=[
                "Studio stores Feedbax-local refs in the workspace.",
                "Mandible custody hints must not be required to reopen locally.",
            ],
        ),
        "materialize_studio_pipeline": CapabilitySpec(
            input_schema="StudioPipelineMaterializationRequest",
            output_schema="StudioPipelineMaterializationResult",
            description=(
                "Materialize Studio eval, analysis, and report stages from upstream "
                "workspace collections and return updated lineage refs."
            ),
            transports=["python", "http"],
            action="execute",
            compatibility_predicates=[
                "selected Studio stage has compatible upstream manifest collection"
            ],
            mutates_state=True,
            artifact_roles=["evaluation_result", "analysis_table", "report", "manifest"],
            selected_node_kinds=[
                "feedbax.studio_stage.eval",
                "feedbax.studio_stage.analysis",
                "feedbax.studio_stage.report",
            ],
        ),
        "enumerate_studio_schemas": CapabilitySpec(
            input_schema="StudioSchemaEnumerationRequest",
            output_schema="StudioSchemaRegistry",
            description=(
                "Enumerate static Studio graph port, task data, selector target, "
                "and validation schemas without JAX compilation or training."
            ),
            transports=["python", "http"],
            action="inspect",
            compatibility_predicates=["graph/spec payload is optional"],
        ),
        "list_components": CapabilitySpec(
            output_schema="ComponentRegistrySnapshot",
            action="inspect",
        ),
        "list_tasks": CapabilitySpec(output_schema="TaskRegistrySnapshot", action="inspect"),
        "list_losses": CapabilitySpec(output_schema="LossRegistrySnapshot", action="inspect"),
        "list_protocols": CapabilitySpec(
            output_schema="ProtocolRegistrySnapshot",
            action="inspect",
        ),
        "list_analyses": CapabilitySpec(
            output_schema="AnalysisRegistrySnapshot",
            action="inspect",
        ),
    }
    return ProviderManifest(
        capabilities=capabilities,
        artifact_roles=[
            "training_checkpoint",
            "training_history",
            "model_artifact_manifest",
            "model_parameters",
            "model_state",
            "optimizer_state",
            "array_store",
            "retention_plan",
            "retained_observables",
            "trajectory_dataset",
            "evaluation_result",
            "analysis_table",
            "figure",
            "report",
            "manifest",
            "execution_plan",
            "execution_log",
        ],
        schemas=_schemas(),
        mandible_manifest_mappings=_mandible_manifest_mappings(),
        entry_points={
            "python": "feedbax.provider:provider_manifest",
            "cli": "feedbax-provider manifest",
            "http": "/api/provider/manifest",
        },
    )


def component_registry_snapshot() -> RegistrySnapshot:
    from feedbax.component_registry import ComponentRegistry

    registry = ComponentRegistry()
    entries: list[RegistryEntry] = []
    for definition in registry.list_all():
        entries.append(
            RegistryEntry(
                type_id=f"feedbax.component.{definition.name}",
                name=definition.name,
                category=definition.category,
                description=definition.description,
                version=feedbax_version(),
                input_ports=definition.input_ports,
                output_ports=definition.output_ports,
                parameter_schema=[
                    schema.model_dump(mode="json", exclude_none=True)
                    for schema in definition.param_schema
                ],
                metadata={
                    "icon": definition.icon,
                    "default_params": definition.default_params,
                    "is_composite": definition.is_composite,
                    "has_template_graph": definition.template_graph is not None,
                    "provenance": definition.provenance,
                    "port_types": (
                        definition.port_types.model_dump(mode="json", exclude_none=True)
                        if definition.port_types is not None
                        else None
                    ),
                },
            )
        )
    return RegistrySnapshot(kind="components", entries=entries)


def task_registry_snapshot() -> RegistrySnapshot:
    task_types = [
        ("feedbax.task.ReachingTask", "ReachingTask", "Current Studio reaching task spec."),
        ("feedbax.task.SimpleReaches", "SimpleReaches", "Built-in reaching task."),
        ("feedbax.task.DelayedReaches", "DelayedReaches", "Built-in delayed reaching task."),
        ("feedbax.task.Stabilization", "Stabilization", "Built-in stabilization task."),
    ]
    return RegistrySnapshot(
        kind="tasks",
        entries=[
            RegistryEntry(
                type_id=type_id,
                name=name,
                category="Task",
                description=description,
                import_path=type_id,
                parameter_schema=TaskSpec.model_json_schema().get("properties", {}),
            )
            for type_id, name, description in task_types
        ],
    )


def loss_registry_snapshot() -> RegistrySnapshot:
    from feedbax.loss_service import NORM_FUNCTIONS

    return RegistrySnapshot(
        kind="losses",
        entries=[
            RegistryEntry(
                type_id="feedbax.loss.LossTermSpec",
                name="LossTermSpec",
                category="Loss",
                description="Structured Studio loss term with optional recursive children.",
                parameter_schema=LossTermSpec.model_json_schema().get("properties", {}),
                metadata={"norm_functions": NORM_FUNCTIONS},
            )
        ],
    )


def analysis_registry_snapshot() -> RegistrySnapshot:
    entries: list[RegistryEntry] = []
    for module_info in pkgutil.iter_modules(analysis_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        entries.append(
            RegistryEntry(
                type_id=f"feedbax.analysis.{module_info.name}",
                name=module_info.name,
                category="Analysis module",
                import_path=f"feedbax.analysis.{module_info.name}",
                description="Importable Feedbax analysis module.",
            )
        )
    return RegistrySnapshot(kind="analyses", entries=entries)


def protocol_registry_snapshot() -> RegistrySnapshot:
    from feedbax.plugins import EXPERIMENT_REGISTRY

    entries: list[RegistryEntry] = []
    for package_name in EXPERIMENT_REGISTRY.get_package_names():
        metadata = EXPERIMENT_REGISTRY.get_package_metadata(package_name)
        entries.extend(
            [
                RegistryEntry(
                    type_id=f"{package_name}.training_modules",
                    name=f"{package_name} training modules",
                    category="Training protocol package",
                    package=package_name,
                    import_path=f"{metadata.package_module.__name__}.{metadata.training_module_root}",
                    metadata={"parts": metadata.parts},
                ),
                RegistryEntry(
                    type_id=f"{package_name}.analysis_modules",
                    name=f"{package_name} analysis modules",
                    category="Analysis protocol package",
                    package=package_name,
                    import_path=f"{metadata.package_module.__name__}.{metadata.analysis_module_root}",
                    metadata={"parts": metadata.parts},
                ),
            ]
        )
    return RegistrySnapshot(kind="protocols", entries=entries)


def registry_snapshot(kind: str) -> RegistrySnapshot:
    if kind == "components":
        return component_registry_snapshot()
    if kind == "tasks":
        return task_registry_snapshot()
    if kind == "losses":
        return loss_registry_snapshot()
    if kind == "analyses":
        return analysis_registry_snapshot()
    if kind == "protocols":
        return protocol_registry_snapshot()
    raise ValueError(f"Unknown registry kind: {kind!r}")


def _pydantic_errors(exc: PydanticValidationError) -> list[ValidationIssue]:
    return [
        ValidationIssue(
            type="schema_error",
            message=str(error.get("msg", "Invalid value")),
            location={"path": "/" + "/".join(str(part) for part in error.get("loc", ()))},
        )
        for error in exc.errors()
    ]


def _component_ports(node_type: str, node_ports: list[str], attr: str, registry: Any) -> list[str]:
    if node_ports:
        return node_ports
    meta = registry.get(node_type)
    if meta is None:
        return []
    return list(getattr(meta, attr))


def _schema_issues_to_provider(
    issues: list[SchemaValidationIssue],
) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    for issue in issues:
        converted = ValidationIssue(
            type=issue.type,
            message=issue.message,
            location=issue.location,
        )
        if issue.severity == "error":
            errors.append(converted)
        else:
            warnings.append(converted)
    return errors, warnings


def validate_graph_spec(payload: dict[str, Any] | GraphSpec) -> ProviderValidationResult:
    from feedbax.graph_normalization import normalize_graph_for_studio_authoring
    from feedbax.component_registry import ComponentRegistry

    try:
        parsed = payload if isinstance(payload, GraphSpec) else GraphSpec.model_validate(payload)
        spec = normalize_graph_for_studio_authoring(parsed)
    except PydanticValidationError as exc:
        errors = _pydantic_errors(exc)
        return ProviderValidationResult(valid=False, errors=errors)

    registry = ComponentRegistry()
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    def _validate_graph(graph: GraphSpec, prefix: str = "") -> None:
        for node_name, node_spec in graph.nodes.items():
            node_path = f"{prefix}/nodes/{node_name}"
            if node_spec.type in TASK_COMPONENT_TYPES:
                errors.append(
                    ValidationIssue(
                        type="task_node_not_allowed",
                        message=(
                            f"Task component {node_spec.type!r} must be stored on "
                            "StudioScenarioSpec.task_spec/task_binding_spec, not GraphSpec.nodes"
                        ),
                        location={"path": node_path},
                    )
                )
                continue
            meta = registry.get(node_spec.type)
            if meta is None:
                errors.append(
                    ValidationIssue(
                        type="unknown_component_type",
                        message=f"Unknown component type: {node_spec.type}",
                        location={"path": node_path},
                    )
                )
                continue
            for schema in meta.param_schema:
                if (
                    schema.required
                    and schema.name not in node_spec.params
                    and schema.default is None
                ):
                    errors.append(
                        ValidationIssue(
                            type="missing_required_param",
                            message=(
                                f"Component {node_name!r} is missing required "
                                f"parameter {schema.name!r}"
                            ),
                            location={"path": f"{node_path}/params/{schema.name}"},
                        )
                    )
            if node_spec.type in {"Subgraph", "Network"}:
                if not graph.subgraphs or node_name not in graph.subgraphs:
                    errors.append(
                        ValidationIssue(
                            type="missing_subgraph",
                            message=f"{node_spec.type} node {node_name!r} requires a subgraph",
                            location={"path": node_path},
                        )
                    )

        graph_errors, graph_warnings = _schema_issues_to_provider(
            validate_graph_connection_schema(graph, prefix)
        )
        errors.extend(graph_errors)
        warnings.extend(graph_warnings)

        if graph.subgraphs:
            for subgraph_node, subgraph in graph.subgraphs.items():
                if subgraph_node not in graph.nodes:
                    warnings.append(
                        ValidationIssue(
                            type="orphan_subgraph",
                            message=f"Subgraph {subgraph_node!r} has no matching node",
                            location={"path": f"{prefix}/subgraphs/{subgraph_node}"},
                        )
                    )
                _validate_graph(subgraph, f"{prefix}/subgraphs/{subgraph_node}")

    _validate_graph(spec)
    if spec.retained_observables:
        from feedbax.retained_observables import RetentionPlanError, lower_retention_plan

        try:
            lower_retention_plan(spec)
        except RetentionPlanError as exc:
            errors.append(
                ValidationIssue(
                    type="invalid_retained_observable",
                    message=str(exc),
                    location={"path": exc.path, "selector": exc.selector or ""},
                )
            )
    return ProviderValidationResult(valid=not errors, errors=errors, warnings=warnings)


def _validate_loss_term(term: LossTermSpec, path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if term.weight < 0:
        issues.append(
            ValidationIssue(
                type="invalid_loss_weight",
                message="Loss weight must be non-negative",
                location={"path": f"{path}/weight"},
            )
        )
    if term.time_agg is not None:
        if term.time_agg.mode == "range" and (
            term.time_agg.start is None or term.time_agg.end is None
        ):
            issues.append(
                ValidationIssue(
                    type="invalid_time_aggregation",
                    message="Range time aggregation requires start and end",
                    location={"path": f"{path}/time_agg"},
                )
            )
        if term.time_agg.mode == "custom" and not term.time_agg.time_idxs:
            issues.append(
                ValidationIssue(
                    type="invalid_time_aggregation",
                    message="Custom time aggregation requires time_idxs",
                    location={"path": f"{path}/time_agg"},
                )
            )
    if term.children:
        for child_name, child in term.children.items():
            issues.extend(_validate_loss_term(child, f"{path}/children/{child_name}"))
    return issues


def validate_training_spec(
    payload: dict[str, Any] | TrainingSpec,
    *,
    graph_spec: Optional[dict[str, Any] | GraphSpec] = None,
    task_spec: Optional[dict[str, Any] | TaskSpec] = None,
) -> ProviderValidationResult:
    try:
        spec = (
            payload if isinstance(payload, TrainingSpec) else TrainingSpec.model_validate(payload)
        )
    except PydanticValidationError as exc:
        errors = _pydantic_errors(exc)
        return ProviderValidationResult(valid=False, errors=errors)

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    if spec.n_batches <= 0:
        errors.append(
            ValidationIssue(
                type="invalid_n_batches",
                message="n_batches must be positive",
                location={"path": "/n_batches"},
            )
        )
    if spec.batch_size <= 0:
        errors.append(
            ValidationIssue(
                type="invalid_batch_size",
                message="batch_size must be positive",
                location={"path": "/batch_size"},
            )
        )
    optimizer_type = spec.optimizer.type.lower()
    if optimizer_type not in {"adamw", "adam", "sgd", "rmsprop"}:
        warnings.append(
            ValidationIssue(
                type="unknown_optimizer",
                message=f"Optimizer {spec.optimizer.type!r} is not a built-in worker optimizer",
                location={"path": "/optimizer/type"},
            )
        )
    errors.extend(_validate_loss_term(spec.loss, "/loss"))
    if graph_spec is not None:
        graph_result = validate_graph_spec(graph_spec)
        errors.extend(graph_result.errors)
        warnings.extend(graph_result.warnings)
        if graph_result.valid:
            from feedbax.retained_observables import RetentionPlanError, lower_retention_plan

            graph = (
                graph_spec
                if isinstance(graph_spec, GraphSpec)
                else GraphSpec.model_validate(graph_spec)
            )
            try:
                lower_retention_plan(graph, spec, task_spec=task_spec)
            except RetentionPlanError as exc:
                errors.append(
                    ValidationIssue(
                        type="loss_graph_mismatch",
                        message=str(exc),
                        location={"path": exc.path, "selector": exc.selector or ""},
                    )
                )
    return ProviderValidationResult(valid=not errors, errors=errors, warnings=warnings)


def validate_task_spec(payload: dict[str, Any] | TaskSpec) -> ProviderValidationResult:
    try:
        spec = payload if isinstance(payload, TaskSpec) else TaskSpec.model_validate(payload)
    except PydanticValidationError as exc:
        errors = _pydantic_errors(exc)
        return ProviderValidationResult(valid=False, errors=errors)

    allowed = {entry.name for entry in task_registry_snapshot().entries} | {
        entry.type_id for entry in task_registry_snapshot().entries
    }
    errors: list[ValidationIssue] = []
    if spec.type not in allowed:
        errors.append(
            ValidationIssue(
                type="unknown_task_type",
                message=f"Unknown task type: {spec.type}",
                location={"path": "/type"},
            )
        )
    errors.extend(_validate_task_n_steps(spec))
    if spec.type in {"DelayedReaches", "feedbax.task.DelayedReaches"}:
        errors.extend(_validate_delayed_reaches_task_params(spec.params))
    return ProviderValidationResult(valid=not errors, errors=errors)


def _validate_task_n_steps(spec: TaskSpec) -> list[ValidationIssue]:
    errors: list[ValidationIssue] = []
    parsed_values: list[tuple[str, int]] = []
    for path, value in task_n_steps_values(spec):
        parsed = parse_positive_n_steps(value)
        if parsed is None:
            errors.append(
                ValidationIssue(
                    type="invalid_task_n_steps",
                    message="Task step count must be a positive integer",
                    location={"path": path},
                )
            )
        else:
            parsed_values.append((path, parsed))

    distinct = {value for _path, value in parsed_values}
    if len(distinct) > 1:
        paths = ", ".join(f"{path}={value}" for path, value in parsed_values)
        errors.append(
            ValidationIssue(
                type="task_n_steps_mismatch",
                message=f"Task step-count declarations disagree: {paths}",
                location={"path": "/"},
            )
        )
    return errors


def _validate_delayed_reaches_task_params(params: dict[str, Any]) -> list[ValidationIssue]:
    """Validate compact DelayedReaches task params at the Studio boundary."""
    errors: list[ValidationIssue] = []
    dense_keys = [
        key for key in ("targets", "target_pos", "target_vel", "validation_trials") if key in params
    ]
    for key in dense_keys:
        errors.append(
            ValidationIssue(
                type="dense_task_trajectory_not_allowed",
                message=(
                    f"DelayedReaches params must store compact task parameters; "
                    f"remove dense trajectory field {key!r}"
                ),
                location={"path": f"/params/{key}"},
            )
        )

    epoch_len_ranges = params.get("epoch_len_ranges", [[5, 15], [10, 20]])
    if not isinstance(epoch_len_ranges, list):
        errors.append(
            ValidationIssue(
                type="invalid_epoch_len_ranges",
                message="DelayedReaches epoch_len_ranges must be a list of [min, max] pairs",
                location={"path": "/params/epoch_len_ranges"},
            )
        )
        return errors

    for index, item in enumerate(epoch_len_ranges):
        if not isinstance(item, list) or len(item) != 2:
            errors.append(
                ValidationIssue(
                    type="invalid_epoch_len_range",
                    message="Each DelayedReaches epoch_len_ranges entry must be [min, max]",
                    location={"path": f"/params/epoch_len_ranges/{index}"},
                )
            )
            continue
        try:
            lower = int(item[0])
            upper = int(item[1])
        except (TypeError, ValueError):
            errors.append(
                ValidationIssue(
                    type="invalid_epoch_len_range",
                    message="DelayedReaches epoch lengths must be integers",
                    location={"path": f"/params/epoch_len_ranges/{index}"},
                )
            )
            continue
        if lower < 0 or upper < lower:
            errors.append(
                ValidationIssue(
                    type="invalid_epoch_len_range",
                    message="DelayedReaches epoch length ranges must satisfy 0 <= min <= max",
                    location={"path": f"/params/epoch_len_ranges/{index}"},
                )
            )

    epoch_count = len(epoch_len_ranges) + 1
    for key in ("target_on_epochs", "hold_epochs", "move_epochs"):
        value = params.get(key, [])
        if not isinstance(value, list):
            errors.append(
                ValidationIssue(
                    type="invalid_epoch_index_list",
                    message=f"DelayedReaches {key} must be a list of epoch indexes",
                    location={"path": f"/params/{key}"},
                )
            )
            continue
        for index, item in enumerate(value):
            try:
                epoch_index = int(item)
            except (TypeError, ValueError):
                errors.append(
                    ValidationIssue(
                        type="invalid_epoch_index",
                        message=f"DelayedReaches {key} entries must be integer epoch indexes",
                        location={"path": f"/params/{key}/{index}"},
                    )
                )
                continue
            if epoch_index < 0 or epoch_index >= epoch_count:
                errors.append(
                    ValidationIssue(
                        type="invalid_epoch_index",
                        message=(
                            f"DelayedReaches {key} entry {epoch_index} is outside "
                            f"the configured epoch range [0, {epoch_count - 1}]"
                        ),
                        location={"path": f"/params/{key}/{index}"},
                    )
                )
    return errors


def validate_evaluation_spec(
    payload: dict[str, Any] | EvaluationRunSpec,
) -> ProviderValidationResult:
    try:
        spec = (
            payload
            if isinstance(payload, EvaluationRunSpec)
            else EvaluationRunSpec.model_validate(payload)
        )
    except PydanticValidationError as exc:
        errors = _pydantic_errors(exc)
        return ProviderValidationResult(valid=False, errors=errors)

    errors: list[ValidationIssue] = []
    if not spec.evaluation_type.strip():
        errors.append(
            ValidationIssue(
                type="missing_evaluation_type",
                message="evaluation_type must not be empty",
                location={"path": "/evaluation_type"},
            )
        )
    if not spec.training_run_ids and not spec.inputs:
        errors.append(
            ValidationIssue(
                type="missing_inputs",
                message="Evaluation specs require training_run_ids or input refs",
                location={"path": "/inputs"},
            )
        )
    return ProviderValidationResult(valid=not errors, errors=errors)


def validate_analysis_spec(
    payload: dict[str, Any] | AnalysisRunSpec,
    *,
    graph_spec: Optional[dict[str, Any] | GraphSpec] = None,
) -> ProviderValidationResult:
    try:
        spec = (
            payload
            if isinstance(payload, AnalysisRunSpec)
            else AnalysisRunSpec.model_validate(payload)
        )
    except PydanticValidationError as exc:
        errors = _pydantic_errors(exc)
        return ProviderValidationResult(valid=False, errors=errors)

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    if not spec.analysis_type.strip():
        errors.append(
            ValidationIssue(
                type="missing_analysis_type",
                message="analysis_type must not be empty",
                location={"path": "/analysis_type"},
            )
        )
    known = {entry.type_id for entry in analysis_registry_snapshot().entries} | {
        entry.name for entry in analysis_registry_snapshot().entries
    }
    if spec.analysis_type not in known:
        warnings.append(
            ValidationIssue(
                type="unregistered_analysis",
                message=(
                    f"Analysis type {spec.analysis_type!r} is not in the built-in "
                    "analysis snapshot; project registries may still provide it."
                ),
                location={"path": "/analysis_type"},
            )
        )
    if not spec.inputs:
        errors.append(
            ValidationIssue(
                type="missing_inputs",
                message="Analysis specs require at least one input ref",
                location={"path": "/inputs"},
            )
        )
    if graph_spec is not None:
        graph_result = validate_graph_spec(graph_spec)
        errors.extend(graph_result.errors)
        warnings.extend(graph_result.warnings)
        if graph_result.valid:
            from feedbax.retained_observables import RetentionPlanError, lower_retention_plan

            graph = (
                graph_spec
                if isinstance(graph_spec, GraphSpec)
                else GraphSpec.model_validate(graph_spec)
            )
            try:
                lower_retention_plan(
                    graph,
                    analysis_input_requirements=spec.input_requirements,
                )
            except RetentionPlanError as exc:
                errors.append(
                    ValidationIssue(
                        type="analysis_input_graph_mismatch",
                        message=str(exc),
                        location={"path": exc.path, "selector": exc.selector or ""},
                    )
                )
    else:
        from feedbax.retained_observables import RetentionPlanError, normalize_selector_ref

        for index, requirement in enumerate(spec.input_requirements):
            selector_value = (
                requirement.target if requirement.target is not None else requirement.selector
            )
            if selector_value is None:
                errors.append(
                    ValidationIssue(
                        type="invalid_analysis_input",
                        message="Analysis input requirement is missing a selector",
                        location={"path": f"/input_requirements/{index}/selector"},
                    )
                )
                continue
            try:
                normalize_selector_ref(
                    selector_value,
                    path=f"/input_requirements/{index}/target",
                )
            except RetentionPlanError as exc:
                errors.append(
                    ValidationIssue(
                        type="invalid_analysis_input",
                        message=str(exc),
                        location={"path": exc.path, "selector": exc.selector or ""},
                    )
                )
    return ProviderValidationResult(valid=not errors, errors=errors, warnings=warnings)


def validate_spec(
    kind: str,
    payload: dict[str, Any],
    *,
    graph_spec: Optional[dict[str, Any]] = None,
) -> ProviderValidationResult:
    if kind == "graph":
        return validate_graph_spec(payload)
    if kind == "training":
        return validate_training_spec(payload, graph_spec=graph_spec)
    if kind == "task":
        return validate_task_spec(payload)
    if kind == "evaluation":
        return validate_evaluation_spec(payload)
    if kind == "analysis":
        return validate_analysis_spec(payload)
    raise ValueError(f"Unknown spec kind: {kind!r}")
