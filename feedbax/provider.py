"""Feedbax provider capability, registry, and validation contract."""

from __future__ import annotations

import pkgutil
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

import feedbax.analysis as analysis_pkg
from feedbax.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArtifactRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    GraphSpecManifest,
    PROVIDER_VERSION,
    ReportManifest,
    ReportSpec,
    SCHEMA_VERSION,
    TrainingRunManifest,
    TrainingRunSetManifest,
    feedbax_version,
    utc_now,
)
from feedbax.execution import ExecutionPlan, ExecutionSpec, LocalExecutionResult
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
    StudioSchemaEnumerationRequest,
    StudioSchemaRegistry,
    TaskDataSchema,
    ValueSchema,
)
from feedbax.web.models.graph import GraphSpec
from feedbax.web.models.training import LossTermSpec, TaskSpec, TrainingSpec

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
        "TrainingSpec": TrainingSpec,
        "TaskSpec": TaskSpec,
        "LossTermSpec": LossTermSpec,
        "EvaluationRunSpec": EvaluationRunSpec,
        "AnalysisRunSpec": AnalysisRunSpec,
        "ReportSpec": ReportSpec,
        "ArtifactRef": ArtifactRef,
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
        "StudioSchemaRegistry": StudioSchemaRegistry,
        "StudioSchemaEnumerationRequest": StudioSchemaEnumerationRequest,
        "GraphSpecManifest": GraphSpecManifest,
        "TrainingRunSetManifest": TrainingRunSetManifest,
        "TrainingRunManifest": TrainingRunManifest,
        "EvaluationRunManifest": EvaluationRunManifest,
        "AnalysisRunManifest": AnalysisRunManifest,
        "ReportManifest": ReportManifest,
    }


def _schemas() -> dict[str, dict[str, Any]]:
    return {name: model.model_json_schema() for name, model in _schema_models().items()}


def provider_manifest() -> ProviderManifest:
    capabilities = {
        "health": CapabilitySpec(output_schema="ProviderHealth"),
        "provider_manifest": CapabilitySpec(output_schema="ProviderManifest"),
        "validate_graph_spec": CapabilitySpec(
            input_schema="GraphSpec",
            output_schema="ProviderValidationResult",
        ),
        "validate_training_spec": CapabilitySpec(
            input_schema="TrainingSpec",
            output_schema="ProviderValidationResult",
        ),
        "validate_task_spec": CapabilitySpec(
            input_schema="TaskSpec",
            output_schema="ProviderValidationResult",
        ),
        "validate_evaluation_spec": CapabilitySpec(
            input_schema="EvaluationRunSpec",
            output_schema="ProviderValidationResult",
        ),
        "validate_analysis_spec": CapabilitySpec(
            input_schema="AnalysisRunSpec",
            output_schema="ProviderValidationResult",
        ),
        "start_training_run": CapabilitySpec(
            input_schema="TrainingSpec",
            output_schema="TrainingRunManifest",
            requires_review=True,
            description="Start a local or configured worker training run.",
        ),
        "prepare_execution_plan": CapabilitySpec(
            input_schema="ExecutionSpec",
            output_schema="ExecutionPlan",
            description="Prepare a deterministic local, SSH, RunPod, or Modal execution plan.",
        ),
        "run_local_execution": CapabilitySpec(
            input_schema="ExecutionSpec",
            output_schema="LocalExecutionResult",
            requires_review=True,
            description="Run an explicitly local execution and emit a durable manifest.",
        ),
        "prepare_studio_training_execution": CapabilitySpec(
            input_schema="StudioTrainingExecutionRequest",
            output_schema="StudioTrainingExecutionPreparation",
            description="Lower a Studio train-stage scenario into a provider execution plan.",
            transports=["python", "http"],
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
        ),
        "materialize_studio_pipeline": CapabilitySpec(
            input_schema="StudioPipelineMaterializationRequest",
            output_schema="StudioPipelineMaterializationResult",
            description=(
                "Materialize Studio eval, analysis, and report stages from upstream "
                "workspace collections and return updated lineage refs."
            ),
            transports=["python", "http"],
        ),
        "enumerate_studio_schemas": CapabilitySpec(
            input_schema="StudioSchemaEnumerationRequest",
            output_schema="StudioSchemaRegistry",
            description=(
                "Enumerate static Studio graph port, task data, selector target, "
                "and validation schemas without JAX compilation or training."
            ),
            transports=["python", "http"],
        ),
        "list_components": CapabilitySpec(output_schema="ComponentRegistrySnapshot"),
        "list_tasks": CapabilitySpec(output_schema="TaskRegistrySnapshot"),
        "list_losses": CapabilitySpec(output_schema="LossRegistrySnapshot"),
        "list_protocols": CapabilitySpec(output_schema="ProtocolRegistrySnapshot"),
        "list_analyses": CapabilitySpec(output_schema="AnalysisRegistrySnapshot"),
    }
    return ProviderManifest(
        capabilities=capabilities,
        artifact_roles=[
            "training_checkpoint",
            "training_history",
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
        entry_points={
            "python": "feedbax.provider:provider_manifest",
            "cli": "feedbax-provider manifest",
            "http": "/api/provider/manifest",
        },
    )


def component_registry_snapshot() -> RegistrySnapshot:
    from feedbax.web.services.component_registry import ComponentRegistry

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
    from feedbax.web.services.loss_service import NORM_FUNCTIONS

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


def validate_graph_spec(payload: dict[str, Any] | GraphSpec) -> ProviderValidationResult:
    from feedbax.web.services.component_registry import ComponentRegistry

    try:
        spec = payload if isinstance(payload, GraphSpec) else GraphSpec.model_validate(payload)
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

        for idx, wire in enumerate(graph.wires):
            wire_path = f"{prefix}/wires/{idx}"
            source = graph.nodes.get(wire.source_node)
            target = graph.nodes.get(wire.target_node)
            if source is None:
                errors.append(
                    ValidationIssue(
                        type="unknown_source_node",
                        message=f"Wire source node {wire.source_node!r} does not exist",
                        location={"path": wire_path},
                    )
                )
            elif wire.source_port not in _component_ports(
                source.type, source.output_ports, "output_ports", registry
            ):
                errors.append(
                    ValidationIssue(
                        type="unknown_source_port",
                        message=(
                            f"Wire source port {wire.source_node}.{wire.source_port} does not exist"
                        ),
                        location={"path": wire_path},
                    )
                )
            if target is None:
                errors.append(
                    ValidationIssue(
                        type="unknown_target_node",
                        message=f"Wire target node {wire.target_node!r} does not exist",
                        location={"path": wire_path},
                    )
                )
            elif wire.target_port not in _component_ports(
                target.type, target.input_ports, "input_ports", registry
            ):
                errors.append(
                    ValidationIssue(
                        type="unknown_target_port",
                        message=(
                            f"Wire target port {wire.target_node}.{wire.target_port} does not exist"
                        ),
                        location={"path": wire_path},
                    )
                )

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
            from feedbax.web.services.loss_service import loss_service

            graph = (
                graph_spec
                if isinstance(graph_spec, GraphSpec)
                else GraphSpec.model_validate(graph_spec)
            )
            for error in loss_service.validate_loss_spec(spec.loss, graph):
                errors.append(
                    ValidationIssue(
                        type="loss_graph_mismatch",
                        message=error["message"],
                        location={
                            "path": "/loss/" + "/".join(error.get("path", [])),
                            "field": error.get("field", ""),
                        },
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
    return ProviderValidationResult(valid=not errors, errors=errors)


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


def validate_analysis_spec(payload: dict[str, Any] | AnalysisRunSpec) -> ProviderValidationResult:
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
