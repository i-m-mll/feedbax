"""Pydantic models for graph specifications."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Use Any for nested param values to avoid recursive type issues
ParamValue = Union[int, float, str, bool, None, List[Any], Dict[str, Any]]


class ParamSchema(BaseModel):
    """Schema for a component parameter."""

    name: str
    type: Literal["int", "float", "bool", "str", "enum", "array", "object", "bounds2d"]
    default: Optional[ParamValue] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    description: Optional[str] = None
    required: bool = False
    nested_schema: Optional[List["ParamSchema"]] = None


class ComponentSpec(BaseModel):
    """Specification for a component instance in a graph."""

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)
    input_ports: List[str] = Field(default_factory=list)
    output_ports: List[str] = Field(default_factory=list)


class WireSpec(BaseModel):
    """Specification for a wire connecting two ports."""

    source_node: str
    source_port: str
    target_node: str
    target_port: str
    temporality: Literal["instant", "recurrent"] = "instant"
    recurrent_initializer: Optional[Dict[str, Any]] = None


class UserPortSpec(BaseModel):
    """User-defined ports for a subgraph."""

    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)


class TapTransform(BaseModel):
    """Transform applied by a tap."""

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)
    intervention: Optional["StudioInterventionTransformSpec"] = None


class TapSpec(BaseModel):
    """Specification for a tap (probe or intervention point)."""

    id: str
    type: Literal["probe", "intervention"]
    position: Dict[str, Any]
    paths: Dict[str, str] = Field(default_factory=dict)
    transform: Optional[TapTransform] = None


class BarnacleSpec(BaseModel):
    """Specification for a barnacle (attached probe/intervention)."""

    id: str
    kind: Literal["probe", "intervention"]
    timing: Literal["input", "output"]
    label: str
    read_paths: List[str] = Field(default_factory=list)
    write_paths: List[str] = Field(default_factory=list)
    transform: str = ""


class RetentionPolicySpec(BaseModel):
    """How long a selected observable must be retained during execution."""

    mode: Literal["stream", "window", "trajectory"] = "trajectory"
    window_size: Optional[int] = None
    order: Optional[int] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetainedObservableTargetSpec(BaseModel):
    """Target of a retained observable/probe."""

    kind: Literal[
        "port",
        "edge",
        "graph_output",
        "recurrent_carry",
        "state_path",
        "task_data",
    ]
    selector: str
    node_id: Optional[str] = None
    port: Optional[str] = None
    edge_id: Optional[str] = None
    path: Optional[str] = None
    timing: Optional[Literal["input", "output", "step", "initial", "final"]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetainedObservableSpec(BaseModel):
    """Declarative probe/data-retention request.

    This replaces treating probes as default graph nodes. A retained
    observable may attach to a port, edge, graph output, recurrent carry,
    state path, or task data selector, and can be consumed by losses,
    rollout artifacts, Studio Data projection, or analysis.
    """

    id: str = Field(default_factory=lambda: f"observable:{uuid.uuid4().hex}")
    label: Optional[str] = None
    selector: Optional[str] = None
    target: Optional[RetainedObservableTargetSpec] = None
    retention: RetentionPolicySpec = Field(default_factory=RetentionPolicySpec)
    value_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphMetadata(BaseModel):
    """Metadata for a graph."""

    name: str
    description: Optional[str] = None
    created_at: str
    updated_at: str
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: Optional[List[str]] = None


class GraphSpec(BaseModel):
    """Complete specification for a computation graph."""

    nodes: Dict[str, ComponentSpec] = Field(default_factory=dict)
    wires: List[WireSpec] = Field(default_factory=list)
    input_ports: List[str] = Field(default_factory=list)
    output_ports: List[str] = Field(default_factory=list)
    input_bindings: Dict[str, Tuple[str, str]] = Field(default_factory=dict)
    output_bindings: Dict[str, Tuple[str, str]] = Field(default_factory=dict)
    subgraphs: Optional[Dict[str, "GraphSpec"]] = None
    barnacles: Optional[Dict[str, List[BarnacleSpec]]] = None
    user_ports: Optional[Dict[str, UserPortSpec]] = None
    taps: Optional[List[TapSpec]] = None
    retained_observables: Optional[List[RetainedObservableSpec]] = None
    metadata: Optional[GraphMetadata] = None


class EdgeRoutingPoint(BaseModel):
    """A point in edge routing."""

    x: float
    y: float


class EdgeRouting(BaseModel):
    """Routing information for an edge."""

    style: Literal["bezier", "elbow"] = "bezier"
    points: List[EdgeRoutingPoint] = Field(default_factory=list)


class EdgeUIState(BaseModel):
    """UI state for an edge."""

    routing: EdgeRouting


class NodeUIState(BaseModel):
    """UI state for a node."""

    position: Dict[str, float]
    collapsed: bool = False
    selected: bool = False
    reversed: bool = False
    size: Optional[Dict[str, float]] = None


class TapUIState(BaseModel):
    """UI state for a tap."""

    position: Dict[str, float]
    selected: Optional[bool] = None


class GraphUIState(BaseModel):
    """UI state for the entire graph."""

    viewport: Dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0, "zoom": 1})
    node_states: Dict[str, NodeUIState] = Field(default_factory=dict)
    edge_states: Optional[Dict[str, EdgeUIState]] = None
    subgraph_states: Optional[Dict[str, "GraphUIState"]] = None
    tap_states: Optional[Dict[str, TapUIState]] = None


class AnalysisPageSpec(BaseModel):
    """Specification for a single analysis page within a project."""

    id: str
    name: str
    graph_spec: Dict[str, Any] = Field(default_factory=dict)
    eval_params: Dict[str, Any] = Field(default_factory=dict)
    viewport: Dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0, "zoom": 1})
    eval_run_id: Optional[str] = None
    expanded_field_paths: List[str] = Field(default_factory=list)


STUDIO_WORKSPACE_SCHEMA_VERSION = "feedbax.studio.workspace.v1"
STUDIO_SCENARIO_SCHEMA_VERSION = "feedbax.studio.scenario.v1"

StudioStageKind = Literal[
    "train",
    "eval",
    "analysis",
    "report",
    "import",
    "compare",
    "export",
    "protocol",
]
StudioStageStatus = Literal[
    "draft",
    "invalid",
    "ready",
    "running",
    "completed",
    "failed",
    "cancelled",
]


class StudioValidationIssue(BaseModel):
    """Validation issue attached to durable Studio workspace state."""

    type: str
    message: str
    location: Optional[Dict[str, str]] = None
    severity: Literal["error", "warning", "info"] = "error"


class StudioValidationState(BaseModel):
    """Last known validation state for a scenario, stage, or workspace."""

    valid: Optional[bool] = None
    checked_at: Optional[str] = None
    errors: List[StudioValidationIssue] = Field(default_factory=list)
    warnings: List[StudioValidationIssue] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioManifestRef(BaseModel):
    """Reference to a manifest produced or consumed by a Studio stage."""

    kind: str
    id: str
    role: Optional[str] = None
    provider: str = "feedbax"
    uri: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioArtifactRef(BaseModel):
    """Reference to a non-manifest artifact produced or consumed by Studio."""

    kind: str
    id: str
    role: Optional[str] = None
    provider: str = "feedbax"
    uri: Optional[str] = None
    media_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioCollectionRef(BaseModel):
    """Reference to a collection flowing between Studio pipeline stages."""

    id: str
    kind: str
    label: Optional[str] = None
    source_stage_id: Optional[str] = None
    item_refs: List[StudioManifestRef] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    facets: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioValueSpec(BaseModel):
    """Declarative value used by Studio task timelines and parameter fields."""

    schema_version: str = "feedbax.studio.value.v1"
    mode: str
    value: Optional[Any] = None
    reference: Optional[Dict[str, Any]] = None
    expression: Optional[str] = None
    function_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    distribution: Optional[Dict[str, Any]] = None
    schedule: Optional[Dict[str, Any]] = None
    sampling_scope: Optional[str] = None
    dtype: Optional[str] = None
    shape: Optional[List[Any]] = None
    units: Optional[str] = None
    frame: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioSelectorRef(BaseModel):
    """Typed reference to a schema-backed Studio selector target."""

    namespace: str
    compact: str
    target_id: Optional[str] = None
    path: Optional[str] = None
    expected_shape: Optional[List[Any]] = None
    dtype: Optional[str] = None
    units: Optional[str] = None
    frame: Optional[str] = None
    role: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioInterventionValueBounds(BaseModel):
    """Optional lower and upper bounds for clamp-style interventions."""

    min: Optional[Any] = None
    max: Optional[Any] = None


class StudioInterventionTransformSpec(BaseModel):
    """Narrow Studio-authored intervention semantics for graph taps."""

    operation: str
    target_selector: Optional[StudioSelectorRef] = None
    value: Optional[StudioValueSpec] = None
    bounds: Optional[StudioInterventionValueBounds] = None
    parameters: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioTaskEpochSpec(BaseModel):
    """Task timeline epoch/phase authored in Studio."""

    id: str
    label: str
    index: int
    length: StudioValueSpec
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioTaskTimelineSignalSpec(BaseModel):
    """Task signal membership across authored timeline epochs."""

    id: str
    label: str
    kind: str
    path: str
    epoch_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioTaskTimelineSpec(BaseModel):
    """Structured task timeline stored under Studio task specs."""

    schema_version: str = "feedbax.studio.task_timeline.v1"
    epochs: List[StudioTaskEpochSpec] = Field(default_factory=list)
    signals: List[StudioTaskTimelineSignalSpec] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioTaskDataSpec(BaseModel):
    """Scenario-owned task data that may be bound into a model graph."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    kind: str
    role: Optional[str] = None
    path: str
    bindable: bool
    expected_shape: Optional[List[Any]] = None
    dtype: Optional[str] = None
    units: Optional[str] = None
    frame: Optional[str] = None
    value_spec: Optional[StudioValueSpec] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioTaskBinding(BaseModel):
    """Binding from a scenario task data into a graph input port."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_data_id: str
    target_node_id: str
    target_port: str
    role: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_task_binding_field_names(cls, data: Any) -> Any:
        if isinstance(data, dict) and "source_output_id" in data:
            raise ValueError(
                "task_binding_spec.bindings[].source_output_id was renamed to "
                "source_data_id in feedbax.studio.task_bindings.v2"
            )
        return data


class StudioTaskBindingSpec(BaseModel):
    """Scenario-owned task data surface and its model bindings."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "feedbax.studio.task_bindings.v2"
    exposed_data: List[StudioTaskDataSpec] = Field(default_factory=list)
    bindings: List[StudioTaskBinding] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_task_binding_contract(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("schema_version") == "feedbax.studio.task_bindings.v1":
            raise ValueError(
                "feedbax.studio.task_bindings.v1 is no longer accepted; use "
                "feedbax.studio.task_bindings.v2 with exposed_data and source_data_id"
            )
        if "exposed_outputs" in data:
            raise ValueError(
                "task_binding_spec.exposed_outputs was renamed to exposed_data in "
                "feedbax.studio.task_bindings.v2"
            )
        return data


class StudioScenarioSpec(BaseModel):
    """Stage-owned structural scenario draft for model/task/objective state.

    The first implementation intentionally wraps current graph/task/loss state
    while reserving explicit slots for richer biomechanics, selector,
    objective, analysis, and report authoring. Later product work should extend
    these typed fields rather than replacing this workspace boundary.
    """

    model_config = ConfigDict(validate_assignment=True)

    id: str
    schema_version: str = STUDIO_SCENARIO_SCHEMA_VERSION
    label: str
    stage_id: Optional[str] = None
    parent_scenario_id: Optional[str] = None
    graph: Optional[GraphSpec] = None
    graph_ui_state: Optional[GraphUIState] = None
    training_spec: Optional[Dict[str, Any]] = None
    task_spec: Optional[Dict[str, Any]] = None
    task_binding_spec: Optional[StudioTaskBindingSpec] = None
    objective_spec: Optional[Dict[str, Any]] = None
    probe_specs: List[RetainedObservableSpec] = Field(default_factory=list)
    temporal_spec: Optional[Dict[str, Any]] = None
    biomechanics_spec: Optional[Dict[str, Any]] = None
    analysis_spec: Optional[Dict[str, Any]] = None
    report_spec: Optional[Dict[str, Any]] = None
    validation: StudioValidationState = Field(default_factory=StudioValidationState)
    ui_state: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioStageSpec(BaseModel):
    """Pipeline stage over scenario drafts, collections, and manifests."""

    id: str
    kind: StudioStageKind
    label: str
    status: StudioStageStatus = "draft"
    scenario_id: Optional[str] = None
    input_collections: List[StudioCollectionRef] = Field(default_factory=list)
    output_collections: List[StudioCollectionRef] = Field(default_factory=list)
    manifest_refs: List[StudioManifestRef] = Field(default_factory=list)
    artifact_refs: List[StudioArtifactRef] = Field(default_factory=list)
    execution_spec: Optional[Dict[str, Any]] = None
    selection_spec: Dict[str, Any] = Field(default_factory=dict)
    validation: StudioValidationState = Field(default_factory=StudioValidationState)
    ui_state: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioWorkspaceSpec(BaseModel):
    """Durable Studio workspace/pipeline model stored with a project."""

    id: str
    schema_version: str = STUDIO_WORKSPACE_SCHEMA_VERSION
    label: str
    active_stage_id: Optional[str] = None
    stages: List[StudioStageSpec] = Field(default_factory=list)
    scenarios: Dict[str, StudioScenarioSpec] = Field(default_factory=dict)
    collections: List[StudioCollectionRef] = Field(default_factory=list)
    manifest_refs: List[StudioManifestRef] = Field(default_factory=list)
    artifact_refs: List[StudioArtifactRef] = Field(default_factory=list)
    validation: StudioValidationState = Field(default_factory=StudioValidationState)
    ui_state: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphProject(BaseModel):
    """A complete graph project with metadata and UI state."""

    metadata: GraphMetadata
    graph: GraphSpec
    ui_state: Optional[GraphUIState] = None
    demo_training_data: Optional[Any] = None
    analysis_pages: Optional[List[AnalysisPageSpec]] = None
    active_analysis_page_id: Optional[str] = None
    workspace: Optional[StudioWorkspaceSpec] = None


def build_default_studio_workspace(
    *,
    label: str,
    graph: GraphSpec,
    ui_state: Optional[GraphUIState] = None,
    analysis_pages: Optional[List[AnalysisPageSpec]] = None,
    active_analysis_page_id: Optional[str] = None,
) -> StudioWorkspaceSpec:
    """Build the initial durable Studio workspace for a legacy project.

    This is a migration scaffold, not a narrowed MVP-only model. It creates the
    canonical stage/scenario anchors that later eval, analysis, report, visual
    objective, cloud execution, and Mandible-ingest work should extend.
    """

    workspace_id = f"studio-workspace:{uuid.uuid4().hex}"
    train_stage_id = "stage:train"
    eval_stage_id = "stage:eval"
    analysis_stage_id = "stage:analysis"
    report_stage_id = "stage:report"
    train_scenario_id = "scenario:train"
    eval_scenario_id = "scenario:eval"
    analysis_scenario_id = "scenario:analysis"
    report_scenario_id = "scenario:report"

    analysis_spec: Optional[Dict[str, Any]] = None
    if analysis_pages:
        analysis_spec = {
            "pages": [page.model_dump() for page in analysis_pages],
            "active_page_id": active_analysis_page_id,
        }

    scenarios = {
        train_scenario_id: StudioScenarioSpec(
            id=train_scenario_id,
            label="Training scenario",
            stage_id=train_stage_id,
            graph=graph,
            graph_ui_state=ui_state,
            metadata={"source": "graph_project_migration"},
        ),
        eval_scenario_id: StudioScenarioSpec(
            id=eval_scenario_id,
            label="Evaluation scenario",
            stage_id=eval_stage_id,
            parent_scenario_id=train_scenario_id,
            metadata={
                "source": "graph_project_migration",
                "inheritance": "training_default",
            },
        ),
        analysis_scenario_id: StudioScenarioSpec(
            id=analysis_scenario_id,
            label="Analysis scenario",
            stage_id=analysis_stage_id,
            analysis_spec=analysis_spec,
            metadata={"source": "graph_project_migration"},
        ),
        report_scenario_id: StudioScenarioSpec(
            id=report_scenario_id,
            label="Report scenario",
            stage_id=report_stage_id,
            metadata={"source": "graph_project_migration"},
        ),
    }

    stages = [
        StudioStageSpec(
            id=train_stage_id,
            kind="train",
            label="Train",
            scenario_id=train_scenario_id,
            output_collections=[
                StudioCollectionRef(
                    id="collection:training-runs",
                    kind="training_runs",
                    label="Training runs",
                    source_stage_id=train_stage_id,
                )
            ],
        ),
        StudioStageSpec(
            id=eval_stage_id,
            kind="eval",
            label="Evaluate",
            scenario_id=eval_scenario_id,
            input_collections=[
                StudioCollectionRef(
                    id="collection:training-runs",
                    kind="training_runs",
                    label="Training runs",
                    source_stage_id=train_stage_id,
                )
            ],
            output_collections=[
                StudioCollectionRef(
                    id="collection:evaluation-runs",
                    kind="evaluation_runs",
                    label="Evaluation runs",
                    source_stage_id=eval_stage_id,
                )
            ],
        ),
        StudioStageSpec(
            id=analysis_stage_id,
            kind="analysis",
            label="Analyze",
            scenario_id=analysis_scenario_id,
            input_collections=[
                StudioCollectionRef(
                    id="collection:evaluation-runs",
                    kind="evaluation_runs",
                    label="Evaluation runs",
                    source_stage_id=eval_stage_id,
                )
            ],
            output_collections=[
                StudioCollectionRef(
                    id="collection:analysis-products",
                    kind="analysis_products",
                    label="Analysis products",
                    source_stage_id=analysis_stage_id,
                )
            ],
        ),
        StudioStageSpec(
            id=report_stage_id,
            kind="report",
            label="Report",
            scenario_id=report_scenario_id,
            input_collections=[
                StudioCollectionRef(
                    id="collection:analysis-products",
                    kind="analysis_products",
                    label="Analysis products",
                    source_stage_id=analysis_stage_id,
                )
            ],
            output_collections=[
                StudioCollectionRef(
                    id="collection:reports",
                    kind="reports",
                    label="Reports",
                    source_stage_id=report_stage_id,
                )
            ],
        ),
    ]

    return StudioWorkspaceSpec(
        id=workspace_id,
        label=label,
        active_stage_id=train_stage_id,
        stages=stages,
        scenarios=scenarios,
        metadata={"source": "graph_project_migration"},
    )


class ValidationError(BaseModel):
    """A validation error."""

    type: str
    message: str
    location: Optional[Dict[str, str]] = None


class ValidationWarning(BaseModel):
    """A validation warning."""

    type: str
    message: str
    location: Optional[Dict[str, str]] = None


class ValidationResult(BaseModel):
    """Result of graph validation."""

    valid: bool
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationWarning] = Field(default_factory=list)
    cycles: List[List[str]] = Field(default_factory=list)


# Enable forward references
GraphSpec.model_rebuild()
GraphUIState.model_rebuild()
ParamSchema.model_rebuild()
