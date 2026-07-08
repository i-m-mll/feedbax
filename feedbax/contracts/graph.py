"""Pydantic models for graph specifications."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from feedbax.contracts.value_schema import ValueSchema


GRAPH_SPEC_SCHEMA_ID = "feedbax.spec.graph"
GRAPH_SPEC_SCHEMA_VERSION_V2 = "feedbax.spec.graph.v2"
GRAPH_SPEC_SCHEMA_VERSION_V3 = "feedbax.spec.graph.v3"
GRAPH_SPEC_SCHEMA_VERSION = "feedbax.spec.graph.v4"
LEGACY_GRAPH_SPEC_SCHEMA_VERSION = "1.0.0"
ANALYSIS_DATA_PRODUCT_REQUIREMENT_SCHEMA_ID = (
    "feedbax.spec.analysis_data_product_requirement"
)
ANALYSIS_DATA_PRODUCT_REQUIREMENT_SCHEMA_VERSION = (
    "feedbax.spec.analysis_data_product_requirement.v1"
)


STUDIO_VALUE_SPEC_SCHEMA_VERSION = "feedbax.spec.studio.value.v2"
LEGACY_STUDIO_VALUE_SPEC_SCHEMA_V1 = "feedbax.spec.studio.value.v1"
LEGACY_STUDIO_VALUE_SPEC_FRONTEND_SCHEMA_V1 = "feedbax.studio.value.v1"
STUDIO_VALUE_SPEC_SCHEMA_ID = "feedbax.spec.studio.value"


def _is_value_spec_like_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    schema_id = value.get("schema_id")
    schema_version = value.get("schema_version")
    return (
        schema_id == STUDIO_VALUE_SPEC_SCHEMA_ID
        or (
            isinstance(schema_version, str)
            and schema_version.startswith("feedbax.studio.value.")
        )
        or (
            isinstance(schema_version, str)
            and schema_version.startswith(f"{STUDIO_VALUE_SPEC_SCHEMA_ID}.")
        )
    )


def _validate_nested_value_specs(value: Any) -> Any:
    if _is_value_spec_like_payload(value):
        return StudioValueSpec.model_validate(value).model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_validate_nested_value_specs(item) for item in value]
    if isinstance(value, dict):
        return {key: _validate_nested_value_specs(item) for key, item in value.items()}
    return value


# Use Any for nested param values to avoid recursive type issues
ParamValue = Union[int, float, str, bool, None, List[Any], Dict[str, Any]]


class ParamSchema(BaseModel):
    """Schema for a component parameter."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)
    param_schema_version: Optional[str] = None
    input_ports: List[str] = Field(default_factory=list)
    output_ports: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def validate_value_spec_params(cls, data: Any) -> Any:
        """Normalize Studio-authored ValueSpec params instead of passing dicts through."""
        if isinstance(data, dict) and isinstance(data.get("params"), dict):
            return {
                **data,
                "params": {
                    key: _validate_nested_value_specs(value)
                    for key, value in data["params"].items()
                },
            }
        return data


class ParameterConstraintSpec(BaseModel):
    """Structural constraint for a graph node parameter array.

    ``mask`` uses trainability semantics: truthy entries keep the parameter value,
    while falsy entries are projected to ``value`` at initialization and after
    optimizer updates.
    """

    node: str
    role: str
    mask: ParamValue
    value: ParamValue = 0.0


class WireSpec(BaseModel):
    """Specification for a wire connecting two ports."""

    model_config = ConfigDict(extra="forbid")

    source_node: str
    source_port: str
    target_node: str
    target_port: str
    temporality: Literal["instant", "recurrent"] = "instant"
    recurrent_initializer: Optional[Dict[str, Any]] = None


class AdditiveGraphChannelTargetSpec(BaseModel):
    """Target for a named additive graph-channel adapter."""

    kind: Literal["edge", "input"]
    target_node: str
    target_port: str
    source_node: Optional[str] = None
    source_port: Optional[str] = None

    @model_validator(mode="after")
    def validate_target(self) -> "AdditiveGraphChannelTargetSpec":
        if self.kind == "edge" and (self.source_node is None or self.source_port is None):
            raise ValueError("edge additive channel adapters require source_node and source_port")
        if self.kind == "input" and (self.source_node is not None or self.source_port is not None):
            raise ValueError("input additive channel adapters must not set source_node/source_port")
        return self


class AdditiveGraphChannelAdapterSpec(BaseModel):
    """Named external additive input to materialize into a graph."""

    label: str
    input_key: str
    target: AdditiveGraphChannelTargetSpec
    adapter_node: Optional[str] = None
    payload_shape: Optional[List[int]] = None
    payload_dtype: Optional[str] = None
    provenance_role: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DerivedDimensionRuleSpec(BaseModel):
    """Rule that derives one node builder parameter from resolved graph shapes."""

    model_config = ConfigDict(extra="forbid")

    node: str
    param: str
    source: Literal["input_port"] = "input_port"
    port: str
    axis: int = -1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserPortSpec(BaseModel):
    """User-defined ports for a subgraph."""

    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)


class CanvasPositionSpec(BaseModel):
    """Canvas x/y position."""

    x: float
    y: float


class CanvasSizeSpec(BaseModel):
    """Canvas width/height dimensions."""

    width: float
    height: float


class CanvasViewportSpec(CanvasPositionSpec):
    """Canvas viewport including zoom."""

    zoom: float


class TapPositionSpec(BaseModel):
    """Tap placement relative to graph nodes."""

    afterNode: str
    targetNode: Optional[str] = None


class TapTransform(BaseModel):
    """Transform applied by a tap."""

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)
    intervention: Optional["StudioInterventionTransformSpec"] = None


class TapSpec(BaseModel):
    """Specification for a tap (probe or intervention point)."""

    id: str
    type: Literal["probe", "intervention"]
    position: TapPositionSpec
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
    value_schema: Optional[ValueSchema] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisInputConsumerSpec(BaseModel):
    """Studio consumer metadata for an analysis input requirement."""

    page_id: Optional[str] = None
    node_id: Optional[str] = None
    input_port: Optional[str] = None
    analysis_type: Optional[str] = None
    role: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisDataProductRequirement(BaseModel):
    """Typed requirement for a resolved analysis data product.

    ``descriptor_basis_hash`` is reserved for the descriptor identity contract
    owned by issue 844acc6. It is present when descriptor or component-selector
    identity is part of the product semantics, and otherwise remains ``None``.
    The descriptor vocabulary itself is intentionally not defined here.
    """

    schema_id: str = ANALYSIS_DATA_PRODUCT_REQUIREMENT_SCHEMA_ID
    schema_version: str = ANALYSIS_DATA_PRODUCT_REQUIREMENT_SCHEMA_VERSION
    role: str
    product_schema_id: str
    exact_product_schema_version: Optional[str] = None
    min_product_schema_version: Optional[str] = None
    max_product_schema_version: Optional[str] = None
    logical_name: Optional[str] = None
    descriptor_selector_requirements: Dict[str, Any] = Field(default_factory=dict)
    descriptor_basis_hash: Optional[str] = None
    product_identity_hash: Optional[str] = None
    artifact_sha256: Optional[str] = None
    producer_manifest_id: Optional[str] = None
    producer_manifest_hash: Optional[str] = None
    parent_manifest_ids: List[str] = Field(default_factory=list)
    parent_manifest_hashes: List[str] = Field(default_factory=list)
    checkpoint_policy: Optional[Dict[str, Any]] = None
    rollout_policy: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    fallback_policy: Literal["forbid", "allow_declared"] = "forbid"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_schema_identity(self) -> "AnalysisDataProductRequirement":
        if self.schema_id != ANALYSIS_DATA_PRODUCT_REQUIREMENT_SCHEMA_ID:
            raise ValueError(
                "unsupported AnalysisDataProductRequirement schema_id: "
                f"{self.schema_id!r}, expected "
                f"{ANALYSIS_DATA_PRODUCT_REQUIREMENT_SCHEMA_ID!r}"
            )
        if self.schema_version != ANALYSIS_DATA_PRODUCT_REQUIREMENT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported AnalysisDataProductRequirement schema_version: "
                f"{self.schema_version!r}, expected "
                f"{ANALYSIS_DATA_PRODUCT_REQUIREMENT_SCHEMA_VERSION!r}"
            )
        if not self.role.strip():
            raise ValueError("AnalysisDataProductRequirement role must not be empty")
        if not self.product_schema_id.strip():
            raise ValueError(
                "AnalysisDataProductRequirement product_schema_id must not be empty"
            )
        return self


class AnalysisInputRequirement(BaseModel):
    """Observable requirement owned by an analysis consumer.

    Analysis inputs use the same selector space as retained observables but
    lower as implicit requirements for a run/artifact, rather than as explicit
    always-captured graph observables.
    """

    id: str = Field(default_factory=lambda: f"analysis-input:{uuid.uuid4().hex}")
    label: Optional[str] = None
    selector: Optional[str] = None
    target: Optional[RetainedObservableTargetSpec] = None
    retention: RetentionPolicySpec = Field(default_factory=RetentionPolicySpec)
    value_schema: Optional[Dict[str, Any]] = None
    data_product: Optional[AnalysisDataProductRequirement] = None
    consumer: AnalysisInputConsumerSpec = Field(default_factory=AnalysisInputConsumerSpec)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphMetadata(BaseModel):
    """Metadata for a graph."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    created_at: str
    updated_at: str
    version: str = "1.0.0"
    save_revision: int = Field(
        default=0,
        ge=0,
        description="Optimistic-concurrency revision for Studio project saves.",
    )
    author: Optional[str] = None
    tags: Optional[List[str]] = None


class GraphSpec(BaseModel):
    """Complete specification for a computation graph."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[GRAPH_SPEC_SCHEMA_ID] = Field(
        default=GRAPH_SPEC_SCHEMA_ID,
        description="Stable machine-readable schema identity for GraphSpec payloads.",
    )
    schema_version: Literal[GRAPH_SPEC_SCHEMA_VERSION] = Field(
        default=GRAPH_SPEC_SCHEMA_VERSION,
        description="Machine-readable GraphSpec schema version.",
    )
    nodes: Dict[str, ComponentSpec] = Field(default_factory=dict)
    wires: List[WireSpec] = Field(default_factory=list)
    additive_channel_adapters: List[AdditiveGraphChannelAdapterSpec] = Field(default_factory=list)
    input_ports: List[str] = Field(default_factory=list)
    output_ports: List[str] = Field(default_factory=list)
    input_bindings: Dict[str, Tuple[str, str]] = Field(default_factory=dict)
    output_bindings: Dict[str, Tuple[str, str]] = Field(default_factory=dict)
    subgraphs: Optional[Dict[str, "GraphSubgraphSpec"]] = None
    derived_dimensions: List[DerivedDimensionRuleSpec] = Field(default_factory=list)
    barnacles: Optional[Dict[str, List[BarnacleSpec]]] = None
    user_ports: Optional[Dict[str, UserPortSpec]] = None
    taps: Optional[List[TapSpec]] = None
    retained_observables: Optional[List[RetainedObservableSpec]] = None
    parameter_constraints: List[ParameterConstraintSpec] = Field(default_factory=list)
    metadata: Optional[GraphMetadata] = None

    @field_validator("subgraphs", mode="before")
    @classmethod
    def validate_subgraph_schema_family(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, dict):
            return value
        from feedbax.contracts.acausal import ACAUSAL_GRAPH_SCHEMA_ID, AcausalGraphSpec

        subgraphs: dict[str, GraphSubgraphSpec] = {}
        for node_id, raw_subgraph in value.items():
            if isinstance(raw_subgraph, (GraphSpec, AcausalGraphSpec)):
                subgraphs[str(node_id)] = raw_subgraph
                continue
            if not isinstance(raw_subgraph, dict):
                raise ValueError(
                    "GraphSpec.subgraphs values must be schema-bearing graph payloads: "
                    f"node_id={node_id!r}, value_type={type(raw_subgraph).__name__}"
                )
            schema_id = raw_subgraph.get("schema_id", GRAPH_SPEC_SCHEMA_ID)
            if schema_id == GRAPH_SPEC_SCHEMA_ID:
                subgraphs[str(node_id)] = GraphSpec.model_validate(raw_subgraph)
            elif schema_id == ACAUSAL_GRAPH_SCHEMA_ID:
                subgraphs[str(node_id)] = AcausalGraphSpec.model_validate(raw_subgraph)
            else:
                raise ValueError(
                    "Unsupported GraphSpec subgraph schema_id: "
                    f"node_id={node_id!r}, schema_id={schema_id!r}, "
                    f"expected one of {GRAPH_SPEC_SCHEMA_ID!r}, {ACAUSAL_GRAPH_SCHEMA_ID!r}"
                )
        return subgraphs


from feedbax.contracts.acausal import ACAUSAL_GRAPH_SCHEMA_ID, AcausalGraphSpec

GraphSubgraphSpec = Union[GraphSpec, AcausalGraphSpec]


class GraphInteriorDomainMismatch(ValueError):
    """Raised when a consumer receives a subgraph from an unsupported domain."""


def schema_id_for_subgraph(subgraph: GraphSubgraphSpec | Mapping[str, Any]) -> str | None:
    """Return the stable schema id for a causal or acausal subgraph payload."""
    if isinstance(subgraph, GraphSpec):
        return GRAPH_SPEC_SCHEMA_ID
    if isinstance(subgraph, AcausalGraphSpec):
        return ACAUSAL_GRAPH_SCHEMA_ID
    schema_id = subgraph.get("schema_id") if isinstance(subgraph, dict) else None
    return schema_id if isinstance(schema_id, str) else None


def expected_subgraph_schema_id_for_domain(domain_id: str | None) -> str | None:
    """Return the interior schema id for domains this backend can distinguish."""
    if domain_id is None:
        return None
    if domain_id == "feedbax.domain.causal":
        return GRAPH_SPEC_SCHEMA_ID
    if domain_id in {"feedbax.domain.acausal", "feedbax.domain.mechanics"}:
        return ACAUSAL_GRAPH_SCHEMA_ID
    return None


def require_causal_subgraph(
    subgraph: GraphSubgraphSpec,
    *,
    node_name: str,
    node_type: str,
    consumer: str,
) -> GraphSpec:
    """Return a causal subgraph or raise a typed domain-mismatch error."""
    if isinstance(subgraph, GraphSpec):
        return subgraph
    raise GraphInteriorDomainMismatch(
        f"{consumer} requires a causal GraphSpec interior for node {node_name!r} "
        f"({node_type}), but received schema_id={ACAUSAL_GRAPH_SCHEMA_ID!r}. "
        "Acausal interiors must be handled by an acausal domain compiler."
    )


def validate_subgraph_domain(
    subgraph: GraphSubgraphSpec,
    *,
    expected_domain: str | None,
    node_name: str,
    node_type: str,
    consumer: str,
) -> None:
    """Raise when a subgraph payload does not match the node's interior domain."""
    expected_schema_id = expected_subgraph_schema_id_for_domain(expected_domain)
    actual_schema_id = schema_id_for_subgraph(subgraph)
    if expected_schema_id is None or actual_schema_id == expected_schema_id:
        return
    raise GraphInteriorDomainMismatch(
        f"{consumer} received an interior for node {node_name!r} ({node_type}) "
        f"with schema_id={actual_schema_id!r}, but domain {expected_domain!r} "
        f"requires schema_id={expected_schema_id!r}."
    )


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

    position: CanvasPositionSpec
    collapsed: bool = False
    selected: bool = False
    reversed: bool = False
    size: Optional[CanvasSizeSpec] = None


class TapUIState(BaseModel):
    """UI state for a tap."""

    position: CanvasPositionSpec
    selected: Optional[bool] = None


class GraphUIState(BaseModel):
    """UI state for the entire graph."""

    viewport: CanvasViewportSpec = Field(
        default_factory=lambda: CanvasViewportSpec(x=0, y=0, zoom=1)
    )
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
    input_requirements: List[AnalysisInputRequirement] = Field(default_factory=list)
    viewport: Dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0, "zoom": 1})
    eval_run_id: Optional[str] = None
    expanded_field_paths: List[str] = Field(default_factory=list)


STUDIO_WORKSPACE_SCHEMA_VERSION = "feedbax.spec.studio.workspace.v1"
STUDIO_SCENARIO_SCHEMA_VERSION = "feedbax.spec.studio.scenario.v1"
STUDIO_STAGE_SCHEMA_VERSION = "feedbax.spec.studio.stage.v1"

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


class StudioSelectorRef(BaseModel):
    """Typed reference to a schema-backed Studio selector target."""

    namespace: Literal[
        "graph_port",
        "graph_edge",
        "graph_output",
        "recurrent_carry",
        "retained_observable",
        "probe",
        "state_path",
        "task_object",
        "task_data",
        "task_binding",
        "mechanics_object",
        "biomechanics_object",
        "artifact_field",
        "analysis_output",
        "custom",
    ]
    compact: str
    target_id: Optional[str] = None
    path: Optional[str] = None
    expected_shape: Optional[List[Any]] = None
    dtype: Optional[str] = None
    units: Optional[str] = None
    frame: Optional[str] = None
    role: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioValueEnumerableSpec(BaseModel):
    """Enumerable ValueSpec axis payload consumed by train matrix expansion."""

    model_config = ConfigDict(extra="forbid")

    form: Literal["list", "range", "sampler"]
    values: Optional[List[Any]] = None
    start: Optional[float] = None
    stop: Optional[float] = None
    count: Optional[int] = None
    scale: Literal["linear", "log"] = "linear"
    sampler: Optional[Dict[str, Any]] = None
    n: Optional[int] = None

    @model_validator(mode="after")
    def validate_form_payload(self) -> "StudioValueEnumerableSpec":
        if self.form == "list":
            if not self.values:
                raise ValueError("sweep list enumerables require at least one value")
            return self
        if self.form == "range":
            if self.start is None or self.stop is None or self.count is None:
                raise ValueError("sweep range enumerables require start, stop, and count")
            if self.count < 1:
                raise ValueError("sweep range count must be positive")
            if self.scale == "log" and (self.start <= 0 or self.stop <= 0):
                raise ValueError("log sweep ranges require positive start and stop")
            return self
        if self.sampler is None or self.n is None:
            raise ValueError("sweep sampler enumerables require sampler and n")
        if self.n < 1:
            raise ValueError("sweep sampler n must be positive")
        return self


class StudioValueVariationSpec(BaseModel):
    """Where and how a Studio value varies."""

    model_config = ConfigDict(extra="forbid")

    scope: Literal["fixed", "snapshot", "run", "replicate", "trial", "epoch", "timestep", "sweep"]
    enumerable: Optional[StudioValueEnumerableSpec] = None
    stochastic_policy: Optional[Literal["shared_per_run", "resample_per_replicate"]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scope_payload(self) -> "StudioValueVariationSpec":
        if self.scope == "sweep" and self.enumerable is None:
            raise ValueError("sweep variation requires an enumerable list, range, or sampler+n")
        if self.scope != "sweep" and self.enumerable is not None:
            raise ValueError("enumerable payloads are only valid for sweep variation")
        if self.scope == "run" and self.stochastic_policy not in (None, "shared_per_run"):
            raise ValueError("run variation samples once and shares within the run")
        if self.scope == "replicate" and self.stochastic_policy not in (
            None,
            "resample_per_replicate",
        ):
            raise ValueError("replicate variation resamples per replicate")
        return self


def _legacy_value_form(mode: Any) -> str:
    return "literal" if mode == "constant" else str(mode)


def _legacy_variation(data: Dict[str, Any]) -> Dict[str, Any]:
    mode = data.get("mode")
    metadata = dict(data.get("metadata") or {})
    scope = data.get("sampling_scope") or ("fixed" if mode == "constant" else "run")
    if scope == "replicate":
        policy = "resample_per_replicate"
    elif scope == "run":
        policy = "shared_per_run"
    else:
        policy = None
    variation: Dict[str, Any] = {
        "scope": scope,
        "metadata": {"migrated_from_sampling_scope": data.get("sampling_scope")},
    }
    if policy:
        variation["stochastic_policy"] = policy
    if metadata.get("indexed_constant") is True:
        values = data.get("value") if isinstance(data.get("value"), list) else [data.get("value")]
        variation["scope"] = "sweep"
        variation["enumerable"] = {"form": "list", "values": values}
    elif scope == "sweep" and isinstance(data.get("distribution"), dict):
        distribution = data["distribution"]
        parameters = distribution.get("parameters")
        if distribution.get("family") == "categorical" and isinstance(parameters, dict):
            values = parameters.get("values")
            if isinstance(values, list):
                variation["enumerable"] = {"form": "list", "values": values}
        if "enumerable" not in variation:
            n = metadata.get("sweep_n") or metadata.get("axis_count")
            if not isinstance(n, int):
                raise ValueError("legacy distribution sweep ValueSpec requires metadata.sweep_n")
            variation["enumerable"] = {"form": "sampler", "sampler": distribution, "n": n}
    return variation


class StudioValueSpec(BaseModel):
    """Declarative value used by Studio task timelines and parameter fields."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = STUDIO_VALUE_SPEC_SCHEMA_VERSION
    value_form: Literal["literal", "reference", "expression", "function", "schedule", "distribution"]
    variation: StudioValueVariationSpec = Field(
        default_factory=lambda: StudioValueVariationSpec(scope="fixed")
    )
    mode: str
    value: Optional[Any] = None
    reference: Optional[StudioSelectorRef] = None
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

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_value_spec(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        version = data.get("schema_version", LEGACY_STUDIO_VALUE_SPEC_SCHEMA_V1)
        if version == STUDIO_VALUE_SPEC_SCHEMA_VERSION:
            if "mode" not in data and "value_form" in data:
                mode = "constant" if data["value_form"] == "literal" else data["value_form"]
                return {**data, "mode": mode}
            return data
        if version not in {
            LEGACY_STUDIO_VALUE_SPEC_SCHEMA_V1,
            LEGACY_STUDIO_VALUE_SPEC_FRONTEND_SCHEMA_V1,
        }:
            raise ValueError(
                f"unsupported StudioValueSpec schema_version {version!r}; "
                f"expected {STUDIO_VALUE_SPEC_SCHEMA_VERSION}"
            )
        if "mode" not in data:
            raise ValueError("legacy StudioValueSpec requires mode")
        return {
            **data,
            "schema_version": STUDIO_VALUE_SPEC_SCHEMA_VERSION,
            "value_form": _legacy_value_form(data["mode"]),
            "variation": _legacy_variation(data),
        }

    @model_validator(mode="after")
    def validate_value_variation(self) -> "StudioValueSpec":
        expected_mode = "constant" if self.value_form == "literal" else self.value_form
        if self.mode != expected_mode:
            raise ValueError("StudioValueSpec mode must match value_form compatibility alias")
        if self.value_form == "literal" and self.variation.scope != "sweep":
            if self.sampling_scope not in (None, "fixed"):
                raise ValueError("literal fixed values must not carry a sampling_scope")
        if self.value_form == "distribution" and self.variation.scope == "run":
            if self.variation.stochastic_policy not in (None, "shared_per_run"):
                raise ValueError("run distribution specs sample once and share")
        if self.value_form == "distribution" and self.variation.scope == "replicate":
            if self.variation.stochastic_policy not in (None, "resample_per_replicate"):
                raise ValueError("replicate distribution specs resample per replicate")
        return self


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
    task_data_id: Optional[str] = None
    path: str
    epoch_ids: List[str] = Field(default_factory=list)
    value_spec: Optional[StudioValueSpec] = None
    value_schema: Optional[Dict[str, Any]] = None
    task_data_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioTaskTimelineSegmentSpec(BaseModel):
    """Named group of task timeline epochs for objectives and analyses."""

    id: str
    label: str
    epoch_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioTaskTimelineSpec(BaseModel):
    """Structured task timeline stored under Studio task specs."""

    schema_version: str = "feedbax.spec.studio.task_timeline.v1"
    epochs: List[StudioTaskEpochSpec] = Field(default_factory=list)
    signals: List[StudioTaskTimelineSignalSpec] = Field(default_factory=list)
    segments: List[StudioTaskTimelineSegmentSpec] = Field(default_factory=list)
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
                "source_data_id in feedbax.spec.studio.task_bindings.v2"
            )
        return data


class StudioTaskBindingSpec(BaseModel):
    """Scenario-owned task data surface and its model bindings."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "feedbax.spec.studio.task_bindings.v2"
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
                "feedbax.spec.studio.task_bindings.v2 with exposed_data and source_data_id"
            )
        if "exposed_outputs" in data:
            raise ValueError(
                "task_binding_spec.exposed_outputs was renamed to exposed_data in "
                "feedbax.spec.studio.task_bindings.v2"
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
    schema_version: str = STUDIO_STAGE_SCHEMA_VERSION
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
AcausalGraphSpec.model_rebuild(
    _types_namespace={"ComponentSpec": ComponentSpec, "GraphMetadata": GraphMetadata}
)
GraphSpec.model_rebuild(_types_namespace={"AcausalGraphSpec": AcausalGraphSpec})
GraphUIState.model_rebuild()
ParamSchema.model_rebuild()
