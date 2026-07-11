"""Workspace representation contracts for Studio component catalogs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedbax.contracts.graph import ParamSchema, ParamValue, StudioSelectorRef


REPRESENTATION_SCHEMA_ID = "feedbax.spec.studio.representation"
REPRESENTATION_SCHEMA_VERSION = "feedbax.spec.studio.representation.v3"
REPRESENTATION_SCHEMA_VERSION_V2 = "feedbax.spec.studio.representation.v2"
REPRESENTATION_SCHEMA_VERSION_V1 = "feedbax.spec.studio.representation.v1"
REPRESENTATION_SCHEMA_VERSION_V0 = "feedbax.spec.studio.representation.v0"
MUSCLE_PATH_GEOMETRY_SCHEMA_ID = "feedbax.spec.studio.muscle_path_geometry"
MUSCLE_PATH_GEOMETRY_SCHEMA_VERSION = "feedbax.spec.studio.muscle_path_geometry.v1"

RepresentationArchetype = Literal[
    "point_body",
    "planar_chain",
    "muscle_path",
    "marker",
    "region",
    "distribution_glyph",
    "vector",
    "trace",
    "objective_link",
    "annotation",
    "registered_renderer",
]
RepresentationSemanticRole = Literal[
    "origin",
    "insertion",
    "center",
    "endpoint",
    "joint",
    "body",
    "target",
    "path",
    "region",
    "glyph",
    "vector_start",
    "vector_end",
    "label",
    "custom",
]
RepresentationInteractionRole = Literal[
    "selectable",
    "hoverable",
    "draggable",
    "editable",
    "connectable",
    "snap_target",
    "read_only",
]
RepresentationAnchorSubpath = Literal[
    "position",
    "velocity",
    "orientation",
    "origin",
    "insertion",
    "center",
    "endpoint",
    "target",
    "path",
]
RepresentationStyleChannel = Literal[
    "stroke",
    "fill",
    "radius",
    "opacity",
    "line_width",
    "dash",
    "label",
    "glyph",
    "colormap",
    "z_index",
    "visibility",
]


class RepresentationContractModel(BaseModel):
    """Base model for representation contract records."""

    model_config = ConfigDict(extra="forbid")


class RepresentationParamPathBinding(RepresentationContractModel):
    """Bind a representation value to a component parameter path."""

    kind: Literal["param_path"] = "param_path"
    path: str
    expected_type: Optional[str] = None
    dim: Optional[int] = None


class RepresentationStateAnchorSelectorBinding(RepresentationContractModel):
    """Bind a representation value to a state or named object selector."""

    kind: Literal["selector"] = "selector"
    selector: StudioSelectorRef
    anchor_subpath: Optional[RepresentationAnchorSubpath] = None
    dim: Optional[int] = None

    @model_validator(mode="after")
    def validate_anchor_subpath_namespace(self) -> "RepresentationStateAnchorSelectorBinding":
        object_namespaces = {"mechanics_object", "biomechanics_object", "task_object"}
        if self.anchor_subpath is not None and self.selector.namespace not in object_namespaces:
            raise ValueError(
                "representation anchor_subpath is only valid for mechanics_object, "
                "biomechanics_object, or task_object selectors"
            )
        return self


class RepresentationTrialSpecPathBinding(RepresentationContractModel):
    """Bind a representation value to a trial-spec path."""

    kind: Literal["trial_spec_path"] = "trial_spec_path"
    path: str
    expected_type: Optional[str] = None
    dim: Optional[int] = None


class RepresentationLiteralBinding(RepresentationContractModel):
    """Bind a representation value to a JSON literal."""

    kind: Literal["literal"] = "literal"
    value: ParamValue
    dim: Optional[int] = None


RepresentationBinding = Union[
    RepresentationParamPathBinding,
    RepresentationStateAnchorSelectorBinding,
    RepresentationTrialSpecPathBinding,
    RepresentationLiteralBinding,
]


class RepresentationFrameProvider(RepresentationContractModel):
    """Declare how a representation frame is resolved."""

    kind: Literal["fixed", "from_input_port", "registered_renderer"]
    frame: Optional[str] = None
    input_port: Optional[str] = None
    renderer_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provider(self) -> "RepresentationFrameProvider":
        if self.kind == "fixed" and not self.frame:
            raise ValueError("fixed frame providers require frame")
        if self.kind == "from_input_port" and not self.input_port:
            raise ValueError("from_input_port frame providers require input_port")
        if self.kind == "registered_renderer" and not self.renderer_id:
            raise ValueError("registered_renderer frame providers require renderer_id")
        return self


class RepresentationPlanarChainSpec(RepresentationContractModel):
    """Typed kinematics capability exposed by a planar-chain element."""

    frame_ids: List[str] = Field(min_length=2)
    pose_fallback: Optional[Literal["zero"]] = None

    @model_validator(mode="after")
    def validate_frame_ids(self) -> "RepresentationPlanarChainSpec":
        duplicate_frames = sorted(_duplicates(self.frame_ids))
        if duplicate_frames:
            raise ValueError(f"Duplicate planar-chain frame ids: {duplicate_frames!r}")
        if self.frame_ids[0] != "world":
            raise ValueError("planar-chain frame_ids must start with 'world'")
        return self


class RepresentationAnchorSpec(RepresentationContractModel):
    """Named semantic anchor exposed by a component representation."""

    id: str
    semantic_role: RepresentationSemanticRole
    interaction_roles: List[RepresentationInteractionRole] = Field(default_factory=list)
    label: Optional[str] = None
    binding: Optional[RepresentationBinding] = None
    frame: Optional[str] = None
    units: Optional[str] = None
    dim: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RepresentationStyleSpec(RepresentationContractModel):
    """Style value or binding for one renderer channel."""

    channel: RepresentationStyleChannel
    value: Optional[ParamValue] = None
    binding: Optional[RepresentationBinding] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_style_source(self) -> "RepresentationStyleSpec":
        if self.value is None and self.binding is None:
            raise ValueError("representation style channels require value or binding")
        return self


class RepresentationElementSpec(RepresentationContractModel):
    """One typed representation element in a component representation."""

    id: str
    archetype: RepresentationArchetype
    anchors: List[str] = Field(default_factory=list)
    bindings: Dict[str, RepresentationBinding] = Field(default_factory=dict)
    style: List[RepresentationStyleSpec] = Field(default_factory=list)
    frame: Optional[str] = None
    frame_provider: Optional[RepresentationFrameProvider] = None
    units: Optional[str] = None
    dim: Optional[int] = None
    scale_invariant: bool = False
    renderer_id: Optional[str] = None
    planar_chain: Optional[RepresentationPlanarChainSpec] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_registered_renderer(self) -> "RepresentationElementSpec":
        if self.archetype == "registered_renderer" and not self.renderer_id:
            raise ValueError("registered_renderer representation elements require renderer_id")
        if self.planar_chain is not None and self.archetype != "planar_chain":
            raise ValueError("planar_chain capability is only valid on planar_chain elements")
        return self


class RepresentationReachabilitySpec(RepresentationContractModel):
    """Declare a component-owned radial reach envelope for workspace validation."""

    kind: Literal["radial"] = "radial"
    origin_anchor: str
    radius_binding: Union[RepresentationParamPathBinding, RepresentationLiteralBinding]
    radius_transform: Literal["identity", "sum_abs"] = "identity"
    label: Optional[str] = None
    units: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RepresentationMusclePathPointSpec(RepresentationContractModel):
    """One body-local attachment point in a muscle path."""

    frame: str
    position: List[float] = Field(min_length=2, max_length=2)


class RepresentationMusclePathSpec(RepresentationContractModel):
    """Ordered attachment points for one rendered muscle path."""

    id: str
    points: List[RepresentationMusclePathPointSpec] = Field(min_length=2)


class RepresentationMusclePathGeometrySpec(RepresentationContractModel):
    """Provider-owned body-local attachment topology for muscle paths."""

    schema_id: Literal[MUSCLE_PATH_GEOMETRY_SCHEMA_ID] = MUSCLE_PATH_GEOMETRY_SCHEMA_ID
    schema_version: Literal[MUSCLE_PATH_GEOMETRY_SCHEMA_VERSION] = (
        MUSCLE_PATH_GEOMETRY_SCHEMA_VERSION
    )
    paths: List[RepresentationMusclePathSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "RepresentationMusclePathGeometrySpec":
        duplicate_paths = sorted(_duplicates([path.id for path in self.paths]))
        if duplicate_paths:
            raise ValueError(f"Duplicate muscle path ids: {duplicate_paths!r}")
        return self


class RepresentationSpec(RepresentationContractModel):
    """Versioned representation declaration served through component metadata."""

    schema_id: Literal[REPRESENTATION_SCHEMA_ID] = REPRESENTATION_SCHEMA_ID
    schema_version: Literal[REPRESENTATION_SCHEMA_VERSION] = REPRESENTATION_SCHEMA_VERSION
    anchors: List[RepresentationAnchorSpec] = Field(default_factory=list)
    elements: List[RepresentationElementSpec] = Field(default_factory=list)
    style: List[RepresentationStyleSpec] = Field(default_factory=list)
    frame: Optional[str] = None
    frame_provider: Optional[RepresentationFrameProvider] = None
    units: Optional[str] = None
    dim: Optional[int] = None
    scale_invariant: bool = False
    reachability: Optional[RepresentationReachabilitySpec] = None
    muscle_path_geometry: Optional[RepresentationMusclePathGeometrySpec] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def canonicalize_and_validate(self) -> "RepresentationSpec":
        anchor_ids = [anchor.id for anchor in self.anchors]
        duplicate_anchor_ids = sorted(_duplicates(anchor_ids))
        if duplicate_anchor_ids:
            raise ValueError(f"Duplicate representation anchor ids: {duplicate_anchor_ids!r}")

        element_ids = [element.id for element in self.elements]
        duplicate_element_ids = sorted(_duplicates(element_ids))
        if duplicate_element_ids:
            raise ValueError(f"Duplicate representation element ids: {duplicate_element_ids!r}")

        known_anchors = set(anchor_ids)
        if self.reachability is not None and self.reachability.origin_anchor not in known_anchors:
            raise ValueError(
                "Representation reachability origin_anchor references an unknown anchor: "
                f"{self.reachability.origin_anchor!r}"
            )
        for element in self.elements:
            missing = sorted(anchor for anchor in element.anchors if anchor not in known_anchors)
            if missing:
                raise ValueError(
                    f"Representation element {element.id!r} references unknown anchors: {missing!r}"
                )

        self.anchors = sorted(self.anchors, key=lambda anchor: anchor.id)
        self.elements = sorted(self.elements, key=lambda element: element.id)
        self.style = sorted(self.style, key=lambda item: item.channel)
        for element in self.elements:
            element.style = sorted(element.style, key=lambda item: item.channel)
        return self


class RepresentationValidationIssue(RepresentationContractModel):
    """Catalog-load issue for a component representation declaration."""

    type: str
    message: str
    path: str


def validate_representation_against_component(
    representation: RepresentationSpec,
    *,
    component_name: str,
    param_schema: List[ParamSchema],
    input_ports: List[str],
) -> List[RepresentationValidationIssue]:
    """Validate static representation bindings known at component-catalog load."""

    issues: list[RepresentationValidationIssue] = []
    param_names = {schema.name for schema in param_schema}
    input_port_names = set(input_ports)

    for path, binding in _iter_bindings(representation):
        if isinstance(binding, RepresentationParamPathBinding):
            root = binding.path.split(".", 1)[0].split("[", 1)[0]
            if root not in param_names:
                issues.append(
                    RepresentationValidationIssue(
                        type="unknown_param_path",
                        message=(
                            f"{component_name!r} representation binding {binding.path!r} "
                            f"does not match any declared parameter"
                        ),
                        path=path,
                    )
                )

    for path, provider in _iter_frame_providers(representation):
        if provider.kind == "from_input_port" and provider.input_port not in input_port_names:
            issues.append(
                RepresentationValidationIssue(
                    type="unknown_input_port",
                    message=(
                        f"{component_name!r} representation frame provider references "
                        f"input port {provider.input_port!r}, but declared inputs are "
                        f"{sorted(input_port_names)!r}"
                    ),
                    path=path,
                )
            )

    return issues


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _iter_bindings(spec: RepresentationSpec) -> list[tuple[str, RepresentationBinding]]:
    bindings: list[tuple[str, RepresentationBinding]] = []
    if spec.reachability is not None:
        bindings.append(("/reachability/radius_binding", spec.reachability.radius_binding))
    for index, anchor in enumerate(spec.anchors):
        if anchor.binding is not None:
            bindings.append((f"/anchors/{index}/binding", anchor.binding))
    for index, element in enumerate(spec.elements):
        for name, binding in element.bindings.items():
            bindings.append((f"/elements/{index}/bindings/{name}", binding))
        for style_index, style in enumerate(element.style):
            if style.binding is not None:
                bindings.append((f"/elements/{index}/style/{style_index}/binding", style.binding))
    for index, style in enumerate(spec.style):
        if style.binding is not None:
            bindings.append((f"/style/{index}/binding", style.binding))
    return bindings


def _iter_frame_providers(spec: RepresentationSpec) -> list[tuple[str, RepresentationFrameProvider]]:
    providers: list[tuple[str, RepresentationFrameProvider]] = []
    if spec.frame_provider is not None:
        providers.append(("/frame_provider", spec.frame_provider))
    for index, element in enumerate(spec.elements):
        if element.frame_provider is not None:
            providers.append((f"/elements/{index}/frame_provider", element.frame_provider))
    return providers
