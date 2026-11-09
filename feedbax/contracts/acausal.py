"""Durable contract models for acausal graph interiors."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedbax.contracts.array_values import (
    ARRAY_VALUE_SCHEMA_VERSION,
    _parse_array_value_payload,
)
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.base import StrictModel

if TYPE_CHECKING:
    from feedbax.contracts.graph import ComponentSpec, GraphMetadata


ACAUSAL_GRAPH_SCHEMA_ID = "feedbax.spec.acausal_graph"
ACAUSAL_GRAPH_SCHEMA_VERSION_V1 = "feedbax.spec.acausal_graph.v1"
ACAUSAL_GRAPH_SCHEMA_VERSION = "feedbax.spec.acausal_graph.v2"
AcausalPhysicalDomain = Literal[
    "translational", "rotational", "planar_multibody", "spatial_multibody"
]
SolverType = Literal["euler", "implicit_euler", "kvaerno5", "tsit5", "mujoco"]


class AcausalConnectionSpec(BaseModel):
    """Undirected connection between two acausal component ports."""

    model_config = ConfigDict(extra="forbid")

    a: Tuple[str, str]
    b: Tuple[str, str]

    @model_validator(mode="after")
    def validate_and_canonicalize(self) -> "AcausalConnectionSpec":
        if self.a == self.b:
            raise ValueError("AcausalConnectionSpec endpoints must be distinct")
        if self.b < self.a:
            return self.model_copy(update={"a": self.b, "b": self.a})
        return self


class RootFinderSpec(BaseModel):
    """Minimal root-finder settings persisted with an acausal solver."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["newton"] = "newton"
    rtol: float = 1e-6
    atol: float = 1e-6
    max_steps: int = 32


class SolverConfigSpec(BaseModel):
    """Solver settings for the executable acausal interior.

    Nested scalar interiors flatten into their enclosing system and inherit
    the outer graph's solver. Scalar implicit solvers use Newton at tolerances
    of 1e-6 and Diffrax's iteration budget when ``root_finder`` is omitted.
    Supplying a root-finder block configures both tolerances and the budget,
    including the block's defaults for any omitted fields.
    """

    model_config = ConfigDict(extra="forbid")

    solver_type: SolverType
    dt: float
    root_finder: Optional[RootFinderSpec] = None
    planar_substeps: int | None = Field(default=None, ge=1, strict=True)


class JointDampingMode(StrictModel):
    """Passive damping of a weighted sum of named planar joint velocities."""

    model_config = ConfigDict(allow_inf_nan=False)

    coefficients: dict[str, float] = Field(min_length=1)
    damping: float = Field(gt=0)


class PlanarDynamicsSpec(StrictModel):
    """Optional dynamics for an authored planar chain."""

    model_config = ConfigDict(allow_inf_nan=False)

    gravity: tuple[float, float] = (0.0, -9.81)
    joint_dampers: dict[str, JointDampingMode] = Field(default_factory=dict)


class SpatialSite(StrictModel):
    """A named overlay attachment in a native body's local coordinates, in m."""

    model_config = ConfigDict(allow_inf_nan=False)

    body: str = Field(min_length=1)
    position: tuple[float, float, float]


class MuscleWrap(StrictModel):
    """Wrap one segment between consecutive sites around a named geometry."""

    after_site: int = Field(ge=0, strict=True)
    geom: str = Field(min_length=1)
    side_site: str = Field(min_length=1)


class PlanarWrapGeometry(StrictModel):
    """An XY sphere or Z-axis cylinder, local to a named planar frame."""

    model_config = ConfigDict(allow_inf_nan=False)

    kind: Literal["sphere", "cylinder"]
    frame: str = Field(min_length=1)
    radius: float = Field(gt=0)
    offset: tuple[float, float] = (0.0, 0.0)


class SpatialMusclePath(StrictModel):
    """A path through native or overlay sites, with its external muscle law."""

    sites: tuple[str, ...] = Field(min_length=2)
    wraps: tuple[MuscleWrap, ...] = ()
    law: str = Field(min_length=1)
    state_size: int = Field(ge=0, strict=True)
    law_revision: int | None = Field(default=None, ge=1, strict=True)
    parameters: dict[str, Any] = Field(default_factory=dict)


class MuJoCoAssetBinding(StrictModel):
    """A self-contained MJCF asset and graph-owned, named overlays."""

    asset: ContentObjectRef
    effector_site: str = Field(min_length=1)
    sites: dict[str, SpatialSite] = Field(default_factory=dict)
    paths: dict[str, SpatialMusclePath] = Field(default_factory=dict)


class AcausalGraphSpec(BaseModel):
    """Durable acausal graph interior specification."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[ACAUSAL_GRAPH_SCHEMA_ID] = ACAUSAL_GRAPH_SCHEMA_ID
    schema_version: Literal[ACAUSAL_GRAPH_SCHEMA_VERSION] = ACAUSAL_GRAPH_SCHEMA_VERSION
    physical_domain: AcausalPhysicalDomain
    nodes: Dict[str, "ComponentSpec"] = Field(default_factory=dict)
    connections: List[AcausalConnectionSpec] = Field(default_factory=list)
    solver: SolverConfigSpec
    subgraphs: Optional[Dict[str, "AcausalGraphSpec"]] = None
    metadata: Optional["GraphMetadata"] = None
    mujoco_asset: MuJoCoAssetBinding | None = None
    planar_dynamics: PlanarDynamicsSpec | None = None
    wrap_geometries: dict[str, PlanarWrapGeometry] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def validate_subgraph_family(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw_subgraphs = data.get("subgraphs")
        if not isinstance(raw_subgraphs, dict):
            return data
        subgraphs: dict[str, AcausalGraphSpec] = {}
        for node_id, raw_subgraph in raw_subgraphs.items():
            if isinstance(raw_subgraph, AcausalGraphSpec):
                subgraphs[str(node_id)] = raw_subgraph
                continue
            if not isinstance(raw_subgraph, dict):
                raise ValueError(
                    "AcausalGraphSpec.subgraphs values must be acausal graph payloads: "
                    f"node_id={node_id!r}, value_type={type(raw_subgraph).__name__}"
                )
            schema_id = raw_subgraph.get("schema_id", ACAUSAL_GRAPH_SCHEMA_ID)
            if schema_id != ACAUSAL_GRAPH_SCHEMA_ID:
                raise ValueError(
                    "AcausalGraphSpec nested subgraphs must use schema_id "
                    f"{ACAUSAL_GRAPH_SCHEMA_ID!r}: node_id={node_id!r}, "
                    f"schema_id={schema_id!r}"
                )
            subgraphs[str(node_id)] = AcausalGraphSpec.model_validate(raw_subgraph)
        return {**data, "subgraphs": subgraphs}

    @model_validator(mode="after")
    def reject_component_param_array_values(self) -> "AcausalGraphSpec":
        if (self.physical_domain == "spatial_multibody") != (self.mujoco_asset is not None):
            raise ValueError("A MuJoCo asset binding is required exactly for spatial multibody")
        if self.physical_domain != "planar_multibody" and (
            self.planar_dynamics is not None or self.solver.planar_substeps is not None
        ):
            raise ValueError("Planar dynamics and substeps require planar multibody")
        if self.wrap_geometries and self.physical_domain != "planar_multibody":
            raise ValueError("Wrap geometry overlays require a planar multibody graph")
        for node_id, node in self.nodes.items():
            for param_name, value in node.params.items():
                if _contains_array_value(value):
                    raise ValueError(
                        "AcausalGraphSpec does not support component-param array value "
                        f"schema {ARRAY_VALUE_SCHEMA_VERSION!r}: "
                        f"nodes[{node_id!r}].params[{param_name!r}]. The acausal compiler "
                        "must define a separate materialization boundary before adopting it."
                    )
        return self


def _contains_array_value(value: Any) -> bool:
    if _parse_array_value_payload(value) is not None:
        return True
    if isinstance(value, dict):
        return any(_contains_array_value(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_array_value(item) for item in value)
    return False


def _browser_json_value(value: Any) -> Any:
    """Admit decimal integer spellings of a browser's finite numbers."""
    if isinstance(value, dict):
        return {key: _browser_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_browser_json_value(item) for item in value]
    if type(value) is int and 2**53 <= abs(value) < 10**21:
        number = float(value)
        if canonical_json_v2_bytes(number) == str(value).encode():
            return number
    return value


def _canonicalize_report_payload(payload: dict[str, Any], authored: dict[str, Any]) -> None:
    # Endpoint tuples use the same UTF-16 order as the browser and canonical JSON v2.
    def endpoint_key(endpoint: list[str]) -> tuple[bytes, ...]:
        return tuple(part.encode("utf-16-be", errors="surrogatepass") for part in endpoint)

    if "connections" in payload:
        pairs = [
            sorted((connection["a"], connection["b"]), key=endpoint_key)
            for connection in payload["connections"]
        ]
        pairs.sort(key=lambda pair: (endpoint_key(pair[0]), endpoint_key(pair[1])))
        payload["connections"] = [{"a": left, "b": right} for left, right in pairs]

    for node_id, node in authored.get("nodes", {}).items():
        # Parameters are authored JSON. Component validation can materialize typed
        # parameter declarations before compilation.
        if params := node.get("params"):
            payload["nodes"][node_id]["params"] = params

    for node_id, child in (authored.get("subgraphs") or {}).items():
        _canonicalize_report_payload(payload["subgraphs"][node_id], child)


def canonical_acausal_graph_payload(
    graph: AcausalGraphSpec, *, authored: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Project an admitted graph onto the Studio authoring report subject."""
    payload = graph.model_dump(mode="json", exclude_defaults=True)
    _canonicalize_report_payload(payload, payload if authored is None else authored)
    return _browser_json_value(payload)


def canonical_acausal_graph_json(
    graph: AcausalGraphSpec, *, authored: dict[str, Any] | None = None
) -> str:
    """Return canonical JSON for an acausal authoring report subject."""
    payload = canonical_acausal_graph_payload(graph, authored=authored)
    return canonical_json_v2_bytes(payload).decode("utf-8")


def acausal_interior_content_hash(
    graph: AcausalGraphSpec, *, authored: dict[str, Any] | None = None
) -> str:
    """Identify a Studio report subject independently of scientific execution evidence."""
    return hashlib.sha256(
        canonical_acausal_graph_json(graph, authored=authored).encode()
    ).hexdigest()
