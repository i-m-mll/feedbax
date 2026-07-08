"""Durable contract models for acausal graph interiors."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from feedbax.contracts.graph import ComponentSpec, GraphMetadata


ACAUSAL_GRAPH_SCHEMA_ID = "feedbax.spec.acausal_graph"
ACAUSAL_GRAPH_SCHEMA_VERSION = "feedbax.spec.acausal_graph.v1"
ACAUSAL_PHYSICAL_DOMAINS = ("translational", "rotational", "planar_multibody")
AcausalPhysicalDomain = Literal["translational", "rotational", "planar_multibody"]
SolverType = Literal["euler", "implicit_euler", "kvaerno5", "tsit5"]


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
    """Acausal solver configuration persisted as contract data."""

    model_config = ConfigDict(extra="forbid")

    solver_type: SolverType
    dt: float
    root_finder: Optional[RootFinderSpec] = None


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


def canonical_acausal_graph_payload(graph: AcausalGraphSpec) -> dict[str, Any]:
    """Return a stable JSON payload for acausal interior hashing."""

    payload = graph.model_dump(mode="json", exclude_none=True)
    payload["connections"] = sorted(
        payload.get("connections", []),
        key=lambda item: (tuple(item["a"]), tuple(item["b"])),
    )
    subgraphs = payload.get("subgraphs")
    if isinstance(subgraphs, dict):
        payload["subgraphs"] = {
            key: canonical_acausal_graph_payload(graph.subgraphs[key])  # type: ignore[index]
            for key in sorted(subgraphs)
            if graph.subgraphs is not None
        }
    return payload


def canonical_acausal_graph_json(graph: AcausalGraphSpec) -> str:
    """Return canonical JSON for an acausal graph interior."""

    return json.dumps(
        canonical_acausal_graph_payload(graph),
        sort_keys=True,
        separators=(",", ":"),
    )


def acausal_interior_content_hash(graph: AcausalGraphSpec) -> str:
    """Return the sha256 hash used for acausal authoring compile caches."""

    return hashlib.sha256(canonical_acausal_graph_json(graph).encode("utf-8")).hexdigest()
