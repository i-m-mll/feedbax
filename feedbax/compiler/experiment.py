"""Authoritative experiment and campaign documents owned by the compiler boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.scientific_compiler_schema import (
    CAMPAIGN_DOCUMENT_SCHEMA_ID,
    CAMPAIGN_DOCUMENT_SCHEMA_VERSION,
    EXPERIMENT_DOCUMENT_SCHEMA_ID,
    EXPERIMENT_DOCUMENT_SCHEMA_VERSION,
    RESOLVED_EXPERIMENT_SCHEMA_ID,
    RESOLVED_EXPERIMENT_SCHEMA_VERSION,
)

from .graph import (
    GRAPH_DOCUMENT_SCHEMA_ID,
    DocumentRoot,
    ResolvedGraph,
    _require_version,
)


class DeclarationRef(BaseModel):
    """Exact neutral declaration identity used by an authored scientific root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["trial_source", "objective", "training_program"]
    type_id: str = Field(min_length=1)
    schema_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    owner: str = Field(min_length=1)


class ScientificSeedDomain(BaseModel):
    """One named scientific random domain and its authored root seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: str = Field(min_length=1)
    root_seed: int = Field(ge=0, le=2**64 - 1)


class ExperimentDocument(BaseModel):
    """Complete authored binding for one scientific experiment family member."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[EXPERIMENT_DOCUMENT_SCHEMA_ID] = EXPERIMENT_DOCUMENT_SCHEMA_ID
    schema_version: str = EXPERIMENT_DOCUMENT_SCHEMA_VERSION
    document_id: str = Field(min_length=1)
    graph: DocumentRoot
    trial_source: DeclarationRef
    objective: DeclarationRef
    training_program: DeclarationRef
    observation_policy: DocumentRoot
    scientific_seeds: tuple[ScientificSeedDomain, ...]

    @field_validator("schema_version")
    @classmethod
    def reject_unsupported_version(cls, value: str) -> str:
        return _require_version(
            "ExperimentDocument", value, EXPERIMENT_DOCUMENT_SCHEMA_VERSION
        )

    @model_validator(mode="after")
    def validate_bindings(self) -> "ExperimentDocument":
        expected_kinds = {
            "trial_source": self.trial_source.kind,
            "objective": self.objective.kind,
            "training_program": self.training_program.kind,
        }
        mismatches = [name for name, kind in expected_kinds.items() if name != kind]
        if mismatches:
            raise ValueError(f"ExperimentDocument declaration kind mismatch: {mismatches!r}")
        domains = [seed.domain for seed in self.scientific_seeds]
        if not domains:
            raise ValueError("ExperimentDocument requires at least one scientific seed domain")
        if len(domains) != len(set(domains)):
            raise ValueError("ExperimentDocument scientific seed domains must be unique")
        return self


class CampaignVariant(BaseModel):
    """One exact experiment revision admitted to a finite campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant_id: str = Field(min_length=1)
    experiment: DocumentRoot


class RequestedProduct(BaseModel):
    """One requested finite workflow product keyed by its operation declaration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    product_id: str = Field(min_length=1)
    operation_type_id: str = Field(min_length=1)


class BoundedBranch(BaseModel):
    """One closed campaign branch with every possible outcome named up front."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    branch_id: str = Field(min_length=1)
    predicate_ref: str = Field(min_length=1)
    outcomes: tuple[str, ...]

    @field_validator("outcomes")
    @classmethod
    def require_closed_unique_outcomes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) < 2 or any(not outcome for outcome in value):
            raise ValueError("BoundedBranch requires at least two named outcomes")
        if len(value) != len(set(value)):
            raise ValueError("BoundedBranch outcomes must be unique")
        return value


class CampaignDocument(BaseModel):
    """Finite variants, products, and closed branches for one campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[CAMPAIGN_DOCUMENT_SCHEMA_ID] = CAMPAIGN_DOCUMENT_SCHEMA_ID
    schema_version: str = CAMPAIGN_DOCUMENT_SCHEMA_VERSION
    document_id: str = Field(min_length=1)
    variants: tuple[CampaignVariant, ...]
    requested_products: tuple[RequestedProduct, ...]
    branches: tuple[BoundedBranch, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def reject_unsupported_version(cls, value: str) -> str:
        return _require_version("CampaignDocument", value, CAMPAIGN_DOCUMENT_SCHEMA_VERSION)

    @model_validator(mode="after")
    def validate_finite_identities(self) -> "CampaignDocument":
        groups = {
            "variants": [variant.variant_id for variant in self.variants],
            "requested_products": [product.product_id for product in self.requested_products],
            "branches": [branch.branch_id for branch in self.branches],
        }
        for name, identities in groups.items():
            if name != "branches" and not identities:
                raise ValueError(f"CampaignDocument requires at least one {name}")
            if len(identities) != len(set(identities)):
                raise ValueError(f"CampaignDocument {name} identities must be unique")
        return self


class ResolvedExperiment(BaseModel):
    """Immutable exact graph and declaration bindings derived from one experiment root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[RESOLVED_EXPERIMENT_SCHEMA_ID] = RESOLVED_EXPERIMENT_SCHEMA_ID
    schema_version: str = RESOLVED_EXPERIMENT_SCHEMA_VERSION
    source_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph: ResolvedGraph
    trial_source: DeclarationRef
    objective: DeclarationRef
    training_program: DeclarationRef
    observation_policy: DocumentRoot
    scientific_seeds: tuple[ScientificSeedDomain, ...]

    @field_validator("schema_version")
    @classmethod
    def reject_unsupported_version(cls, value: str) -> str:
        return _require_version(
            "ResolvedExperiment", value, RESOLVED_EXPERIMENT_SCHEMA_VERSION
        )

    @model_validator(mode="after")
    def validate_identity(self) -> "ResolvedExperiment":
        if self.resolved_sha256 != _resolved_experiment_sha256(self):
            raise ValueError(
                "ResolvedExperiment resolved_sha256 does not match its canonical semantic payload"
            )
        return self


def _resolved_experiment_payload(value: ResolvedExperiment | dict[str, Any]) -> dict[str, Any]:
    payload = (
        value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, ResolvedExperiment)
        else dict(value)
    )
    return {
        key: item
        for key, item in payload.items()
        if key not in {"schema_id", "schema_version", "resolved_sha256"}
    }


def _resolved_experiment_sha256(value: ResolvedExperiment | dict[str, Any]) -> str:
    return canonical_sha256(_resolved_experiment_payload(value))


def resolve_experiment(
    document: ExperimentDocument,
    resolved_graph: ResolvedGraph,
) -> ResolvedExperiment:
    """Bind one exact experiment revision to its already resolved semantic graph."""
    if document.graph.schema_id != GRAPH_DOCUMENT_SCHEMA_ID:
        raise ValueError(
            f"ExperimentDocument graph root must identify {GRAPH_DOCUMENT_SCHEMA_ID!r}"
        )
    if document.graph.content_sha256 != resolved_graph.document_sha256:
        raise ValueError(
            "ExperimentDocument graph revision does not match ResolvedGraph document_sha256"
        )
    source_document_sha256 = canonical_sha256(
        document.model_dump(mode="json", exclude_none=True)
    )
    payload = {
        "source_document_sha256": source_document_sha256,
        "graph": resolved_graph.model_dump(mode="json", exclude_none=True),
        "trial_source": document.trial_source.model_dump(mode="json"),
        "objective": document.objective.model_dump(mode="json"),
        "training_program": document.training_program.model_dump(mode="json"),
        "observation_policy": document.observation_policy.model_dump(mode="json"),
        "scientific_seeds": [seed.model_dump(mode="json") for seed in document.scientific_seeds],
    }
    return ResolvedExperiment(
        **payload,
        resolved_sha256=_resolved_experiment_sha256(payload),
    )


__all__ = [
    "CAMPAIGN_DOCUMENT_SCHEMA_ID",
    "CAMPAIGN_DOCUMENT_SCHEMA_VERSION",
    "EXPERIMENT_DOCUMENT_SCHEMA_ID",
    "EXPERIMENT_DOCUMENT_SCHEMA_VERSION",
    "RESOLVED_EXPERIMENT_SCHEMA_ID",
    "RESOLVED_EXPERIMENT_SCHEMA_VERSION",
    "BoundedBranch",
    "CampaignDocument",
    "CampaignVariant",
    "DeclarationRef",
    "ExperimentDocument",
    "RequestedProduct",
    "ResolvedExperiment",
    "ScientificSeedDomain",
    "resolve_experiment",
]
