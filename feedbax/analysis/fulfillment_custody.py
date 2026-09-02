"""Transactional repair and rebuild-as-verification for fulfillment receipts.

Admission failure is refusal, never a cache miss. This module holds the two
explicit operations that follow from that refusal.

**Repair** is the recovery path for a receipt that exists but fails admission.
It quarantines the failed bytes keyed by their *observed* SHA-256, executes a
candidate into a shadow root, validates the candidate completely through the
same per-kind admission validator, promotes into the authoritative location only
on full success, and emits a durable repair record linking the old bytes to
their replacement. The authoritative manifest is published by write-then-rename
as the final step, so a crash mid-repair never leaves the authoritative path
holding an unvalidated candidate.

**Rebuild** is verification, not maintenance. It executes a node into shadow
custody without touching the authoritative receipt and compares a defined
deterministic output projection: artifact roles, names, media types, SHA-256,
sizes, products, and summaries. It never compares manifest bytes, because a
manifest carries ``created_at`` and is not byte-reproducible. On equality the
original receipt is preserved untouched, so its authenticated ``ParentRef``s
stay stable; on difference the operation fails with a per-node drift report and
the original receipt intact. Corrupted stored bytes are an admission failure
before any rebuild comparison happens: that is custody loss, not drift.

Rebuild reports what the executors really produce, and the projection
special-cases no producer. A stored artifact whose digest the receipt
authenticates is exactly what rebuild exists to verify, and excluding one would
make verification blind to real changes in the same bytes. When a producer is
not byte-reproducible, the fix belongs in the producer, not in an exclusion
here: ``save_figure_with_spec`` formerly stamped the ``figure_spec`` sidecar
with the wall clock and so drifted on every intact figure node, and it was
fixed by removing that field from the authenticated bytes rather than by
exempting the artifact from comparison.

Shadow custody mirrors the authoritative root's readable bytes so parents
resolve identically in both roots. Mirroring copies; it never hard-links,
because a hard-linked canonical artifact name would make the secure
content-addressed store refuse to publish over an aliased name.

Stability note: this module is not part of the owner-ratified downstream
inventory in ``docs/design/downstream_interface_stability.md``. The
``feedbax.fulfillment.output_projection`` and
``feedbax.fulfillment.repair_record`` families nonetheless carry explicit schema
identity and reject unsupported versions, because a drift report crosses a
process boundary and a repair record is written to durable storage.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal
import uuid

from pydantic import Field, model_validator

from feedbax.analysis.fulfillment import (
    AdmissionOutcome,
    AdmittedManifestBytes,
    FulfillmentAdmissionError,
    FulfillmentNodeKind,
    FulfillmentReceipt,
    artifact_bytes_path,
    resolve_artifact_bytes,
)
from feedbax.analysis.fulfillment_adapters import (
    EvaluationMatrixNodeRequest,
    FulfillmentEnvironment,
    NodeRequest,
    admit_node,
    canonical_fulfillment_order,
    execute_node,
    expand_evaluation_matrix_node,
)
from feedbax.contracts.base import (
    ArtifactRef,
    StrictModel,
    sha256_bytes,
    utc_now,
)
from feedbax.contracts.artifact_store import store_bytes_artifact
from feedbax.contracts.manifest import (
    AnalysisDataProduct,
    AnyManifest,
    load_manifest_bytes,
    normalize_manifest_spec_payloads,
    safe_manifest_key,
)
from feedbax.persistence.manifest_index import index_manifest_file


FULFILLMENT_PROJECTION_SCHEMA_ID = "feedbax.fulfillment.output_projection"
FULFILLMENT_PROJECTION_SCHEMA_VERSION = "feedbax.fulfillment.output_projection.v1"
#: Versions this loader accepts, mapped to the version they migrate to.
FULFILLMENT_PROJECTION_MIGRATION_TABLE: dict[str, str] = {}

FULFILLMENT_REPAIR_SCHEMA_ID = "feedbax.fulfillment.repair_record"
FULFILLMENT_REPAIR_SCHEMA_VERSION = "feedbax.fulfillment.repair_record.v1"
#: Versions this loader accepts, mapped to the version they migrate to.
FULFILLMENT_REPAIR_MIGRATION_TABLE: dict[str, str] = {}

#: Root-relative directory holding quarantined bytes, keyed by observed SHA-256.
FULFILLMENT_QUARANTINE_DIRECTORY = "quarantine"

#: Root-relative directory holding durable repair records.
FULFILLMENT_REPAIR_DIRECTORY = "repairs"

#: Root-relative entries a shadow root never mirrors from its authoritative root.
#: The manifest index stores absolute paths, so a copied index would resolve
#: shadow lookups back to authoritative bytes.
SHADOW_MIRROR_EXCLUDED: frozenset[str] = frozenset(
    {"index", FULFILLMENT_QUARANTINE_DIRECTORY, FULFILLMENT_REPAIR_DIRECTORY}
)


class ArtifactProjection(StrictModel):
    """The deterministic output identity of one artifact.

    ``uri`` and ``metadata`` are excluded: both are machine-local and differ
    between an authoritative root and a shadow root by construction.
    """

    role: str
    logical_name: str
    media_type: str
    sha256: str | None = None
    size_bytes: int | None = None


class ProductProjection(StrictModel):
    """The deterministic output identity of one analysis data product.

    ``producer_manifest_hash``, ``parent_manifests[].manifest_hash``, and
    ``product_identity_hash`` are excluded because they hash manifest bytes,
    which carry ``created_at`` and are therefore not byte-reproducible.
    ``materialization`` is excluded because it records machine-local storage
    locations.
    """

    role: str
    logical_name: str
    product_schema_id: str
    product_schema_version: str
    label: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    checkpoint_policy: dict[str, Any] = Field(default_factory=dict)
    rollout_policy: dict[str, Any] = Field(default_factory=dict)
    descriptor_basis_hash: str | None = None
    artifacts: list[ArtifactProjection] = Field(default_factory=list)


class OutputProjection(StrictModel):
    """The canonical comparable structure one executed node produces.

    This is the whole of what a rebuild compares. It contains exactly the
    node's content-minted identity, its declared status, its artifact byte
    identities, its typed data products, and its recorded summaries. It contains
    no manifest bytes, no timestamps, and no machine-local location.
    """

    schema_id: Literal["feedbax.fulfillment.output_projection"] = (
        FULFILLMENT_PROJECTION_SCHEMA_ID
    )
    schema_version: Literal["feedbax.fulfillment.output_projection.v1"] = (
        FULFILLMENT_PROJECTION_SCHEMA_VERSION
    )
    node_kind: FulfillmentNodeKind
    manifest_kind: str
    manifest_id: str
    status: str | None = None
    artifacts: list[ArtifactProjection] = Field(default_factory=list)
    products: list[ProductProjection] = Field(default_factory=list)
    summaries: dict[str, Any] = Field(default_factory=dict)


class ProjectionDifference(StrictModel):
    """One projection field on which a rebuild disagreed with the receipt."""

    schema_id: Literal["feedbax.fulfillment.output_projection"] = (
        FULFILLMENT_PROJECTION_SCHEMA_ID
    )
    schema_version: Literal["feedbax.fulfillment.output_projection.v1"] = (
        FULFILLMENT_PROJECTION_SCHEMA_VERSION
    )
    field_path: str
    expected: Any = None
    observed: Any = None


class NodeRebuildOutcome(StrictModel):
    """The structured result of rebuilding one node as verification."""

    schema_id: Literal["feedbax.fulfillment.output_projection"] = (
        FULFILLMENT_PROJECTION_SCHEMA_ID
    )
    schema_version: Literal["feedbax.fulfillment.output_projection.v1"] = (
        FULFILLMENT_PROJECTION_SCHEMA_VERSION
    )
    node_key: str
    node_kind: FulfillmentNodeKind
    manifest_id: str
    matched: bool
    expected: OutputProjection
    observed: OutputProjection
    differences: list[ProjectionDifference] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_decision(self) -> "NodeRebuildOutcome":
        if self.matched and self.differences:
            raise ValueError("a matched rebuild cannot carry projection differences")
        if not self.matched and not self.differences:
            raise ValueError("a drifted rebuild must name at least one projection difference")
        return self

    @property
    def field_paths(self) -> tuple[str, ...]:
        """Return the differing projection fields in deterministic order."""
        return tuple(difference.field_path for difference in self.differences)

    def describe(self) -> str:
        """Return a stable one-line description of the comparison."""
        if self.matched:
            return f"{self.node_kind} node {self.node_key} rebuilt with zero drift"
        fields = "; ".join(self.field_paths)
        return f"{self.node_kind} node {self.node_key} drifted: {fields}"


class QuarantinedBytes(StrictModel):
    """One byte stream a repair preserved before replacing it."""

    schema_id: Literal["feedbax.fulfillment.repair_record"] = FULFILLMENT_REPAIR_SCHEMA_ID
    schema_version: Literal["feedbax.fulfillment.repair_record.v1"] = (
        FULFILLMENT_REPAIR_SCHEMA_VERSION
    )
    origin: Literal["manifest", "artifact"]
    role: str | None = None
    logical_name: str | None = None
    declared_sha256: str | None = None
    observed_sha256: str
    size_bytes: int
    source_relative_path: str
    quarantine_relative_path: str


class RepairRecord(StrictModel):
    """Durable provenance linking failed receipt bytes to their replacement.

    Timestamps are recorded here deliberately: a repair record is provenance,
    not identity. The record's own storage location is minted from the observed
    SHA-256 of the failed manifest bytes, so repairing the same failed bytes
    twice addresses the same record.
    """

    schema_id: Literal["feedbax.fulfillment.repair_record"] = FULFILLMENT_REPAIR_SCHEMA_ID
    schema_version: Literal["feedbax.fulfillment.repair_record.v1"] = (
        FULFILLMENT_REPAIR_SCHEMA_VERSION
    )
    node_key: str
    node_kind: FulfillmentNodeKind
    manifest_kind: str
    manifest_id: str
    manifest_relative_path: str
    triggering_admission: AdmissionOutcome
    quarantined: list[QuarantinedBytes] = Field(default_factory=list)
    replacement_manifest_sha256: str
    replacement_artifacts: list[ArtifactProjection] = Field(default_factory=list)
    admission_after_repair: AdmissionOutcome
    started_at: datetime
    completed_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_decision(self) -> "RepairRecord":
        if self.triggering_admission.admitted:
            raise ValueError("a repair record must record the admission failure that triggered it")
        if not self.admission_after_repair.admitted:
            raise ValueError("a repair record is written only after the replacement admits")
        return self


def migrate_output_projection(payload: OutputProjection | Mapping[str, Any]) -> OutputProjection:
    """Load a current output projection or reject an unsupported version.

    Unknown and removed versions fail closed with the family, current version,
    and migration table named. There is no inference and no compatibility shim.
    """
    if isinstance(payload, OutputProjection):
        return OutputProjection.model_validate(payload.model_dump(mode="json"))
    data = dict(payload)
    _require_schema_identity(
        data,
        schema_id=FULFILLMENT_PROJECTION_SCHEMA_ID,
        schema_version=FULFILLMENT_PROJECTION_SCHEMA_VERSION,
        migration_table=FULFILLMENT_PROJECTION_MIGRATION_TABLE,
        label="OutputProjection",
    )
    return OutputProjection.model_validate(data)


def migrate_repair_record(payload: RepairRecord | Mapping[str, Any]) -> RepairRecord:
    """Load a current repair record or reject an unsupported version."""
    if isinstance(payload, RepairRecord):
        return RepairRecord.model_validate(payload.model_dump(mode="json"))
    data = dict(payload)
    _require_schema_identity(
        data,
        schema_id=FULFILLMENT_REPAIR_SCHEMA_ID,
        schema_version=FULFILLMENT_REPAIR_SCHEMA_VERSION,
        migration_table=FULFILLMENT_REPAIR_MIGRATION_TABLE,
        label="RepairRecord",
    )
    return RepairRecord.model_validate(data)


def _require_schema_identity(
    data: Mapping[str, Any],
    *,
    schema_id: str,
    schema_version: str,
    migration_table: Mapping[str, str],
    label: str,
) -> None:
    if data.get("schema_id") != schema_id:
        raise ValueError(
            f"unsupported {label} schema_id: {data.get('schema_id')!r}; expected {schema_id!r}"
        )
    version = data.get("schema_version")
    if version != schema_version:
        raise ValueError(
            f"unsupported {label} schema_version: {version!r}; expected {schema_version!r}; "
            f"migration table={dict(migration_table)!r}"
        )


class FulfillmentDriftError(RuntimeError):
    """Raised when a rebuild produced a different output projection than the receipt."""

    def __init__(self, outcomes: Sequence[NodeRebuildOutcome]) -> None:
        drifted = [outcome for outcome in outcomes if not outcome.matched]
        super().__init__("; ".join(outcome.describe() for outcome in drifted))
        self.outcomes = tuple(outcomes)
        self.drifted = tuple(drifted)


class FulfillmentRepairError(RuntimeError):
    """Raised when a repair refused: nothing was promoted into the authoritative root."""

    def __init__(self, message: str, *, outcome: AdmissionOutcome | None = None) -> None:
        super().__init__(message)
        self.outcome = outcome


def output_projection(
    manifest: AnyManifest,
    *,
    node_kind: FulfillmentNodeKind,
) -> OutputProjection:
    """Project one manifest onto the canonical structure a rebuild compares.

    The projection is exactly the node's identity, status, artifact byte
    identities, typed data products, and recorded summaries, each in the order
    the executor declared it. Everything that is not byte-reproducible across
    two clean executions — manifest bytes, ``created_at``, provenance hashes of
    manifest bytes, and machine-local URIs — is excluded by construction rather
    than normalized away afterwards.

    Args:
        manifest: The manifest to project.
        node_kind: The fulfillment node kind that produced it.
    """
    return OutputProjection(
        node_kind=node_kind,
        manifest_kind=manifest.kind,
        manifest_id=manifest.id,
        status=manifest.status,
        artifacts=[_artifact_projection(artifact) for artifact in manifest.artifacts],
        products=[
            _product_projection(product) for product in getattr(manifest, "produced_data", [])
        ],
        summaries=_projected_summaries(manifest),
    )


def _artifact_projection(artifact: ArtifactRef) -> ArtifactProjection:
    return ArtifactProjection(
        role=artifact.role,
        logical_name=artifact.logical_name,
        media_type=artifact.media_type,
        sha256=artifact.sha256,
        size_bytes=artifact.size_bytes,
    )


def _product_projection(product: AnalysisDataProduct) -> ProductProjection:
    return ProductProjection(
        role=product.role,
        logical_name=product.logical_name,
        product_schema_id=product.product_schema_id,
        product_schema_version=product.product_schema_version,
        label=product.label,
        parameters=dict(product.parameters),
        checkpoint_policy=dict(product.checkpoint_policy),
        rollout_policy=dict(product.rollout_policy),
        descriptor_basis_hash=product.descriptor_basis_hash,
        artifacts=[_artifact_projection(artifact) for artifact in product.artifacts],
    )


def _projected_summaries(manifest: AnyManifest) -> dict[str, Any]:
    """Return the recorded summaries a rebuild compares, in declared order.

    Evaluation and analysis manifests carry ``summary_metrics``; report recipes
    record their summary under ``metadata["summary"]``. No other metadata is
    projected, because arbitrary manifest metadata carries execution detail that
    is not part of a node's output.
    """
    summaries: dict[str, Any] = {}
    metrics = getattr(manifest, "summary_metrics", None)
    if metrics is not None:
        summaries["summary_metrics"] = dict(metrics)
    recorded = manifest.metadata.get("summary")
    if recorded is not None:
        summaries["metadata_summary"] = recorded
    return summaries


def compare_output_projections(
    expected: OutputProjection,
    observed: OutputProjection,
) -> tuple[ProjectionDifference, ...]:
    """Return every projection field on which two projections disagree.

    Fields are named by dotted path with list indices, such as
    ``artifacts[0].sha256``. Keys are walked in sorted order so two comparisons
    of the same pair always produce the same report. The schema identity fields
    are not compared: a projection that declared a different version would have
    been rejected at load.
    """
    differences: list[ProjectionDifference] = []
    _collect_differences(
        "",
        expected.model_dump(mode="json", exclude={"schema_id", "schema_version"}),
        observed.model_dump(mode="json", exclude={"schema_id", "schema_version"}),
        differences,
    )
    return tuple(differences)


def _collect_differences(
    path: str,
    expected: Any,
    observed: Any,
    differences: list[ProjectionDifference],
) -> None:
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in sorted(set(expected) | set(observed)):
            child = f"{path}.{key}" if path else key
            if key not in expected or key not in observed:
                differences.append(
                    ProjectionDifference(
                        field_path=child,
                        expected=expected.get(key),
                        observed=observed.get(key),
                    )
                )
                continue
            _collect_differences(child, expected[key], observed[key], differences)
        return
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            differences.append(
                ProjectionDifference(
                    field_path=path or "<projection>",
                    expected=expected,
                    observed=observed,
                )
            )
            return
        for index, (expected_item, observed_item) in enumerate(zip(expected, observed)):
            _collect_differences(f"{path}[{index}]", expected_item, observed_item, differences)
        return
    if expected != observed:
        differences.append(
            ProjectionDifference(
                field_path=path or "<projection>",
                expected=expected,
                observed=observed,
            )
        )


@dataclass(frozen=True)
class ShadowCustody:
    """A seeded shadow receipt root that never holds authoritative bytes."""

    authoritative_root: Path
    root: Path
    environment: FulfillmentEnvironment


@contextmanager
def shadow_custody(
    environment: FulfillmentEnvironment,
    *,
    shadow_root: Path | str | None = None,
    retain: bool = False,
) -> Iterator[ShadowCustody]:
    """Open a shadow receipt root mirroring the authoritative root's readable bytes.

    Parents must resolve identically in both roots, so the mirror copies every
    top-level entry of the authoritative root except the manifest index (whose
    stored paths are absolute) and this module's own quarantine and repair
    directories. Mirroring never hard-links: a hard-linked canonical artifact
    name makes the secure content-addressed store refuse to publish over an
    aliased name, which would break later writes into the authoritative root.

    Args:
        environment: The authoritative environment being shadowed.
        shadow_root: An explicit shadow location. A caller rebuilding many nodes
            passes one root and pays the mirror once. It must not live inside
            the authoritative root.
        retain: Keep the shadow root after the block instead of discarding it.
    """
    authoritative = Path(environment.root)
    if shadow_root is None:
        chosen = authoritative.parent / f".fulfillment-shadow-{uuid.uuid4().hex}"
    else:
        chosen = Path(shadow_root)
    if _is_inside(chosen, authoritative):
        raise ValueError(
            f"shadow root {chosen} must not live inside the authoritative root {authoritative}"
        )
    seeded = chosen.exists()
    if not seeded:
        _mirror_root(authoritative, chosen)
    try:
        yield ShadowCustody(
            authoritative_root=authoritative,
            root=chosen,
            environment=FulfillmentEnvironment(
                root=chosen,
                registries=environment.registries,
                repo_root=environment.repo_root,
                execution_context=environment.execution_context,
                issues=environment.issues,
            ),
        )
    finally:
        if not retain and not seeded:
            shutil.rmtree(chosen, ignore_errors=True)


def _is_inside(candidate: Path, parent: Path) -> bool:
    """Return whether *candidate* physically resolves at or inside *parent*.

    Containment is decided after resolving symlinks on both sides. A lexical
    comparison accepts a shadow root that merely *looks* like a sibling while
    being a symlink into authoritative custody, and a shadow that resolves into
    the authoritative root writes candidate bytes over authoritative ones. Both
    sides are resolved because a receipt root is itself commonly reached through
    a symlink, and resolving only one side would refuse legitimate siblings.
    ``os.path.realpath`` is non-strict, so a shadow root that does not exist yet
    resolves through its existing prefix.
    """
    absolute_candidate = Path(os.path.realpath(candidate))
    absolute_parent = Path(os.path.realpath(parent))
    return absolute_candidate == absolute_parent or absolute_parent in absolute_candidate.parents


def _mirror_root(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        return
    for entry in sorted(source.iterdir()):
        if entry.name in SHADOW_MIRROR_EXCLUDED:
            continue
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True, symlinks=True)
        elif entry.is_file():
            shutil.copy2(entry, target)


def rebuild_node(
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
    shadow: ShadowCustody | None = None,
) -> NodeRebuildOutcome:
    """Rebuild one node in shadow custody and compare its output projection.

    The authoritative receipt is admitted first and is never written to. An
    admission failure — including altered content-addressed bytes — refuses
    before any shadow execution, because corrupted storage is custody loss, not
    drift.

    Args:
        request: The node to verify.
        environment: The authoritative environment holding its receipt.
        shadow: A shadow root to execute into. When omitted, an ephemeral shadow
            is opened and discarded; pass one from :func:`shadow_custody` with
            ``retain=True`` to keep the rebuilt output. A supplied shadow belongs
            to exactly this node: executing a node overwrites its own receipt
            inside the shadow, which would invalidate the authenticated parent
            bindings a dependent node resolves there.
    """
    if isinstance(request, EvaluationMatrixNodeRequest):
        raise TypeError(
            f"evaluation matrix node {request.node_key!r} is not a single rebuild; expand it "
            "with expand_evaluation_matrix_node or use rebuild_nodes"
        )
    if shadow is not None:
        return _rebuild_in_shadow(request, environment=environment, shadow=shadow)
    with shadow_custody(environment) as opened:
        return _rebuild_in_shadow(request, environment=environment, shadow=opened)


def _rebuild_in_shadow(
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
    shadow: ShadowCustody,
) -> NodeRebuildOutcome:
    admitted_bytes = AdmittedManifestBytes()
    admission = admit_node(request, environment=environment, capture=admitted_bytes)
    if not admission.admitted:
        raise FulfillmentAdmissionError(admission)
    if admission.manifest_path is None:  # pragma: no cover - admitted implies a path
        raise RuntimeError("an admitted receipt must name the bytes that authenticated it")
    # The projection compared against the rebuild is the projection of the bytes
    # admission authenticated. Reopening the path here would project whatever is
    # stored by the time the rebuild finishes, and a verification that reads its
    # own subject twice verifies nothing about the first read.
    expected = output_projection(
        admitted_bytes.receipt(
            node_kind=request.node_kind, root=Path(environment.root)
        ).manifest,
        node_kind=request.node_kind,
    )
    rebuilt, _path = execute_node(request, environment=shadow.environment)
    observed = output_projection(rebuilt, node_kind=request.node_kind)
    differences = compare_output_projections(expected, observed)
    return NodeRebuildOutcome(
        node_key=request.node_key,
        node_kind=request.node_kind,
        manifest_id=admission.manifest_id,
        matched=not differences,
        expected=expected,
        observed=observed,
        differences=list(differences),
    )


@dataclass(frozen=True)
class RebuildRun:
    """The deterministic result listing for one strictly serial rebuild."""

    outcomes: tuple[NodeRebuildOutcome, ...]

    @property
    def verification_order(self) -> tuple[str, ...]:
        return tuple(outcome.node_key for outcome in self.outcomes)

    @property
    def drifted(self) -> tuple[NodeRebuildOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if not outcome.matched)


def rebuild_nodes(
    requests: Sequence[NodeRequest],
    *,
    environment: FulfillmentEnvironment,
) -> RebuildRun:
    """Rebuild every node serially in canonical order, refusing on any drift.

    Each node is verified in its own shadow root, freshly mirrored from the
    authoritative root. Sharing one shadow across nodes would let an earlier
    node's rebuilt receipt stand in as a later node's parent, so a dependent
    node would be verified against rebuilt bytes instead of the authoritative
    ones it really binds. Every node is verified before the refusal is raised,
    so the drift report is complete rather than truncated at the first
    difference. Evaluation matrix nodes are expanded to their canonical rows.

    Raises:
        FulfillmentDriftError: If any node's rebuilt projection differs.
    """
    outcomes = [
        rebuild_node(node, environment=environment)
        for node in _expanded_order(requests, environment=environment)
    ]
    run = RebuildRun(outcomes=tuple(outcomes))
    if run.drifted:
        raise FulfillmentDriftError(run.outcomes)
    return run


def _expanded_order(
    requests: Sequence[NodeRequest],
    *,
    environment: FulfillmentEnvironment,
) -> tuple[NodeRequest, ...]:
    expanded: list[NodeRequest] = []
    for request in canonical_fulfillment_order(requests):
        if isinstance(request, EvaluationMatrixNodeRequest):
            expanded.extend(expand_evaluation_matrix_node(request, environment=environment))
        else:
            expanded.append(request)
    return tuple(expanded)


def require_no_drift(outcome: NodeRebuildOutcome) -> NodeRebuildOutcome:
    """Return *outcome* when it matched, otherwise refuse with its drift report."""
    if not outcome.matched:
        raise FulfillmentDriftError([outcome])
    return outcome


@dataclass(frozen=True)
class RepairResult:
    """The outcome of one completed repair."""

    record: RepairRecord
    record_path: Path
    receipt: FulfillmentReceipt


def repair_node(
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
    shadow: ShadowCustody | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RepairResult:
    """Repair one node whose stored receipt exists but fails admission.

    The sequence is fixed and each step gates the next: quarantine the failed
    bytes by observed SHA-256, execute a candidate into shadow custody, admit
    the candidate completely in the shadow root, promote its artifacts, re-admit
    the promoted manifest against the authoritative root, and only then publish
    the manifest by write-then-rename. A failure at any gate leaves the
    authoritative manifest exactly as it was.

    Args:
        request: The node to repair.
        environment: The authoritative environment holding the failed receipt.
        shadow: A shadow root to execute the candidate into.
        metadata: Extra provenance recorded on the repair record.

    Raises:
        FulfillmentRepairError: If there is nothing to repair, the receipt is
            absent, or the candidate fails admission. Nothing is promoted.
    """
    if isinstance(request, EvaluationMatrixNodeRequest):
        raise TypeError(
            f"evaluation matrix node {request.node_key!r} is not a single receipt; expand it "
            "with expand_evaluation_matrix_node and repair each row"
        )
    if shadow is not None:
        return _repair_in_shadow(request, environment=environment, shadow=shadow, metadata=metadata)
    with shadow_custody(environment) as opened:
        return _repair_in_shadow(request, environment=environment, shadow=opened, metadata=metadata)


def _repair_in_shadow(
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
    shadow: ShadowCustody,
    metadata: Mapping[str, Any] | None,
) -> RepairResult:
    started_at = utc_now()
    root = Path(environment.root)
    failed_bytes = AdmittedManifestBytes()
    failure = admit_node(request, environment=environment, capture=failed_bytes)
    if failure.admitted:
        raise FulfillmentRepairError(
            f"node {request.node_key!r} admits cleanly; repair never runs over a healthy receipt",
            outcome=failure,
        )
    if not failure.manifest_present:
        raise FulfillmentRepairError(
            f"node {request.node_key!r} has no stored receipt at {failure.manifest_path}; "
            "an absent receipt is a fulfillment gap, not a repair",
            outcome=failure,
        )
    if failure.manifest_path is None:  # pragma: no cover - present implies a path
        raise RuntimeError("a present receipt must name the bytes that failed admission")

    manifest_path = Path(failure.manifest_path)
    if failed_bytes.raw is None:  # pragma: no cover - present implies bytes were read
        raise RuntimeError("a present receipt must carry the bytes admission read")
    stored_manifest = load_manifest_bytes(failed_bytes.raw)
    quarantined = _quarantine_failed_bytes(
        stored_manifest,
        manifest_path=manifest_path,
        raw=failed_bytes.raw,
        root=root,
    )
    old_manifest_sha256 = quarantined[0].observed_sha256
    _purge_mirrored_corruption(stored_manifest, shadow_root=shadow.root)

    candidate, candidate_path = execute_node(request, environment=shadow.environment)
    candidate_outcome = admit_node(request, environment=shadow.environment, path=candidate_path)
    if not candidate_outcome.admitted:
        raise FulfillmentRepairError(
            f"repair candidate for node {request.node_key!r} failed admission in shadow custody: "
            f"{candidate_outcome.describe()}",
            outcome=candidate_outcome,
        )

    promoted = normalize_manifest_spec_payloads(
        candidate.model_copy(
            update={
                "artifacts": [
                    _promote_artifact_bytes(artifact, shadow_root=shadow.root, root=root)
                    for artifact in candidate.artifacts
                ]
            }
        )
    )
    promoted_outcome = admit_node(request, environment=environment, manifest=promoted)
    if not promoted_outcome.admitted:
        raise FulfillmentRepairError(
            f"promoted candidate for node {request.node_key!r} failed admission against the "
            f"authoritative root: {promoted_outcome.describe()}",
            outcome=promoted_outcome,
        )
    replacement_sha256 = _publish_manifest_bytes(promoted, path=manifest_path, root=root)

    repaired_bytes = AdmittedManifestBytes()
    admission_after = admit_node(request, environment=environment, capture=repaired_bytes)
    if not admission_after.admitted:  # pragma: no cover - promoted admission already passed
        raise FulfillmentRepairError(
            f"repaired node {request.node_key!r} does not admit from its published bytes: "
            f"{admission_after.describe()}",
            outcome=admission_after,
        )

    record = RepairRecord(
        node_key=request.node_key,
        node_kind=request.node_kind,
        manifest_kind=failure.manifest_kind,
        manifest_id=failure.manifest_id,
        manifest_relative_path=manifest_path.relative_to(root).as_posix(),
        triggering_admission=failure,
        quarantined=list(quarantined),
        replacement_manifest_sha256=replacement_sha256,
        replacement_artifacts=[_artifact_projection(a) for a in promoted.artifacts],
        admission_after_repair=admission_after,
        started_at=started_at,
        completed_at=utc_now(),
        metadata=dict(metadata or {}),
    )
    record_path = _publish_repair_record(
        record,
        root=root,
        failed_manifest_sha256=old_manifest_sha256,
    )
    return RepairResult(
        record=record,
        record_path=record_path,
        receipt=repaired_bytes.receipt(node_kind=request.node_kind, root=root),
    )


def _quarantine_failed_bytes(
    manifest: AnyManifest,
    *,
    manifest_path: Path,
    raw: bytes,
    root: Path,
) -> tuple[QuarantinedBytes, ...]:
    """Preserve every byte stream the repair may replace, keyed by observed hash.

    The failed manifest bytes are always quarantined, and they are the bytes
    admission read rather than a fresh reopening of the path: quarantine exists
    to preserve exactly what failed, and re-reading would preserve — and key the
    whole repair record on — whatever replaced it.

    An artifact is quarantined when its stored bytes do not hash to their
    declared digest, because those are the only artifact bytes a promotion may
    unlink. Bytes are copied, never moved: quarantine adds custody, it never
    removes it.
    """
    entries = [
        _quarantine_bytes(
            raw,
            root=root,
            origin="manifest",
            source_path=manifest_path,
            suffix=".json",
        )
    ]
    for artifact in manifest.artifacts:
        path = artifact_bytes_path(artifact, root=root)
        if path is None or not path.is_file():
            continue
        data = path.read_bytes()
        observed = hashlib.sha256(data).hexdigest()
        if observed == artifact.sha256:
            continue
        entries.append(
            _quarantine_bytes(
                data,
                root=root,
                origin="artifact",
                source_path=path,
                suffix=_artifact_suffix(path, digest=artifact.sha256),
                role=artifact.role,
                logical_name=artifact.logical_name,
                declared_sha256=artifact.sha256,
            )
        )
    return tuple(entries)


def _purge_mirrored_corruption(manifest: AnyManifest, *, shadow_root: Path) -> None:
    """Drop the failed receipt's corrupt artifact copies from the shadow root.

    The shadow mirrors the authoritative root, so it also mirrors the corruption
    being repaired. A content-addressed name holding bytes that disagree with it
    makes the secure store refuse to publish, which would fail the candidate
    execution for the wrong reason. Only the repaired node's own mismatched
    copies are dropped, and only inside the shadow, whose bytes are scratch: the
    authoritative bytes are already quarantined.
    """
    for artifact in manifest.artifacts:
        path = artifact_bytes_path(artifact, root=shadow_root)
        if path is None or not path.is_file():
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact.sha256:
            path.unlink()


def _quarantine_bytes(
    data: bytes,
    *,
    root: Path,
    origin: Literal["manifest", "artifact"],
    source_path: Path,
    suffix: str,
    role: str | None = None,
    logical_name: str | None = None,
    declared_sha256: str | None = None,
) -> QuarantinedBytes:
    observed = hashlib.sha256(data).hexdigest()
    relative = f"{FULFILLMENT_QUARANTINE_DIRECTORY}/sha256/{observed[:2]}/{observed}{suffix}"
    destination = root / Path(relative)
    if destination.is_file():
        if destination.read_bytes() != data:
            raise FulfillmentRepairError(
                f"quarantine entry {destination} holds different bytes than its own SHA-256 names"
            )
    else:
        _write_bytes_atomically(data, path=destination)
    return QuarantinedBytes(
        origin=origin,
        role=role,
        logical_name=logical_name,
        declared_sha256=declared_sha256,
        observed_sha256=observed,
        size_bytes=len(data),
        source_relative_path=source_path.relative_to(root).as_posix(),
        quarantine_relative_path=relative,
    )


def _artifact_suffix(path: Path, *, digest: str | None) -> str:
    """Return the content-addressed file's suffix after its digest."""
    if digest and path.name.startswith(digest):
        return path.name[len(digest) :]
    return path.suffix


def _promote_artifact_bytes(
    artifact: ArtifactRef,
    *,
    shadow_root: Path,
    root: Path,
) -> ArtifactRef:
    """Publish one shadow-validated artifact into the authoritative store.

    Content-addressed bytes are self-authenticating and additive, so artifacts
    are promoted before the manifest: a crash here leaves the authoritative
    manifest untouched. When the authoritative name already holds bytes that
    disagree with their own digest — the corruption being repaired, already
    quarantined — that name is unlinked immediately before the verified
    replacement is stored, so the window in which neither is present is a single
    filesystem operation wide.

    The bytes this read returns are authenticated against the candidate's own
    declared digest and size *at this read*, before anything authoritative is
    touched. Shadow admission proved the shadow bytes earlier, and that proof is
    stale by the time promotion reads them again: without re-authentication the
    promotion would publish whatever the shadow path holds now under a manifest
    that declares what it held then.
    """
    resolution = resolve_artifact_bytes(artifact, root=shadow_root)
    source = resolution.path
    if source is None or not source.is_file():
        raise FulfillmentRepairError(
            f"shadow artifact {artifact.logical_name!r} has no resolvable bytes beneath "
            f"{shadow_root} ({resolution.describe()}); a validated candidate must carry its "
            "own bytes"
        )
    if not artifact.sha256:
        raise FulfillmentRepairError(
            f"shadow artifact {artifact.logical_name!r} declares no SHA-256; a promotion never "
            "publishes bytes it cannot authenticate"
        )
    data = source.read_bytes()
    observed_digest = hashlib.sha256(data).hexdigest()
    if observed_digest != artifact.sha256:
        raise FulfillmentRepairError(
            f"shadow artifact {artifact.logical_name!r} at {source} hashes to "
            f"{observed_digest}, not the declared {artifact.sha256}; the bytes promotion read "
            "are not the bytes shadow admission authenticated"
        )
    if artifact.size_bytes is not None and len(data) != artifact.size_bytes:
        raise FulfillmentRepairError(
            f"shadow artifact {artifact.logical_name!r} at {source} is {len(data)} bytes, not "
            f"the declared {artifact.size_bytes}; the bytes promotion read are not the bytes "
            "shadow admission authenticated"
        )
    destination = artifact_bytes_path(artifact, root=root)
    if destination is not None and destination.is_file():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != artifact.sha256:
            destination.unlink()
    stored = store_bytes_artifact(
        data,
        root=root,
        role=artifact.role,
        logical_name=artifact.logical_name,
        media_type=artifact.media_type,
        metadata=dict(artifact.metadata),
    )
    return artifact.model_copy(update={"uri": stored.uri})


def _publish_manifest_bytes(manifest: AnyManifest, *, path: Path, root: Path) -> str:
    """Publish manifest bytes by write-then-rename and return their SHA-256.

    ``os.replace`` inside the manifest's own directory is as close to atomic as
    the filesystem allows: the authoritative path holds either the previous
    bytes or the fully written replacement, never a partial candidate.
    """
    data = manifest.model_dump_json(indent=2, exclude_none=True).encode("utf-8") + b"\n"
    _write_bytes_atomically(data, path=path)
    index_manifest_file(path, root=root)
    return sha256_bytes(data)


def _publish_repair_record(
    record: RepairRecord,
    *,
    root: Path,
    failed_manifest_sha256: str,
) -> Path:
    """Write one repair record at its content-minted location.

    The location is keyed by the observed SHA-256 of the failed manifest bytes,
    so the record addresses the exact custody event it describes and a repeated
    repair of the same failed bytes does not accumulate near-duplicates.
    """
    directory = (
        root
        / FULFILLMENT_REPAIR_DIRECTORY
        / safe_manifest_key(record.manifest_id)
    )
    path = directory / f"{failed_manifest_sha256}.json"
    _write_bytes_atomically(
        record.model_dump_json(indent=2).encode("utf-8") + b"\n",
        path=path,
    )
    return path


def _write_bytes_atomically(data: bytes, *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=".fulfillment-staging-",
        delete=False,
    )
    staging = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
