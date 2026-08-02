"""Uniform per-kind admission of cached manifests as fulfillment receipts.

A fulfillment receipt is an authenticated fact: the manifest at the canonical
location for one node really is the completed record of exactly the requested
work. Admission is uniform across manifest kinds and requires *all* of:

1. ``status == "completed"``;
2. the manifest kind matches the requested node kind;
3. the recomputed content-minted identity equals the stored identifier;
4. the embedded spec equals the requested spec;
5. the exact expected parents (and, for figures, the runtime bindings) match;
6. every required output role is present;
7. every artifact's SHA-256 and byte size verify against its stored bytes.

Failure is refusal, not a cache miss. A manifest that exists but fails
admission stops fulfillment with named, structured failures rather than being
silently overwritten by re-execution, because silent re-execution over a failed
record conceals custody loss. Failures are returned as data so a later repair
mode can decide what to quarantine.

Stability note: this module is not part of the owner-ratified downstream
inventory in ``docs/design/downstream_interface_stability.md``. The
``feedbax.fulfillment.admission_outcome`` family nonetheless carries explicit
schema identity and rejects unsupported versions, because a refusal record that
crosses a process boundary is a durable emitted spec.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.analysis.figures import FigureExecutionPlan
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    AnyManifest,
    ArtifactRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    FigureManifest,
    ParentRef,
    ReportManifest,
    ReportSpec,
    SpecPayload,
    StagedEvaluationPrerequisite,
    StrictModel,
    analysis_run_manifest_id,
    canonical_json_bytes,
    canonical_manifest_path,
    evaluation_run_manifest_id,
    load_manifest_bytes,
    report_manifest_id,
)


FULFILLMENT_ADMISSION_SCHEMA_ID = "feedbax.fulfillment.admission_outcome"
FULFILLMENT_ADMISSION_SCHEMA_VERSION = "feedbax.fulfillment.admission_outcome.v1"
#: Versions this loader accepts, mapped to the version they migrate to.
FULFILLMENT_ADMISSION_MIGRATION_TABLE: dict[str, str] = {}

#: Node kinds the fulfillment surface admits, in deterministic declared order.
FulfillmentNodeKind = Literal["evaluation", "analysis", "figure", "report"]

FULFILLMENT_NODE_KINDS: tuple[FulfillmentNodeKind, ...] = (
    "evaluation",
    "analysis",
    "figure",
    "report",
)

#: Manifest kind that carries each node kind's receipt.
MANIFEST_KIND_BY_NODE_KIND: dict[FulfillmentNodeKind, str] = {
    "evaluation": "EvaluationRunManifest",
    "analysis": "AnalysisRunManifest",
    "figure": "FigureManifest",
    "report": "ReportManifest",
}

#: The named admission criteria, in evaluation order.
AdmissionCriterion = Literal[
    "manifest_present",
    "status_completed",
    "kind_matches",
    "identity_recomputed",
    "spec_matches",
    "parents_match",
    "runtime_bindings_match",
    "output_roles_present",
    "artifacts_verified",
]

#: The named reasons an admission criterion can fail.
AdmissionFailureCode = Literal[
    "manifest_absent",
    "status_not_completed",
    "kind_mismatch",
    "identity_mismatch",
    "spec_kind_mismatch",
    "spec_invalid",
    "spec_mismatch",
    "parents_mismatch",
    "runtime_binding_absent",
    "runtime_binding_unexpected",
    "runtime_binding_mismatch",
    "runtime_inputs_mismatch",
    "output_role_absent",
    "artifact_digest_absent",
    "artifact_size_absent",
    "artifact_bytes_unresolvable",
    "artifact_bytes_absent",
    "artifact_sha256_mismatch",
    "artifact_size_mismatch",
]


class AdmissionFailure(StrictModel):
    """One named admission criterion that a candidate receipt failed."""

    schema_id: Literal["feedbax.fulfillment.admission_outcome"] = FULFILLMENT_ADMISSION_SCHEMA_ID
    schema_version: Literal[
        "feedbax.fulfillment.admission_outcome.v1"
    ] = FULFILLMENT_ADMISSION_SCHEMA_VERSION
    criterion: AdmissionCriterion
    code: AdmissionFailureCode
    message: str
    node_kind: FulfillmentNodeKind
    manifest_id: str
    manifest_kind: str
    manifest_path: str | None = None
    expected: Any = None
    observed: Any = None
    details: dict[str, Any] = Field(default_factory=dict)


class AdmissionOutcome(StrictModel):
    """Structured admission decision for one candidate fulfillment receipt."""

    schema_id: Literal["feedbax.fulfillment.admission_outcome"] = FULFILLMENT_ADMISSION_SCHEMA_ID
    schema_version: Literal[
        "feedbax.fulfillment.admission_outcome.v1"
    ] = FULFILLMENT_ADMISSION_SCHEMA_VERSION
    admitted: bool
    node_kind: FulfillmentNodeKind
    manifest_id: str
    manifest_kind: str
    manifest_path: str | None = None
    manifest_present: bool
    failures: list[AdmissionFailure] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_decision(self) -> "AdmissionOutcome":
        if self.admitted and self.failures:
            raise ValueError("an admitted outcome cannot carry admission failures")
        if not self.admitted and not self.failures:
            raise ValueError("a refused outcome must name at least one admission failure")
        return self

    @property
    def codes(self) -> tuple[AdmissionFailureCode, ...]:
        """Return the failed codes in deterministic evaluation order."""
        return tuple(failure.code for failure in self.failures)

    def describe(self) -> str:
        """Return a stable one-line description of the decision."""
        if self.admitted:
            return f"{self.node_kind} receipt {self.manifest_id} admitted"
        reasons = "; ".join(f"{failure.criterion}={failure.code}" for failure in self.failures)
        return f"{self.node_kind} receipt {self.manifest_id} refused: {reasons}"


def migrate_admission_outcome(payload: AdmissionOutcome | Mapping[str, Any]) -> AdmissionOutcome:
    """Load a current admission outcome or reject an unsupported version.

    Unknown and removed versions fail closed with the family, current version,
    and migration table named. There is no inference and no compatibility shim.
    """
    if isinstance(payload, AdmissionOutcome):
        return AdmissionOutcome.model_validate(payload.model_dump(mode="json"))
    data = dict(payload)
    if data.get("schema_id") != FULFILLMENT_ADMISSION_SCHEMA_ID:
        raise ValueError(
            "unsupported AdmissionOutcome schema_id: "
            f"{data.get('schema_id')!r}; expected {FULFILLMENT_ADMISSION_SCHEMA_ID!r}"
        )
    version = data.get("schema_version")
    if version != FULFILLMENT_ADMISSION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported AdmissionOutcome schema_version: "
            f"{version!r}; expected {FULFILLMENT_ADMISSION_SCHEMA_VERSION!r}; "
            f"migration table={FULFILLMENT_ADMISSION_MIGRATION_TABLE!r}"
        )
    return AdmissionOutcome.model_validate(data)


class FulfillmentAdmissionError(RuntimeError):
    """Raised when a manifest exists for a node but is not an admissible receipt."""

    def __init__(self, outcome: AdmissionOutcome) -> None:
        super().__init__(outcome.describe())
        self.outcome = outcome


@dataclass(frozen=True)
class FulfillmentReceipt:
    """An admitted manifest bound to the exact bytes that authenticated it.

    This is an in-process value, not a durable emitted format: it carries a
    machine-local path and the loaded manifest object. Durable forward binding
    happens through ``ParentRef``/``StagedExactParents``, which are minted from
    a receipt rather than stored as one.

    The byte profile is part of the receipt rather than something a later
    consumer measures. A receipt exists because bytes at :attr:`path` were
    admitted, and :attr:`manifest_sha256` with :attr:`size_bytes` names *those*
    bytes. Anything that re-opened the path to describe what was admitted would
    be describing a second file: the bytes can change between admission and
    binding, and the profile a consumer then records as proof would be the
    replacement's, minted by the very step that was supposed to authenticate it.
    So the receipt is always built through :meth:`of_admitted_bytes` or
    :meth:`admitted`, both of which read once, and every forward binding is
    derived from the profile they settled.

    Attributes:
        node_kind: The fulfillment node kind whose receipt this is.
        manifest: The manifest parsed from the admitted bytes.
        path: Where those bytes were read from.
        root: The receipt root ``path`` lives under.
        manifest_sha256: The SHA-256 of the admitted bytes.
        size_bytes: The length of the admitted bytes.
    """

    node_kind: FulfillmentNodeKind
    manifest: AnyManifest
    path: Path
    root: Path
    manifest_sha256: str
    size_bytes: int

    @property
    def manifest_id(self) -> str:
        return self.manifest.id

    @property
    def manifest_kind(self) -> str:
        return self.manifest.kind

    @classmethod
    def of_admitted_bytes(
        cls,
        raw: bytes,
        *,
        node_kind: FulfillmentNodeKind,
        path: Path | str,
        root: Path | str,
        manifest: AnyManifest | None = None,
    ) -> "FulfillmentReceipt":
        """Return the receipt one already-read admitted byte string makes.

        The manifest is parsed from *raw* rather than taken from the caller, so
        the receipt's identity and its profile describe the same bytes. When the
        caller also holds a manifest object — an executor's return value, say —
        it is used only to prove the bytes are the ones it means.
        """
        parsed = load_manifest_bytes(raw)
        if manifest is not None and (
            parsed.kind != manifest.kind or parsed.id != manifest.id
        ):
            raise ValueError(
                f"manifest bytes at {path} are {parsed.kind} {parsed.id!r} but the receipt "
                f"names {manifest.kind} {manifest.id!r}"
            )
        return cls(
            node_kind=node_kind,
            manifest=parsed,
            path=Path(path),
            root=Path(root),
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
        )

    @classmethod
    def admitted(
        cls,
        *,
        node_kind: FulfillmentNodeKind,
        path: Path | str,
        root: Path | str,
        manifest: AnyManifest | None = None,
    ) -> "FulfillmentReceipt":
        """Return the receipt the bytes stored at *path* make, read exactly once."""
        return cls.of_admitted_bytes(
            Path(path).read_bytes(),
            node_kind=node_kind,
            path=path,
            root=root,
            manifest=manifest,
        )


@dataclass(frozen=True)
class _Candidate:
    """A located candidate receipt plus its load outcome.

    ``raw`` is the exact byte string admission read, or ``None`` when the caller
    supplied an in-memory manifest and nothing was read at all.
    """

    manifest: AnyManifest | None
    path: Path
    present: bool
    raw: bytes | None = None


@dataclass
class AdmittedManifestBytes:
    """The exact manifest bytes one admission read, handed back to its caller.

    Admission reads the stored manifest in order to decide. A caller that then
    binds the admitted receipt forward needs *those* bytes: re-opening the path
    to describe what was admitted describes whatever is there afterwards, and
    the profile it mints is the replacement's — produced, with perfect
    confidence, by the very step whose job was to authenticate. Passing one of
    these into an admission is how a caller collects the read that already
    happened rather than paying for a second one.

    An admission that read nothing — because the caller supplied an in-memory
    manifest — leaves :attr:`raw` unset, and :meth:`receipt` refuses rather than
    reading the path to fill the gap.
    """

    raw: bytes | None = None
    path: Path | None = None

    def receipt(
        self, *, node_kind: FulfillmentNodeKind, root: Path | str
    ) -> "FulfillmentReceipt":
        """Return the receipt these admitted bytes make."""
        if self.raw is None or self.path is None:
            raise RuntimeError(
                "no manifest bytes were captured by this admission, so no receipt can be "
                "bound from it; an admission over an in-memory manifest authenticates no "
                "stored bytes, and re-reading the path here would authenticate whatever is "
                "there now instead"
            )
        return FulfillmentReceipt.of_admitted_bytes(
            self.raw, node_kind=node_kind, path=self.path, root=root
        )


def receipt_path(
    node_kind: FulfillmentNodeKind,
    manifest_id: str,
    *,
    root: Path | str,
) -> Path:
    """Return the canonical receipt path for one node kind and manifest id."""
    return canonical_manifest_path(
        MANIFEST_KIND_BY_NODE_KIND[node_kind],
        manifest_id,
        root=root,
    )


def _load_candidate(
    node_kind: FulfillmentNodeKind,
    manifest_id: str,
    *,
    root: Path,
    manifest: AnyManifest | None,
    path: Path | None,
    capture: AdmittedManifestBytes | None = None,
) -> _Candidate:
    """Locate and read one candidate receipt exactly once.

    The stored manifest is parsed from the bytes this read returned rather than
    loaded a second time from the path, and those bytes are handed to *capture*
    so the caller can bind the receipt forward from the same read admission
    decided on.
    """
    resolved = path if path is not None else receipt_path(node_kind, manifest_id, root=root)
    if manifest is not None:
        return _Candidate(manifest=manifest, path=resolved, present=True)
    try:
        raw = resolved.read_bytes()
    except OSError:
        return _Candidate(manifest=None, path=resolved, present=False)
    found = load_manifest_bytes(raw)
    if capture is not None:
        capture.raw = raw
        capture.path = resolved
    return _Candidate(manifest=found, path=resolved, present=True, raw=raw)


class _FailureCollector:
    """Accumulates named admission failures in deterministic criterion order."""

    def __init__(
        self,
        *,
        node_kind: FulfillmentNodeKind,
        manifest_id: str,
        manifest_kind: str,
        manifest_path: Path | None,
    ) -> None:
        self._node_kind = node_kind
        self._manifest_id = manifest_id
        self._manifest_kind = manifest_kind
        self._manifest_path = manifest_path
        self.failures: list[AdmissionFailure] = []

    def add(
        self,
        criterion: AdmissionCriterion,
        code: AdmissionFailureCode,
        message: str,
        *,
        expected: Any = None,
        observed: Any = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.failures.append(
            AdmissionFailure(
                criterion=criterion,
                code=code,
                message=message,
                node_kind=self._node_kind,
                manifest_id=self._manifest_id,
                manifest_kind=self._manifest_kind,
                manifest_path=str(self._manifest_path) if self._manifest_path else None,
                expected=expected,
                observed=observed,
                details=dict(details or {}),
            )
        )


def _outcome(
    collector: _FailureCollector,
    *,
    node_kind: FulfillmentNodeKind,
    manifest_id: str,
    manifest_kind: str,
    manifest_path: Path | None,
    manifest_present: bool,
) -> AdmissionOutcome:
    return AdmissionOutcome(
        admitted=not collector.failures,
        node_kind=node_kind,
        manifest_id=manifest_id,
        manifest_kind=manifest_kind,
        manifest_path=str(manifest_path) if manifest_path else None,
        manifest_present=manifest_present,
        failures=collector.failures,
    )


def _check_common(
    collector: _FailureCollector,
    manifest: AnyManifest,
    *,
    expected_kind: str,
    expected_id: str,
) -> None:
    if manifest.kind != expected_kind:
        collector.add(
            "kind_matches",
            "kind_mismatch",
            f"cached manifest kind is {manifest.kind!r}, not {expected_kind!r}",
            expected=expected_kind,
            observed=manifest.kind,
        )
    if manifest.status != "completed":
        collector.add(
            "status_completed",
            "status_not_completed",
            f"cached manifest status is {manifest.status!r}, not 'completed'",
            expected="completed",
            observed=manifest.status,
        )
    if manifest.id != expected_id:
        collector.add(
            "identity_recomputed",
            "identity_mismatch",
            "recomputed content identity does not equal the stored manifest id",
            expected=expected_id,
            observed=manifest.id,
        )


def _check_embedded_spec(
    collector: _FailureCollector,
    payload: SpecPayload,
    *,
    spec_kind: str,
    model: type[StrictModel],
    requested_spec: StrictModel,
) -> None:
    if payload.kind != spec_kind:
        collector.add(
            "spec_matches",
            "spec_kind_mismatch",
            f"embedded spec kind is {payload.kind!r}, not {spec_kind!r}",
            expected=spec_kind,
            observed=payload.kind,
        )
        return
    try:
        cached_spec = model.model_validate(payload.inline)
    except ValueError as exc:
        collector.add(
            "spec_matches",
            "spec_invalid",
            f"embedded spec is not a valid {spec_kind}: {exc}",
            expected=spec_kind,
        )
        return
    if canonical_json_bytes(cached_spec) != canonical_json_bytes(requested_spec):
        collector.add(
            "spec_matches",
            "spec_mismatch",
            f"embedded {spec_kind} does not equal the requested {spec_kind}",
            expected=requested_spec.model_dump(mode="json", exclude_none=True),
            observed=cached_spec.model_dump(mode="json", exclude_none=True),
        )


def _check_parents(
    collector: _FailureCollector,
    manifest: AnyManifest,
    *,
    expected_parents: Sequence[ParentRef],
) -> None:
    observed = list(manifest.provenance.parents)
    expected = list(expected_parents)
    if observed != expected:
        collector.add(
            "parents_match",
            "parents_mismatch",
            "cached provenance parents do not equal the exact expected parents",
            expected=[parent.model_dump(mode="json", exclude_none=True) for parent in expected],
            observed=[parent.model_dump(mode="json", exclude_none=True) for parent in observed],
        )


def _check_output_roles(
    collector: _FailureCollector,
    manifest: AnyManifest,
    *,
    required_output_roles: Sequence[str],
) -> None:
    present = {artifact.role for artifact in manifest.artifacts}
    for role in required_output_roles:
        if role not in present:
            collector.add(
                "output_roles_present",
                "output_role_absent",
                f"required output role {role!r} is absent from the cached manifest",
                expected=role,
                observed=sorted(present),
            )


def artifact_bytes_path(artifact: ArtifactRef, *, root: Path) -> Path | None:
    """Return the in-root location of one artifact's bytes, or ``None``.

    Resolution is root-relative and never follows a machine-local ``uri``: a
    relocated receipt root must still verify. The recorded ``relative_path`` is
    preferred; otherwise the content-addressed layout is derived from the
    declared digest.
    """
    relative = artifact.metadata.get("relative_path")
    if isinstance(relative, str) and relative:
        pure = PurePosixPath(relative.replace("\\", "/"))
        if not pure.is_absolute() and not any(part in {"", ".", ".."} for part in pure.parts):
            return root.joinpath(*pure.parts)
        return None
    digest = artifact.sha256
    if not digest:
        return None
    directory = root / "artifacts" / "sha256" / digest[:2]
    if not directory.is_dir():
        return None
    matches = sorted(
        candidate
        for candidate in directory.iterdir()
        if candidate.is_file() and candidate.name.startswith(digest)
    )
    return matches[0] if matches else None


def _check_artifacts(
    collector: _FailureCollector,
    manifest: AnyManifest,
    *,
    root: Path,
) -> None:
    for index, artifact in enumerate(manifest.artifacts):
        label = f"artifacts[{index}] role={artifact.role!r} name={artifact.logical_name!r}"
        if not artifact.sha256:
            collector.add(
                "artifacts_verified",
                "artifact_digest_absent",
                f"{label} declares no SHA-256, so its bytes cannot be authenticated",
                observed=artifact.model_dump(mode="json", exclude_none=True),
            )
            continue
        if artifact.size_bytes is None:
            collector.add(
                "artifacts_verified",
                "artifact_size_absent",
                f"{label} declares no byte size, so its bytes cannot be authenticated",
                observed=artifact.model_dump(mode="json", exclude_none=True),
            )
            continue
        path = artifact_bytes_path(artifact, root=root)
        if path is None:
            collector.add(
                "artifacts_verified",
                "artifact_bytes_unresolvable",
                f"{label} has no resolvable location beneath the receipt root",
                observed=artifact.metadata.get("relative_path"),
                details={"root": str(root)},
            )
            continue
        try:
            data = path.read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            collector.add(
                "artifacts_verified",
                "artifact_bytes_absent",
                f"{label} bytes are missing at {path}",
                expected=artifact.sha256,
                details={"path": str(path)},
            )
            continue
        observed_digest = hashlib.sha256(data).hexdigest()
        if observed_digest != artifact.sha256:
            collector.add(
                "artifacts_verified",
                "artifact_sha256_mismatch",
                f"{label} stored bytes do not match the recorded SHA-256",
                expected=artifact.sha256,
                observed=observed_digest,
                details={"path": str(path)},
            )
            continue
        if len(data) != artifact.size_bytes:
            collector.add(
                "artifacts_verified",
                "artifact_size_mismatch",
                f"{label} stored bytes do not match the recorded size",
                expected=artifact.size_bytes,
                observed=len(data),
                details={"path": str(path)},
            )


def evaluation_expected_parents(spec: EvaluationRunSpec) -> list[ParentRef]:
    """Return the exact provenance parents one evaluation spec must produce.

    This mirrors the executor exactly: declared inputs followed by the parents
    of each staged prerequisite, in declaration order.
    """
    staged = spec.params.get("staged_prerequisites") or {}
    if not isinstance(staged, Mapping):
        raise TypeError("evaluation params.staged_prerequisites must be a mapping or null")
    prerequisite_parents = [
        StagedEvaluationPrerequisite.model_validate(value).parent for value in staged.values()
    ]
    return [*spec.inputs, *prerequisite_parents]


def admit_evaluation_receipt(
    spec: EvaluationRunSpec,
    *,
    root: Path | str,
    manifest: EvaluationRunManifest | None = None,
    path: Path | str | None = None,
    expected_parents: Sequence[ParentRef] | None = None,
    required_output_roles: Sequence[str] = (),
    capture: AdmittedManifestBytes | None = None,
) -> AdmissionOutcome:
    """Admit a cached ``EvaluationRunManifest`` as this spec's fulfillment receipt."""
    root_path = Path(root)
    manifest_id = evaluation_run_manifest_id(spec)
    candidate = _load_candidate(
        "evaluation",
        manifest_id,
        root=root_path,
        manifest=manifest,
        path=Path(path) if path is not None else None,
        capture=capture,
    )
    collector = _FailureCollector(
        node_kind="evaluation",
        manifest_id=manifest_id,
        manifest_kind="EvaluationRunManifest",
        manifest_path=candidate.path,
    )
    if candidate.manifest is None:
        collector.add(
            "manifest_present",
            "manifest_absent",
            f"no EvaluationRunManifest is stored at {candidate.path}",
            details={"path": str(candidate.path)},
        )
        return _outcome(
            collector,
            node_kind="evaluation",
            manifest_id=manifest_id,
            manifest_kind="EvaluationRunManifest",
            manifest_path=candidate.path,
            manifest_present=False,
        )
    found = candidate.manifest
    _check_common(collector, found, expected_kind="EvaluationRunManifest", expected_id=manifest_id)
    if isinstance(found, EvaluationRunManifest):
        _check_embedded_spec(
            collector,
            found.evaluation_spec,
            spec_kind="EvaluationRunSpec",
            model=EvaluationRunSpec,
            requested_spec=spec,
        )
    _check_parents(
        collector,
        found,
        expected_parents=(
            expected_parents
            if expected_parents is not None
            else evaluation_expected_parents(spec)
        ),
    )
    _check_output_roles(collector, found, required_output_roles=required_output_roles)
    _check_artifacts(collector, found, root=root_path)
    return _outcome(
        collector,
        node_kind="evaluation",
        manifest_id=manifest_id,
        manifest_kind="EvaluationRunManifest",
        manifest_path=candidate.path,
        manifest_present=True,
    )


def admit_analysis_receipt(
    spec: AnalysisRunSpec,
    *,
    root: Path | str,
    manifest: AnalysisRunManifest | None = None,
    path: Path | str | None = None,
    expected_parents: Sequence[ParentRef] | None = None,
    required_output_roles: Sequence[str] = (),
    capture: AdmittedManifestBytes | None = None,
) -> AdmissionOutcome:
    """Admit a cached ``AnalysisRunManifest`` as this spec's fulfillment receipt."""
    root_path = Path(root)
    manifest_id = analysis_run_manifest_id(spec)
    candidate = _load_candidate(
        "analysis",
        manifest_id,
        root=root_path,
        manifest=manifest,
        path=Path(path) if path is not None else None,
        capture=capture,
    )
    collector = _FailureCollector(
        node_kind="analysis",
        manifest_id=manifest_id,
        manifest_kind="AnalysisRunManifest",
        manifest_path=candidate.path,
    )
    if candidate.manifest is None:
        collector.add(
            "manifest_present",
            "manifest_absent",
            f"no AnalysisRunManifest is stored at {candidate.path}",
            details={"path": str(candidate.path)},
        )
        return _outcome(
            collector,
            node_kind="analysis",
            manifest_id=manifest_id,
            manifest_kind="AnalysisRunManifest",
            manifest_path=candidate.path,
            manifest_present=False,
        )
    found = candidate.manifest
    _check_common(collector, found, expected_kind="AnalysisRunManifest", expected_id=manifest_id)
    if isinstance(found, AnalysisRunManifest):
        _check_embedded_spec(
            collector,
            found.analysis_spec,
            spec_kind="AnalysisRunSpec",
            model=AnalysisRunSpec,
            requested_spec=spec,
        )
    _check_parents(
        collector,
        found,
        expected_parents=expected_parents if expected_parents is not None else spec.inputs,
    )
    _check_output_roles(collector, found, required_output_roles=required_output_roles)
    _check_artifacts(collector, found, root=root_path)
    return _outcome(
        collector,
        node_kind="analysis",
        manifest_id=manifest_id,
        manifest_kind="AnalysisRunManifest",
        manifest_path=candidate.path,
        manifest_present=True,
    )


def admit_report_receipt(
    spec: ReportSpec,
    *,
    root: Path | str,
    manifest: ReportManifest | None = None,
    path: Path | str | None = None,
    expected_parents: Sequence[ParentRef] | None = None,
    required_output_roles: Sequence[str] = (),
    capture: AdmittedManifestBytes | None = None,
) -> AdmissionOutcome:
    """Admit a cached ``ReportManifest`` as this spec's fulfillment receipt."""
    root_path = Path(root)
    manifest_id = report_manifest_id(spec)
    candidate = _load_candidate(
        "report",
        manifest_id,
        root=root_path,
        manifest=manifest,
        path=Path(path) if path is not None else None,
        capture=capture,
    )
    collector = _FailureCollector(
        node_kind="report",
        manifest_id=manifest_id,
        manifest_kind="ReportManifest",
        manifest_path=candidate.path,
    )
    if candidate.manifest is None:
        collector.add(
            "manifest_present",
            "manifest_absent",
            f"no ReportManifest is stored at {candidate.path}",
            details={"path": str(candidate.path)},
        )
        return _outcome(
            collector,
            node_kind="report",
            manifest_id=manifest_id,
            manifest_kind="ReportManifest",
            manifest_path=candidate.path,
            manifest_present=False,
        )
    found = candidate.manifest
    _check_common(collector, found, expected_kind="ReportManifest", expected_id=manifest_id)
    if isinstance(found, ReportManifest):
        _check_embedded_spec(
            collector,
            found.report_spec,
            spec_kind="ReportSpec",
            model=ReportSpec,
            requested_spec=spec,
        )
    _check_parents(
        collector,
        found,
        expected_parents=expected_parents if expected_parents is not None else spec.inputs,
    )
    _check_output_roles(collector, found, required_output_roles=required_output_roles)
    _check_artifacts(collector, found, root=root_path)
    return _outcome(
        collector,
        node_kind="report",
        manifest_id=manifest_id,
        manifest_kind="ReportManifest",
        manifest_path=candidate.path,
        manifest_present=True,
    )


def _figure_runtime_binding_from_manifest(manifest: FigureManifest) -> SpecPayload | None:
    """Return the single recorded runtime-binding payload, if the manifest has one."""
    payloads = [
        entry
        for entry in manifest.regeneration_specs
        if isinstance(entry, SpecPayload) and entry.kind == "FigureRuntimeBindingSpec"
    ]
    if len(payloads) > 1:
        raise ValueError(
            f"FigureManifest {manifest.id!r} records {len(payloads)} runtime-binding payloads"
        )
    return payloads[0] if payloads else None


def admit_figure_receipt(
    plan: FigureExecutionPlan,
    *,
    root: Path | str,
    manifest: FigureManifest | None = None,
    path: Path | str | None = None,
    expected_parents: Sequence[ParentRef] | None = None,
    required_output_roles: Sequence[str] = (),
    capture: AdmittedManifestBytes | None = None,
) -> AdmissionOutcome:
    """Admit a cached ``FigureManifest`` as this figure node's fulfillment receipt.

    Figure identity is minted from the *resolved authored* spec, and runtime
    inputs, input authorities, and artifact-provider bindings are overlaid
    afterwards. A matching manifest id therefore proves only that the same
    authored figure was rendered at some point — not that it was rendered
    against the runtime inputs this node requests. Admission compares the
    overlaid runtime inputs and the recorded binding provenance as well, so
    id-only admission is never sufficient for a figure.

    Args:
        plan: The plan produced by ``plan_figure_execution`` for this node. It
            carries the same identity, embedded spec, and runtime-binding
            derivation the executor would use.
    """
    root_path = Path(root)
    manifest_id = plan.manifest_id
    candidate = _load_candidate(
        "figure",
        manifest_id,
        root=root_path,
        manifest=manifest,
        path=Path(path) if path is not None else None,
        capture=capture,
    )
    collector = _FailureCollector(
        node_kind="figure",
        manifest_id=manifest_id,
        manifest_kind="FigureManifest",
        manifest_path=candidate.path,
    )
    if candidate.manifest is None:
        collector.add(
            "manifest_present",
            "manifest_absent",
            f"no FigureManifest is stored at {candidate.path}",
            details={"path": str(candidate.path)},
        )
        return _outcome(
            collector,
            node_kind="figure",
            manifest_id=manifest_id,
            manifest_kind="FigureManifest",
            manifest_path=candidate.path,
            manifest_present=False,
        )
    found = candidate.manifest
    _check_common(collector, found, expected_kind="FigureManifest", expected_id=manifest_id)
    if isinstance(found, FigureManifest):
        _check_embedded_spec(
            collector,
            found.figure_spec,
            spec_kind="FigureSpec",
            model=FigureSpec,
            requested_spec=FigureSpec.model_validate(plan.authored_payload.inline),
        )
        _check_figure_runtime_bindings(collector, found, plan=plan)
    _check_parents(
        collector,
        found,
        expected_parents=(
            expected_parents
            if expected_parents is not None
            else plan.execution_spec.inputs
        ),
    )
    _check_output_roles(collector, found, required_output_roles=required_output_roles)
    _check_artifacts(collector, found, root=root_path)
    return _outcome(
        collector,
        node_kind="figure",
        manifest_id=manifest_id,
        manifest_kind="FigureManifest",
        manifest_path=candidate.path,
        manifest_present=True,
    )


def _check_figure_runtime_bindings(
    collector: _FailureCollector,
    manifest: FigureManifest,
    *,
    plan: FigureExecutionPlan,
) -> None:
    expected_inputs = list(plan.execution_spec.inputs)
    if list(manifest.inputs) != expected_inputs:
        collector.add(
            "runtime_bindings_match",
            "runtime_inputs_mismatch",
            "cached figure runtime inputs do not equal the requested runtime inputs",
            expected=[ref.model_dump(mode="json", exclude_none=True) for ref in expected_inputs],
            observed=[
                ref.model_dump(mode="json", exclude_none=True) for ref in manifest.inputs
            ],
        )
    observed_payload = _figure_runtime_binding_from_manifest(manifest)
    expected_payload = plan.runtime_binding_payload
    if expected_payload is None and observed_payload is not None:
        collector.add(
            "runtime_bindings_match",
            "runtime_binding_unexpected",
            "cached figure records a runtime binding but this node declares none",
            observed=observed_payload.inline,
        )
        return
    if expected_payload is not None and observed_payload is None:
        collector.add(
            "runtime_bindings_match",
            "runtime_binding_absent",
            "cached figure records no runtime binding but this node declares one",
            expected=expected_payload.inline,
        )
        return
    if expected_payload is None or observed_payload is None:
        return
    if canonical_json_bytes(observed_payload.inline) != canonical_json_bytes(
        expected_payload.inline
    ):
        collector.add(
            "runtime_bindings_match",
            "runtime_binding_mismatch",
            "cached figure runtime binding provenance does not equal the requested binding",
            expected=expected_payload.inline,
            observed=observed_payload.inline,
        )


def require_admitted(outcome: AdmissionOutcome, receipt: FulfillmentReceipt) -> FulfillmentReceipt:
    """Return *receipt* when *outcome* admitted it, otherwise refuse."""
    if not outcome.admitted:
        raise FulfillmentAdmissionError(outcome)
    return receipt
