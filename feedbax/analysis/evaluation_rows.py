"""Typed projection of authenticated evaluation rows for downstream analyses."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from feedbax.contracts.evaluation_states import EVALUATION_STATES_ARTIFACT_ROLE
from feedbax.contracts.manifest import (
    AnalysisEvaluationStateSource,
    EvaluationManifestProvenanceEnvelope,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    verify_evaluation_manifest_provenance,
)

if TYPE_CHECKING:
    from feedbax.analysis.specs import ResolvedAnalysisInput


StateT = TypeVar("StateT")
ParametersT = TypeVar("ParametersT")
MetadataT = TypeVar("MetadataT")
RowKeyT = TypeVar("RowKeyT", bound=Hashable)


class EvaluationRowProjectionErrorCategory(StrEnum):
    """Stable categories for row-projection failures."""

    INPUT_CONTRACT = "input_contract"
    MANIFEST_AUTHORITY = "manifest_authority"
    STATE_AUTHORITY = "state_authority"
    PROVENANCE = "provenance"
    PROJECTION = "projection"
    DUPLICATE_ROW_KEY = "duplicate_row_key"
    COVERAGE = "coverage"


class EvaluationRowProjectionError(ValueError):
    """A categorized failure that never requires parsing its message."""

    def __init__(
        self,
        category: EvaluationRowProjectionErrorCategory,
        message: str,
        *,
        row_index: int | None = None,
        manifest_id: str | None = None,
        projection_field: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.row_index = row_index
        self.manifest_id = manifest_id
        self.projection_field = projection_field
        self.__cause__ = cause


class EvaluationRowCoverageError(EvaluationRowProjectionError):
    """Exact authored-Cartesian coverage mismatch."""

    def __init__(
        self,
        message: str,
        *,
        missing: Sequence[Hashable] = (),
        unexpected: Sequence[Hashable] = (),
        duplicates: Sequence[Hashable] = (),
    ) -> None:
        super().__init__(EvaluationRowProjectionErrorCategory.COVERAGE, message)
        self.missing = tuple(missing)
        self.unexpected = tuple(unexpected)
        self.duplicates = tuple(duplicates)


@dataclass(frozen=True, slots=True)
class AuthenticatedEvaluationRow:
    """One row after Feedbax has verified manifest, state, and source authority."""

    manifest_authority: ParentRef
    manifest: EvaluationRunManifest
    run_spec: EvaluationRunSpec
    states: Any
    state_source: AnalysisEvaluationStateSource
    provenance: EvaluationManifestProvenanceEnvelope
    parameters: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationRowProjector(Generic[StateT, ParametersT, MetadataT, RowKeyT]):
    """Downstream-owned typed projections over one authenticated row."""

    state: Callable[[AuthenticatedEvaluationRow], StateT]
    parameters: Callable[[AuthenticatedEvaluationRow], ParametersT]
    metadata: Callable[[AuthenticatedEvaluationRow], MetadataT]
    row_key: Callable[
        [AuthenticatedEvaluationRow, StateT, ParametersT, MetadataT],
        RowKeyT,
    ]


@dataclass(frozen=True, slots=True)
class ProjectedEvaluationRow(Generic[StateT, ParametersT, MetadataT, RowKeyT]):
    """Typed downstream values plus the already-verified Feedbax authority."""

    row_key: RowKeyT
    state: StateT
    parameters: ParametersT
    metadata: MetadataT
    authority: AuthenticatedEvaluationRow


def project_authenticated_evaluation_rows(
    inputs: Sequence[ResolvedAnalysisInput],
    *,
    projector: EvaluationRowProjector[StateT, ParametersT, MetadataT, RowKeyT],
) -> tuple[ProjectedEvaluationRow[StateT, ParametersT, MetadataT, RowKeyT], ...]:
    """Authenticate and project resolved analysis inputs through downstream types.

    ``inputs`` are the public ``ResolvedAnalysisInput`` values supplied to an
    analysis recipe. Feedbax verifies their exact manifest bytes, completed
    producer/source provenance, and durable evaluation-state authority before
    invoking any downstream projector. Scientific interpretation remains in the
    four projector callbacks.
    """

    projected: list[ProjectedEvaluationRow[StateT, ParametersT, MetadataT, RowKeyT]] = []
    seen: dict[RowKeyT, int] = {}
    for index, item in enumerate(inputs):
        row = _authenticate_row(item, row_index=index)
        state = _project_field(projector.state, row, "state", row_index=index)
        parameters = _project_field(
            projector.parameters, row, "parameters", row_index=index
        )
        metadata = _project_field(projector.metadata, row, "metadata", row_index=index)
        try:
            row_key = projector.row_key(row, state, parameters, metadata)
            hash(row_key)
        except Exception as exc:
            raise EvaluationRowProjectionError(
                EvaluationRowProjectionErrorCategory.PROJECTION,
                f"evaluation row {row.manifest.id!r} row_key projection failed",
                row_index=index,
                manifest_id=row.manifest.id,
                projection_field="row_key",
                cause=exc,
            ) from exc
        if row_key in seen:
            raise EvaluationRowProjectionError(
                EvaluationRowProjectionErrorCategory.DUPLICATE_ROW_KEY,
                f"evaluation row key {row_key!r} is duplicated",
                row_index=index,
                manifest_id=row.manifest.id,
                projection_field="row_key",
            )
        seen[row_key] = index
        projected.append(
            ProjectedEvaluationRow(
                row_key=row_key,
                state=state,
                parameters=parameters,
                metadata=metadata,
                authority=row,
            )
        )
    return tuple(projected)


def require_exact_authored_cartesian_coverage(
    observed_keys: Sequence[RowKeyT],
    *,
    axes: Mapping[str, Sequence[Any]],
    row_key: Callable[[Mapping[str, Any]], RowKeyT],
) -> tuple[RowKeyT, ...]:
    """Require observed keys to equal an authored Cartesian product exactly.

    Axis meaning, membership, ordering, and the row-key representation are all
    caller-owned. Feedbax supplies only generic Cartesian expansion, duplicate
    detection, and exact missing/unexpected diagnostics.
    """

    if not axes:
        raise EvaluationRowCoverageError("authored Cartesian coverage requires axes")
    normalized: list[tuple[str, tuple[Any, ...]]] = []
    for name, values in axes.items():
        axis_values = tuple(values)
        if not isinstance(name, str) or not name or not axis_values:
            raise EvaluationRowCoverageError(
                "authored Cartesian axes require non-empty names and values"
            )
        if _duplicates(axis_values):
            raise EvaluationRowCoverageError(
                f"authored Cartesian axis {name!r} contains duplicate values"
            )
        normalized.append((name, axis_values))

    expected: list[RowKeyT] = []

    def expand(position: int, coordinate: dict[str, Any]) -> None:
        if position == len(normalized):
            try:
                key = row_key(MappingProxyType(dict(coordinate)))
                hash(key)
            except Exception as exc:
                raise EvaluationRowCoverageError(
                    "authored Cartesian row_key projection failed"
                ) from exc
            expected.append(key)
            return
        name, values = normalized[position]
        for value in values:
            coordinate[name] = value
            expand(position + 1, coordinate)
        coordinate.pop(name, None)

    expand(0, {})
    duplicate_expected = _duplicates(expected)
    if duplicate_expected:
        raise EvaluationRowCoverageError(
            "authored Cartesian row_key projection is not one-to-one",
            duplicates=duplicate_expected,
        )
    duplicate_observed = _duplicates(observed_keys)
    if duplicate_observed:
        raise EvaluationRowCoverageError(
            "observed evaluation row keys contain duplicates",
            duplicates=duplicate_observed,
        )
    expected_set = set(expected)
    observed_set = set(observed_keys)
    missing = tuple(key for key in expected if key not in observed_set)
    unexpected = tuple(key for key in observed_keys if key not in expected_set)
    if missing or unexpected:
        raise EvaluationRowCoverageError(
            f"evaluation row coverage mismatch: missing={missing!r}, "
            f"unexpected={unexpected!r}",
            missing=missing,
            unexpected=unexpected,
        )
    return tuple(expected)


def _authenticate_row(item: Any, *, row_index: int) -> AuthenticatedEvaluationRow:
    manifest = getattr(item, "manifest", None)
    manifest_id = getattr(manifest, "id", None)
    ref = getattr(item, "ref", None)
    manifest_input = getattr(item, "manifest_input", None)
    if (
        not isinstance(ref, ParentRef)
        or not isinstance(manifest, EvaluationRunManifest)
        or manifest_input is None
        or getattr(item, "states", None) is None
    ):
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.INPUT_CONTRACT,
            "evaluation row projection requires a resolved evaluation manifest and states",
            row_index=row_index,
            manifest_id=manifest_id,
        )
    if (
        manifest_input.ref != ref
        or manifest_input.manifest != manifest
        or ref.kind != "EvaluationRunManifest"
        or ref.role != "evaluation_run"
    ):
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.MANIFEST_AUTHORITY,
            f"evaluation row {manifest.id!r} disagrees with its authenticated authority",
            row_index=row_index,
            manifest_id=manifest.id,
        )
    source = getattr(item, "evaluation_state_source", None)
    if not isinstance(source, AnalysisEvaluationStateSource):
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.STATE_AUTHORITY,
            f"evaluation row {manifest.id!r} lacks typed state-source authority",
            row_index=row_index,
            manifest_id=manifest.id,
        )
    portable_ref = ref.model_copy(update={"uri": None})
    artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
    ]
    if (
        source.source_kind != "durable"
        or source.requested_evaluation_manifest_id != manifest.id
        or source.supplying_evaluation_manifest_id != manifest.id
        or source.evaluation_manifest_authority != portable_ref
        or len(artifacts) != 1
        or source.artifact_id != artifacts[0].artifact_id
        or source.artifact_sha256 != artifacts[0].sha256
        or source.artifact_size_bytes != artifacts[0].size_bytes
        or source.artifact_storage_backend != artifacts[0].storage_backend
    ):
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.STATE_AUTHORITY,
            f"evaluation row {manifest.id!r} state authority is not canonical",
            row_index=row_index,
            manifest_id=manifest.id,
        )
    try:
        run_spec = EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline)
        provenance = verify_evaluation_manifest_provenance(
            ref,
            manifest_input.raw_bytes,
            expected_producer_identity=run_spec.evaluation_type,
        )
    except Exception as exc:
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.PROVENANCE,
            f"evaluation row {manifest.id!r} provenance authentication failed",
            row_index=row_index,
            manifest_id=manifest.id,
            cause=exc,
        ) from exc
    return AuthenticatedEvaluationRow(
        manifest_authority=portable_ref,
        manifest=manifest,
        run_spec=run_spec,
        states=item.states,
        state_source=source,
        provenance=provenance,
        parameters=MappingProxyType(dict(run_spec.params)),
        metadata=MappingProxyType(dict(manifest.metadata)),
    )


def _project_field(
    projector: Callable[[AuthenticatedEvaluationRow], StateT],
    row: AuthenticatedEvaluationRow,
    field: str,
    *,
    row_index: int,
) -> StateT:
    try:
        return projector(row)
    except Exception as exc:
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.PROJECTION,
            f"evaluation row {row.manifest.id!r} {field} projection failed",
            row_index=row_index,
            manifest_id=row.manifest.id,
            projection_field=field,
            cause=exc,
        ) from exc


def _duplicates(values: Sequence[Any]) -> tuple[Any, ...]:
    seen: set[Any] = set()
    duplicates: list[Any] = []
    try:
        for value in values:
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
    except TypeError as exc:
        raise EvaluationRowCoverageError(
            "Cartesian axis values and row keys must be hashable"
        ) from exc
    return tuple(duplicates)


__all__ = [
    "AuthenticatedEvaluationRow",
    "EvaluationRowCoverageError",
    "EvaluationRowProjectionError",
    "EvaluationRowProjectionErrorCategory",
    "EvaluationRowProjector",
    "ProjectedEvaluationRow",
    "project_authenticated_evaluation_rows",
    "require_exact_authored_cartesian_coverage",
]
