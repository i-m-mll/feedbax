"""Verified-provenance projection of resolver-issued evaluation rows."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from feedbax.contracts.manifest import (
    AnalysisEvaluationStateSource,
    EvaluationManifestProvenanceEnvelope,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    verify_evaluation_manifest_provenance,
)

if TYPE_CHECKING:
    from feedbax.analysis.specs import (
        EvaluationStateMaterializationReceipt,
        ResolvedAnalysisInput,
    )


StateT = TypeVar("StateT")
ParametersT = TypeVar("ParametersT")
MetadataT = TypeVar("MetadataT")
RowKeyT = TypeVar("RowKeyT", bound=Hashable)


class EvaluationRowProjectionErrorCategory(StrEnum):
    """Coarse stable stage for a row-projection failure."""

    INPUT_CONTRACT = "input_contract"
    MANIFEST_AUTHORITY = "manifest_authority"
    STATE_MATERIALIZATION = "state_materialization"
    PROVENANCE = "provenance"
    PROJECTION = "projection"
    DUPLICATE_ROW_KEY = "duplicate_row_key"
    COVERAGE = "coverage"


class EvaluationRowProjectionErrorReason(StrEnum):
    """Stable reason codes; callers never need to inspect exception text."""

    INPUT_MANIFEST_MISSING = "input_manifest_missing"
    INPUT_STATES_MISSING = "input_states_missing"
    MANIFEST_AUTHORITY_MISMATCH = "manifest_authority_mismatch"
    STATE_SOURCE_MISSING = "state_source_missing"
    STATE_RECEIPT_MISSING = "state_receipt_missing"
    STATE_RECEIPT_MISMATCH = "state_receipt_mismatch"
    MANIFEST_PROVENANCE_INVALID = "manifest_provenance_invalid"
    PROJECTOR_FAILED = "projector_failed"
    PROJECTED_KEY_UNHASHABLE = "projected_key_unhashable"
    PROJECTED_KEY_DUPLICATE = "projected_key_duplicate"
    COVERAGE_AXES_EMPTY = "coverage_axes_empty"
    COVERAGE_AXIS_INVALID = "coverage_axis_invalid"
    COVERAGE_AXIS_VALUE_UNHASHABLE = "coverage_axis_value_unhashable"
    COVERAGE_AXIS_VALUE_DUPLICATE = "coverage_axis_value_duplicate"
    COVERAGE_KEY_PROJECTION_FAILED = "coverage_key_projection_failed"
    COVERAGE_EXPECTED_KEY_UNHASHABLE = "coverage_expected_key_unhashable"
    COVERAGE_EXPECTED_KEY_COLLISION = "coverage_expected_key_collision"
    COVERAGE_OBSERVED_KEY_UNHASHABLE = "coverage_observed_key_unhashable"
    COVERAGE_OBSERVED_KEY_DUPLICATE = "coverage_observed_key_duplicate"
    COVERAGE_KEY_SET_MISMATCH = "coverage_key_set_mismatch"


class EvaluationRowProjectionError(ValueError):
    """Categorized and reason-coded row-projection failure."""

    def __init__(
        self,
        category: EvaluationRowProjectionErrorCategory,
        reason: EvaluationRowProjectionErrorReason,
        message: str,
        *,
        row_index: int | None = None,
        manifest_id: str | None = None,
        row_key: Hashable | None = None,
        first_index: int | None = None,
        source_kind: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.reason = reason
        self.row_index = row_index
        self.manifest_id = manifest_id
        self.row_key = row_key
        self.first_index = first_index
        self.source_kind = source_kind
        self.__cause__ = cause


class EvaluationRowCoverageError(EvaluationRowProjectionError):
    """Structured exact authored-Cartesian coverage failure."""

    def __init__(
        self,
        reason: EvaluationRowProjectionErrorReason,
        message: str,
        *,
        axis_name: str | None = None,
        coordinate: Mapping[str, Any] | None = None,
        missing: Sequence[Hashable] = (),
        unexpected: Sequence[Hashable] = (),
        duplicates: Sequence[Hashable] = (),
        duplicate_indices: Sequence[tuple[int, int]] = (),
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            EvaluationRowProjectionErrorCategory.COVERAGE,
            reason,
            message,
            cause=cause,
        )
        self.axis_name = axis_name
        self.coordinate = MappingProxyType(dict(coordinate)) if coordinate is not None else None
        self.missing = tuple(missing)
        self.unexpected = tuple(unexpected)
        self.duplicates = tuple(duplicates)
        self.duplicate_indices = tuple(duplicate_indices)


@dataclass(frozen=True, slots=True)
class VerifiedEvaluationRowFacts:
    """Facts verified before downstream interpretation.

    Manifest bytes and producer provenance are authenticated. ``state_receipt``
    binds the complete typed source and exact in-memory state object issued by
    the resolver, and truthfully identifies whether it came from an
    authenticated artifact, a versioned manifest-keyed cache, or an
    authenticated recomputation. The latter two are not mislabeled as
    content-authenticated state bytes.
    """

    manifest_authority: ParentRef
    manifest: EvaluationRunManifest
    run_spec: EvaluationRunSpec
    states: Any
    state_source: AnalysisEvaluationStateSource
    state_receipt: EvaluationStateMaterializationReceipt
    provenance: EvaluationManifestProvenanceEnvelope
    parameters: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationRowProjection(Generic[StateT, ParametersT, MetadataT, RowKeyT]):
    """One downstream-owned, cross-field typed row projection."""

    row_key: RowKeyT
    state: StateT
    parameters: ParametersT
    metadata: MetadataT


@dataclass(frozen=True, slots=True)
class ProjectedEvaluationRow(Generic[StateT, ParametersT, MetadataT, RowKeyT]):
    """Typed downstream values plus their verified upstream facts."""

    row_key: RowKeyT
    state: StateT
    parameters: ParametersT
    metadata: MetadataT
    facts: VerifiedEvaluationRowFacts


def project_verified_evaluation_rows(
    inputs: Sequence[ResolvedAnalysisInput],
    *,
    project: Callable[
        [VerifiedEvaluationRowFacts],
        EvaluationRowProjection[StateT, ParametersT, MetadataT, RowKeyT],
    ],
) -> tuple[ProjectedEvaluationRow[StateT, ParametersT, MetadataT, RowKeyT], ...]:
    """Verify resolver facts, then invoke one downstream cross-field projector."""

    projected: list[ProjectedEvaluationRow[StateT, ParametersT, MetadataT, RowKeyT]] = []
    first_by_key: dict[RowKeyT, int] = {}
    for index, item in enumerate(inputs):
        facts = _verify_row_facts(item, row_index=index)
        try:
            value = project(facts)
        except Exception as exc:
            raise EvaluationRowProjectionError(
                EvaluationRowProjectionErrorCategory.PROJECTION,
                EvaluationRowProjectionErrorReason.PROJECTOR_FAILED,
                f"evaluation row {facts.manifest.id!r} projection failed",
                row_index=index,
                manifest_id=facts.manifest.id,
                source_kind=facts.state_source.source_kind,
                cause=exc,
            ) from exc
        if not isinstance(value, EvaluationRowProjection):
            exc = TypeError("project must return EvaluationRowProjection")
            raise EvaluationRowProjectionError(
                EvaluationRowProjectionErrorCategory.PROJECTION,
                EvaluationRowProjectionErrorReason.PROJECTOR_FAILED,
                f"evaluation row {facts.manifest.id!r} projector returned {type(value).__name__}",
                row_index=index,
                manifest_id=facts.manifest.id,
                source_kind=facts.state_source.source_kind,
                cause=exc,
            ) from exc
        try:
            hash(value.row_key)
        except Exception as exc:
            raise EvaluationRowProjectionError(
                EvaluationRowProjectionErrorCategory.PROJECTION,
                EvaluationRowProjectionErrorReason.PROJECTED_KEY_UNHASHABLE,
                f"evaluation row {facts.manifest.id!r} projected an unhashable key",
                row_index=index,
                manifest_id=facts.manifest.id,
                source_kind=facts.state_source.source_kind,
                cause=exc,
            ) from exc
        first_index = first_by_key.get(value.row_key)
        if first_index is not None:
            raise EvaluationRowProjectionError(
                EvaluationRowProjectionErrorCategory.DUPLICATE_ROW_KEY,
                EvaluationRowProjectionErrorReason.PROJECTED_KEY_DUPLICATE,
                f"evaluation row key {value.row_key!r} is duplicated",
                row_index=index,
                manifest_id=facts.manifest.id,
                row_key=value.row_key,
                first_index=first_index,
                source_kind=facts.state_source.source_kind,
            )
        first_by_key[value.row_key] = index
        projected.append(
            ProjectedEvaluationRow(
                row_key=value.row_key,
                state=value.state,
                parameters=value.parameters,
                metadata=value.metadata,
                facts=facts,
            )
        )
    return tuple(projected)


def require_exact_authored_cartesian_coverage(
    observed_keys: Sequence[RowKeyT],
    *,
    axes: Mapping[str, Sequence[Any]],
    row_key: Callable[[Mapping[str, Any]], RowKeyT],
) -> tuple[RowKeyT, ...]:
    """Require observed keys to equal a downstream-authored Cartesian product."""

    if not axes:
        raise EvaluationRowCoverageError(
            EvaluationRowProjectionErrorReason.COVERAGE_AXES_EMPTY,
            "authored Cartesian coverage requires axes",
        )
    normalized: list[tuple[str, tuple[Any, ...]]] = []
    for name, values in axes.items():
        axis_values = tuple(values)
        if not isinstance(name, str) or not name or not axis_values:
            raise EvaluationRowCoverageError(
                EvaluationRowProjectionErrorReason.COVERAGE_AXIS_INVALID,
                "authored Cartesian axes require non-empty names and values",
                axis_name=name if isinstance(name, str) else None,
            )
        duplicates, _ = _duplicate_details(
            axis_values,
            unhashable_reason=(EvaluationRowProjectionErrorReason.COVERAGE_AXIS_VALUE_UNHASHABLE),
            duplicate_reason=(EvaluationRowProjectionErrorReason.COVERAGE_AXIS_VALUE_DUPLICATE),
            axis_name=name,
        )
        if duplicates:
            raise EvaluationRowCoverageError(
                EvaluationRowProjectionErrorReason.COVERAGE_AXIS_VALUE_DUPLICATE,
                f"authored Cartesian axis {name!r} contains duplicate values",
                axis_name=name,
                duplicates=duplicates,
            )
        normalized.append((name, axis_values))

    expected: list[RowKeyT] = []

    def expand(position: int, coordinate: dict[str, Any]) -> None:
        if position == len(normalized):
            frozen = MappingProxyType(dict(coordinate))
            try:
                key = row_key(frozen)
            except Exception as exc:
                raise EvaluationRowCoverageError(
                    EvaluationRowProjectionErrorReason.COVERAGE_KEY_PROJECTION_FAILED,
                    "authored Cartesian row-key projection failed",
                    coordinate=frozen,
                    cause=exc,
                ) from exc
            try:
                hash(key)
            except Exception as exc:
                raise EvaluationRowCoverageError(
                    EvaluationRowProjectionErrorReason.COVERAGE_EXPECTED_KEY_UNHASHABLE,
                    "authored Cartesian row-key projection returned an unhashable key",
                    coordinate=frozen,
                    cause=exc,
                ) from exc
            expected.append(key)
            return
        name, values = normalized[position]
        for value in values:
            coordinate[name] = value
            expand(position + 1, coordinate)
        coordinate.pop(name, None)

    expand(0, {})
    duplicate_expected, expected_indices = _duplicate_details(
        expected,
        unhashable_reason=(EvaluationRowProjectionErrorReason.COVERAGE_EXPECTED_KEY_UNHASHABLE),
        duplicate_reason=(EvaluationRowProjectionErrorReason.COVERAGE_EXPECTED_KEY_COLLISION),
    )
    if duplicate_expected:
        raise EvaluationRowCoverageError(
            EvaluationRowProjectionErrorReason.COVERAGE_EXPECTED_KEY_COLLISION,
            "authored Cartesian row-key projection is not one-to-one",
            duplicates=duplicate_expected,
            duplicate_indices=expected_indices,
        )
    duplicate_observed, observed_indices = _duplicate_details(
        observed_keys,
        unhashable_reason=(EvaluationRowProjectionErrorReason.COVERAGE_OBSERVED_KEY_UNHASHABLE),
        duplicate_reason=(EvaluationRowProjectionErrorReason.COVERAGE_OBSERVED_KEY_DUPLICATE),
    )
    if duplicate_observed:
        raise EvaluationRowCoverageError(
            EvaluationRowProjectionErrorReason.COVERAGE_OBSERVED_KEY_DUPLICATE,
            "observed evaluation row keys contain duplicates",
            duplicates=duplicate_observed,
            duplicate_indices=observed_indices,
        )
    expected_set = set(expected)
    observed_set = set(observed_keys)
    missing = tuple(key for key in expected if key not in observed_set)
    unexpected = tuple(key for key in observed_keys if key not in expected_set)
    if missing or unexpected:
        raise EvaluationRowCoverageError(
            EvaluationRowProjectionErrorReason.COVERAGE_KEY_SET_MISMATCH,
            f"evaluation row coverage mismatch: missing={missing!r}, unexpected={unexpected!r}",
            missing=missing,
            unexpected=unexpected,
        )
    return tuple(expected)


def _verify_row_facts(
    item: ResolvedAnalysisInput,
    *,
    row_index: int,
) -> VerifiedEvaluationRowFacts:
    manifest = item.manifest
    manifest_id = getattr(manifest, "id", None)
    if not isinstance(manifest, EvaluationRunManifest) or item.manifest_input is None:
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.INPUT_CONTRACT,
            EvaluationRowProjectionErrorReason.INPUT_MANIFEST_MISSING,
            "evaluation row projection requires a resolved authenticated manifest",
            row_index=row_index,
            manifest_id=manifest_id,
        )
    if item.states is None:
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.INPUT_CONTRACT,
            EvaluationRowProjectionErrorReason.INPUT_STATES_MISSING,
            f"evaluation row {manifest.id!r} has no resolved states",
            row_index=row_index,
            manifest_id=manifest.id,
        )
    ref = item.ref
    if (
        item.manifest_input.ref != ref
        or item.manifest_input.manifest != manifest
        or ref.kind != "EvaluationRunManifest"
        or ref.role != "evaluation_run"
    ):
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.MANIFEST_AUTHORITY,
            EvaluationRowProjectionErrorReason.MANIFEST_AUTHORITY_MISMATCH,
            f"evaluation row {manifest.id!r} disagrees with its manifest authority",
            row_index=row_index,
            manifest_id=manifest.id,
        )
    source = item.evaluation_state_source
    receipt = item.evaluation_state_receipt
    if source is None:
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.STATE_MATERIALIZATION,
            EvaluationRowProjectionErrorReason.STATE_SOURCE_MISSING,
            f"evaluation row {manifest.id!r} lacks a resolved state source",
            row_index=row_index,
            manifest_id=manifest.id,
        )
    if receipt is None:
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.STATE_MATERIALIZATION,
            EvaluationRowProjectionErrorReason.STATE_RECEIPT_MISSING,
            f"evaluation row {manifest.id!r} lacks a resolver-issued state receipt",
            row_index=row_index,
            manifest_id=manifest.id,
            source_kind=source.source_kind,
        )
    portable_ref = ref.model_copy(update={"uri": None})
    if not receipt.matches(item.states, source):
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.STATE_MATERIALIZATION,
            EvaluationRowProjectionErrorReason.STATE_RECEIPT_MISMATCH,
            f"evaluation row {manifest.id!r} states disagree with their resolver receipt",
            row_index=row_index,
            manifest_id=manifest.id,
            source_kind=source.source_kind,
        )
    try:
        run_spec = EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline)
        provenance = verify_evaluation_manifest_provenance(
            ref,
            item.manifest_input.raw_bytes,
            expected_producer_identity=run_spec.evaluation_type,
        )
    except Exception as exc:
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCategory.PROVENANCE,
            EvaluationRowProjectionErrorReason.MANIFEST_PROVENANCE_INVALID,
            f"evaluation row {manifest.id!r} provenance authentication failed",
            row_index=row_index,
            manifest_id=manifest.id,
            source_kind=source.source_kind,
            cause=exc,
        ) from exc
    return VerifiedEvaluationRowFacts(
        manifest_authority=portable_ref,
        manifest=manifest,
        run_spec=run_spec,
        states=item.states,
        state_source=source,
        state_receipt=receipt,
        provenance=provenance,
        parameters=MappingProxyType(dict(run_spec.params)),
        metadata=MappingProxyType(dict(manifest.metadata)),
    )


def _duplicate_details(
    values: Sequence[Any],
    *,
    unhashable_reason: EvaluationRowProjectionErrorReason,
    duplicate_reason: EvaluationRowProjectionErrorReason,
    axis_name: str | None = None,
) -> tuple[tuple[Any, ...], tuple[tuple[int, int], ...]]:
    first_by_value: dict[Any, int] = {}
    duplicates: list[Any] = []
    indices: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        try:
            hash(value)
        except Exception as exc:
            raise EvaluationRowCoverageError(
                unhashable_reason,
                "Cartesian axis values and row keys must be hashable",
                axis_name=axis_name,
                cause=exc,
            ) from exc
        first = first_by_value.get(value)
        if first is None:
            first_by_value[value] = index
        else:
            if value not in duplicates:
                duplicates.append(value)
            indices.append((first, index))
    return tuple(duplicates), tuple(indices)


__all__ = [
    "EvaluationRowCoverageError",
    "EvaluationRowProjection",
    "EvaluationRowProjectionError",
    "EvaluationRowProjectionErrorCategory",
    "EvaluationRowProjectionErrorReason",
    "ProjectedEvaluationRow",
    "VerifiedEvaluationRowFacts",
    "project_verified_evaluation_rows",
    "require_exact_authored_cartesian_coverage",
]
