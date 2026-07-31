"""Thin projection of resolver-issued evaluation rows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from feedbax.analysis.specs import ResolvedAnalysisInput, ResolvedEvaluationStateHandle
from feedbax.contracts.manifest import (
    AnalysisEvaluationStateSource,
    EvaluationManifestProvenanceEnvelope,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    load_manifest_bytes,
    verify_evaluation_manifest_provenance,
)

ProjectedT = TypeVar("ProjectedT")


class EvaluationRowProjectionErrorCode(StrEnum):
    """Stable boundary failures exposed by evaluation-row projection."""

    INPUT_MANIFEST_MISSING = "input_manifest_missing"
    MANIFEST_AUTHORITY_MISMATCH = "manifest_authority_mismatch"
    MANIFEST_PROVENANCE_INVALID = "manifest_provenance_invalid"
    STATE_HANDLE_MISSING = "state_handle_missing"
    STATE_HANDLE_INVALID = "state_handle_invalid"
    STATE_HANDLE_MISMATCH = "state_handle_mismatch"
    PROJECTOR_FAILED = "projector_failed"


class EvaluationRowProjectionError(ValueError):
    """One stable error code plus the row identity and original cause."""

    def __init__(
        self,
        code: EvaluationRowProjectionErrorCode,
        message: str,
        *,
        row_index: int,
        manifest_id: str,
        source_kind: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.row_index = row_index
        self.manifest_id = manifest_id
        self.source_kind = source_kind
        self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class ResolvedEvaluationRow:
    """Resolver materialization plus authority parsed from authenticated bytes.

    ``handle`` records the source and state supplied by ``resolve_analysis_inputs``.
    Durable artifact bytes are authenticated by the durable loader. Cache and
    recompute state bytes are not durably content-authenticated, and mutation after
    resolution is outside the handle guarantee. Scientific admissibility remains a
    responsibility of the downstream projector.
    """

    manifest_authority: ParentRef
    manifest: EvaluationRunManifest
    run_spec: EvaluationRunSpec
    states: Any
    state_source: AnalysisEvaluationStateSource
    handle: ResolvedEvaluationStateHandle
    provenance: EvaluationManifestProvenanceEnvelope
    parameters: Mapping[str, Any]
    metadata: Mapping[str, Any]


def project_evaluation_rows(
    inputs: Sequence[ResolvedAnalysisInput],
    *,
    project: Callable[[ResolvedEvaluationRow], ProjectedT],
) -> tuple[ProjectedT, ...]:
    """Authenticate row authority, then invoke one downstream cross-field callback."""

    projected: list[ProjectedT] = []
    for index, item in enumerate(inputs):
        row = _resolve_row(item, row_index=index)
        try:
            projected.append(project(row))
        except Exception as exc:
            raise EvaluationRowProjectionError(
                EvaluationRowProjectionErrorCode.PROJECTOR_FAILED,
                f"evaluation row {row.manifest.id!r} projection failed",
                row_index=index,
                manifest_id=row.manifest.id,
                source_kind=row.state_source.source_kind,
                cause=exc,
            ) from exc
    return tuple(projected)


def _resolve_row(
    item: ResolvedAnalysisInput,
    *,
    row_index: int,
) -> ResolvedEvaluationRow:
    manifest_id = item.ref.id
    manifest_input = item.manifest_input
    if manifest_input is None:
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCode.INPUT_MANIFEST_MISSING,
            "evaluation row projection requires an authenticated manifest input",
            row_index=row_index,
            manifest_id=manifest_id,
        )
    ref = item.ref
    if (
        manifest_input.ref != ref
        or ref.kind != "EvaluationRunManifest"
        or ref.role != "evaluation_run"
    ):
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCode.MANIFEST_AUTHORITY_MISMATCH,
            f"evaluation row {manifest_id!r} disagrees with its manifest authority",
            row_index=row_index,
            manifest_id=manifest_id,
        )
    handle = item.evaluation_state_handle
    if handle is None:
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCode.STATE_HANDLE_MISSING,
            f"evaluation row {manifest_id!r} lacks a resolved state handle",
            row_index=row_index,
            manifest_id=manifest_id,
        )
    if type(handle) is not ResolvedEvaluationStateHandle or not handle._is_resolver_issued():
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCode.STATE_HANDLE_INVALID,
            f"evaluation row {manifest_id!r} has an invalid state handle type",
            row_index=row_index,
            manifest_id=manifest_id,
        )
    if (
        item.states is not handle.states
        or item.evaluation_state_source is None
        or not handle._matches_source(item.evaluation_state_source)
    ):
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCode.STATE_HANDLE_MISMATCH,
            f"evaluation row {manifest_id!r} does not match its resolver-issued handle",
            row_index=row_index,
            manifest_id=manifest_id,
            source_kind=handle.source.source_kind,
        )
    try:
        manifest = load_manifest_bytes(manifest_input.raw_bytes)
        if not isinstance(manifest, EvaluationRunManifest):
            raise TypeError("authenticated input is not an EvaluationRunManifest")
        run_spec = EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline)
        provenance = verify_evaluation_manifest_provenance(
            ref,
            manifest_input.raw_bytes,
            expected_producer_identity=run_spec.evaluation_type,
        )
    except Exception as exc:
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCode.MANIFEST_PROVENANCE_INVALID,
            f"evaluation row {manifest_id!r} provenance authentication failed",
            row_index=row_index,
            manifest_id=manifest_id,
            cause=exc,
        ) from exc

    portable_ref = ref.model_copy(update={"uri": None})
    if not handle._matches_authority(portable_ref):
        raise EvaluationRowProjectionError(
            EvaluationRowProjectionErrorCode.STATE_HANDLE_MISMATCH,
            f"evaluation row {manifest.id!r} does not match its resolver-issued handle",
            row_index=row_index,
            manifest_id=manifest.id,
            source_kind=handle.source.source_kind,
        )
    return ResolvedEvaluationRow(
        manifest_authority=portable_ref,
        manifest=manifest,
        run_spec=run_spec,
        states=handle.states,
        state_source=handle.source,
        handle=handle,
        provenance=provenance,
        parameters=run_spec.params,
        metadata=manifest.metadata,
    )


__all__ = [
    "EvaluationRowProjectionError",
    "EvaluationRowProjectionErrorCode",
    "ResolvedEvaluationRow",
    "project_evaluation_rows",
]
