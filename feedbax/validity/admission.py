"""Compare retained observations and apply receipt-scoped validity decisions."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.code_validity import CodeTrace, RegistryBinding, SourceDifference, SourcePath
from feedbax.contracts.receipt_validity import (
    CheckConditions, CodeDifference, CodeReason, EnvironmentDifference, EnvironmentNotice,
    EnvironmentReason, FileDifference, InputDifference, ReceiptValidity, RegistryDifference,
    ValidityAssessment, ValidityJudgement, ValidityReason, ValidityRecord,
)
from feedbax.contracts.request_refs import ReferenceValue

if TYPE_CHECKING:
    from feedbax.validity.source import SourceCatalog


QUICK_RECEIPT_DURATION = timedelta(seconds=2)


def _canonical_order[T: ReferenceValue](items: Iterable[T]) -> tuple[T, ...]:
    distinct = {canonical_json_v2_bytes(item.model_dump(mode="json")): item for item in items}
    return tuple(distinct[key] for key in sorted(distinct))


def _registry_identity(binding: RegistryBinding) -> dict:
    return binding.model_dump(exclude={"version"})


def observe_differences(
    traces: Sequence[CodeTrace], *,
    catalog: SourceCatalog,
    registry: Mapping[tuple[str, str], RegistryBinding],
    environment: Mapping[str, str],
    read_object: Callable[[ContentObjectRef], bytes],
) -> tuple[CheckConditions, tuple[FileDifference, ...]]:
    """Read available source, file and metadata observations without imports or execution."""
    code = []
    outside_computation = []
    versions = []
    files = []
    for trace in traces:
        indeterminate = set(trace.indeterminate)
        indeterminate_files = set(trace.indeterminate_files)
        for source in trace.sources:
            difference = catalog.compare_source(
                source.path, source.content.sha256,
                lambda: read_object(source.content), source.entered,
            )
            if source.path in indeterminate:
                indeterminate.discard(source.path)
                difference = SourceDifference(
                    path=source.path, before=source.content.sha256,
                    after=_file_digest(catalog, source.path), status="indeterminate",
                    statements=difference.statements if difference else (),
                    functions=difference.functions if difference else (),
                )
            if difference is not None:
                packaging = source.host_product and (
                    difference.status != "changed" or difference.statements
                    or not set(difference.functions).isdisjoint(source.host_product)
                )
                (outside_computation if packaging else code).append(difference)
        outside_computation.extend(SourceDifference(
            path=path, before=None, after=_file_digest(catalog, path), status="indeterminate",
        ) for path in indeterminate)
        for binding in trace.registry:
            current = registry.get((binding.family, binding.requested_key))
            if current is None or _registry_identity(binding) != _registry_identity(current):
                code.append(RegistryDifference(before=binding, after=current))
        for recorded in trace.files:
            current_digest = _file_digest(catalog, recorded.path)
            unsettled = recorded.path in indeterminate_files
            indeterminate_files.discard(recorded.path)
            if current_digest != recorded.sha256 or unsettled:
                files.append(FileDifference(
                    path=recorded.path, before=recorded.sha256, after=current_digest,
                    status="indeterminate" if unsettled else "changed",
                ))
        files.extend(FileDifference(
            path=path, before=None, after=_file_digest(catalog, path), status="indeterminate",
        ) for path in indeterminate_files)
        versions.extend(
            EnvironmentDifference(distribution=name, before=version, after=environment.get(name))
            for name, version in trace.environment.items()
            if environment.get(name) != version
        )
    return CheckConditions(
        code=_canonical_order(code), outside_computation=_canonical_order(outside_computation),
        environment=_canonical_order(versions),
    ), _canonical_order(files)


def _file_digest(catalog: SourceCatalog, path: SourcePath) -> str | None:
    try:
        return catalog.file_digest(path)
    except OSError:
        return None


def decide_validity(
    receipt: ContentObjectRef, *,
    conditions: CheckConditions,
    records: Sequence[ValidityRecord] = (),
    input_differences: Sequence[InputDifference] = (),
    file_differences: Sequence[FileDifference] = (),
    code_checkable: bool = False,
    code_unavailable: str | None = None,
    environment_checkable: bool = False,
    has_trace: bool = True,
) -> ReceiptValidity:
    """Decide freshness independently of actions, runtime duration and request identity."""
    reasons: list[ValidityReason] = [*input_differences, *file_differences]
    if not has_trace:
        reasons.append(CodeReason(basis="unverifiable", differences=(), detail="No recorded code trace."))
        return ReceiptValidity(receipt=receipt, state="stale", reasons=tuple(reasons))

    scoped = [record for record in records if record.receipt == receipt]
    assessed_code = assessed_environment = None
    accepted = set()
    for record in scoped:
        if isinstance(record, ValidityJudgement):
            accepted.update(record.differences)
        elif isinstance(record, ValidityAssessment) and record.conditions == conditions:
            assessed_code = record.evidence.code or assessed_code
            assessed_environment = record.evidence.environment or assessed_environment
    outside = tuple(difference for difference in conditions.outside_computation if difference not in accepted)
    if outside:
        reasons.append(CodeReason(basis="source", differences=outside))
    code = tuple(difference for difference in conditions.code if difference not in accepted)
    assess = not input_differences and not file_differences
    checks = []
    unchecked_reason = None
    if code and not (assessed_code and assessed_code.outcome == "equal"):
        reason = CodeReason(
            basis="computation" if assessed_code and assessed_code.outcome == "different" else "source",
            differences=code, detail=assessed_code.detail if assessed_code is not None else code_unavailable,
        )
        if assess and code_checkable and assessed_code is None:
            checks.append("code")
            unchecked_reason = reason
        else:
            reasons.append(reason)

    notice = None
    if conditions.environment:
        detail = assessed_environment.detail if assessed_environment is not None else None
        outcome = assessed_environment.outcome if assessed_environment is not None else None
        if outcome == "different":
            check = "different"
            reasons.append(EnvironmentReason(differences=conditions.environment, detail=detail))
        elif outcome == "equal":
            check = "confirmed"
        elif outcome is None and assess and environment_checkable:
            check = "pending"
            checks.append("environment")
        else:
            check = "not_checked"
        notice = EnvironmentNotice(differences=conditions.environment, check=check, detail=detail)

    state = "stale" if reasons else "unchecked" if unchecked_reason else "current"
    if unchecked_reason:
        reasons.append(unchecked_reason)
    return ReceiptValidity(
        receipt=receipt, state=state, reasons=tuple(reasons), checks=tuple(checks), environment=notice,
    )


def validity_reason(validity: ReceiptValidity) -> str | None:
    """Project structured observations to the existing plain-text reason field."""
    descriptions: list[str] = []
    for reason in validity.reasons:
        match reason:
            case InputDifference(role=role, member=member, detail=detail):
                descriptions.append(f"Input {role}/{member} changed." if detail is None else detail)
            case FileDifference(path=path, status=status):
                detail = "could not be authenticated" if status == "indeterminate" else "changed"
                descriptions.append(f"File {path.scope}/{path.path} {detail}.")
            case EnvironmentReason(differences=changes, detail=detail):
                descriptions.append(detail or f"Numerical environment changed ({_versions(changes)}).")
            case CodeReason(differences=changes, detail=detail):
                names = ", ".join(_code_name(difference) for difference in changes)
                if detail is not None:
                    descriptions.append(f"{detail} ({names})" if names else detail)
                elif names:
                    descriptions.append(f"Code changed in {names}.")
    if validity.state == "unchecked":
        descriptions.append("Checking the lowered computation.")
    if validity.environment is not None and validity.environment.check != "different":
        notice = validity.environment
        descriptions.append(f"Environment changed ({_versions(notice.differences)}).")
        if notice.detail:
            descriptions.append(notice.detail)
    return " ".join(descriptions) or None


def _versions(changes: Sequence[EnvironmentDifference]) -> str:
    return ", ".join(f"{change.distribution} {change.before} → {change.after or 'absent'}" for change in changes)


def _code_name(difference: CodeDifference) -> str:
    match difference:
        case SourceDifference(path=path, statements=statements, functions=functions):
            named = (*functions, *statements)
            location = f"{path.scope}/{path.path}"
            return f"{location} ({', '.join(named)})" if named else location
        case RegistryDifference(before=binding):
            return f"{binding.family} {binding.requested_key}"
