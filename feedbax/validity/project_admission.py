"""Read retained results and current project observations for one admission batch."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.code_validity import CodeTrace, ReceiptEvidence
from feedbax.contracts.computation_validity import NORMALIZATION_VERSION, ComputationTrace
from feedbax.contracts.effective_success import EvaluationSubjectValue, Success, select_bound_product
from feedbax.contracts.receipt_validity import CheckConditions, InputDifference, ReceiptValidity
from feedbax.contracts.request_refs import ContentRef, LastMember, ProductRef
from feedbax.persistence.receipt_validity import read_validity_records
from feedbax.validity.admission import QUICK_RECEIPT_DURATION, decide_validity, observe_differences
from feedbax.validity.environment import read_environment
from feedbax.validity.evidence import execution_evidence_identity, read_execution_evidence
from feedbax.validity.registry import snapshot_registry
from feedbax.validity.source import source_catalog

if TYPE_CHECKING:
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.persistence.effective_success import PublishedSuccess, SuccessView
    from feedbax.plugins.application import ApplicationRegistryBundle


@dataclass(frozen=True)
class ObservedReceiptValidity:
    decision: ReceiptValidity
    conditions: CheckConditions


class ProjectValidity:
    """One lock and registry observation, shared across a saved Plan or admission."""

    def __init__(
        self, root: Path, registries: ApplicationRegistryBundle, *, quick_duration: timedelta = QUICK_RECEIPT_DURATION,
    ):
        self.root = root
        self.registries = registries
        self.quick_duration = quick_duration

    def refresh(self) -> ProjectValidity:
        return ProjectValidity(self.root, self.registries, quick_duration=self.quick_duration)

    @cached_property
    def _registry(self):
        return {
            (binding.family, binding.requested_key): binding
            for binding in snapshot_registry(self.registries, self._catalog)
        }

    @cached_property
    def _catalog(self):
        return source_catalog(self.root)

    @cached_property
    def _environment(self):
        return read_environment(self.root).versions

    def inspect(
        self, view: SuccessView, published: PublishedSuccess, *,
        origins: Mapping[str, ContentObjectRef] | None = None,
        checkpoint_differences: Sequence[InputDifference] = (),
    ) -> ObservedReceiptValidity:
        receipt = published.receipt
        inputs = (*checkpoint_differences, *current_input_differences(view, receipt, origins=origins))
        if receipt.trace is None:
            conditions = CheckConditions(code=(), environment=())
            return ObservedReceiptValidity(decide_validity(
                published.ref, conditions=conditions, has_trace=False, input_differences=inputs,
            ), conditions)

        traces = [receipt.trace, *(evidence.trace for evidence in _prefixes(view.store, receipt.trace))]

        conditions, files = observe_differences(
            traces, catalog=self._catalog, registry=self._registry, environment=self._environment,
            read_object=view.store.get_object,
        )
        computations = tuple(trace.computation for trace in traces)
        opaque = next((trace.computation_unavailable for trace in traces if trace.computation_unavailable), None)
        code_unavailable = (
            f"Starting arguments for {opaque.driver.name} are unavailable for comparison."
            if opaque is not None else _unavailable_computation(computations)
        )
        code_checkable = code_unavailable is None and all(computation is not None for computation in computations)
        environment_checkable = (
            all(computation is not None and computation.prefix is not None for computation in computations)
            or (receipt.request.id.partition(":")[0] in ("analyze", "render", "report") and is_quick(receipt, self.quick_duration, view.store))
        )
        records = read_validity_records(view.store, receipt.request.id, published.ref)
        return ObservedReceiptValidity(decide_validity(
            published.ref, conditions=conditions, records=records,
            input_differences=inputs, file_differences=files,
            code_checkable=code_checkable, code_unavailable=code_unavailable,
            environment_checkable=environment_checkable,
        ), conditions)


def _unavailable_computation(computations: tuple[ComputationTrace | None, ...]) -> str | None:
    for computation in computations:
        if computation is None:
            continue
        for identity in computation.programs:
            for program in (identity.reference, identity.execution):
                if program is None:
                    return "The fixed reference-platform program is unavailable."
                unidentified = tuple(call.target for call in program.calls if call.kind == "unidentified")
                if unidentified:
                    return f"Unidentified foreign calls prevent an exact check ({', '.join(unidentified)})."
                if computation.normalization_version != NORMALIZATION_VERSION and program.raw_program is None:
                    return "The retained program cannot be compared under the current normalization."
    return None


def _prefixes(store: ImmutableArtifactBlobProvider, trace: CodeTrace) -> Iterator[ReceiptEvidence]:
    pending = list(trace.prefixes)
    visited = set()
    while pending:
        ref = pending.pop()
        identity = execution_evidence_identity(ref)
        if identity in visited:
            continue
        visited.add(identity)
        evidence = read_execution_evidence(store, ref)
        yield evidence
        pending.extend(evidence.trace.prefixes)


def is_quick(receipt: Success, duration: timedelta, store: ImmutableArtifactBlobProvider) -> bool:
    if receipt.provenance is None or receipt.trace is None:
        return False
    elapsed = receipt.provenance.ended_at - receipt.provenance.started_at
    if elapsed > duration:
        return False
    for prefix in _prefixes(store, receipt.trace):
        elapsed += prefix.provenance.ended_at - prefix.provenance.started_at
        if elapsed > duration:
            return False
    return elapsed <= duration


def current_input_differences(
    view: SuccessView, receipt: Success, *,
    origins: Mapping[str, ContentObjectRef] | None = None,
) -> tuple[InputDifference, ...]:
    decoded = view.read_request(receipt.request)
    differences = []
    for role, members in decoded.inputs.items():
        for member, bound in zip(members, receipt.inputs[role], strict=True):
            reference = member.reference
            if not isinstance(reference, ProductRef) or bound.checkpoint_model is not None:
                continue
            if bound.evaluation_subject is not None and not isinstance(reference.select, LastMember):
                continue
            origin = origins.get(reference.request_id) if origins is not None else None
            if origin is None:
                origin = view.authored_ref(reference.request_id)
            if bound.selection is not None and bound.selection.receipt == origin:
                continue

            content: ContentRef | None = None
            descriptor = None
            if origin is None and bound.evaluation_subject is not None:
                from feedbax.persistence.effective_success import terminal_train_subject

                terminal = terminal_train_subject(view.store, reference.request_id)
                if terminal is not None:
                    content = terminal.model.content
            elif origin is not None:
                try:
                    producer = view.read_receipt(origin)
                    product, _ = select_bound_product(decoded, reference, producer)
                    content, descriptor = product.content, product.descriptor
                except (KeyError, OSError, ValueError):
                    pass
            if bound.evaluation_subject is not None:
                value = EvaluationSubjectValue.model_validate(bound.descriptor)
                if content is not None and value.matches_terminal_model(content):
                    continue
            elif (content, descriptor) == (bound.content, bound.descriptor):
                continue
            differences.append(InputDifference(role=role, member=member.key, before=bound.content, after=content))
    return tuple(differences)
