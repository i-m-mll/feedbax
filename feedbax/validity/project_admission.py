"""Read retained results and current project observations for one admission batch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.computation_validity import NORMALIZATION_VERSION
from feedbax.contracts.effective_success import EvaluationSubjectValue, Success, select_bound_product
from feedbax.contracts.receipt_validity import CheckConditions, InputDifference, ReceiptValidity
from feedbax.contracts.request_refs import ContentRef, LastMember, ProductRef
from feedbax.persistence.receipt_validity import read_validity_records
from feedbax.validity.admission import QUICK_RECEIPT_DURATION, decide_validity, observe_differences
from feedbax.validity.environment import read_environment
from feedbax.validity.evidence import read_execution_evidence
from feedbax.validity.registry import snapshot_registry
from feedbax.validity.source import source_catalog

if TYPE_CHECKING:
    from feedbax.persistence.effective_success import PublishedSuccess, SuccessView
    from feedbax.plugins.application import ApplicationRegistryBundle


_project_catalog = lru_cache(maxsize=16)(source_catalog)


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

    @cached_property
    def _registry(self):
        return {
            (binding.family, binding.requested_key): binding
            for binding in snapshot_registry(self.registries, self._catalog)
        }

    @cached_property
    def _catalog(self):
        return _project_catalog(self.root)

    @cached_property
    def _environment(self):
        return read_environment(self.root).versions

    def inspect(
        self, view: SuccessView, published: PublishedSuccess, *,
        origins: Mapping[str, ContentObjectRef] | None = None,
    ) -> ObservedReceiptValidity:
        receipt = published.receipt
        inputs = current_input_differences(view, receipt, origins=origins)
        if receipt.trace is None:
            conditions = CheckConditions(code=(), environment=())
            return ObservedReceiptValidity(decide_validity(
                published.ref, conditions=conditions, has_trace=False, input_differences=inputs,
            ), conditions)

        traces = [receipt.trace]
        pending = list(receipt.trace.prefixes)
        visited = set()
        while pending:
            ref = pending.pop()
            if ref in visited:
                continue
            visited.add(ref)
            evidence = read_execution_evidence(view.store, ref)
            traces.append(evidence.trace)
            pending.extend(evidence.trace.prefixes)

        conditions, files = observe_differences(
            traces, catalog=self._catalog, registry=self._registry, environment=self._environment,
            read_object=view.store.get_object,
        )
        computations = [trace.computation for trace in traces]
        code_checkable = all(
            computation is not None and all(
                all(
                    program is not None and all(call.kind != "unidentified" for call in program.calls)
                    for program in (identity.reference, identity.execution)
                )
                for identity in computation.programs
            )
            for computation in computations
        )
        if code_checkable:
            code_checkable = all(
                computation.normalization_version == NORMALIZATION_VERSION
                or all(
                    program is not None and program.raw_program is not None
                    for identity in computation.programs
                    for program in (identity.reference, identity.execution)
                )
                for computation in computations if computation is not None
            )
        environment_checkable = (
            all(computation is not None and computation.prefix is not None for computation in computations)
            or (receipt.request.id.partition(":")[0] in ("analyze", "render", "report") and is_quick(receipt, self.quick_duration))
        )
        records = read_validity_records(view.store, receipt.request.id, published.ref)
        return ObservedReceiptValidity(decide_validity(
            published.ref, conditions=conditions, records=records,
            input_differences=inputs, file_differences=files,
            code_checkable=code_checkable, environment_checkable=environment_checkable,
        ), conditions)


def is_quick(receipt: Success, duration: timedelta) -> bool:
    return (
        receipt.provenance is not None
        and receipt.provenance.ended_at - receipt.provenance.started_at <= duration
    )


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
