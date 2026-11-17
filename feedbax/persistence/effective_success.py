"""Bind inputs and publish typed successful results in the immutable Store."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from typing import TYPE_CHECKING

from pydantic import JsonValue, TypeAdapter

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.checkpoint_projection import CheckpointOccurrenceRef, ContinuationOccurrence
from feedbax.contracts.execution_target import AttemptTarget, LOCAL_TARGET
from feedbax.contracts.effective_success import (
    INDEX_REF,
    Bindings,
    BoundMember,
    Products,
    RequestObject,
    Selection,
    Success,
    SuccessIndex,
    decode_success_index,
    exact_model_checkpoint_input,
    read_committed_success,
    read_effective_request,
    read_success_request,
    select_bound_product,
)
from feedbax.contracts.request_identity import (
    DecodedRequest,
    EncodedRequest,
    effective_binding_id,
)
from feedbax.contracts.model_input import DOCUMENT_INPUT_MEDIA_TYPES, document_input_object, read_document_input
from feedbax.contracts.request_refs import (
    BestRank,
    ContentRef,
    EffectiveBindingId,
    InputMember,
    ProductRef,
    RequestId,
    RequestInputs,
)
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef, read_store_closure_ref
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.checkpoint_publication import checkpoint_model_objects
from feedbax.persistence.receipt_content import continuation_objects, verify_success_content
from feedbax.persistence.evaluation_subject import (
    evaluation_subject_objects,
    terminal_train_subject_at,
    verify_evaluation_success_evidence,
)
from feedbax.persistence.store_damage import KnownDamage, require_sound

if TYPE_CHECKING:
    from feedbax.contracts.effective_success import TerminalTrainSubject


class PublicationConflict(ValueError):
    """The index generation moved before publication."""


@dataclass(frozen=True)
class PublishedSuccess:
    ref: ContentObjectRef
    receipt: Success


def run_ref_name(request_id: RequestId) -> str:
    return f"runs/{request_id.partition(':')[0]}/{request_id}"


def _receipt_root(selection: StoreClosureRef) -> ReceiptRoot:
    if len(selection.roots) != 1 or not isinstance(root := selection.roots[0], ReceiptRoot):
        raise ValueError("A run ref requires one exact receipt root")
    return root


_REQUEST_ID = TypeAdapter(RequestId)


def terminal_train_subject(store: ImmutableArtifactBlobProvider, run: str) -> TerminalTrainSubject | None:
    """Read an exact terminal model using the authoritative run ref."""
    request_id = _REQUEST_ID.validate_python(run)
    if not request_id.startswith("train:"):
        raise ValueError(f"Evaluate subject {request_id!r} is not a Train result")
    name = run_ref_name(request_id)
    if not os.path.lexists(store._ref_path(name)):
        return None
    with store._exclusive_lock(create=False):
        value = store.read_ref(name)
        if value is None:
            return None
        root = _receipt_root(read_store_closure_ref(value))
        return terminal_train_subject_at(store, root.receipt, request_id, selection=root.product)


def store_request(store: ImmutableArtifactBlobProvider, request: EncodedRequest) -> RequestObject:
    return RequestObject(
        id=request.id,
        object=store.store_rooted_object(request.data, namespace="proof-effective-request"),
        documents=tuple(
            store.store_rooted_object(data, namespace="proof-effective-document")
            for _, data in sorted(request.documents.items())
        ),
    )


def store_document_inputs(
    store: ImmutableArtifactBlobProvider, inputs: RequestInputs, read_document: Callable[[str], bytes],
) -> tuple[ContentObjectRef, ...]:
    """Pin the authored bytes identified by direct document inputs before admission."""
    refs = {}
    for members in inputs.values():
        for member in members:
            content = member.reference
            if not isinstance(content, ContentRef) or content.media_type not in DOCUMENT_INPUT_MEDIA_TYPES:
                continue
            exact = document_input_object(content)
            raw = read_document(exact.sha256)
            if store.store_rooted_object(raw, namespace="project-document-input") != exact:
                raise ValueError("Planned document input differs from its immutable reference")
            read_document_input(content, store)
            refs[exact.sha256] = exact
    return tuple(refs.values())


class SuccessView:
    """One committed index generation, shared by binding and successful-result lookup."""

    def __init__(self, store: ImmutableArtifactBlobProvider):
        self.store = store
        self._expected = store.read_ref(INDEX_REF)
        self._index = decode_success_index(self._expected)

    def read_request(self, request: RequestObject) -> DecodedRequest:
        return read_success_request(request, self.store.get_object)

    def read_receipt(self, ref: ContentObjectRef) -> Success:
        return read_committed_success(self._index, ref, self.store.get_object)

    def resolve(
        self, request: DecodedRequest, member: InputMember, origin: ContentObjectRef | None,
    ) -> BoundMember:
        reference = member.reference
        if isinstance(reference, ContentRef):
            verify_success_content(self.store, reference)
            return BoundMember(key=member.key, content=reference)
        if isinstance(reference, BestRank):
            raise ValueError("Best requires an authenticated Host selection")
        if origin is None:
            raise ValueError(f"Producer success is unavailable for {member.key!r}")
        producer = self.read_receipt(origin)
        read_effective_request(producer, self.store.get_object)
        product, occurrence = select_bound_product(request, reference, producer)
        verify_success_content(self.store, product)
        return BoundMember(
            key=member.key,
            content=product.content,
            descriptor=product.descriptor,
            selection=Selection(receipt=origin, member=product.name, occurrence=occurrence),
        )

    def _verify_bound(self, decoded: DecodedRequest, member: InputMember, bound: BoundMember) -> None:
        if bound.checkpoint_model is not None:
            exact = exact_model_checkpoint_input(decoded)
            if exact is None or exact[1] != member:
                raise ValueError("Unexpected checkpoint model evidence")
            for ref in checkpoint_model_objects(self.store, member, bound):
                require_sound(self.store, ref)
                self.store.verify_object(ref)
        elif bound.evaluation_subject is not None:
            for ref in evaluation_subject_objects(self.store, decoded, member, bound):
                require_sound(self.store, ref)
                self.store.verify_object(ref)
        else:
            origin = bound.selection.receipt if bound.selection else None
            if self.resolve(decoded, member, origin) != bound:
                raise ValueError(f"Receipt input provenance differs for {member.key!r}")

    def bind(
        self, request: RequestObject, *, origins: Mapping[RequestId, ContentObjectRef] | None = None,
        checkpoint_model: BoundMember | None = None,
    ) -> tuple[EffectiveBindingId, Bindings]:
        decoded = self.read_request(request)
        checkpoint = exact_model_checkpoint_input(decoded)
        if (checkpoint is None) != (checkpoint_model is None):
            raise ValueError("Analyze requires the exact Host checkpoint binding")
        model_member = checkpoint[1] if checkpoint is not None else None
        if model_member is not None and checkpoint_model is not None:
            self._verify_bound(decoded, model_member, checkpoint_model)
        if origins is None:
            origins = self._index.authored
        else:
            expected = {
                member.reference.request_id
                for members in decoded.inputs.values()
                for member in members
                if member != model_member and isinstance(member.reference, ProductRef)
            }
            if origins.keys() != expected:
                raise ValueError("Explicit origins must cover the exact authored producer requests")
        inputs = {
            role: tuple(
                checkpoint_model if member == model_member and checkpoint_model is not None else self.resolve(
                    decoded, member,
                    origins.get(member.reference.request_id)
                    if isinstance(member.reference, ProductRef)
                    else None,
                )
                for member in members
            )
            for role, members in decoded.inputs.items()
        }
        return effective_binding_id(decoded, inputs), inputs

    def read_success(self, receipt_ref: ContentObjectRef) -> PublishedSuccess:
        receipt = self.read_receipt(receipt_ref)
        decoded = read_effective_request(receipt, self.store.get_object)
        self._verify_inputs(decoded, receipt.inputs)
        if receipt.evaluation_manifest is not None:
            verify_evaluation_success_evidence(self.store, receipt)
        for products in receipt.products.values():
            for product in products:
                verify_success_content(self.store, product)
        if receipt.continuation is not None:
            for slot_ref in continuation_objects(receipt.continuation):
                require_sound(self.store, slot_ref)
                self.store.verify_object(slot_ref)
        return PublishedSuccess(receipt_ref, receipt)

    def verify_bindings(self, request: RequestObject, bindings: Bindings) -> EffectiveBindingId:
        """Authenticate a frozen binding without consulting current producer aliases."""
        decoded = self.read_request(request)
        effective = effective_binding_id(decoded, bindings)
        self._verify_inputs(decoded, bindings)
        return effective

    def _verify_inputs(self, decoded: DecodedRequest, bindings: Bindings) -> None:
        for role, members in decoded.inputs.items():
            for member, bound in zip(members, bindings[role], strict=True):
                self._verify_bound(decoded, member, bound)

    def read_owned_success(self, receipt_ref: ContentObjectRef) -> PublishedSuccess:
        """Verify committed replay evidence without rereading historical inputs."""
        from feedbax.persistence.receipt_closure import ReceiptClosure

        closure = ReceiptClosure(self.store, self._index)
        receipt = closure.receipt_at(receipt_ref)
        for ref in closure.owned_objects(receipt_ref):
            require_sound(self.store, ref)
            self.store.verify_object(ref)
        read_effective_request(receipt, self.store.get_object)
        if receipt.evaluation_manifest is not None:
            verify_evaluation_success_evidence(self.store, receipt, verify_bound_closure=False)
        return PublishedSuccess(receipt_ref, receipt)

    def owned_damage(self, attempt: str, ref: ContentObjectRef) -> str | None:
        """Check only this result's current read dependencies for known damage."""
        from feedbax.persistence.receipt_closure import ReceiptClosure
        if self.attempt_ref(attempt) != ref:
            raise ValueError(f"Receipt differs from the indexed attempt {attempt!r}")
        closure = ReceiptClosure(self.store, self._index)
        try:
            closure.receipt_for_attempt(attempt)
            for member in closure.owned_objects(ref):
                require_sound(self.store, member)
        except KnownDamage as error:
            return str(error)
        return None


    def authored(self, request_id: RequestId) -> PublishedSuccess | None:
        ref = self._index.authored.get(request_id)
        if ref is None:
            return None
        result = self.read_success(ref)
        if result.receipt.request.id != request_id:
            raise ValueError("Authored index entry points to a different request")
        return result

    def attempt_ref(self, attempt: str) -> ContentObjectRef | None:
        return self._index.attempts.get(attempt)

    def read_attempt(self, attempt: str) -> PublishedSuccess | None:
        ref = self.attempt_ref(attempt)
        return None if ref is None else self.read_success(ref)

    def lookup(self, request: RequestObject) -> PublishedSuccess | None:
        effective, _ = self.bind(request)
        ref = self._index.effective.get(effective)
        if ref is None:
            return None
        result = self.read_success(ref)
        if result.receipt.effective != effective:
            raise ValueError("Effective index entry points to different bindings")
        return result

    def admit_collected(
        self, receipt_ref: ContentObjectRef, *, attempt: str, request_id: RequestId,
        supersession_times: Mapping[str, datetime], promote: bool,
        expected_target: AttemptTarget | None = None,
        selection: StoreClosureRef | None = None,
    ) -> PublishedSuccess:
        """Admit exact transferred bytes and update visibility in one Store generation."""
        from feedbax.persistence.receipt_closure import ReceiptClosure

        with self.store._exclusive_lock():
            closure = ReceiptClosure(self.store)
            receipt = closure.receipt_at(receipt_ref)
            if (receipt.attempt, receipt.request.id) != (attempt, request_id):
                raise ValueError("Collected receipt differs from the expected attempt and request")
            if expected_target is not None and receipt.target != expected_target:
                raise ValueError("Collected receipt differs from the expected execution target")
            if selection is not None:
                if _receipt_root(selection).receipt != receipt_ref or not promote:
                    raise ValueError("Imported selection differs from the receipt being admitted")
                current = self._index.authored.get(request_id)
                if current not in (None, receipt_ref):
                    raise PublicationConflict(
                        f"Run {request_id} already has receipt {current.sha256}; incoming receipt is {receipt_ref.sha256}"
                    )
                self._verify_collected_objects(closure.walk(selection))
                if current == receipt_ref:
                    selection = StoreClosureRef(roots=(ReceiptRoot(receipt=receipt_ref),))
                self._publish_run_ref(request_id, selection, previous=None, importing=True)
                return PublishedSuccess(receipt_ref, receipt)
            graph = closure.for_admission(receipt_ref)
            authored = dict(self._index.authored)
            effective = dict(self._index.effective)
            attempts = dict(self._index.attempts)
            times = dict(self._index.supersession_times)
            current = authored.get(request_id)
            if promote:
                if current not in (None, receipt_ref, receipt.supersedes):
                    raise PublicationConflict("Destination has a different authored success")
                authored[request_id] = receipt_ref
                if effective.get(receipt.effective) in (None, receipt_ref, receipt.supersedes):
                    effective[receipt.effective] = receipt_ref

            for ref, result in graph.receipts:
                if attempts.setdefault(result.attempt, ref) != ref:
                    raise PublicationConflict("Destination attempt names a different success receipt")
                if result.supersedes is not None:
                    when = supersession_times.get(ref.sha256)
                    if when is None or times.setdefault(ref.sha256, when) != when:
                        raise PublicationConflict("Collected receipt has missing or conflicting supersession time")

            self._verify_collected_objects(graph.objects)
            if promote:
                self._check_run_predecessor(request_id, receipt_ref, receipt.supersedes)
            self._commit_index(SuccessIndex(
                authored=authored, effective=effective, attempts=attempts, supersession_times=times,
            ))
            if promote:
                self._publish_run_ref(request_id, StoreClosureRef(roots=(ReceiptRoot(receipt=receipt_ref),)),
                                      previous=receipt.supersedes)
        return PublishedSuccess(receipt_ref, receipt)

    def _verify_collected_objects(self, objects: tuple[ContentObjectRef, ...]) -> None:
        for ref in objects:
            require_sound(self.store, ref)
            self.store.verify_object(ref, verifier="receipt collection admission")

    def _check_run_predecessor(
        self, request_id: RequestId, receipt: ContentObjectRef, previous: ContentObjectRef | None,
    ) -> dict[str, JsonValue] | None:
        value = self.store.read_ref(run_ref_name(request_id))
        if value is not None:
            current = _receipt_root(read_store_closure_ref(value)).receipt
            if current not in (receipt, previous):
                raise PublicationConflict(
                    f"Run {request_id} already has receipt {current.sha256}; incoming receipt is {receipt.sha256}"
                )
        return value

    def _publish_run_ref(
        self, request_id: RequestId, selection: StoreClosureRef, *, previous: ContentObjectRef | None,
        importing: bool = False,
    ) -> None:
        receipt = _receipt_root(selection).receipt
        expected = self._check_run_predecessor(request_id, receipt, previous)
        # An identical receipt already admitted by this Store keeps its authority.
        if importing and expected is not None:
            return
        if not self.store.compare_and_swap_ref(
            run_ref_name(request_id), expected=expected, replacement=selection.model_dump(mode="json"),
        ):
            raise PublicationConflict(f"Run {request_id} moved during publication")

    def ensure_published_run_ref(self, ref: ContentObjectRef, receipt: Success) -> None:
        """Reconcile the ordinary ref when replay observes an already committed result."""
        with self.store._exclusive_lock():
            index = decode_success_index(self.store.read_ref(INDEX_REF))
            if index.authored.get(receipt.request.id) == ref:
                self._publish_run_ref(
                    receipt.request.id, StoreClosureRef(roots=(ReceiptRoot(receipt=ref),)),
                    previous=receipt.supersedes,
                )

    def publish(
        self, request: RequestObject, *, attempt: str, products: Products
    ) -> PublishedSuccess:
        return self._publish(request, attempt, products, bindings=None)

    def publish_bound(
        self, request: RequestObject, *, attempt: str, products: Products, bindings: Bindings,
        continuation: ContinuationOccurrence | None = None,
        recovery: CheckpointOccurrenceRef | None = None,
        evaluation_manifest: ContentObjectRef | None = None,
        target: AttemptTarget = LOCAL_TARGET,
    ) -> PublishedSuccess:
        """Commit an accepted attempt using its pinned bindings, including on replay."""
        return self._publish(
            request, attempt, products, bindings=bindings, continuation=continuation, recovery=recovery,
            evaluation_manifest=evaluation_manifest, target=target,
        )

    def _publish(
        self, request: RequestObject, attempt: str, products: Products, *, bindings: Bindings | None,
        continuation: ContinuationOccurrence | None = None,
        recovery: CheckpointOccurrenceRef | None = None,
        evaluation_manifest: ContentObjectRef | None = None,
        target: AttemptTarget = LOCAL_TARGET,
    ) -> PublishedSuccess:
        with self.store._exclusive_lock():
            if existing := self._index.attempts.get(attempt):
                if bindings is None:
                    raise ValueError("This attempt already published a success")
                published = self.read_owned_success(existing)
                if (
                    published.receipt.request, published.receipt.products, published.receipt.inputs,
                    published.receipt.continuation, published.receipt.recovery, published.receipt.evaluation_manifest,
                    published.receipt.target,
                ) != (request, products, bindings, continuation, recovery, evaluation_manifest, target):
                    raise ValueError("Published attempt has different request, bindings, or products")
                self.ensure_published_run_ref(existing, published.receipt)
                return published
            if bindings is None:
                effective, inputs = self.bind(request)
            else:
                effective = self.verify_bindings(request, bindings)
                inputs = bindings
            previous = self._index.authored.get(request.id)
            if previous is not None:
                try:
                    prior = self.read_receipt(previous)
                except KnownDamage as error:
                    if error.observation.expected != previous:
                        raise
                    # The authenticated index still pins the exact superseded receipt.
                else:
                    if prior.request.id != request.id:
                        raise ValueError("Superseded receipt belongs to a different authored request")
            receipt = Success(
                request=request,
                attempt=attempt,
                target=target,
                effective=effective,
                inputs=inputs,
                products=products,
                supersedes=previous,
                continuation=continuation,
                recovery=recovery,
                evaluation_manifest=evaluation_manifest,
            )
            if evaluation_manifest is not None:
                verify_evaluation_success_evidence(self.store, receipt)
            for members in receipt.products.values():
                for product in members:
                    verify_success_content(self.store, product)
            if continuation is not None:
                for slot_ref in continuation_objects(continuation):
                    self.store.verify_object(slot_ref)
            raw = canonical_json_v2_bytes(receipt.model_dump(mode="json"))
            ref = (
                self.store.store_rooted_object(raw, namespace="proof-effective-receipt")
                if bindings is None else self.store.store_object(raw)
            )
            times = dict(self._index.supersession_times)
            if previous is not None:
                times[ref.sha256] = datetime.now(UTC)
            updated = SuccessIndex(
                authored={**self._index.authored, request.id: ref},
                effective={**self._index.effective, effective: ref},
                attempts={**self._index.attempts, attempt: ref},
                supersession_times=times,
            )
            self._check_run_predecessor(request.id, ref, previous)
            self._commit_index(updated)
            self._publish_run_ref(request.id, StoreClosureRef(roots=(ReceiptRoot(receipt=ref),)), previous=previous)
        return PublishedSuccess(ref, receipt)

    def _commit_index(self, updated: SuccessIndex) -> None:
        if not self.store.compare_and_swap_ref(
            INDEX_REF, expected=self._expected, replacement=updated.model_dump(mode="json"),
        ):
            raise PublicationConflict("Successful-result index moved during publication")
