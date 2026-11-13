"""Bind inputs and publish typed successful results in the immutable Store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.checkpoint_projection import CheckpointOccurrenceRef, ContinuationOccurrence
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
from feedbax.contracts.request_refs import (
    ContentRef,
    EffectiveBindingId,
    InputMember,
    ProductRef,
    RequestId,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.receipt_content import continuation_objects, verify_success_content


class PublicationConflict(ValueError):
    """The index generation moved before publication."""


@dataclass(frozen=True)
class PublishedSuccess:
    ref: ContentObjectRef
    receipt: Success


def store_request(store: ImmutableArtifactBlobProvider, request: EncodedRequest) -> RequestObject:
    return RequestObject(
        id=request.id,
        object=store.store_rooted_object(request.data, namespace="proof-effective-request"),
        documents=tuple(
            store.store_rooted_object(data, namespace="proof-effective-document")
            for _, data in sorted(request.documents.items())
        ),
    )


class SuccessView:
    """One committed index generation, shared by binding and successful-result lookup."""

    def __init__(self, store: ImmutableArtifactBlobProvider):
        self.store = store
        self._expected = store.read_ref(INDEX_REF)
        self._index = decode_success_index(self._expected)

    def read_request(self, request: RequestObject) -> DecodedRequest:
        return read_success_request(request, self.store.get_object)

    def _receipt(self, ref: ContentObjectRef) -> Success:
        return read_committed_success(self._index, ref, self.store.get_object)

    def resolve(
        self, request: DecodedRequest, member: InputMember, origin: ContentObjectRef | None,
    ) -> BoundMember:
        reference = member.reference
        if isinstance(reference, ContentRef):
            verify_success_content(self.store, reference)
            return BoundMember(key=member.key, content=reference)
        if origin is None:
            raise ValueError(f"Producer success is unavailable for {member.key!r}")
        producer = self._receipt(origin)
        read_effective_request(producer, self.store.get_object)
        product, occurrence = select_bound_product(request, reference, producer)
        verify_success_content(self.store, product)
        return BoundMember(
            key=member.key,
            content=product.content,
            descriptor=product.descriptor,
            selection=Selection(receipt=origin, member=product.name, occurrence=occurrence),
        )

    def bind(self, request: RequestObject) -> tuple[EffectiveBindingId, Bindings]:
        decoded = self.read_request(request)
        inputs = {
            role: tuple(
                self.resolve(
                    decoded, member,
                    self._index.authored.get(member.reference.request_id)
                    if isinstance(member.reference, ProductRef)
                    else None,
                )
                for member in members
            )
            for role, members in decoded.inputs.items()
        }
        return effective_binding_id(decoded, inputs), inputs

    def read_success(self, receipt_ref: ContentObjectRef) -> PublishedSuccess:
        receipt = self._receipt(receipt_ref)
        decoded = read_effective_request(receipt, self.store.get_object)
        for role, members in decoded.inputs.items():
            for member, bound in zip(members, receipt.inputs[role], strict=True):
                origin = bound.selection.receipt if bound.selection else None
                if self.resolve(decoded, member, origin) != bound:
                    raise ValueError(f"Receipt input provenance differs for {role}.{member.key}")
        for products in receipt.products.values():
            for product in products:
                verify_success_content(self.store, product)
        if receipt.continuation is not None:
            for slot_ref in continuation_objects(receipt.continuation):
                self.store.verify_object(slot_ref)
        return PublishedSuccess(receipt_ref, receipt)

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

    def publish(
        self, request: RequestObject, *, attempt: str, products: Products
    ) -> PublishedSuccess:
        return self._publish(request, attempt, products, bindings=None)

    def publish_bound(
        self, request: RequestObject, *, attempt: str, products: Products, bindings: Bindings,
        continuation: ContinuationOccurrence | None = None,
        recovery: CheckpointOccurrenceRef | None = None,
    ) -> PublishedSuccess:
        """Commit an accepted attempt using its pinned bindings, including on replay."""
        return self._publish(
            request, attempt, products, bindings=bindings, continuation=continuation, recovery=recovery,
        )

    def _publish(
        self, request: RequestObject, attempt: str, products: Products, *, bindings: Bindings | None,
        continuation: ContinuationOccurrence | None = None,
        recovery: CheckpointOccurrenceRef | None = None,
    ) -> PublishedSuccess:
        with self.store._exclusive_lock():
            if existing := self._index.attempts.get(attempt):
                if bindings is None:
                    raise ValueError("This attempt already published a success")
                published = self.read_success(existing)
                if (
                    published.receipt.request, published.receipt.products, published.receipt.inputs,
                    published.receipt.continuation, published.receipt.recovery,
                ) != (request, products, bindings, continuation, recovery):
                    raise ValueError("Published attempt has different request, bindings, or products")
                return published
            if bindings is None:
                effective, inputs = self.bind(request)
            else:
                decoded = self.read_request(request)
                effective = effective_binding_id(decoded, bindings)
                for role, members in decoded.inputs.items():
                    for member, bound in zip(members, bindings[role], strict=True):
                        origin = bound.selection.receipt if bound.selection else None
                        if self.resolve(decoded, member, origin) != bound:
                            raise ValueError(f"Pinned receipt input differs for {role}.{member.key}")
                inputs = bindings
            previous = self._index.authored.get(request.id)
            if previous is not None and self._receipt(previous).request.id != request.id:
                raise ValueError("Superseded receipt belongs to a different authored request")
            receipt = Success(
                request=request,
                attempt=attempt,
                effective=effective,
                inputs=inputs,
                products=products,
                supersedes=previous,
                continuation=continuation,
                recovery=recovery,
            )
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
            if not self.store.compare_and_swap_ref(
                INDEX_REF,
                expected=self._expected,
                replacement=updated.model_dump(mode="json"),
            ):
                raise PublicationConflict("Successful-result index moved during publication")
        return PublishedSuccess(ref, receipt)
