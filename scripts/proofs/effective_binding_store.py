"""Bounded successful-result publication over the real immutable project Store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.effective_success import (
    INDEX_REF as INDEX_REF,
    Bindings,
    BoundMember,
    Product,
    Products,
    RequestObject,
    Selection,
    Success,
    SuccessIndex,
    decode_success_index,
    read_committed_success,
    success_blob_ref,
)
from feedbax.contracts.request_identity import (
    DecodedRequest,
    EncodedRequest,
    decode_request,
    effective_binding_id,
)
from feedbax.contracts.request_refs import (
    ContentRef,
    EffectiveBindingId,
    InputMember,
    LastMember,
    NamedMember,
    ProductRef,
    RequestId,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


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

    def _decode(self, request: RequestObject) -> DecodedRequest:
        documents = {ref.sha256: ref for ref in request.documents}
        used = set()

        def read_document(digest: str) -> bytes:
            if digest not in documents:
                raise ValueError(f"Request document is unavailable: {digest}")
            used.add(digest)
            return self.store.get_object(documents[digest])

        decoded = decode_request(self.store.get_object(request.object), read_document)
        if (
            decoded.id != request.id
            or used != documents.keys()
            or len(used) != len(request.documents)
        ):
            raise ValueError("Stored request differs from its authored ID or document closure")
        return decoded

    def _verify_content(self, content: ContentRef) -> None:
        self.store.verify_object(success_blob_ref(content))

    def _receipt(self, ref: ContentObjectRef) -> Success:
        return read_committed_success(self._index, ref, self.store.get_object)

    def _resolve(self, member: InputMember, origin: ContentObjectRef | None) -> BoundMember:
        reference = member.reference
        if isinstance(reference, ContentRef):
            self._verify_content(reference)
            return BoundMember(key=member.key, content=reference)
        if origin is None:
            raise ValueError(f"Producer success is unavailable for {member.key!r}")
        producer = self._receipt(origin)
        if producer.request.id != reference.request_id or reference.role not in producer.products:
            raise ValueError(f"Receipt does not supply product for {member.key!r}")
        self._decode(producer.request)
        product = _select_product(reference, producer.products[reference.role])
        self._verify_content(product.content)
        return BoundMember(
            key=member.key,
            content=product.content,
            descriptor=product.descriptor,
            selection=Selection(receipt=origin, member=product.name),
        )

    def bind(self, request: RequestObject) -> tuple[EffectiveBindingId, Bindings]:
        decoded = self._decode(request)
        inputs = {
            role: tuple(
                self._resolve(
                    member,
                    self._index.authored.get(member.reference.request_id)
                    if isinstance(member.reference, ProductRef)
                    else None,
                )
                for member in members
            )
            for role, members in decoded.inputs.items()
        }
        return effective_binding_id(decoded, inputs), inputs

    def read_success(self, ref: ContentObjectRef) -> PublishedSuccess:
        receipt = self._receipt(ref)
        decoded = self._decode(receipt.request)
        if effective_binding_id(decoded, receipt.inputs) != receipt.effective:
            raise ValueError("Receipt effective identity differs from its recorded bindings")
        for role, members in decoded.inputs.items():
            for member, bound in zip(members, receipt.inputs[role], strict=True):
                origin = bound.selection.receipt if bound.selection else None
                if self._resolve(member, origin) != bound:
                    raise ValueError(f"Receipt input provenance differs for {role}.{member.key}")
        for products in receipt.products.values():
            for product in products:
                self._verify_content(product.content)
        return PublishedSuccess(ref, receipt)

    def authored(self, request_id: RequestId) -> PublishedSuccess | None:
        ref = self._index.authored.get(request_id)
        if ref is None:
            return None
        result = self.read_success(ref)
        if result.receipt.request.id != request_id:
            raise ValueError("Authored index entry points to a different request")
        return result

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
        effective, inputs = self.bind(request)
        previous = self._index.authored.get(request.id)
        if attempt in self._index.attempts:
            raise ValueError("This attempt already published a success")
        if previous is not None and self._receipt(previous).request.id != request.id:
            raise ValueError("Superseded receipt belongs to a different authored request")
        receipt = Success(
            request=request,
            attempt=attempt,
            effective=effective,
            inputs=inputs,
            products=products,
            supersedes=previous,
        )
        for members in receipt.products.values():
            for product in members:
                self._verify_content(product.content)
        ref = self.store.store_rooted_object(
            canonical_json_v2_bytes(receipt.model_dump(mode="json")),
            namespace="proof-effective-receipt",
        )
        with self.store._exclusive_lock():
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


def _select_product(reference: ProductRef, members: tuple[Product, ...]) -> Product:
    match reference.select:
        case None if len(members) == 1:
            return members[0]
        case LastMember():
            return members[-1]
        case NamedMember(member=name):
            for product in members:
                if product.name == name:
                    return product
    raise ValueError(f"Product selector has no supported member for {reference.role!r}")
