"""Bounded successful-result publication over the real immutable project Store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.request_identity import (
    DecodedRequest,
    EncodedRequest,
    decode_request,
    effective_binding_id,
)
from feedbax.contracts.request_refs import (
    ContentRef,
    EffectiveBindingId,
    EffectiveInputMember,
    InputMember,
    LastMember,
    Name,
    NamedMember,
    ProductRef,
    ReferenceValue,
    RequestId,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

INDEX_REF = "proof/effective-successes-v1"


class RequestObject(ReferenceValue):
    id: RequestId
    object: ContentObjectRef
    documents: tuple[ContentObjectRef, ...]


class Product(ReferenceValue):
    name: Name
    content: ContentRef
    descriptor: JsonValue = None


type Products = dict[Name, Annotated[tuple[Product, ...], Field(min_length=1)]]


class Selection(ReferenceValue):
    receipt: ContentObjectRef
    member: Name


class BoundMember(EffectiveInputMember):
    selection: Selection | None = None


type Bindings = dict[Name, tuple[BoundMember, ...]]


class Success(ReferenceValue):
    schema_version: Literal["feedbax.proof.effective-success.v1"] = (
        "feedbax.proof.effective-success.v1"
    )
    request: RequestObject
    attempt: Name
    effective: EffectiveBindingId
    inputs: Bindings
    products: Products = Field(min_length=1)
    supersedes: ContentObjectRef | None

    @model_validator(mode="after")
    def unique_products(self) -> Success:
        for role, members in self.products.items():
            if len({member.name for member in members}) != len(members):
                raise ValueError(f"Repeated product member in {role!r}")
        return self


class SuccessIndex(ReferenceValue):
    schema_version: Literal["feedbax.proof.success-index.v1"] = "feedbax.proof.success-index.v1"
    authored: dict[RequestId, ContentObjectRef] = Field(default_factory=dict)
    effective: dict[EffectiveBindingId, ContentObjectRef] = Field(default_factory=dict)
    attempts: dict[Name, ContentObjectRef] = Field(default_factory=dict)


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
        self._index = (
            SuccessIndex.model_validate(self._expected)
            if self._expected is not None
            else SuccessIndex()
        )
        if self._expected is not None and self._index.model_dump(mode="json") != self._expected:
            raise ValueError("Successful-result index has an incomplete schema")

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
        if content.tree:
            raise ValueError("The bounded Store witness accepts blob bindings")
        ref = ContentObjectRef(
            object_id=f"content-object://sha256/{content.sha256}",
            sha256=content.sha256,
            size_bytes=content.size,
        )
        self.store.verify_object(ref)

    def _receipt(self, ref: ContentObjectRef) -> Success:
        raw = self.store.get_object(ref)
        receipt = Success.model_validate_json(raw)
        if raw != canonical_json_v2_bytes(receipt.model_dump(mode="json")):
            raise ValueError("Successful receipt bytes are not canonical")
        if self._index.attempts.get(receipt.attempt) != ref:
            raise ValueError("Receipt has no attempt in this committed Store generation")
        return receipt

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
        updated = SuccessIndex(
            authored={**self._index.authored, request.id: ref},
            effective={**self._index.effective, effective: ref},
            attempts={**self._index.attempts, attempt: ref},
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
