"""Typed records for the bounded effective-binding Store publication."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from feedbax.contracts.artifact_custody import CONTENT_OBJECT_REF_PREFIX, ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.request_identity import DecodedRequest, decode_request
from feedbax.contracts.request_refs import (
    ContentRef,
    EffectiveBindingId,
    EffectiveInputMember,
    LastMember,
    Name,
    NamedMember,
    ProductRef,
    ReferenceValue,
    RequestId,
    Sha256,
)


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
    supersession_times: dict[Sha256, AwareDatetime] = Field(
        default_factory=dict,
        exclude_if=lambda times: not times,
    )

    @model_validator(mode="after")
    def _committed_times(self) -> SuccessIndex:
        if self.supersession_times.keys() - {ref.sha256 for ref in self.attempts.values()}:
            raise ValueError("Supersession time names an uncommitted successor")
        return self


def decode_success_index(value: dict[str, JsonValue] | None) -> SuccessIndex:
    if value is None:
        return SuccessIndex()
    index = SuccessIndex.model_validate(value)
    if index.model_dump(mode="json") != value:
        raise ValueError("Successful-result index has an incomplete or noncanonical schema")
    return index


def success_blob_ref(content: ContentRef) -> ContentObjectRef:
    if content.tree:
        raise ValueError("The bounded Store witness accepts blob bindings")
    return ContentObjectRef(
        object_id=CONTENT_OBJECT_REF_PREFIX + content.sha256,
        sha256=content.sha256,
        size_bytes=content.size,
    )


def read_committed_success(
    index: SuccessIndex,
    ref: ContentObjectRef,
    read_object: Callable[[ContentObjectRef], bytes],
) -> Success:
    receipt = read_success_object(ref, read_object)
    if index.attempts.get(receipt.attempt) != ref:
        raise ValueError("Receipt has no attempt in this committed Store generation")
    return receipt


def read_success_object(
    ref: ContentObjectRef,
    read_object: Callable[[ContentObjectRef], bytes],
) -> Success:
    """Read the immutable receipt bytes at an already authorized exact ref."""
    raw = read_object(ref)
    receipt = Success.model_validate_json(raw)
    if raw != canonical_json_v2_bytes(receipt.model_dump(mode="json")):
        raise ValueError("Successful receipt bytes are not canonical")
    return receipt


def read_success_request(
    request: RequestObject,
    read_object: Callable[[ContentObjectRef], bytes],
) -> DecodedRequest:
    documents = {ref.sha256: ref for ref in request.documents}
    used = set()

    def read_document(digest: str) -> bytes:
        if digest not in documents:
            raise ValueError(f"Request document is unavailable: {digest}")
        used.add(digest)
        return read_object(documents[digest])

    decoded = decode_request(read_object(request.object), read_document)
    if decoded.id != request.id or used != documents.keys() or len(used) != len(request.documents):
        raise ValueError("Stored request differs from its authored ID or document closure")
    return decoded


def select_success_product(reference: ProductRef, receipt: Success) -> Product:
    """Resolve exactly one authored role member from this immutable producer."""
    members = receipt.products.get(reference.role)
    if receipt.request.id != reference.request_id or members is None:
        raise ValueError(f"Receipt does not supply {reference.request_id}/{reference.role}")
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
