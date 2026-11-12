"""Exact root selections for a portable Store closure ref."""

from typing import Literal

from pydantic import Field, JsonValue, StrictBool

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.request_refs import ProductRef, ReferenceValue


STORE_CLOSURE_REF_SCHEMA = "feedbax.store_closure_ref.v1"


class ReceiptRoot(ReferenceValue):
    receipt: ContentObjectRef
    product: ProductRef | None = None


class StoreClosureRef(ReferenceValue):
    schema_version: Literal["feedbax.store_closure_ref.v1"] = STORE_CLOSURE_REF_SCHEMA
    roots: tuple[ReceiptRoot | ContentObjectRef | ContentTreeRef, ...] = Field(min_length=1)
    include_ancestry: StrictBool = False


def read_store_closure_ref(value: dict[str, JsonValue]) -> StoreClosureRef:
    closure = StoreClosureRef.model_validate(value)
    if closure.model_dump(mode="json") != value:
        raise ValueError("Store closure ref is incomplete or noncanonical")
    return closure
