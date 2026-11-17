"""Exact root selections for a portable Store closure ref."""

from typing import Literal

from pydantic import Field, JsonValue, StrictBool, model_validator

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.code_validity import CheckpointEvidenceRoot
from feedbax.contracts.request_refs import ProductRef, ReferenceValue


STORE_CLOSURE_REF_SCHEMA = "feedbax.store_closure_ref.v1"


class ReceiptRoot(ReferenceValue):
    receipt: ContentObjectRef
    product: ProductRef | None = None
    validity_records: tuple[ContentObjectRef, ...] = Field(default=(), exclude_if=lambda refs: not refs)

    @model_validator(mode="after")
    def _whole_receipt_validity(self) -> "ReceiptRoot":
        if self.validity_records and (
            self.product is not None
            or len({ref.sha256 for ref in self.validity_records}) != len(self.validity_records)
        ):
            raise ValueError("Validity records must be distinct and name a whole receipt")
        return self


class StoreClosureRef(ReferenceValue):
    schema_version: Literal["feedbax.store_closure_ref.v1"] = STORE_CLOSURE_REF_SCHEMA
    roots: tuple[ReceiptRoot | CheckpointEvidenceRoot | ContentObjectRef | ContentTreeRef, ...] = Field(min_length=1)
    include_ancestry: StrictBool = False


def read_store_closure_ref(value: dict[str, JsonValue]) -> StoreClosureRef:
    closure = StoreClosureRef.model_validate(value)
    if closure.model_dump(mode="json") != value:
        raise ValueError("Store closure ref is incomplete or noncanonical")
    return closure
