"""Persist immutable assessments and judgements on the exact successful receipt."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.effective_success import INDEX_REF, decode_success_index
from feedbax.contracts.receipt_validity import ValidityRecord
from feedbax.contracts.request_refs import RequestId
from feedbax.contracts.store_closure import ReceiptRoot, read_store_closure_ref
from feedbax.persistence.effective_success import PublicationConflict, run_ref_name
from feedbax.persistence.store_damage import require_sound

if TYPE_CHECKING:
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

_RECORD = TypeAdapter(ValidityRecord)


def read_validity_record(
    store: ImmutableArtifactBlobProvider, ref: ContentObjectRef, *, receipt: ContentObjectRef,
) -> ValidityRecord:
    require_sound(store, ref)
    raw = store.get_object(ref)
    record = _RECORD.validate_json(raw)
    if record.receipt != receipt or canonical_json_v2_bytes(record.model_dump(mode="json")) != raw:
        raise ValueError("Validity record does not authenticate the named receipt")
    return record


def _receipt_root(value: dict | None) -> ReceiptRoot | None:
    if value is None:
        return None
    (root,) = read_store_closure_ref(value).roots
    if not isinstance(root, ReceiptRoot) or root.product is not None:
        raise ValueError("A run ref must name one complete receipt")
    return root


def read_validity_records(
    store: ImmutableArtifactBlobProvider, request: RequestId, receipt: ContentObjectRef,
) -> tuple[ValidityRecord, ...]:
    root = _receipt_root(store.read_ref(run_ref_name(request)))
    if root is None or root.receipt != receipt:
        return ()
    return tuple(read_validity_record(store, ref, receipt=receipt) for ref in root.validity_records)


def record_validity(
    store: ImmutableArtifactBlobProvider, request: RequestId, record: ValidityRecord,
) -> ContentObjectRef:
    """Commit to the named latest success, preserving concurrent records for that receipt."""
    name = run_ref_name(request)
    with store._exclusive_lock():
        expected = store.read_ref(name)
        root = _receipt_root(expected)
        index = decode_success_index(store.read_ref(INDEX_REF))
        if root is None or root.receipt != record.receipt or index.authored.get(request) != record.receipt:
            raise PublicationConflict("The latest successful receipt changed during the validity check")
        ref = store.store_object(canonical_json_v2_bytes(record.model_dump(mode="json")))
        if ref in root.validity_records:
            return ref
        updated = root.model_copy(update={"validity_records": (*root.validity_records, ref)})
        assert expected is not None
        closure = read_store_closure_ref(expected).model_copy(update={"roots": (updated,)})
        if not store.compare_and_swap_ref(name, expected=expected, replacement=closure.model_dump(mode="json")):
            raise PublicationConflict("The request ref moved during the validity check")
    return ref
