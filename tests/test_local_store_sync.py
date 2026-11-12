from datetime import UTC, datetime
import errno
from multiprocessing import get_context
import os
from pathlib import Path

import pytest

from feedbax.contracts.artifact_custody import ArtifactBlobIntegrityError, ContentObjectRef
from feedbax.contracts.effective_success import INDEX_REF, success_blob_ref
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_refs import (
    InputMember,
    LastMember,
    NamedMember,
    ProductRef,
    SlotMember,
)
from feedbax.contracts.request_specs import ReportRequestSpec
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.reachability import measure_store
from feedbax.persistence.store_sync import sync_store
from scripts.proofs.effective_binding_store import SuccessView, store_request
from scripts.proofs.effective_binding_witness import document_product


def _interrupt_after_clone(
    source_root: Path, destination_root: Path, closure: StoreClosureRef, marker: Path
) -> None:
    class InterruptedStore(ImmutableArtifactBlobProvider):
        def _clone_object_from(self, source, object_ref):
            if super()._clone_object_from(source, object_ref):
                marker.write_text(object_ref.model_dump_json())
                os._exit(74)
            return False

    sync_store(
        ImmutableArtifactBlobProvider(source_root),
        InterruptedStore(destination_root),
        closure,
        destination_ref="packets/ancestry",
    )


def test_selected_receipt_sync_survives_interruption(tmp_path):
    source = ImmutableArtifactBlobProvider(tmp_path / "source")
    destination = ImmutableArtifactBlobProvider(tmp_path / "destination")
    producer = store_request(
        source, encode_request(ReportRequestSpec(report_type="proof.sync-producer"), {})
    )
    kept, sibling, last = (
        document_product(source, name, f'{{"kind":"fixture","id":"{name}"}}'.encode())
        for name in ("kept", "sibling", "last")
    )
    original = SuccessView(source).publish(
        producer, attempt="producer-01", products={"products": (kept, sibling, last)}
    )
    consumer = store_request(
        source,
        encode_request(
            ReportRequestSpec(report_type="proof.sync-consumer"),
            {
                "documents": tuple(
                    InputMember(
                        key=key,
                        reference=ProductRef(request_id=producer.id, role="products", select=rule),
                    )
                    for key, rule in (
                        ("first", NamedMember(member="kept")),
                        ("last", LastMember()),
                    )
                )
            },
        ),
    )
    report = document_product(source, "report", b'{"kind":"report"}')
    result = SuccessView(source).publish(
        consumer, attempt="consumer-01", products={"reports": (report, sibling)}
    )
    latest = SuccessView(source).publish(
        producer, attempt="producer-02", products={"products": (sibling,)}
    )
    assert SuccessView(source).authored(producer.id) == latest

    product = ProductRef(
        request_id=consumer.id, role="reports", select=NamedMember(member="report")
    )
    root = ReceiptRoot(receipt=result.ref, product=product)
    selected = StoreClosureRef(roots=(root,))
    report_ref = success_blob_ref(report.content)
    selected_refs = (result.ref, consumer.object, *consumer.documents, report_ref)
    try:
        initial = sync_store(source, destination, selected, destination_ref="packets/selected")
    except OSError as error:
        if error.errno != errno.ENOTSUP:
            raise
        pytest.skip("test volume does not support copy-on-write cloning")
    assert (initial.cloned, initial.reused) == (len(selected_refs), 0)
    for ref in selected_refs:
        assert destination.get_object(ref) == source.get_object(ref)
    with pytest.raises(FileNotFoundError):
        destination.verify_object(original.ref)

    ancestry = StoreClosureRef(roots=(root,), include_ancestry=True)
    marker = tmp_path / "first-clone.json"
    child = get_context("spawn").Process(
        target=_interrupt_after_clone, args=(source.root, destination.root, ancestry, marker)
    )
    child.start()
    try:
        child.join(15)
        assert child.exitcode == 74
    finally:
        if child.is_alive():
            child.kill()
            child.join(5)
    first_clone = ContentObjectRef.model_validate_json(marker.read_bytes())
    assert first_clone not in selected_refs
    assert destination.get_object(first_clone) == source.get_object(first_clone)
    assert destination.read_ref("packets/ancestry") is None

    ancestor_refs = (
        original.ref,
        producer.object,
        *producer.documents,
        success_blob_ref(kept.content),
        success_blob_ref(last.content),
    )
    closure_refs = selected_refs + ancestor_refs
    retry = sync_store(source, destination, ancestry, destination_ref="packets/ancestry")
    assert (retry.cloned, retry.reused) == (len(ancestor_refs) - 1, len(selected_refs) + 1)
    for ref in closure_refs:
        assert destination.get_object(ref) == source.get_object(ref)
        src, dst = source.object_path(ref).stat(), destination.object_path(ref).stat()
        assert src.st_ino != dst.st_ino and src.st_nlink == dst.st_nlink == 1
    for ref in (latest.ref, success_blob_ref(sibling.content)):
        with pytest.raises(FileNotFoundError):
            destination.verify_object(ref)

    assert destination.read_ref(INDEX_REF) is None
    imported = sync_store(destination, source, "packets/ancestry", destination_ref="imported")
    assert imported.closure == ancestry
    assert imported.cloned == 0 and imported.reused == len(closure_refs)
    reachability = measure_store(destination, now=datetime.now(UTC), keep_superseded_days=0)
    assert not reachability.gaps
    assert {root.name: root.closure.objects for root in reachability.roots} == {
        "ref:packets/ancestry": len(closure_refs),
        "ref:packets/selected": len(selected_refs),
    }

    report_path = destination.object_path(report_ref)
    report_bytes = source.get_object(report_ref)
    report_path.write_bytes(b"x" * len(report_bytes))
    with pytest.raises(ArtifactBlobIntegrityError):
        sync_store(source, destination, ancestry, destination_ref="packets/corrupt")
    assert destination.read_ref("packets/corrupt") is None
    report_path.write_bytes(report_bytes)

    sibling_ref = success_blob_ref(sibling.content)
    source.object_path(sibling_ref).write_bytes(b"x" * sibling_ref.size_bytes)
    whole_receipt = StoreClosureRef(roots=(ReceiptRoot(receipt=result.ref),))
    with pytest.raises(ArtifactBlobIntegrityError):
        sync_store(
            source,
            destination,
            whole_receipt,
            destination_ref="packets/selected",
            expected=selected,
        )
    assert destination.read_ref("packets/selected") == selected.model_dump(mode="json")
    with pytest.raises(FileNotFoundError):
        destination.verify_object(sibling_ref)

    slot = ProductRef(request_id=producer.id, role="products", select=SlotMember(slot="model"))
    unsupported = StoreClosureRef(roots=(ReceiptRoot(receipt=original.ref, product=slot),))
    with pytest.raises(ValueError, match="Product selector has no supported member"):
        sync_store(source, destination, unsupported, destination_ref="packets/unsupported")
    assert destination.read_ref("packets/unsupported") is None

    model_directory = tmp_path / "model"
    model_directory.mkdir()
    (model_directory / "model").write_bytes(b"model tree member")
    tree = source.ingest_tree(model_directory)
    direct_tree = StoreClosureRef(roots=(tree,))
    result = sync_store(source, destination, direct_tree, destination_ref="packets/tree")
    assert (result.cloned, result.reused) == (2, 0)
    assert destination.get_tree(tree) == source.get_tree(tree)
    [member] = destination.get_tree(tree).files
    assert destination.get_object(member.object) == b"model tree member"
