from pathlib import Path
from uuid import uuid4

import pytest

from feedbax.contracts.artifact_custody import ArtifactBlobIntegrityError
from feedbax.contracts.effective_success import INDEX_REF, success_content_ref
from feedbax.contracts.execution_target import LOCAL_TARGET, SessionTarget, TargetRealization
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_specs import ReportRequestSpec
from feedbax.contracts.store_closure import StoreClosureRef
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider, content_object_relative_path
from feedbax.persistence.effective_success import SuccessView, store_request
from feedbax.persistence.store_sync import collect_success, sync_store
from scripts.proofs.effective_binding_witness import document_product
from tests.support.store_transport import loopback_store


def test_rsync_retry_sends_only_missing_objects_and_publishes_after_verification(tmp_path):
    source = ImmutableArtifactBlobProvider(tmp_path / "source with spaces")
    venue = ImmutableArtifactBlobProvider(tmp_path / "venue with spaces")
    connection = loopback_store(venue.root, tmp_path / "connection")
    payload = b"large immutable object" * 400_000
    large = source.store_object(payload)
    early_payload = b"small immutable object 20"
    early = source.store_object(early_payload)
    assert early.sha256 < large.sha256
    already = venue.store_object(b"already present")
    assert source.store_object(b"already present") == already
    closure = StoreClosureRef(roots=(large, early, already))
    name = "transfer/witness"

    connection.disconnect.write_text("524288")
    with pytest.raises(OSError, match="Store transport rsync exited"):
        sync_store(source, connection.endpoint, closure, destination_ref=name)
    assert venue.read_ref(name) is None
    assert not venue.object_path(early).exists()
    assert not venue.object_path(large).exists()
    staging = Path(venue.root) / ".incoming"
    assert (staging / content_object_relative_path(early.sha256)).read_bytes() == early_payload
    assert not (staging / content_object_relative_path(large.sha256)).exists()
    assert not (staging / content_object_relative_path(already.sha256)).exists()

    result = sync_store(source, connection.endpoint, closure, destination_ref=name)
    assert (result.cloned, result.reused) == (1, 2)
    assert venue.read_ref(name) == closure.model_dump(mode="json")
    assert venue.get_object(large) == payload and venue.get_object(early) == early_payload
    assert not (venue.object_path(large).stat().st_mode & 0o222)
    assert sync_store(source, connection.endpoint, closure, destination_ref=name).cloned == 0
    assert len(connection.rsync_calls()) == 2

    pulled = ImmutableArtifactBlobProvider(tmp_path / "pulled")
    assert pulled.store_object(early_payload) == early
    assert sync_store(connection.endpoint, pulled, name, destination_ref=name).cloned == 2
    assert pulled.get_object(large) == payload
    assert pulled.read_ref(name) == closure.model_dump(mode="json")


def test_ssh_collection_verifies_the_bytes_before_admitting_success(tmp_path):
    venue = ImmutableArtifactBlobProvider(tmp_path / "venue")
    connection = loopback_store(venue.root, tmp_path / "connection")
    local = ImmutableArtifactBlobProvider(tmp_path / "local")
    request = store_request(venue, encode_request(ReportRequestSpec(report_type="proof.ssh"), {}))
    product = document_product(venue, "report", b'{"kind":"report"}')
    original = SuccessView(venue).publish(request, attempt="first", products={"reports": (product,)})
    target = SessionTarget(
        kind="session", profile="fixture-gpu", driver="runpod", session_id=uuid4(),
        realization=TargetRealization(environment="a" * 64, placement={"device": "fixture"}),
    )
    newer = SuccessView(venue).publish_bound(
        request, attempt="second", products={"reports": (product,)}, bindings={}, target=target,
    )
    assert newer.receipt.effective == original.receipt.effective
    ref = success_content_ref(product.content)
    staged = Path(local.root) / ".incoming" / content_object_relative_path(ref.sha256)
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"x" * ref.size_bytes)

    with pytest.raises(ArtifactBlobIntegrityError, match="SHA-256 mismatch"):
        collect_success(connection.endpoint, local, newer.ref, attempt="second", request_id=request.id)
    assert local.read_ref(INDEX_REF) is None
    assert not local.object_path(ref).exists()
    assert not staged.exists()
    assert not connection.rsync_calls()

    with pytest.raises(OSError, match="expected execution target"):
        collect_success(connection.endpoint, local, newer.ref, attempt="second", request_id=request.id,
                        expected_target=LOCAL_TARGET)
    assert not connection.rsync_calls()

    def reject_observation(success):
        assert success == newer.receipt
        raise ValueError("terminal observation differs")

    with pytest.raises(ValueError, match="terminal observation differs"):
        collect_success(connection.endpoint, local, newer.ref, attempt="second", request_id=request.id,
                        expected_target=target, verify=reject_observation)
    assert local.read_ref(INDEX_REF) is None

    assert collect_success(
        connection.endpoint, local, newer.ref, attempt="second", request_id=request.id, expected_target=target,
    ) == newer
    assert SuccessView(local).read_attempt("first") == original
    assert SuccessView(local).authored(request.id) == newer
    assert collect_success(connection.endpoint, local, newer.ref, attempt="second", request_id=request.id) == newer
    assert len(connection.rsync_calls()) == 1
    assert local.get_object(newer.ref) == venue.get_object(newer.ref)
