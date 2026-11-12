from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json
from multiprocessing import get_context
import os
from pathlib import Path

import pytest

from feedbax.contracts.artifact_custody import ArtifactBlobIntegrityError, ContentObjectRef
from feedbax.contracts.effective_success import decode_success_index
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_refs import InputMember, LastMember, NamedMember, ProductRef
from feedbax.contracts.request_specs import ReportRequestSpec
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.reachability import measure_store
from scripts.proofs.effective_binding_store import (
    INDEX_REF,
    PublicationConflict,
    SuccessView,
    store_request,
)
from scripts.proofs.effective_binding_witness import document_product, prepare, summarize


SUPPLIER = b'{"kind":"EvaluationRunManifest","id":"evaluation-1"}'
ANALYSIS = b'{"kind":"AnalysisRunManifest","id":"analysis-1"}'


def test_frozen_and_rule_lookup_require_the_same_exact_ordered_binding(tmp_path, record_property):
    store = ImmutableArtifactBlobProvider(tmp_path)
    case = prepare(store, SUPPLIER, ANALYSIS)
    view = SuccessView(ImmutableArtifactBlobProvider(tmp_path))
    assert case.rule.id != case.frozen.id
    assert view.authored(case.frozen.id) is None
    assert (
        view.authored(case.rule.id)
        == view.lookup(case.rule)
        == view.lookup(case.frozen)
        == case.original
    )
    assert case.original.receipt.attempt == "synopsis-01"

    frozen_members = tuple(
        InputMember(
            key=key,
            reference=ProductRef(
                request_id=case.producer.id,
                role="manifests",
                select=NamedMember(member=name),
            ),
        )
        for key, name in (("rank-1", "analysis"), ("rank-2", "supplier"))
    )
    synopsis = ReportRequestSpec(report_type="proof.ordered-manifest-synopsis")
    reordered = store_request(store, encode_request(synopsis, {"documents": frozen_members[::-1]}))
    rekeyed = store_request(
        store,
        encode_request(
            synopsis,
            {
                "documents": (
                    frozen_members[0].model_copy(update={"key": "different-key"}),
                    frozen_members[1],
                )
            },
        ),
    )
    assert view.lookup(reordered) is None
    assert view.lookup(rekeyed) is None
    record_property("rule_id", case.rule.id)
    record_property("frozen_id", case.frozen.id)
    record_property("shared_effective", case.original.receipt.effective)
    record_property("reordered_effective", view.bind(reordered)[0])
    record_property("rekeyed_effective", view.bind(rekeyed)[0])

    producer = view.authored(case.producer.id)
    assert producer is not None
    supplier, analysis = producer.receipt.products["manifests"]
    changed_descriptor = analysis.model_copy(update={"descriptor": {"kind": "different"}})
    view.publish(
        case.producer,
        attempt="retain-02",
        products={"manifests": (supplier, changed_descriptor)},
    )
    descriptors = SuccessView(store)
    assert descriptors.lookup(case.rule) is None
    assert descriptors.lookup(case.frozen) is None
    record_property("changed_descriptor_effective", descriptors.bind(case.rule)[0])
    assert descriptors.read_success(case.original.ref) == case.original

    changed_content = document_product(
        store,
        "analysis",
        b'{"kind":"AnalysisRunManifest","id":"analysis-2"}',
    )
    descriptors.publish(
        case.producer,
        attempt="retain-03",
        products={"manifests": (supplier, changed_content)},
    )
    content = SuccessView(store)
    assert content.lookup(case.frozen) is None
    record_property("changed_content_effective", content.bind(case.frozen)[0])
    assert content.read_success(case.original.ref) == case.original

    target = store.object_path(
        ContentObjectRef(
            object_id=f"content-object://sha256/{changed_content.content.sha256}",
            sha256=changed_content.content.sha256,
            size_bytes=changed_content.content.size,
        )
    )
    target.write_bytes(b"x" * changed_content.content.size)
    with pytest.raises(ArtifactBlobIntegrityError):
        content.lookup(case.rule)


def _interrupt_before_index(root: Path, marker: Path, request, products) -> None:
    class InterruptedStore(ImmutableArtifactBlobProvider):
        def compare_and_swap_ref(self, name, *, expected, replacement):
            assert name == INDEX_REF
            marker.write_text(json.dumps(replacement["attempts"]["interrupted"]))
            os._exit(73)

    SuccessView(InterruptedStore(root)).publish(request, attempt="interrupted", products=products)


def test_interruption_and_competing_publication_have_one_commit_point(tmp_path, record_property):
    root = tmp_path / "store"
    store = ImmutableArtifactBlobProvider(root)
    case = prepare(store, SUPPLIER, ANALYSIS)
    original_bytes = store.get_object(case.original.ref)
    original_request = store.get_object(case.original.receipt.request.object)
    before = store.read_ref(INDEX_REF)
    assert before is not None and "supersession_times" not in before
    marker = tmp_path / "interrupted-receipt.json"
    products = case.original.receipt.products
    process = get_context("spawn").Process(
        target=_interrupt_before_index,
        args=(root, marker, case.rule, products),
    )
    process.start()
    try:
        process.join(15)
        assert process.exitcode == 73
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
    staged = ContentObjectRef.model_validate_json(marker.read_bytes())
    reopened = SuccessView(ImmutableArtifactBlobProvider(root))
    assert reopened.store.get_object(staged)
    assert reopened.store.read_ref(INDEX_REF) == before
    assert reopened.authored(case.rule.id) == reopened.lookup(case.frozen) == case.original
    with pytest.raises(ValueError, match="committed Store generation"):
        reopened.read_success(staged)

    def publish(snapshot: SuccessView, attempt: str):
        try:
            return snapshot.publish(case.rule, attempt=attempt, products=products)
        except PublicationConflict:
            return None

    other = SuccessView(ImmutableArtifactBlobProvider(root))
    race_started = datetime.now(UTC)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(publish, reopened, "candidate-01")
        second = workers.submit(publish, other, "candidate-02")
        winners = [result for result in (first.result(), second.result()) if result is not None]
    [winner] = winners
    latest = SuccessView(ImmutableArtifactBlobProvider(root))
    assert latest.authored(case.rule.id) == latest.lookup(case.frozen) == winner
    assert winner.receipt.supersedes == case.original.ref
    assert latest.read_success(case.original.ref) == case.original
    assert reopened.lookup(case.frozen) == case.original
    with pytest.raises(ValueError, match="attempt already published"):
        latest.publish(case.rule, attempt=case.original.receipt.attempt, products=products)
    record_property("interruption_exit", process.exitcode)
    record_property("uncommitted_receipt", staged.sha256)
    record_property("prior_receipt", case.original.ref.sha256)
    record_property("winning_receipt", winner.ref.sha256)
    record_property("winning_attempt", winner.receipt.attempt)

    value = latest.store.read_ref(INDEX_REF)
    index = decode_success_index(value)
    assert set(index.supersession_times) == {winner.ref.sha256}
    when = index.supersession_times[winner.ref.sha256]
    assert race_started <= when <= datetime.now(UTC)
    assert store.get_object(case.original.ref) == original_bytes
    assert store.get_object(case.original.receipt.request.object) == original_request
    assert "supersession_times" not in json.loads(store.get_object(winner.ref))

    cutoff = when + timedelta(days=14)
    before_cutoff = measure_store(
        store,
        now=cutoff - timedelta(microseconds=1),
        keep_superseded_days=14,
    )
    after_cutoff = measure_store(store, now=cutoff, keep_superseded_days=14)
    assert not before_cutoff.gaps and not after_cutoff.gaps
    history = f"superseded-success:{case.original.receipt.attempt}"
    assert history in {root.name for root in before_cutoff.roots}
    assert history not in {root.name for root in after_cutoff.roots}
    assert f"success:{winner.receipt.attempt}" in {root.name for root in after_cutoff.roots}
    assert case.original.ref.sha256 not in (after_cutoff.eligible_digests or ())

    assert value is not None
    legacy = {name: item for name, item in value.items() if name != "supersession_times"}
    assert store.compare_and_swap_ref(INDEX_REF, expected=value, replacement=legacy)
    assert SuccessView(store).read_success(case.original.ref) == case.original
    undated = measure_store(store, now=cutoff, keep_superseded_days=14)
    assert history in {root.name for root in undated.roots}
    assert undated.eligible is None
    assert any("authoritative supersession time unavailable" in gap for gap in undated.gaps)


def test_corrupt_receipt_bytes_never_admit_an_index_hit(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path)
    case = prepare(store, SUPPLIER, ANALYSIS)
    store.object_path(case.original.ref).write_bytes(b"x" * case.original.ref.size_bytes)
    with pytest.raises(ArtifactBlobIntegrityError):
        SuccessView(store).lookup(case.frozen)


def test_digest_frozen_input_keeps_its_result_when_the_rule_resolves_to_new_bytes(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path)
    original = document_product(store, "analysis", ANALYSIS).model_copy(update={"descriptor": None})
    producer = store_request(
        store, encode_request(ReportRequestSpec(report_type="proof.source"), {})
    )
    SuccessView(store).publish(producer, attempt="producer-01", products={"manifests": (original,)})
    spec = ReportRequestSpec(report_type="proof.synopsis")
    rule = store_request(
        store,
        encode_request(
            spec,
            {
                "documents": (
                    InputMember(
                        key="analysis",
                        reference=ProductRef(
                            request_id=producer.id, role="manifests", select=LastMember()
                        ),
                    ),
                )
            },
        ),
    )
    frozen = store_request(
        store,
        encode_request(
            spec, {"documents": (InputMember(key="analysis", reference=original.content),)}
        ),
    )
    before = SuccessView(store)
    first = before.publish(
        rule, attempt="synopsis-01", products={"report": (summarize(before, rule),)}
    )
    assert SuccessView(store).lookup(frozen) == first

    changed = document_product(
        store,
        "analysis",
        b'{"kind":"AnalysisRunManifest","id":"analysis-2"}',
    ).model_copy(update={"descriptor": None})
    SuccessView(store).publish(producer, attempt="producer-02", products={"manifests": (changed,)})
    after = SuccessView(store)
    assert after.lookup(rule) is None
    assert after.lookup(frozen) == first
