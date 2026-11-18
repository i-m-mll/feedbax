from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path

import numpy as np
import pytest

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.computation_validity import StepCountChanged
from feedbax.contracts.code_validity import AttemptProvenance, CodeTrace, ReceiptEvidence, RegistryBinding, SourcePath, SourceSnapshot
from feedbax.contracts.receipt_validity import (
    CheckConditions,
    CheckOutcome,
    FileDifference,
    RegistryDifference,
    ValidityAssessment,
    ValidityCheckEvidence,
    ValidityJudgement,
)
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE
from feedbax.contracts.effective_success import Product
from feedbax.contracts.store_closure import read_store_closure_ref
from feedbax.dataset import Dataset, Variable
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import publish_dataset
from feedbax.persistence.effective_success import PublicationConflict, SuccessView, run_ref_name
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.persistence.receipt_validity import read_validity_records, record_validity
from feedbax.validity.admission import decide_validity, observe_differences
from feedbax.validity.environment import read_environment
from feedbax.validity.light_products import light_product_matches
from feedbax.validity.project_admission import is_quick
from feedbax.validity.source import SourceCatalog
from scripts.proofs.effective_binding_witness import prepare


def _observe(
    project: Path, store: ImmutableArtifactBlobProvider, trace: CodeTrace, environment: dict[str, str],
):
    return observe_differences(
        [trace], catalog=SourceCatalog({"project": project}), registry={},
        environment=environment, read_object=store.get_object,
    )[0]


def test_atomic_lock_replacement_changes_observed_environment_with_equal_size_and_mtime(tmp_path):
    path = tmp_path / "uv.lock"
    path.write_text('[[package]]\nname = "numpy"\nversion = "1.0"\n')
    observed = read_environment(tmp_path, installed={})
    original = path.stat()

    replacement = tmp_path / "next.lock"
    replacement.write_text('[[package]]\nname = "numpy"\nversion = "2.0"\n')
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(path)

    current = read_environment(tmp_path, installed={})
    assert observed.versions == {"numpy": "1.0"}
    assert current.versions == {"numpy": "2.0"}
    assert current.lock_sha256 != observed.lock_sha256


def test_environment_confirmation_never_accepts_code_and_joint_proof_expires(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    (tmp_path / "first.py").write_text("def used(): return 2\n")
    (tmp_path / "second.py").write_text("def used(): return 0\n")
    snapshots = tuple(SourceSnapshot(
        path=SourcePath(scope="project", path=name),
        content=store.store_object(b"def used(): return 1\n"), entered=("used",),
    ) for name in ("first.py", "second.py"))
    trace = CodeTrace(sources=snapshots, registry=(), files=(), environment={"jax": "0.7.2"}, indeterminate=())
    receipt = store.store_object(b"the successful receipt")
    conditions = _observe(tmp_path, store, trace, {"jax": "0.8.3"})
    before = decide_validity(receipt, conditions=conditions, code_checkable=True, environment_checkable=True)
    assert before.state == "unchecked"
    assert set(before.checks) == {"code", "environment"}

    confirmation = ValidityAssessment(
        receipt=receipt, conditions=conditions, assessed_at=datetime.now(UTC),
        evidence=ValidityCheckEvidence(environment=CheckOutcome(outcome="equal")),
    )
    env_only = decide_validity(receipt, conditions=conditions, records=(confirmation,), code_checkable=True)
    assert env_only.state == "unchecked" and set(env_only.checks) == {"code"}
    assert env_only.environment.check == "confirmed"

    steps = (StepCountChanged(previous=2, present=6),)
    for code, expected in (
        (CheckOutcome(outcome="different", changes=steps), ("code", "computation", steps, None)),
        (CheckOutcome(outcome="unavailable", detail="No replay start."), ("code", "source", (), "No replay start.")),
    ):
        assessed = confirmation.model_copy(update={
            "evidence": ValidityCheckEvidence(code=code, environment=CheckOutcome(outcome="equal")),
        })
        result = decide_validity(receipt, conditions=conditions, records=(assessed,), code_checkable=True)
        assert result.state == "stale" and not result.checks
        assert result.environment.check == "confirmed"
        assert {(reason.kind, reason.basis, reason.changes, reason.detail) for reason in result.reasons} == {expected}

    both = confirmation.model_copy(update={
        "evidence": ValidityCheckEvidence(
            code=CheckOutcome(outcome="equal"), environment=CheckOutcome(outcome="equal"),
        ),
    })
    assert decide_validity(receipt, conditions=conditions, records=(both,)).state == "current"
    (tmp_path / "second.py").write_text("def used(): return 1\n")
    changed = _observe(tmp_path, store, trace, {"jax": "0.8.3"})
    stale = decide_validity(receipt, conditions=changed, records=(both,))
    assert stale.state == "stale"
    assert stale.reasons[0].differences[0].path.path == "first.py"
    assert stale.environment.check == "not_checked"


def test_judgement_names_the_exact_registry_binding_and_driver_edit(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    before = RegistryBinding(
        family="operation", requested_key="example.analysis", key="example.analysis",
        distribution="example", version="1", module="example.analysis", qualified_name="exact",
    )
    after = before.model_copy(update={"qualified_name": "revised"})
    driver = SourcePath(scope="project", path="driver.py")
    (tmp_path / "driver.py").write_text("def drive(): return 2\n")
    archive = SourceSnapshot(
        path=driver, content=store.store_object(b"def drive(): return 1\n"), entered=("drive",),
    )
    trace = CodeTrace(sources=(archive,), registry=(before,), files=(), environment={}, indeterminate=())
    conditions, _ = observe_differences(
        (trace,), catalog=SourceCatalog({"project": tmp_path}),
        registry={(after.family, after.requested_key): after}, environment={}, read_object=store.get_object,
    )
    binding = RegistryDifference(before=before, after=after)
    assert binding in conditions.code
    receipt = store.store_object(b"receipt")
    accept_registry = ValidityJudgement(
        receipt=receipt, differences=(binding,), who="Scientist", when=datetime.now(UTC), why="Reviewed registry.",
    )
    outstanding = decide_validity(receipt, conditions=conditions, records=(accept_registry,))
    assert outstanding.state == "stale"
    remaining = {change for reason in outstanding.reasons for change in reason.differences}
    assert {change.path for change in remaining} == {driver}
    accept_driver = accept_registry.model_copy(update={"differences": tuple(remaining), "why": "Reviewed driver."})
    assert decide_validity(receipt, conditions=conditions, records=(accept_registry, accept_driver)).state == "current"
    newer = RegistryDifference(before=before, after=after.model_copy(update={"qualified_name": "newer"}))
    edited = CheckConditions(code=(*remaining, newer), environment=())
    rechecked = decide_validity(receipt, conditions=edited, records=(accept_registry, accept_driver))
    assert {change for reason in rechecked.reasons for change in reason.differences} == {newer}


def test_judgement_records_survive_republication_and_are_bound_to_latest_success(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    case = prepare(store, b'{"kind":"test","id":"producer"}', b'{"kind":"test","id":"analysis"}')
    success = case.original
    snapshots = []
    for name in ("first", "second"):
        path = tmp_path / f"{name}.py"
        path.write_text("def used(): return 2\n")
        snapshots.append(SourceSnapshot(
            path=SourcePath(scope="project", path=path.name),
            content=store.store_object(b"def used(): return 1\n"),
            entered=("used",),
        ))
    trace = CodeTrace(sources=tuple(snapshots), registry=(), files=(), environment={}, indeterminate=())
    conditions = _observe(tmp_path, store, trace, {})
    records = tuple(ValidityJudgement(
        receipt=success.ref, differences=(difference,), who="Scientist",
        when=datetime.now(UTC), why="Reviewed fixed inputs",
    ) for difference in conditions.code)
    with ThreadPoolExecutor(max_workers=2) as pool:
        stored = tuple(pool.map(lambda record: record_validity(store, case.rule.id, record), records))
    SuccessView(store).ensure_published_run_ref(success.ref, success.receipt)
    assert {record.differences for record in read_validity_records(store, case.rule.id, success.ref)} == {
        record.differences for record in records
    }
    closure = read_store_closure_ref(store.read_ref(run_ref_name(case.rule.id)))
    assert {ref.sha256 for ref in stored} <= {ref.sha256 for ref in ReceiptClosure(store).walk(closure)}
    assert decide_validity(success.ref, conditions=conditions, records=records).state == "current"
    file_change = FileDifference(path=snapshots[0].path, before="0" * 64, after="1" * 64)
    still_stale = decide_validity(
        success.ref, conditions=conditions, records=records, file_differences=(file_change,),
    )
    assert still_stale.reasons == (file_change,)

    after = SuccessView(store).publish(
        case.rule, attempt="second-success", products=success.receipt.products,
    )
    assert read_validity_records(store, case.rule.id, after.ref) == ()
    assert SuccessView(store).read_success(success.ref) == success
    with pytest.raises(PublicationConflict):
        record_validity(store, case.rule.id, records[0])


def test_a_missing_original_observation_cannot_appear_current(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    source = SourcePath(scope="project", path="interrupted-import.py")
    data = SourcePath(scope="project", path="interrupted-read.csv")
    trace = CodeTrace(
        sources=(), registry=(), files=(), environment={}, indeterminate=(source,), indeterminate_files=(data,),
    )
    conditions, files = observe_differences(
        (trace,), catalog=SourceCatalog({"project": tmp_path}), registry={},
        environment={}, read_object=store.get_object,
    )
    result = decide_validity(store.store_object(b"receipt"), conditions=conditions, file_differences=files)
    assert result.state == "stale"
    reason_by_kind = {reason.kind: reason for reason in result.reasons}
    assert reason_by_kind["file"].path == data
    assert reason_by_kind["file"].status == "indeterminate"
    assert {difference.path for difference in reason_by_kind["code"].differences} == {source}
    assert not result.checks


def test_quick_replay_tolerates_float_rounding_but_preserves_labels_and_zero(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path / "store")

    def product(values, *, labels=("early", "late")):
        dataset = Dataset.from_arrays(
            {"loss": Variable(np.array(values, dtype=np.float32), ("time",))},
            coords={"time": Variable(np.array(labels), ("time",))},
        )
        ref = publish_dataset(store, dataset)
        content = ref.content.model_copy(update={"media_type": DATASET_MEDIA_TYPE})
        return Product(name="metrics", content=content, descriptor={
            "descriptor_sha256": ref.descriptor_sha256,
        })

    retained = product((0, 1))
    last_bit = product((0, np.nextafter(np.float32(1), np.float32(2))))
    assert retained.content.sha256 != last_bit.content.sha256
    assert light_product_matches(store, retained, last_bit)
    assert not light_product_matches(store, retained, product((1e-12, 1)))
    assert not light_product_matches(store, retained, product((0, 1.00001)))
    assert not light_product_matches(store, retained, product((0, 1), labels=("late", "early")))


def test_an_expensive_prefix_cannot_make_a_short_resume_eligible_for_automatic_run(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    case = prepare(store, b'{"kind":"test","id":"producer"}', b'{"kind":"test","id":"analysis"}')
    started = datetime(2026, 1, 1, tzinfo=UTC)
    trace = CodeTrace(sources=(), registry=(), files=(), environment={}, indeterminate=())
    suffix = AttemptProvenance(
        feedbax_commit="witness", feedbax_dirty=False, lock_sha256=None, platform="cpu",
        started_at=started + timedelta(seconds=4), ended_at=started + timedelta(seconds=5),
    )
    before = ReceiptEvidence(
        attempt="earlier", trace=trace, provenance=suffix.model_copy(update={
            "started_at": started, "ended_at": started + timedelta(seconds=3),
        }),
    )
    initial = case.original.receipt.model_copy(update={"trace": trace, "provenance": suffix})
    assert is_quick(initial, timedelta(seconds=2), store)
    prefix = store.store_object(canonical_json_v2_bytes(before.model_dump(mode="json")))
    resumed = initial.model_copy(update={"trace": trace.model_copy(update={"prefixes": (prefix,)})})
    assert not is_quick(resumed, timedelta(seconds=2), store)
    assert is_quick(resumed, timedelta(seconds=4), store)
