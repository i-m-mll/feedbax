from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from feedbax.contracts.code_validity import CodeTrace, SourcePath, SourceSnapshot
from feedbax.contracts.receipt_validity import (
    CheckOutcome,
    FileDifference,
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
from feedbax.validity.light_products import light_product_matches
from feedbax.validity.source import SourceCatalog
from scripts.proofs.effective_binding_witness import prepare


def _observe(
    project: Path, store: ImmutableArtifactBlobProvider, trace: CodeTrace, environment: dict[str, str],
):
    return observe_differences(
        [trace], catalog=SourceCatalog({"project": project}), registry={},
        environment=environment, read_object=store.get_object,
    )[0]


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
    assert before.checks == ("code", "environment")

    confirmation = ValidityAssessment(
        receipt=receipt, conditions=conditions, assessed_at=datetime.now(UTC),
        evidence=ValidityCheckEvidence(environment=CheckOutcome(outcome="equal")),
    )
    env_only = decide_validity(receipt, conditions=conditions, records=(confirmation,), code_checkable=True)
    assert env_only.state == "unchecked" and env_only.checks == ("code",)
    assert env_only.environment.check == "confirmed"

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
    assert result.reasons[0].path == data
    assert result.reasons[0].status == "indeterminate"
    assert result.reasons[1].differences[0].path == source
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
