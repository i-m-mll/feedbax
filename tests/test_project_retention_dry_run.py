"""Qualify read-only storage accounting over real refs, trees, and a saved checkpoint."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from threading import Event

import pytest

from feedbax.cli import main
from feedbax.contracts.artifact_custody import ContentRootRecord
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host import HostClient
from feedbax.host.plan_history import advance_plan, plan_ref_key, plan_run_key
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.reachability import RootObservation, Size, measure_store, object_target
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput, CanonicalTrainRequestBinding
from feedbax.training.checkpoint_custody import write_checkpoint_transaction
from tests.test_checkpoint_custody import _coordinate, _minimax_slots, _run_spec


def _files(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def _store_age(store, ref, date):
    os.utime(store.object_path(ref), (date.timestamp(), date.timestamp()))


def test_ref_closures_share_bytes_and_use_both_retention_boundaries(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path / "results")
    now = datetime(2026, 11, 12, tzinfo=UTC)
    for name in ("a", "b"):
        source = tmp_path / name
        source.mkdir()
        (source / "shared").write_bytes(b"shared member")
        (source / "own").write_bytes(name.encode())
        tree = store.ingest_tree(source)
        assert store.compare_and_swap_ref(
            f"pins/{name}",
            expected=None,
            replacement=tree.model_dump(mode="json"),
        )
    old = store.store_object(b"orphan")
    recent = store.store_object(b"recent supersession")
    expired = store.store_object(b"expired supersession")
    fresh = store.store_object(b"young object")
    compatibility = store.store_object(b"legacy root")
    store.register_root(
        ContentRootRecord(
            root_id="legacy-import",
            root_kind="import",
            targets=(object_target(compatibility),),
        )
    )
    for ref in (old, recent, expired, compatibility):
        _store_age(store, ref, now - timedelta(days=1))
    _store_age(store, fresh, now - timedelta(hours=23))
    observations = (
        RootObservation("receipt:recent", (object_target(recent),), now - timedelta(days=13)),
        RootObservation("receipt:expired", (object_target(expired),), now - timedelta(days=14)),
    )
    before = _files(store.root)
    report = measure_store(store, now=now, keep_superseded_days=14, observations=observations)
    assert report.gaps == ()
    assert report.shared == Size(1, len(b"shared member"))
    assert report.fresh == Size(1, fresh.size_bytes)
    assert report.eligible_digests == tuple(sorted((old.sha256, expired.sha256)))
    assert report.retained.objects == 7
    for pin in (root for root in report.roots if root.name.startswith("ref:pins/")):
        assert pin.closure.objects == 3 and pin.unshared.objects == 2
        assert pin.shared == report.shared
    no_history = measure_store(store, now=now, keep_superseded_days=0, observations=observations)
    assert set(no_history.eligible_digests or ()) == {old.sha256, expired.sha256, recent.sha256}
    assert _files(store.root) == before

    assert store.compare_and_swap_ref("runs/unknown", expected=None, replacement={"opaque": "data"})
    incomplete = measure_store(store, now=now, keep_superseded_days=14, observations=observations)
    assert incomplete.retained == report.retained and incomplete.fresh == report.fresh
    assert incomplete.eligible is None and incomplete.unclassified == report.eligible
    assert all(root.unshared is None for root in incomplete.roots)
    assert any("runs/unknown" in gap for gap in incomplete.gaps)


def test_project_report_uses_scoped_plan_and_exact_retained_checkpoint(
    tmp_path, monkeypatch, capsys
):
    root = (tmp_path / "saved-folder").resolve()
    spec = _run_spec(minimax=True)
    request = TrainingRequestSpec(
        graph=spec.graph.inline,
        task=spec.task,
        objective=spec.objective,
        method_ref=spec.method_ref,
        method_payload=spec.method_payload,
        training_config=spec.training_config,
    )
    project = Project.model_validate(
        {
            "steps": {
                "fit": {"kind": "train", "base": {"inline": request.model_dump(mode="json")}}
            },
        }
    )
    for relative, raw in dump_text_project(
        project,
        ProjectConfiguration(
            project="renamed", store="results", settings={"keep_superseded": "none"}
        ),
    ).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    store = ImmutableArtifactBlobProvider(root / "results")
    encoded = encode_request(request, {})
    refs = CanonicalTrainRequestBinding(
        request=store.store_object(encoded.data),
        documents=tuple(store.store_object(data) for data in encoded.documents.values()),
    )
    coordinate = _coordinate()
    published = write_checkpoint_transaction(
        store.root / "checkpoints" / coordinate.run_id,
        run_spec=spec,
        phase_program=spec.worker_execution.method_contract.phase_program,
        barrier_name=coordinate.completed_barrier,
        coordinate=coordinate,
        slots=_minimax_slots(),
        completed_training_batches=1,
    )
    checkpoint = PublishedTrainingCheckpoint(
        checkpoint_root=published.root,
        pointer=published.latest_pointer,
        completed_updates=1,
    )
    requests = {plan_run_key("default", "fit", "default"): encoded.id}
    plan = store.store_object(canonical_json_v2_bytes({"requests": requests}))
    key = plan_ref_key(root.name, str(store.root))
    job = {
        "id": coordinate.run_id,
        "project": root.name,
        "request": encoded.id,
        "state": "interrupted",
        "data": {"manifest_root": str(store.root)},
        "edges": [],
        "checkpoint": checkpoint.model_dump(mode="json"),
        "launch": {
            "payload": BoundStudioTrainingJobInput(
                spec=spec,
                project_id=root.name,
                request_id=encoded.id,
                manifest_root=store.root,
                canonical_request=refs,
            ).model_dump(mode="json")
        },
    }

    def snapshot(self, *, project, start_host, include_launch, record_keys):
        assert (project, start_host, include_launch, record_keys) == (
            root.name,
            False,
            True,
            (key,),
        )
        return {
            "cursor": 41,
            "jobs": [
                job,
                job | {"id": "another-store", "data": {"manifest_root": "/another"}, "launch": {}},
            ],
            "records": {key: advance_plan(plan.model_dump(mode="json"), requests, None, {})},
        }

    monkeypatch.setattr(HostClient, "jobs_snapshot", snapshot)
    before = _files(root)
    monkeypatch.chdir(root / "steps")
    assert main(["store", "gc", "--dry-run"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] and report["host_cursor"] == 41
    assert report["project"] == root.name and report["store"] == str(store.root)
    assert report["objects"]["keep_superseded_days"] == 0
    assert report["objects"]["eligible"] is None
    roots = {row["name"]: row for row in report["objects"]["roots"]}
    assert roots["host:admitted-train-plan"]["closure"]["bytes"] == plan.size_bytes
    assert roots[f"job:{coordinate.run_id}"]["closure"]["objects"] == 1 + len(refs.documents)
    assert "job:another-store" not in roots
    checkpoint_sizes = report["checkpoint_files"]
    assert checkpoint_sizes["objects"] == 2 + len(published.manifest.slots)
    assert checkpoint_sizes["bytes"] == sum(
        path.stat().st_size for path in published.transaction_dir.rglob("*") if path.is_file()
    )
    assert any("latest checkpoint only" in gap for gap in report["objects"]["gaps"])
    assert _files(root) == before


def test_dry_run_leaves_an_uninitialized_store_absent(tmp_path):
    path = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        measure_store(
            ImmutableArtifactBlobProvider(path),
            now=datetime.now(UTC),
            keep_superseded_days=14,
        )
    assert not path.exists()


def test_dry_run_protects_a_fresh_object_while_its_ref_is_being_published(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path / "results")
    object_ready = Event()
    publish_ref = Event()

    def publish():
        ref = store.store_object(b"pending publication")
        object_ready.set()
        assert publish_ref.wait(5)
        assert store.compare_and_swap_ref(
            "pins/new", expected=None, replacement=ref.model_dump(mode="json")
        )
        return ref

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(publish)
        try:
            assert object_ready.wait(5)
            pending = measure_store(store, now=datetime.now(UTC), keep_superseded_days=14)
            assert pending.fresh == Size(1, len(b"pending publication"))
            assert pending.eligible == Size(0, 0)
        finally:
            publish_ref.set()
        ref = future.result(timeout=5)
    settled = measure_store(store, now=datetime.now(UTC), keep_superseded_days=14)
    assert settled.retained == Size(1, ref.size_bytes)
    assert settled.fresh == Size(0, 0) and settled.eligible == Size(0, 0)
