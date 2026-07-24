"""Side-effect-free rehearsal probe isolation over immutable custody inputs.

These tests gate the incident where a "run one real update locally" probe was
executed through the training executor against a materialized checkpoint fork
target and advanced the target's ``latest.json`` from one transaction to the
next, so custody archiving of the (now stale) source refused. The gate is
isolation: a bounded probe resumes from a read-only source and writes every new
transaction plus its ``latest.json`` pointer into a disposable output namespace,
guarded by a mutation tripwire and cleaned up on success and failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.training import (
    CheckpointCustodyError,
    CheckpointCustodyMutationError,
    IsolatedCheckpointProbe,
    assert_checkpoint_custody_unchanged,
    authenticate_published_checkpoint_custody,
    execute_training_run_spec,
    fingerprint_checkpoint_custody_inputs,
    isolated_checkpoint_probe,
    produce_checkpoint_custody_archive,
)
from tests.test_checkpoint_custody import (
    _resolver_parent_ref,
    _slot_blob_path,
    _write_resolver_checkpoint,
)
from tests.test_training_run_executor import (
    _mapped_diagnostics_run_spec,
    _mapped_scalar_kernel,
    _mapped_test_registry,
    _valid_materialized_preparation,
)


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    """Return the exact bytes of every regular file under ``root``."""
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


# --- fingerprint + tripwire -------------------------------------------------


def test_assert_unchanged_passes_for_untouched_custody(tmp_path: Path) -> None:
    result = _write_resolver_checkpoint(tmp_path / "source")
    before = fingerprint_checkpoint_custody_inputs(result.root)
    assert assert_checkpoint_custody_unchanged(result.root, before) == before


def test_tripwire_detects_latest_pointer_advance(tmp_path: Path) -> None:
    result = _write_resolver_checkpoint(tmp_path / "source")
    before = fingerprint_checkpoint_custody_inputs(result.root)
    # Reproduce the incident's mutation: a second transaction republishes
    # latest.json in place, moving the pointer off the fingerprinted state.
    _write_resolver_checkpoint(result.root)
    with pytest.raises(CheckpointCustodyMutationError):
        assert_checkpoint_custody_unchanged(result.root, before)


def test_tripwire_detects_in_place_metadata_edit(tmp_path: Path) -> None:
    result = _write_resolver_checkpoint(tmp_path / "source")
    before = fingerprint_checkpoint_custody_inputs(result.root)
    latest = result.root / "latest.json"
    latest.write_bytes(latest.read_bytes() + b" ")
    with pytest.raises(CheckpointCustodyMutationError):
        assert_checkpoint_custody_unchanged(result.root, before)


def test_fingerprint_detects_added_and_removed_paths(tmp_path: Path) -> None:
    result = _write_resolver_checkpoint(tmp_path / "source")
    base = fingerprint_checkpoint_custody_inputs(result.root)
    extra = result.root / "unexpected.bin"
    extra.write_bytes(b"x")
    assert fingerprint_checkpoint_custody_inputs(result.root) != base
    extra.unlink()
    assert fingerprint_checkpoint_custody_inputs(result.root) == base


def test_metadata_default_ignores_but_digest_blobs_catches_blob_edit(
    tmp_path: Path,
) -> None:
    result = _write_resolver_checkpoint(tmp_path / "source")
    slot_name = result.manifest.slots[0].slot
    blob_path = _slot_blob_path(result.manifest_path, slot_name)
    metadata_before = fingerprint_checkpoint_custody_inputs(result.root)
    blobs_before = fingerprint_checkpoint_custody_inputs(result.root, digest_blobs=True)
    original = blob_path.read_bytes()
    # Same path, different content: the default (structure + metadata digests)
    # does not re-hash blobs, but digest_blobs=True does.
    blob_path.write_bytes(original + b"\0")
    assert fingerprint_checkpoint_custody_inputs(result.root) == metadata_before
    assert fingerprint_checkpoint_custody_inputs(result.root, digest_blobs=True) != blobs_before
    blob_path.write_bytes(original)


def test_fingerprint_reports_absent_root(tmp_path: Path) -> None:
    fingerprint = fingerprint_checkpoint_custody_inputs(tmp_path / "missing")
    assert fingerprint.root_present is False
    assert fingerprint.paths == ()


# --- isolated_checkpoint_probe context manager ------------------------------


def test_probe_write_leaves_source_bytes_identical(tmp_path: Path) -> None:
    result = _write_resolver_checkpoint(tmp_path / "source")
    source_root = result.root
    snapshot = _snapshot_tree(source_root)
    with isolated_checkpoint_probe(source_root) as probe:
        assert isinstance(probe, IsolatedCheckpointProbe)
        assert probe.output_root != source_root
        # Simulate the executor's bounded probe: a fresh transaction and its
        # latest.json land entirely in the disposable output namespace.
        _write_resolver_checkpoint(probe.output_root)
        assert (probe.output_root / "latest.json").is_file()
    assert _snapshot_tree(source_root) == snapshot
    assert not probe.output_root.exists()


def test_source_archiving_succeeds_after_probe(tmp_path: Path) -> None:
    result = _write_resolver_checkpoint(tmp_path / "source")
    ref = _resolver_parent_ref(result)
    provider = ImmutableArtifactBlobProvider(tmp_path / "artifacts")
    with isolated_checkpoint_probe(result.root) as probe:
        _write_resolver_checkpoint(probe.output_root)
    # The incident's failure mode is now prevented: the source still selects the
    # archived transaction, so custody archiving accepts it.
    archived = produce_checkpoint_custody_archive(
        ref, allowed_root=result.root, artifact_provider=provider
    )
    assert archived.evidence.parent_ref.id == result.manifest.transaction_id


def test_probe_tripwire_fires_when_body_mutates_source(tmp_path: Path) -> None:
    result = _write_resolver_checkpoint(tmp_path / "source")
    source_root = result.root
    with pytest.raises(CheckpointCustodyMutationError):
        with isolated_checkpoint_probe(source_root) as probe:
            assert probe.output_root.exists()
            # Regression: a probe that wrongly writes into the source.
            _write_resolver_checkpoint(source_root)
    assert not probe.output_root.exists()


def test_probe_cleans_up_output_namespace_on_exception(tmp_path: Path) -> None:
    result = _write_resolver_checkpoint(tmp_path / "source")
    source_root = result.root
    snapshot = _snapshot_tree(source_root)
    captured: dict[str, Path] = {}
    with pytest.raises(RuntimeError, match="probe boom"):
        with isolated_checkpoint_probe(source_root) as probe:
            captured["output_root"] = probe.output_root
            _write_resolver_checkpoint(probe.output_root)
            raise RuntimeError("probe boom")
    assert not captured["output_root"].exists()
    assert _snapshot_tree(source_root) == snapshot


def test_probe_requires_published_source(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CheckpointCustodyError):
        with isolated_checkpoint_probe(empty):
            pass


# --- executor resume probe (the real incident surface) ----------------------


def test_executor_resume_probe_isolates_source(tmp_path: Path) -> None:
    spec = _mapped_diagnostics_run_spec()
    spec.worker_execution.method_contract.phase_program.phases[0].max_steps = 7
    spec.training_config.n_batches = 7
    spec.checkpoint_progress.checkpoint_interval = 3
    spec.checkpoint_progress.progress_interval = 3
    source_root = tmp_path / "source-checkpoint"
    execute_training_run_spec(
        spec,
        preparation=_valid_materialized_preparation(spec),
        registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
        manifest_root=tmp_path / "source-manifest",
        checkpoint_root=source_root,
        run_id="probe-source",
        stop_after_barrier="after_train_batch",
    )
    before = fingerprint_checkpoint_custody_inputs(source_root, digest_blobs=True)
    published_before = authenticate_published_checkpoint_custody(source_root)

    with isolated_checkpoint_probe(source_root) as probe:
        resumed = execute_training_run_spec(
            spec,
            preparation=_valid_materialized_preparation(spec),
            registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
            manifest_root=tmp_path / "probe-manifest",
            checkpoint_root=probe.output_root,
            checkpoint_source_root=probe.source_root,
            run_id="probe-run",
            resume=True,
        )
        # The probe advanced and wrote its new transactions into the output
        # namespace, not the source.
        assert resumed.checkpoint_writes
        assert (probe.output_root / "latest.json").is_file()

    after = fingerprint_checkpoint_custody_inputs(source_root, digest_blobs=True)
    assert after == before
    assert not probe.output_root.exists()
    published_after = authenticate_published_checkpoint_custody(source_root)
    assert published_after.manifest.transaction_id == published_before.manifest.transaction_id


def test_executor_rejects_source_root_without_resume(tmp_path: Path) -> None:
    spec = _mapped_diagnostics_run_spec()
    with pytest.raises(Exception, match="checkpoint_source_root requires resume"):
        execute_training_run_spec(
            spec,
            preparation=_valid_materialized_preparation(spec),
            registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
            manifest_root=tmp_path / "manifest",
            checkpoint_root=tmp_path / "output",
            checkpoint_source_root=tmp_path / "source",
            run_id="no-resume",
        )
