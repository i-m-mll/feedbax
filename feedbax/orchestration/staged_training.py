"""Admission and withdrawal of durable staged training receipts."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from feedbax._atomic import sync_parent_directories
from feedbax.contracts.base import utc_now
from feedbax.contracts.manifest import (
    TrainingRunManifest,
    canonical_manifest_path,
    load_manifest,
    write_manifest,
)
from feedbax.contracts.spec_storage import canonical_training_run_spec_sha256
from feedbax.orchestration.bundle import canonical_run_bundle_sha256
from feedbax.persistence.manifest_discovery import find_manifest_by_id

if TYPE_CHECKING:
    from feedbax.orchestration.bundle import RunBundle, RunRowSpec, StagedTrainingBinding


class StagedTrainingConflict(ValueError):
    """The requested operation cannot act on the current staged receipt."""


@contextmanager
def training_receipt_lock(root: Path, manifest_id: str) -> Iterator[None]:
    """Serialize changes by canonical address, which survives manifest replacement."""
    path = canonical_manifest_path("TrainingRunManifest", manifest_id, root=root)
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def load_training_receipt(root: Path, manifest_id: str) -> tuple[TrainingRunManifest, Path]:
    manifest, path = find_manifest_by_id(
        manifest_id, root=root, expected_kind="TrainingRunManifest"
    )
    assert isinstance(manifest, TrainingRunManifest)
    return manifest, path


def load_staged_training_receipt(root: Path, manifest_id: str) -> tuple[TrainingRunManifest, Path]:
    """Read the staged address. Older discovery aliases carry no launch authority."""
    path = canonical_manifest_path("TrainingRunManifest", manifest_id, root=root)
    manifest = load_manifest(path)
    if not isinstance(manifest, TrainingRunManifest) or manifest.id != manifest_id:
        raise StagedTrainingConflict(
            f"Staged receipt at {path} does not own training run {manifest_id!r}"
        )
    return manifest, path


def staging_generation(metadata: Mapping[str, Any]) -> str | None:
    """Identify the current staging of a deterministic run ID, including older receipts."""
    value = (
        metadata.get("staging_generation")
        or metadata.get("restaged_at")
        or metadata.get("staged_at")
    )
    return value if isinstance(value, str) and value else None


def has_attempt_evidence(manifest: TrainingRunManifest) -> bool:
    return (
        manifest.started_at is not None
        or manifest.stopped
        or manifest.completed_batches is not None
        or manifest.terminal_certification is not None
        or bool(manifest.checkpoint_custody)
        or "staged_attempt" in manifest.metadata
    )


def staged_training_display_status(status: str | None, metadata: Mapping[str, Any]) -> str | None:
    if status == "pending" and "staged_attempt" in metadata:
        return "claimed"
    return status


def require_unclaimed_training_receipt(manifest: TrainingRunManifest) -> None:
    owned = manifest.status == "running" or has_attempt_evidence(manifest)
    if manifest.status != "pending" or owned:
        guidance = " Stop an owned attempt through its job or runner." if owned else ""
        raise StagedTrainingConflict(
            f"Training run {manifest.id!r} has no unclaimed pending intent "
            f"(receipt status {manifest.status!r}).{guidance}"
        )


def withdraw_staged_training(root: Path, manifest_id: str) -> tuple[TrainingRunManifest, Path]:
    with training_receipt_lock(root, manifest_id):
        manifest, _ = load_training_receipt(root, manifest_id)
        require_unclaimed_training_receipt(manifest)
        now = utc_now()
        withdrawn = manifest.model_copy(
            update={
                "status": "cancelled",
                "completed_at": now,
                "metadata": {**manifest.metadata, "cancelled_at": now.isoformat()},
            }
        )
        return withdrawn, write_manifest(withdrawn, root=root)


def _staged_owner(bundle: RunBundle, row: RunRowSpec) -> dict[str, str]:
    return {
        "run_set_dir": str(bundle.run_set_dir),
        "run_set_id": bundle.run_set_id,
        "row_id": row.row_id,
        "bundle_sha256": canonical_run_bundle_sha256(bundle),
    }


def _bound_receipt(
    binding: StagedTrainingBinding, row: RunRowSpec
) -> tuple[TrainingRunManifest, Path]:
    manifest_id = binding.manifest_id
    provenance = row.execution.row_provenance
    if provenance is None or provenance.planned_run_id != manifest_id:
        raise StagedTrainingConflict(f"Row {row.row_id!r} does not match its staged training run")
    try:
        manifest, path = load_staged_training_receipt(Path(binding.manifest_root), manifest_id)
    except FileNotFoundError as exc:
        raise StagedTrainingConflict(str(exc)) from exc
    if staging_generation(manifest.metadata) != binding.generation:
        raise StagedTrainingConflict(
            f"Training run {manifest_id!r} has a different staging generation"
        )
    spec = manifest.training_spec
    if (
        manifest.metadata.get("execution_target") != "local"
        or spec is None
        or spec.kind != "TrainingRunSpec"
        or canonical_training_run_spec_sha256(spec.inline) != row.execution.payload.sha256
    ):
        raise StagedTrainingConflict(
            f"Staged training run {manifest_id!r} does not bind the exact local row payload"
        )
    return manifest, path


def claim_staged_training(bundle: RunBundle) -> None:
    """Claim the bound receipt before provisioning. The exact same owner can resume."""
    binding = bundle.staged_training
    if binding is None:
        return
    if (
        bundle.deployment_policy.driver != "local"
        or bundle.deployment_policy.venue != "local"
        or bundle.deployment_policy.cloud_authorized
        or bundle.execution_family != "native-training"
        or len(bundle.rows) != 1
    ):
        raise StagedTrainingConflict("Staged training admission requires one native local row")
    row = bundle.rows[0]
    owner = _staged_owner(bundle, row)
    root = Path(binding.manifest_root)
    with training_receipt_lock(root, binding.manifest_id):
        manifest, path = _bound_receipt(binding, row)
        if manifest.status == "pending" and manifest.metadata.get("staged_attempt") == owner:
            # The bytes were synced before replacement, but the directory sync may have failed.
            sync_parent_directories(path)
            return
        require_unclaimed_training_receipt(manifest)
        claimed = manifest.model_copy(
            update={"metadata": {**manifest.metadata, "staged_attempt": owner}}
        )
        write_manifest(claimed, root=root)


def require_staged_training_owner(bundle: RunBundle, row: RunRowSpec) -> None:
    """Refuse to start a bound row unless its durable claim belongs to this run set."""
    binding = bundle.staged_training
    if binding is None:
        return
    manifest, _ = _bound_receipt(binding, row)
    if manifest.status != "pending" or manifest.metadata.get("staged_attempt") != _staged_owner(
        bundle, row
    ):
        raise StagedTrainingConflict(f"Training run {manifest.id!r} has no claim for this attempt")
