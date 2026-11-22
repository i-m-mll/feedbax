"""Verify the existing training receipt and its exact terminal checkpoint together."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from feedbax.contracts.base import ParentRef
from feedbax.contracts.method_checkpoint import MethodCheckpointIdentity
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE
from feedbax.contracts.effective_success import Product
from feedbax.contracts.manifest import (
    TrainingRunManifest,
    canonical_manifest_path,
    load_manifest_bytes,
    training_run_manifest_id,
)
from feedbax.contracts.request_refs import ContentRef
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.persistence.checkpoint_selection import selected_coordinates, verify_dependency_files
from feedbax.persistence.completion import CompletionFileError, completion_file, read_completion_bindings, verify_completion_files
from feedbax.persistence.canonical_train import (
    BoundStudioTrainingJobInput,
    verify_training_request_binding,
)
from feedbax.persistence.method_checkpoint import method_checkpoint_files, read_method_checkpoint_manifest
from feedbax.plugins.declarations import DistributionSource

if TYPE_CHECKING:
    from feedbax.host.model_checkpoint import SemanticModelSlot


type CheckpointReadScope = Literal["full", "model_only"]

_PROVIDER = TypeAdapter(DistributionSource)


class CompletionUnavailable(ValueError):
    """A successful producer has no required terminal product."""


def training_binding_path(root: Path, job_id: str) -> Path:
    return root / "checkpoints" / job_id / "binding.json"


def read_training_binding(root: Path, attempt: str, digest: str) -> BoundStudioTrainingJobInput:
    return BoundStudioTrainingJobInput.model_validate_json(
        _read_exact(root, training_binding_path(root, attempt), digest),
    )


def _read_exact(root: Path, path: Path, digest: str) -> bytes:
    relative = path.relative_to(root).as_posix()
    with completion_file(root, relative) as (stream, _):
        raw = stream.read()
    if sha256(raw).hexdigest() != digest:
        raise CompletionFileError(f"Completion file changed: {relative}")
    return raw


def _file(root: Path, path: Path, digest: str, size: int) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": digest, "size_bytes": size}


def training_metric_file(root: Path, attempt: str, manifest: TrainingRunManifest) -> dict | None:
    """Read the declared metrics member; the Store verifies its bytes at ingress."""
    records = [artifact for artifact in manifest.artifacts if artifact.role == "training_metrics"]
    if not records:
        return None
    if len(records) != 1:
        raise CompletionFileError("Train receipt has repeated metric history.")
    (artifact,) = records
    history = training_binding_path(root, attempt).with_name("metrics.jsonl")
    if (
        artifact.uri != history.relative_to(root).as_posix()
        or artifact.sha256 is None or artifact.size_bytes is None or artifact.media_type != "application/x-ndjson"
    ):
        raise CompletionFileError("Train receipt metric history is outside this attempt.")
    return _file(root, history, artifact.sha256, artifact.size_bytes)


def _checkpoint_matches_request(identity: MethodCheckpointIdentity, launch: BoundStudioTrainingJobInput) -> bool:
    return (identity.request_id, identity.method_key) == (launch.request_id, launch.spec.method)

def checkpoint_reference(checkpoint: PublishedTrainingCheckpoint) -> ParentRef:
    pointer = checkpoint.pointer
    return ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id=pointer.transaction_id,
        uri=pointer.manifest_relative_path,
        role="training_checkpoint_custody",
        metadata={"manifest_sha256": pointer.manifest_sha256},
    )


def _checkpoint_files(root: Path, checkpoint: PublishedTrainingCheckpoint, scope: CheckpointReadScope) -> list[dict]:
    return [
        _file(root, checkpoint.checkpoint_root / relative, ref.sha256, ref.size_bytes)
        for relative, ref in method_checkpoint_files(
            checkpoint.checkpoint_root, checkpoint.pointer, include_extra=scope == "full",
        ).items()
    ]


def attest_checkpoint_publication(
    *,
    launch: BoundStudioTrainingJobInput,
    binding_sha256: str,
    checkpoint: PublishedTrainingCheckpoint,
) -> dict:
    """Attest the published checkpoint before notifying the Host."""
    root = launch.manifest_root
    binding_path = training_binding_path(root, checkpoint.pointer.run_id)
    return {
        "producer_job_id": checkpoint.pointer.run_id,
        "project_id": launch.project_id,
        "request_id": launch.request_id,
        "manifest_root": str(root),
        "binding_sha256": binding_sha256,
        "checkpoint": checkpoint.model_dump(mode="json"),
        "files": [
            _file(root, binding_path, binding_sha256, binding_path.stat().st_size),
            *_checkpoint_files(root, checkpoint, "full"),
        ],
    }


def verify_training_completion(
    *,
    producer_job_id: str,
    launch: BoundStudioTrainingJobInput,
    binding_sha256: str,
    receipt: dict,
    checkpoint: PublishedTrainingCheckpoint | None,
    scope: CheckpointReadScope = "full",
) -> dict:
    """Authenticate the producer's bound terminal pair without reading a latest pointer."""
    root = launch.manifest_root
    if not root.is_absolute() or root != root.resolve():
        raise CompletionFileError("Completion manifest root must be absolute and canonical.")
    try:
        request_files = verify_training_request_binding(launch)
        expected_id = training_run_manifest_id(producer_job_id)
        if receipt.get("manifest_id") != expected_id:
            raise CompletionFileError("Completion receipt id does not identify this attempt.")
        binding_path = training_binding_path(root, producer_job_id)
        binding_raw = _read_exact(root, binding_path, binding_sha256)
        if BoundStudioTrainingJobInput.model_validate_json(binding_raw) != launch:
            raise CompletionFileError("Completion does not match the producer's admitted input.")
        receipt_path = canonical_manifest_path("TrainingRunManifest", expected_id, root=root)
        raw = _read_exact(root, receipt_path, receipt["manifest_sha256"])
        manifest = load_manifest_bytes(raw)
        target = launch.spec.target_total
        if (
            not isinstance(manifest, TrainingRunManifest)
            or manifest.id != expected_id
            or manifest.job_id != producer_job_id
            or manifest.status != "completed"
            or manifest.stopped
            or manifest.completed_batches != target
        ):
            raise CompletionFileError("Receipt does not certify this attempt at its authored target.")
        provenance = manifest.provenance.metadata
        if (provenance.get("request_id"), provenance.get("binding_sha256")) != (launch.request_id, binding_sha256):
            raise CompletionFileError("Receipt does not bind the admitted Train request and launch.")
        if launch.collected_resume == producer_job_id or (
            launch.collected_resume is None and (launch.canonical_request.checkpoint is None) != (launch.checkpoint_input is None)
        ):
            raise CompletionFileError("Completion does not bind its declared Train checkpoint.")
        if launch.checkpoint_input is not None and launch.resume is None:
            expected = checkpoint_reference(launch.checkpoint_input.checkpoint)
            observed = manifest.provenance.metadata.get("checkpoint_initialization")
            if observed != expected.model_dump(mode="json"):
                raise CompletionFileError("Completion initialized from a different checkpoint.")
        if checkpoint is None:
            raise CompletionUnavailable("Successful producer has no terminal checkpoint.")
        pointer = checkpoint.pointer
        expected_root = training_binding_path(root, pointer.run_id).parent
        owner = launch.resume.pointer.run_id if manifest.inherited_terminal_checkpoint and launch.resume else producer_job_id
        if (
            pointer.run_id != owner
            or checkpoint.checkpoint_root != expected_root
            or manifest.terminal_checkpoint != checkpoint_reference(checkpoint)
            or checkpoint.completed_updates != target
        ):
            raise CompletionFileError("Receipt, published pointer and authored completion disagree.")
        transaction = read_method_checkpoint_manifest(expected_root, pointer)
        if not _checkpoint_matches_request(transaction.identity, launch) or transaction.clock.completed_updates != target:
            raise CompletionFileError("Authenticated checkpoint disagrees with its admitted method or request.")
        files = [
            _file(root, binding_path, binding_sha256, len(binding_raw)),
            _file(root, receipt_path, receipt["manifest_sha256"], len(raw)),
            *(_file(root, path, ref.sha256, ref.size_bytes) for path, ref in request_files),
            *_checkpoint_files(root, checkpoint, scope),
        ]
        metrics = training_metric_file(root, producer_job_id, manifest)
        method = dict(provenance["method"])
        if method.pop("key") != launch.spec.method:
            raise CompletionFileError("Train receipt names a different method.")
        provider = _PROVIDER.validate_python(method)
        completion = {
            "producer_job_id": producer_job_id,
            "project_id": launch.project_id,
            "request_id": launch.request_id,
            "manifest_root": str(root),
            "binding_sha256": binding_sha256,
            "receipt": receipt,
            "checkpoint": checkpoint.model_dump(mode="json"),
            "files": files,
            "metrics": metrics,
            "provider": _PROVIDER.dump_python(provider, mode="json"),
        }
        verify_completion_files(completion, checkpoint_model=scope == "model_only")
        return completion
    except (CompletionFileError, CompletionUnavailable):
        raise
    except (ValueError, KeyError, OSError) as error:
        raise CompletionFileError(str(error)) from error


def verify_bound_training_completion(
    completion: dict, *, scope: CheckpointReadScope = "full",
) -> BoundStudioTrainingJobInput:
    """Recheck a pinned producer binding immediately before a consumer uses it."""
    try:
        root = Path(completion["manifest_root"])
        launch = read_training_binding(root, completion["producer_job_id"], completion["binding_sha256"])
        checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
        actual = verify_training_completion(
            producer_job_id=completion["producer_job_id"], launch=launch,
            binding_sha256=completion["binding_sha256"], receipt=completion["receipt"],
            checkpoint=checkpoint, scope=scope,
        )
        recorded = completion
        if scope == "model_only":
            from feedbax.contracts.method_checkpoint import method_checkpoint_extra_path

            manifest = read_method_checkpoint_manifest(checkpoint.checkpoint_root, checkpoint.pointer)
            extra = (checkpoint.checkpoint_root / Path(checkpoint.pointer.manifest_relative_path).parent
                     / method_checkpoint_extra_path(manifest.extra)).relative_to(root).as_posix()
            recorded = completion | {"files": [file for file in completion["files"] if file["path"] != extra]}
        if actual != recorded:
            raise CompletionFileError("Rechecked completion differs from its pinned binding.")
        return launch
    except CompletionFileError:
        raise
    except (ValueError, KeyError, OSError) as error:
        raise CompletionFileError(str(error)) from error


def verify_training_model_product(completion: dict) -> tuple[Product, SemanticModelSlot]:
    """Project an exact approved Train completion to its independent model product."""
    from feedbax.application.composition import load_application
    from feedbax.host.method_checkpoint import read_method_checkpoint

    verify_bound_training_completion(completion, scope="model_only")
    checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
    model = read_method_checkpoint(
        checkpoint.checkpoint_root, checkpoint.pointer, components=load_application().bundle.components,
    ).model
    descriptor = model.tree.descriptor
    return Product(
        name="model",
        content=ContentRef(
            sha256=descriptor.sha256, size=descriptor.size_bytes, media_type=MODEL_TREE_MEDIA_TYPE, tree=True,
        ),
        descriptor=model.encoding.model_dump(mode="json"),
    ), model


def read_host_completion_binding() -> dict:
    """Authenticate the single subject edge supplied to this worker delivery."""
    (binding,) = read_completion_bindings()["bindings"]
    return binding


def load_bound_checkpoint_selection(
    binding: dict, *, root: Path, producer_request: str, coordinates: tuple[int, ...],
    project: str | None = None, allow_gaps: bool = False,
) -> tuple[tuple[PublishedTrainingCheckpoint, BoundStudioTrainingJobInput], ...]:
    """Open exactly the ordered checkpoint publications already bound by the Host."""
    selection = binding["checkpoint_selection"]
    if selection["requested"] != list(coordinates) or ("gaps" in selection and not allow_gaps):
        raise CompletionFileError("Host selection differs from the authored checkpoint coordinates.")
    verify_dependency_files(binding)
    if closure := selection.get("closure"):
        product = closure["product"]
        if (
            product["request_id"] != producer_request or product["manifest_root"] != str(root)
            or (project is not None and product["project_id"] != project)
        ):
            raise CompletionFileError("/inputs: checkpoint coverage closed on a different Train request")
        if closure["kind"] == "completion":
            verify_bound_training_completion(product, scope="model_only")
    pinned = []
    for update, member in zip(selected_coordinates(selection), selection["members"], strict=True):
        if (
            member["request_id"] != producer_request
            or member["manifest_root"] != str(root)
            or (project is not None and member["project_id"] != project)
        ):
            raise CompletionFileError("/inputs: selected member is from a different Train request")
        checkpoint = PublishedTrainingCheckpoint.model_validate(member["checkpoint"])
        attempt = member["producer_job_id"]
        if (
            checkpoint.pointer.run_id != attempt
            or checkpoint.completed_updates != update
            or checkpoint.checkpoint_root != root / "checkpoints" / attempt
            or checkpoint.pointer.completed_updates != update
        ):
            raise CompletionFileError("/inputs: selected checkpoint has a different attempt or batch")
        launch = read_training_binding(root, attempt, member["binding_sha256"])
        if (
            launch.request_id != producer_request
            or launch.project_id != member["project_id"]
            or launch.manifest_root != root
            or launch.canonical_request is None
        ):
            raise CompletionFileError("/inputs: selected checkpoint lacks its canonical Train binding")
        verify_training_request_binding(launch)
        manifest = read_method_checkpoint_manifest(checkpoint.checkpoint_root, checkpoint.pointer)
        if manifest.clock.completed_updates != update or not _checkpoint_matches_request(manifest.identity, launch):
            raise CompletionFileError("/inputs: selected checkpoint differs from the verified transaction")
        pinned.append((checkpoint, launch))
    return tuple(pinned)
