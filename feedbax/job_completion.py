"""Verify the existing training receipt and its exact terminal checkpoint together."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path

from feedbax.contracts.base import ParentRef
from feedbax.contracts.effective_success import Product
from feedbax.contracts.manifest import (
    TrainingRunManifest,
    canonical_manifest_path,
    load_manifest_bytes,
    training_run_manifest_id,
)
from feedbax.contracts.request_refs import ContentRef
from feedbax.contracts.studio_training import (
    PublishedTrainingCheckpoint,
    StudioGraphMethodPayload,
)
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.contracts.training import TrainingRunSpec
from feedbax.host.checkpoint_selection import selected_coordinates, verify_dependency_files
from feedbax.host.completion import CompletionFileError, completion_file
from feedbax.persistence.array_container import open_array_container
from feedbax.studio.canonical_train import (
    BoundStudioTrainingJobInput,
    verify_training_request_binding,
)
from feedbax.training.checkpoint_custody import (
    CheckpointCustodyError,
    CheckpointReadScope,
    CheckpointWriteResult,
    _checkpoint_latest_parent_ref,
    authenticate_checkpoint_custody_ref,
    checkpoint_slot_files,
    open_published_model_slot,
    validate_checkpoint_run_contract,
)
from feedbax.training.model_checkpoint import MODEL_TREE_MEDIA_TYPE, SemanticModelSlot


class CompletionUnavailable(ValueError):
    """A successful producer has no required terminal product."""


def training_binding_path(root: Path, job_id: str) -> Path:
    return root / "checkpoints" / job_id / "binding.json"


def _read_exact(root: Path, path: Path, digest: str) -> bytes:
    relative = path.relative_to(root).as_posix()
    with completion_file(root, relative) as (stream, _):
        raw = stream.read()
    if sha256(raw).hexdigest() != digest:
        raise CompletionFileError(f"Completion file changed: {relative}")
    return raw


def _file(root: Path, path: Path, digest: str, size: int) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": digest, "size_bytes": size}


def attest_checkpoint_publication(
    *,
    launch: BoundStudioTrainingJobInput,
    binding_sha256: str,
    checkpoint: PublishedTrainingCheckpoint,
    write: CheckpointWriteResult,
) -> dict:
    """Attest an already published checkpoint's exact file closure for the local Host."""
    from feedbax.training.standard_train_checkpoint import STANDARD_TRAIN_STATE_MEDIA_TYPE

    media = {slot.slot: slot.media_type for slot in write.manifest.slots}
    if media != {
        "model": MODEL_TREE_MEDIA_TYPE,
        "optimizer": STANDARD_TRAIN_STATE_MEDIA_TYPE,
        "prng": STANDARD_TRAIN_STATE_MEDIA_TYPE,
        "batch_counter": STANDARD_TRAIN_STATE_MEDIA_TYPE,
    }:
        raise ValueError("/slots: checkpoint selection requires the typed standard Train profile")
    root = launch.manifest_root
    binding_path = training_binding_path(root, checkpoint.pointer.run_id)
    files = [
        _file(root, binding_path, binding_sha256, binding_path.stat().st_size),
        _file(root, write.manifest_path, checkpoint.pointer.manifest_sha256, write.manifest_path.stat().st_size),
    ]
    files.extend(
        _file(root, path, digest, size)
        for slot in write.manifest.slots
        for path, (digest, size) in checkpoint_slot_files(write.manifest_path.parent, slot).items()
    )
    return {
        "producer_job_id": checkpoint.pointer.run_id, "project_id": launch.project_id,
        "request_id": launch.request_id, "manifest_root": str(root),
        "binding_sha256": binding_sha256, "checkpoint": checkpoint.model_dump(mode="json"),
        "files": files,
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
        method = StudioGraphMethodPayload.model_validate(launch.spec.method_payload.payload)
        if (
            not isinstance(manifest, TrainingRunManifest)
            or manifest.id != expected_id
            or manifest.job_id != producer_job_id
            or manifest.status != "completed"
            or manifest.stopped
            or manifest.completed_batches != method.target_total
        ):
            raise CompletionFileError(
                "Receipt does not certify this attempt at its authored target."
            )
        if request_files and (
            manifest.training_spec is None
            or TrainingRunSpec.model_validate(manifest.training_spec.inline) != launch.spec
        ):
            raise CompletionFileError("Receipt does not bind the admitted Train execution payload.")
        checkpoint_ref = launch.canonical_request.checkpoint if launch.canonical_request else None
        if (checkpoint_ref is None) != (launch.checkpoint_input is None):
            raise CompletionFileError("Completion does not bind its declared Train checkpoint.")
        if launch.checkpoint_input is not None and launch.resume is None:
            provenance = manifest.provenance.metadata if manifest.provenance is not None else {}
            initialization = provenance.get("checkpoint_initialization")
            mode = "continue_from" if method.continuation is not None else "initialize_from"
            if not isinstance(initialization, dict) or initialization.get("mode") != mode:
                raise CompletionFileError("Completion lacks its Train initialization provenance.")
            source_ref = (
                initialization.get("transformation", {}).get("producer_source")
                if method.continuation is not None else initialization.get("source")
            )
            source = ParentRef.model_validate(source_ref)
            expected = launch.checkpoint_input.checkpoint.pointer
            if (
                source.id != expected.transaction_id
                or source.metadata.get("manifest_sha256") != expected.manifest_sha256
            ):
                raise CompletionFileError("Completion initialized from a different checkpoint.")
        terminal = manifest.terminal_checkpoint
        if terminal is None or checkpoint is None:
            raise CompletionUnavailable("Successful producer has no terminal checkpoint.")
        if not isinstance(terminal, ParentRef):
            raise CompletionFileError("Receipt does not bind a checkpoint transaction reference.")
        pointer = checkpoint.pointer
        expected_root = training_binding_path(root, pointer.run_id).parent
        if manifest.inherited_terminal_checkpoint is not None:
            if launch.resume != checkpoint or pointer.run_id == producer_job_id:
                raise CompletionFileError(
                    "Inherited terminal checkpoint differs from the admitted resume input."
                )
        elif pointer.run_id != producer_job_id:
            raise CompletionFileError("Terminal checkpoint belongs to a different attempt.")
        if (
            checkpoint.checkpoint_root != expected_root
            or terminal.id != pointer.transaction_id
            or terminal.uri != pointer.manifest_relative_path
            or terminal.metadata.get("manifest_sha256") != pointer.manifest_sha256
            or checkpoint.completed_updates != method.target_total
            or pointer.completed_training_batches != method.target_total
        ):
            raise CompletionFileError(
                "Receipt, published pointer and authored completion disagree."
            )
        custody = authenticate_checkpoint_custody_ref(
            terminal, allowed_root=expected_root, scope=scope,
        )
        transaction = custody.manifest
        if (
            transaction.run_id != pointer.run_id
            or transaction.completed_coordinate != pointer.completed_coordinate
            or transaction.completed_training_batches != method.target_total
            or transaction.content_integrity_digest.transaction_root_sha256
            != pointer.transaction_root_sha256
        ):
            raise CompletionFileError(
                "Authenticated checkpoint disagrees with its completion pointer."
            )
        validate_checkpoint_run_contract(
            transaction, launch.spec, launch.spec.worker_execution.effective_phase.phase_program
        )
        transaction_path = expected_root / pointer.manifest_relative_path
        relative_transaction = transaction_path.relative_to(root).as_posix()
        with completion_file(root, relative_transaction) as (_, size):
            files = [
                _file(root, binding_path, binding_sha256, len(binding_raw)),
                _file(root, receipt_path, receipt["manifest_sha256"], len(raw)),
                _file(root, transaction_path, pointer.manifest_sha256, size),
            ]
        files.extend(_file(root, path, ref.sha256, ref.size_bytes) for path, ref in request_files)
        files.extend(
            _file(root, path, digest, size)
            for slot in transaction.slots
            for path, (digest, size) in checkpoint_slot_files(transaction_path.parent, slot).items()
        )
        return {
            "producer_job_id": producer_job_id,
            "project_id": launch.project_id,
            "request_id": launch.request_id,
            "manifest_root": str(root),
            "binding_sha256": binding_sha256,
            "receipt": receipt,
            "checkpoint": checkpoint.model_dump(mode="json"),
            "files": files,
        }
    except (CompletionFileError, CompletionUnavailable):
        raise
    except (ValueError, KeyError, OSError, CheckpointCustodyError) as error:
        raise CompletionFileError(str(error)) from error


def verify_bound_training_completion(
    completion: dict, *, scope: CheckpointReadScope = "full",
) -> BoundStudioTrainingJobInput:
    """Recheck a pinned producer binding immediately before a consumer uses it."""
    try:
        root = Path(completion["manifest_root"])
        producer_job_id = completion["producer_job_id"]
        binding_sha256 = completion["binding_sha256"]
        path = training_binding_path(root, producer_job_id)
        launch = BoundStudioTrainingJobInput.model_validate_json(_read_exact(root, path, binding_sha256))
        checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
        actual = verify_training_completion(
            producer_job_id=producer_job_id,
            launch=launch,
            binding_sha256=binding_sha256,
            receipt=completion["receipt"],
            checkpoint=checkpoint,
            scope=scope,
        )
        if actual != completion:
            raise CompletionFileError("Rechecked completion differs from its pinned binding.")
        return launch
    except CompletionFileError:
        raise
    except (ValueError, KeyError, OSError) as error:
        raise CompletionFileError(str(error)) from error

def verify_training_model_product(completion: dict) -> tuple[Product, SemanticModelSlot]:
    """Project the exact saved Train terminal checkpoint to its model product."""
    launch = verify_bound_training_completion(completion, scope="model_only")
    if launch.canonical_request is None:
        raise CompletionUnavailable("Model product requires a canonical saved Train request.")
    checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
    try:
        model = open_published_model_slot(checkpoint.checkpoint_root, checkpoint.pointer)
        if model is None:
            raise CompletionUnavailable("Terminal checkpoint has no semantic model tree.")
        with (
            model.provider.open_owned_object(model.arrays) as path,
            open_array_container(path, expected_descriptor_sha256=model.encoding.array_descriptor_sha256),
        ):
            pass
    except CompletionUnavailable:
        raise
    except (CheckpointCustodyError, ValueError, OSError) as error:
        raise CompletionFileError(str(error)) from error

    descriptor = model.tree.descriptor
    return Product(
        name="model",
        content=ContentRef(
            sha256=descriptor.sha256,
            size=descriptor.size_bytes,
            media_type=MODEL_TREE_MEDIA_TYPE,
            tree=True,
        ),
        descriptor=model.encoding.model_dump(mode="json"),
    ), model


def read_host_completion_binding() -> dict:
    """Authenticate the single subject edge supplied to this worker delivery."""
    path = Path(os.environ["FEEDBAX_HOST_COMPLETION_BINDINGS"])
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != os.environ["FEEDBAX_HOST_COMPLETION_BINDINGS_SHA256"]:
        raise CompletionFileError("Host completion binding changed after assignment.")
    (binding,) = strict_json_loads(raw)["bindings"]
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
            or checkpoint.pointer.completed_training_batches != update
        ):
            raise CompletionFileError("/inputs: selected checkpoint has a different attempt or batch")
        launch = BoundStudioTrainingJobInput.model_validate_json(
            _read_exact(root, training_binding_path(root, attempt), member["binding_sha256"]),
        )
        if (
            launch.request_id != producer_request
            or launch.project_id != member["project_id"]
            or launch.manifest_root != root
            or launch.canonical_request is None
        ):
            raise CompletionFileError("/inputs: selected checkpoint lacks its canonical Train binding")
        verify_training_request_binding(launch)
        custody = authenticate_checkpoint_custody_ref(
            _checkpoint_latest_parent_ref(checkpoint.pointer),
            allowed_root=checkpoint.checkpoint_root,
        )
        manifest = custody.manifest
        if (
            manifest.completed_training_batches != update
            or manifest.barrier != "after_train_batch"
            or manifest.completed_coordinate != checkpoint.pointer.completed_coordinate
            or manifest.content_integrity_digest.transaction_root_sha256
            != checkpoint.pointer.transaction_root_sha256
        ):
            raise CompletionFileError("/inputs: selected checkpoint differs from the verified transaction")
        validate_checkpoint_run_contract(
            manifest,
            launch.spec,
            launch.spec.worker_execution.effective_phase.phase_program,
        )
        pinned.append((checkpoint, launch))
    return tuple(pinned)
