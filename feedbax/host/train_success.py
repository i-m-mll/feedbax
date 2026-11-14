"""Commit typed checkpoint evidence for an accepted managed Train attempt."""

from hashlib import sha256
from pathlib import Path, PurePosixPath
import re

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.checkpoint_projection import (
    STANDARD_TRAIN_STATE_MEDIA_TYPE,
    CarriedOptimizer,
    CheckpointClock,
    CheckpointOccurrenceRef,
    ContinuationOccurrence,
    ContinuationSlot,
)
from feedbax.contracts.checkpoints import PHASE_BOUNDARY_DECISION_METADATA_KEY, CheckpointTransactionManifest
from feedbax.contracts.effective_success import (
    Bindings, Product, RequestObject, read_success_object, select_success_product,
    success_content_ref,
)
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE
from feedbax.contracts.request_refs import ProductRef
from feedbax.contracts.strict_json import strict_model_validate_json
from feedbax.contracts.studio_training import StudioGraphMethodPayload
from feedbax.contracts.training import TrainingRunSpec
from feedbax.host.completion import completion_file
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.receipt_content import success_content_objects


def publish_train_success(
    store_root: Path, attempt: str, request_id: str, launch: dict, outcome: dict,
    predecessor: tuple[dict, dict] | None,
) -> dict:
    """Use only the accepted completion and the producer pinned at Host delivery."""
    completion = outcome["completion"]
    checkpoint = completion["checkpoint"]
    pointer = checkpoint["pointer"]
    owner = pointer["run_id"]
    relative = PurePosixPath(pointer["manifest_relative_path"])
    checkpoint_root = store_root / "checkpoints" / owner
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", owner)
        or (owner != attempt and checkpoint != launch.get("resume"))
        or checkpoint["checkpoint_root"] != str(checkpoint_root)
        or relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2
    ):
        raise ValueError("Model product completion differs from the accepted Train attempt")
    product = Product.model_validate(outcome["model_product"])
    tree = success_content_ref(product.content)
    if product.name != "model" or product.content.media_type != MODEL_TREE_MEDIA_TYPE or not isinstance(tree, ContentTreeRef):
        raise ValueError("Train result requires a semantic checkpoint/model tree")

    binding = launch["canonical_request"]
    request = RequestObject.model_validate({
        "id": request_id, "object": binding["request"], "documents": binding["documents"],
    })
    model_root = checkpoint_root / relative.parent / "semantic-model"
    resume = launch.get("resume")
    recovery = _occurrence(resume["pointer"]) if resume is not None else None
    store = ImmutableArtifactBlobProvider(store_root)
    with store._exclusive_lock():
        view = SuccessView(store)
        if existing := view.attempt_ref(attempt):
            committed = view.read_owned_success(existing).receipt
            if (
                committed.attempt != attempt or committed.request != request
                or committed.products != {"checkpoint": (product,)}
                or committed.continuation is None
                or committed.continuation.provenance != _occurrence(pointer)
                or committed.recovery != recovery
                or (predecessor is None) != (not committed.inputs)
            ):
                raise ValueError("Published Train attempt differs from the accepted result")
            if predecessor is not None:
                completion, prior = predecessor
                (bound,) = committed.inputs["checkpoint"]
                if (
                    bound.selection is None
                    or bound.selection.receipt.model_dump(mode="json") != prior.get("store_success")
                    or (bound.selection.occurrence is not None
                        and bound.selection.occurrence != _occurrence(completion["checkpoint"]["pointer"]))
                ):
                    raise ValueError("Published Train inputs differ from the delivered checkpoint")
            return existing.model_dump(mode="json")
        bindings = _train_inputs(view, request, predecessor)
        source = ImmutableArtifactBlobProvider(model_root)
        files = {file["path"]: (file["sha256"], file["size_bytes"]) for file in completion["files"]}
        manifest_path = checkpoint_root / relative
        manifest = _check_model_slot(store_root, manifest_path, pointer, product)

        def retain(path: Path, digest: str, size: int) -> ContentObjectRef:
            if files.get(path.relative_to(store_root).as_posix()) != (digest, size):
                raise ValueError("Checkpoint object is absent from the accepted completion evidence")
            return store._store_file(path, sha256=digest, size_bytes=size)

        for ref in success_content_objects(source, product):
            retain(source.object_path(ref), ref.sha256, ref.size_bytes)
        slots = {slot.slot: slot for slot in manifest.slots}
        if any(
            name not in slots or slots[name].media_type != STANDARD_TRAIN_STATE_MEDIA_TYPE
            for name in ("batch_counter", "optimizer", "prng")
        ):
            raise ValueError("Train result lacks typed batch counter, optimizer, or PRNG state")
        arrays = {
            name: ContinuationSlot(
                content=retain(manifest_path.parent / slot.relative_path, slot.sha256, slot.size_bytes),
                array_descriptor_sha256=slot.metadata["array_descriptor_sha256"],
                semantic_abi_sha256=slot.semantic_abi.fingerprint_sha256,
            )
            for name in ("batch_counter", "optimizer", "prng")
            for slot in (slots[name],)
        }
        coordinate = manifest.completed_coordinate
        boundary = manifest.metadata.get(PHASE_BOUNDARY_DECISION_METADATA_KEY)
        source_spec = TrainingRunSpec.model_validate(
            manifest.run_contract_binding.canonical_projection["training_run_spec"]
        )
        optimizer = StudioGraphMethodPayload.model_validate(source_spec.method_payload.payload).optimizer
        continuation = ContinuationOccurrence(
            provenance=_occurrence(pointer),
            learning_rate=optimizer.lr_schedule or optimizer.params["learning_rate"],
            carried_optimizer=CarriedOptimizer.from_spec(optimizer, source_spec.training_config.grad_clip),
            clock=CheckpointClock(
                completed_batches=manifest.completed_training_batches,
                segment_start_batch=manifest.segment_lineage.start_batch,
                phase=coordinate.phase,
                barrier=manifest.barrier,
                program_step=coordinate.program_step,
                inner_step=coordinate.inner_step,
                outer_step=coordinate.outer_step,
                schedule_origin_step=coordinate.schedule_origin_step,
                phase_boundary=None if boundary is None else (boundary["next_phase"],),
            ),
            model_semantic_abi_sha256=slots["model"].semantic_abi.fingerprint_sha256,
            batch_counter=arrays["batch_counter"],
            optimizer=arrays["optimizer"],
            prng=arrays["prng"],
        )
        published = view.publish_bound(
            request, attempt=attempt, products={"checkpoint": (product,)}, bindings=bindings,
            continuation=continuation, recovery=recovery,
        )
    return published.ref.model_dump(mode="json")


def _occurrence(pointer: dict) -> CheckpointOccurrenceRef:
    return CheckpointOccurrenceRef(
        run_id=pointer["run_id"], transaction_id=pointer["transaction_id"],
        manifest_sha256=pointer["manifest_sha256"],
    )


def _check_model_slot(
    root: Path, manifest_path: Path, pointer: dict, product: Product,
) -> CheckpointTransactionManifest:
    with completion_file(root, manifest_path.relative_to(root).as_posix()) as (stream, _):
        raw = stream.read()
    if sha256(raw).hexdigest() != pointer["manifest_sha256"]:
        raise ValueError("Train checkpoint manifest changed before Store publication")
    manifest = strict_model_validate_json(CheckpointTransactionManifest, raw)
    (slot,) = (slot for slot in manifest.slots if slot.slot == "model")
    content = product.content
    if (
        manifest.transaction_id != pointer["transaction_id"]
        or (slot.role, slot.media_type, slot.sha256, slot.size_bytes)
        != ("model", content.media_type, content.sha256, content.size)
        or slot.metadata["semantic_model"] != product.descriptor
    ):
        raise ValueError("Train model product differs from the authenticated checkpoint slot")
    return manifest


def _train_inputs(view: SuccessView, request: RequestObject, predecessor: tuple[dict, dict] | None) -> Bindings:
    decoded = view.read_request(request)
    if not decoded.inputs and predecessor is None:
        return {}
    if predecessor is None or decoded.inputs.keys() != {"checkpoint"}:
        raise ValueError("Train result lacks its exact predecessor binding")
    completion, outcome = predecessor
    (member,) = decoded.inputs["checkpoint"]
    reference = member.reference
    if not isinstance(reference, ProductRef) or reference.request_id != completion["request_id"]:
        raise ValueError("Train result names a different predecessor request")
    ref = view.attempt_ref(completion["producer_job_id"])
    if ref is None or outcome.get("store_success") != ref.model_dump(mode="json"):
        raise ValueError("Pinned predecessor lacks its committed typed model receipt")
    parent = read_success_object(ref, view.store.get_object)
    model = select_success_product(reference, parent)
    if model.model_dump(mode="json") != outcome.get("model_product"):
        raise ValueError("Pinned predecessor typed product differs from the accepted Train outcome")
    if isinstance(decoded.spec.method_payload.payload.get("checkpoint_initialization"), dict):
        if parent.continuation is None:
            raise ValueError("Pinned predecessor has no typed continuation evidence")
        if parent.continuation.provenance != _occurrence(completion["checkpoint"]["pointer"]):
            raise ValueError("Pinned predecessor receipt names a different checkpoint occurrence")
    return {"checkpoint": (view.resolve(decoded, member, ref),)}
