"""Commit typed checkpoint evidence for an accepted managed Train attempt."""

from hashlib import sha256
from pathlib import Path, PurePosixPath
import re

from pydantic import TypeAdapter

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.checkpoint_projection import CarriedMethodState, CheckpointOccurrenceRef, ContinuationOccurrence
from feedbax.contracts.method_checkpoint import CheckpointLatestPointer, MethodCheckpointManifest, method_checkpoint_extra_path
from feedbax.contracts.effective_success import (
    Bindings, Product, RequestObject, read_success_object, select_success_product,
    success_content_ref,
)
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE
from feedbax.contracts.request_refs import ContentRef, ProductRef, is_synthesized_model_product
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.strict_json import strict_model_validate_json
from feedbax.persistence.completion import completion_file
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.receipt_content import success_content_objects
from feedbax.persistence.train_job import TrainJobEntry
from feedbax.plugins.declarations import DistributionSource
from feedbax.contracts.execution_target import AttemptTarget, LOCAL_TARGET


def publish_train_success(
    store_root: Path, attempt: str, request_id: str, launch: dict, outcome: dict,
    predecessor: tuple[dict, dict] | None, *, collected: TrainJobEntry | None = None,
    target: AttemptTarget = LOCAL_TARGET,
    execution_evidence: ContentObjectRef | None = None,
) -> dict:
    """Use only the accepted completion and the producer pinned at Host delivery."""
    completion = outcome["completion"]
    checkpoint = completion["checkpoint"]
    pointer = checkpoint["pointer"]
    owner = pointer["run_id"]
    relative = PurePosixPath(CheckpointLatestPointer.model_validate(pointer).manifest_relative_path)
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
    synthesized = ContentObjectRef.model_validate(binding["model_receipt"]) if "model_receipt" in binding else None
    request = RequestObject.model_validate({
        "id": request_id, "object": binding["request"], "documents": binding["documents"],
    })
    model_root = checkpoint_root / relative.parent / "semantic-model"
    resume = launch.get("resume")
    recovery = _occurrence(resume["pointer"]) if resume is not None else None
    if collected is not None:
        if collected.request != request or collected.attempt != launch.get("collected_resume"):
            raise ValueError("Collected Resume differs from its original Train request")
        recovery = _occurrence(collected.recovery.pointer.model_dump(mode="json"))
    store = ImmutableArtifactBlobProvider(store_root)
    metrics = completion.get("metrics")
    provider = TypeAdapter(DistributionSource).validate_python(completion["provider"])
    products = {"checkpoint": (product,)}
    if metrics is not None:
        products["metrics"] = (Product(name="updates", content=ContentRef(
            sha256=metrics["sha256"], size=metrics["size_bytes"], media_type="application/x-ndjson",
        )),)
    with store._exclusive_lock():
        view = SuccessView(store)
        if existing := view.attempt_ref(attempt):
            committed = view.read_owned_success(existing).receipt
            if (
                committed.attempt != attempt or committed.request != request or committed.target != target
                or committed.products != products or committed.provider != provider
                or committed.continuation is None
                or committed.continuation.provenance != _occurrence(pointer)
                or committed.recovery != recovery
                or (collected is not None and committed.inputs != collected.inputs)
                or (collected is None and bool(committed.inputs) != (
                    predecessor is not None or binding.get("imported_model") is not None or synthesized is not None
                ))
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
            elif (binding.get("imported_model") is not None or synthesized is not None) and committed.inputs != bind_train_inputs(
                view, request, None, synthesized=synthesized,
            ):
                raise ValueError("Published Train inputs differ from the delivered model source")
            view.ensure_published_run_ref(existing, committed)
            return existing.model_dump(mode="json")
        bindings = collected.inputs if collected is not None else bind_train_inputs(
            view, request, predecessor, synthesized=synthesized,
        )
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
        extra = manifest.extra
        retain(manifest_path.parent / method_checkpoint_extra_path(extra), extra.object.sha256, extra.object.size_bytes)
        if metrics is not None:
            expected_metrics = store_root / "checkpoints" / attempt / "metrics.jsonl"
            if metrics["path"] != expected_metrics.relative_to(store_root).as_posix():
                raise ValueError("Train metric history belongs to a different attempt")
            store._store_file(expected_metrics, sha256=metrics["sha256"], size_bytes=metrics["size_bytes"])
        continuation = ContinuationOccurrence(
            provenance=_occurrence(pointer),
            clock=manifest.clock,
            state=CarriedMethodState(
                method_key=manifest.identity.method_key,
                extra=manifest.extra,
            ),
        )
        published = view.publish_bound(
            request, attempt=attempt, products=products, bindings=bindings,
            continuation=continuation, recovery=recovery, provider=provider, target=target,
            execution_evidence=execution_evidence,
        )
    return published.ref.model_dump(mode="json")


def _occurrence(pointer: dict) -> CheckpointOccurrenceRef:
    return CheckpointOccurrenceRef(
        run_id=pointer["run_id"], transaction_id=pointer["transaction_id"],
        manifest_sha256=pointer["manifest_sha256"],
    )


def _check_model_slot(
    root: Path, manifest_path: Path, pointer: dict, product: Product,
) -> MethodCheckpointManifest:
    with completion_file(root, manifest_path.relative_to(root).as_posix()) as (stream, _):
        raw = stream.read()
    if sha256(raw).hexdigest() != pointer["manifest_sha256"]:
        raise ValueError("Train checkpoint manifest changed before Store publication")
    manifest = strict_model_validate_json(MethodCheckpointManifest, raw)
    identity = CheckpointLatestPointer.model_validate(pointer)
    model = manifest.model
    if (
        manifest.transaction_id != identity.transaction_id or manifest.run_id != identity.run_id
        or manifest.state_sha256 != identity.transaction_root_sha256
        or model.tree != success_content_ref(product.content)
        or model.encoding.model_dump(mode="json") != product.descriptor
    ):
        raise ValueError("Train model product differs from the authenticated checkpoint")
    return manifest


def bind_train_inputs(
    view: SuccessView, request: RequestObject, predecessor: tuple[dict, dict] | None,
    *, synthesized: ContentObjectRef | None = None,
) -> Bindings:
    decoded = view.read_request(request)
    if not decoded.inputs and predecessor is None:
        return {}
    if decoded.inputs.keys() != {"checkpoint"}:
        raise ValueError("Train result lacks its exact predecessor binding")
    (member,) = decoded.inputs["checkpoint"]
    reference = member.reference
    if predecessor is None and synthesized is not None and is_synthesized_model_product(reference):
        return {"checkpoint": (view.resolve(decoded, member, synthesized),)}
    if isinstance(reference, ContentRef) and predecessor is None:
        from feedbax.contracts.loaded_model import IMPORTED_MODEL_MEDIA_TYPE

        if reference.media_type == IMPORTED_MODEL_MEDIA_TYPE:
            return {"checkpoint": (view.resolve(decoded, member, None),)}
    if predecessor is None:
        raise ValueError("Train result lacks its exact predecessor binding")
    completion, outcome = predecessor
    if not isinstance(reference, ProductRef) or reference.request_id != completion["request_id"]:
        raise ValueError("Train result names a different predecessor request")
    ref = view.attempt_ref(completion["producer_job_id"])
    if ref is None or outcome.get("store_success") != ref.model_dump(mode="json"):
        raise ValueError("Pinned predecessor lacks its committed typed model receipt")
    parent = read_success_object(ref, view.store.get_object)
    model = select_success_product(reference, parent)
    if model.model_dump(mode="json") != outcome.get("model_product"):
        raise ValueError("Pinned predecessor typed product differs from the accepted Train outcome")
    if isinstance(decoded.spec, DocumentTrainingRequestSpec) and decoded.spec.continuation is not None:
        if parent.continuation is None:
            raise ValueError("Pinned predecessor has no typed continuation evidence")
        if parent.continuation.provenance != _occurrence(completion["checkpoint"]["pointer"]):
            raise ValueError("Pinned predecessor receipt names a different checkpoint occurrence")
    return {"checkpoint": (view.resolve(decoded, member, ref),)}
