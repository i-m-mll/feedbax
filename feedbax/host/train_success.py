"""Commit the typed model product of an accepted managed Train attempt."""

from hashlib import sha256
from pathlib import Path, PurePosixPath
import re

from feedbax.contracts.artifact_custody import ContentTreeRef
from feedbax.contracts.effective_success import (
    Bindings, BoundMember, Product, RequestObject, Selection, select_success_product, success_content_ref,
)
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE
from feedbax.contracts.request_refs import ProductRef
from feedbax.contracts.strict_json import strict_json_loads
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
    store = ImmutableArtifactBlobProvider(store_root)
    with store._exclusive_lock():
        view = SuccessView(store)
        bindings = _train_inputs(view, request, predecessor)
        if not view.has_attempt(attempt):
            source = ImmutableArtifactBlobProvider(model_root)
            files = {file["path"]: (file["sha256"], file["size_bytes"]) for file in completion["files"]}
            _check_model_slot(store_root, checkpoint_root / relative, pointer, product)
            for ref in success_content_objects(source, product):
                path = source.object_path(ref)
                if files.get(path.relative_to(store_root).as_posix()) != (ref.sha256, ref.size_bytes):
                    raise ValueError("Model tree object is absent from the accepted checkpoint file evidence")
                store._store_file(path, sha256=ref.sha256, size_bytes=ref.size_bytes)
        published = view.publish_bound(request, attempt=attempt, products={"checkpoint": (product,)}, bindings=bindings)
    return published.ref.model_dump(mode="json")


def _check_model_slot(root: Path, manifest_path: Path, pointer: dict, product: Product) -> None:
    with completion_file(root, manifest_path.relative_to(root).as_posix()) as (stream, _):
        raw = stream.read()
    if sha256(raw).hexdigest() != pointer["manifest_sha256"]:
        raise ValueError("Train checkpoint manifest changed before Store publication")
    manifest = strict_json_loads(raw)
    (slot,) = (slot for slot in manifest["slots"] if slot["slot"] == "model")
    content = product.content
    if (
        manifest["transaction_id"] != pointer["transaction_id"]
        or (slot["role"], slot["media_type"], slot["sha256"], slot["size_bytes"])
        != ("model", content.media_type, content.sha256, content.size)
        or slot["metadata"]["semantic_model"] != product.descriptor
    ):
        raise ValueError("Train model product differs from the authenticated checkpoint slot")


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
    parent = view.read_attempt(completion["producer_job_id"])
    if parent is None or outcome.get("store_success") != parent.ref.model_dump(mode="json"):
        raise ValueError("Pinned predecessor lacks its committed typed model receipt")
    model = select_success_product(reference, parent.receipt)
    if model.model_dump(mode="json") != outcome.get("model_product"):
        raise ValueError("Pinned predecessor typed product differs from the accepted Train outcome")
    return {"checkpoint": (BoundMember(
        key=member.key, content=model.content, descriptor=model.descriptor,
        selection=Selection(receipt=parent.ref, member=model.name),
    ),)}
