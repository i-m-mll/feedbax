"""Validate portable checkpoint inputs for accepted Evaluate successes."""

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTree, ContentTreeFile, ContentTreeRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.checkpoints import CheckpointTransactionManifest
from feedbax.contracts.effective_success import (
    EVALUATION_SUBJECT_MEDIA_TYPE, BoundMember, EvaluationCheckpointSource,
    EvaluationSubjectValue, Product, RequestObject, Success, read_success_request, success_content_ref,
)
from feedbax.contracts.base import ParentRef
from feedbax.contracts.manifest import TrainingRunManifest, load_manifest_bytes, training_run_manifest_id
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE
from feedbax.contracts.request_identity import DecodedRequest
from feedbax.contracts.request_refs import BatchMember, BatchSetMember, BestRank, ContentRef, InputMember, LastMember, ProductRef, UpdateSeriesMember
from feedbax.contracts.request_specs import EvaluationRequestSpec
from feedbax.contracts.spec_storage import canonical_training_run_spec_projection
from feedbax.contracts.studio_training import StudioGraphMethodPayload
from feedbax.contracts.training import TrainingRunSpec
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.contracts.rollout import ROLLOUT_TYPE, RolloutParameters
from feedbax.contracts.jobs import JobReceipt
from feedbax.studio.canonical_evaluate import decode_evaluation_manifest, evaluation_manifest_id, states_product
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.receipt_content import ClosureVisitors, success_content_objects
from feedbax.host.best_selection import rank_train_metric


def checkpoint_transaction(
    store: ImmutableArtifactBlobProvider, source: EvaluationCheckpointSource,
) -> CheckpointTransactionManifest:
    manifest = CheckpointTransactionManifest.model_validate_json(store.get_object(source.transaction))
    if (manifest.run_id, manifest.transaction_id) != (source.occurrence.run_id, source.occurrence.transaction_id):
        raise ValueError("Evaluate checkpoint transaction has a different occurrence")
    if manifest.barrier != "after_train_batch":
        raise ValueError("Evaluate checkpoint transaction has a different barrier")
    return manifest


def _verify_train_contract(
    store: ImmutableArtifactBlobProvider, checkpoint: EvaluationCheckpointSource,
    manifest: CheckpointTransactionManifest, request: DecodedRequest, producer: RequestObject,
) -> StudioGraphMethodPayload:
    launch = BoundStudioTrainingJobInput.model_validate_json(store.get_object(checkpoint.train_binding))
    binding = launch.canonical_request
    spec = request.spec
    authored_method = spec.method_payload.payload
    if spec.seed is not None:
        authored_method = authored_method | {"seed": spec.seed}
    projection = manifest.run_contract_binding.canonical_projection
    if (
        binding is None or launch.request_id != producer.id
        or (binding.request, binding.documents) != (producer.object, producer.documents)
        or projection is None
        or projection["training_run_spec"] != canonical_training_run_spec_projection(launch.spec)
        or launch.spec.trial_documents != spec.trial_documents
        or launch.spec.method_ref != spec.method_ref or launch.spec.method_payload.payload != authored_method
    ):
        raise ValueError("Evaluate checkpoint does not match its authenticated Train definition")
    return StudioGraphMethodPayload.model_validate(launch.spec.method_payload.payload)


def model_product(manifest: CheckpointTransactionManifest) -> Product:
    (slot,) = (slot for slot in manifest.slots if slot.slot == "model" and slot.media_type == MODEL_TREE_MEDIA_TYPE)
    return Product(
        name="model", content=ContentRef(
            sha256=slot.sha256, size=slot.size_bytes, media_type=slot.media_type, tree=True,
        ), descriptor=slot.metadata["semantic_model"],
    )


def _verify_terminal(
    store: ImmutableArtifactBlobProvider, ref: ContentObjectRef, checkpoint: EvaluationCheckpointSource,
    completed_updates: int | None,
) -> None:
    receipt = load_manifest_bytes(store.get_object(ref))
    occurrence = checkpoint.occurrence
    if not isinstance(receipt, TrainingRunManifest) or receipt.job_id is None:
        raise ValueError("Evaluate terminal evidence does not name a Train receipt")
    terminal = receipt.terminal_checkpoint
    if (
        receipt.id != training_run_manifest_id(receipt.job_id)
        or receipt.status != "completed" or receipt.stopped or receipt.completed_batches != completed_updates
        or not isinstance(terminal, ParentRef) or terminal.id != occurrence.transaction_id
        or terminal.metadata.get("manifest_sha256") != occurrence.manifest_sha256
        or (receipt.inherited_terminal_checkpoint is None and receipt.job_id != occurrence.run_id)
    ):
        raise ValueError("Evaluate terminal evidence differs from its exact checkpoint")


def subject_tree(
    store: ImmutableArtifactBlobProvider, value: ContentObjectRef, models: tuple[ContentTreeRef, ...],
) -> ContentTree:
    files = [ContentTreeFile(relative_path="binding.json", object=value)]
    directories = {"models"} if models else set()
    for index, model in enumerate(models):
        prefix = f"models/{index:03}"
        tree = store.get_tree(model)
        directories.update((prefix, f"{prefix}/content"))
        directories.update(f"{prefix}/content/{directory}" for directory in tree.directories)
        files.append(ContentTreeFile(relative_path=f"{prefix}/tree.json", object=model.descriptor))
        files.extend(
            ContentTreeFile(relative_path=f"{prefix}/content/{file.relative_path}", object=file.object)
            for file in tree.files
        )
    return ContentTree(directories=tuple(sorted(directories)), files=tuple(sorted(files, key=lambda file: file.relative_path)))


def evaluation_subject_objects(
    store: ImmutableArtifactBlobProvider, request: DecodedRequest, member: InputMember, bound: BoundMember,
    *, visitors: ClosureVisitors | None = None,
) -> tuple[ContentObjectRef, ...]:
    """Verify a frozen Evaluate projection without selecting a current Train result."""
    source, reference = bound.evaluation_subject, member.reference
    if (
        source is None or bound.key != member.key
        or not isinstance(request.spec, EvaluationRequestSpec) or request.spec.evaluation_type != ROLLOUT_TYPE
        or bound.content.media_type != EVALUATION_SUBJECT_MEDIA_TYPE or not bound.content.tree
    ):
        raise ValueError("Malformed managed Evaluate subject")
    objects = [source.producer.object, *source.producer.documents]
    if isinstance(reference, BestRank):
        scores = []
        for candidate, claim in zip(reference.candidates, source.ranked_candidates, strict=True):
            if claim.producer.id != candidate.reference.request_id:
                raise ValueError("Best receipt lists a different candidate")
            checkpoint = claim.checkpoint
            refs = (
                claim.receipt, checkpoint.transaction, checkpoint.train_binding,
                claim.producer.object, *claim.producer.documents,
            )
            if visitors is not None:
                for ref in refs:
                    visitors.object(ref)
            raw = store.get_object(claim.receipt)
            scores.append(rank_train_metric(raw, JobReceipt(
                manifest_id=training_run_manifest_id(claim.attempt), manifest_sha256=claim.receipt.sha256,
            ), claim.attempt, reference.metric))
            producer_request = read_success_request(claim.producer, store.get_object)
            transaction = checkpoint_transaction(store, checkpoint)
            method = _verify_train_contract(store, checkpoint, transaction, producer_request, claim.producer)
            _verify_terminal(store, claim.receipt, checkpoint, method.target_total)
            receipt = load_manifest_bytes(raw)
            launch = BoundStudioTrainingJobInput.model_validate_json(store.get_object(checkpoint.train_binding))
            if (
                not isinstance(receipt, TrainingRunManifest) or receipt.training_spec is None
                or TrainingRunSpec.model_validate(receipt.training_spec.inline) != launch.spec
                or transaction.completed_training_batches != method.target_total
            ):
                raise ValueError("Best score is detached from its admitted Train execution")
            if receipt.inherited_terminal_checkpoint is not None:
                resume = launch.resume
                if (
                    resume is None or resume.completed_updates != method.target_total
                    or (resume.pointer.run_id, resume.pointer.transaction_id, resume.pointer.manifest_sha256)
                    != (checkpoint.occurrence.run_id, checkpoint.occurrence.transaction_id, checkpoint.occurrence.manifest_sha256)
                ):
                    raise ValueError("Best receipt inherited a different terminal checkpoint")
            objects.extend(refs)
        winner = reference.winner(scores)
        if source.terminal_receipt != source.ranked_candidates[winner].receipt:
            raise ValueError("Evaluate receipt does not retain the ranked producer's terminal receipt")
        reference = reference.candidates[winner].reference
    elif source.ranked_candidates:
        raise ValueError("Fixed Evaluate subject contains Best evidence")
    if not isinstance(reference, ProductRef) or reference.role != "checkpoint" or source.producer.id != reference.request_id:
        raise ValueError("Evaluate receipt names a different Train request")
    if visitors is not None:
        for ref in objects:
            visitors.object(ref)
    producer = read_success_request(source.producer, store.get_object)
    if producer.kind != "train":
        raise ValueError("Evaluate subject does not name a Train request")
    value = EvaluationSubjectValue.model_validate(bound.descriptor)
    params = RolloutParameters.model_validate(request.spec.params)
    selector = reference.select
    if isinstance(selector, UpdateSeriesMember):
        requested, axis = selector.updates, "update"
    elif isinstance(selector, BatchSetMember):
        requested, axis = selector.batches, "checkpoint"
    elif isinstance(selector, BatchMember):
        requested, axis = (selector.batch,), "checkpoint"
    elif isinstance(selector, LastMember) and len(value.requested) == 1:
        requested, axis = value.requested, "checkpoint"
    else:
        raise ValueError("Evaluate subject has an unsupported checkpoint selector")
    if (
        (value.requested, value.axis) != (requested, axis) or bool(source.closure) != bool(value.gaps)
        or (source.terminal_receipt is not None) != isinstance(selector, LastMember)
    ):
        raise ValueError("Evaluate coverage differs from the authored selector")
    resolved = value.resolved
    if resolved is not None and (
        resolved.seed != params.seed or resolved.trial_policy != params.trial_policy
        or any(item.trials != params.trials for item in resolved.members)
    ):
        raise ValueError("Evaluate trial binding differs from its authored parameters")

    if source.closure is not None:
        if visitors is not None:
            for ref in (source.closure.checkpoint.transaction, source.closure.checkpoint.train_binding):
                visitors.object(ref)
            if source.closure.receipt is not None:
                visitors.object(source.closure.receipt)
        closure = checkpoint_transaction(store, source.closure.checkpoint)
        method = _verify_train_contract(store, source.closure.checkpoint, closure, producer, source.producer)
        if source.closure.kind == "recovery" and any(
            gap.update > closure.completed_training_batches for gap in value.gaps
        ):
            raise ValueError("Evaluate gaps extend beyond the recovery checkpoint")
        objects.extend((source.closure.checkpoint.transaction, source.closure.checkpoint.train_binding))
        if source.closure.receipt is not None:
            if closure.completed_training_batches != method.target_total:
                raise ValueError("Evaluate coverage completion differs from its authored horizon")
            _verify_terminal(store, source.closure.receipt, source.closure.checkpoint, closure.completed_training_batches)
            objects.append(source.closure.receipt)

    models = []
    for checkpoint, item in zip(source.checkpoints, resolved.members if resolved else (), strict=True):
        if visitors is not None:
            visitors.object(checkpoint.transaction)
            visitors.object(checkpoint.train_binding)
            if source.terminal_receipt is not None:
                visitors.object(source.terminal_receipt)
        manifest = checkpoint_transaction(store, checkpoint)
        method = _verify_train_contract(store, checkpoint, manifest, producer, source.producer)
        product = model_product(manifest)
        if (manifest.completed_training_batches, product.content.sha256) != (item.coordinate, item.model_tree):
            raise ValueError("Evaluate checkpoint differs from its resolved model and coordinate")
        if source.terminal_receipt is not None:
            if item.coordinate != method.target_total:
                raise ValueError("Evaluate terminal checkpoint differs from its authored horizon")
            _verify_terminal(store, source.terminal_receipt, checkpoint, item.coordinate)
            objects.append(source.terminal_receipt)
        model_objects = success_content_objects(store, product, visitors=visitors)
        models.append(success_content_ref(product.content))
        objects.extend((checkpoint.transaction, checkpoint.train_binding, *model_objects))

    tree_ref = success_content_ref(bound.content)
    if visitors is not None:
        visitors.object(tree_ref.descriptor)
    tree = store.get_tree(tree_ref)
    descriptor = tree.files[0].object
    if visitors is not None:
        visitors.object(descriptor)
    if (
        tree != subject_tree(store, descriptor, tuple(models))
        or store.get_object(descriptor) != canonical_json_v2_bytes(value.model_dump(mode="json"))
    ):
        raise ValueError("Evaluate subject tree differs from the resolved inputs")
    return (tree_ref.descriptor, descriptor, *objects)


def verify_evaluation_success_evidence(
    store: ImmutableArtifactBlobProvider, success: Success, *, visitors: ClosureVisitors | None = None,
    verify_bound_closure: bool = True,
) -> tuple[ContentObjectRef, ...]:
    ref = success.evaluation_manifest
    assert ref is not None
    manifest, subject = decode_evaluation_manifest(
        store, store.get_object(ref), ref, success.attempt, success.request.id,
        JobReceipt(manifest_id=evaluation_manifest_id(success.attempt), manifest_sha256=ref.sha256),
    )
    (bound,) = success.inputs["subject"]
    source = bound.evaluation_subject
    assert source is not None
    value = EvaluationSubjectValue.model_validate(bound.descriptor)
    expected_binding = value.resolved.model_dump(mode="json") if value.resolved else None
    expected_digest = value.resolved.digest() if value.resolved else None
    if (
        success.products != {"states": (states_product(manifest),)}
        or manifest.metadata["canonical_request"] != success.request.object.model_dump(mode="json")
        or manifest.metadata.get("effective_binding") != expected_binding
        or manifest.metadata.get("effective_binding_sha256") != expected_digest
        or len(manifest.provenance.parents) != len(source.checkpoints)
    ):
        raise ValueError("Accepted Evaluate manifest differs from the typed Store success")
    for parent, checkpoint in zip(manifest.provenance.parents, source.checkpoints, strict=True):
        occurrence = checkpoint.occurrence
        if (
            parent.kind != "TrainingCheckpointTransactionManifest" or parent.id != occurrence.transaction_id
            or parent.metadata["manifest_sha256"] != occurrence.manifest_sha256
            or parent.metadata["producer_job_id"] != occurrence.run_id
        ):
            raise ValueError("Accepted Evaluate manifest names a different checkpoint occurrence")
    coverage = manifest.provenance.metadata.get("checkpoint_selection")
    if coverage is not None and (
        coverage["requested"] != list(value.requested)
        or coverage.get("gaps", []) != [gap.model_dump(mode="json") for gap in value.gaps]
    ):
        raise ValueError("Accepted Evaluate manifest names different checkpoint coverage")
    evidence = [ref]
    best = manifest.provenance.metadata.get("best_rank")
    if isinstance(subject, BestRank):
        if (
            not isinstance(best, dict)
            or (best.get("rank"), best.get("key")) != (subject.rank, subject.key)
            or len(source.ranked_candidates) != len(subject.candidates)
        ):
            raise ValueError("Accepted Evaluate manifest lacks its ranked candidate evidence")
        for evidence_item, claim in zip(best["candidates"], source.ranked_candidates, strict=True):
            if (
                evidence_item["attempt"], evidence_item["request_id"],
                evidence_item["receipt_sha256"], evidence_item["binding_sha256"],
            ) != (claim.attempt, claim.producer.id, claim.receipt.sha256, claim.checkpoint.train_binding.sha256):
                raise ValueError("Accepted Evaluate manifest names different ranked candidate evidence")
    elif best is not None or source.ranked_candidates:
        raise ValueError("Fixed Evaluate manifest contains ranked candidate evidence")
    if source.closure is not None and verify_bound_closure:
        evidence.append(source.closure.checkpoint.transaction)
        if visitors is not None:
            visitors.object(source.closure.checkpoint.transaction)
        settled = coverage["closure"]
        transaction = checkpoint_transaction(store, source.closure.checkpoint)
        if (
            settled["kind"] != source.closure.kind
            or settled["checkpoint"] != transaction.content_integrity_digest.transaction_root_sha256
            or (source.closure.kind == "recovery" and settled["attempt"] != transaction.run_id)
        ):
            raise ValueError("Accepted Evaluate manifest names different closure evidence")
    terminal = source.terminal_receipt or (source.closure.receipt if source.closure else None)
    if terminal is not None and (
        len(manifest.input_training_runs) != 1
        or manifest.input_training_runs[0].metadata["manifest_sha256"] != terminal.sha256
    ):
        raise ValueError("Accepted Evaluate manifest names a different Train receipt")
    return tuple(evidence)
