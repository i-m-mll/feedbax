"""Validate portable checkpoint inputs for accepted Evaluate successes."""

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTree, ContentTreeFile, ContentTreeRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import DatasetDescriptor
from feedbax.contracts.effective_success import (
    EVALUATION_SUBJECT_MEDIA_TYPE, BoundMember, CheckpointPublicationSource,
    EvaluationSubjectValue, Success, TerminalTrainSubject,
    read_effective_request, read_success_object, read_success_request, select_success_product, success_content_ref,
)
from feedbax.contracts.base import ParentRef, Provenance
from feedbax.contracts.manifest import EvaluationRunManifest, EvaluationRunSpec, TrainingRunManifest, load_manifest_bytes, training_run_manifest_id
from feedbax.contracts.request_identity import DecodedRequest
from feedbax.contracts.request_refs import BatchMember, BatchSetMember, BestRank, ContentRef, InputMember, LastMember, ProductRef, RequestId, UpdateSeriesMember
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec, EvaluationRequestSpec
from feedbax.persistence.canonical_train import BoundStudioTrainingJobInput
from feedbax.contracts.rollout import ROLLOUT_TYPE, RolloutParameters
from feedbax.contracts.jobs import JobReceipt
from feedbax.persistence.canonical_evaluate import (
    decode_evaluation_manifest, evaluation_manifest_id, initial_evaluation_references, states_product, states_ref,
    verified_imported_subject, verify_initial_rollout,
)
from feedbax.persistence.array_container import open_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.checkpoint_publication import model_product, read_retained_checkpoint
from feedbax.persistence.method_checkpoint import parse_method_checkpoint_manifest
from feedbax.persistence.receipt_content import ClosureVisitors, success_content_objects, verify_success_content
from feedbax.persistence.best_selection import rank_train_metric


def read_terminal_train_subject(
    store: ImmutableArtifactBlobProvider, pin: TerminalTrainSubject, request_id: RequestId | None = None,
) -> tuple[Success, DocumentTrainingRequestSpec]:
    """Authenticate the exact terminal model and its authored Train documents."""
    success = read_success_object(pin.receipt, store.get_object)
    decoded = read_effective_request(success, store.get_object)
    request_id = request_id or decoded.id
    checkpoint = success.continuation
    if (
        decoded.id != request_id or not isinstance(decoded.spec, DocumentTrainingRequestSpec)
        or checkpoint is None or checkpoint.clock.completed_updates != decoded.spec.target_total
        or pin.model != select_success_product(ProductRef(
            request_id=request_id, role="checkpoint", select=LastMember(),
        ), success)
    ):
        raise ValueError("Pinned subject differs from the exact terminal Train model")
    verify_success_content(store, pin.model)
    return success, decoded.spec


def terminal_train_subject_at(
    store: ImmutableArtifactBlobProvider, receipt: ContentObjectRef, request_id: RequestId,
    *, attempt: str | None = None, selection: ProductRef | None = None,
) -> TerminalTrainSubject:
    """Pin the exact committed receipt and its selected model product."""
    success = read_success_object(receipt, store.get_object)
    if attempt is not None and attempt != success.attempt:
        raise ValueError("Pinned Train receipt names a different attempt")
    model = select_success_product(ProductRef(request_id=request_id, role="checkpoint", select=LastMember()), success)
    if selection is not None and select_success_product(selection, success) != model:
        raise ValueError(f"Train {request_id} checkpoint/model was not imported")
    pin = TerminalTrainSubject(receipt=receipt, model=model)
    read_terminal_train_subject(store, pin, request_id)
    return pin


def terminal_train_provenance(
    pin: TerminalTrainSubject, success: Success,
) -> tuple[list[ParentRef], Provenance]:
    checkpoint = success.continuation
    assert checkpoint is not None
    occurrence = checkpoint.provenance
    return [ParentRef(
        kind="Success", id=success.attempt, role="subject",
        metadata={"manifest_sha256": pin.receipt.sha256, "request_id": success.request.id},
    )], Provenance(parents=[ParentRef(
        kind="TrainingCheckpointTransactionManifest", id=occurrence.transaction_id, role="checkpoint",
        metadata={
            "manifest_sha256": occurrence.manifest_sha256,
            "completed_updates": checkpoint.clock.completed_updates,
            "producer_job_id": occurrence.run_id,
        },
    )])



def _verify_terminal(
    store: ImmutableArtifactBlobProvider, ref: ContentObjectRef, checkpoint: CheckpointPublicationSource,
    completed_updates: int | None,
) -> TrainingRunManifest:
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
    return receipt


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
            transaction, spec = read_retained_checkpoint(store, checkpoint, claim.producer)
            receipt = _verify_terminal(store, claim.receipt, checkpoint, spec.target_total)
            if transaction.clock.completed_updates != spec.target_total:
                raise ValueError("Best score is detached from its admitted Train execution")
            if receipt.inherited_terminal_checkpoint is not None:
                launch = BoundStudioTrainingJobInput.model_validate_json(store.get_object(checkpoint.train_binding))
                resume = launch.resume
                if (
                    resume is None or resume.completed_updates != spec.target_total
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
        closure, spec = read_retained_checkpoint(store, source.closure.checkpoint, source.producer)
        if source.closure.kind == "recovery" and any(
            gap.update > closure.clock.completed_updates for gap in value.gaps
        ):
            raise ValueError("Evaluate gaps extend beyond the recovery checkpoint")
        objects.extend((source.closure.checkpoint.transaction, source.closure.checkpoint.train_binding))
        if source.closure.receipt is not None:
            if closure.clock.completed_updates != spec.target_total:
                raise ValueError("Evaluate coverage completion differs from its authored horizon")
            _verify_terminal(store, source.closure.receipt, source.closure.checkpoint, closure.clock.completed_updates)
            objects.append(source.closure.receipt)

    models = []
    if source.terminal_checkpoint is not None:
        if (
            not isinstance(selector, LastMember) or source.terminal_receipt is None
            or source.checkpoints or source.ranked_candidates or resolved is None
        ):
            raise ValueError("Malformed terminal Train subject evidence")
        success = read_success_object(source.terminal_receipt, store.get_object)
        product = select_success_product(reference, success)
        pin = TerminalTrainSubject(receipt=source.terminal_receipt, model=product)
        success, _ = read_terminal_train_subject(store, pin, reference.request_id)
        occurrence = success.continuation
        (item,) = resolved.members
        if (
            occurrence is None or occurrence.provenance != source.terminal_checkpoint
            or source.producer != success.request
            or (occurrence.clock.completed_updates, product.content.sha256) != (item.coordinate, item.model_tree)
        ):
            raise ValueError("Evaluate terminal Train evidence differs from the resolved model")
        if visitors is not None:
            visitors.object(pin.receipt)
        models.append(success_content_ref(product.content))
        objects.extend((pin.receipt, *success_content_objects(store, product, visitors=visitors)))
    for checkpoint, item in zip(
        source.checkpoints, resolved.members if resolved and source.terminal_checkpoint is None else (), strict=True,
    ):
        if visitors is not None:
            visitors.object(checkpoint.transaction)
            visitors.object(checkpoint.train_binding)
            if source.terminal_receipt is not None:
                visitors.object(source.terminal_receipt)
        manifest, spec = read_retained_checkpoint(store, checkpoint, source.producer)
        product = model_product(manifest)
        if (manifest.clock.completed_updates, product.content.sha256) != (item.coordinate, item.model_tree):
            raise ValueError("Evaluate checkpoint differs from its resolved model and coordinate")
        if source.terminal_receipt is not None:
            if item.coordinate != spec.target_total:
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


def verify_initial_evaluation_states(
    store: ImmutableArtifactBlobProvider, manifest: EvaluationRunManifest, refs: dict[str, ContentRef],
) -> None:
    from math import isfinite

    member = verify_initial_rollout(store, manifest, refs)
    spec = EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline)
    params = RolloutParameters.model_validate(spec.params)
    states = states_ref(manifest)
    should_score = "objective" in refs
    with (
        store.open_owned_object(states.object_ref) as path,
        open_array_container(path, expected_descriptor_sha256=states.descriptor_sha256) as reader,
    ):
        descriptor = DatasetDescriptor.from_json_value(reader.descriptor.metadata)
        descriptor.validate_tensor_shapes({name: value.shape for name, value in reader.descriptor.tensors.items()})
        if (
            descriptor.schema_id != params.states_schema
            or set(descriptor.dimensions).intersection(("checkpoint", "update"))
            or (
                descriptor.coords["replicate"].inline_values if "replicate" in descriptor.coords else ()
            ) != (member.replica_ids if len(member.replica_ids) > 1 else ())
            or descriptor.coords["trial"].inline_values != member.trials
            or member.trials != params.trials
            or ("objective/total" in descriptor.variables) != should_score
            or ("objective/total" in manifest.summary_metrics) != should_score
        ):
            raise ValueError("Initial Evaluate states differ from the accepted document rollout")
        if should_score:
            totals = reader.read_tensor("objective/total").ravel()
            total = sum(float(value) for value in totals) / len(totals)
            if not isfinite(total) or manifest.summary_metrics["objective/total"] != total:
                raise ValueError("Initial Evaluate summary differs from the retained objective")


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
    if isinstance(subject, ContentRef):
        decoded = read_effective_request(success, store.get_object)
        if (
            success.products != {"states": (states_product(manifest),)}
            or manifest.metadata["canonical_request"] != success.request.object.model_dump(mode="json")
        ):
            raise ValueError("Accepted initial rollout differs from its Store success")
        refs = initial_evaluation_references(decoded)
        verify_initial_evaluation_states(store, manifest, refs)
        _, imported = verified_imported_subject(store, refs["subject"])
        if visitors is not None:
            for item in imported:
                visitors.object(item)
        return ref, *imported
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
        or len(manifest.provenance.parents) != len(source.checkpoints) + int(source.terminal_checkpoint is not None)
    ):
        raise ValueError("Accepted Evaluate manifest differs from the typed Store success")
    occurrences = (source.terminal_checkpoint,) if source.terminal_checkpoint is not None else tuple(
        checkpoint.occurrence for checkpoint in source.checkpoints
    )
    for parent, occurrence in zip(manifest.provenance.parents, occurrences, strict=True):
        if (
            parent.kind != "TrainingCheckpointTransactionManifest" or parent.id != occurrence.transaction_id
            or parent.metadata["manifest_sha256"] != occurrence.manifest_sha256
            or parent.metadata["producer_job_id"] != occurrence.run_id
            or (source.terminal_checkpoint is not None and parent.metadata["completed_updates"] != value.requested[0])
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
        publication = source.closure.checkpoint
        transaction = parse_method_checkpoint_manifest(store.get_object(publication.transaction), publication.pointer)
        if (
            settled["kind"] != source.closure.kind
            or settled["checkpoint"] != transaction.state_sha256
            or (source.closure.kind == "recovery" and settled["attempt"] != transaction.run_id)
        ):
            raise ValueError("Accepted Evaluate manifest names different closure evidence")
    terminal = source.terminal_receipt or (source.closure.receipt if source.closure else None)
    if terminal is not None and (
        len(manifest.input_training_runs) != 1
        or manifest.input_training_runs[0].metadata["manifest_sha256"] != terminal.sha256
        or (source.terminal_checkpoint is not None and (
            manifest.input_training_runs[0].kind != "Success"
            or manifest.input_training_runs[0].metadata["request_id"] != source.producer.id
        ))
    ):
        raise ValueError("Accepted Evaluate manifest names a different Train receipt")
    return tuple(evidence)
