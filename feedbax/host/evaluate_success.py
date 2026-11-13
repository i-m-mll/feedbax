"""Publish the typed success for one accepted managed Evaluate delivery."""

from pathlib import Path

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.checkpoint_projection import CheckpointOccurrenceRef
from feedbax.contracts.effective_success import (
    EVALUATION_SUBJECT_MEDIA_TYPE, BoundMember, EvaluationCheckpointSource, EvaluationClosure,
    EvaluationSubjectSource, EvaluationSubjectValue, RequestObject, success_content_ref,
)
from feedbax.contracts.dataset import DatasetDescriptor
from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.manifest import canonical_manifest_path
from feedbax.contracts.request_refs import ContentRef, UpdateSeriesMember
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host.checkpoint_selection import verify_dependency_files
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.array_container import open_array_container
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.evaluation_subject import checkpoint_transaction, model_product, subject_tree
from feedbax.persistence.receipt_content import success_content_objects
from feedbax.studio.canonical_evaluate import (
    BoundEvaluationJobInput, RolloutParameters, evaluation_checkpoint_coordinates,
    states_product, states_ref, verify_evaluation_receipt,
)
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput


def _retain_checkpoint(
    store: ImmutableArtifactBlobProvider, checkpoint: PublishedTrainingCheckpoint, publication: dict,
) -> EvaluationCheckpointSource:
    evidence = _files(publication)
    pointer = checkpoint.pointer
    path = checkpoint.checkpoint_root / pointer.manifest_relative_path
    digest, size = evidence[path.relative_to(store.root).as_posix()]
    if digest != pointer.manifest_sha256:
        raise ValueError("Checkpoint transaction is absent from the exact Host binding")
    return EvaluationCheckpointSource(
        occurrence=CheckpointOccurrenceRef(
            run_id=pointer.run_id, transaction_id=pointer.transaction_id, manifest_sha256=digest,
        ),
        transaction=store._store_file(path, sha256=digest, size_bytes=size),
        train_binding=_retain_train_binding(store, publication),
    )


def publish_evaluate_success(
    *, attempt: str, request_id: str, project: str, payload: dict, receipt: dict, binding: dict,
) -> ContentObjectRef:
    launch = BoundEvaluationJobInput.model_validate(payload)
    if (launch.project_id, launch.request_id) != (project, request_id):
        raise ValueError("Evaluate launch differs from the accepted attempt")
    root = launch.manifest_root
    store = ImmutableArtifactBlobProvider(root)
    worker_receipt = JobReceipt.model_validate(receipt)
    with store._exclusive_lock():
        view = SuccessView(store)
        existing = view.read_attempt(attempt)
        manifest_object = existing.receipt.evaluation_manifest if existing is not None else None
        if existing is not None and manifest_object is None:
            raise ValueError("Published Evaluate attempt lacks its accepted worker receipt")
        manifest, subject = verify_evaluation_receipt(
            root, attempt=attempt, request_id=request_id, project=project, receipt=worker_receipt, binding=binding,
            manifest_object=manifest_object,
        )
        if manifest_object is None:
            manifest_ref, _ = store.resolve_manifest(manifest.kind, manifest.id)
            manifest_object = manifest_ref.object
        if (
            manifest.metadata["canonical_request"] != launch.canonical_request.request.model_dump(mode="json")
            or subject != launch.canonical_request.checkpoint
        ):
            raise ValueError("Evaluate receipt differs from its admitted canonical request")
        request = RequestObject(id=request_id, object=launch.canonical_request.request, documents=())
        states = states_ref(manifest)
        product = states_product(manifest)
        decoded = view.read_request(request)
        (member,) = decoded.inputs["subject"]
        if existing is not None:
            saved = existing.receipt
            if (
                saved.request != request or saved.evaluation_manifest != manifest_object
                or saved.products != {"states": (product,)}
            ):
                raise ValueError("Published Evaluate attempt differs from its accepted result")
            return existing.ref

        verify_dependency_files(binding)
        closure = None
        terminal_receipt = None
        if (coordinates := evaluation_checkpoint_coordinates(subject)) is not None:
            selected = binding["checkpoint_selection"]
            publications = selected["members"]
            if settled := selected.get("closure"):
                data = settled["product"]
                checkpoint = PublishedTrainingCheckpoint.model_validate(data["checkpoint"])
                closure = EvaluationClosure(
                    kind=settled["kind"], checkpoint=_retain_checkpoint(store, checkpoint, data),
                    receipt=_retain_train_receipt(store, data) if settled["kind"] == "completion" else None,
                )
        else:
            completion = binding["completion"]
            terminal_receipt = _retain_train_receipt(store, completion)
            checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
            coordinates = (checkpoint.completed_updates,)
            publications = (completion,)
            selected = {}
        resolved = manifest.metadata.get("effective_binding")
        value = EvaluationSubjectValue(
            axis="update" if isinstance(subject.select, UpdateSeriesMember) else "checkpoint",
            requested=coordinates, gaps=selected.get("gaps", ()), resolved=resolved,
        )
        if value.resolved is not None and value.resolved.digest() != manifest.metadata.get("effective_binding_sha256"):
            raise ValueError("Evaluate resolved-input digest differs from its worker receipt")
        if value.resolved is None and "effective_binding_sha256" in manifest.metadata:
            raise ValueError("Evaluate receipt has a digest without resolved inputs")
        params = RolloutParameters.model_validate(decoded.spec.params)
        observed = [item.coordinate for item in value.resolved.members] if value.resolved else []
        with (
            store.open_owned_object(states.object_ref) as path,
            open_array_container(path, expected_descriptor_sha256=states.descriptor_sha256) as reader,
        ):
            descriptor = DatasetDescriptor.from_json_value(reader.descriptor.metadata)
            descriptor.validate_tensor_shapes({name: value.shape for name, value in reader.descriptor.tensors.items()})
            if (
                descriptor.schema_id != params.states_schema or reader.read_tensor(value.axis).tolist() != observed
                or (observed and descriptor.coords["trial"].inline_values != params.trials)
                or (not observed and descriptor.variables)
            ):
                raise ValueError("Evaluate states differ from accepted checkpoint coverage or trials")

        checkpoints, trees = [], []
        producer = None
        for publication in publications:
            checkpoint = PublishedTrainingCheckpoint.model_validate(publication["checkpoint"])
            pinned = _retain_checkpoint(store, checkpoint, publication)
            current = _producer(store, publication, pinned.train_binding)
            if producer is not None and current != producer:
                raise ValueError("Evaluate series changes the exact Train definition")
            producer = current
            files = _files(publication)
            transaction = checkpoint_transaction(store, pinned)
            if (
                transaction.completed_training_batches != checkpoint.completed_updates
                or transaction.content_integrity_digest.transaction_root_sha256 != checkpoint.pointer.transaction_root_sha256
                or transaction.completed_coordinate != checkpoint.pointer.completed_coordinate
            ):
                raise ValueError("Evaluate checkpoint differs from its admitted occurrence")
            model_slot = model_product(transaction)
            tree = success_content_ref(model_slot.content)
            source = ImmutableArtifactBlobProvider(
                checkpoint.checkpoint_root / Path(checkpoint.pointer.manifest_relative_path).parent / "semantic-model",
            )
            for ref in success_content_objects(source, model_slot):
                path = source.object_path(ref)
                if files.get(path.relative_to(root).as_posix()) != (ref.sha256, ref.size_bytes):
                    raise ValueError("Evaluate model object is absent from its exact Host binding")
                store._store_file(path, sha256=ref.sha256, size_bytes=ref.size_bytes)
            checkpoints.append(pinned)
            trees.append(tree)
        if closure is not None:
            closing = _producer(store, selected["closure"]["product"], closure.checkpoint.train_binding)
            if producer is not None and producer != closing:
                raise ValueError("Evaluate closure changes the exact Train definition")
            producer = closing
        if producer is None:
            raise ValueError("Evaluate result has no authenticated Train subject")
        value_ref = store.store_object(canonical_json_v2_bytes(value.model_dump(mode="json")))
        tree = store.store_tree(subject_tree(store, value_ref, tuple(trees)))
        bound = BoundMember(
            key=member.key, content=ContentRef(
                sha256=tree.descriptor.sha256, size=tree.descriptor.size_bytes,
                media_type=EVALUATION_SUBJECT_MEDIA_TYPE, tree=True,
            ),
            descriptor=value.model_dump(mode="json"),
            evaluation_subject=EvaluationSubjectSource(
                producer=producer, checkpoints=tuple(checkpoints), closure=closure,
                terminal_receipt=terminal_receipt,
            ),
        )
        result = view.publish_bound(
            request, attempt=attempt, products={"states": (product,)}, bindings={"subject": (bound,)},
            evaluation_manifest=manifest_object,
        )
        return result.ref


def _retain_train_receipt(store: ImmutableArtifactBlobProvider, publication: dict) -> ContentObjectRef:
    receipt = JobReceipt.model_validate(publication["receipt"])
    path = canonical_manifest_path("TrainingRunManifest", receipt.manifest_id, root=store.root)
    digest, size = _files(publication)[path.relative_to(store.root).as_posix()]
    if digest != receipt.manifest_sha256:
        raise ValueError("Train receipt is absent from the exact Host binding")
    return store._store_file(path, sha256=digest, size_bytes=size)


def _files(publication: dict) -> dict[str, tuple[str, int]]:
    return {file["path"]: (file["sha256"], file["size_bytes"]) for file in publication["files"]}


def _retain_train_binding(store: ImmutableArtifactBlobProvider, publication: dict) -> ContentObjectRef:
    relative = f"checkpoints/{publication['producer_job_id']}/binding.json"
    digest, size = _files(publication)[relative]
    if digest != publication["binding_sha256"]:
        raise ValueError("Train binding is absent from the exact Host evidence")
    return store._store_file(store.root / relative, sha256=digest, size_bytes=size)


def _producer(store: ImmutableArtifactBlobProvider, publication: dict, ref: ContentObjectRef) -> RequestObject:
    launch = BoundStudioTrainingJobInput.model_validate_json(store.get_object(ref))
    if (
        launch.canonical_request is None or launch.request_id != publication["request_id"]
        or launch.project_id != publication["project_id"] or launch.manifest_root != store.root
    ):
        raise ValueError("Checkpoint lacks its canonical Train request")
    canonical = launch.canonical_request
    return RequestObject(id=launch.request_id, object=canonical.request, documents=canonical.documents)
