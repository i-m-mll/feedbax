"""Publish the typed success for one accepted managed Evaluate delivery."""

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.effective_success import (
    EVALUATION_SUBJECT_MEDIA_TYPE, BestCandidateEvidence, BoundMember, EvaluationClosure,
    EvaluationSubjectSource, EvaluationSubjectValue, RequestObject, Success, TerminalTrainSubject, success_content_ref,
)
from feedbax.contracts.dataset import DatasetDescriptor
from feedbax.contracts.execution_target import AttemptTarget, LOCAL_TARGET
from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.live_evaluate import (
    BoundLiveEvaluationPoint, ScoredEvaluationPoint, require_evaluation_coverage, rollout_slot,
)
from feedbax.contracts.manifest import canonical_manifest_path
from feedbax.contracts.request_refs import ContentRef, ProductRef, UpdateSeriesMember, is_synthesized_model_product
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.persistence.checkpoint_selection import verify_dependency_files
from feedbax.host.model_checkpoint_input import checkpoint_producer, retain_checkpoint, retain_checkpoint_model
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.array_container import open_array_container
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.evaluation_subject import (
    read_terminal_train_subject, subject_tree, verify_initial_evaluation_states,
)
from feedbax.persistence.canonical_evaluate import (
    BoundEvaluationJobInput, RolloutParameters, evaluation_checkpoint_coordinates,
    exact_evaluation_subject, states_product, states_ref, verify_evaluation_receipt,
)
from feedbax.persistence.canonical_train import BoundStudioTrainingJobInput
from feedbax.persistence.evidence import read_execution_evidence, release_staged_evidence, verify_execution_replay


def publish_evaluate_success(
    *, attempt: str, request_id: str, project: str, payload: dict, receipt: dict, binding: dict,
    target: AttemptTarget = LOCAL_TARGET,
    execution_evidence: ContentObjectRef | None = None,
) -> ContentObjectRef:
    launch = BoundEvaluationJobInput.model_validate(payload)
    if (launch.project_id, launch.request_id) != (project, request_id):
        raise ValueError("Evaluate launch differs from the accepted attempt")
    root = launch.manifest_root
    store = ImmutableArtifactBlobProvider(root)
    worker_receipt = JobReceipt.model_validate(receipt)
    with store._exclusive_lock():
        view = SuccessView(store)
        committed = view.attempt_ref(attempt)
        existing = view.read_owned_success(committed) if committed is not None else None
        if existing is None:
            manifest_ref, _ = store.resolve_manifest("EvaluationRunManifest", worker_receipt.manifest_id)
            manifest_object = manifest_ref.object
        else:
            manifest_object = existing.receipt.evaluation_manifest
            if manifest_object is None:
                raise ValueError("Published Evaluate attempt lacks its accepted worker receipt")
        manifest, subject = verify_evaluation_receipt(
            root, attempt=attempt, request_id=request_id, project=project, receipt=worker_receipt, binding=binding,
            manifest_object=manifest_object,
        )
        if (
            manifest.metadata["canonical_request"] != launch.canonical_request.request.model_dump(mode="json")
            or subject != launch.canonical_request.subject
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
                saved.request != request or saved.evaluation_manifest != manifest_object or saved.target != target
                or saved.products != {"states": (product,)}
            ):
                raise ValueError("Published Evaluate attempt differs from its accepted result")
            verify_execution_replay(saved, execution_evidence)
            view.ensure_published_run_ref(existing.ref, existing.receipt)
            _release_adopted_points(store, saved, binding)
            return existing.ref
        if execution_evidence is None:
            raise ValueError("Managed Evaluate success lacks its execution evidence")
        trace = read_execution_evidence(store, execution_evidence).trace
        computation = trace.computation or trace.computation_unavailable
        if isinstance(subject, ContentRef):
            from feedbax.persistence.canonical_evaluate import initial_evaluation_references

            verify_initial_evaluation_states(store, manifest, initial_evaluation_references(decoded))
            replicas = manifest.loaded_model.replicas if manifest.loaded_model else ("0",)
            require_evaluation_coverage(computation, tuple(rollout_slot(None, replica) for replica in replicas))
            _, bindings = view.bind(request, origins={})
            return view.publish_bound(
                request, attempt=attempt, products={"states": (product,)}, bindings=bindings,
                evaluation_manifest=manifest_object, target=target,
                execution_evidence=execution_evidence,
            ).ref

        if is_synthesized_model_product(subject):
            from feedbax.persistence.canonical_evaluate import SynthesizedSubjectBinding

            source = SynthesizedSubjectBinding.model_validate(binding["model_subject"])
            require_evaluation_coverage(computation, (rollout_slot(None, "0"),))
            _, bindings = view.bind(request, origins={subject.request_id: source.receipt})
            return view.publish_bound(
                request, attempt=attempt, products={"states": (product,)}, bindings=bindings,
                evaluation_manifest=manifest_object, target=target,
                execution_evidence=execution_evidence,
            ).ref

        closure = None
        terminal_receipt = None
        terminal_checkpoint = None
        selected = {}
        publications = ()
        checkpoints, trees = [], []
        producer = None
        if "terminal" in binding:
            if not isinstance(subject, ProductRef):
                raise ValueError("A ranked Evaluate cannot publish a terminal Store subject")
            pin = TerminalTrainSubject.model_validate(binding["terminal"])
            if launch.canonical_request.terminal not in (None, pin):
                raise ValueError("Evaluate Store subject differs from its exact admitted pin")
            train, _ = read_terminal_train_subject(store, pin, subject.request_id)
            assert train.continuation is not None
            producer = train.request
            terminal_receipt = pin.receipt
            terminal_checkpoint = train.continuation.provenance
            coordinates = (train.continuation.clock.completed_updates,)
            trees.append(success_content_ref(pin.model.content))
        elif (coordinates := evaluation_checkpoint_coordinates(subject)) is not None:
            selected = binding["checkpoint_selection"]
            publications = selected["members"]
            if settled := selected.get("closure"):
                data = settled["product"]
                checkpoint = PublishedTrainingCheckpoint.model_validate(data["checkpoint"])
                closure = EvaluationClosure(
                    kind=settled["kind"], checkpoint=retain_checkpoint(store, checkpoint, data),
                    receipt=_retain_train_receipt(store, data) if settled["kind"] == "completion" else None,
                )
        else:
            completion = binding["completion"]
            terminal_receipt = _retain_train_receipt(store, completion)
            checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
            coordinates = (checkpoint.completed_updates,)
            publications = (completion,)
            selected = {}
        if terminal_checkpoint is None:
            verify_dependency_files(binding)
            exact_evaluation_subject(subject, binding, verify_metrics=True)
        resolved = manifest.metadata.get("effective_binding")
        value = EvaluationSubjectValue(
            axis="update" if isinstance(subject, ProductRef) and isinstance(subject.select, UpdateSeriesMember) else "checkpoint",
            requested=coordinates, gaps=selected.get("gaps", ()), resolved=resolved,
        )
        if value.resolved is not None and value.resolved.digest() != manifest.metadata.get("effective_binding_sha256"):
            raise ValueError("Evaluate resolved-input digest differs from its worker receipt")
        if value.resolved is None and "effective_binding_sha256" in manifest.metadata:
            raise ValueError("Evaluate receipt has a digest without resolved inputs")
        members = value.resolved.members if value.resolved is not None else ()
        require_evaluation_coverage(computation, tuple(
            rollout_slot(item.coordinate, replica)
            for item in members
            for replica in item.replica_ids or ("0",)
        ))
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

        for publication in publications:
            checkpoint = PublishedTrainingCheckpoint.model_validate(publication["checkpoint"])
            pinned = retain_checkpoint(store, checkpoint, publication)
            current = _producer(store, publication, pinned.train_binding)
            if producer is not None and current != producer:
                raise ValueError("Evaluate series changes the exact Train definition")
            producer = current
            model_slot = retain_checkpoint_model(store, checkpoint, publication, pinned, current)
            tree = success_content_ref(model_slot.content)
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
        ranked_candidates = []
        for claim in binding.get("best_rank", {}).get("candidates", ()):
            completion = claim["completion"]
            checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
            source = retain_checkpoint(store, checkpoint, completion)
            ranked_candidates.append(BestCandidateEvidence(
                attempt=completion["producer_job_id"],
                producer=_producer(store, completion, source.train_binding),
                receipt=_retain_train_receipt(store, completion),
                checkpoint=source,
            ))
        bound = BoundMember(
            key=member.key, content=ContentRef(
                sha256=tree.descriptor.sha256, size=tree.descriptor.size_bytes,
                media_type=EVALUATION_SUBJECT_MEDIA_TYPE, tree=True,
            ),
            descriptor=value.model_dump(mode="json"),
            evaluation_subject=EvaluationSubjectSource(
                producer=producer, checkpoints=tuple(checkpoints), closure=closure,
                terminal_receipt=terminal_receipt, terminal_checkpoint=terminal_checkpoint,
                ranked_candidates=tuple(ranked_candidates),
            ),
        )
        result = view.publish_bound(
            request, attempt=attempt, products={"states": (product,)}, bindings={"subject": (bound,)},
            evaluation_manifest=manifest_object, target=target,
            execution_evidence=execution_evidence,
        )
        _release_adopted_points(store, result.receipt, binding)
        return result.ref


def _release_adopted_points(store: ImmutableArtifactBlobProvider, receipt: Success, binding: dict) -> None:
    trace = receipt.trace
    if trace is None or not trace.prefixes:
        return
    for raw in binding.get("live_evaluation_points", ()):
        point = BoundLiveEvaluationPoint.model_validate(raw)
        if point.evaluation_job_id == receipt.attempt and isinstance(point.result, ScoredEvaluationPoint) and (
            point.result.evidence in trace.prefixes
        ):
            release_staged_evidence(store, receipt.attempt, scope=point.root_id)


def _retain_train_receipt(store: ImmutableArtifactBlobProvider, publication: dict) -> ContentObjectRef:
    receipt = JobReceipt.model_validate(publication["receipt"])
    path = canonical_manifest_path("TrainingRunManifest", receipt.manifest_id, root=store.root)
    digest, size = _files(publication)[path.relative_to(store.root).as_posix()]
    if digest != receipt.manifest_sha256:
        raise ValueError("Train receipt is absent from the exact Host binding")
    return store._store_file(path, sha256=digest, size_bytes=size)


def _files(publication: dict) -> dict[str, tuple[str, int]]:
    return {file["path"]: (file["sha256"], file["size_bytes"]) for file in publication["files"]}


def _producer(store: ImmutableArtifactBlobProvider, publication: dict, ref: ContentObjectRef) -> RequestObject:
    launch = BoundStudioTrainingJobInput.model_validate_json(store.get_object(ref))
    if (
        launch.canonical_request is None or launch.request_id != publication["request_id"]
        or launch.project_id != publication["project_id"] or launch.manifest_root != store.root
    ):
        raise ValueError("Checkpoint lacks its canonical Train request")
    return checkpoint_producer(launch)
