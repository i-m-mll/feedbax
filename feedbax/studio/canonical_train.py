"""Realize the supported Studio Train request as the existing native worker input."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from pathlib import Path

from pydantic import Field, field_validator

from feedbax.checkpoint_coordinates import finite_coordinates
from feedbax.contracts.artifact_custody import ArtifactBlobCustodyError, ContentObjectRef
from feedbax.contracts.base import StrictModel, canonical_json_bytes
from feedbax.contracts.model_values import model_owner_problem
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import LastMember, ProductRef
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec, TrainingRequestSpec
from feedbax.contracts.spec_storage import canonical_training_run_spec_sha256
from feedbax.contracts.studio_training import (
    STUDIO_GRAPH_METHOD,
    PublishedTrainingCheckpoint,
    StudioGraphMethodPayload,
    StudioTrainingJobInput,
)
from feedbax.contracts.training import (
    RiskAggregationSpec,
    TrainingConfig,
    TrainingProgramRegistry,
    TrainingRunSpec,
    WorkerExecutionSpec,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


class CanonicalTrainRequestBinding(StrictModel):
    request: ContentObjectRef
    documents: tuple[ContentObjectRef, ...]
    checkpoint: ProductRef | None = None


class PinnedTrainCheckpoint(StrictModel):
    producer_job_id: str
    receipt: dict[str, str]
    checkpoint: PublishedTrainingCheckpoint


class BoundStudioTrainingJobInput(StudioTrainingJobInput):
    """Internal Host payload and attempt sidecar, with canonical custody when admitted by Plan."""

    canonical_request: CanonicalTrainRequestBinding | None = None
    checkpoint_input: PinnedTrainCheckpoint | None = None
    checkpoint_catalog: tuple[int, ...] = Field(default=(), exclude_if=lambda value: not value)

    @field_validator("checkpoint_catalog", mode="before")
    @classmethod
    def _validate_catalog(cls, value):
        return finite_coordinates(value) if value else ()


def _decode_train(
    request_id: str, data: bytes, read_document: Callable[[str], bytes]
) -> tuple[TrainingRequestSpec | DocumentTrainingRequestSpec, ProductRef | None]:
    decoded = decode_request(data, read_document)
    if decoded.id != request_id or not isinstance(
        decoded.spec, (TrainingRequestSpec, DocumentTrainingRequestSpec)
    ):
        raise ValueError("Training launch does not match its canonical Train request.")
    checkpoint = None
    if decoded.inputs:
        members = decoded.inputs.get("checkpoint", ())
        if decoded.inputs.keys() != {"checkpoint"} or len(members) != 1:
            raise ValueError("/inputs/checkpoint: expected one predecessor checkpoint")
        reference = members[0].reference
        if (
            not isinstance(reference, ProductRef)
            or reference.role != "checkpoint"
            or not reference.request_id.startswith("train:")
            or not isinstance(reference.select, LastMember)
        ):
            raise ValueError("/inputs/checkpoint: expected the last checkpoint of a Train request")
        checkpoint = reference
    policy = decoded.spec.method_payload.payload.get("checkpoint_initialization")
    if checkpoint is None and policy is not None:
        raise ValueError("/inputs/checkpoint: initialization requires a predecessor checkpoint")
    if checkpoint is not None and policy != "initialize_from" and (
        not isinstance(policy, dict) or policy.get("mode") != "continue_from"
    ):
        raise ValueError(
            "/method_payload/payload/checkpoint_initialization: declare initialize_from or an explicit continue_from policy"
        )
    return decoded.spec, checkpoint


def lower_canonical_train(
    request: TrainingRequestSpec | DocumentTrainingRequestSpec,
    execution: TrainingProgramRegistry | WorkerExecutionSpec,
    *,
    registries=None,
) -> TrainingRunSpec:
    """Project one authored request, using registered or already bound worker declarations."""
    if request.method_ref != STUDIO_GRAPH_METHOD:
        raise ValueError(
            "/method_ref: canonical admission currently supports the Studio graph method"
        )
    if request.numerical_settings.model_dump(exclude_none=True):
        raise ValueError("/numerical_settings: overrides are unavailable for Studio graph Train")
    if request.risk_aggregation != RiskAggregationSpec():
        raise ValueError(
            "/risk_aggregation: Studio graph Train uses its ordinary objective reduction"
        )
    documents = request.trial_documents
    if documents is None:
        raise ValueError("/trial_documents: Managed Train requires Model, Task, and Objective documents.")
    from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
    from feedbax.plugins.application import new_application_registry_bundle

    registries = registries or new_application_registry_bundle()
    from feedbax.persistence.model_assets import model_artifact_provider
    from feedbax.compiler.model_document import lower_model_document

    graph_spec = lower_model_document(documents.model_name, documents.model, registries.components).graph
    paired = compile_trial_execution(
        documents.model_name, documents.model, documents.task_source,
        registries.components, registries.expression_functions, python_tasks=registries.python_tasks, objective=documents.objective,
        input_overrides=documents.input_overrides,
        artifact_provider=model_artifact_provider(graph_spec),
    )
    if not isinstance(paired, PreparedTrialExecution):
        problem = paired[0]
        raise ValueError("/trial_documents/" + "/".join(map(str, problem.path)) + ": " + problem.reason)
    if problem := model_owner_problem(paired.executable.resolved.graph, "checkpoint"):
        address = paired.lowered.source_address(pointer(problem.path))
        path = ("trial_documents", "model", *address.stored_path()) if address is not None else problem.path
        raise ValueError(f"{pointer(path)}: {problem.reason}")
    raw = request.method_payload.payload
    if request.seed is not None and "seed" in raw and request.seed != raw["seed"]:
        raise ValueError("/seed: top-level and method seeds disagree")
    payload = request.method_payload
    if request.seed is not None:
        payload = payload.model_copy(update={"payload": raw | {"seed": request.seed}})
    method = StudioGraphMethodPayload.model_validate(payload.payload)
    method.require_supported_launch(resume=False)
    authored_config = request.training_config
    continuation = method.continuation.progress if method.continuation is not None else None
    batches = continuation.additional_batches if continuation is not None else method.target_total
    if authored_config.n_batches != batches:
        raise ValueError("/training_config/n_batches: does not match the method target")
    if isinstance(authored_config, TrainingConfig) and (
        authored_config.n_reach_steps <= 0 or authored_config.n_reach_steps != paired.execution.task.n_steps
    ):
        raise ValueError("/training_config/n_reach_steps: does not match the authored task horizon")
    if authored_config.batch_size <= 0 or authored_config.snapshot_interval <= 0:
        raise ValueError("/training_config: batch size and snapshot interval must be positive")
    schedule = method.optimizer.lr_schedule
    rate = schedule.learning_rate_0 if schedule is not None else method.optimizer.params.get("learning_rate")
    if isinstance(authored_config, TrainingConfig):
        if type(rate) not in (float, int) or not isfinite(rate) or authored_config.learning_rate != rate:
            raise ValueError("/training_config/learning_rate: does not match the finite method rate")
        config = authored_config
    else:
        if type(rate) not in (float, int) or not isfinite(rate) or rate <= 0:
            path = "lr_schedule/learning_rate_0" if schedule is not None else "params/learning_rate"
            raise ValueError(f"/method_payload/payload/optimizer/{path}: requires a finite positive rate")
        if schedule is not None and "learning_rate" in method.optimizer.params:
            raise ValueError(
                "/method_payload/payload/optimizer/params/learning_rate: the schedule owns the rate"
            )
        config = TrainingConfig(
            **authored_config.model_dump(exclude={"n_reach_steps"}),
            learning_rate=rate,
            n_reach_steps=paired.execution.task.n_steps,
        )
    if isinstance(execution, TrainingProgramRegistry):
        resolved = execution.resolve_execution(request.method_ref, payload)
        contract, phase = resolved.contract, resolved.effective_phase
    else:
        contract, phase = execution.method_contract, execution.effective_phase
    return TrainingRunSpec(
        graph={"inline": paired.executable.resolved.graph.model_dump(mode="json", exclude_none=True)},
        trial_documents=documents,
        training_config=config,
        risk_aggregation=request.risk_aggregation,
        method_ref=request.method_ref,
        method_payload=payload,
        on_nan=request.on_nan,
        worker_execution=WorkerExecutionSpec(method_contract=contract, effective_phase=phase),
        checkpoint_progress={
            "checkpoint_interval": config.snapshot_interval,
            "progress_interval": 1,
            **({"continuation": continuation} if continuation is not None else {}),
        },
    )


def bind_canonical_train(
    request_id: str,
    data: bytes,
    documents: Mapping[str, bytes],
    registry: TrainingProgramRegistry,
    store: ImmutableArtifactBlobProvider,
    *, registries=None,
) -> tuple[TrainingRunSpec, CanonicalTrainRequestBinding]:
    """Validate a compiler request and retain its exact factored objects in the project store."""
    closure: dict[str, bytes] = {}

    def read_document(digest: str) -> bytes:
        raw = documents[digest]
        closure[digest] = raw
        return raw

    request, checkpoint = _decode_train(request_id, data, read_document)
    spec = lower_canonical_train(request, registry, registries=registries)
    refs = tuple(
        store.store_rooted_object(raw, namespace="studio-train-document")
        for _, raw in sorted(closure.items())
    )
    return spec, CanonicalTrainRequestBinding(
        request=store.store_rooted_object(data, namespace="studio-train-request"),
        documents=refs,
        checkpoint=checkpoint,
    )


def verify_training_request_binding(
    launch: BoundStudioTrainingJobInput,
) -> tuple[tuple[Path, ContentObjectRef], ...]:
    """Authenticate the request and the exact executable payload pinned by the launch."""
    if launch.canonical_request is None:
        if launch.request_id != canonical_training_run_spec_sha256(launch.spec):
            raise ValueError("Training launch does not match its authored request id.")
        return ()
    binding = launch.canonical_request
    store = ImmutableArtifactBlobProvider(launch.manifest_root)
    documents = {ref.sha256: ref for ref in binding.documents}
    if len(documents) != len(binding.documents):
        raise ValueError("Canonical Train binding repeats a document object.")
    used: set[str] = set()

    def read_object(ref: ContentObjectRef) -> bytes:
        try:
            return store.get_object(ref)
        except (ArtifactBlobCustodyError, OSError) as error:
            raise ValueError(
                f"Cannot authenticate canonical Train object {ref.object_id}: {error}"
            ) from error

    def read_document(digest: str) -> bytes:
        if digest not in documents:
            raise ValueError(f"Canonical Train binding is missing document object {digest}.")
        used.add(digest)
        return read_object(documents[digest])

    request, checkpoint = _decode_train(
        launch.request_id, read_object(binding.request), read_document
    )
    if checkpoint != binding.checkpoint:
        raise ValueError("Canonical Train binding does not match its checkpoint input.")
    if used != documents.keys():
        raise ValueError("Canonical Train binding does not name its exact document closure.")
    if launch.bootstrap_modules:
        import asyncio
        from feedbax.plugins.composition import compose_application

        registries = asyncio.run(compose_application(modules=launch.bootstrap_modules)).bundle
    else:
        registries = None
    expected = lower_canonical_train(request, launch.spec.worker_execution, registries=registries)
    if canonical_json_bytes(expected.model_dump(mode="json")) != canonical_json_bytes(
        launch.spec.model_dump(mode="json")
    ):
        raise ValueError("Canonical Train request does not match its bound execution payload.")
    return tuple((store.object_path(ref), ref) for ref in (binding.request, *binding.documents))
