"""Realize the supported Studio Train request as the existing native worker input."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from pathlib import Path

from pydantic import Field, field_validator

from feedbax.checkpoint_coordinates import finite_coordinates
from feedbax.contracts.artifact_custody import ArtifactBlobCustodyError, ContentObjectRef
from feedbax.contracts.base import StrictModel, canonical_json_bytes
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import LastMember, ProductRef
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.contracts.spec_storage import canonical_training_run_spec_sha256
from feedbax.contracts.studio_training import (
    STUDIO_GRAPH_METHOD,
    PublishedTrainingCheckpoint,
    StudioGraphMethodPayload,
    StudioTrainingJobInput,
)
from feedbax.contracts.training import (
    RiskAggregationSpec,
    TrainingProgramRegistry,
    TrainingRunSpec,
    WorkerExecutionSpec,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.studio.protocol import infer_task_n_steps


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
) -> tuple[TrainingRequestSpec, ProductRef | None]:
    decoded = decode_request(data, read_document)
    if decoded.id != request_id or not isinstance(decoded.spec, TrainingRequestSpec):
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
    request: TrainingRequestSpec, execution: TrainingProgramRegistry | WorkerExecutionSpec
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
    objective = request.objective
    if objective.kind != "loss_term" or objective.loss is None or objective.payload is not None:
        raise ValueError("/objective: Studio graph Train requires an authored loss term")
    if request.task_binding_spec is None:
        raise ValueError("/task_binding_spec: Studio graph Train requires task bindings")
    raw = request.method_payload.payload
    if request.seed is not None and "seed" in raw and request.seed != raw["seed"]:
        raise ValueError("/seed: top-level and method seeds disagree")
    payload = request.method_payload
    if request.seed is not None:
        payload = payload.model_copy(update={"payload": raw | {"seed": request.seed}})
    method = StudioGraphMethodPayload.model_validate(payload.payload)
    method.require_supported_launch(resume=False)
    config = request.training_config
    continuation = method.continuation.progress if method.continuation is not None else None
    batches = continuation.additional_batches if continuation is not None else method.target_total
    if config.n_batches != batches:
        raise ValueError("/training_config/n_batches: does not match the method target")
    if config.n_reach_steps <= 0 or config.n_reach_steps != infer_task_n_steps(request.task):
        raise ValueError("/training_config/n_reach_steps: does not match the authored task horizon")
    if config.batch_size <= 0 or config.snapshot_interval <= 0:
        raise ValueError("/training_config: batch size and snapshot interval must be positive")
    schedule = method.optimizer.lr_schedule
    rate = schedule.learning_rate_0 if schedule is not None else method.optimizer.params.get("learning_rate")
    if type(rate) not in (float, int) or not isfinite(rate) or config.learning_rate != rate:
        raise ValueError("/training_config/learning_rate: does not match the finite method rate")
    if isinstance(execution, TrainingProgramRegistry):
        resolved = execution.resolve_execution(request.method_ref, payload)
        contract, phase = resolved.contract, resolved.effective_phase
    else:
        contract, phase = execution.method_contract, execution.effective_phase
    return TrainingRunSpec(
        graph={"inline": request.graph.model_dump(mode="json", exclude_none=True)},
        task=request.task,
        task_binding_spec=request.task_binding_spec,
        training_config=config,
        objective=objective,
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
) -> tuple[TrainingRunSpec, CanonicalTrainRequestBinding]:
    """Validate a compiler request and retain its exact factored objects in the project store."""
    closure: dict[str, bytes] = {}

    def read_document(digest: str) -> bytes:
        raw = documents[digest]
        closure[digest] = raw
        return raw

    request, checkpoint = _decode_train(request_id, data, read_document)
    spec = lower_canonical_train(request, registry)
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
    expected = lower_canonical_train(request, launch.spec.worker_execution)
    if canonical_json_bytes(expected.model_dump(mode="json")) != canonical_json_bytes(
        launch.spec.model_dump(mode="json")
    ):
        raise ValueError("Canonical Train request does not match its bound execution payload.")
    return tuple((store.object_path(ref), ref) for ref in (binding.request, *binding.documents))
