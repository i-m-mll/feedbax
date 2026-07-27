"""Typed, side-effect-free training-method authoring compilation."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

import feedbax.contracts.training as training_contracts
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.run_matrix import (
    AuthoredTrainingRow,
    RowLowererIdentity,
    TrainingRowLoweringResult,
)
from feedbax.contracts.spec_storage import (
    canonical_training_run_spec_projection,
    training_spec_sha256,
)
from feedbax.contracts.training import (
    ArtifactPolicySpec,
    CheckpointProgressPolicySpec,
    GraphTopologySourceSpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    ObjectiveSlotSpec,
    RiskAggregationSpec,
    RunControlSpec,
    TaskSpec,
    TrainingConfig,
    TrainingMethodAuthoringContribution,
    TrainingMethodDescriptor,
    TrainingRunSpec,
    WorkerExecutionSpec,
)


TRAINING_METHOD_AUTHORING_LOWERER_ID = "feedbax.training.method_authoring"
TRAINING_METHOD_AUTHORING_LOWERER_VERSION = "feedbax.training.method_authoring.v1"
TRAINING_METHOD_AUTHORING_LOWERER_IDENTITY = RowLowererIdentity(
    lowerer_id=TRAINING_METHOD_AUTHORING_LOWERER_ID,
    lowerer_version=TRAINING_METHOD_AUTHORING_LOWERER_VERSION,
)

AUTHORING_RESERVED_METADATA_KEYS = frozenset(
    {
        "schema_id",
        "schema_version",
        "method_ref",
        "method_payload",
        "run_control",
        "graph",
        "task",
        "objective",
        "worker_execution",
    }
)

PayloadT = TypeVar("PayloadT", bound=BaseModel)
ModelT = TypeVar("ModelT", bound=BaseModel)


class TrainingMethodAuthoringError(ValueError):
    """Raised when compact method authoring cannot lower without ambiguity."""


@dataclass(frozen=True)
class TrainingMethodAuthoringProjectors(Generic[PayloadT]):
    """The four explicit semantic projections required by authoring compilation."""

    graph: Callable[[PayloadT], object]
    task: Callable[[PayloadT], object]
    objective: Callable[[PayloadT], object]
    domain: Callable[[PayloadT], Mapping[str, Any]]

    def __post_init__(self) -> None:
        invalid = [
            name
            for name in ("graph", "task", "objective", "domain")
            if not callable(getattr(self, name))
        ]
        if invalid:
            raise TypeError(f"training authoring projectors must be callable; invalid={invalid!r}")


@dataclass(frozen=True)
class TrainingMethodAuthoringCompilation(Mapping[str, Any]):
    """Validated canonical products of one compact authored training row."""

    run_spec: TrainingRunSpec
    worker_execution: WorkerExecutionSpec
    lowering_result: TrainingRowLoweringResult

    def __getitem__(self, key: str) -> Any:
        """Expose the lowering result through the TrainingRowLowerer mapping protocol."""
        if key not in {"schema_id", "schema_version", "execution_payload", "lowerer_identities"}:
            raise KeyError(key)
        return getattr(self.lowering_result, key)

    def __iter__(self) -> Iterator[str]:
        """Iterate TrainingRowLoweringResult fields for Pydantic validation."""
        return iter(("schema_id", "schema_version", "execution_payload", "lowerer_identities"))

    def __len__(self) -> int:
        """Return the stable number of TrainingRowLoweringResult fields."""
        return 4


def _normalize_method_ref(method_ref: MethodRefSpec | str | Mapping[str, Any]) -> MethodRefSpec:
    if isinstance(method_ref, MethodRefSpec):
        return method_ref.model_copy(deep=True)
    if isinstance(method_ref, str):
        parts = method_ref.split("/")
        if len(parts) != 3 or any(not part or part != part.strip() for part in parts):
            raise TrainingMethodAuthoringError(
                "/method_ref must be exactly '<package>/<name>/<version>'"
            )
        return MethodRefSpec(package=parts[0], name=parts[1], version=parts[2])
    try:
        return MethodRefSpec.model_validate(method_ref)
    except Exception as exc:
        raise TrainingMethodAuthoringError(f"/method_ref is invalid: {exc}") from exc


def _validate_run_control(
    run_control: RunControlSpec | Mapping[str, Any],
) -> RunControlSpec:
    if isinstance(run_control, RunControlSpec):
        return RunControlSpec.model_validate(run_control.model_dump(mode="python"))
    if "schema_id" not in run_control or "schema_version" not in run_control:
        raise TrainingMethodAuthoringError(
            "/run_control requires explicit schema_id and schema_version"
        )
    try:
        migrated = default_spec_registry.migrate("RunControlSpec", run_control)
        return RunControlSpec.model_validate(migrated.payload)
    except Exception as exc:
        raise TrainingMethodAuthoringError(f"/run_control is invalid: {exc}") from exc


def _project_model(
    name: str,
    projector: Callable[[PayloadT], object],
    payload: PayloadT,
    model: type[BaseModel],
) -> BaseModel:
    try:
        projected = projector(payload.model_copy(deep=True))
    except Exception as exc:
        raise TrainingMethodAuthoringError(f"/{name} projector failed: {exc}") from exc
    try:
        validated = model.model_validate(projected)
        return model.model_validate(validated.model_dump(mode="json", exclude_none=True))
    except Exception as exc:
        raise TrainingMethodAuthoringError(
            f"/{name} projector returned invalid output: {exc}"
        ) from exc


def _project_domain(
    projector: Callable[[PayloadT], Mapping[str, Any]],
    payload: PayloadT,
) -> dict[str, Any]:
    try:
        projected = projector(payload.model_copy(deep=True))
    except Exception as exc:
        raise TrainingMethodAuthoringError(f"/domain projector failed: {exc}") from exc
    if not isinstance(projected, Mapping):
        raise TrainingMethodAuthoringError("/domain projector must return a mapping")
    if any(not isinstance(key, str) or not key for key in projected):
        raise TrainingMethodAuthoringError(
            "/domain projector mapping keys must be non-empty strings"
        )
    collisions = sorted(AUTHORING_RESERVED_METADATA_KEYS.intersection(projected))
    if collisions:
        raise TrainingMethodAuthoringError(
            f"/domain projector collides with reserved metadata keys={collisions!r}"
        )
    try:
        encoded = json.dumps(
            dict(projected),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        canonical = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TrainingMethodAuthoringError(
            f"/domain projector output must be canonical JSON: {exc}"
        ) from exc
    if canonical != dict(projected):
        raise TrainingMethodAuthoringError(
            "/domain projector output must already use canonical JSON value types"
        )
    return canonical


def _copy_typed_option(
    value: ModelT | None,
    model: type[ModelT],
    *,
    path: str,
) -> ModelT:
    if value is None:
        return model()
    if type(value) is not model:
        raise TrainingMethodAuthoringError(f"{path} must be an exact {model.__name__} instance")
    return model.model_validate(value.model_dump(mode="python"))


def _descriptor_for_authoring(
    method_ref: MethodRefSpec,
) -> TrainingMethodDescriptor[Any]:
    registry = training_contracts.DEFAULT_TRAINING_METHOD_REGISTRY
    registry.resolve(method_ref, path="/method_ref")
    descriptor = registry.descriptor(method_ref)
    if descriptor is None:
        raise TrainingMethodAuthoringError(
            f"/method_ref {method_ref.key!r} is a low-level-only registration; "
            "typed authoring requires a TrainingMethodDescriptor"
        )
    if descriptor.authoring_hook is None:
        raise TrainingMethodAuthoringError(
            f"/method_ref {method_ref.key!r} descriptor has no authoring_hook"
        )
    return descriptor


def compile_training_method_authoring(
    row: AuthoredTrainingRow | Mapping[str, Any],
    _context: Any | None = None,
    *,
    method_ref: MethodRefSpec | str | Mapping[str, Any],
    run_control: RunControlSpec | Mapping[str, Any],
    projectors: TrainingMethodAuthoringProjectors[Any],
    artifacts: ArtifactPolicySpec | None = None,
    risk_aggregation: RiskAggregationSpec | None = None,
) -> TrainingMethodAuthoringCompilation:
    """Compile one compact typed method payload into canonical run contracts.

    The function is a ``TrainingRowLowerer``-compatible callable when its
    keyword-only authoring arguments are bound with ``functools.partial``.
    It performs no storage, custody, environment, or launch operations.
    """
    try:
        authored_row = (
            row.model_copy(deep=True)
            if isinstance(row, AuthoredTrainingRow)
            else AuthoredTrainingRow.model_validate(row)
        )
    except Exception as exc:
        raise TrainingMethodAuthoringError(f"/row is invalid: {exc}") from exc
    source_snapshot = authored_row.model_dump(mode="python")
    expected_source_hash = training_spec_sha256(authored_row.payload)
    if authored_row.payload_hash != expected_source_hash:
        raise TrainingMethodAuthoringError(
            "/row/payload_hash does not match the canonical authored payload"
        )
    normalized_ref = _normalize_method_ref(method_ref)
    control = _validate_run_control(run_control)
    descriptor = _descriptor_for_authoring(normalized_ref)
    artifact_policy = _copy_typed_option(
        artifacts,
        ArtifactPolicySpec,
        path="/artifacts",
    )
    risk_policy = _copy_typed_option(
        risk_aggregation,
        RiskAggregationSpec,
        path="/risk_aggregation",
    )

    authoring_hook = descriptor.authoring_hook
    if authoring_hook is None:  # Narrowed by _descriptor_for_authoring.
        raise AssertionError("descriptor authoring hook unexpectedly missing")
    try:
        hook_identity = RowLowererIdentity(
            lowerer_id=authoring_hook.lowerer_id,
            lowerer_version=authoring_hook.lowerer_version,
        )
    except Exception as exc:
        raise TrainingMethodAuthoringError(f"/authoring_hook identity is invalid: {exc}") from exc
    if hook_identity == TRAINING_METHOD_AUTHORING_LOWERER_IDENTITY:
        raise TrainingMethodAuthoringError(
            "/authoring_hook identity duplicates the reserved authoring compiler identity"
        )

    try:
        typed_payload = descriptor.payload_model.model_validate(copy.deepcopy(authored_row.payload))
    except Exception as exc:
        raise TrainingMethodAuthoringError(
            f"/row/payload does not match method payload schema: {exc}"
        ) from exc
    envelope = MethodPayloadEnvelope(
        schema_id=descriptor.payload_schema_id,
        schema_version=descriptor.payload_schema_version,
        payload=typed_payload.model_dump(mode="json"),
    )
    registry = training_contracts.DEFAULT_TRAINING_METHOD_REGISTRY
    registry.validate_payload(normalized_ref, envelope, path="/method_payload")

    graph = _project_model("graph", projectors.graph, typed_payload, GraphTopologySourceSpec)
    task = _project_model("task", projectors.task, typed_payload, TaskSpec)
    objective = _project_model("objective", projectors.objective, typed_payload, ObjectiveSlotSpec)
    domain = _project_domain(projectors.domain, typed_payload)

    try:
        resolved = registry.resolve_execution(normalized_ref, envelope)
    except Exception as exc:
        raise TrainingMethodAuthoringError(
            f"/method_ref registry execution resolution failed: {exc}"
        ) from exc
    callback_payload = typed_payload.model_copy(deep=True)
    callback_snapshot = callback_payload.model_dump(mode="python")
    try:
        raw_contribution = authoring_hook.compile(callback_payload)
    except Exception as exc:
        raise TrainingMethodAuthoringError(f"/authoring_hook failed: {exc}") from exc
    if callback_payload.model_dump(mode="python") != callback_snapshot:
        raise TrainingMethodAuthoringError("/authoring_hook mutated its typed payload input")
    if authored_row.model_dump(mode="python") != source_snapshot:
        raise TrainingMethodAuthoringError("authoring compilation mutated the source row")
    if type(raw_contribution) is not TrainingMethodAuthoringContribution:
        raise TrainingMethodAuthoringError(
            "/authoring_hook must return an exact TrainingMethodAuthoringContribution instance"
        )
    try:
        contribution = TrainingMethodAuthoringContribution.model_validate(
            raw_contribution.model_dump(mode="python")
        )
    except Exception as exc:
        raise TrainingMethodAuthoringError(
            f"/authoring_hook returned an invalid contribution: {exc}"
        ) from exc
    training_config = TrainingConfig.model_validate(
        {
            **contribution.training_config.model_dump(mode="python"),
            "n_batches": control.n_batches,
            "batch_size": control.batch_size,
        }
    )
    expected_worker = WorkerExecutionSpec(
        method_contract=resolved.contract,
        effective_phase=resolved.effective_phase,
        mapping_levels=contribution.mapping_levels,
    )
    run_spec = TrainingRunSpec(
        graph=graph,
        task=task,
        training_config=training_config,
        objective=objective,
        risk_aggregation=risk_policy,
        method_ref=normalized_ref,
        method_payload=envelope,
        method_extensions=contribution.method_extensions,
        worker_execution=expected_worker,
        artifacts=artifact_policy,
        checkpoint_progress=CheckpointProgressPolicySpec(
            checkpoint_interval=control.checkpoint_interval,
            progress_interval=control.progress_interval,
            continuation=control.continuation,
        ),
        metadata=domain,
    )
    canonical_payload = canonical_training_run_spec_projection(run_spec)

    lowering_result = TrainingRowLoweringResult(
        execution_payload=canonical_payload,
        lowerer_identities=[
            TRAINING_METHOD_AUTHORING_LOWERER_IDENTITY,
            hook_identity,
        ],
    )
    return TrainingMethodAuthoringCompilation(
        run_spec=run_spec,
        worker_execution=expected_worker,
        lowering_result=lowering_result,
    )
