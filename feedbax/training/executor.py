"""Native TrainingRunSpec executor and manifest emitter."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
from pydantic import ValidationError

from feedbax.contracts.checkpoints import (
    CheckpointSegmentLineage,
    CheckpointLineageRef,
    CheckpointTransactionManifest,
)
from feedbax.contracts.manifest import (
    EntrypointRef,
    ManifestStatus,
    ParentRef,
    Provenance,
    TrainingRunManifest,
    TrainingManifestMetadataProjectionCustody,
    TRAINING_MANIFEST_METADATA_PROJECTION_PROVENANCE_KEY,
    ArtifactRef,
    canonical_json_bytes,
    default_manifest_root,
    safe_manifest_key,
    sha256_bytes,
    store_bytes_artifact,
    store_json_artifact,
    training_run_manifest_id,
    utc_now,
    write_manifest,
)
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    GraphTopologySourceSpec,
    OptimizerSpec,
    ResolvedTrainingMethod,
    TrainingMethodRegistry,
    TrainingManifestMetadataProjection,
    TrainingRunSpec,
)
from feedbax.contracts.spec_storage import training_spec_canonical_bytes, training_spec_sha256
from feedbax.contracts.worker import (
    AxisCoordinateSpec,
    BarrierArtifactSinkSpec,
    MaterializedSlotAxisBinding,
    MethodContractSpec,
    MethodTrainingDiagnosticsSpec,
    ProgressCoordinate,
)
from feedbax.objectives.service import LossService, ObjectiveLoweringError
from feedbax.orchestration.events import (
    MAPPED_METRIC_VALUE_SCHEMA_VERSION,
    STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_ID,
    STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_VERSION,
    RunEventEmitter,
    normalize_serialized_metrics,
)
from feedbax.orchestration.schedule_eval import project_training_schedules
from feedbax.training.checkpoint_custody import (
    CheckpointWriteResult,
    ResumeSlotTransform,
    load_latest_checkpoint,
    write_checkpoint_transaction,
    materialize_concatenated_checkpoint_histories,
)
from feedbax.training.phase_executor import (
    InMemoryCheckpointStore,
    PhaseCheckpoint,
    PhaseProgramExecutor,
    StepGuardResult,
)
from feedbax.training.schedule_clocks import resolve_schedule_window
from feedbax.training.manifest_preflight import (
    build_training_run_manifest_spec_payloads,
    preflight_training_run_manifest_payloads,
    resolve_training_spec_payload_ref,
    validate_training_run_spec,
)
from feedbax.training.interruption import CancellationDecision
from feedbax.training.preparation import (
    ExecutionPreparationResult,
    MaterializedExecutionPreparation,
    RestoredScheduleContextBinder,
    _thaw_runtime_value,
    executor_owned_initial_slot_names,
    validate_materialized_execution_slots,
    validate_materialized_execution_preparation,
)
from feedbax.training.diagnostics import (
    CheckpointTransactionDiagnostic,
    LearningRateDiagnostic,
    MethodTrainingTrace,
    MethodTrainingTraceRecord,
    NativeExecutionProducerContext,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
    TrainingDiagnostics,
)
from feedbax.training.worker_validation import (
    WorkerExecutabilityEnvironment,
    resolve_execution_mapping,
    validate_worker_contract,
)


ManifestConflictPolicy = Literal["error", "reuse-identical"]
ProgressCallback = Callable[[Mapping[str, Any]], None]
CancellationProbe = Callable[[ProgressCoordinate], CancellationDecision | None]
_RESERVED_KERNEL_CONTEXT_KEYS = frozenset(
    {"run_spec", "method_payload", "schedule_projection"}
)
_FEEDBAX_METADATA_NAMESPACE_PREFIX = "feedbax_"
_METADATA_VALUE_ABSENT = object()
_METHOD_OBSERVATION_BUFFER_CAP = 500


class TrainingRunExecutorError(ValueError):
    """Base class for native training-run executor failures."""


class ManifestEmissionConflictError(TrainingRunExecutorError):
    """Raised when a manifest id already exists with different content."""


class DiagnosticsEmissionConflictError(TrainingRunExecutorError):
    """Raised when native diagnostics already exist with different content."""


class _ImmediateCancellation(Exception):
    """Internal control signal for an operator-requested immediate termination."""

    def __init__(self, coordinate: ProgressCoordinate, decision: CancellationDecision) -> None:
        self.coordinate = coordinate
        self.decision = decision


@dataclass
class _BufferedStaticValues:
    """Non-array leaves retained in update order across one buffered chunk."""

    values: tuple[Any, ...]


@dataclass
class _MethodObservationBuffer:
    """Bound raw method observations until an evidence publication boundary."""

    observations: list[dict[str, Any]]
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]]
    capacity: int = _METHOD_OBSERVATION_BUFFER_CAP

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("method observation buffer capacity must be positive")
        self._coordinates: list[ProgressCoordinate] = []
        self.flush_sizes: list[int] = []
        self.peak_buffered_updates = 0

    @property
    def buffered_updates(self) -> int:
        """Return the number of raw updates awaiting materialization."""
        return len(self._coordinates)

    def append(self, coordinate: ProgressCoordinate) -> None:
        """Retain one immutable raw coordinate, flushing only at hard capacity."""
        self._coordinates.append(coordinate)
        self.peak_buffered_updates = max(
            self.peak_buffered_updates,
            len(self._coordinates),
        )
        if len(self._coordinates) >= self.capacity:
            self.flush()

    def flush(self) -> None:
        """Materialize one stacked metrics PyTree and normalize each host record."""
        if not self._coordinates:
            return
        coordinates = tuple(self._coordinates)
        metrics = tuple(coordinate.metrics for coordinate in coordinates)

        def stack_leaves(*leaves: Any) -> Any:
            if all(leaf is None for leaf in leaves):
                return None
            try:
                return jnp.stack(leaves)
            except (TypeError, ValueError):
                if any(isinstance(leaf, (jax.Array, np.ndarray)) for leaf in leaves):
                    raise
                return _BufferedStaticValues(tuple(leaves))

        try:
            expected_structure = jt.structure(metrics[0], is_leaf=lambda item: item is None)
            if any(
                jt.structure(item, is_leaf=lambda value: value is None) != expected_structure
                for item in metrics[1:]
            ):
                raise TrainingRunExecutorError(
                    "buffered method observations changed metrics PyTree structure"
                )
            stacked_metrics = jt.map(
                stack_leaves,
                *metrics,
                is_leaf=lambda item: item is None,
            )
        except (TypeError, ValueError) as exc:
            raise TrainingRunExecutorError(
                "buffered method observations could not be stacked by corresponding leaf"
            ) from exc
        host_metrics = jax.device_get(stacked_metrics)
        normalized: list[dict[str, Any]] = []
        for index, coordinate in enumerate(coordinates):
            record_metrics = jt.map(
                lambda leaf: (
                    None
                    if leaf is None
                    else leaf.values[index]
                    if isinstance(leaf, _BufferedStaticValues)
                    else leaf[index]
                ),
                host_metrics,
                is_leaf=lambda item: item is None or isinstance(item, _BufferedStaticValues),
            )
            normalized.append(
                _history_event(
                    coordinate.model_copy(update={"metrics": record_metrics}),
                    self.slot_axis_bindings,
                )
            )
        self.observations.extend(normalized)
        self.flush_sizes.append(len(coordinates))
        self._coordinates.clear()

    def assert_empty(self) -> None:
        """Reject an execution result that omitted a buffered observation tail."""
        if self._coordinates:
            raise TrainingRunExecutorError(
                f"{len(self._coordinates)} method observations remain unflushed"
            )


def _compilation_cache_snapshot() -> dict[str, Any]:
    """Return a precisely labelled filesystem snapshot of JAX's persistent cache."""

    configured = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not configured:
        return {"configured": False, "directory": None, "entry_count": None, "bytes": None}
    root = Path(configured).expanduser()
    entry_count = 0
    total_bytes = 0
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            entry_count += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
    return {
        "configured": True,
        "directory": str(root),
        "entry_count": entry_count,
        "bytes": total_bytes,
    }


def _first_progress_telemetry(
    *,
    started_at: float,
    start_semantics: str,
    cache_before: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure first-progress latency and cache filesystem changes.

    These are operational proxies. They are deliberately not labelled as JAX
    compile duration or cache hits because the public JAX runtime does not
    expose either quantity reliably at this boundary.
    """

    cache_at_first_progress = _compilation_cache_snapshot()
    before_count = cache_before.get("entry_count")
    after_count = cache_at_first_progress.get("entry_count")
    if not cache_before.get("configured"):
        cache_effect = "not_configured"
        entry_delta = None
    elif isinstance(before_count, int) and isinstance(after_count, int):
        entry_delta = after_count - before_count
        if entry_delta > 0:
            cache_effect = "new_entries_observed"
        elif before_count > 0:
            cache_effect = "preexisting_entries_no_new_entries_observed"
        else:
            cache_effect = "empty_cache_no_new_entries_observed"
    else:
        entry_delta = None
        cache_effect = "snapshot_unavailable"
    return {
        "measurement_semantics": "measurement_start_to_first_progress_callback",
        "measurement_start_semantics": start_semantics,
        "start_to_first_progress_seconds": time.perf_counter() - started_at,
        "compile_time_estimate_seconds": None,
        "compile_time_estimate_semantics": "unavailable_without_runtime_compile_instrumentation",
        "persistent_cache_effectiveness_proxy": cache_effect,
        "persistent_cache_entry_delta_to_first_progress": entry_delta,
        "persistent_cache_before": dict(cache_before),
        "persistent_cache_at_first_progress": cache_at_first_progress,
    }


@dataclass(frozen=True)
class _PreparedBarrierArtifact:
    sink: BarrierArtifactSinkSpec
    data: bytes
    logical_name: str
    media_type: str
    suffix: str


@dataclass(frozen=True)
class TrainingRunExecutionResult:
    """Result returned by ``execute_training_run_spec``."""

    run_id: str
    status: ManifestStatus
    manifest: TrainingRunManifest
    manifest_path: Path
    diagnostics: TrainingDiagnostics
    diagnostics_path: Path
    final_slots: dict[str, Any]
    final_coordinate: ProgressCoordinate
    checkpoint_writes: tuple[CheckpointWriteResult, ...]
    history_events: tuple[dict[str, Any], ...]
    payload_binding_status: Literal["verified", "not_bound"]


class StreamingCheckpointStore(InMemoryCheckpointStore):
    """Checkpoint store that writes custody transactions at barrier time."""

    def __init__(
        self,
        *,
        root: Path | str,
        artifact_root: Path | str,
        run_spec: TrainingRunSpec,
        phase_program: Any,
        parent_lineage: Sequence[CheckpointLineageRef] = (),
        segment_start_batch: int = 0,
        segment_batch_count: int | None = None,
        segment_parent_transaction_id: str | None = None,
        resolved_method: ResolvedTrainingMethod[Any],
        slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]] | None = None,
        run_event_emitter: RunEventEmitter | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.artifact_root = Path(artifact_root)
        self.run_spec = run_spec
        self.phase_program = phase_program
        self.parent_lineage = tuple(parent_lineage)
        self.segment_start_batch = segment_start_batch
        self.segment_batch_count = segment_batch_count
        self.segment_parent_transaction_id = segment_parent_transaction_id
        self.resolved_method = resolved_method
        self.slot_axis_bindings = dict(slot_axis_bindings or {})
        self.run_event_emitter = run_event_emitter
        self._barrier_specs = {
            barrier.name: barrier for barrier in phase_program.checkpoint_barriers
        }
        self._writes: list[CheckpointWriteResult] = []
        self._barrier_artifacts: list[ArtifactRef] = []
        self._before_save: Callable[[ProgressCoordinate], None] | None = None

    def set_before_save(self, callback: Callable[[ProgressCoordinate], None] | None) -> None:
        """Install an executor-local validation hook before checkpoint publication."""
        self._before_save = callback

    @property
    def writes(self) -> tuple[CheckpointWriteResult, ...]:
        """Return successful custody writes in barrier firing order."""
        return tuple(self._writes)

    @property
    def barrier_artifacts(self) -> tuple[ArtifactRef, ...]:
        """Return local artifacts captured from barrier sink slots."""
        return tuple(self._barrier_artifacts)

    def save(self, checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        if self._before_save is not None:
            self._before_save(checkpoint.coordinate)
        saved = super().save(checkpoint)
        barrier_spec = self._barrier_specs[saved.barrier]
        prepared_artifacts = _prepare_barrier_artifacts(
            saved,
            barrier_spec.artifact_sinks,
            run_id=saved.coordinate.run_id,
        )
        segment_batch_count = self.segment_batch_count
        if segment_batch_count is None and self.segment_start_batch > 0:
            cumulative_completed_batches = _terminal_completed_batches(
                program=self.phase_program,
                final_slots=saved.slots,
                fallback=saved.coordinate.program_step,
                require_authority=True,
                slot_axis_bindings=self.slot_axis_bindings,
            )
            segment_batch_count = cumulative_completed_batches - self.segment_start_batch
            if segment_batch_count < 0:
                raise TrainingRunExecutorError(
                    "resumed checkpoint progress precedes its validated segment start: "
                    f"completed_batches={cumulative_completed_batches} "
                    f"segment_start_batch={self.segment_start_batch}"
                )
        checkpoint_metadata: dict[str, Any] = {"barrier_visit_ordinal": saved.visit_ordinal}
        descriptor = self.resolved_method.descriptor
        if descriptor is not None and descriptor.optimizer_step_extractor is not None:
            checkpoint_metadata["optimizer_step"] = _synchronized_optimizer_step(
                descriptor.optimizer_step_extractor,
                self.resolved_method.payload,
                saved.slots,
                self.slot_axis_bindings,
            )
        write = write_checkpoint_transaction(
            self.root,
            run_spec=self.run_spec,
            phase_program=self.phase_program,
            barrier_name=saved.barrier,
            coordinate=saved.coordinate,
            slots=saved.slots,
            status="partial",
            parent_lineage=self.parent_lineage,
            segment_start_batch=self.segment_start_batch,
            segment_batch_count=segment_batch_count,
            segment_parent_transaction_id=self.segment_parent_transaction_id,
            history_availability={"progress": True},
            metadata=checkpoint_metadata,
        )
        self._writes.append(write)
        if self.run_event_emitter is not None:
            coordinate_payload, checkpoint_metrics = normalize_serialized_metrics(
                saved.coordinate,
                saved.coordinate.metrics,
                self.slot_axis_bindings,
            )
            checkpoint_payload: dict[str, Any] = {
                "barrier": saved.barrier,
                "barrier_visit_ordinal": saved.visit_ordinal,
                "transaction_id": write.manifest.transaction_id,
                "coordinate": coordinate_payload,
                "program_step": saved.coordinate.program_step,
            }
            if any(
                name in saved.coordinate.metrics
                and bindings
                and bindings[0].mode == "mapped"
                for name, bindings in self.slot_axis_bindings.items()
            ):
                checkpoint_payload["metrics"] = checkpoint_metrics
            self.run_event_emitter.emit(
                "checkpoint_written",
                checkpoint_payload,
            )
        for prepared in prepared_artifacts:
            self._barrier_artifacts.append(
                store_bytes_artifact(
                    prepared.data,
                    root=self.artifact_root,
                    role=prepared.sink.role,
                    logical_name=prepared.logical_name,
                    media_type=prepared.media_type,
                    suffix=prepared.suffix,
                    metadata={
                        **prepared.sink.metadata,
                        "slot": prepared.sink.slot,
                        "barrier": saved.barrier,
                        "barrier_phase": barrier_spec.phase,
                        "barrier_visit_ordinal": saved.visit_ordinal,
                        "program_step": saved.coordinate.program_step,
                        "checkpoint_transaction_id": write.manifest.transaction_id,
                    },
                )
            )
        return saved


def execute_training_run_spec(
    spec: TrainingRunSpec | Mapping[str, Any],
    *,
    run_id: str | None = None,
    preparation: ExecutionPreparationResult | MaterializedExecutionPreparation | None = None,
    initial_slots: Mapping[str, Any] | None = None,
    kernel_context: Mapping[str, Any] | None = None,
    manifest_root: Path | str | None = None,
    checkpoint_root: Path | str | None = None,
    registry: TrainingMethodRegistry | None = None,
    loss_service: LossService | None = None,
    environment: WorkerExecutabilityEnvironment | None = None,
    training_spec_payload: Mapping[str, Any] | None = None,
    training_spec_payload_kind: str = "TrainingRunSpec",
    training_spec_payload_schema_id: str | None = None,
    training_spec_payload_schema_version: str | None = None,
    training_spec_payload_ref: str | None = None,
    manifest_metadata_projection: TrainingManifestMetadataProjection
    | Mapping[str, Any]
    | None = None,
    task_binding_spec: Mapping[str, Any] | None = None,
    resume: bool = False,
    resume_slot_transform: ResumeSlotTransform | None = None,
    stop_after_barrier: str | None = None,
    one_update: bool = False,
    manifest_conflict_policy: ManifestConflictPolicy = "error",
    issues: Sequence[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    run_event_emitter: RunEventEmitter | None = None,
    cancellation_probe: CancellationProbe | None = None,
    execution_started_at_monotonic: float | None = None,
    execution_start_semantics: str = "executor_entry",
    execution_context: NativeExecutionProducerContext | Mapping[str, Any] | None = None,
) -> TrainingRunExecutionResult:
    """Validate, execute, checkpoint, and natively emit one training-run manifest.

    ``progress_callback`` is called once for each generated training-progress
    history event, in history order, as each progress coordinate is produced
    during execution. Callback payloads have the same shape as stored history
    events. Exceptions raised by the callback propagate to the caller.

    ``run_event_emitter`` optionally receives the versioned run-event protocol.
    Emitter I/O diagnostics are contained by the emitter and do not change
    training-loop failure behavior.

    ``cancellation_probe`` is checked at progress boundaries. A ``stop``
    decision completes the next durable checkpoint before returning a cancelled
    manifest; ``terminate`` emits a cancelled manifest immediately.
    """
    run_spec = _validate_spec(spec)
    mapping_levels, slot_axis_bindings = resolve_execution_mapping(run_spec.worker_execution)
    loose_preparation = any(
        value is not None
        for value in (initial_slots, kernel_context, loss_service, resume_slot_transform)
    )
    if preparation is not None and loose_preparation:
        raise TrainingRunExecutorError(
            "preparation is mutually exclusive with initial_slots, kernel_context, "
            "loss_service, and resume_slot_transform"
        )
    if mapping_levels:
        if not isinstance(preparation, MaterializedExecutionPreparation):
            raise TrainingRunExecutorError(
                "active mapping levels require a Feedbax-materialized execution preparation; "
                "loose, scalar, and pre-stacked inputs are not accepted"
            )
        try:
            validate_materialized_execution_preparation(preparation, run_spec=run_spec)
        except ValueError as exc:
            raise TrainingRunExecutorError(
                f"mapped execution preparation validation failed: {exc}"
            ) from exc
        initial_slots = _thaw_runtime_value(preparation.initial_slots)
        kernel_context = preparation.kernel_context
        loss_service = preparation.loss_service
        resume_slot_transform = preparation.resume_slot_transform
        restored_schedule_context_binder = preparation.restored_schedule_context_binder
    elif isinstance(preparation, MaterializedExecutionPreparation):
        raise TrainingRunExecutorError(
            "MaterializedExecutionPreparation is invalid when mapping_levels are absent"
        )
    elif isinstance(preparation, ExecutionPreparationResult):
        initial_slots = preparation.initial_slots
        kernel_context = preparation.kernel_context
        loss_service = preparation.loss_service
        resume_slot_transform = preparation.resume_slot_transform
        restored_schedule_context_binder = preparation.restored_schedule_context_binder
    else:
        restored_schedule_context_binder = None
    _validate_checkpoint_progress_policy(run_spec)
    if not isinstance(one_update, bool):
        raise TrainingRunExecutorError("one_update must be a boolean")
    producer_context = _validate_execution_context(execution_context)
    payload_binding_status = _validate_execution_payload_binding(
        spec,
        run_spec=run_spec,
        execution_context=producer_context,
    )
    method_registry = registry or DEFAULT_TRAINING_METHOD_REGISTRY
    resolved_method = (
        run_spec.resolved_method
        if method_registry is DEFAULT_TRAINING_METHOD_REGISTRY
        else method_registry.resolve_execution(
            run_spec.method_ref,
            run_spec.method_payload,
            worker_execution=run_spec.worker_execution,
        )
    )
    embedded_training_spec_payload = (
        dict(training_spec_payload)
        if training_spec_payload is not None
        else run_spec.model_dump(
            mode="json",
            exclude_none=producer_context is not None,
        )
    )
    authoritative_training_spec_ref = _authoritative_training_spec_payload_ref(
        run_spec,
        training_spec_payload=embedded_training_spec_payload,
        training_spec_payload_kind=training_spec_payload_kind,
        training_spec_payload_schema_id=training_spec_payload_schema_id,
        training_spec_payload_schema_version=training_spec_payload_schema_version,
        execution_context=producer_context,
        explicit_ref=training_spec_payload_ref,
    )
    preflight_training_run_manifest_payloads(
        run_spec,
        training_spec_payload=embedded_training_spec_payload,
        training_spec_payload_kind=training_spec_payload_kind,
        training_spec_payload_schema_id=training_spec_payload_schema_id,
        training_spec_payload_schema_version=training_spec_payload_schema_version,
        training_spec_payload_ref=training_spec_payload_ref,
        authoritative_training_spec_payload_ref=authoritative_training_spec_ref,
        task_binding_spec=task_binding_spec,
    )
    effective_training_spec_ref = resolve_training_spec_payload_ref(
        training_spec_payload_ref,
        authoritative_ref=authoritative_training_spec_ref,
    )
    projection_custody = prepare_training_manifest_metadata_projection(
        manifest_metadata_projection,
        registry=method_registry,
        run_spec=run_spec,
        training_spec_payload=embedded_training_spec_payload,
        training_spec_payload_kind=training_spec_payload_kind,
        training_spec_payload_schema_id=training_spec_payload_schema_id,
        training_spec_payload_schema_version=training_spec_payload_schema_version,
        resolved_method=resolved_method,
    )
    root_path = (
        Path(manifest_root)
        if manifest_root is not None
        else (
            Path(run_spec.artifacts.manifest_root)
            if run_spec.artifacts.manifest_root is not None
            else default_manifest_root()
        )
    )
    collection_root = _collection_root(producer_context)
    if producer_context is not None:
        row_provenance = producer_context.execution.row_provenance
        assert row_provenance is not None
        planned_run_id = row_provenance.planned_run_id
        if run_id is not None and run_id != planned_run_id:
            raise TrainingRunExecutorError(
                "run_id must equal execution.row_provenance.planned_run_id for native "
                f"row execution; run_id={run_id!r}, planned_run_id={planned_run_id!r}"
            )
        resolved_run_id = planned_run_id
        exact_manifest_id = planned_run_id
    else:
        resolved_run_id = run_id or _default_run_id(run_spec)
        exact_manifest_id = training_run_manifest_id(resolved_run_id)
    manifest_output_path = (
        collection_root / "manifest.json"
        if collection_root is not None
        else root_path
        / "manifests"
        / "training_runs"
        / f"{exact_manifest_id.replace(':', '_')}.json"
    )
    if manifest_conflict_policy == "error" and manifest_output_path.exists():
        raise ManifestEmissionConflictError(
            f"Training-run manifest already exists at {manifest_output_path}"
        )
    diagnostics_path = _diagnostics_output_path(
        root_path=root_path,
        run_id=resolved_run_id,
        collection_root=collection_root,
    )
    _preflight_training_diagnostics_emission(diagnostics_path)
    method_payload = resolved_method.payload
    method_contract = resolved_method.contract
    _validate_declarations_match_spec(run_spec, method_contract.method_ref)

    graph_inline = _graph_inline(run_spec.graph)
    lowered = _lower_objective(
        run_spec,
        graph_inline=graph_inline,
        loss_service=loss_service or LossService(),
    )
    kernels = dict(resolved_method.update_kernels)
    guards = dict(resolved_method.guard_predicates)
    effective_phase = validate_worker_contract(
        method_contract,
        environment=environment,
        update_kernels=kernels,
        guard_predicates=guards,
        task_binding_spec=task_binding_spec,
        objective_requirements=lowered.requirements,
        active_mapping_axes=tuple(level.axis for level in mapping_levels),
    )
    if (
        resolved_method.descriptor is not None
        and effective_phase != run_spec.worker_execution.effective_phase
    ):
        raise TrainingRunExecutorError(
            "/worker_execution/effective_phase does not match the executable method contract"
        )
    program = effective_phase.phase_program
    executor_owned_slots = executor_owned_initial_slot_names(run_spec, slot_axis_bindings)
    slots = _initial_slots(
        initial_slots,
        lowered_objective=lowered.loss,
        executor_owned_slots=executor_owned_slots,
    )
    slots = _normalize_batch_progress_slot(slots, program=program)
    if mapping_levels:
        try:
            validate_materialized_execution_slots(
                slots,
                slot_axis_bindings,
                required_slots={slot.name for slot in effective_phase.state_slots if slot.required},
            )
        except ValueError as exc:
            raise TrainingRunExecutorError(
                f"mapped execution state validation failed after executor slot merge: {exc}"
            ) from exc
    caller_kernel_context = dict(kernel_context or {})
    reserved_context_keys = sorted(
        set(caller_kernel_context).intersection(_RESERVED_KERNEL_CONTEXT_KEYS)
    )
    if reserved_context_keys:
        raise TrainingRunExecutorError(
            f"kernel_context contains executor-reserved keys: {reserved_context_keys!r}"
        )
    executor_context = {
        **caller_kernel_context,
        "run_spec": run_spec,
        "method_payload": method_payload,
    }
    custody_root = _checkpoint_root(
        root_path=root_path,
        configured_root=checkpoint_root or run_spec.artifacts.artifact_root,
        run_id=resolved_run_id,
    )
    resume_barrier: str | None = None
    parent_lineage: list[CheckpointLineageRef] = []
    loaded_resume_checkpoint: PhaseCheckpoint | None = None
    segment_start_batch = 0
    segment_batch_count: int | None = None
    segment_parent_transaction_id: str | None = None
    if resume:
        loaded = load_latest_checkpoint(
            custody_root,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=slots,
            resume_slot_transform=resume_slot_transform,
            continuation_request=run_spec.checkpoint_progress.continuation,
            allow_new_lineage_override=(run_spec.checkpoint_progress.continuation is not None),
        )
        loaded_resume_checkpoint = PhaseCheckpoint(
            barrier=loaded.manifest.barrier,
            coordinate=loaded.manifest.completed_coordinate,
            slots=loaded.slots,
            visit_ordinal=_checkpoint_visit_ordinal(loaded.manifest.metadata),
        )
        resume_barrier = loaded.manifest.barrier
        parent_lineage.append(
            CheckpointLineageRef(
                transaction_id=loaded.manifest.transaction_id,
                manifest=ParentRef(
                    kind="TrainingCheckpointTransactionManifest",
                    id=loaded.manifest.transaction_id,
                    role="resume_parent",
                ),
            )
        )
        continuation = run_spec.checkpoint_progress.continuation
        if continuation is not None:
            segment_start_batch = continuation.source_completed_batches
            segment_batch_count = continuation.additional_batches
            segment_parent_transaction_id = loaded.manifest.transaction_id
        else:
            segment_start_batch = _same_row_resume_start_batch(loaded.manifest)
            if segment_start_batch > 0:
                segment_parent_transaction_id = loaded.manifest.transaction_id

    checkpoint_store = StreamingCheckpointStore(
        root=custody_root,
        artifact_root=root_path,
        run_spec=run_spec,
        phase_program=program,
        parent_lineage=parent_lineage,
        segment_start_batch=segment_start_batch,
        segment_batch_count=segment_batch_count,
        segment_parent_transaction_id=segment_parent_transaction_id,
        resolved_method=resolved_method,
        slot_axis_bindings=slot_axis_bindings,
        run_event_emitter=run_event_emitter,
    )
    executor = PhaseProgramExecutor(
        program,
        kernels,
        guard_predicates=guards,
        checkpoint_store=checkpoint_store,
        state_slots=effective_phase.state_slots,
        mapping_levels=mapping_levels,
        slot_axis_bindings=slot_axis_bindings,
    )
    prepared_slots = (
        executor.merge_resume_slots(slots, loaded_resume_checkpoint)
        if loaded_resume_checkpoint is not None
        else slots
    )
    if loaded_resume_checkpoint is not None:
        executor_context = _bind_restored_schedule_context(
            run_spec=run_spec,
            resolved_method=resolved_method,
            slots=prepared_slots,
            slot_axis_bindings=slot_axis_bindings,
            kernel_context=executor_context,
            binder=restored_schedule_context_binder,
        )
    try:
        executor.preflight(
            prepared_slots,
            run_id=resolved_run_id,
            context=executor_context,
        )
    except ValueError as exc:
        raise TrainingRunExecutorError(f"mapped execution preflight failed: {exc}") from exc
    if loaded_resume_checkpoint is not None:
        checkpoint_store.remember(loaded_resume_checkpoint)
    live_history_events: list[dict[str, Any]] = []
    method_observations: list[dict[str, Any]] = []
    method_observation_buffer = (
        _MethodObservationBuffer(method_observations, slot_axis_bindings)
        if method_contract.training_diagnostics is not None
        else None
    )
    if method_observation_buffer is not None:
        checkpoint_store.set_before_save(lambda _coordinate: method_observation_buffer.flush())
    method_observation_origin_batch = (
        _terminal_completed_batches(
            program=program,
            final_slots=prepared_slots,
            fallback=segment_start_batch,
            require_authority=True,
            slot_axis_bindings=slot_axis_bindings,
        )
        if method_contract.training_diagnostics is not None
        else segment_start_batch
    )
    total_batches = int(run_spec.training_config.n_batches or 0)
    started_at = (
        time.perf_counter()
        if execution_started_at_monotonic is None
        else execution_started_at_monotonic
    )
    cache_before = _compilation_cache_snapshot()
    runtime_telemetry: dict[str, Any] = {}
    cancellation: CancellationDecision | None = None

    def observe_cancellation(coordinate: ProgressCoordinate) -> None:
        nonlocal cancellation
        if cancellation_probe is None or cancellation is not None:
            return
        decision = cancellation_probe(coordinate)
        if decision is None or decision.action == "continue":
            return
        if decision.action == "terminate":
            raise _ImmediateCancellation(coordinate, decision)
        cancellation = decision

    try:
        live_progress_callback = _live_progress_callback(
            progress_callback,
            live_history_events,
            run_event_emitter=run_event_emitter,
            total_batches=total_batches,
            started_at=started_at,
            cancellation_observer=observe_cancellation,
            runtime_telemetry=runtime_telemetry,
            cache_before=cache_before,
            start_semantics=execution_start_semantics,
            slot_axis_bindings=slot_axis_bindings,
        )

        def publish_progress(coordinate: ProgressCoordinate) -> None:
            if method_observation_buffer is not None:
                method_observation_buffer.flush()
            live_progress_callback(coordinate)

        execution = executor.run(
            slots,
            run_id=resolved_run_id,
            resume_from_barrier=resume_barrier,
            stop_after_barrier=stop_after_barrier,
            stop_after_next_barrier=lambda _barrier: cancellation is not None,
            context=executor_context,
            progress_callback=publish_progress,
            method_observation_callback=(
                method_observation_buffer.append
                if method_observation_buffer is not None
                else None
            ),
            step_guard=_executor_nan_guard(
                run_spec=run_spec,
                custody_root=custody_root,
                program=program,
                expected_slots=slots,
                resume_slot_transform=resume_slot_transform,
            ),
            checkpoint_interval=run_spec.checkpoint_progress.checkpoint_interval,
            progress_interval=run_spec.checkpoint_progress.progress_interval,
            include_completed_batches_in_progress=method_contract.training_diagnostics is not None,
            one_update=one_update,
        )
        if method_observation_buffer is not None:
            method_observation_buffer.flush()
            method_observation_buffer.assert_empty()
        checkpoint_writes = checkpoint_store.writes
        continuation = run_spec.checkpoint_progress.continuation
        if continuation is not None and continuation.self_contained and checkpoint_writes:
            materialize_concatenated_checkpoint_histories(
                custody_root,
                custody_root / "derived" / f"{resolved_run_id}-stitched-histories.pkl",
                parent_roots={lineage.transaction_id: custody_root for lineage in parent_lineage},
            )
        barrier_artifacts = checkpoint_store.barrier_artifacts
        history_events = live_history_events
        final_metrics = _final_metrics(
            execution.slots,
            execution.coordinate,
            slot_axis_bindings,
        )
        diagnostics = _build_training_diagnostics(
            manifest_id=exact_manifest_id,
            run_id=resolved_run_id,
            status="cancelled" if cancellation is not None else "completed",
            final_coordinate=execution.coordinate,
            final_slots=execution.slots,
            program=program,
            checkpoint_writes=checkpoint_writes,
            segment_start_batch=segment_start_batch,
            method_observation_origin_batch=method_observation_origin_batch,
            history_events=history_events,
            method_observations=method_observations,
            execution_context=producer_context,
            method_contract=method_contract,
            slot_axis_bindings=slot_axis_bindings,
        )
        diagnostics_path = _diagnostics_output_path(
            root_path=root_path,
            run_id=resolved_run_id,
            collection_root=collection_root,
        )
        diagnostics_ref = _training_diagnostics_ref(diagnostics, diagnostics_path)
        manifest = _build_manifest(
            run_spec,
            run_id=resolved_run_id,
            manifest_id=exact_manifest_id,
            root_path=root_path,
            training_spec_payload=embedded_training_spec_payload,
            training_spec_payload_kind=training_spec_payload_kind,
            training_spec_payload_schema_id=training_spec_payload_schema_id,
            training_spec_payload_schema_version=training_spec_payload_schema_version,
            training_spec_payload_ref=effective_training_spec_ref,
            task_binding_spec=dict(task_binding_spec) if task_binding_spec is not None else None,
            checkpoint_writes=checkpoint_writes,
            barrier_artifacts=barrier_artifacts,
            history_events=history_events,
            final_metrics=final_metrics,
            issues=issues,
            status="cancelled" if cancellation is not None else "completed",
            cancellation=cancellation,
            runtime_telemetry=runtime_telemetry,
            execution_context=producer_context,
            diagnostics_ref=diagnostics_ref,
            completed_batches=diagnostics.completed_batches,
            projection_custody=projection_custody,
        )
        manifest_output_path = (
            collection_root / "manifest.json" if collection_root is not None else None
        )
        _preflight_manifest_emission(
            manifest,
            root=root_path,
            conflict_policy=manifest_conflict_policy,
            path=manifest_output_path,
        )
        _emit_training_diagnostics(diagnostics, path=diagnostics_path)
        manifest_path = _emit_manifest(
            manifest,
            root=root_path,
            conflict_policy=manifest_conflict_policy,
            path=manifest_output_path,
        )
        if run_event_emitter is not None:
            terminal_coordinate, _ = normalize_serialized_metrics(
                execution.coordinate,
                execution.coordinate.metrics,
                slot_axis_bindings,
            )
            run_event_emitter.emit_terminal(
                "complete",
                {
                    "run_id": resolved_run_id,
                    "status": "cancelled" if cancellation is not None else "completed",
                    "coordinate": terminal_coordinate,
                    "program_step": execution.coordinate.program_step,
                    "manifest_path": str(manifest_path),
                    "manifest_id": manifest.id,
                    "metrics": final_metrics,
                    "runtime_telemetry": runtime_telemetry,
                    **(
                        {"cancellation": cancellation.as_provenance()}
                        if cancellation is not None
                        else {}
                    ),
                },
            )
    except _ImmediateCancellation as interrupted:
        checkpoint_writes = checkpoint_store.writes
        barrier_artifacts = checkpoint_store.barrier_artifacts
        history_events = live_history_events
        final_metrics = _final_metrics({}, interrupted.coordinate, slot_axis_bindings)
        diagnostics = _build_training_diagnostics(
            manifest_id=exact_manifest_id,
            run_id=resolved_run_id,
            status="cancelled",
            final_coordinate=interrupted.coordinate,
            final_slots={},
            program=program,
            checkpoint_writes=checkpoint_writes,
            segment_start_batch=segment_start_batch,
            method_observation_origin_batch=method_observation_origin_batch,
            history_events=history_events,
            method_observations=method_observations,
            execution_context=producer_context,
            method_contract=method_contract,
            slot_axis_bindings=slot_axis_bindings,
        )
        diagnostics_path = _diagnostics_output_path(
            root_path=root_path,
            run_id=resolved_run_id,
            collection_root=collection_root,
        )
        diagnostics_ref = _training_diagnostics_ref(diagnostics, diagnostics_path)
        manifest = _build_manifest(
            run_spec,
            run_id=resolved_run_id,
            manifest_id=exact_manifest_id,
            root_path=root_path,
            training_spec_payload=embedded_training_spec_payload,
            training_spec_payload_kind=training_spec_payload_kind,
            training_spec_payload_schema_id=training_spec_payload_schema_id,
            training_spec_payload_schema_version=training_spec_payload_schema_version,
            training_spec_payload_ref=effective_training_spec_ref,
            task_binding_spec=dict(task_binding_spec) if task_binding_spec is not None else None,
            checkpoint_writes=checkpoint_writes,
            barrier_artifacts=barrier_artifacts,
            history_events=history_events,
            final_metrics=final_metrics,
            issues=issues,
            status="cancelled",
            cancellation=interrupted.decision,
            runtime_telemetry=runtime_telemetry,
            execution_context=producer_context,
            diagnostics_ref=diagnostics_ref,
            completed_batches=diagnostics.completed_batches,
            projection_custody=projection_custody,
        )
        manifest_output_path = (
            collection_root / "manifest.json" if collection_root is not None else None
        )
        _preflight_manifest_emission(
            manifest,
            root=root_path,
            conflict_policy=manifest_conflict_policy,
            path=manifest_output_path,
        )
        _emit_training_diagnostics(diagnostics, path=diagnostics_path)
        manifest_path = _emit_manifest(
            manifest,
            root=root_path,
            conflict_policy=manifest_conflict_policy,
            path=manifest_output_path,
        )
        if run_event_emitter is not None:
            terminal_coordinate, _ = normalize_serialized_metrics(
                interrupted.coordinate,
                interrupted.coordinate.metrics,
                slot_axis_bindings,
            )
            run_event_emitter.emit_terminal(
                "complete",
                {
                    "run_id": resolved_run_id,
                    "status": "cancelled",
                    "coordinate": terminal_coordinate,
                    "program_step": interrupted.coordinate.program_step,
                    "manifest_path": str(manifest_path),
                    "manifest_id": manifest.id,
                    "metrics": final_metrics,
                    "cancellation": interrupted.decision.as_provenance(),
                    "runtime_telemetry": runtime_telemetry,
                },
            )
        return TrainingRunExecutionResult(
            run_id=resolved_run_id,
            status="cancelled",
            manifest=manifest,
            manifest_path=manifest_path,
            diagnostics=diagnostics,
            diagnostics_path=diagnostics_path,
            final_slots={},
            final_coordinate=interrupted.coordinate,
            checkpoint_writes=tuple(checkpoint_writes),
            history_events=tuple(history_events),
            payload_binding_status=payload_binding_status,
        )
    except Exception as exc:
        if run_event_emitter is not None:
            run_event_emitter.emit_terminal(
                "failed",
                {
                    "run_id": resolved_run_id,
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
        raise
    return TrainingRunExecutionResult(
        run_id=resolved_run_id,
        status="cancelled" if cancellation is not None else "completed",
        manifest=manifest,
        manifest_path=manifest_path,
        diagnostics=diagnostics,
        diagnostics_path=diagnostics_path,
        final_slots=dict(execution.slots),
        final_coordinate=execution.coordinate,
        checkpoint_writes=tuple(checkpoint_writes),
        history_events=tuple(history_events),
        payload_binding_status=payload_binding_status,
    )


def _validate_spec(spec: TrainingRunSpec | Mapping[str, Any]) -> TrainingRunSpec:
    try:
        return validate_training_run_spec(spec)
    except ValidationError as exc:
        raise TrainingRunExecutorError(f"/training_run_spec validation failed: {exc}") from exc


def _validate_checkpoint_progress_policy(run_spec: TrainingRunSpec) -> None:
    policy = run_spec.checkpoint_progress
    for name in ("checkpoint_interval", "progress_interval"):
        interval = getattr(policy, name)
        if interval is not None and (
            isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0
        ):
            raise TrainingRunExecutorError(
                f"/checkpoint_progress/{name} must be a positive integer"
            )
    if policy.resume_from is not None:
        raise NotImplementedError(
            "/checkpoint_progress/resume_from is authored but not consumed by the native "
            "executor; use the explicit resume execution input until coordinate-selected "
            "resume is implemented"
        )
    if policy.checkpoint_slots is not None:
        raise NotImplementedError(
            "/checkpoint_progress/checkpoint_slots is authored but not consumed by the "
            "native executor; checkpoint slots remain governed by the phase program"
        )


def _validate_execution_context(
    context: NativeExecutionProducerContext | Mapping[str, Any] | None,
) -> NativeExecutionProducerContext | None:
    if context is None:
        return None
    try:
        validated = NativeExecutionProducerContext.model_validate(context)
    except ValidationError as exc:
        raise TrainingRunExecutorError(f"/execution_context validation failed: {exc}") from exc
    if validated.execution.row_provenance is None:
        raise TrainingRunExecutorError(
            "/execution_context/execution/row_provenance is required for native execution"
        )
    return validated


def prepare_training_manifest_metadata_projection(
    supplied: TrainingManifestMetadataProjection | Mapping[str, Any] | None,
    *,
    registry: TrainingMethodRegistry,
    run_spec: TrainingRunSpec,
    training_spec_payload: Mapping[str, Any] | None,
    training_spec_payload_kind: str,
    training_spec_payload_schema_id: str | None,
    training_spec_payload_schema_version: str | None,
    resolved_method: ResolvedTrainingMethod[Any] | None = None,
) -> TrainingManifestMetadataProjectionCustody | None:
    """Validate and bind explicit or descriptor metadata before any side effect."""
    if supplied is None:
        descriptor = resolved_method.descriptor if resolved_method is not None else None
        projector = descriptor.metadata_projector if descriptor is not None else None
        if projector is None:
            return None
        if not descriptor.package:
            raise TrainingRunExecutorError(
                "descriptor metadata projection requires a non-empty package identity"
            )
        try:
            projected = projector.project(resolved_method.payload)
            canonical_values = json.loads(
                training_spec_canonical_bytes(projected.model_dump(mode="json", exclude_none=True))
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise TrainingRunExecutorError(
                f"/descriptor/metadata_projector validation failed: {exc}"
            ) from exc
        _reject_reserved_projection_keys(canonical_values, run_spec=run_spec)
        source_payload, source_schema_id, source_schema_version = (
            _training_manifest_source_payload_identity(
                run_spec,
                training_spec_payload=training_spec_payload,
                training_spec_payload_kind=training_spec_payload_kind,
                training_spec_payload_schema_id=training_spec_payload_schema_id,
                training_spec_payload_schema_version=training_spec_payload_schema_version,
            )
        )
        return TrainingManifestMetadataProjectionCustody(
            source_payload_kind=training_spec_payload_kind,
            source_payload_schema_id=source_schema_id,
            source_payload_schema_version=source_schema_version,
            source_payload_sha256=training_spec_sha256(source_payload),
            projection_schema_id=projector.schema_id,
            projection_schema_version=projector.schema_version,
            values=canonical_values,
            values_sha256=training_spec_sha256(canonical_values),
            registration_owner=descriptor.owner,
            registration_package=descriptor.package,
        )
    raw_projection: Any = (
        supplied.model_dump(mode="json", exclude_none=True)
        if isinstance(supplied, TrainingManifestMetadataProjection)
        else supplied
    )
    try:
        # Canonicalization before Pydantic prevents key coercion and rejects
        # NaN, infinities, arrays, and other unsupported JSON objects.
        training_spec_canonical_bytes(raw_projection)
        projection = TrainingManifestMetadataProjection.model_validate(raw_projection)
    except (TypeError, ValueError, ValidationError) as exc:
        raise TrainingRunExecutorError(
            f"/manifest_metadata_projection validation failed: {exc}"
        ) from exc

    source_payload, source_schema_id, source_schema_version = (
        _training_manifest_source_payload_identity(
            run_spec,
            training_spec_payload=training_spec_payload,
            training_spec_payload_kind=training_spec_payload_kind,
            training_spec_payload_schema_id=training_spec_payload_schema_id,
            training_spec_payload_schema_version=training_spec_payload_schema_version,
        )
    )
    observed_source = (
        training_spec_payload_kind,
        source_schema_id,
        source_schema_version,
    )
    declared_source = (
        projection.source_payload_kind,
        projection.source_payload_schema_id,
        projection.source_payload_schema_version,
    )
    if declared_source != observed_source:
        raise TrainingRunExecutorError(
            "/manifest_metadata_projection source payload identity mismatch; "
            f"declared={declared_source!r}, observed={observed_source!r}"
        )
    observed_source_sha256 = training_spec_sha256(source_payload)
    if projection.source_payload_sha256 != observed_source_sha256:
        raise TrainingRunExecutorError(
            "/manifest_metadata_projection source payload sha256 mismatch; "
            f"declared={projection.source_payload_sha256}, "
            f"observed={observed_source_sha256}"
        )
    try:
        registration, values = registry.validate_manifest_metadata_projection(
            projection,
            path="/manifest_metadata_projection",
        )
        canonical_values = json.loads(training_spec_canonical_bytes(values))
    except (TypeError, ValueError, ValidationError) as exc:
        raise TrainingRunExecutorError(
            f"/manifest_metadata_projection validation failed: {exc}"
        ) from exc
    _reject_reserved_projection_keys(canonical_values, run_spec=run_spec)
    return TrainingManifestMetadataProjectionCustody(
        source_payload_kind=projection.source_payload_kind,
        source_payload_schema_id=projection.source_payload_schema_id,
        source_payload_schema_version=projection.source_payload_schema_version,
        source_payload_sha256=projection.source_payload_sha256,
        projection_schema_id=projection.projection_schema_id,
        projection_schema_version=projection.projection_schema_version,
        values=canonical_values,
        values_sha256=training_spec_sha256(canonical_values),
        registration_owner=registration.owner,
        registration_package=registration.package,
    )


def _training_manifest_source_payload_identity(
    run_spec: TrainingRunSpec,
    *,
    training_spec_payload: Mapping[str, Any] | None,
    training_spec_payload_kind: str,
    training_spec_payload_schema_id: str | None,
    training_spec_payload_schema_version: str | None,
) -> tuple[dict[str, Any], str, str]:
    """Return the exact embedded payload and its validated schema identity."""
    source_payload = dict(training_spec_payload or run_spec.model_dump(mode="json"))
    schema_id = training_spec_payload_schema_id
    schema_version = training_spec_payload_schema_version
    if training_spec_payload_kind == "TrainingRunSpec":
        schema_id = source_payload.get("schema_id")
        schema_version = source_payload.get("schema_version")
    if not isinstance(schema_id, str) or not isinstance(schema_version, str):
        raise TrainingRunExecutorError(
            "/manifest_metadata_projection source payload requires explicit schema identity"
        )
    return source_payload, schema_id, schema_version


def _reject_reserved_projection_keys(
    canonical_values: Mapping[str, Any],
    *,
    run_spec: TrainingRunSpec,
) -> None:
    owned_keys = _feedbax_owned_training_manifest_metadata(
        training_run_spec_schema_version=run_spec.schema_version,
        include_all_owned_keys=True,
    )
    collisions = sorted(
        key
        for key in canonical_values
        if key in owned_keys or key.startswith(_FEEDBAX_METADATA_NAMESPACE_PREFIX)
    )
    if collisions:
        raise TrainingRunExecutorError(
            "/manifest_metadata_projection reserved Feedbax metadata key collision; "
            f"keys={collisions!r}"
        )


def _feedbax_owned_training_manifest_metadata(
    *,
    training_run_spec_schema_version: str,
    environment_fingerprint: Any = _METADATA_VALUE_ABSENT,
    training_row_provenance: Any = _METADATA_VALUE_ABSENT,
    runtime_telemetry: Any = _METADATA_VALUE_ABSENT,
    include_all_owned_keys: bool = False,
) -> dict[str, Any]:
    """Build Feedbax-owned root metadata and define its collision policy.

    ``include_all_owned_keys`` exposes the complete key policy before runtime
    values exist, allowing projection validation to fail before output or
    training side effects. The ``feedbax_`` namespace is reserved for future
    Feedbax-owned root metadata so adding a key cannot silently create a
    downstream collision.
    """
    candidates = {
        "training_run_spec_schema_version": training_run_spec_schema_version,
        "environment_fingerprint": environment_fingerprint,
        "training_row_provenance": training_row_provenance,
        "runtime_telemetry": runtime_telemetry,
    }
    if include_all_owned_keys:
        return candidates
    return {key: value for key, value in candidates.items() if value is not _METADATA_VALUE_ABSENT}


def _validate_execution_payload_binding(
    supplied_spec: TrainingRunSpec | Mapping[str, Any],
    *,
    run_spec: TrainingRunSpec,
    execution_context: NativeExecutionProducerContext | None,
) -> Literal["verified", "not_bound"]:
    """Fail closed when an assembly envelope disagrees with the executed payload."""

    if execution_context is None:
        return "not_bound"
    payload = (
        run_spec.model_dump(mode="json", exclude_none=True)
        if isinstance(supplied_spec, TrainingRunSpec)
        else dict(supplied_spec)
    )
    ref = execution_context.execution.payload
    observed_schema_id = payload.get("schema_id")
    observed_schema_version = payload.get("schema_version")
    if observed_schema_id != ref.schema_id or observed_schema_version != ref.schema_version:
        raise TrainingRunExecutorError(
            "/execution_context/execution/payload schema identity does not match the "
            "executed TrainingRunSpec; "
            f"expected=({ref.schema_id!r}, {ref.schema_version!r}), "
            f"observed=({observed_schema_id!r}, {observed_schema_version!r})"
        )
    canonical = training_spec_canonical_bytes(payload)
    observed_sha256 = hashlib.sha256(canonical).hexdigest()
    if observed_sha256 != ref.sha256:
        raise TrainingRunExecutorError(
            "/execution_context/execution/payload sha256 does not match the executed "
            f"TrainingRunSpec; expected={ref.sha256}, observed={observed_sha256}"
        )
    digest_prefix = "artifact://sha256/"
    if ref.artifact_id.startswith(digest_prefix):
        artifact_digest = ref.artifact_id.removeprefix(digest_prefix)
        if artifact_digest != ref.sha256:
            raise TrainingRunExecutorError(
                "/execution_context/execution/payload artifact_id digest disagrees with "
                f"sha256; artifact_id={artifact_digest}, sha256={ref.sha256}"
            )
    if ref.uri is None:
        return "verified"
    custody_path = _local_custody_path(ref.uri)
    if custody_path is None:
        return "verified"
    try:
        if not custody_path.is_file():
            raise TrainingRunExecutorError(
                "/execution_context/execution/payload local custody URI does not "
                f"resolve to a readable file: {ref.uri!r}"
            )
        custody_bytes = custody_path.read_bytes()
    except TrainingRunExecutorError:
        raise
    except OSError as exc:
        raise TrainingRunExecutorError(
            f"/execution_context/execution/payload local custody URI could not be read: {ref.uri!r}"
        ) from exc
    custody_sha256 = hashlib.sha256(custody_bytes).hexdigest()
    if custody_sha256 != ref.sha256:
        raise TrainingRunExecutorError(
            "/execution_context/execution/payload custody bytes fail sha256 validation; "
            f"expected={ref.sha256}, observed={custody_sha256}, uri={ref.uri!r}"
        )
    try:
        custody_payload = json.loads(custody_bytes)
    except json.JSONDecodeError as exc:
        raise TrainingRunExecutorError(
            "/execution_context/execution/payload custody bytes are not valid JSON"
        ) from exc
    if training_spec_canonical_bytes(custody_payload) != canonical:
        raise TrainingRunExecutorError(
            "/execution_context/execution/payload custody document differs from the "
            "executed TrainingRunSpec"
        )
    return "verified"


def _authoritative_training_spec_payload_ref(
    run_spec: TrainingRunSpec,
    *,
    training_spec_payload: Mapping[str, Any] | None,
    training_spec_payload_kind: str,
    training_spec_payload_schema_id: str | None,
    training_spec_payload_schema_version: str | None,
    execution_context: NativeExecutionProducerContext | None,
    explicit_ref: str | None,
) -> str | None:
    """Return the execution-envelope ref only for the same embedded payload."""
    if execution_context is None:
        return None
    payload = dict(training_spec_payload or run_spec.model_dump(mode="json"))
    if training_spec_payload_kind == "TrainingRunSpec":
        schema_id = payload.get("schema_id")
        schema_version = payload.get("schema_version")
    else:
        schema_id = training_spec_payload_schema_id
        schema_version = training_spec_payload_schema_version
    authoritative = execution_context.execution.payload
    identities_match = (
        schema_id == authoritative.schema_id
        and schema_version == authoritative.schema_version
        and training_spec_sha256(payload) == authoritative.sha256
    )
    if identities_match:
        return authoritative.artifact_id
    if explicit_ref is not None:
        raise TrainingRunExecutorError(
            "/training_spec_payload_ref cannot identify a payload that disagrees with "
            "the authoritative execution envelope"
        )
    return None


def _validated_optimizer_step(
    extractor: Callable[[Any, Mapping[str, Any]], int],
    payload: Any,
    runtime: Mapping[str, Any],
) -> int:
    """Invoke a descriptor extractor and enforce its runtime-only int contract."""
    try:
        value = extractor(payload, runtime)
    except Exception as exc:
        raise TrainingRunExecutorError(
            f"/descriptor/optimizer_step_extractor failed: {exc}"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingRunExecutorError(
            "/descriptor/optimizer_step_extractor must return a non-bool integer; "
            f"observed={value!r}"
        )
    if value < 0:
        raise TrainingRunExecutorError(
            "/descriptor/optimizer_step_extractor must return a non-negative integer; "
            f"observed={value}"
        )
    return value


def _instance_runtime_slots(
    runtime: Mapping[str, Any],
    bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
    instance: int,
) -> dict[str, Any]:
    mapped = {
        name
        for name, slot_bindings in bindings.items()
        if slot_bindings and slot_bindings[0].mode == "mapped"
    }
    return {
        name: jt.map(
            lambda leaf: leaf[instance] if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
            value,
        )
        if name in mapped
        else value
        for name, value in runtime.items()
    }


def _synchronized_optimizer_step(
    extractor: Callable[[Any, Mapping[str, Any]], int],
    payload: Any,
    runtime: Mapping[str, Any],
    bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
) -> int:
    mapped_bindings = [
        value[0] for value in bindings.values() if value and value[0].mode == "mapped"
    ]
    if not mapped_bindings:
        return _validated_optimizer_step(extractor, payload, runtime)
    values = [
        _validated_optimizer_step(
            extractor,
            payload,
            _instance_runtime_slots(runtime, bindings, instance),
        )
        for instance in range(mapped_bindings[0].size)
    ]
    if any(value != values[0] for value in values[1:]):
        raise TrainingRunExecutorError(
            f"/descriptor/optimizer_step_extractor mapped authorities diverged: {values!r}"
        )
    return values[0]


def _bind_restored_schedule_context(
    *,
    run_spec: TrainingRunSpec,
    resolved_method: ResolvedTrainingMethod[Any],
    slots: Mapping[str, Any],
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
    kernel_context: Mapping[str, Any],
    binder: RestoredScheduleContextBinder | None,
) -> dict[str, Any]:
    continuation = run_spec.checkpoint_progress.continuation
    descriptor = resolved_method.descriptor
    if continuation is None:
        return dict(kernel_context)
    current_step = continuation.source_completed_batches
    projection = project_training_schedules(
        run_spec,
        coordinates=(current_step, current_step + 1, current_step + 2),
        lineage=CheckpointSegmentLineage(
            start_batch=current_step,
            segment_batch_count=continuation.additional_batches or 0,
            parent_transaction_id="restored" if current_step else None,
        ),
        resolved_method=resolved_method,
    )
    projected_context = {
        **kernel_context,
        "schedule_projection": projection.model_dump(mode="json"),
    }
    # Schedule projection already enforces both descriptor requirements.
    assert descriptor is not None and descriptor.optimizer_spec_projector is not None
    optimizer_spec = OptimizerSpec.model_validate(
        descriptor.optimizer_spec_projector(resolved_method.payload)
    )
    schedule = optimizer_spec.lr_schedule
    if schedule is None or schedule.kind == "constant":
        return projected_context
    if descriptor.optimizer_step_extractor is None:
        raise TrainingRunExecutorError(
            "scheduled continuation requires /descriptor/optimizer_step_extractor"
        )
    restored_count = _synchronized_optimizer_step(
        descriptor.optimizer_step_extractor,
        resolved_method.payload,
        slots,
        slot_axis_bindings,
    )
    if restored_count != current_step:
        raise TrainingRunExecutorError(
            "restored optimizer count disagrees with continuation source; "
            f"optimizer_count={restored_count}, source_completed_batches={current_step}"
        )
    if binder is None:
        raise TrainingRunExecutorError(
            "nonconstant scheduled continuation requires restored_schedule_context_binder"
        )
    window = resolve_schedule_window(
        schedule.origin,
        lineage=CheckpointSegmentLineage(
            start_batch=current_step,
            segment_batch_count=continuation.additional_batches or 0,
            parent_transaction_id="restored" if current_step else None,
        ),
        duration=schedule.total_steps,
        allow_inert=schedule.allow_inert,
    )
    patch = binder(slots, window.start_batch, current_step, restored_count)
    if not isinstance(patch, Mapping):
        raise TrainingRunExecutorError("restored_schedule_context_binder must return a mapping")
    reserved = sorted(set(patch).intersection(_RESERVED_KERNEL_CONTEXT_KEYS))
    if reserved:
        raise TrainingRunExecutorError(
            f"restored_schedule_context_binder returned executor-reserved keys: {reserved!r}"
        )
    return {**projected_context, **patch}


def _local_custody_path(uri: str) -> Path | None:
    """Resolve locally addressable custody URIs and ignore remote registry schemes."""

    parsed = urlsplit(uri)
    if parsed.scheme == "":
        return Path(uri)
    if parsed.scheme != "file":
        return None
    if parsed.netloc not in {"", "localhost"}:
        raise TrainingRunExecutorError(
            "/execution_context/execution/payload file URI must address the local host; "
            f"uri={uri!r}"
        )
    return Path(unquote(parsed.path))


def _collection_root(context: NativeExecutionProducerContext | None) -> Path | None:
    configured = context.collection_root if context is not None else None
    configured = configured or os.environ.get("FEEDBAX_ROW_DIR")
    return Path(configured) if configured else None


def _default_run_id(spec: TrainingRunSpec) -> str:
    payload = spec.model_dump(mode="json", exclude_none=True)
    digest = sha256_bytes(canonical_json_bytes(payload))
    return digest[:32]


def _validate_declarations_match_spec(spec: TrainingRunSpec, resolved_method_ref: str) -> None:
    requested = spec.method_ref.key
    if resolved_method_ref != requested:
        raise TrainingRunExecutorError(
            "/method_ref registry declaration mismatch: "
            f"resolved {resolved_method_ref!r}, expected {requested!r}"
        )
    if spec.worker_execution.method_contract.method_ref != resolved_method_ref:
        raise TrainingRunExecutorError(
            "/worker_execution/method_contract/method_ref must match registry declaration"
        )


def _graph_inline(graph: GraphTopologySourceSpec) -> dict[str, Any] | None:
    return dict(graph.inline) if graph.inline is not None else None


def _lower_objective(
    spec: TrainingRunSpec,
    *,
    graph_inline: dict[str, Any] | None,
    loss_service: LossService,
) -> Any:
    try:
        from feedbax.contracts.graph import GraphSpec

        graph = GraphSpec.model_validate(graph_inline) if graph_inline is not None else None
        return loss_service.lower_objective_slot(
            spec.objective,
            graph=graph,
            trial_axis="batch",
            path="/objective",
        )
    except ObjectiveLoweringError as exc:
        raise TrainingRunExecutorError(str(exc)) from exc


def _initial_slots(
    initial_slots: Mapping[str, Any] | None,
    *,
    lowered_objective: Any,
    executor_owned_slots: Sequence[str],
) -> dict[str, Any]:
    if initial_slots is None:
        raise TrainingRunExecutorError("/initial_slots are required for native execution")
    slots = dict(initial_slots)
    for slot in executor_owned_slots:
        slots.setdefault(slot, lowered_objective)
    return slots


def _normalize_batch_progress_slot(
    slots: Mapping[str, Any],
    *,
    program: Any,
) -> dict[str, Any]:
    """Give a declared scalar batch authority a value-independent checkpoint ABI."""
    authority = program.batch_progress
    if authority is None or authority.field_path or authority.slot not in slots:
        return dict(slots)
    normalized = dict(slots)
    try:
        normalized[authority.slot] = jnp.asarray(normalized[authority.slot], dtype=jnp.int32)
    except (TypeError, ValueError) as exc:
        raise TrainingRunExecutorError(
            "/phase_program/batch_progress/slot "
            f"{authority.slot!r} must initialize to one integer scalar"
        ) from exc
    return normalized


def _executor_nan_guard(
    *,
    run_spec: TrainingRunSpec,
    custody_root: Path,
    program: Any,
    expected_slots: Mapping[str, Any],
    resume_slot_transform: ResumeSlotTransform | None,
) -> Callable[[Mapping[str, Any], ProgressCoordinate, Mapping[str, Any]], StepGuardResult | None]:
    def guard(
        slots: Mapping[str, Any],
        coordinate: ProgressCoordinate,
        _context: Mapping[str, Any],
    ) -> StepGuardResult | None:
        nan_metrics = _nan_metric_names(coordinate.metrics)
        if not nan_metrics:
            return None

        program_step = coordinate.program_step
        step = coordinate.inner_step
        message = (
            f"NaN detected after program_step {program_step} inner_step {step}; "
            f"metrics={nan_metrics!r}; on_nan={run_spec.on_nan!r}"
        )
        if run_spec.on_nan == "raise":
            raise FloatingPointError(message)

        try:
            loaded = load_latest_checkpoint(
                custody_root,
                expected_run_spec=run_spec,
                expected_phase_program=program,
                expected_slots=expected_slots,
                resume_slot_transform=resume_slot_transform,
            )
        except Exception as exc:
            raise FloatingPointError(
                f"{message}; no custody-consistent checkpoint could be restored"
            ) from exc

        metric_slots = set(coordinate.metrics)
        restored_slots = {
            slot: value
            for slot, value in slots.items()
            if slot not in loaded.slots and slot not in metric_slots
        }
        restored_slots.update(loaded.slots)
        return StepGuardResult(
            halt=True,
            slots=restored_slots,
            coordinate=loaded.manifest.completed_coordinate,
        )

    return guard


def _nan_metric_names(metrics: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    flags: list[jax.Array] = []
    for name, value in sorted(metrics.items()):
        leaf_flags: list[jax.Array] = []
        for leaf in jt.leaves(value, is_leaf=lambda item: item is None):
            if leaf is None:
                continue
            try:
                array = jnp.asarray(leaf)
            except (TypeError, ValueError):
                continue
            if jnp.issubdtype(array.dtype, jnp.number):
                leaf_flags.append(jnp.any(jnp.isnan(array)))
        if leaf_flags:
            names.append(name)
            flags.append(jnp.any(jnp.stack(leaf_flags)))
    if not flags:
        return []
    observed = jax.device_get(jnp.stack(flags))
    return [name for name, flag in zip(names, observed, strict=True) if flag]


def _checkpoint_root(
    *,
    root_path: Path,
    configured_root: Path | str | None,
    run_id: str,
) -> Path:
    if configured_root is not None:
        return Path(configured_root)
    return root_path / "checkpoints" / run_id


def _checkpoint_visit_ordinal(metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("barrier_visit_ordinal")
    return value if isinstance(value, int) and value >= 0 else None


def _history_event(
    coordinate: ProgressCoordinate,
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
) -> dict[str, Any]:
    coordinate_payload, metrics = normalize_serialized_metrics(
        coordinate,
        coordinate.metrics,
        slot_axis_bindings,
    )
    return {
        "type": "training_progress",
        "coordinate": coordinate_payload,
        "metrics": metrics,
    }


def _live_progress_callback(
    progress_callback: ProgressCallback | None,
    history_events: list[dict[str, Any]],
    *,
    run_event_emitter: RunEventEmitter | None = None,
    total_batches: int | None = None,
    started_at: float | None = None,
    cancellation_observer: Callable[[ProgressCoordinate], None] | None = None,
    runtime_telemetry: dict[str, Any] | None = None,
    cache_before: Mapping[str, Any] | None = None,
    start_semantics: str = "executor_entry",
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]] | None = None,
) -> Callable[[ProgressCoordinate], None]:
    ready_emitted = False

    def emit(coordinate: ProgressCoordinate) -> None:
        nonlocal ready_emitted
        bindings = dict(slot_axis_bindings or {})
        event = _history_event(coordinate, bindings)
        if runtime_telemetry is not None and not runtime_telemetry:
            runtime_telemetry.update(
                _first_progress_telemetry(
                    started_at=started_at if started_at is not None else time.perf_counter(),
                    start_semantics=start_semantics,
                    cache_before=cache_before or {},
                )
            )
            event["runtime_telemetry"] = deepcopy(runtime_telemetry)
        history_events.append(event)
        if run_event_emitter is not None:
            if not ready_emitted:
                ready_emitted = True
                ready_coordinate, ready_metrics = normalize_serialized_metrics(
                    coordinate,
                    coordinate.metrics,
                    bindings,
                )
                ready_payload: dict[str, Any] = {
                    "run_id": coordinate.run_id,
                    "phase": coordinate.phase,
                    "program_step": coordinate.program_step,
                    "coordinate": ready_coordinate,
                    "runtime_telemetry": deepcopy(runtime_telemetry or {}),
                }
                if any(
                    name in coordinate.metrics
                    and axis_bindings
                    and axis_bindings[0].mode == "mapped"
                    for name, axis_bindings in bindings.items()
                ):
                    ready_payload["metrics"] = ready_metrics
                run_event_emitter.emit(
                    "ready",
                    ready_payload,
                )
            elapsed_seconds = time.perf_counter() - started_at if started_at is not None else None
            payload = _run_progress_event_payload(
                event,
                coordinate=coordinate,
                total_batches=total_batches,
                elapsed_seconds=elapsed_seconds,
            )
            run_event_emitter.emit_progress(
                payload,
                batch=coordinate.program_step,
                total_batches=total_batches,
            )
        if progress_callback is not None:
            progress_callback(deepcopy(event))
        if cancellation_observer is not None:
            cancellation_observer(coordinate)

    return emit


def _run_progress_event_payload(
    event: Mapping[str, Any],
    *,
    coordinate: ProgressCoordinate,
    total_batches: int | None,
    elapsed_seconds: float | None,
) -> dict[str, Any]:
    metrics = event.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    coordinate_payload = event.get("coordinate")
    payload: dict[str, Any] = {
        "run_id": coordinate.run_id,
        "phase": coordinate.phase,
        "program_step": coordinate.program_step,
        "total_batches": total_batches,
        "coordinate": coordinate_payload,
        "metrics": dict(metrics),
        "status": "running",
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = elapsed_seconds
    loss = metrics.get("train_loss")
    if loss is not None and not (
        isinstance(loss, Mapping)
        and loss.get("schema_id") == "feedbax.manifest.mapped_metric_value"
    ):
        try:
            payload["loss"] = float(jax.device_get(loss))
        except (TypeError, ValueError):
            payload["loss"] = loss
    return payload


def _prepare_barrier_artifacts(
    checkpoint: PhaseCheckpoint,
    artifact_sinks: Sequence[BarrierArtifactSinkSpec],
    *,
    run_id: str,
) -> list[_PreparedBarrierArtifact]:
    prepared: list[_PreparedBarrierArtifact] = []
    for sink in artifact_sinks:
        if sink.slot not in checkpoint.slots:
            if sink.required:
                raise TrainingRunExecutorError(
                    f"/checkpoint_barriers/{checkpoint.barrier}/artifact_sinks/{sink.slot}: "
                    "required artifact sink slot was not captured"
                )
            continue
        data, media_type = _barrier_artifact_bytes(checkpoint.slots[sink.slot], sink)
        suffix = _barrier_artifact_suffix(sink)
        prepared.append(
            _PreparedBarrierArtifact(
                sink=sink,
                data=data,
                logical_name=_barrier_artifact_logical_name(
                    run_id=run_id,
                    barrier=checkpoint.barrier,
                    visit_ordinal=checkpoint.visit_ordinal,
                    sink=sink,
                    suffix=suffix,
                ),
                media_type=media_type,
                suffix=suffix,
            )
        )
    return prepared


def _barrier_artifact_bytes(
    value: Any,
    sink: BarrierArtifactSinkSpec,
) -> tuple[bytes, str]:
    if sink.encoding == "raw":
        if isinstance(value, bytes):
            return value, sink.media_type
        if isinstance(value, bytearray):
            return bytes(value), sink.media_type
        if isinstance(value, memoryview):
            return value.tobytes(), sink.media_type
        raise TrainingRunExecutorError(
            f"/artifact_sinks/{sink.slot}/encoding: raw artifact sinks require bytes-like "
            f"slot values; found {type(value).__name__}"
        )
    if sink.encoding == "json":
        media_type = (
            "application/json" if sink.media_type == "application/octet-stream" else sink.media_type
        )
        return canonical_json_bytes(value), media_type
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), sink.media_type


def _barrier_artifact_suffix(sink: BarrierArtifactSinkSpec) -> str:
    if sink.suffix is not None:
        return sink.suffix
    if sink.logical_name is not None:
        suffix = Path(sink.logical_name).suffix
        if suffix:
            return suffix
    if sink.encoding == "json":
        return ".json"
    if sink.encoding == "pickle":
        return ".pkl"
    return ""


def _barrier_artifact_logical_name(
    *,
    run_id: str,
    barrier: str,
    visit_ordinal: int | None,
    sink: BarrierArtifactSinkSpec,
    suffix: str,
) -> str:
    source_name = sink.logical_name or sink.slot
    stem = Path(source_name).stem if Path(source_name).suffix else source_name
    visit = "unknown" if visit_ordinal is None else str(visit_ordinal)
    parts = [
        safe_manifest_key(run_id),
        safe_manifest_key(barrier),
        f"visit-{visit}",
        safe_manifest_key(sink.slot),
        safe_manifest_key(stem),
    ]
    return "_".join(parts) + suffix


def _final_metrics(
    slots: Mapping[str, Any],
    coordinate: ProgressCoordinate,
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
) -> dict[str, Any]:
    metrics = dict(coordinate.metrics)
    for key, value in slots.items():
        if key.endswith("loss"):
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                metrics[key] = value
    metrics.setdefault("program_step", coordinate.program_step)
    _, normalized = normalize_serialized_metrics(coordinate, metrics, slot_axis_bindings)
    return normalized


def _build_training_diagnostics(
    *,
    manifest_id: str,
    run_id: str,
    status: Literal["completed", "cancelled"],
    final_coordinate: ProgressCoordinate,
    final_slots: Mapping[str, Any],
    program: Any,
    checkpoint_writes: Sequence[CheckpointWriteResult],
    segment_start_batch: int,
    method_observation_origin_batch: int,
    history_events: Sequence[Mapping[str, Any]],
    method_observations: Sequence[Mapping[str, Any]],
    execution_context: NativeExecutionProducerContext | None,
    method_contract: MethodContractSpec,
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]] | None = None,
) -> TrainingDiagnostics:
    transactions: list[CheckpointTransactionDiagnostic] = []
    for write in checkpoint_writes:
        cumulative_completed = write.manifest.completed_training_batches
        if cumulative_completed is None:
            cumulative_completed = write.manifest.completed_coordinate.program_step
        transactions.append(
            CheckpointTransactionDiagnostic(
                transaction_id=write.manifest.transaction_id,
                completed_batches=max(0, cumulative_completed - segment_start_batch),
                cumulative_completed_batches=cumulative_completed,
                coordinate=write.manifest.completed_coordinate,
            )
        )
    checkpoint_completed_batches = (
        transactions[-1].cumulative_completed_batches if transactions else None
    )
    cumulative_completed_batches = _terminal_completed_batches(
        program=program,
        final_slots=final_slots,
        fallback=(
            final_coordinate.completed_batches
            if final_coordinate.completed_batches is not None
            else checkpoint_completed_batches
            if checkpoint_completed_batches is not None
            else final_coordinate.program_step
        ),
        require_authority=status == "completed",
        slot_axis_bindings=slot_axis_bindings,
    )
    segment_completed_batches = max(
        0,
        cumulative_completed_batches - segment_start_batch,
    )
    diagnostic_input = execution_context.diagnostics if execution_context is not None else None
    seeds = list(diagnostic_input.seeds) if diagnostic_input is not None else []
    if (
        execution_context is not None
        and execution_context.execution.row_provenance is not None
        and execution_context.execution.row_provenance.seed is not None
        and execution_context.execution.row_provenance.seed not in seeds
    ):
        seeds.append(execution_context.execution.row_provenance.seed)
    lr_trace = _realized_lr_trace(
        history_events,
        declared=(diagnostic_input.lr_trace if diagnostic_input is not None else ()),
    )
    declaration = method_contract.training_diagnostics
    method_trace = _method_training_trace(
        declaration,
        method_ref=method_contract.method_ref,
        method_observations=method_observations,
        observation_origin_batch=method_observation_origin_batch,
        completed_batches=cumulative_completed_batches,
        replica_count=next(
            (
                axis.size
                for axis in method_contract.axes
                if declaration is not None and axis.name == declaration.replica_axis
            ),
            None,
        ),
    )
    return TrainingDiagnostics(
        schema_version=(
            TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3
            if method_trace is not None
            else TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2
        ),
        manifest_id=manifest_id,
        run_id=run_id,
        terminal_status=status,
        completed_batches=cumulative_completed_batches,
        segment_completed_batches=segment_completed_batches,
        cumulative_completed_batches=cumulative_completed_batches,
        seeds=seeds,
        lr_trace=lr_trace,
        checkpoint_coordinates=[item.completed_batches for item in transactions],
        checkpoint_transactions=transactions,
        resume_context=(diagnostic_input.resume_context if diagnostic_input is not None else None),
        optimizer_build_context=(
            diagnostic_input.optimizer_build_context if diagnostic_input is not None else None
        ),
        method_trace=method_trace,
        metadata=(dict(diagnostic_input.metadata) if diagnostic_input is not None else {}),
    )


def _method_training_trace(
    declaration: MethodTrainingDiagnosticsSpec | None,
    *,
    method_ref: str,
    method_observations: Sequence[Mapping[str, Any]],
    observation_origin_batch: int,
    completed_batches: int,
    replica_count: int | None,
) -> MethodTrainingTrace | None:
    if declaration is None:
        return None
    if replica_count is None:
        raise TrainingRunExecutorError("method trace lacks sized replica-axis authority")
    observed: dict[int, list[MethodTrainingTraceRecord]] = {}
    for event in method_observations:
        coordinate = event.get("coordinate")
        metrics = event.get("metrics")
        if not isinstance(coordinate, Mapping) or not isinstance(metrics, Mapping):
            raise TrainingRunExecutorError("method trace requires typed progress events")
        batch = coordinate.get("completed_batches")
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise TrainingRunExecutorError("method trace progress is missing completed_batches")
        if batch <= observation_origin_batch or batch > completed_batches:
            raise TrainingRunExecutorError(
                f"method trace completed batch {batch} is outside the active segment"
            )
        if batch in observed:
            raise TrainingRunExecutorError(f"method trace duplicates completed batch {batch}")
        payload = metrics.get(declaration.metric_payload_slot)
        if not isinstance(payload, Mapping) or payload.get("schema_id") not in {
            "feedbax.manifest.mapped_metric_value",
            STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_ID,
        }:
            raise TrainingRunExecutorError("method trace metric payload is not a mapped metric")
        axes = payload.get("axes")
        if not isinstance(axes, list) or len(axes) != 1:
            raise TrainingRunExecutorError("method trace metric requires one mapped replica axis")
        axis = axes[0]
        if (
            not isinstance(axis, Mapping)
            or axis.get("axis") != declaration.replica_axis
            or (axis.get("role"), axis.get("level"), axis.get("mode"), axis.get("array_axis"))
            != ("replicate", 0, "mapped", 0)
        ):
            raise TrainingRunExecutorError("method trace mapped axis does not match replica_axis")
        size = axis.get("size")
        structured = payload.get("schema_id") == STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_ID
        expected_version = (
            STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_VERSION
            if structured
            else MAPPED_METRIC_VALUE_SCHEMA_VERSION
        )
        if payload.get("schema_version") != expected_version:
            raise TrainingRunExecutorError("method trace metric schema version is unsupported")
        values = payload.get("value")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size != replica_count
        ):
            raise TrainingRunExecutorError("method trace does not cover the replica axis")
        if structured:
            if not isinstance(values, Mapping):
                raise TrainingRunExecutorError("structured method trace value must be named")
            replica_values = [
                _structured_replica_value(values, index=index, size=size) for index in range(size)
            ]
        else:
            array = np.asarray(values)
            if array.shape[:1] != (size,):
                raise TrainingRunExecutorError("method trace does not cover the replica axis")
            replica_values = [array[index].tolist() for index in range(size)]
        observed[batch] = [
            MethodTrainingTraceRecord(
                completed_batch=batch,
                replica_index=index,
                value=replica_values[index],
            )
            for index in range(size)
        ]
    expected = set(range(observation_origin_batch + 1, completed_batches + 1))
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        raise TrainingRunExecutorError(f"method trace has incomplete batch coverage: {missing!r}")
    return MethodTrainingTrace(
        method_ref=method_ref,
        trace_schema_id=declaration.trace_schema_id,
        trace_schema_version=declaration.trace_schema_version,
        measurement_basis=declaration.measurement_basis,
        metric_payload_slot=declaration.metric_payload_slot,
        replica_axis=declaration.replica_axis,
        records=[record for batch in sorted(observed) for record in observed[batch]],
    )


def _structured_replica_value(value: Mapping[str, Any], *, index: int, size: int) -> dict[str, Any]:
    if not value:
        raise TrainingRunExecutorError("structured method trace named nodes cannot be empty")
    replica: dict[str, Any] = {}
    for name, item in value.items():
        if not isinstance(name, str):
            raise TrainingRunExecutorError("structured method trace names must be strings")
        if isinstance(item, Mapping):
            replica[name] = _structured_replica_value(item, index=index, size=size)
            continue
        if not isinstance(item, list) or len(item) != size:
            raise TrainingRunExecutorError("structured method trace does not cover the replica axis")
        replica[name] = item[index]
    return replica


def _same_row_resume_start_batch(manifest: CheckpointTransactionManifest) -> int:
    """Return the authoritative segment origin for an operational resume."""

    completed_batches = manifest.completed_training_batches
    if (
        isinstance(completed_batches, bool)
        or not isinstance(completed_batches, int)
        or completed_batches < 0
    ):
        raise TrainingRunExecutorError(
            "same-row resume checkpoint lacks authoritative completed-training progress"
        )
    lineage_total = (
        manifest.segment_lineage.start_batch + manifest.segment_lineage.segment_batch_count
    )
    if lineage_total != completed_batches:
        raise TrainingRunExecutorError(
            "same-row resume checkpoint progress disagrees with segment lineage: "
            f"completed_batches={completed_batches} lineage_total={lineage_total}"
        )
    return completed_batches


def _terminal_completed_batches(
    *,
    program: Any,
    final_slots: Mapping[str, Any],
    fallback: int,
    require_authority: bool,
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]] | None = None,
) -> int:
    """Read the terminal cumulative batch count from the declared authority."""

    authority = program.batch_progress
    if authority is None:
        return fallback
    path = "/phase_program/batch_progress"
    if authority.slot not in final_slots:
        if not require_authority:
            return fallback
        raise TrainingRunExecutorError(
            f"{path}/slot={authority.slot!r} is missing from terminal slots"
        )
    bindings = dict(slot_axis_bindings or {})
    authority_bindings = bindings.get(authority.slot, ())
    if authority_bindings and authority_bindings[0].mode == "mapped":
        values = [
            _terminal_completed_batches(
                program=program,
                final_slots=_instance_runtime_slots(final_slots, bindings, instance),
                fallback=fallback,
                require_authority=require_authority,
            )
            for instance in range(authority_bindings[0].size)
        ]
        if any(value != values[0] for value in values[1:]):
            raise TrainingRunExecutorError(f"{path} mapped authorities diverged: {values!r}")
        return values[0]
    value = final_slots[authority.slot]
    for index, segment in enumerate(authority.field_path):
        segment_path = f"{path}/field_path/{index}"
        if isinstance(value, Mapping):
            if segment not in value:
                raise TrainingRunExecutorError(
                    f"{segment_path}={segment!r} is missing in slot {authority.slot!r}"
                )
            value = value[segment]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not isinstance(segment, int) or segment >= len(value):
                raise TrainingRunExecutorError(
                    f"{segment_path}={segment!r} is not a valid index in slot {authority.slot!r}"
                )
            value = value[segment]
        else:
            raise TrainingRunExecutorError(
                f"{segment_path} cannot traverse non-container in slot {authority.slot!r}"
            )
    try:
        scalar_array = np.asarray(jax.device_get(value))
        if scalar_array.size != 1:
            raise ValueError(f"expected one scalar, got shape={scalar_array.shape!r}")
        batches = int(scalar_array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingRunExecutorError(
            f"{path} resolved slot {authority.slot!r} to non-integer {value!r}"
        ) from exc
    if batches < 0:
        raise TrainingRunExecutorError(
            f"{path} resolved slot {authority.slot!r} to negative batches={batches}"
        )
    return batches


def _realized_lr_trace(
    history_events: Sequence[Mapping[str, Any]],
    *,
    declared: Sequence[LearningRateDiagnostic],
) -> list[LearningRateDiagnostic]:
    observed: dict[tuple[tuple[tuple[str, int], ...], int], LearningRateDiagnostic] = {}

    def add(sample: LearningRateDiagnostic) -> None:
        coordinates = tuple(
            (coordinate.axis, coordinate.index)
            for coordinate in (sample.axis_coordinates or ())
        )
        key = (coordinates, sample.step)
        if key in observed:
            raise TrainingRunExecutorError(
                f"learning-rate diagnostics duplicate coordinate/step identity {key!r}"
            )
        observed[key] = sample

    for sample in declared:
        add(sample)
    for event in history_events:
        coordinate = event.get("coordinate")
        metrics = event.get("metrics")
        if not isinstance(coordinate, Mapping) or not isinstance(metrics, Mapping):
            continue
        learning_rate = metrics.get("learning_rate", metrics.get("lr"))
        if learning_rate is None:
            continue
        step = coordinate.get("completed_batches")
        if step is None:
            raise TrainingRunExecutorError(
                "learning-rate diagnostics require coordinate.completed_batches "
                "training-batch authority"
            )
        if (
            isinstance(learning_rate, Mapping)
            and learning_rate.get("schema_id") == "feedbax.manifest.mapped_metric_value"
        ):
            axes = learning_rate.get("axes")
            values = np.asarray(learning_rate.get("value"))
            if not isinstance(axes, list) or len(axes) != 1:
                raise TrainingRunExecutorError("mapped learning-rate metric requires one axis")
            axis = axes[0]
            if values.ndim < 1 or values.shape[0] != axis.get("size"):
                raise TrainingRunExecutorError(
                    "mapped learning-rate metric does not cover every axis coordinate"
                )
            for index in range(values.shape[0]):
                add(
                    LearningRateDiagnostic(
                        step=int(step),
                        learning_rate=float(values[index]),
                        axis_coordinates=(
                            AxisCoordinateSpec(axis=str(axis["axis"]), index=index),
                        ),
                    )
                )
        else:
            add(
                LearningRateDiagnostic(
                    step=int(step),
                    learning_rate=float(jax.device_get(learning_rate)),
                )
            )
    return [observed[key] for key in sorted(observed)]


def _diagnostics_output_path(
    *,
    root_path: Path,
    run_id: str,
    collection_root: Path | None,
) -> Path:
    if collection_root is not None:
        return collection_root / "training-diagnostics.json"
    return root_path / "diagnostics" / f"{safe_manifest_key(run_id)}.json"


def _training_diagnostics_ref(
    diagnostics: TrainingDiagnostics,
    path: Path,
) -> ArtifactRef:
    payload = diagnostics.model_dump_json(indent=2, exclude_none=True).encode("utf-8") + b"\n"
    return ArtifactRef(
        role="training_diagnostics",
        logical_name="training-diagnostics.json",
        sha256=sha256_bytes(payload),
        media_type="application/json",
        size_bytes=len(payload),
        uri=str(path),
        metadata={
            "schema_id": diagnostics.schema_id,
            "schema_version": diagnostics.schema_version,
        },
    )


def _preflight_training_diagnostics_emission(path: Path) -> None:
    """Reject existing diagnostics before training because output is not yet knowable."""

    if path.exists():
        raise DiagnosticsEmissionConflictError(
            f"training diagnostics output already exists before execution: {path}"
        )


def _emit_training_diagnostics(diagnostics: TrainingDiagnostics, *, path: Path) -> Path:
    payload = diagnostics.model_dump_json(indent=2, exclude_none=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return path
        raise DiagnosticsEmissionConflictError(
            f"training diagnostics already exist with different content: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _build_manifest(
    spec: TrainingRunSpec,
    *,
    run_id: str,
    manifest_id: str,
    root_path: Path,
    training_spec_payload: dict[str, Any],
    training_spec_payload_kind: str,
    training_spec_payload_schema_id: str | None,
    training_spec_payload_schema_version: str | None,
    training_spec_payload_ref: str | None,
    task_binding_spec: dict[str, Any] | None,
    checkpoint_writes: Sequence[CheckpointWriteResult],
    barrier_artifacts: Sequence[ArtifactRef],
    history_events: list[dict[str, Any]],
    final_metrics: dict[str, Any],
    issues: Sequence[str] | None,
    status: ManifestStatus = "completed",
    cancellation: CancellationDecision | None = None,
    runtime_telemetry: Mapping[str, Any] | None = None,
    execution_context: NativeExecutionProducerContext | None = None,
    diagnostics_ref: ArtifactRef,
    completed_batches: int,
    projection_custody: TrainingManifestMetadataProjectionCustody | None,
) -> TrainingRunManifest:
    artifacts = [*barrier_artifacts, diagnostics_ref]
    if history_events:
        artifacts.append(
            store_json_artifact(
                history_events,
                root=root_path,
                role="training_history",
                logical_name=f"feedbax_training_history_{run_id}.json",
            )
        )
    checkpoint_refs = [
        ParentRef(
            kind="TrainingCheckpointTransactionManifest",
            id=write.manifest.transaction_id,
            role="training_checkpoint_custody",
            uri=write.manifest_path.relative_to(write.root).as_posix(),
            metadata={"manifest_sha256": write.latest_pointer.manifest_sha256},
        )
        for write in checkpoint_writes
    ]
    payloads = build_training_run_manifest_spec_payloads(
        spec,
        training_spec_payload=training_spec_payload,
        training_spec_payload_kind=training_spec_payload_kind,
        training_spec_payload_schema_id=training_spec_payload_schema_id,
        training_spec_payload_schema_version=training_spec_payload_schema_version,
        training_spec_payload_ref=training_spec_payload_ref,
        task_binding_spec=task_binding_spec,
    )
    execution_identity = execution_context.execution if execution_context is not None else None
    row_provenance = (
        execution_identity.row_provenance.model_dump(mode="json", exclude_none=True)
        if execution_identity is not None and execution_identity.row_provenance is not None
        else None
    )
    projection_values = projection_custody.values if projection_custody is not None else {}
    projection_provenance = (
        {
            TRAINING_MANIFEST_METADATA_PROJECTION_PROVENANCE_KEY: projection_custody.provenance_summary()
        }
        if projection_custody is not None
        else {}
    )
    return TrainingRunManifest(
        id=manifest_id,
        job_id=run_id,
        status=status,
        started_at=utc_now(),
        completed_at=utc_now(),
        graph_spec=payloads.graph_spec,
        training_spec=payloads.training_spec,
        task_spec=payloads.task_spec,
        task_binding_spec=payloads.task_binding_spec,
        checkpoint_custody=checkpoint_refs,
        completed_batches=completed_batches,
        intent_hash=(
            execution_identity.authored_intent.intent_hash
            if execution_identity is not None
            else None
        ),
        resolved_semantics_root_hash=(
            execution_identity.resolved_snapshot.root_hash
            if execution_identity is not None
            else None
        ),
        execution_hash=(
            execution_identity.execution_capsule.execution_hash
            if execution_identity is not None
            else None
        ),
        input_data_identities=(
            [
                identity.model_dump(mode="json", exclude_none=True)
                for identity in execution_identity.immutable_inputs
            ]
            if execution_identity is not None
            else []
        ),
        summary_metrics={
            **final_metrics,
            **({"runtime_telemetry": dict(runtime_telemetry)} if runtime_telemetry else {}),
        },
        provenance=Provenance(
            entrypoint=EntrypointRef(
                kind="feedbax-training-executor",
                command="python -m feedbax execute-training-run-spec",
            ),
            issues=list(issues or ()),
            metadata={
                "training_executor": "native",
                **projection_provenance,
                **(
                    {"environment_fingerprint": execution_context.environment_fingerprint}
                    if execution_context is not None
                    else {}
                ),
                **(
                    {"cancellation": cancellation.as_provenance()}
                    if cancellation is not None
                    else {}
                ),
            },
        ),
        artifacts=artifacts,
        metadata={
            **_feedbax_owned_training_manifest_metadata(
                training_run_spec_schema_version=spec.schema_version,
                environment_fingerprint=(
                    execution_context.environment_fingerprint
                    if execution_context is not None
                    else _METADATA_VALUE_ABSENT
                ),
                training_row_provenance=(
                    row_provenance if execution_context is not None else _METADATA_VALUE_ABSENT
                ),
                runtime_telemetry=(
                    dict(runtime_telemetry) if runtime_telemetry else _METADATA_VALUE_ABSENT
                ),
            ),
            **projection_values,
        },
        metadata_projection_custody=projection_custody,
    )


def _emit_manifest(
    manifest: TrainingRunManifest,
    *,
    root: Path,
    conflict_policy: ManifestConflictPolicy,
    path: Path | None = None,
) -> Path:
    path = path or (root / "manifests" / "training_runs" / f"{manifest.id.replace(':', '_')}.json")
    _preflight_manifest_emission(
        manifest,
        root=root,
        conflict_policy=conflict_policy,
        path=path,
    )
    payload = manifest.model_dump_json(indent=2, exclude_none=True) + "\n"
    if path.exists():
        return path
    if path == root / "manifests" / "training_runs" / f"{manifest.id.replace(':', '_')}.json":
        return write_manifest(manifest, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _preflight_manifest_emission(
    manifest: TrainingRunManifest,
    *,
    root: Path,
    conflict_policy: ManifestConflictPolicy,
    path: Path | None = None,
) -> None:
    path = path or (root / "manifests" / "training_runs" / f"{manifest.id.replace(':', '_')}.json")
    if not path.exists():
        return
    payload = manifest.model_dump_json(indent=2, exclude_none=True) + "\n"
    existing = path.read_text(encoding="utf-8")
    if existing != payload:
        raise ManifestEmissionConflictError(
            f"manifest identity already exists with different content: {manifest.id!r}"
        )
    if conflict_policy != "reuse-identical":
        raise ManifestEmissionConflictError(f"manifest identity already exists: {manifest.id!r}")


def load_training_run_spec(path: Path | str) -> TrainingRunSpec:
    """Load a TrainingRunSpec JSON file."""
    return TrainingRunSpec.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
