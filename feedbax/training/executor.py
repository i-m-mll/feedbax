"""Native TrainingRunSpec executor and manifest emitter."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from feedbax.contracts.checkpoints import CheckpointLineageRef
from feedbax.contracts.manifest import (
    EntrypointRef,
    ManifestStatus,
    ParentRef,
    Provenance,
    SpecPayload,
    TrainingRunManifest,
    canonical_json_bytes,
    default_manifest_root,
    sha256_bytes,
    spec_payload,
    store_json_artifact,
    training_run_manifest_id,
    utc_now,
    write_manifest,
)
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    GraphTopologySourceSpec,
    TrainingMethodRegistry,
    TrainingRunSpec,
)
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.objectives.service import LossService, ObjectiveLoweringError
from feedbax.training.checkpoint_custody import (
    CheckpointWriteResult,
    load_latest_checkpoint,
    write_checkpoint_transaction,
)
from feedbax.training.phase_executor import (
    InMemoryCheckpointStore,
    PhaseCheckpoint,
    PhaseProgramExecutor,
)
from feedbax.training.worker_validation import (
    WorkerExecutabilityEnvironment,
    validate_worker_contract,
)


ManifestConflictPolicy = Literal["error", "reuse-identical"]
ProgressCallback = Callable[[Mapping[str, Any]], None]


class TrainingRunExecutorError(ValueError):
    """Base class for native training-run executor failures."""


class ManifestEmissionConflictError(TrainingRunExecutorError):
    """Raised when a manifest id already exists with different content."""


@dataclass(frozen=True)
class TrainingRunExecutionResult:
    """Result returned by ``execute_training_run_spec``."""

    run_id: str
    status: ManifestStatus
    manifest: TrainingRunManifest
    manifest_path: Path
    final_slots: dict[str, Any]
    final_coordinate: ProgressCoordinate
    checkpoint_writes: tuple[CheckpointWriteResult, ...]
    history_events: tuple[dict[str, Any], ...]


class StreamingCheckpointStore(InMemoryCheckpointStore):
    """Checkpoint store that writes custody transactions at barrier time."""

    def __init__(
        self,
        *,
        root: Path | str,
        run_spec: TrainingRunSpec,
        phase_program: Any,
        parent_lineage: Sequence[CheckpointLineageRef] = (),
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.run_spec = run_spec
        self.phase_program = phase_program
        self.parent_lineage = tuple(parent_lineage)
        self._writes: list[CheckpointWriteResult] = []

    @property
    def writes(self) -> tuple[CheckpointWriteResult, ...]:
        """Return successful custody writes in barrier firing order."""
        return tuple(self._writes)

    def save(self, checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        saved = super().save(checkpoint)
        write = write_checkpoint_transaction(
            self.root,
            run_spec=self.run_spec,
            phase_program=self.phase_program,
            barrier_name=saved.barrier,
            coordinate=saved.coordinate,
            slots=saved.slots,
            status="partial",
            parent_lineage=self.parent_lineage,
            history_availability={"progress": True},
            metadata={"barrier_visit_ordinal": saved.visit_ordinal},
        )
        self._writes.append(write)
        return saved


def execute_training_run_spec(
    spec: TrainingRunSpec | Mapping[str, Any],
    *,
    run_id: str | None = None,
    initial_slots: Mapping[str, Any] | None = None,
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
    task_binding_spec: Mapping[str, Any] | None = None,
    resume: bool = False,
    stop_after_barrier: str | None = None,
    manifest_conflict_policy: ManifestConflictPolicy = "error",
    issues: Sequence[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TrainingRunExecutionResult:
    """Validate, execute, checkpoint, and natively emit one training-run manifest.

    ``progress_callback`` is called once for each generated training-progress
    history event, in history order, as each progress coordinate is produced
    during execution. Callback payloads have the same shape as stored history
    events. Exceptions raised by the callback propagate to the caller.
    """
    run_spec = _validate_spec(spec)
    root_path = Path(manifest_root) if manifest_root is not None else (
        Path(run_spec.artifacts.manifest_root)
        if run_spec.artifacts.manifest_root is not None
        else default_manifest_root()
    )
    resolved_run_id = run_id or _default_run_id(run_spec)
    method_registry = registry or DEFAULT_TRAINING_METHOD_REGISTRY
    registration = method_registry.resolve(run_spec.method_ref, path="/method_ref")
    method_payload = method_registry.validate_payload(
        run_spec.method_ref,
        run_spec.method_payload,
        path="/method_payload",
    )
    method_contract = registration.contract_factory()
    _validate_declarations_match_spec(run_spec, method_contract.method_ref)

    graph_inline = _graph_inline(run_spec.graph)
    lowered = _lower_objective(
        run_spec,
        graph_inline=graph_inline,
        loss_service=loss_service or LossService(),
    )
    kernels = dict(registration.update_kernels_factory(method_payload))
    guards = dict(registration.guard_predicates_factory(method_payload))
    effective_phase = validate_worker_contract(
        method_contract,
        environment=environment,
        update_kernels=kernels,
        task_binding_spec=task_binding_spec,
        objective_requirements=lowered.requirements,
    )
    program = effective_phase.phase_program
    slots = _initial_slots(initial_slots, lowered_objective=lowered.loss)
    custody_root = _checkpoint_root(
        root_path=root_path,
        configured_root=checkpoint_root or run_spec.artifacts.artifact_root,
        run_id=resolved_run_id,
    )
    resume_barrier: str | None = None
    parent_lineage: list[CheckpointLineageRef] = []
    loaded_resume_checkpoint: PhaseCheckpoint | None = None
    if resume:
        loaded = load_latest_checkpoint(
            custody_root,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=slots,
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

    checkpoint_store = StreamingCheckpointStore(
        root=custody_root,
        run_spec=run_spec,
        phase_program=program,
        parent_lineage=parent_lineage,
    )
    if loaded_resume_checkpoint is not None:
        checkpoint_store.remember(loaded_resume_checkpoint)

    executor = PhaseProgramExecutor(
        program,
        kernels,
        guard_predicates=guards,
        checkpoint_store=checkpoint_store,
        state_slots=effective_phase.state_slots,
    )
    live_history_events: list[dict[str, Any]] = []
    execution = executor.run(
        slots,
        run_id=resolved_run_id,
        resume_from_barrier=resume_barrier,
        stop_after_barrier=stop_after_barrier,
        context={"run_spec": run_spec, "method_payload": method_payload},
        progress_callback=(
            _live_progress_callback(progress_callback, live_history_events)
            if progress_callback is not None
            else None
        ),
    )
    checkpoint_writes = checkpoint_store.writes
    history_events = (
        live_history_events
        if progress_callback is not None
        else _history_events(execution.progress)
    )
    final_metrics = _final_metrics(execution.slots, execution.coordinate)
    manifest = _build_manifest(
        run_spec,
        run_id=resolved_run_id,
        root_path=root_path,
        graph_inline=graph_inline,
        training_spec_payload=dict(training_spec_payload or run_spec.model_dump(mode="json")),
        training_spec_payload_kind=training_spec_payload_kind,
        training_spec_payload_schema_id=training_spec_payload_schema_id,
        training_spec_payload_schema_version=training_spec_payload_schema_version,
        training_spec_payload_ref=training_spec_payload_ref,
        task_binding_spec=dict(task_binding_spec) if task_binding_spec is not None else None,
        checkpoint_writes=checkpoint_writes,
        history_events=history_events,
        final_metrics=final_metrics,
        issues=issues,
    )
    manifest_path = _emit_manifest(
        manifest,
        root=root_path,
        conflict_policy=manifest_conflict_policy,
    )
    return TrainingRunExecutionResult(
        run_id=resolved_run_id,
        status="completed",
        manifest=manifest,
        manifest_path=manifest_path,
        final_slots=dict(execution.slots),
        final_coordinate=execution.coordinate,
        checkpoint_writes=tuple(checkpoint_writes),
        history_events=tuple(history_events),
    )


def _validate_spec(spec: TrainingRunSpec | Mapping[str, Any]) -> TrainingRunSpec:
    try:
        return spec if isinstance(spec, TrainingRunSpec) else TrainingRunSpec.model_validate(spec)
    except ValidationError as exc:
        raise TrainingRunExecutorError(f"/training_run_spec validation failed: {exc}") from exc


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
) -> dict[str, Any]:
    if initial_slots is None:
        raise TrainingRunExecutorError("/initial_slots are required for native execution")
    slots = dict(initial_slots)
    slots.setdefault("objective", lowered_objective)
    return slots


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


def _history_events(progress: Sequence[ProgressCoordinate]) -> list[dict[str, Any]]:
    return [_history_event(coordinate) for coordinate in progress]


def _history_event(coordinate: ProgressCoordinate) -> dict[str, Any]:
    return {
        "type": "training_progress",
        "coordinate": coordinate.model_dump(mode="json", exclude_none=True),
        "metrics": dict(coordinate.metrics),
    }


def _live_progress_callback(
    progress_callback: ProgressCallback | None,
    history_events: list[dict[str, Any]],
) -> Callable[[ProgressCoordinate], None]:
    def emit(coordinate: ProgressCoordinate) -> None:
        event = _history_event(coordinate)
        history_events.append(event)
        if progress_callback is not None:
            progress_callback(deepcopy(event))

    return emit


def _final_metrics(slots: Mapping[str, Any], coordinate: ProgressCoordinate) -> dict[str, Any]:
    metrics = dict(coordinate.metrics)
    for key, value in slots.items():
        if key.endswith("loss"):
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                metrics[key] = value
    metrics.setdefault("global_step", coordinate.global_step)
    return metrics


def _build_manifest(
    spec: TrainingRunSpec,
    *,
    run_id: str,
    root_path: Path,
    graph_inline: dict[str, Any] | None,
    training_spec_payload: dict[str, Any],
    training_spec_payload_kind: str,
    training_spec_payload_schema_id: str | None,
    training_spec_payload_schema_version: str | None,
    training_spec_payload_ref: str | None,
    task_binding_spec: dict[str, Any] | None,
    checkpoint_writes: Sequence[CheckpointWriteResult],
    history_events: list[dict[str, Any]],
    final_metrics: dict[str, Any],
    issues: Sequence[str] | None,
) -> TrainingRunManifest:
    artifacts = []
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
            uri=str(write.manifest_path),
            metadata={"manifest_sha256": write.latest_pointer.manifest_sha256},
        )
        for write in checkpoint_writes
    ]
    return TrainingRunManifest(
        id=training_run_manifest_id(run_id),
        job_id=run_id,
        status="completed",
        started_at=utc_now(),
        completed_at=utc_now(),
        graph_spec=spec_payload("GraphSpec", graph_inline) if graph_inline is not None else None,
        training_spec=_training_spec_payload(
            training_spec_payload_kind,
            training_spec_payload,
            schema_id=training_spec_payload_schema_id,
            schema_version=training_spec_payload_schema_version,
            ref=training_spec_payload_ref,
        ),
        task_spec=spec_payload("TaskSpec", spec.task.model_dump(mode="json")),
        task_binding_spec=(
            spec_payload("StudioTaskBindingSpec", task_binding_spec)
            if task_binding_spec is not None
            else None
        ),
        checkpoint_custody=checkpoint_refs,
        summary_metrics=final_metrics,
        provenance=Provenance(
            entrypoint=EntrypointRef(
                kind="feedbax-training-executor",
                command="python -m feedbax execute-training-run-spec",
            ),
            issues=list(issues or ()),
            metadata={"training_executor": "native"},
        ),
        artifacts=artifacts,
        metadata={"training_run_spec_schema_version": spec.schema_version},
    )


def _training_spec_payload(
    kind: str,
    inline: dict[str, Any],
    *,
    schema_id: str | None,
    schema_version: str | None,
    ref: str | None,
) -> SpecPayload:
    if kind == "TrainingRunSpec":
        return spec_payload(kind, inline, ref=ref)
    if schema_id is None or schema_version is None:
        raise TrainingRunExecutorError(
            "/training_spec_payload requires schema_id and schema_version for external kinds"
        )
    return SpecPayload(
        kind=kind,
        inline=inline,
        schema_id=schema_id,
        schema_version=schema_version,
        ref=ref,
        sha256=sha256_bytes(canonical_json_bytes(inline)),
    )


def _emit_manifest(
    manifest: TrainingRunManifest,
    *,
    root: Path,
    conflict_policy: ManifestConflictPolicy,
) -> Path:
    path = root / "manifests" / "training_runs" / f"{manifest.id.replace(':', '_')}.json"
    payload = manifest.model_dump_json(indent=2, exclude_none=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != payload:
            raise ManifestEmissionConflictError(
                "manifest identity already exists with different content: "
                f"{manifest.id!r}"
            )
        if conflict_policy == "reuse-identical":
            return path
        raise ManifestEmissionConflictError(
            f"manifest identity already exists: {manifest.id!r}"
        )
    return write_manifest(manifest, root=root)


def load_training_run_spec(path: Path | str) -> TrainingRunSpec:
    """Load a TrainingRunSpec JSON file."""
    return TrainingRunSpec.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
