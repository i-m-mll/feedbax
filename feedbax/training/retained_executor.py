"""Retained TrainingRunSpec entry for envelope and scenario execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from feedbax.contracts.artifact_custody import ArtifactBlobProvider, ContentBinding
from feedbax.contracts.checkpoint_initialization import (
    CheckpointInitializationPlan,
    CheckpointInitializationRequest,
)
from feedbax.contracts.checkpoints import CheckpointForkPlan, CheckpointLatestPointer
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.training import (
    TrainingManifestMetadataProjection,
    TrainingProgramRegistry,
    TrainingRunSpec,
)
from feedbax.objectives.service import LossService, ObjectiveLoweringError
from feedbax.orchestration.events import RunEventEmitter
from feedbax.training.checkpoint_custody import (
    CheckpointForkPlanBindings,
    CheckpointWriteResult,
    ResumeSlotTransform,
)
from feedbax.training.diagnostics import NativeExecutionProducerContext
from feedbax.training.executor import (
    CancellationProbe,
    ManifestConflictPolicy,
    ProgressCallback,
    TrainingRunExecutionResult,
    TrainingRunExecutorError,
    WorkerExecutabilityEnvironment,
    _execute_training_run,
)
from feedbax.training.model_checkpoint import SemanticModelWriter
from feedbax.training.phase_executor import CommittedUpdateCallback
from feedbax.training.preparation import (
    ExecutionPreparationResult,
    MaterializedExecutionPreparation,
    ObjectiveLowerer,
)
from feedbax.training.standard_train_checkpoint import TrainStateCodec


def execute_training_run_spec(
    spec: TrainingRunSpec | Mapping[str, Any],
    *,
    run_id: str | None = None,
    preparation: ExecutionPreparationResult | MaterializedExecutionPreparation | None = None,
    initial_slots: Mapping[str, Any] | None = None,
    kernel_context: Mapping[str, Any] | None = None,
    manifest_root: Path | str | None = None,
    checkpoint_root: Path | str | None = None,
    checkpoint_source_root: Path | str | None = None,
    checkpoint_source_pointer: CheckpointLatestPointer | None = None,
    checkpoint_history_parent_roots: Mapping[str, Path | str] | None = None,
    checkpoint_initialization_plan: CheckpointInitializationPlan | None = None,
    checkpoint_initialization_request: CheckpointInitializationRequest | None = None,
    checkpoint_fork_plan: CheckpointForkPlan | None = None,
    checkpoint_fork_plan_bindings: CheckpointForkPlanBindings | None = None,
    checkpoint_fork_target_id: str | None = None,
    checkpoint_content_provider: ArtifactBlobProvider | None = None,
    semantic_model: SemanticModelWriter | None = None,
    train_state_codec: TrainStateCodec = "legacy",
    checkpoint_source_content_binding: ContentBinding | None = None,
    registry: TrainingProgramRegistry,
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
    update_budget: int | None = None,
    manifest_conflict_policy: ManifestConflictPolicy = "error",
    issues: Sequence[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    retain_history: bool = True,
    run_event_emitter: RunEventEmitter | None = None,
    cancellation_probe: CancellationProbe | None = None,
    stop_before_update: Callable[[], bool] | None = None,
    start_callback: Callable[[int], None] | None = None,
    committed_update_callback: CommittedUpdateCallback | None = None,
    checkpoint_published: Callable[[CheckpointWriteResult], None] | None = None,
    before_finalize: Callable[[], None] | None = None,
    execution_started_at_monotonic: float | None = None,
    execution_start_semantics: str = "executor_entry",
    execution_context: NativeExecutionProducerContext | Mapping[str, Any] | None = None,
) -> TrainingRunExecutionResult:
    """Validate, execute, checkpoint, and natively emit one training-run manifest.

    ``progress_callback`` is called once for each generated training-progress
    history event, in history order, as each progress coordinate is produced
    during execution. Callback payloads have the same shape as stored history
    events. Exceptions raised by the callback propagate to the caller.

    With ``retain_history=False``, the result omits the in-memory history and
    diagnostics (``history_events=()``, ``diagnostics=None``). The published
    artifacts and progress callbacks are produced in either mode.

    ``run_event_emitter`` optionally receives the versioned run-event protocol.
    Emitter I/O diagnostics are contained by the emitter and do not change
    training-loop failure behavior.

    ``cancellation_probe`` is checked at progress boundaries. A ``stop``
    decision completes the next durable checkpoint before returning a cancelled
    manifest; ``terminate`` emits a cancelled manifest immediately.
    ``stop_before_update`` can interrupt before any pending update, finalizing
    the last committed state. Unsaved progress is checkpointed through the
    committed phase's authored barrier. A stop before any new progress leaves
    existing custody in place.

    ``checkpoint_source_root`` names a read-only source for legacy resume or an
    explicit initialization/fork plan. Plan-driven execution additionally
    requires the source's canonical content binding and publishes every output
    checkpoint through that immutable content provider.
    For same-request resume, ``checkpoint_source_pointer`` authenticates one
    specific published transaction without consulting a mutable latest pointer.
    ``checkpoint_history_parent_roots`` locates ancestor transactions for
    self-contained history materialization when custody spans multiple roots.
    """
    arguments = locals().copy()
    return _execute_training_run(lower_objective=_lower_objective, **arguments)


def _lower_objective(spec: TrainingRunSpec, service: ObjectiveLowerer | None):
    graph = GraphSpec.model_validate(spec.graph.inline) if spec.graph.inline is not None else None
    try:
        lowered = (service or LossService()).lower_objective_slot(
            spec.objective, graph=graph, trial_axis="batch", path="/objective",
        )
    except ObjectiveLoweringError as exc:
        raise TrainingRunExecutorError(str(exc)) from exc
    return lowered.loss, lowered.requirements
