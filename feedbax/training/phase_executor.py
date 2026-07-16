"""Generic phase-program executor for worker-contract tests and integrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from operator import index
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.tree as jt
import numpy as np

from feedbax.contracts.worker import (
    MaterializedMappingLevelSpec,
    MaterializedSlotAxisBinding,
    PhaseProgramSpec,
    ProgressCoordinate,
    StateSlotSpec,
)
from feedbax.training.worker_validation import (
    WorkerContractValidationError,
    validate_update_kernel_callable,
)

UpdateKernel = Callable[
    [Mapping[str, Any], ProgressCoordinate, Mapping[str, Any]], Mapping[str, Any]
]
ProgressCallback = Callable[[ProgressCoordinate], None]
MethodObservationCallback = Callable[[ProgressCoordinate], None]
StepGuard = Callable[
    [Mapping[str, Any], ProgressCoordinate, Mapping[str, Any]], "StepGuardResult | None"
]


def _is_shareable_jax_array(value: Any) -> bool:
    return isinstance(value, jax.Array)


def _copy_executor_value(value: Any) -> Any:
    """Copy executor pytrees while sharing immutable JAX array leaves."""
    return jt.map(
        lambda leaf: leaf if _is_shareable_jax_array(leaf) else deepcopy(leaf),
        value,
        is_leaf=_is_shareable_jax_array,
    )


def _copy_executor_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _copy_executor_value(value) for key, value in mapping.items()}


def _copy_phase_checkpoint(checkpoint: "PhaseCheckpoint") -> "PhaseCheckpoint":
    return PhaseCheckpoint(
        barrier=checkpoint.barrier,
        coordinate=deepcopy(checkpoint.coordinate),
        slots=_copy_executor_mapping(checkpoint.slots),
        visit_ordinal=checkpoint.visit_ordinal,
    )


def _copy_progress_coordinate(coordinate: ProgressCoordinate) -> ProgressCoordinate:
    return coordinate.model_copy(update={"metrics": _copy_executor_mapping(coordinate.metrics)})


def _progress_observation(
    coordinate: ProgressCoordinate,
    completed_batches: int,
    *,
    include_completed_batches: bool,
) -> ProgressCoordinate:
    if include_completed_batches:
        return coordinate.model_copy(update={"completed_batches": completed_batches})
    return coordinate


@dataclass
class PhaseCheckpoint:
    """In-memory checkpoint captured at a phase barrier."""

    barrier: str
    coordinate: ProgressCoordinate
    slots: dict[str, Any]
    visit_ordinal: int | None = None


@dataclass
class PhaseExecutionResult:
    """Result from executing a phase program."""

    slots: dict[str, Any]
    coordinate: ProgressCoordinate
    progress: list[ProgressCoordinate] = field(default_factory=list)
    checkpoints: dict[str, PhaseCheckpoint] = field(default_factory=dict)
    checkpoint_visits: tuple[PhaseCheckpoint, ...] = ()


@dataclass(frozen=True)
class StepGuardResult:
    """Optional method-neutral control decision after one executor step."""

    halt: bool = False
    slots: Mapping[str, Any] | None = None
    coordinate: ProgressCoordinate | None = None


class PhaseCheckpointStore(Protocol):
    """Minimal checkpoint-store contract consumed by ``PhaseProgramExecutor``."""

    def save(self, checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        """Persist one checkpoint visit and return the stored checkpoint."""

    def remember(self, checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        """Record a pre-existing checkpoint visit without publishing side effects."""

    def load(self, barrier: str) -> PhaseCheckpoint:
        """Return the latest checkpoint visit for ``barrier``."""

    def as_dict(self) -> dict[str, PhaseCheckpoint]:
        """Return the latest checkpoint for each barrier."""

    def visits(self, barrier: str | None = None) -> tuple[PhaseCheckpoint, ...]:
        """Return stored checkpoint visits in append order."""


class InMemoryCheckpointStore:
    """Append-only checkpoint store used by the generic phase executor.

    Repeated visits to the same barrier are retained with per-barrier visit
    ordinals. ``load()`` and ``as_dict()`` expose the latest visit for resume and
    legacy callers.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, list[PhaseCheckpoint]] = {}
        self._visit_log: list[PhaseCheckpoint] = []
        self._next_visit_ordinal: dict[str, int] = {}

    def save(self, checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        return self.remember(checkpoint)

    def remember(self, checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        saved = self._checkpoint_with_visit_ordinal(checkpoint)
        self._checkpoints.setdefault(saved.barrier, []).append(saved)
        self._visit_log.append(saved)
        return _copy_phase_checkpoint(saved)

    def load(self, barrier: str) -> PhaseCheckpoint:
        try:
            return _copy_phase_checkpoint(self._checkpoints[barrier][-1])
        except (KeyError, IndexError) as exc:
            raise WorkerContractValidationError(
                f"/checkpoint_store/{barrier}",
                f"unknown checkpoint barrier {barrier!r}",
            ) from exc

    def as_dict(self) -> dict[str, PhaseCheckpoint]:
        """Return a defensive copy of the latest checkpoint for each barrier."""
        return {
            barrier: _copy_phase_checkpoint(checkpoints[-1])
            for barrier, checkpoints in self._checkpoints.items()
            if checkpoints
        }

    def visits(self, barrier: str | None = None) -> tuple[PhaseCheckpoint, ...]:
        """Return defensive copies of checkpoint visits in append order."""
        visits: Sequence[PhaseCheckpoint]
        if barrier is None:
            visits = self._visit_log
        else:
            visits = self._checkpoints.get(barrier, ())
        return tuple(_copy_phase_checkpoint(visit) for visit in visits)

    def _checkpoint_with_visit_ordinal(self, checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        next_ordinal = self._next_visit_ordinal.get(checkpoint.barrier, 0)
        visit_ordinal = (
            next_ordinal
            if checkpoint.visit_ordinal is None
            else max(checkpoint.visit_ordinal, next_ordinal)
        )
        self._next_visit_ordinal[checkpoint.barrier] = visit_ordinal + 1
        return _copy_phase_checkpoint(
            PhaseCheckpoint(
                barrier=checkpoint.barrier,
                coordinate=checkpoint.coordinate,
                slots=checkpoint.slots,
                visit_ordinal=visit_ordinal,
            )
        )


class PhaseProgramExecutor:
    """Execute a validated phase program through fixed-signature kernels."""

    def __init__(
        self,
        program: PhaseProgramSpec,
        kernels: Mapping[str, UpdateKernel],
        *,
        guard_predicates: Mapping[str, UpdateKernel] | None = None,
        checkpoint_store: PhaseCheckpointStore | None = None,
        state_slots: Sequence[StateSlotSpec] = (),
        mapping_levels: Sequence[MaterializedMappingLevelSpec] = (),
        slot_axis_bindings: Mapping[str, Sequence[MaterializedSlotAxisBinding]] | None = None,
    ) -> None:
        self.program = program
        self.kernels = dict(kernels)
        self.guard_predicates = dict(guard_predicates or {})
        self.checkpoint_store = checkpoint_store or InMemoryCheckpointStore()
        self._metric_slots = tuple(slot.name for slot in state_slots if slot.role == "metric")
        self.mapping_levels = tuple(mapping_levels)
        self.slot_axis_bindings = {
            name: tuple(bindings) for name, bindings in (slot_axis_bindings or {}).items()
        }
        self._active_axis = self.mapping_levels[0].axis if self.mapping_levels else None
        self._mapped_size = self.mapping_levels[0].size if self.mapping_levels else None
        self._mapped_slots = frozenset(
            name
            for name, bindings in self.slot_axis_bindings.items()
            if bindings and bindings[0].mode == "mapped"
        )
        self._shared_slots = frozenset(self.slot_axis_bindings) - self._mapped_slots
        self._phases = {phase.name: phase for phase in program.phases}
        self._steps = {step.name: step for step in program.update_steps}
        self._transitions = {
            (transition.source, transition.target): transition for transition in program.transitions
        }
        self._barriers = {barrier.name: barrier for barrier in program.checkpoint_barriers}
        for kernel_ref, kernel in self.kernels.items():
            validate_update_kernel_callable(kernel, path=f"/kernels/{kernel_ref}")
        for predicate_ref, predicate in self.guard_predicates.items():
            validate_update_kernel_callable(predicate, path=f"/guard_predicates/{predicate_ref}")

    def _mapped_kernel(self, kernel: UpdateKernel) -> Callable[..., Mapping[str, Any]]:
        def scalar_call(
            mapped_slots: Mapping[str, Any],
            shared_slots: Mapping[str, Any],
            coordinate: ProgressCoordinate,
            context: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            return kernel({**shared_slots, **mapped_slots}, coordinate, context)

        return eqx.filter_vmap(
            scalar_call,
            in_axes=(eqx.if_array(0), None, None, None),
            out_axes=eqx.if_array(0),
        )

    def preflight(
        self,
        slots: Mapping[str, Any],
        *,
        run_id: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Shape-check every mapped kernel through its exact execution wrapper."""
        if self._active_axis is None:
            return
        coordinate = ProgressCoordinate(run_id=run_id, phase=self.program.initial_phase)
        mapped = {name: slots[name] for name in self._mapped_slots if name in slots}
        shared = {name: slots[name] for name in self._shared_slots if name in slots}
        for phase in self.program.phases:
            for step_name in phase.update_steps:
                step = self._steps[step_name]
                if self._active_axis not in step.axes:
                    continue
                kernel = self.kernels[step.kernel.kernel_ref]
                try:
                    updates = eqx.filter_eval_shape(
                        self._mapped_kernel(kernel),
                        mapped,
                        shared,
                        coordinate.model_copy(update={"phase": phase.name}),
                        dict(context or {}),
                    )
                    self._validate_mapped_updates(
                        step.name, step.writes, updates, {**shared, **mapped}
                    )
                except Exception as exc:
                    raise WorkerContractValidationError(
                        f"/phase_program/phases/{phase.name}/update_steps/{step.name}",
                        f"mapped filtered-vmap preflight failed: {exc}",
                    ) from exc

    def merge_resume_slots(
        self,
        slots: Mapping[str, Any],
        checkpoint: PhaseCheckpoint,
    ) -> dict[str, Any]:
        """Overlay declared checkpoint authority onto prepared runtime slots."""
        barrier = self._barriers.get(checkpoint.barrier)
        if barrier is None:
            raise WorkerContractValidationError(
                f"/checkpoint_barriers/{checkpoint.barrier}",
                f"unknown checkpoint barrier {checkpoint.barrier!r}",
            )
        declared = {slot.slot: slot for slot in barrier.slots}
        unknown = sorted(set(checkpoint.slots) - set(declared))
        if unknown:
            raise WorkerContractValidationError(
                f"/checkpoint_barriers/{checkpoint.barrier}/slots",
                f"checkpoint contains undeclared slots {unknown!r}",
            )
        missing_required_checkpoint = [
            name
            for name, spec in declared.items()
            if spec.required and name not in checkpoint.slots
        ]
        if missing_required_checkpoint:
            raise WorkerContractValidationError(
                f"/checkpoint_barriers/{checkpoint.barrier}/slots",
                f"checkpoint is missing required slots {missing_required_checkpoint!r}",
            )
        if not slots:
            return _copy_executor_mapping(checkpoint.slots)
        missing_prepared = sorted(set(checkpoint.slots) - set(slots))
        if missing_prepared:
            raise WorkerContractValidationError(
                f"/checkpoint_barriers/{checkpoint.barrier}/slots",
                "checkpoint slots lack prepared runtime templates "
                f"{missing_prepared!r}",
            )
        missing_required_prepared = [
            name for name, spec in declared.items() if spec.required and name not in slots
        ]
        if missing_required_prepared:
            raise WorkerContractValidationError(
                f"/checkpoint_barriers/{checkpoint.barrier}/slots",
                "prepared runtime is missing required checkpoint slots "
                f"{missing_required_prepared!r}",
            )
        merged = _copy_executor_mapping(slots)
        merged.update(_copy_executor_mapping(checkpoint.slots))
        return merged

    def run(
        self,
        slots: Mapping[str, Any],
        *,
        run_id: str,
        resume_from_barrier: str | None = None,
        stop_after_barrier: str | None = None,
        stop_after_next_barrier: Callable[[str], bool] | None = None,
        context: Mapping[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
        method_observation_callback: MethodObservationCallback | None = None,
        step_guard: StepGuard | None = None,
        checkpoint_interval: int | None = None,
        progress_interval: int | None = None,
        include_completed_batches_in_progress: bool = False,
    ) -> PhaseExecutionResult:
        """Execute phases from the start or from a checkpoint barrier.

        Authored intervals are evaluated against the phase program's declared
        training-batch authority, not against phase boundaries or checkpoint
        visit ordinals. A terminal partial interval is always emitted.
        """
        self._validate_interval(checkpoint_interval, name="checkpoint_interval")
        self._validate_interval(progress_interval, name="progress_interval")
        progress: list[ProgressCoordinate] = []
        checkpoint_context = dict(context or {})
        resume_inner_step = 0
        resuming_within_phase = False
        if resume_from_barrier is not None:
            checkpoint = self.checkpoint_store.load(resume_from_barrier)
            current_slots = self.merge_resume_slots(slots, checkpoint)
            start_phase, resume_inner_step = self._resume_position(checkpoint)
            resuming_within_phase = resume_inner_step > 0
            coordinate = checkpoint.coordinate.model_copy(
                update={
                    "run_id": run_id,
                    "phase": start_phase,
                    "completed_barrier": resume_from_barrier,
                }
            )
        else:
            current_slots = _copy_executor_mapping(slots)
            start_phase = self.program.initial_phase
            coordinate = ProgressCoordinate(run_id=run_id, phase=start_phase)

        initial_batch = self._completed_training_batches(current_slots, coordinate)
        previous_completed_batch = initial_batch
        last_progress_batch = initial_batch
        last_checkpoint_batch = initial_batch
        phase_name: str | None = start_phase
        while phase_name is not None:
            phase = self._phases[phase_name]
            schedule_origin_step = (
                coordinate.schedule_origin_step
                if resuming_within_phase
                else self._schedule_origin_step(phase.name, coordinate)
            )
            coordinate = coordinate.model_copy(
                update={
                    "phase": phase.name,
                    "schedule_origin_step": schedule_origin_step,
                }
            )
            for inner_step in range(resume_inner_step, phase.max_steps):
                coordinate = coordinate.model_copy(update={"inner_step": inner_step})
                for step_name in phase.update_steps:
                    step = self._steps[step_name]
                    kernel = self.kernels.get(step.kernel.kernel_ref)
                    if kernel is None:
                        raise WorkerContractValidationError(
                            f"/phase_program/update_steps/{step_name}/kernel_ref",
                            f"missing callable for kernel_ref {step.kernel.kernel_ref!r}",
                        )
                    if self._active_axis is not None and self._active_axis in step.axes:
                        updates = self._mapped_kernel(kernel)(
                            {
                                name: current_slots[name]
                                for name in self._mapped_slots
                                if name in current_slots
                            },
                            {
                                name: current_slots[name]
                                for name in self._shared_slots
                                if name in current_slots
                            },
                            coordinate,
                            checkpoint_context,
                        )
                        self._validate_mapped_updates(
                            step.name, step.writes, updates, current_slots
                        )
                    else:
                        updates = kernel(current_slots, coordinate, checkpoint_context)
                    unknown = sorted(set(updates) - set(step.writes))
                    if unknown:
                        raise WorkerContractValidationError(
                            f"/phase_program/update_steps/{step_name}/writes",
                            f"kernel returned undeclared writes {unknown!r}",
                        )
                    current_slots.update(_copy_executor_mapping(updates))
                coordinate = coordinate.model_copy(
                    update={
                        "program_step": coordinate.program_step + 1,
                        "metrics": self._progress_metrics(current_slots),
                    }
                )
                if step_guard is not None:
                    guard_result = step_guard(current_slots, coordinate, checkpoint_context)
                    if guard_result is not None and guard_result.halt:
                        halted_slots = (
                            current_slots if guard_result.slots is None else guard_result.slots
                        )
                        halted_coordinate = guard_result.coordinate or coordinate
                        return PhaseExecutionResult(
                            slots=_copy_executor_mapping(halted_slots),
                            coordinate=halted_coordinate,
                            progress=progress,
                            checkpoints=self.checkpoint_store.as_dict(),
                            checkpoint_visits=self.checkpoint_store.visits(),
                        )
                completed_batches = self._completed_training_batches(current_slots, coordinate)
                if completed_batches < previous_completed_batch:
                    raise WorkerContractValidationError(
                        "/phase_program/batch_progress",
                        "completed training batches must be monotonic; "
                        f"previous={previous_completed_batch}, current={completed_batches}",
                    )
                previous_completed_batch = completed_batches
                if method_observation_callback is not None:
                    method_observation_callback(
                        _copy_progress_coordinate(
                            _progress_observation(
                                coordinate,
                                completed_batches,
                                include_completed_batches=True,
                            )
                        )
                    )
                if progress_interval is None or self._interval_due(
                    completed_batches,
                    progress_interval,
                    last_progress_batch,
                ):
                    observed_coordinate = _progress_observation(
                        coordinate,
                        completed_batches,
                        include_completed_batches=include_completed_batches_in_progress,
                    )
                    progress.append(observed_coordinate)
                    last_progress_batch = completed_batches
                    if progress_callback is not None:
                        progress_callback(_copy_progress_coordinate(observed_coordinate))
                if checkpoint_interval is not None and self._interval_due(
                    completed_batches,
                    checkpoint_interval,
                    last_checkpoint_batch,
                ):
                    saved_checkpoint = self._save_phase_barrier(
                        phase.name,
                        coordinate,
                        current_slots,
                    )
                    last_checkpoint_batch = completed_batches
                    coordinate = coordinate.model_copy(
                        update={"completed_barrier": saved_checkpoint.barrier}
                    )
                    if self._should_stop_at_barrier(
                        saved_checkpoint.barrier,
                        stop_after_barrier=stop_after_barrier,
                        stop_after_next_barrier=stop_after_next_barrier,
                    ):
                        return self._result(current_slots, coordinate, progress)

            next_phase = self._next_phase(
                phase.name,
                current_slots,
                coordinate,
                checkpoint_context,
            )
            completed_batches = self._completed_training_batches(current_slots, coordinate)
            if (
                next_phase is None
                and progress_interval is not None
                and completed_batches != last_progress_batch
            ):
                observed_coordinate = _progress_observation(
                    coordinate,
                    completed_batches,
                    include_completed_batches=include_completed_batches_in_progress,
                )
                progress.append(observed_coordinate)
                last_progress_batch = completed_batches
                if progress_callback is not None:
                    progress_callback(_copy_progress_coordinate(observed_coordinate))

            saved_checkpoint: PhaseCheckpoint | None = None
            if checkpoint_interval is None and phase.checkpoint_barrier is not None:
                saved_checkpoint = self._save_barrier(
                    phase.checkpoint_barrier,
                    coordinate,
                    current_slots,
                )
            elif (
                checkpoint_interval is not None
                and next_phase is None
                and completed_batches != last_checkpoint_batch
            ):
                saved_checkpoint = self._save_phase_barrier(
                    phase.name,
                    coordinate,
                    current_slots,
                )
                last_checkpoint_batch = completed_batches
            if saved_checkpoint is not None:
                coordinate = coordinate.model_copy(
                    update={"completed_barrier": saved_checkpoint.barrier}
                )
                if self._should_stop_at_barrier(
                    saved_checkpoint.barrier,
                    stop_after_barrier=stop_after_barrier,
                    stop_after_next_barrier=stop_after_next_barrier,
                ):
                    return self._result(current_slots, coordinate, progress)

            phase_name = next_phase
            resume_inner_step = 0
            resuming_within_phase = False

        return self._result(current_slots, coordinate, progress)

    def _validate_mapped_updates(
        self,
        step_name: str,
        declared_writes: Sequence[str],
        updates: Mapping[str, Any],
        input_slots: Mapping[str, Any],
    ) -> None:
        unknown = sorted(set(updates) - set(declared_writes))
        if unknown:
            raise WorkerContractValidationError(
                f"/phase_program/update_steps/{step_name}/writes",
                f"kernel returned undeclared writes {unknown!r}",
            )
        array_types = (jax.Array, np.ndarray, jax.ShapeDtypeStruct)
        for slot, value in updates.items():
            if slot not in input_slots:
                raise WorkerContractValidationError(
                    f"/phase_program/update_steps/{step_name}/writes/{slot}",
                    "mapped write destination is absent from executor state",
                )
            output_leaves, output_def = jt.flatten(value)
            input_leaves, input_def = jt.flatten(input_slots[slot])
            if output_def != input_def:
                raise WorkerContractValidationError(
                    f"/phase_program/update_steps/{step_name}/writes/{slot}",
                    "mapped write must retain destination PyTree definition",
                )
            for leaf_index, (source, target) in enumerate(
                zip(input_leaves, output_leaves, strict=True)
            ):
                source_array = isinstance(source, array_types)
                target_array = isinstance(target, array_types)
                if source_array != target_array:
                    raise WorkerContractValidationError(
                        f"/phase_program/update_steps/{step_name}/writes/{slot}/{leaf_index}",
                        "mapped write changes array/static leaf category",
                    )
                if source_array and (
                    source.shape != target.shape or source.dtype != target.dtype
                ):
                    raise WorkerContractValidationError(
                        f"/phase_program/update_steps/{step_name}/writes/{slot}/{leaf_index}",
                        "mapped write changes destination shape or dtype; "
                        f"expected={source.shape!r}/{source.dtype}, "
                        f"observed={target.shape!r}/{target.dtype}",
                    )

    def _result(
        self,
        slots: Mapping[str, Any],
        coordinate: ProgressCoordinate,
        progress: list[ProgressCoordinate],
    ) -> PhaseExecutionResult:
        return PhaseExecutionResult(
            slots=_copy_executor_mapping(slots),
            coordinate=coordinate,
            progress=progress,
            checkpoints=self.checkpoint_store.as_dict(),
            checkpoint_visits=self.checkpoint_store.visits(),
        )

    @staticmethod
    def _validate_interval(value: int | None, *, name: str) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise WorkerContractValidationError(
                f"/checkpoint_progress/{name}",
                "interval must be a positive integer",
            )

    @staticmethod
    def _interval_due(completed_batches: int, interval: int, last_batch: int) -> bool:
        return (
            completed_batches > last_batch
            and completed_batches // interval > last_batch // interval
        )

    def _completed_training_batches(
        self,
        slots: Mapping[str, Any],
        coordinate: ProgressCoordinate,
    ) -> int:
        authority = self.program.batch_progress
        if authority is None:
            return coordinate.program_step
        path = "/phase_program/batch_progress"
        if authority.slot not in slots:
            raise WorkerContractValidationError(
                f"{path}/slot",
                f"declared batch-progress slot {authority.slot!r} is missing",
            )
        values = (
            [
                self._slice_mapped_value(slots[authority.slot], instance)
                for instance in range(self._mapped_size or 0)
            ]
            if authority.slot in self._mapped_slots
            else [slots[authority.slot]]
        )
        batches = [
            self._batch_progress_value(value, authority.field_path, path) for value in values
        ]
        if any(value != batches[0] for value in batches[1:]):
            raise WorkerContractValidationError(
                path,
                f"mapped batch-progress authorities diverged: {batches!r}",
            )
        return batches[0]

    @staticmethod
    def _slice_mapped_value(value: Any, instance: int) -> Any:
        return jt.map(
            lambda leaf: leaf[instance] if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
            value,
        )

    @staticmethod
    def _batch_progress_value(
        value: Any,
        field_path: Sequence[str | int],
        path: str,
    ) -> int:
        for position, segment in enumerate(field_path):
            segment_path = f"{path}/field_path/{position}"
            if isinstance(value, Mapping):
                if segment not in value:
                    raise WorkerContractValidationError(
                        segment_path,
                        f"key {segment!r} is missing from batch-progress slot",
                    )
                value = value[segment]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if not isinstance(segment, int) or segment >= len(value):
                    raise WorkerContractValidationError(
                        segment_path,
                        f"index {segment!r} is invalid for batch-progress slot",
                    )
                value = value[segment]
            else:
                raise WorkerContractValidationError(
                    segment_path,
                    "cannot traverse non-container in batch-progress slot",
                )
        try:
            scalar = jax.device_get(value)
            if hasattr(scalar, "item"):
                scalar = scalar.item()
            if isinstance(scalar, bool):
                raise TypeError("boolean is not a batch coordinate")
            batches = index(scalar)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise WorkerContractValidationError(
                path,
                "batch-progress slot must resolve to one integer scalar",
            ) from exc
        if batches < 0:
            raise WorkerContractValidationError(
                path,
                f"batch-progress slot resolved to negative batches={batches}",
            )
        return batches

    def _save_phase_barrier(
        self,
        phase_name: str,
        coordinate: ProgressCoordinate,
        slots: Mapping[str, Any],
    ) -> PhaseCheckpoint:
        barrier_name = self._phases[phase_name].checkpoint_barrier
        if barrier_name is None:
            raise WorkerContractValidationError(
                f"/phase_program/phases/{phase_name}/checkpoint_barrier",
                "authored checkpoint cadence requires a barrier for every phase where it fires",
            )
        return self._save_barrier(barrier_name, coordinate, slots)

    @staticmethod
    def _should_stop_at_barrier(
        barrier_name: str,
        *,
        stop_after_barrier: str | None,
        stop_after_next_barrier: Callable[[str], bool] | None,
    ) -> bool:
        return stop_after_barrier == barrier_name or (
            stop_after_next_barrier is not None and stop_after_next_barrier(barrier_name)
        )

    def _save_barrier(
        self,
        barrier_name: str,
        coordinate: ProgressCoordinate,
        slots: Mapping[str, Any],
    ) -> PhaseCheckpoint:
        barrier = self._barriers[barrier_name]
        capture_slots = [slot.slot for slot in barrier.slots]
        capture_slots.extend(
            sink.slot for sink in barrier.artifact_sinks if sink.slot not in capture_slots
        )
        captured = {
            slot: _copy_executor_value(slots[slot]) for slot in capture_slots if slot in slots
        }
        missing = [
            slot.slot for slot in barrier.slots if slot.required and slot.slot not in captured
        ]
        missing_artifact_sinks = [
            sink.slot
            for sink in barrier.artifact_sinks
            if sink.required and sink.slot not in captured
        ]
        if missing:
            raise WorkerContractValidationError(
                f"/checkpoint_barriers/{barrier_name}/slots",
                f"missing required checkpoint slots {missing!r}",
            )
        if missing_artifact_sinks:
            raise WorkerContractValidationError(
                f"/checkpoint_barriers/{barrier_name}/artifact_sinks",
                f"missing required artifact sink slots {missing_artifact_sinks!r}",
            )
        return self.checkpoint_store.save(
            PhaseCheckpoint(
                barrier=barrier_name,
                coordinate=coordinate.model_copy(update={"completed_barrier": barrier_name}),
                slots=captured,
            )
        )

    def _progress_metrics(self, slots: Mapping[str, Any]) -> dict[str, Any]:
        return {
            slot: _copy_executor_value(slots[slot]) for slot in self._metric_slots if slot in slots
        }

    def _resume_phase_for_barrier(self, barrier_name: str) -> str:
        barrier = self._barriers.get(barrier_name)
        if barrier is None:
            raise WorkerContractValidationError(
                f"/checkpoint_barriers/{barrier_name}",
                f"unknown checkpoint barrier {barrier_name!r}",
            )
        if barrier.resume_coordinate is not None:
            return barrier.resume_coordinate.phase
        return self._next_phase(barrier.phase) or barrier.phase

    def _resume_position(self, checkpoint: PhaseCheckpoint) -> tuple[str, int]:
        phase = self._phases.get(checkpoint.coordinate.phase)
        inner_step = checkpoint.coordinate.inner_step
        if phase is not None and inner_step is not None and 0 <= inner_step < phase.max_steps - 1:
            return phase.name, inner_step + 1
        return self._resume_phase_for_barrier(checkpoint.barrier), 0

    def _schedule_origin_step(self, phase_name: str, coordinate: ProgressCoordinate) -> int | None:
        phase = self._phases[phase_name]
        if phase.schedule_origin is None:
            return coordinate.schedule_origin_step
        if phase.schedule_origin.mode == "run_start":
            return phase.schedule_origin.step_offset
        if phase.schedule_origin.mode in {"phase_entry", "resume_barrier"}:
            return coordinate.program_step + phase.schedule_origin.step_offset
        return coordinate.schedule_origin_step

    def _next_phase(
        self,
        phase_name: str,
        slots: Mapping[str, Any] | None = None,
        coordinate: ProgressCoordinate | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> str | None:
        phase = self._phases[phase_name]
        if not phase.legal_next:
            return None
        for target in phase.legal_next:
            transition = self._transitions.get((phase_name, target))
            if transition is None:
                raise WorkerContractValidationError(
                    f"/phase_program/phases/{phase_name}/legal_next",
                    f"missing transition {phase_name!r} -> {target!r}",
                )
            if transition.guard is None:
                return target
            if slots is None or coordinate is None:
                continue
            predicate = self.guard_predicates.get(transition.guard.predicate_ref)
            if predicate is None:
                raise WorkerContractValidationError(
                    f"/phase_program/transitions/{phase_name}->{target}/guard",
                    f"missing predicate for guard {transition.guard.predicate_ref!r}",
                )
            if bool(predicate(slots, coordinate, dict(context or {}))):
                return target
        return None

    def _required_transition(self, phase_name: str, target: str) -> None:
        if (phase_name, target) not in self._transitions:
            raise WorkerContractValidationError(
                f"/phase_program/phases/{phase_name}/legal_next",
                f"missing transition {phase_name!r} -> {target!r}",
            )
