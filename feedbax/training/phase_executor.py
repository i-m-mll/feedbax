"""Generic phase-program executor for worker-contract tests and integrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from feedbax.contracts.worker import (
    PhaseProgramSpec,
    ProgressCoordinate,
)
from feedbax.training.worker_validation import (
    WorkerContractValidationError,
    validate_update_kernel_callable,
)

UpdateKernel = Callable[[Mapping[str, Any], ProgressCoordinate, Mapping[str, Any]], Mapping[str, Any]]


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
        return deepcopy(saved)

    def load(self, barrier: str) -> PhaseCheckpoint:
        try:
            return deepcopy(self._checkpoints[barrier][-1])
        except (KeyError, IndexError) as exc:
            raise WorkerContractValidationError(
                f"/checkpoint_store/{barrier}",
                f"unknown checkpoint barrier {barrier!r}",
            ) from exc

    def as_dict(self) -> dict[str, PhaseCheckpoint]:
        """Return a defensive copy of the latest checkpoint for each barrier."""
        return {
            barrier: deepcopy(checkpoints[-1])
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
        return tuple(deepcopy(visit) for visit in visits)

    def _checkpoint_with_visit_ordinal(self, checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        next_ordinal = self._next_visit_ordinal.get(checkpoint.barrier, 0)
        visit_ordinal = (
            next_ordinal
            if checkpoint.visit_ordinal is None
            else max(checkpoint.visit_ordinal, next_ordinal)
        )
        self._next_visit_ordinal[checkpoint.barrier] = visit_ordinal + 1
        return deepcopy(
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
    ) -> None:
        self.program = program
        self.kernels = dict(kernels)
        self.guard_predicates = dict(guard_predicates or {})
        self.checkpoint_store = checkpoint_store or InMemoryCheckpointStore()
        self._phases = {phase.name: phase for phase in program.phases}
        self._steps = {step.name: step for step in program.update_steps}
        self._transitions = {
            (transition.source, transition.target): transition
            for transition in program.transitions
        }
        self._barriers = {barrier.name: barrier for barrier in program.checkpoint_barriers}
        for kernel_ref, kernel in self.kernels.items():
            validate_update_kernel_callable(kernel, path=f"/kernels/{kernel_ref}")
        for predicate_ref, predicate in self.guard_predicates.items():
            validate_update_kernel_callable(predicate, path=f"/guard_predicates/{predicate_ref}")

    def run(
        self,
        slots: Mapping[str, Any],
        *,
        run_id: str,
        resume_from_barrier: str | None = None,
        stop_after_barrier: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> PhaseExecutionResult:
        """Execute phases from the start or from a checkpoint barrier."""
        progress: list[ProgressCoordinate] = []
        checkpoint_context = dict(context or {})
        if resume_from_barrier is not None:
            checkpoint = self.checkpoint_store.load(resume_from_barrier)
            current_slots = dict(checkpoint.slots)
            start_phase = self._resume_phase_for_barrier(resume_from_barrier)
            coordinate = ProgressCoordinate(
                run_id=run_id,
                phase=start_phase,
                global_step=checkpoint.coordinate.global_step,
                completed_barrier=resume_from_barrier,
            )
        else:
            current_slots = dict(deepcopy(slots))
            start_phase = self.program.initial_phase
            coordinate = ProgressCoordinate(run_id=run_id, phase=start_phase)

        phase_name: str | None = start_phase
        while phase_name is not None:
            phase = self._phases[phase_name]
            coordinate = coordinate.model_copy(
                update={
                    "phase": phase.name,
                    "schedule_origin_step": self._schedule_origin_step(phase.name, coordinate),
                }
            )
            for inner_step in range(phase.max_steps):
                coordinate = coordinate.model_copy(update={"inner_step": inner_step})
                for step_name in phase.update_steps:
                    step = self._steps[step_name]
                    kernel = self.kernels.get(step.kernel.kernel_ref)
                    if kernel is None:
                        raise WorkerContractValidationError(
                            f"/phase_program/update_steps/{step_name}/kernel_ref",
                            f"missing callable for kernel_ref {step.kernel.kernel_ref!r}",
                        )
                    updates = kernel(current_slots, coordinate, checkpoint_context)
                    unknown = sorted(set(updates) - set(step.writes))
                    if unknown:
                        raise WorkerContractValidationError(
                            f"/phase_program/update_steps/{step_name}/writes",
                            f"kernel returned undeclared writes {unknown!r}",
                        )
                    current_slots.update(deepcopy(dict(updates)))
                coordinate = coordinate.model_copy(
                    update={"global_step": coordinate.global_step + 1}
                )
                progress.append(coordinate)

            if phase.checkpoint_barrier is not None:
                saved_checkpoint = self._save_barrier(
                    phase.checkpoint_barrier,
                    coordinate,
                    current_slots,
                )
                coordinate = coordinate.model_copy(
                    update={"completed_barrier": saved_checkpoint.barrier}
                )
                if stop_after_barrier == phase.checkpoint_barrier:
                    return PhaseExecutionResult(
                        slots=current_slots,
                        coordinate=coordinate,
                        progress=progress,
                        checkpoints=self.checkpoint_store.as_dict(),
                        checkpoint_visits=self.checkpoint_store.visits(),
                    )

            phase_name = self._next_phase(
                phase.name,
                current_slots,
                coordinate,
                checkpoint_context,
            )

        return PhaseExecutionResult(
            slots=current_slots,
            coordinate=coordinate,
            progress=progress,
            checkpoints=self.checkpoint_store.as_dict(),
            checkpoint_visits=self.checkpoint_store.visits(),
        )

    def _save_barrier(
        self,
        barrier_name: str,
        coordinate: ProgressCoordinate,
        slots: Mapping[str, Any],
    ) -> PhaseCheckpoint:
        barrier = self._barriers[barrier_name]
        captured = {
            slot.slot: deepcopy(slots[slot.slot])
            for slot in barrier.slots
            if slot.slot in slots
        }
        missing = [slot.slot for slot in barrier.slots if slot.required and slot.slot not in captured]
        if missing:
            raise WorkerContractValidationError(
                f"/checkpoint_barriers/{barrier_name}/slots",
                f"missing required checkpoint slots {missing!r}",
            )
        return self.checkpoint_store.save(
            PhaseCheckpoint(
                barrier=barrier_name,
                coordinate=coordinate.model_copy(update={"completed_barrier": barrier_name}),
                slots=captured,
            )
        )

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

    def _schedule_origin_step(self, phase_name: str, coordinate: ProgressCoordinate) -> int | None:
        phase = self._phases[phase_name]
        if phase.schedule_origin is None:
            return coordinate.schedule_origin_step
        if phase.schedule_origin.mode == "run_start":
            return phase.schedule_origin.step_offset
        if phase.schedule_origin.mode in {"phase_entry", "resume_barrier"}:
            return coordinate.global_step + phase.schedule_origin.step_offset
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
