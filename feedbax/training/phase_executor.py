"""Generic phase-program executor for worker-contract tests and integrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class PhaseExecutionResult:
    """Result from executing a phase program."""

    slots: dict[str, Any]
    coordinate: ProgressCoordinate
    progress: list[ProgressCoordinate] = field(default_factory=list)
    checkpoints: dict[str, PhaseCheckpoint] = field(default_factory=dict)


class InMemoryCheckpointStore:
    """Simple checkpoint store used by the generic phase executor."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, PhaseCheckpoint] = {}

    def save(self, checkpoint: PhaseCheckpoint) -> None:
        self._checkpoints[checkpoint.barrier] = deepcopy(checkpoint)

    def load(self, barrier: str) -> PhaseCheckpoint:
        try:
            return deepcopy(self._checkpoints[barrier])
        except KeyError as exc:
            raise WorkerContractValidationError(
                f"/checkpoint_store/{barrier}",
                f"unknown checkpoint barrier {barrier!r}",
            ) from exc

    def as_dict(self) -> dict[str, PhaseCheckpoint]:
        """Return a defensive copy of all stored checkpoints."""
        return deepcopy(self._checkpoints)


class PhaseProgramExecutor:
    """Execute a validated phase program through fixed-signature kernels."""

    def __init__(
        self,
        program: PhaseProgramSpec,
        kernels: Mapping[str, UpdateKernel],
        *,
        checkpoint_store: InMemoryCheckpointStore | None = None,
    ) -> None:
        self.program = program
        self.kernels = dict(kernels)
        self.checkpoint_store = checkpoint_store or InMemoryCheckpointStore()
        self._phases = {phase.name: phase for phase in program.phases}
        self._steps = {step.name: step for step in program.update_steps}
        self._transitions = {(transition.source, transition.target): transition for transition in program.transitions}
        self._barriers = {barrier.name: barrier for barrier in program.checkpoint_barriers}
        for kernel_ref, kernel in self.kernels.items():
            validate_update_kernel_callable(kernel, path=f"/kernels/{kernel_ref}")

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
            coordinate = coordinate.model_copy(update={"phase": phase.name})
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
                self._save_barrier(phase.checkpoint_barrier, coordinate, current_slots)
                coordinate = coordinate.model_copy(
                    update={"completed_barrier": phase.checkpoint_barrier}
                )
                if stop_after_barrier == phase.checkpoint_barrier:
                    return PhaseExecutionResult(
                        slots=current_slots,
                        coordinate=coordinate,
                        progress=progress,
                        checkpoints=self.checkpoint_store.as_dict(),
                    )

            phase_name = self._next_phase(phase.name)

        return PhaseExecutionResult(
            slots=current_slots,
            coordinate=coordinate,
            progress=progress,
            checkpoints=self.checkpoint_store.as_dict(),
        )

    def _save_barrier(
        self,
        barrier_name: str,
        coordinate: ProgressCoordinate,
        slots: Mapping[str, Any],
    ) -> None:
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
        self.checkpoint_store.save(
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

    def _next_phase(self, phase_name: str) -> str | None:
        phase = self._phases[phase_name]
        if not phase.legal_next:
            return None
        target = phase.legal_next[0]
        if (phase_name, target) not in self._transitions:
            raise WorkerContractValidationError(
                f"/phase_program/phases/{phase_name}/legal_next",
                f"missing transition {phase_name!r} -> {target!r}",
            )
        return target
