"""Generic phase-program executor for worker-contract tests and integrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from operator import index
import time
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np

from feedbax.contracts.checkpoints import PHASE_BOUNDARY_DECISION_METADATA_KEY
from feedbax.contracts.worker import (
    MaterializedMappingLevelSpec,
    MaterializedSlotAxisBinding,
    PhaseProgramSpec,
    ProgressCoordinate,
    StateSlotSpec,
    UpdateStepSpec,
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
CommittedUpdateCallback = Callable[[ProgressCoordinate, int, Mapping[str, Any]], None]
StepGuard = Callable[
    [Mapping[str, Any], ProgressCoordinate, Mapping[str, Any], "_StepObservation"],
    "StepGuardResult | None",
]


@dataclass(frozen=True)
class NumericalUpdateKernel:
    """Declare a pure numerical update and its runtime operand projection.

    Host coordinates are excluded. Selected keys and static values are fixed at
    preparation; changing numerical context (including counters) must use arrays.
    Ordinary calls evaluate the scientific function directly, without compilation.
    """

    update: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
    context_names: tuple[str, ...] = ()
    optional_context_names: tuple[str, ...] = ()
    excluded_slots: tuple[str, ...] = ()

    def project(
        self,
        slots: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        numeric_slots = {k: v for k, v in slots.items() if k not in self.excluded_slots}
        numeric_context = {k: context[k] for k in self.context_names}
        numeric_context.update({k: context[k] for k in self.optional_context_names if k in context})
        return numeric_slots, numeric_context

    def __call__(
        self,
        slots: Mapping[str, Any],
        coordinate: ProgressCoordinate,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del coordinate
        numeric_slots, numeric_context = self.project(slots, context)
        return _reconstruct_updates(self.update(numeric_slots, numeric_context), numeric_slots)


def _reconstruct_updates(
    updates: Mapping[str, Any],
    slots: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(updates)
    for name, value in result.items():
        if name in slots and jt.structure(value) == jt.structure(slots[name]):
            output_static = eqx.filter(value, lambda x: not eqx.is_array(x))
            input_static = eqx.filter(slots[name], lambda x: not eqx.is_array(x))
            if eqx.tree_equal(output_static, input_static) is True:
                result[name] = eqx.combine(eqx.filter(value, eqx.is_array), input_static)
    return result


@dataclass(frozen=True)
class _NumericalWrite:
    """Rebuild a write from compiled arrays and its static leaf sources."""

    tree: Any
    array_indices: tuple[int, ...]
    static_leaves: tuple[tuple[int, Any], ...]
    input_offset: int | None

    def restore(
        self,
        arrays: tuple[jax.Array, ...],
        input_leaves: list[Any],
        *,
        mapped: bool,
    ) -> Any:
        leaves = [None] * self.tree.num_leaves
        offset = self.input_offset
        for i, value in self.static_leaves:
            leaves[i] = value if offset is None else deepcopy(input_leaves[offset + i])
        for i, array in zip(self.array_indices, arrays, strict=True):
            if mapped and offset is not None:
                source = input_leaves[offset + i]
                # AOT canonicalizes NumPy dtypes, but a mapped write must keep
                # the live destination's dtype and array category.
                if not isinstance(source, jax.Array) and (
                    not isinstance(source, np.ndarray) or source.dtype != array.dtype
                ):
                    raise ValueError("mapped write changes destination array category or dtype")
            leaves[i] = array
        return self.tree.unflatten(leaves)


@dataclass(frozen=True)
class _PreparedNumericalUpdate:
    """A fixed input and write ABI around an array-only JAX executable."""

    declaration: NumericalUpdateKernel
    executable: Any
    static_inputs: tuple[Any, ...]
    static_indices: tuple[int, ...]
    array_indices: tuple[int, ...]
    input_tree: Any
    slot_names: tuple[str, ...]
    context_names: frozenset[str]
    writes: tuple[tuple[str, _NumericalWrite], ...]
    unprojected_writes: tuple[str, ...]
    mapped: bool
    custom_projection: bool

    def __call__(
        self,
        slots: Mapping[str, Any],
        coordinate: ProgressCoordinate,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del coordinate
        projected_slots, selected_context = self.declaration.project(slots, context)
        if selected_context.keys() != self.context_names:
            raise ValueError("selected numerical context keys changed after preparation")
        try:
            selected = {k: projected_slots[k] for k in self.slot_names}
        except KeyError as exc:
            missing = sorted(set(self.slot_names) - projected_slots.keys())
            raise ValueError(f"selected numerical slots disappeared: {missing!r}") from exc
        leaves, tree = jt.flatten((selected, selected_context))
        if tree != self.input_tree:
            raise ValueError("numerical input PyTree structure changed after preparation")
        static = tuple(leaves[i] for i in self.static_indices)
        if static and eqx.tree_equal(static, self.static_inputs) is not True:
            raise ValueError("numerical static inputs changed after preparation")
        arrays = tuple(leaves[i] for i in self.array_indices)
        if not all(map(eqx.is_array, arrays)):
            raise ValueError("numerical input array/static leaf category changed after preparation")
        outputs = self.executable(arrays)
        updates = {
            name: write.restore(outputs[name], leaves, mapped=self.mapped)
            for name, write in self.writes
        }
        if self.custom_projection:
            return _reconstruct_updates(updates, selected)
        return updates


@jax.jit
def _numeric_nan_flags(metrics: tuple[tuple[Any, ...], ...]) -> jax.Array:
    if not metrics:
        return jnp.zeros((0,), dtype=jnp.bool_)
    return jnp.stack(
        [jnp.any(jnp.stack([jnp.any(jnp.isnan(leaf)) for leaf in leaves])) for leaves in metrics]
    )


def _metric_nan_flags(metrics: Mapping[str, Any]) -> tuple[tuple[str, ...], jax.Array]:
    names = []
    numeric = []
    for name, value in sorted(metrics.items()):
        leaves = []
        for leaf in jt.leaves(value):
            try:
                array = leaf if isinstance(leaf, (jax.Array, np.ndarray)) else np.asarray(leaf)
                if jnp.issubdtype(array.dtype, jnp.number):
                    leaves.append(array)
            except (TypeError, ValueError):
                continue
        if leaves:
            names.append(name)
            numeric.append(tuple(leaves))
    return tuple(names), _numeric_nan_flags(tuple(numeric))


@dataclass(frozen=True)
class _StepObservation:
    tripped_metrics: tuple[str, ...]
    raw_batches: Any
    mapped: bool = False
    authority_error: WorkerContractValidationError | None = None

    def completed_batches(self) -> int:
        if self.authority_error is not None:
            raise self.authority_error
        values = self.raw_batches if self.mapped else (self.raw_batches,)
        path = "/phase_program/batch_progress"
        batches = [PhaseProgramExecutor._batch_progress_value(value, path) for value in values]
        if any(value != batches[0] for value in batches[1:]):
            raise WorkerContractValidationError(
                path, f"mapped batch-progress authorities diverged: {batches!r}"
            )
        return batches[0]


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
        boundary_decision=checkpoint.boundary_decision,
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


@dataclass(frozen=True)
class PhaseBoundaryDecision:
    """The transition selected at a completed phase, with None for termination."""

    next_phase: str | None

    @classmethod
    def from_checkpoint_metadata(
        cls, metadata: Mapping[str, Any]
    ) -> PhaseBoundaryDecision | None:
        if PHASE_BOUNDARY_DECISION_METADATA_KEY not in metadata:
            return None
        decision = metadata[PHASE_BOUNDARY_DECISION_METADATA_KEY]
        path = "/checkpoint/phase_boundary_decision"
        if (
            not isinstance(decision, Mapping)
            or set(decision) != {"schema_version", "next_phase"}
            or type(decision["schema_version"]) is not int
            or decision["schema_version"] != 1
        ):
            raise WorkerContractValidationError(path, "invalid or unsupported boundary decision")
        target = decision["next_phase"]
        if target is not None and (not isinstance(target, str) or not target):
            raise WorkerContractValidationError(f"{path}/next_phase", "invalid next phase")
        return cls(target)


@dataclass
class PhaseCheckpoint:
    """In-memory checkpoint captured at a phase barrier."""

    barrier: str
    coordinate: ProgressCoordinate
    slots: dict[str, Any]
    visit_ordinal: int | None = None
    boundary_decision: PhaseBoundaryDecision | None = None


@dataclass
class PhaseExecutionResult:
    """Result from executing a phase program."""

    slots: dict[str, Any]
    coordinate: ProgressCoordinate
    start_completed_batches: int
    end_completed_batches: int
    progress: list[ProgressCoordinate] = field(default_factory=list)
    checkpoints: dict[str, PhaseCheckpoint] = field(default_factory=dict)
    checkpoint_visits: tuple[PhaseCheckpoint, ...] = ()
    interrupted: bool = False
    resumed_terminal: bool = False


@dataclass(frozen=True)
class StepGuardResult:
    """Optional method-neutral control decision after one executor step."""

    halt: bool = False
    slots: Mapping[str, Any] | None = None
    coordinate: ProgressCoordinate | None = None


class PhaseCheckpointWriter(Protocol):
    """Accept checkpoint payloads captured by the phase executor."""

    def save(self, checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        """Persist one checkpoint visit and return the stored checkpoint."""


@runtime_checkable
class PhaseCheckpointStore(PhaseCheckpointWriter, Protocol):
    """Checkpoint writer that also keeps payloads available in memory."""

    def load(self, barrier: str) -> PhaseCheckpoint:
        """Return the latest checkpoint visit for ``barrier``."""

    def as_dict(self) -> dict[str, PhaseCheckpoint]:
        """Return the latest checkpoint for each barrier."""

    def visits(self, barrier: str | None = None) -> tuple[PhaseCheckpoint, ...]:
        """Return stored checkpoint visits in append order."""


def _next_checkpoint_visit_ordinal(
    checkpoint: PhaseCheckpoint,
    previous_ordinal: int | None,
) -> int:
    next_ordinal = 0 if previous_ordinal is None else previous_ordinal + 1
    return max(next_ordinal, checkpoint.visit_ordinal or 0)


class InMemoryCheckpointStore:
    """Append-only checkpoint store used by the generic phase executor.

    Repeated visits to the same barrier are retained with per-barrier visit
    ordinals. ``load()`` and ``as_dict()`` expose the latest visit for resume and
    legacy callers.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, list[PhaseCheckpoint]] = {}
        self._visit_log: list[PhaseCheckpoint] = []

    def save(self, checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        return self.remember(checkpoint)

    def remember(self, checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        visits = self._checkpoints.get(checkpoint.barrier)
        previous_ordinal = visits[-1].visit_ordinal if visits else None
        ordinal = _next_checkpoint_visit_ordinal(checkpoint, previous_ordinal)
        saved = _copy_phase_checkpoint(replace(checkpoint, visit_ordinal=ordinal))
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


class PhaseProgramExecutor:
    """Execute a validated phase program through fixed-signature kernels."""

    def __init__(
        self,
        program: PhaseProgramSpec,
        kernels: Mapping[str, UpdateKernel],
        *,
        guard_predicates: Mapping[str, UpdateKernel] | None = None,
        checkpoint_store: PhaseCheckpointWriter | None = None,
        state_slots: Sequence[StateSlotSpec] = (),
        mapping_levels: Sequence[MaterializedMappingLevelSpec] = (),
        slot_axis_bindings: Mapping[str, Sequence[MaterializedSlotAxisBinding]] | None = None,
    ) -> None:
        self.program = program
        self.kernels = dict(kernels)
        self._prepared_numerical: dict[str, _PreparedNumericalUpdate] = {}
        self.preparation_telemetry: dict[str, Any] = {}
        self.guard_predicates = dict(guard_predicates or {})
        self.checkpoint_store = (
            InMemoryCheckpointStore() if checkpoint_store is None else checkpoint_store
        )
        self._checkpoint_history = (
            self.checkpoint_store
            if isinstance(self.checkpoint_store, PhaseCheckpointStore)
            else None
        )
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

    def _prepare_numerical(
        self,
        kernel: NumericalUpdateKernel,
        step: UpdateStepSpec,
        slots: Mapping[str, Any],
        context: Mapping[str, Any],
        *,
        mapped_step: bool,
    ) -> _PreparedNumericalUpdate:
        inputs = kernel.project(slots, context)
        numeric_slots, numeric_context = inputs
        input_leaves, input_tree = jt.flatten(inputs)
        slot_names = tuple(sorted(numeric_slots))
        array_indices = tuple(i for i, x in enumerate(input_leaves) if eqx.is_array(x))
        static_indices = tuple(i for i, x in enumerate(input_leaves) if not eqx.is_array(x))
        destinations = {}
        custom_projection = type(kernel).project is not NumericalUpdateKernel.project
        # Only the declared key projection proves that the operand is the destination.
        if not custom_projection:
            offset = 0
            for name, tree in zip(slot_names, input_tree.children()[0].children(), strict=True):
                destinations[name] = offset
                offset += tree.num_leaves
        writes = {}

        def numerical_call(arrays):
            leaves = input_leaves.copy()
            for i, array in zip(array_indices, arrays, strict=True):
                leaves[i] = array
            selected_slots, selected_context = input_tree.unflatten(leaves)
            if mapped_step:
                mapped = {k: v for k, v in selected_slots.items() if k in self._mapped_slots}
                shared = {
                    k: v
                    for k, v in selected_slots.items()
                    if k not in self._mapped_slots
                }

                def scalar_call(mapped, shared, ctx):
                    return kernel.update({**shared, **mapped}, ctx)

                updates = eqx.filter_vmap(
                    scalar_call,
                    in_axes=(eqx.if_array(0), None, None),
                    out_axes=eqx.if_array(0),
                )(mapped, shared, selected_context)
            else:
                updates = kernel.update(selected_slots, selected_context)
            output_template = jt.map(
                lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype)
                if eqx.is_array(x)
                else x,
                updates,
            )
            self._validate_destination_updates(
                step.name,
                step.writes,
                output_template,
                slots,
                fixed_static=True,
            )
            outputs = {}
            for name, value in updates.items():
                leaves, tree = jt.flatten(value)
                array_positions = tuple(i for i, x in enumerate(leaves) if eqx.is_array(x))
                writes[name] = _NumericalWrite(
                    tree=tree,
                    array_indices=array_positions,
                    static_leaves=tuple(
                        (i, x) for i, x in enumerate(leaves) if not eqx.is_array(x)
                    ),
                    input_offset=destinations.get(name),
                )
                outputs[name] = tuple(leaves[i] for i in array_positions)
            return outputs

        if mapped_step and set(step.writes) & self._shared_slots:
            raise ValueError("mapped numerical step writes shared slots")
        executable = jax.jit(numerical_call).lower(
            tuple(input_leaves[i] for i in array_indices)
        ).compile()
        return _PreparedNumericalUpdate(
            declaration=kernel,
            executable=executable,
            static_inputs=tuple(deepcopy(input_leaves[i]) for i in static_indices),
            static_indices=static_indices,
            array_indices=array_indices,
            input_tree=input_tree,
            slot_names=slot_names,
            context_names=frozenset(numeric_context),
            writes=tuple(writes.items()),
            unprojected_writes=tuple(name for name in writes if name not in destinations),
            mapped=mapped_step,
            custom_projection=custom_projection,
        )

    def preflight(
        self,
        slots: Mapping[str, Any],
        *,
        run_id: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Prepare numerical executables and shape-check remaining mapped kernels."""
        self._prepared_numerical = {}
        preparation_seconds = 0.0
        coordinate = ProgressCoordinate(run_id=run_id, phase=self.program.initial_phase)
        prepared = []
        for step in self.program.update_steps:
            phases = [phase for phase in self.program.phases if step.name in phase.update_steps]
            if not phases:
                continue
            kernel = self.kernels[step.kernel.kernel_ref]
            mapped_step = self._active_axis is not None and self._active_axis in step.axes
            try:
                if isinstance(kernel, NumericalUpdateKernel):
                    started = time.perf_counter()
                    self._prepared_numerical[step.name] = self._prepare_numerical(
                        kernel, step, slots, dict(context or {}), mapped_step=mapped_step
                    )
                    preparation_seconds += time.perf_counter() - started
                    prepared.append({"step": step.name, "kernel_ref": step.kernel.kernel_ref})
                elif mapped_step:
                    mapped = {k: slots[k] for k in self._mapped_slots if k in slots}
                    shared = {k: slots[k] for k in self._shared_slots if k in slots}
                    for phase in phases:
                        updates = eqx.filter_eval_shape(
                            self._mapped_kernel(kernel),
                            mapped,
                            shared,
                            coordinate.model_copy(update={"phase": phase.name}),
                            dict(context or {}),
                        )
                        self._validate_destination_updates(step.name, step.writes, updates, slots)

            except Exception as exc:
                raise WorkerContractValidationError(
                    f"/phase_program/update_steps/{step.name}",
                    f"kernel preparation failed: {exc}",
                ) from exc
        self.preparation_telemetry = {
            "kernel_preparation_seconds": preparation_seconds if prepared else None,
            "kernel_preparation_steps": prepared,
            "kernel_preparation_scope": "declared numerical updates: tracing, lowering, compilation/cache load",
        }

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
                f"checkpoint slots lack prepared runtime templates {missing_prepared!r}",
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
        resume_checkpoint: PhaseCheckpoint | None = None,
        extend_terminal_phase: bool = False,
        stop_after_barrier: str | None = None,
        stop_after_next_barrier: Callable[[str], bool] | None = None,
        context: Mapping[str, Any] | None = None,
        start_callback: Callable[[int], None] | None = None,
        progress_callback: ProgressCallback | None = None,
        retain_progress: bool = True,
        method_observation_callback: MethodObservationCallback | None = None,
        committed_update_callback: CommittedUpdateCallback | None = None,
        step_guard: StepGuard | None = None,
        checkpoint_interval: int | None = None,
        progress_interval: int | None = None,
        include_completed_batches_in_progress: bool = False,
        update_budget: int | None = None,
        stop_before_update: Callable[[], bool] | None = None,
    ) -> PhaseExecutionResult:
        """Execute phases from the start or from a checkpoint barrier.

        Authored intervals are evaluated against the phase program's declared
        training-batch authority, not against phase boundaries or checkpoint
        visit ordinals. ``update_budget`` uses the same authority as a delta
        from the entry counter and may end naturally before the budget is used.
        A terminal partial interval is always emitted. A stop observed before
        an update saves any unsaved progress at the last committed coordinate
        through that phase's authored barrier, even without a checkpoint interval.
        Set `retain_progress=False` to omit the observations from the result
        while still delivering them to `progress_callback`.

        ``start_callback`` receives the entry batch after restoration and before updates.
        ``committed_update_callback`` receives the accepted coordinate, batch count,
        and read-only slot mapping after each update, independent of history cadence.

        A checkpoint restored by external custody can be passed directly as
        ``resume_checkpoint``. ``resume_from_barrier`` selects a stored in-memory visit.
        Set ``extend_terminal_phase`` only for an authorized additional-batch
        continuation that can restart a terminal phase.
        """
        self._validate_interval(checkpoint_interval, name="checkpoint_interval")
        self._validate_interval(progress_interval, name="progress_interval")
        if update_budget is not None and (
            isinstance(update_budget, bool)
            or not isinstance(update_budget, int)
            or update_budget <= 0
        ):
            raise WorkerContractValidationError(
                "/runtime/update_budget",
                "update_budget must be a positive non-boolean integer",
            )
        progress: list[ProgressCoordinate] = []
        checkpoint_context = dict(context or {})
        mapped_kernels = {
            kernel_ref: self._mapped_kernel(kernel) for kernel_ref, kernel in self.kernels.items()
        }
        resume_inner_step = 0
        resuming_within_phase = False
        if resume_from_barrier is not None:
            if resume_checkpoint is not None or self._checkpoint_history is None:
                raise WorkerContractValidationError(
                    "/checkpoint_store",
                    "resume_from_barrier requires an in-memory checkpoint store "
                    "and cannot be used with resume_checkpoint",
                )
            resume_checkpoint = self._checkpoint_history.load(resume_from_barrier)
        if resume_checkpoint is not None:
            current_slots = self.merge_resume_slots(slots, resume_checkpoint)
            start_phase, resume_inner_step = self.resume_position(
                resume_checkpoint.barrier,
                resume_checkpoint.coordinate,
                resume_checkpoint.boundary_decision,
                extend_terminal_phase=extend_terminal_phase,
            )
            resuming_within_phase = resume_inner_step > 0
            coordinate = resume_checkpoint.coordinate.model_copy(
                update={
                    "run_id": run_id,
                    "completed_barrier": resume_checkpoint.barrier,
                }
            )
        else:
            current_slots = _copy_executor_mapping(slots)
            start_phase = self.program.initial_phase
            coordinate = ProgressCoordinate(run_id=run_id, phase=start_phase)

        if (
            start_phase is not None
            and any(isinstance(k, NumericalUpdateKernel) for k in self.kernels.values())
            and not self._prepared_numerical
        ):
            self.preflight(current_slots, run_id=run_id, context=checkpoint_context)
        initial_batch = self._completed_training_batches(current_slots, coordinate)
        if start_callback is not None:
            start_callback(initial_batch)
        previous_completed_batch = initial_batch
        last_progress_batch = initial_batch
        last_checkpoint_batch = initial_batch
        last_checkpoint_step = coordinate.program_step
        phase_name: str | None = start_phase
        interrupted = False
        last_committed_boundary_decision = None
        while phase_name is not None:
            phase = self._phases[phase_name]
            entry_coordinate = coordinate
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
            boundary_decision = None
            for inner_step in range(resume_inner_step, phase.max_steps):
                if phase.update_steps and stop_before_update is not None and stop_before_update():
                    if inner_step == resume_inner_step:
                        coordinate = entry_coordinate
                    interrupted = True
                    break
                coordinate = coordinate.model_copy(update={"inner_step": inner_step})
                for step_name in phase.update_steps:
                    step = self._steps[step_name]
                    kernel = self.kernels.get(step.kernel.kernel_ref)
                    if kernel is None:
                        raise WorkerContractValidationError(
                            f"/phase_program/update_steps/{step_name}/kernel_ref",
                            f"missing callable for kernel_ref {step.kernel.kernel_ref!r}",
                        )
                    prepared = None
                    if isinstance(kernel, NumericalUpdateKernel):
                        try:
                            prepared = self._prepared_numerical[step.name]
                            updates = prepared(current_slots, coordinate, checkpoint_context)
                        except Exception as exc:
                            raise WorkerContractValidationError(
                                f"/phase_program/update_steps/{step.name}",
                                f"prepared numerical execution failed: {exc}",
                            ) from exc
                        if prepared.mapped and prepared.unprojected_writes:
                            self._validate_destination_updates(
                                step.name,
                                step.writes,
                                {name: updates[name] for name in prepared.unprojected_writes},
                                current_slots,
                            )
                    elif self._active_axis is not None and self._active_axis in step.axes:
                        updates = mapped_kernels[step.kernel.kernel_ref](
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
                        self._validate_destination_updates(
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
                    if prepared is None:
                        updates = _copy_executor_mapping(updates)
                    elif prepared.unprojected_writes:
                        updates = {
                            **updates,
                            **{
                                name: _copy_executor_value(updates[name])
                                for name in prepared.unprojected_writes
                            },
                        }
                    current_slots.update(updates)
                coordinate = coordinate.model_copy(
                    update={
                        "program_step": coordinate.program_step + 1,
                        "metrics": self._progress_metrics(current_slots),
                    }
                )
                observation = self._step_observation(
                    current_slots, coordinate, include_health=step_guard is not None
                )
                if step_guard is not None:
                    guard_result = step_guard(
                        current_slots, coordinate, checkpoint_context, observation
                    )
                    if guard_result is not None and guard_result.halt:
                        halted_slots = (
                            current_slots if guard_result.slots is None else guard_result.slots
                        )
                        halted_coordinate = guard_result.coordinate or coordinate
                        return self._result(
                            halted_slots,
                            halted_coordinate,
                            progress,
                            start_completed_batches=initial_batch,
                        )
                completed_batches = observation.completed_batches()
                if completed_batches < previous_completed_batch:
                    raise WorkerContractValidationError(
                        "/phase_program/batch_progress",
                        "completed training batches must be monotonic; "
                        f"previous={previous_completed_batch}, current={completed_batches}",
                    )
                previous_completed_batch = completed_batches
                if committed_update_callback is not None:
                    committed_update_callback(
                        coordinate, completed_batches, MappingProxyType(current_slots)
                    )
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
                    if retain_progress:
                        progress.append(observed_coordinate)
                    last_progress_batch = completed_batches
                    if progress_callback is not None:
                        progress_callback(_copy_progress_coordinate(observed_coordinate))
                completed_batch_delta = completed_batches - initial_batch
                if update_budget is not None and completed_batch_delta > update_budget:
                    raise WorkerContractValidationError(
                        "/runtime/update_budget",
                        "completed training batches exceeded update_budget; "
                        f"start={initial_batch}, current={completed_batches}, "
                        f"budget={update_budget}",
                    )
                budget_reached = update_budget is not None and completed_batch_delta == update_budget
                interval_due = checkpoint_interval is not None and self._interval_due(
                    completed_batches, checkpoint_interval, last_checkpoint_batch
                )
                if inner_step == phase.max_steps - 1:
                    boundary_coordinate = coordinate
                    if interval_due and phase.checkpoint_barrier is not None:
                        boundary_coordinate = coordinate.model_copy(
                            update={"completed_barrier": phase.checkpoint_barrier}
                        )
                    boundary_decision = PhaseBoundaryDecision(
                        self._next_phase(
                            phase.name, (current_slots, boundary_coordinate, checkpoint_context)
                        )
                    )
                if budget_reached or interval_due:
                    saved_barrier = self._save_phase_barrier(
                        phase.name, coordinate, current_slots, boundary_decision
                    )
                    last_checkpoint_batch = completed_batches
                    last_checkpoint_step = coordinate.program_step
                    coordinate = coordinate.model_copy(
                        update={"completed_barrier": saved_barrier}
                    )
                    if budget_reached or self._should_stop_at_barrier(
                        saved_barrier,
                        stop_after_barrier=stop_after_barrier,
                        stop_after_next_barrier=stop_after_next_barrier,
                    ):
                        return self._result(
                            current_slots,
                            coordinate,
                            progress,
                            start_completed_batches=initial_batch,
                        )

            next_phase = boundary_decision.next_phase if boundary_decision is not None else None
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
                if retain_progress:
                    progress.append(observed_coordinate)
                last_progress_batch = completed_batches
                if progress_callback is not None:
                    progress_callback(_copy_progress_coordinate(observed_coordinate))

            saved_barrier: str | None = None
            if interrupted:
                if coordinate.program_step != last_checkpoint_step:
                    saved_barrier = self._save_phase_barrier(
                        coordinate.phase,
                        coordinate,
                        current_slots,
                        last_committed_boundary_decision
                        if coordinate.program_step == entry_coordinate.program_step
                        else None,
                    )
            elif checkpoint_interval is None:
                if phase.checkpoint_barrier is not None:
                    saved_barrier = self._save_barrier(
                        phase.checkpoint_barrier, coordinate, current_slots, boundary_decision
                    )
            elif (
                next_phase is None
                and completed_batches != last_checkpoint_batch
            ):
                saved_barrier = self._save_phase_barrier(
                    coordinate.phase, coordinate, current_slots, boundary_decision
                )
            if saved_barrier is not None:
                last_checkpoint_step = coordinate.program_step
                coordinate = coordinate.model_copy(update={"completed_barrier": saved_barrier})
                if not interrupted and self._should_stop_at_barrier(
                    saved_barrier,
                    stop_after_barrier=stop_after_barrier,
                    stop_after_next_barrier=stop_after_next_barrier,
                ):
                    return self._result(
                        current_slots,
                        coordinate,
                        progress,
                        start_completed_batches=initial_batch,
                    )

            last_committed_boundary_decision = boundary_decision
            phase_name = next_phase
            resume_inner_step = 0
            resuming_within_phase = False

        return self._result(
            current_slots,
            coordinate,
            progress,
            start_completed_batches=initial_batch,
            interrupted=interrupted,
            resumed_terminal=resume_checkpoint is not None and start_phase is None,
        )

    def _validate_destination_updates(
        self,
        step_name: str,
        declared_writes: Sequence[str],
        updates: Mapping[str, Any],
        input_slots: Mapping[str, Any],
        *,
        fixed_static: bool = False,
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
                    "write destination is absent from executor state",
                )
            output_leaves, output_def = jt.flatten(value)
            input_leaves, input_def = jt.flatten(input_slots[slot])
            if output_def != input_def:
                raise WorkerContractValidationError(
                    f"/phase_program/update_steps/{step_name}/writes/{slot}",
                    "write must retain destination PyTree definition",
                )
            for leaf_index, (source, target) in enumerate(
                zip(input_leaves, output_leaves, strict=True)
            ):
                source_array = isinstance(source, array_types)
                target_array = isinstance(target, array_types)
                if source_array != target_array:
                    raise WorkerContractValidationError(
                        f"/phase_program/update_steps/{step_name}/writes/{slot}/{leaf_index}",
                        "write changes array/static leaf category",
                    )
                if source_array and (source.shape != target.shape or source.dtype != target.dtype):
                    raise WorkerContractValidationError(
                        f"/phase_program/update_steps/{step_name}/writes/{slot}/{leaf_index}",
                        "write changes destination shape or dtype; "
                        f"expected={source.shape!r}/{source.dtype}, "
                        f"observed={target.shape!r}/{target.dtype}",
                    )
                if fixed_static and not source_array and eqx.tree_equal(source, target) is not True:
                    raise WorkerContractValidationError(
                        f"/phase_program/update_steps/{step_name}/writes/{slot}/{leaf_index}",
                        "numerical write changes destination static value",
                    )

    def _result(
        self,
        slots: Mapping[str, Any],
        coordinate: ProgressCoordinate,
        progress: list[ProgressCoordinate],
        *,
        start_completed_batches: int,
        interrupted: bool = False,
        resumed_terminal: bool = False,
    ) -> PhaseExecutionResult:
        result = PhaseExecutionResult(
            slots=_copy_executor_mapping(slots),
            coordinate=coordinate,
            start_completed_batches=start_completed_batches,
            end_completed_batches=self._completed_training_batches(slots, coordinate),
            progress=progress,
            interrupted=interrupted,
            resumed_terminal=resumed_terminal,
        )
        if self._checkpoint_history is not None:
            result.checkpoints = self._checkpoint_history.as_dict()
            result.checkpoint_visits = self._checkpoint_history.visits()
        return result

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
        return self._step_observation(slots, coordinate, include_health=False).completed_batches()

    def _step_observation(
        self,
        slots: Mapping[str, Any],
        coordinate: ProgressCoordinate,
        *,
        include_health: bool = True,
    ) -> _StepObservation:
        names, flags = _metric_nan_flags(coordinate.metrics) if include_health else ((), ())
        authority = self.program.batch_progress
        error = None
        mapped = False
        raw = coordinate.program_step
        if authority is not None:
            try:
                if authority.slot not in slots:
                    raise WorkerContractValidationError(
                        "/phase_program/batch_progress/slot",
                        f"declared batch-progress slot {authority.slot!r} is missing",
                    )
                raw = self._batch_progress_field(
                    slots[authority.slot], authority.field_path, "/phase_program/batch_progress"
                )
                mapped = authority.slot in self._mapped_slots and isinstance(
                    raw, (jax.Array, np.ndarray)
                )
            except WorkerContractValidationError as exc:
                error = exc
                raw = None
        observed_flags, observed_raw = jax.device_get((flags, raw))
        return _StepObservation(
            tuple(name for name, flag in zip(names, observed_flags, strict=True) if flag),
            observed_raw,
            mapped,
            error,
        )

    @staticmethod
    def _slice_mapped_value(value: Any, instance: int) -> Any:
        return jt.map(
            lambda leaf: leaf[instance] if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
            value,
        )

    @staticmethod
    def _batch_progress_field(
        value: Any,
        field_path: Sequence[str | int],
        path: str,
    ) -> Any:
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
        return value

    @staticmethod
    def _batch_progress_value(value: Any, path: str) -> int:
        try:
            scalar = value.item() if hasattr(value, "item") else value
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
        boundary_decision: PhaseBoundaryDecision | None = None,
    ) -> str:
        barrier_name = self._phases[phase_name].checkpoint_barrier
        if barrier_name is None:
            raise WorkerContractValidationError(
                f"/phase_program/phases/{phase_name}/checkpoint_barrier",
                "cannot checkpoint a phase without an authored barrier",
            )
        return self._save_barrier(barrier_name, coordinate, slots, boundary_decision)

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
        boundary_decision: PhaseBoundaryDecision | None = None,
    ) -> str:
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
        saved = self.checkpoint_store.save(
            PhaseCheckpoint(
                barrier=barrier_name,
                coordinate=coordinate.model_copy(update={"completed_barrier": barrier_name}),
                slots=captured,
                boundary_decision=boundary_decision,
            )
        )
        return saved.barrier

    def _progress_metrics(self, slots: Mapping[str, Any]) -> dict[str, Any]:
        return {
            slot: _copy_executor_value(slots[slot]) for slot in self._metric_slots if slot in slots
        }

    def resume_position(
        self,
        barrier_name: str,
        coordinate: ProgressCoordinate,
        decision: PhaseBoundaryDecision | None,
        *,
        extend_terminal_phase: bool = False,
    ) -> tuple[str | None, int]:
        """Resolve the resume point using the program and recorded checkpoint metadata."""
        barrier = self._barriers[barrier_name]
        coordinate_phase = self._phases.get(coordinate.phase)
        inner_step = coordinate.inner_step
        next_inner_step = (
            inner_step + 1
            if coordinate_phase is not None
            and inner_step is not None
            and 0 <= inner_step < coordinate_phase.max_steps - 1
            else 0
        )
        if decision is None and next_inner_step:
            return coordinate.phase, next_inner_step
        if barrier.resume_coordinate is not None:
            return barrier.resume_coordinate.phase, 0
        if decision is None:
            phase_name = barrier.phase
            next_phase = self._next_phase(phase_name, None)
        else:
            if coordinate_phase is None or coordinate_phase.checkpoint_barrier != barrier.name:
                raise WorkerContractValidationError(
                    "/checkpoint/phase_boundary_decision",
                    "recorded phase boundary does not belong to the checkpoint barrier",
                )
            phase_name = coordinate_phase.name
            next_phase = decision.next_phase
            if inner_step != coordinate_phase.max_steps - 1 and not (
                extend_terminal_phase and next_phase is None and next_inner_step
            ):
                raise WorkerContractValidationError(
                    "/checkpoint/phase_boundary_decision",
                    "recorded decision requires a completed phase boundary",
                )
            if next_phase is not None and (
                next_phase not in coordinate_phase.legal_next
                or (phase_name, next_phase) not in self._transitions
            ):
                raise WorkerContractValidationError(
                    "/checkpoint/phase_boundary_decision/next_phase",
                    f"recorded successor {next_phase!r} is not legal for phase {phase_name!r}",
                )
        if next_phase is None and extend_terminal_phase:
            return phase_name, next_inner_step
        return next_phase, 0

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
        guard_inputs: tuple[Mapping[str, Any], ProgressCoordinate, Mapping[str, Any]] | None,
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
            if guard_inputs is None:
                raise WorkerContractValidationError(
                    "/checkpoint/phase_boundary_decision",
                    f"legacy checkpoint for guarded phase {phase_name!r} lacks a recorded "
                    "boundary decision; resume from a checkpoint with a recorded decision "
                    "or an authored resume coordinate",
                )
            slots, coordinate, context = guard_inputs
            predicate = self.guard_predicates.get(transition.guard.predicate_ref)
            if predicate is None:
                raise WorkerContractValidationError(
                    f"/phase_program/transitions/{phase_name}->{target}/guard",
                    f"missing predicate for guard {transition.guard.predicate_ref!r}",
                )
            if bool(predicate(slots, coordinate, dict(context))):
                return target
        return None

    def _required_transition(self, phase_name: str, target: str) -> None:
        if (phase_name, target) not in self._transitions:
            raise WorkerContractValidationError(
                f"/phase_program/phases/{phase_name}/legal_next",
                f"missing transition {phase_name!r} -> {target!r}",
            )
