"""Fail-closed validation for method-neutral worker contracts."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import Field

from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.worker import (
    FIXED_UPDATE_KERNEL_SIGNATURE,
    AxisRole,
    CheckpointSlotManifest,
    EffectivePhaseSpec,
    MethodContractSpec,
    PhaseKind,
    ProgressCoordinate,
    ReducerRequirement,
    StateSlotRole,
    UpdateStepKind,
    derive_consistency_predicate,
)
from feedbax.studio.protocol import GRAPH_BINDABLE_TASK_DATA_ROLES


SUPPORTED_AXIS_ROLES: frozenset[AxisRole] = frozenset(
    {
        "authored_sweep",
        "batch",
        "environment",
        "replicate",
        "member",
        "realization",
        "phase",
        "rollout",
        "epoch",
        "minibatch",
    }
)
SUPPORTED_STATE_SLOT_ROLES: frozenset[StateSlotRole] = frozenset(
    {
        "model",
        "optimizer",
        "prng",
        "auxiliary",
        "population",
        "environment",
        "objective",
        "checkpoint",
        "metric",
    }
)
SUPPORTED_PHASE_KINDS: frozenset[PhaseKind] = frozenset(
    {
        "warmup",
        "collect",
        "advantage",
        "inner_loop",
        "outer_loop",
        "adversarial",
        "evaluation",
        "checkpoint",
        "custom",
    }
)
SUPPORTED_UPDATE_STEP_KINDS: frozenset[UpdateStepKind] = frozenset(
    {
        "collect",
        "gradient",
        "projection",
        "reduce",
        "checkpoint",
        "measurement",
        "control",
        "custom",
    }
)


class WorkerContractValidationError(ValueError):
    """Path-addressed worker contract validation failure."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.detail = message
        super().__init__(f"{path}: {message}")


class WorkerExecutabilityEnvironment(StrictModel):
    """Capabilities available to one concrete worker launch."""

    supported_axis_roles: set[AxisRole] = Field(default_factory=lambda: set(SUPPORTED_AXIS_ROLES))
    supported_state_slot_roles: set[StateSlotRole] = Field(
        default_factory=lambda: set(SUPPORTED_STATE_SLOT_ROLES)
    )
    supported_phase_kinds: set[PhaseKind] = Field(
        default_factory=lambda: set(SUPPORTED_PHASE_KINDS)
    )
    supported_update_step_kinds: set[UpdateStepKind] = Field(
        default_factory=lambda: set(SUPPORTED_UPDATE_STEP_KINDS)
    )
    supported_axes: set[str] | None = None
    supported_state_slots: set[str] | None = None
    require_dry_run_check: bool = False


class DryRunShapeCheckResult(StrictModel):
    """Result of the one-step dry-run/JIT shape check seam."""

    passed: bool
    exercised_payload_fields: list[str] = Field(default_factory=list)
    message: str | None = None


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise WorkerContractValidationError(path, message)


def _unique_by_name(items: Sequence[Any], path: str) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for index, item in enumerate(items):
        name = getattr(item, "name")
        if name in seen:
            raise WorkerContractValidationError(f"{path}/{index}/name", f"duplicate name {name!r}")
        seen[name] = item
    return seen


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata(value: Any) -> dict[str, Any]:
    metadata = _field(value, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def validate_update_kernel_callable(
    kernel: Callable[..., Mapping[str, Any]],
    *,
    path: str,
) -> None:
    """Validate a runtime update kernel uses the fixed worker signature."""
    try:
        signature = inspect.signature(kernel)
    except (TypeError, ValueError) as exc:
        raise WorkerContractValidationError(path, "kernel signature is not inspectable") from exc
    parameters = tuple(signature.parameters)
    if parameters != FIXED_UPDATE_KERNEL_SIGNATURE:
        raise WorkerContractValidationError(
            path,
            "kernel callable must have signature "
            f"{FIXED_UPDATE_KERNEL_SIGNATURE!r}; found {parameters!r}",
        )


def validate_per_trial_bindings(
    *,
    exposed_data: Sequence[Any],
    bindings: Sequence[Any],
    path: str = "/task_binding_spec",
) -> None:
    """Validate graph-facing per-trial bindings against the current role set.

    This check is intentionally port-agnostic. It only verifies that each
    binding resolves to a declared, role-bindable task-data producer and that
    optional dtype/shape metadata matches when both sides provide it.
    """
    data_by_id: dict[str, Any] = {}
    for index, data in enumerate(exposed_data):
        data_id = _field(data, "id")
        _require(
            isinstance(data_id, str) and bool(data_id),
            f"{path}/exposed_data/{index}/id",
            "missing data id",
        )
        data_by_id[data_id] = data

    for index, binding in enumerate(bindings):
        binding_path = f"{path}/bindings/{index}"
        source_id = _field(binding, "source_data_id")
        _require(
            isinstance(source_id, str) and source_id in data_by_id,
            f"{binding_path}/source_data_id",
            f"unknown source data id {source_id!r}",
        )
        data = data_by_id[source_id]
        role = (
            _field(binding, "role") or _field(data, "role") or _metadata(data).get("task_data_role")
        )
        _require(
            role in GRAPH_BINDABLE_TASK_DATA_ROLES,
            f"{binding_path}/role",
            f"role {role!r} is not graph-bindable; expected one of "
            f"{sorted(GRAPH_BINDABLE_TASK_DATA_ROLES)!r}",
        )
        _require(
            bool(_field(data, "bindable", False)),
            f"{binding_path}/source_data_id",
            f"source data {source_id!r} is not marked bindable",
        )
        data_dtype = _field(data, "dtype") or _metadata(data).get("dtype")
        target_dtype = _field(binding, "target_dtype") or _metadata(binding).get("target_dtype")
        if data_dtype is not None and target_dtype is not None:
            _require(
                data_dtype == target_dtype,
                f"{binding_path}/target_dtype",
                f"dtype mismatch: source={data_dtype!r}, target={target_dtype!r}",
            )
        data_shape = _field(data, "expected_shape") or _metadata(data).get("shape")
        target_shape = _field(binding, "target_shape") or _metadata(binding).get("target_shape")
        if data_shape is not None and target_shape is not None:
            _require(
                tuple(data_shape) == tuple(target_shape),
                f"{binding_path}/target_shape",
                f"shape mismatch: source={tuple(data_shape)!r}, target={tuple(target_shape)!r}",
            )


def _validate_reducers(
    contract: MethodContractSpec,
    axes: Mapping[str, Any],
    objective_requirements: Any | None = None,
) -> None:
    requirements: list[ReducerRequirement] = []
    requirements.extend(contract.objective_reducers)
    requirements.extend(contract.worker_reducers)
    if objective_requirements is not None:
        objective_axes = _field(objective_requirements, "requires_axes", ())
        objective_semantics = _field(objective_requirements, "aggregation_semantics", {})
        for axis in objective_axes:
            semantic = objective_semantics.get(axis, "none")
            if semantic != "none":
                requirements.append(
                    ReducerRequirement(
                        axis=axis,
                        owner="objective",
                        path=f"/objective_execution_requirements/aggregation_semantics/{axis}",
                    )
                )
    for index, axis in enumerate(contract.axes):
        if axis.reducer is not None:
            requirements.append(
                ReducerRequirement(
                    axis=axis.name,
                    owner=axis.reducer.owner,
                    path=axis.reducer.path or f"/axes/{index}/reducer",
                )
            )

    by_axis: dict[str, ReducerRequirement] = {}
    for requirement in requirements:
        _require(
            requirement.axis in axes,
            requirement.path,
            f"reducer references unknown axis {requirement.axis!r}",
        )
        existing = by_axis.get(requirement.axis)
        if existing is not None:
            raise WorkerContractValidationError(
                requirement.path,
                "axis has more than one reducer owner: "
                f"first={existing.owner!r} at {existing.path}, "
                f"second={requirement.owner!r}",
            )
        by_axis[requirement.axis] = requirement

    if objective_requirements is not None:
        objective_semantics = _field(objective_requirements, "aggregation_semantics", {})
        expected_semantics = _metadata(contract).get("required_objective_aggregation", {})
        if isinstance(expected_semantics, Mapping):
            for axis, expected in expected_semantics.items():
                if axis not in axes:
                    raise WorkerContractValidationError(
                        f"/metadata/required_objective_aggregation/{axis}",
                        f"required objective aggregation references unknown axis {axis!r}",
                    )
                actual = objective_semantics.get(axis)
                if actual != expected:
                    raise WorkerContractValidationError(
                        f"/objective_execution_requirements/aggregation_semantics/{axis}",
                        "objective aggregation semantic mismatch: "
                        f"expected {expected!r}, found {actual!r}",
                    )


def _validate_dry_run(
    contract: MethodContractSpec,
    *,
    environment: WorkerExecutabilityEnvironment,
    dry_run_shape_check: Callable[[MethodContractSpec], DryRunShapeCheckResult | bool | None]
    | None,
) -> None:
    declared_fields = {
        field
        for step in contract.phase_program.update_steps
        for field in (
            step.kernel.jit_static_payload_fields + step.kernel.resource_significant_payload_fields
        )
    }
    if dry_run_shape_check is None:
        _require(
            not environment.require_dry_run_check and not declared_fields,
            "/dry_run_shape_check",
            "dry-run shape check is required but no check was provided",
        )
        return

    raw = dry_run_shape_check(contract)
    if raw is None:
        result = DryRunShapeCheckResult(passed=True)
    elif isinstance(raw, bool):
        result = DryRunShapeCheckResult(passed=raw)
    else:
        result = raw
    _require(result.passed, "/dry_run_shape_check", result.message or "dry-run check failed")
    missing = sorted(declared_fields - set(result.exercised_payload_fields))
    _require(
        not missing,
        "/dry_run_shape_check/exercised_payload_fields",
        f"dry-run did not exercise resource-significant payload fields {missing!r}",
    )


def validate_worker_contract(
    contract: MethodContractSpec,
    *,
    environment: WorkerExecutabilityEnvironment | None = None,
    dry_run_shape_check: Callable[[MethodContractSpec], DryRunShapeCheckResult | bool | None]
    | None = None,
    update_kernels: Mapping[str, Callable[..., Mapping[str, Any]]] | None = None,
    task_binding_spec: Any | None = None,
    objective_requirements: Any | None = None,
) -> EffectivePhaseSpec:
    """Validate a method declaration for executability before launch."""
    env = environment or WorkerExecutabilityEnvironment()
    axes = _unique_by_name(contract.axes, "/axes")
    slots = _unique_by_name(contract.state_slots, "/state_slots")
    phases = _unique_by_name(contract.phase_program.phases, "/phase_program/phases")
    steps = _unique_by_name(contract.phase_program.update_steps, "/phase_program/update_steps")
    bindings = _unique_by_name(
        contract.phase_program.optimizer_bindings,
        "/phase_program/optimizer_bindings",
    )
    barriers = _unique_by_name(
        contract.phase_program.checkpoint_barriers,
        "/phase_program/checkpoint_barriers",
    )

    for index, axis in enumerate(contract.axes):
        _require(
            axis.role in env.supported_axis_roles,
            f"/axes/{index}/role",
            f"unsupported axis role {axis.role!r}",
        )
        if env.supported_axes is not None:
            _require(
                axis.name in env.supported_axes,
                f"/axes/{index}/name",
                f"unsupported axis {axis.name!r}",
            )
    for index, slot in enumerate(contract.state_slots):
        _require(
            slot.role in env.supported_state_slot_roles,
            f"/state_slots/{index}/role",
            f"unsupported state slot role {slot.role!r}",
        )
        if env.supported_state_slots is not None:
            _require(
                slot.name in env.supported_state_slots,
                f"/state_slots/{index}/name",
                f"unsupported state slot {slot.name!r}",
            )
        if slot.axis is not None:
            _require(
                slot.axis in axes,
                f"/state_slots/{index}/axis",
                f"unknown axis {slot.axis!r}",
            )

    _require(
        contract.phase_program.initial_phase in phases,
        "/phase_program/initial_phase",
        f"unknown initial phase {contract.phase_program.initial_phase!r}",
    )
    for phase_index, phase in enumerate(contract.phase_program.phases):
        phase_path = f"/phase_program/phases/{phase_index}"
        _require(
            phase.kind in env.supported_phase_kinds,
            f"{phase_path}/kind",
            f"unsupported phase kind {phase.kind!r}",
        )
        for field_name in ("reads", "writes", "initializes"):
            for slot in getattr(phase, field_name):
                _require(
                    slot in slots,
                    f"{phase_path}/{field_name}",
                    f"unknown state slot {slot!r}",
                )
        for step_name in phase.update_steps:
            _require(
                step_name in steps,
                f"{phase_path}/update_steps",
                f"unknown update step {step_name!r}",
            )
        for target in phase.legal_next:
            _require(
                target in phases,
                f"{phase_path}/legal_next",
                f"unknown next phase {target!r}",
            )
        if phase.loop_axis is not None:
            _require(
                phase.loop_axis in axes,
                f"{phase_path}/loop_axis",
                f"unknown loop axis {phase.loop_axis!r}",
            )
        if phase.checkpoint_barrier is not None:
            _require(
                phase.checkpoint_barrier in barriers,
                f"{phase_path}/checkpoint_barrier",
                f"unknown checkpoint barrier {phase.checkpoint_barrier!r}",
            )

    transition_pairs = set()
    for index, transition in enumerate(contract.phase_program.transitions):
        transition_path = f"/phase_program/transitions/{index}"
        _require(
            transition.source in phases,
            f"{transition_path}/source",
            f"unknown source phase {transition.source!r}",
        )
        _require(
            transition.target in phases,
            f"{transition_path}/target",
            f"unknown target phase {transition.target!r}",
        )
        source_phase = phases[transition.source]
        if source_phase.legal_next:
            _require(
                transition.target in source_phase.legal_next,
                f"{transition_path}/target",
                f"transition not listed in {transition.source!r}.legal_next",
            )
        if transition.barrier is not None:
            _require(
                transition.barrier in barriers,
                f"{transition_path}/barrier",
                f"unknown barrier {transition.barrier!r}",
            )
        if transition.guard is not None:
            _require(
                transition.barrier is not None,
                f"{transition_path}/guard",
                "metric guard must be evaluated at a checkpoint barrier",
            )
            for slot_name in transition.guard.metric_slots:
                _require(
                    slot_name in slots,
                    f"{transition_path}/guard/metric_slots",
                    f"unknown metric slot {slot_name!r}",
                )
                _require(
                    slots[slot_name].role == "metric",
                    f"{transition_path}/guard/metric_slots",
                    f"guard slot {slot_name!r} is not a metric slot",
                )
            for slot_name in transition.guard.bookkeeping_slots:
                _require(
                    slot_name in slots,
                    f"{transition_path}/guard/bookkeeping_slots",
                    f"unknown guard bookkeeping slot {slot_name!r}",
                )
                _require(
                    slots[slot_name].role == "auxiliary",
                    f"{transition_path}/guard/bookkeeping_slots",
                    f"guard bookkeeping slot {slot_name!r} is not an auxiliary slot",
                )
        transition_pairs.add((transition.source, transition.target))

    for phase_index, phase in enumerate(contract.phase_program.phases):
        for target in phase.legal_next:
            _require(
                (phase.name, target) in transition_pairs,
                f"/phase_program/phases/{phase_index}/legal_next",
                f"missing explicit transition {phase.name!r} -> {target!r}",
            )

    for step_index, step in enumerate(contract.phase_program.update_steps):
        step_path = f"/phase_program/update_steps/{step_index}"
        _require(
            step.kind in env.supported_update_step_kinds,
            f"{step_path}/kind",
            f"unsupported update step kind {step.kind!r}",
        )
        for slot in step.reads:
            _require(slot in slots, f"{step_path}/reads", f"unknown state slot {slot!r}")
        for slot in step.writes:
            _require(slot in slots, f"{step_path}/writes", f"unknown state slot {slot!r}")
        for axis in step.axes:
            _require(axis in axes, f"{step_path}/axes", f"unknown axis {axis!r}")
        if step.optimizer_binding is not None:
            _require(
                step.optimizer_binding in bindings,
                f"{step_path}/optimizer_binding",
                f"unknown optimizer binding {step.optimizer_binding!r}",
            )
        if step.kind == "measurement":
            _require(
                step.data_member is not None,
                f"{step_path}/data_member",
                "measurement step must declare a held-out/realization data member",
            )
            _require(
                step.optimizer_binding is None,
                f"{step_path}/optimizer_binding",
                "measurement step must not declare an optimizer binding",
            )
            _require(
                bool(step.writes),
                f"{step_path}/writes",
                "measurement step must write at least one metric slot",
            )
            for slot_name in step.writes:
                _require(
                    slots[slot_name].role == "metric",
                    f"{step_path}/writes",
                    f"measurement step may only write metric slots; {slot_name!r} "
                    f"has role {slots[slot_name].role!r}",
                )
        if step.kind == "control":
            _require(
                step.schedule_coordinate is not None,
                f"{step_path}/schedule_coordinate",
                "control step must declare the schedule coordinate it reads",
            )
            _require(
                step.optimizer_binding is None,
                f"{step_path}/optimizer_binding",
                "control step must not declare an optimizer binding",
            )
            _require(
                bool(step.reads),
                f"{step_path}/reads",
                "control step must read at least one metric slot",
            )
            for slot_name in step.reads:
                _require(
                    slots[slot_name].role == "metric",
                    f"{step_path}/reads",
                    f"control step may only read metric slots; {slot_name!r} "
                    f"has role {slots[slot_name].role!r}",
                )
            for slot_name in step.writes:
                _require(
                    slots[slot_name].role == "auxiliary",
                    f"{step_path}/writes",
                    f"control step may only write auxiliary slots; {slot_name!r} "
                    f"has role {slots[slot_name].role!r}",
                )
        if update_kernels is not None:
            kernel = update_kernels.get(step.kernel.kernel_ref)
            _require(
                kernel is not None,
                f"{step_path}/kernel/kernel_ref",
                f"missing callable for kernel_ref {step.kernel.kernel_ref!r}",
            )
            validate_update_kernel_callable(kernel, path=f"{step_path}/kernel")

    for binding_index, binding in enumerate(contract.phase_program.optimizer_bindings):
        binding_path = f"/phase_program/optimizer_bindings/{binding_index}"
        _require(
            binding.optimizer_slot in slots,
            f"{binding_path}/optimizer_slot",
            f"unknown optimizer slot {binding.optimizer_slot!r}",
        )
        _require(
            slots[binding.optimizer_slot].role == "optimizer",
            f"{binding_path}/optimizer_slot",
            f"slot {binding.optimizer_slot!r} is not an optimizer slot",
        )
        _require(
            binding.target_slot in slots,
            f"{binding_path}/target_slot",
            f"unknown target slot {binding.target_slot!r}",
        )
        for phase in binding.phase_scope:
            _require(
                phase in phases,
                f"{binding_path}/phase_scope",
                f"unknown phase {phase!r}",
            )
        for slot_name in binding.objective_reads:
            _require(
                slot_name in slots,
                f"{binding_path}/objective_reads",
                f"unknown objective read slot {slot_name!r}",
            )

    for barrier_index, barrier in enumerate(contract.phase_program.checkpoint_barriers):
        barrier_path = f"/phase_program/checkpoint_barriers/{barrier_index}"
        _require(
            barrier.phase in phases,
            f"{barrier_path}/phase",
            f"unknown phase {barrier.phase!r}",
        )
        for slot_index, checkpoint_slot in enumerate(barrier.slots):
            _require(
                checkpoint_slot.slot in slots,
                f"{barrier_path}/slots/{slot_index}/slot",
                f"unknown checkpoint slot {checkpoint_slot.slot!r}",
            )
            if checkpoint_slot.axis is not None:
                _require(
                    checkpoint_slot.axis in axes,
                    f"{barrier_path}/slots/{slot_index}/axis",
                    f"unknown axis {checkpoint_slot.axis!r}",
                )
        for sink_index, artifact_sink in enumerate(barrier.artifact_sinks):
            _require(
                artifact_sink.slot in slots,
                f"{barrier_path}/artifact_sinks/{sink_index}/slot",
                f"artifact sink slot must be a declared state slot; found {artifact_sink.slot!r}",
            )
        if barrier.resume_coordinate is not None:
            _require(
                barrier.resume_coordinate.phase in phases,
                f"{barrier_path}/resume_coordinate/phase",
                f"unknown resume phase {barrier.resume_coordinate.phase!r}",
            )
            if barrier.resume_coordinate.completed_barrier is not None:
                _require(
                    barrier.resume_coordinate.completed_barrier in barriers,
                    f"{barrier_path}/resume_coordinate/completed_barrier",
                    "resume coordinate references unknown barrier "
                    f"{barrier.resume_coordinate.completed_barrier!r}",
                )

    _validate_reducers(contract, axes, objective_requirements)
    _validate_dry_run(contract, environment=env, dry_run_shape_check=dry_run_shape_check)

    if task_binding_spec is not None:
        validate_per_trial_bindings(
            exposed_data=_field(task_binding_spec, "exposed_data", ()),
            bindings=_field(task_binding_spec, "bindings", ()),
        )

    predicate = derive_consistency_predicate(contract.phase_program)
    return EffectivePhaseSpec(
        method_ref=contract.method_ref,
        axes=contract.axes,
        state_slots=contract.state_slots,
        phase_program=contract.phase_program,
        consistency_predicate=predicate,
    )


def validate_checkpoint_population_payload(
    contract: MethodContractSpec,
    payload: Mapping[str, Any],
) -> None:
    """Validate population slot payload lengths against declared axis sizes."""
    axes = {axis.name: axis for axis in contract.axes}
    for slot_index, slot in enumerate(contract.state_slots):
        if slot.role != "population" or slot.axis is None:
            continue
        axis = axes.get(slot.axis)
        if axis is None or axis.size is None:
            continue
        _require(
            slot.name in payload,
            f"/checkpoint_payload/{slot.name}",
            f"missing population slot {slot.name!r}",
        )
        try:
            size = len(payload[slot.name])  # noqa: B905 - generic PyTree/list payload.
        except TypeError as exc:
            raise WorkerContractValidationError(
                f"/checkpoint_payload/{slot.name}",
                "population slot payload has no length",
            ) from exc
        _require(
            size == axis.size,
            f"/checkpoint_payload/{slot.name}",
            f"population length mismatch for axis {slot.axis!r}: expected {axis.size}, found {size}",
        )
        _require(slot_index >= 0, "/state_slots", "unreachable")


def validate_resume_checkpoint_pairing(
    manifest: CheckpointSlotManifest,
    *,
    required_slots: Sequence[str],
) -> None:
    """Reject cross-slot checkpoint pairing from different barriers or steps."""
    by_slot = {record.slot: record for record in manifest.slots}
    missing = sorted(set(required_slots) - set(by_slot))
    _require(
        not missing,
        "/checkpoint_slots",
        f"checkpoint manifest missing required slots {missing!r}",
    )
    records = [by_slot[slot] for slot in required_slots]
    barriers = {record.barrier for record in records}
    steps = {record.global_step for record in records}
    _require(
        len(barriers) == 1,
        "/checkpoint_slots",
        f"checkpoint slots cross barriers {sorted(barriers)!r}",
    )
    if len(steps) != 1:
        offenders = {
            record.slot: {"barrier": record.barrier, "global_step": record.global_step}
            for record in records
        }
        raise WorkerContractValidationError(
            "/checkpoint_slots",
            f"checkpoint slots cross global steps: {offenders!r}",
        )


def progress_coordinate_for_barrier(
    *,
    run_id: str,
    phase: str,
    global_step: int,
    completed_barrier: str | None,
) -> ProgressCoordinate:
    """Build a generic progress coordinate for checkpoint/resume tests."""
    return ProgressCoordinate(
        run_id=run_id,
        phase=phase,
        global_step=global_step,
        completed_barrier=completed_barrier,
    )
