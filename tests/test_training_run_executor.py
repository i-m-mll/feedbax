from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.tree as jt
import numpy as np
import pytest
import jax.numpy as jnp
import optax
from pydantic import BaseModel, ConfigDict

import feedbax.training.executor as training_executor
import feedbax.training.phase_executor as phase_executor
import feedbax.orchestration.events as orchestration_events
from feedbax.analysis.execution_context import (
    StagedCheckpointCustodyRootBinding,
    resolve_staged_execution_context,
)
from feedbax.contracts.manifest import TrainingRunManifest, load_manifest, sha256_bytes
from feedbax.contracts.checkpoints import (
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_ID,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION,
    BatchHistory,
    CheckpointContinuationRequest,
)
from feedbax.contracts.run_matrix import RowLowererIdentity, TrainingRowProvenance
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedCheckpointCustodySpec,
    StagedExecutionDescriptor,
)
from feedbax.contracts.spec_storage import (
    training_run_execution_hash,
    training_spec_canonical_bytes,
    training_spec_sha256,
)
from feedbax.contracts.training import (
    LrScheduleSpec,
    LossTermSpec,
    ObjectiveSlotSpec,
    OptimizerSpec,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    StandardSupervisedMethodPayload,
    TaskSpec,
    TrainingConfig,
    TrainingMethodRegistration,
    TrainingMethodMetadataProjector,
    TrainingMethodRegistry,
    TrainingManifestMetadataProjection,
    TrainingManifestMetadataProjectionRegistration,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_update_kernels,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_descriptor,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
    default_training_method_registry,
)
from feedbax.contracts.worker import (
    AxisCoordinateSpec,
    AxisSpec,
    BarrierArtifactSinkSpec,
    CheckpointSlotSpec,
    MethodTrainingDiagnosticsSpec,
    MappingLevelSpec,
    MetricGuardSpec,
    PhaseTransitionSpec,
    ProgressCoordinate,
    StateSlotSpec,
    SlotAxisBindingSpec,
)
from feedbax.orchestration.events import RunEventEmitter, RunEventReader
from feedbax.orchestration.conformance import (
    ConformanceRowArtifacts,
    check_checkpoint_cadence,
)
from feedbax.orchestration.bundle import (
    AuthoredIntentRef,
    ExecutionCapsuleRef,
    ExecutionIdentityEnvelope,
    ResolvedSnapshotRef,
    RowLaunchSpec,
    RunRowSpec,
    SchemaArtifactRef,
)
from feedbax.orchestration.drivers.native_execution import (
    NativeExecutionContextError,
    inject_native_execution_context,
)
from feedbax.training.checkpoint_custody import (
    CheckpointCompatibilityError,
    concatenate_checkpoint_histories,
    fork_checkpoint_transaction,
    load_latest_checkpoint,
    write_checkpoint_transaction,
)
from feedbax.training.executor import (
    DiagnosticsEmissionConflictError,
    ManifestEmissionConflictError,
    TrainingRunExecutorError,
    _feedbax_owned_training_manifest_metadata,
    _method_training_trace,
    _same_row_resume_start_batch,
    _preflight_manifest_emission,
    _realized_lr_trace,
    _synchronized_optimizer_step,
    execute_training_run_spec,
)
from feedbax.training.preparation import (
    ExecutionPreparationRequest,
    ExecutionPreparationPlan,
    ExecutionPreparationResult,
    MaterializedExecutionPreparation,
    _build_materialized_execution_preparation,
    _freeze_runtime_value,
    _identity_fingerprint,
    _run_spec_sha256,
    ScalarInstancePreparationResult,
    materialize_execution_preparation,
    validate_materialized_execution_preparation,
)
from feedbax.training.optimizers import build_optimizer
from feedbax.training.phase_executor import PhaseProgramExecutor
from feedbax.training.worker_validation import (
    WorkerContractValidationError,
    resolve_execution_mapping,
    validate_worker_contract,
)
from feedbax.training.manifest_preflight import (
    TrainingRunManifestPreflightError,
    build_training_run_manifest_spec_payloads,
    preflight_training_run_manifest_payloads,
)
from feedbax.training.interruption import CancellationDecision
from feedbax.training.diagnostics import (
    LearningRateDiagnostic,
    NativeExecutionProducerContext,
    NativeTrainingDiagnosticsInput,
    ScheduleContextDiagnostic,
    TrainingDiagnostics,
)


def _minimal_graph() -> dict[str, object]:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": ["input"],
        "output_ports": ["output"],
        "input_bindings": {"input": ("gain", "input")},
        "output_bindings": {"output": ("gain", "output")},
    }


def _run_spec() -> TrainingRunSpec:
    return TrainingRunSpec(
        graph={"inline": _minimal_graph()},
        task=TaskSpec(type="ToyTask", params={"n_steps": 1}),
        training_config=TrainingConfig(n_batches=1, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(
                type="target_state",
                label="target",
                selector="port:gain.output",
                target_value=[0.0],
            )
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
    )


def _scheduled_continuation_spec(
    *,
    origin: dict[str, object] | None = None,
    allow_inert: bool = False,
) -> TrainingRunSpec:
    spec = _mapped_run_spec()
    payload = StandardSupervisedMethodPayload(
        gradient_clip=0.125,
        optimizer=OptimizerSpec(
            type="adamw",
            lr_schedule=LrScheduleSpec(
                origin=origin or {"kind": "segment_start"},
                allow_inert=allow_inert,
                kind="warmup_cosine",
                learning_rate_0=3e-4,
                total_steps=3_500,
                constant_lr_iterations=1_000,
                warmup_init_fraction=0.1,
                cosine_annealing_alpha=0.1,
            ),
        )
    )
    return spec.model_copy(
        update={
            "method_payload": spec.method_payload.model_copy(
                update={"payload": payload.model_dump(mode="json")}
            ),
            "checkpoint_progress": spec.checkpoint_progress.model_copy(
                update={
                    "continuation": CheckpointContinuationRequest(
                        source_completed_batches=12_000,
                        additional_batches=1,
                    )
                }
            ),
        }
    )


def _mapped_run_spec() -> TrainingRunSpec:
    spec = _run_spec()
    worker = spec.worker_execution.model_copy(deep=True)
    contract = worker.method_contract
    contract.axes.append(AxisSpec(name="ensemble", role="replicate", size=5))
    for slot in contract.state_slots:
        slot.axis_bindings = [SlotAxisBindingSpec(axis="ensemble", mode="mapped", array_axis=0)]
    for step in contract.phase_program.update_steps:
        step.axes.append("ensemble")
    worker.mapping_levels = [MappingLevelSpec(axis="ensemble")]
    return spec.model_copy(update={"worker_execution": worker})


def _restore_optimizer_count(state, count: int):
    injected = state[-1]
    hyperparams_states = dict(injected.hyperparams_states)
    hyperparams_states["learning_rate"] = hyperparams_states["learning_rate"]._replace(
        count=jnp.asarray(count, dtype=jnp.int32)
    )
    return (*state[:-1], injected._replace(count=jnp.asarray(count), hyperparams_states=hyperparams_states))


def _write_mapped_schedule_checkpoint(spec: TrainingRunSpec, root: Path):
    payload = StandardSupervisedMethodPayload.model_validate(spec.method_payload.payload)
    optimizer = build_optimizer(
        payload.optimizer,
        schedule_origin_step=12_000,
        current_step=12_000,
        optimizer_count_at_current_step=12_000,
        gradient_clip=payload.gradient_clip,
    )
    restored = _restore_optimizer_count(optimizer.init(jnp.asarray(1.0)), 12_000)
    mapped_optimizer = jt.map(lambda leaf: jnp.stack([leaf] * 5), restored)
    slots = {
        "model": jnp.ones((5,)),
        "optimizer": mapped_optimizer,
        "prng": jnp.zeros((5, 2), dtype=jnp.uint32),
        "batch_counter": jnp.full((5,), 12_000, dtype=jnp.int32),
    }
    source_spec = spec.model_copy(
        update={"checkpoint_progress": spec.checkpoint_progress.model_copy(update={"continuation": None})}
    )
    write_checkpoint_transaction(
        root,
        run_spec=source_spec,
        phase_program=spec.worker_execution.effective_phase.phase_program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(run_id="source", phase="train_batch", program_step=12_000),
        slots=slots,
        completed_training_batches=12_000,
        metadata={"optimizer_step": 12_000},
    )
    return mapped_optimizer


@pytest.mark.parametrize(
    ("origin", "allow_inert", "inert"),
    [
        (None, False, False),
        ({"kind": "absolute", "batch": 0}, False, True),
        ({"kind": "absolute", "batch": 0}, True, False),
    ],
)
def test_restored_schedule_context_rebuilds_mapped_jit_optimizer(
    tmp_path: Path, origin: dict[str, object] | None, allow_inert: bool, inert: bool
) -> None:
    spec = _scheduled_continuation_spec(origin=origin, allow_inert=allow_inert)
    checkpoint_root = tmp_path / "checkpoint"
    payload = StandardSupervisedMethodPayload.model_validate(spec.method_payload.payload)

    def bind(_slots, origin, current, count):
        return {
            "optimizer": build_optimizer(
                payload.optimizer,
                schedule_origin_step=origin,
                current_step=current,
                optimizer_count_at_current_step=count,
                gradient_clip=payload.gradient_clip,
            )
        }

    @eqx.filter_jit
    def transition(model, opt_state, optimizer):
        updates, next_state = optimizer.update(jnp.full_like(model, 10.0), opt_state, model)
        learning_rate = optax.tree_utils.tree_get(
            next_state,
            "learning_rate",
            filtering=lambda path, _value: any(
                getattr(item, "name", None) == "hyperparams" for item in path
            ),
        )
        return optax.apply_updates(model, updates), next_state, learning_rate

    def kernel(slots, coordinate, context):
        del coordinate
        model, optimizer_state, learning_rate = transition(
            slots["model"], slots["optimizer"], context["optimizer"]
        )
        return {
            "model": model,
            "optimizer": optimizer_state,
            "prng": slots["prng"],
            "train_loss": learning_rate,
            "batch_counter": slots["batch_counter"] + 1,
        }

    def optimizer_step(_payload, runtime):
        return int(runtime["optimizer"][-1].count)

    worker = spec.worker_execution.model_copy(deep=True)
    worker.effective_phase = validate_worker_contract(
        worker.method_contract, update_kernels={"feedbax.training.standard_supervised.gradient_update": kernel}
    )
    spec = spec.model_copy(update={"worker_execution": worker})
    template_optimizer = _write_mapped_schedule_checkpoint(spec, checkpoint_root)
    descriptor = replace(
        standard_supervised_method_descriptor(),
        contract_compiler=lambda _payload: spec.worker_execution.method_contract,
        optimizer_step_extractor=optimizer_step,
        update_kernels_factory=lambda _payload: {"feedbax.training.standard_supervised.gradient_update": kernel},
    )
    registry = TrainingMethodRegistry()
    registry.register_descriptor(descriptor)
    prepared_slots = {
        "model": jnp.zeros((5,)),
        "optimizer": template_optimizer,
        "prng": jnp.ones((5, 2), dtype=jnp.uint32),
        "objective": jnp.zeros((5,)),
        "train_loss": jnp.zeros((5,)),
        "batch_counter": jnp.zeros((5,), dtype=jnp.int32),
    }
    preparation = _build_materialized_execution_preparation(
        request=ExecutionPreparationRequest(run_spec=spec),
        provider_identity="tests.restored_schedule",
        initial_slots=prepared_slots,
        kernel_context={},
        loss_service=None,
        resume_slot_transform=None,
        restored_schedule_context_binder=bind,
        coordinate_order=tuple((AxisCoordinateSpec(axis="ensemble", index=index),) for index in range(5)),
    )
    def execute():
        return execute_training_run_spec(
            spec, preparation=preparation, registry=registry,
            manifest_root=tmp_path / "manifests", checkpoint_root=checkpoint_root, resume=True
        )
    if inert:
        with pytest.raises(ValueError, match="absolute schedule window lies entirely before"):
            execute()
        return
    resumed = execute()
    _, bindings = resolve_execution_mapping(spec.worker_execution)
    assert payload.gradient_clip == 0.125
    assert _synchronized_optimizer_step(
        optimizer_step, payload, resumed.final_slots, bindings
    ) == 12_001
    if origin is None:
        assert jnp.asarray(resumed.final_slots["train_loss"]).view(jnp.uint32).tolist() == [0x37FBA890] * 5
    assert not jnp.array_equal(resumed.final_slots["model"], jnp.ones((5,)))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jt.leaves(resumed.final_slots) if isinstance(leaf, jax.Array))


def test_noncallable_restored_schedule_binder_fails_at_preparation() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        ExecutionPreparationResult({}, restored_schedule_context_binder=object())


def _mapped_diagnostics_run_spec() -> TrainingRunSpec:
    spec = _mapped_run_spec()
    worker = spec.worker_execution.model_copy(deep=True)
    worker.method_contract.training_diagnostics = MethodTrainingDiagnosticsSpec(
        trace_schema_id="feedbax.tests.method_trace",
        trace_schema_version="feedbax.tests.method_trace.v1",
        measurement_basis="kernel_input",
        metric_payload_slot="train_loss",
        replica_axis="ensemble",
    )
    return spec.model_copy(update={"worker_execution": worker})


def _mapped_shared_objective_run_spec(kernel) -> TrainingRunSpec:
    spec = _mapped_run_spec()
    worker = spec.worker_execution.model_copy(deep=True)
    contract = worker.method_contract
    for slot in contract.state_slots:
        if slot.name == "objective":
            slot.axis_bindings = [SlotAxisBindingSpec(axis="ensemble", mode="shared")]
    program = contract.phase_program
    program.phases[0].writes.remove("train_loss")
    program.phases[0].checkpoint_barrier = None
    program.update_steps[0].writes.remove("train_loss")
    program.checkpoint_barriers = []
    kernels = {"feedbax.training.standard_supervised.gradient_update": kernel}
    worker.effective_phase = validate_worker_contract(contract, update_kernels=kernels)
    return spec.model_copy(update={"worker_execution": worker})


def _mapped_test_registry(spec: TrainingRunSpec, kernel) -> TrainingMethodRegistry:
    registry = TrainingMethodRegistry()
    registry.register(
        TrainingMethodRegistration(
            method_ref=standard_supervised_method_ref().key,
            payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=StandardSupervisedMethodPayload,
            contract_factory=lambda: spec.worker_execution.method_contract,
            update_kernels_factory=lambda _payload: {
                "feedbax.training.standard_supervised.gradient_update": kernel
            },
        )
    )
    return registry


def _valid_materialized_preparation(
    spec: TrainingRunSpec,
    *,
    initial_batch: int = 0,
) -> MaterializedExecutionPreparation:
    slots = {
        slot.name: jnp.zeros((5,), dtype=jnp.float32)
        for slot in spec.worker_execution.method_contract.state_slots
    }
    if initial_batch:
        slots["batch_counter"] = jnp.full((5,), initial_batch, dtype=jnp.int32)
    coordinates = tuple((AxisCoordinateSpec(axis="ensemble", index=index),) for index in range(5))
    return _build_materialized_execution_preparation(
        request=ExecutionPreparationRequest(run_spec=spec),
        provider_identity="tests.mapped_preparation",
        initial_slots=slots,
        kernel_context={},
        loss_service=None,
        resume_slot_transform=None,
        coordinate_order=coordinates,
    )


def _mapped_runtime_resume_preparation(
    spec: TrainingRunSpec,
    **overrides: object,
) -> MaterializedExecutionPreparation:
    slots: dict[str, object] = {
        "model": jnp.zeros((5,), dtype=jnp.float32),
        "optimizer": jnp.zeros((5,), dtype=jnp.float32),
        "prng": jnp.zeros((5, 2), dtype=jnp.uint32),
        "objective": jnp.zeros((5,), dtype=jnp.float32),
        "train_loss": jnp.zeros((5,), dtype=jnp.float32),
        "batch_counter": jnp.zeros((5,), dtype=jnp.int32),
        "adaptive_state": jnp.zeros((5,), dtype=jnp.float32),
        "runtime_trace": jnp.zeros((5,), dtype=jnp.float32),
    }
    slots.update(overrides)
    coordinates = tuple((AxisCoordinateSpec(axis="ensemble", index=index),) for index in range(5))
    return _build_materialized_execution_preparation(
        request=ExecutionPreparationRequest(run_spec=spec),
        provider_identity="tests.mapped_runtime_resume",
        initial_slots=slots,
        kernel_context={},
        loss_service=None,
        resume_slot_transform=None,
        coordinate_order=coordinates,
    )


def _mapped_plan(materializer) -> ExecutionPreparationPlan:
    return ExecutionPreparationPlan(
        shared_slots={},
        kernel_context={},
        loss_service=None,
        resume_slot_transform=None,
        rng_roots={"model": jax.random.key(7)},
        materialize_instance=materializer,
    )


def _shared_objective_materializer(request):
    index = request.axis_coordinates[0].index
    return ScalarInstancePreparationResult(
        mapped_slots={
            "model": jnp.array(float(index)),
            "optimizer": jnp.array(0),
            "prng": request.rng.keys["model"],
            "batch_counter": jnp.array(0, dtype=jnp.int32),
        }
    )


def _shared_objective_kernel(slots, coordinate, context):
    del coordinate, context
    assert "objective" in slots
    return {
        "model": slots["model"] + 1,
        "optimizer": slots["optimizer"] + 1,
        "prng": slots["prng"],
        "batch_counter": slots["batch_counter"] + 1,
    }


def test_executor_injects_declared_shared_objective_before_mapped_preflight(
    tmp_path: Path,
) -> None:
    spec = _mapped_shared_objective_run_spec(_shared_objective_kernel)
    preparation = materialize_execution_preparation(
        ExecutionPreparationRequest(run_spec=spec),
        _mapped_plan(_shared_objective_materializer),
        provider_identity="tests.shared_objective",
    )
    assert "objective" not in preparation.initial_slots

    result = execute_training_run_spec(
        spec,
        preparation=preparation,
        registry=_mapped_test_registry(spec, _shared_objective_kernel),
        manifest_root=tmp_path / "manifest",
        checkpoint_root=tmp_path / "checkpoints",
        run_id="mapped-shared-objective",
    )

    assert result.final_slots["model"].tolist() == [1, 2, 3, 4, 5]
    assert "objective" in result.final_slots


def test_mapped_plan_still_rejects_missing_provider_owned_shared_slot() -> None:
    spec = _mapped_run_spec()
    for slot in spec.worker_execution.method_contract.state_slots:
        if slot.name == "objective":
            slot.axis_bindings = [SlotAxisBindingSpec(axis="ensemble", mode="shared")]
    spec.worker_execution.method_contract.state_slots.append(
        StateSlotSpec(
            name="shared_config",
            role="auxiliary",
            axis_bindings=[SlotAxisBindingSpec(axis="ensemble", mode="shared")],
        )
    )

    with pytest.raises(ValueError, match=r"missing_required=\['shared_config'\]"):
        materialize_execution_preparation(
            ExecutionPreparationRequest(run_spec=spec),
            _mapped_plan(_shared_objective_materializer),
            provider_identity="tests.missing_shared",
        )


def test_mapped_preflight_failure_has_no_publication_side_effects(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def unsafe_kernel(slots, coordinate, context):
        del coordinate, context
        return {"model": int(slots["model"])}

    spec = _mapped_shared_objective_run_spec(unsafe_kernel)
    preparation = materialize_execution_preparation(
        ExecutionPreparationRequest(run_spec=spec),
        _mapped_plan(_shared_objective_materializer),
        provider_identity="tests.preflight_failure",
    )
    emitter = RunEventEmitter(
        run_set_id="mapped-preflight",
        row_id="row",
        heartbeat_seconds=None,
    )
    events = []
    monkeypatch.setattr(emitter, "emit", lambda event_type, payload=None: events.append(event_type))
    manifest_root = tmp_path / "manifest"
    checkpoint_root = tmp_path / "checkpoints"

    with pytest.raises(TrainingRunExecutorError, match="mapped execution preflight failed"):
        execute_training_run_spec(
            spec,
            preparation=preparation,
            registry=_mapped_test_registry(spec, unsafe_kernel),
            manifest_root=manifest_root,
            checkpoint_root=checkpoint_root,
            run_id="mapped-preflight-failure",
            run_event_emitter=emitter,
        )

    assert events == []
    assert not manifest_root.exists()
    assert not checkpoint_root.exists()


def test_mapped_plan_materializes_ordered_distinct_instances_and_stacks_pytrees(
    tmp_path: Path,
) -> None:
    spec = _mapped_run_spec()
    requests = []
    optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3)

    def materialize(request):
        requests.append(request)
        key = request.rng.keys["model"]
        linear = eqx.nn.Linear(1, 1, key=key)
        return ScalarInstancePreparationResult(
            mapped_slots={
                "model": linear,
                "optimizer": optimizer.init(linear),
                "prng": jax.random.key_data(key),
                "objective": jnp.array(0.0),
                "train_loss": np.array(0.0, dtype=np.float32),
                "batch_counter": jnp.array(0, dtype=jnp.int32),
            }
        )

    prepared = materialize_execution_preparation(
        ExecutionPreparationRequest(run_spec=spec),
        _mapped_plan(materialize),
        provider_identity="tests.mapped_plan",
    )

    assert [item.axis_coordinates[0].index for item in requests] == list(range(5))
    assert (
        len({tuple(map(int, jax.random.key_data(item.rng.keys["model"]))) for item in requests})
        == 5
    )
    assert prepared.initial_slots["model"].weight.shape[0] == 5
    assert not prepared.initial_slots["train_loss"].flags.writeable
    assert any(
        getattr(leaf, "shape", None) == (5,)
        for leaf in jax.tree.leaves(prepared.initial_slots["optimizer"])
    )
    assert isinstance(
        prepared.initial_slots["optimizer"],
        optax.schedules.InjectStatefulHyperparamsState,
    )
    validate_materialized_execution_preparation(prepared, run_spec=spec)
    instance_model = jax.tree.map(lambda leaf: leaf[0], prepared.initial_slots["model"])
    instance_state = jax.tree.map(lambda leaf: leaf[0], prepared.initial_slots["optimizer"])
    updates, updated_state = optimizer.update(
        jax.tree.map(jnp.ones_like, instance_model),
        instance_state,
        instance_model,
    )
    assert jax.tree.leaves(updates)
    assert updated_state.count == 1

    def gradient_update(slots, coordinate, context):
        del coordinate, context
        gradients = jax.tree.map(jnp.ones_like, slots["model"])
        model_updates, optimizer_state = optimizer.update(
            gradients,
            slots["optimizer"],
            slots["model"],
        )
        return {
            "model": optax.apply_updates(slots["model"], model_updates),
            "optimizer": optimizer_state,
            "prng": slots["prng"],
            "train_loss": jnp.asarray(0.0),
            "batch_counter": slots["batch_counter"] + 1,
        }

    result = execute_training_run_spec(
        spec,
        preparation=prepared,
        registry=_mapped_test_registry(spec, gradient_update),
        manifest_root=tmp_path / "manifest",
        checkpoint_root=tmp_path / "checkpoint",
        run_id="mapped-injected-adamw",
    )
    assert result.final_slots["optimizer"].count.tolist() == [1] * 5
    assert type(result.final_slots["optimizer"].hyperparams) is dict
    loaded = load_latest_checkpoint(
        tmp_path / "checkpoint",
        expected_run_spec=spec,
        expected_phase_program=spec.worker_execution.method_contract.phase_program,
        expected_slots=prepared.initial_slots,
    )
    assert jax.tree.structure(loaded.slots["optimizer"]) == jax.tree.structure(
        result.final_slots["optimizer"]
    )
    forked = fork_checkpoint_transaction(
        tmp_path / "checkpoint",
        tmp_path / "fork",
        target_run_spec=spec,
        expected_slots=prepared.initial_slots,
        source_slot_transforms={"optimizer": lambda slots: dict(slots)},
        source_transform_metadata={"optimizer": {"identity": "tests.identity", "parameters": {}}},
    )
    assert forked.slot_transfer_modes["optimizer"] == "serialized"
    reloaded = load_latest_checkpoint(
        forked.root,
        expected_run_spec=spec,
        expected_phase_program=spec.worker_execution.method_contract.phase_program,
        expected_slots=prepared.initial_slots,
    )
    assert jax.tree.structure(reloaded.slots["optimizer"]) == jax.tree.structure(
        result.final_slots["optimizer"]
    )


def test_runtime_freeze_preserves_optax_tuple_pytree_types() -> None:
    parameters = {"weight": jnp.array(1.0)}
    transformation = optax.adam(1e-3)
    injected = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3)

    frozen_transformation = _freeze_runtime_value(transformation)
    frozen_state = _freeze_runtime_value(injected.init(parameters))

    assert type(frozen_transformation) is type(transformation)
    assert callable(frozen_transformation.update)
    assert isinstance(frozen_state, optax.schedules.InjectStatefulHyperparamsState)
    assert frozen_state.hyperparams["learning_rate"] == pytest.approx(1e-3)
    with pytest.raises(TypeError):
        frozen_state.hyperparams["learning_rate"] = 2e-3
    updates, updated_state = frozen_transformation.update(
        {"weight": jnp.array(1.0)},
        frozen_transformation.init(parameters),
        parameters,
    )
    assert updates["weight"] < 0
    assert jax.tree.leaves(updated_state)

    source_array = np.array([1.0])
    ordinary = _freeze_runtime_value((["item"], {"array": source_array}, 3))
    source_array[0] = 9.0
    assert type(ordinary) is tuple
    assert ordinary[0] == ("item",)
    assert ordinary[1]["array"].tolist() == [1.0]
    assert not ordinary[1]["array"].flags.writeable
    assert ordinary[2] == 3
    with pytest.raises(ValueError, match="object-backed NumPy"):
        _freeze_runtime_value(np.array([object()], dtype=object))


def test_mapped_plan_rejects_batch_history_immediately() -> None:
    calls = []

    def materialize(request):
        calls.append(request)
        return ScalarInstancePreparationResult(
            mapped_slots={
                "model": BatchHistory(jnp.zeros((1,))),
                **{
                    name: jnp.array(0)
                    for name in {"optimizer", "prng", "objective", "train_loss", "batch_counter"}
                },
            }
        )

    with pytest.raises(ValueError, match="contains BatchHistory"):
        materialize_execution_preparation(
            ExecutionPreparationRequest(run_spec=_mapped_run_spec()),
            _mapped_plan(materialize),
            provider_identity="tests.mapped_history",
        )
    assert len(calls) == 1


def test_mapped_plan_rejects_optional_presence_and_leaf_drift() -> None:
    spec = _mapped_run_spec()
    slots = spec.worker_execution.method_contract.state_slots
    slots[-2].required = False
    calls = 0

    def materialize(_request):
        nonlocal calls
        calls += 1
        values = {name: jnp.zeros((2,)) for name in {slot.name for slot in slots}}
        if calls == 2:
            values.pop("train_loss")
        return ScalarInstancePreparationResult(mapped_slots=values)

    with pytest.raises(ValueError, match="optional mapped slots"):
        materialize_execution_preparation(
            ExecutionPreparationRequest(run_spec=spec),
            _mapped_plan(materialize),
            provider_identity="tests.optional_drift",
        )


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("shape", "changes shape or dtype"),
        ("type", "changes array/static type"),
        ("tree", "divergent PyTree"),
        ("static", "divergent static values"),
    ],
)
def test_mapped_plan_rejects_tree_shape_type_and_static_drift(drift, message) -> None:
    calls = 0

    def materialize(_request):
        nonlocal calls
        calls += 1
        model = (jnp.zeros((2,)), "same") if drift == "static" else jnp.zeros((2,))
        if calls == 2:
            if drift == "shape":
                model = jnp.zeros((3,))
            elif drift == "type":
                model = 1
            elif drift == "tree":
                model = {"array": jnp.zeros((2,))}
            else:
                model = (jnp.zeros((2,)), "different")
        return ScalarInstancePreparationResult(
            mapped_slots={
                name: model if name == "model" else jnp.array(0)
                for name in {
                    "model",
                    "optimizer",
                    "prng",
                    "objective",
                    "train_loss",
                    "batch_counter",
                }
            }
        )

    with pytest.raises(ValueError, match=message):
        materialize_execution_preparation(
            ExecutionPreparationRequest(run_spec=_mapped_run_spec()),
            _mapped_plan(materialize),
            provider_identity="tests.tree_drift",
        )
    assert calls == 2


def _mapped_phase_executor(kernel, spec: TrainingRunSpec | None = None):
    spec = spec or _mapped_run_spec()
    levels, bindings = resolve_execution_mapping(spec.worker_execution)
    return PhaseProgramExecutor(
        spec.worker_execution.method_contract.phase_program,
        {"feedbax.training.standard_supervised.gradient_update": kernel},
        state_slots=spec.worker_execution.method_contract.state_slots,
        mapping_levels=levels,
        slot_axis_bindings=bindings,
    ), bindings


def _mapped_scalar_kernel(slots, coordinate, context):
    del coordinate, context
    return {
        "model": slots["model"] + 1,
        "optimizer": slots["optimizer"] + 1,
        "prng": slots["prng"],
        "train_loss": slots["model"] * 2,
        "batch_counter": slots["batch_counter"] + 1,
    }


def _mapped_slots() -> dict[str, object]:
    return {
        "model": jnp.arange(5.0),
        "optimizer": jnp.zeros((5,)),
        "prng": jnp.zeros((5, 2), dtype=jnp.uint32),
        "objective": jnp.zeros((5,)),
        "train_loss": jnp.zeros((5,)),
        "batch_counter": jnp.zeros((5,), dtype=jnp.int32),
    }


def test_mapped_executor_preflight_and_execution_match_manual_scalar_loop() -> None:
    executor, _ = _mapped_phase_executor(_mapped_scalar_kernel)
    slots = _mapped_slots()
    executor.preflight(slots, run_id="mapped", context={})
    result = executor.run(slots, run_id="mapped")

    assert result.slots["model"].tolist() == [1, 2, 3, 4, 5]
    assert result.slots["train_loss"].tolist() == [0, 2, 4, 6, 8]
    assert result.slots["batch_counter"].tolist() == [1] * 5


def test_mapped_executor_constructs_wrapper_once_across_changing_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _mapped_run_spec()
    spec.worker_execution.method_contract.phase_program.phases[0].max_steps = 3

    def coordinate_kernel(slots, coordinate, context):
        del context
        increment = coordinate.program_step + 1
        updates = _mapped_scalar_kernel(slots, coordinate, {})
        updates["model"] = slots["model"] + increment
        return updates

    executor, _ = _mapped_phase_executor(coordinate_kernel, spec)
    constructed = 0
    mapped_kernel = executor._mapped_kernel

    def counted_mapped_kernel(kernel):
        nonlocal constructed
        constructed += 1
        return mapped_kernel(kernel)

    monkeypatch.setattr(executor, "_mapped_kernel", counted_mapped_kernel)
    result = executor.run(_mapped_slots(), run_id="mapped-dynamic-coordinate")

    assert constructed == 1
    assert result.slots["model"].tolist() == [6, 7, 8, 9, 10]
    assert result.slots["batch_counter"].tolist() == [3] * 5


def test_mapped_batch_authority_uses_one_transfer_and_rejects_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _ = _mapped_phase_executor(_mapped_scalar_kernel)
    slots = _mapped_slots()
    transfers = 0
    device_get = jax.device_get

    def counted_device_get(value):
        nonlocal transfers
        transfers += 1
        return device_get(value)

    monkeypatch.setattr(phase_executor.jax, "device_get", counted_device_get)
    completed = executor._completed_training_batches(
        slots, phase_executor.ProgressCoordinate(run_id="mapped", phase="train_batch")
    )
    assert completed == 0
    assert transfers == 1

    transfers = 0
    slots["batch_counter"] = jnp.arange(5, dtype=jnp.int32)
    with pytest.raises(WorkerContractValidationError, match="authorities diverged"):
        executor._completed_training_batches(
            slots, phase_executor.ProgressCoordinate(run_id="mapped", phase="train_batch")
        )
    assert transfers == 1


@pytest.mark.parametrize("authority", [2**100, pytest.param(None, id="supports-index")])
def test_mapped_batch_authority_preserves_host_integer_semantics(authority) -> None:
    class HostBatchAuthority:
        def __index__(self) -> int:
            return 7

    expected = 7 if authority is None else authority
    authority = HostBatchAuthority() if authority is None else authority
    executor, _ = _mapped_phase_executor(_mapped_scalar_kernel)
    slots = _mapped_slots()
    slots["batch_counter"] = authority

    completed = executor._completed_training_batches(
        slots, phase_executor.ProgressCoordinate(run_id="mapped", phase="train_batch")
    )

    assert completed == expected


def test_nan_metric_names_uses_one_transfer_for_nested_mapped_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfers = 0
    device_get = jax.device_get

    def counted_device_get(value):
        nonlocal transfers
        transfers += 1
        return device_get(value)

    monkeypatch.setattr(training_executor.jax, "device_get", counted_device_get)
    observed = training_executor._nan_metric_names(
        {
            "healthy": {"replicas": jnp.arange(5.0), "nested": (jnp.inf, None)},
            "replica_edge": {
                "replicas": jnp.array([0.0, 1.0, 2.0, 3.0, jnp.nan]),
                "label": "ignored",
            },
        }
    )
    assert observed == ["replica_edge"]
    assert transfers == 1


def test_native_mapped_nan_guard_fails_closed_after_batched_health_reduction(
    tmp_path: Path,
) -> None:
    spec = _mapped_run_spec()

    def nan_kernel(slots, coordinate, context):
        updates = _mapped_scalar_kernel(slots, coordinate, context)
        updates["train_loss"] = jnp.full_like(slots["train_loss"], jnp.nan)
        return updates

    with pytest.raises(FloatingPointError, match="train_loss"):
        execute_training_run_spec(
            spec,
            preparation=_valid_materialized_preparation(spec),
            registry=_mapped_test_registry(spec, nan_kernel),
            manifest_root=tmp_path / "manifest",
            checkpoint_root=tmp_path / "checkpoints",
            run_id="mapped-nan-health",
        )


def test_native_mapped_resume_retains_prepared_runtime_slots_and_checkpoint_authority(
    tmp_path: Path,
) -> None:
    spec = _mapped_run_spec()
    worker = spec.worker_execution.model_copy(deep=True)
    contract = worker.method_contract
    mapped_binding = [SlotAxisBindingSpec(axis="ensemble", mode="mapped", array_axis=0)]
    contract.state_slots.extend(
        [
            StateSlotSpec(
                name="adaptive_state",
                role="auxiliary",
                axis_bindings=mapped_binding,
            ),
            StateSlotSpec(
                name="runtime_trace",
                role="metric",
                axis_bindings=mapped_binding,
            ),
        ]
    )
    phase = contract.phase_program.phases[0]
    phase.reads.extend(["adaptive_state", "runtime_trace"])
    phase.writes.extend(["adaptive_state", "runtime_trace"])
    step = contract.phase_program.update_steps[0]
    step.reads.extend(["adaptive_state", "runtime_trace"])
    step.writes.extend(["adaptive_state", "runtime_trace"])
    contract.phase_program.checkpoint_barriers[0].slots.append(
        CheckpointSlotSpec(slot="adaptive_state")
    )

    @jax.jit
    def jitted_values(model, optimizer, batch_counter, adaptive_state, runtime_trace):
        return (
            model + 1,
            optimizer + 1,
            model * 2,
            batch_counter + 1,
            adaptive_state + 1,
            runtime_trace + 1,
        )

    def update(slots, coordinate, context):
        del coordinate, context
        model, optimizer, train_loss, batch_counter, adaptive_state, runtime_trace = (
            jitted_values(
                slots["model"],
                slots["optimizer"],
                slots["batch_counter"],
                slots["adaptive_state"],
                slots["runtime_trace"],
            )
        )
        return {
            "model": model,
            "optimizer": optimizer,
            "prng": slots["prng"],
            "train_loss": train_loss,
            "batch_counter": batch_counter,
            "adaptive_state": adaptive_state,
            "runtime_trace": runtime_trace,
        }

    kernels = {"feedbax.training.standard_supervised.gradient_update": update}
    worker.effective_phase = validate_worker_contract(contract, update_kernels=kernels)
    spec = spec.model_copy(update={"worker_execution": worker})
    registry = _mapped_test_registry(spec, update)
    checkpoint_root = tmp_path / "checkpoint"
    source = execute_training_run_spec(
        spec,
        preparation=_mapped_runtime_resume_preparation(
            spec,
            model=jnp.full((5,), 2.0),
            optimizer=jnp.full((5,), 4.0),
            adaptive_state=jnp.full((5,), 6.0),
            runtime_trace=jnp.full((5,), 20.0),
        ),
        registry=registry,
        manifest_root=tmp_path / "source-manifest",
        checkpoint_root=checkpoint_root,
        run_id="mapped-runtime-source",
        stop_after_barrier="after_train_batch",
    )
    persisted_slots = {slot.slot for slot in source.checkpoint_writes[0].manifest.slots}
    assert persisted_slots == {
        "model",
        "optimizer",
        "prng",
        "adaptive_state",
        "batch_counter",
    }
    assert "runtime_trace" not in persisted_slots

    resumed = execute_training_run_spec(
        spec,
        preparation=_mapped_runtime_resume_preparation(
            spec,
            model=jnp.full((5,), 100.0),
            optimizer=jnp.full((5,), 100.0),
            prng=jnp.ones((5, 2), dtype=jnp.uint32),
            adaptive_state=jnp.full((5,), 100.0),
            batch_counter=jnp.zeros((5,), dtype=jnp.int32),
            runtime_trace=jnp.full((5,), 40.0),
        ),
        registry=registry,
        manifest_root=tmp_path / "resume-manifest",
        checkpoint_root=checkpoint_root,
        run_id="mapped-runtime-resumed",
        resume=True,
    )

    assert resumed.final_slots["model"].tolist() == [4.0] * 5
    assert resumed.final_slots["optimizer"].tolist() == [6.0] * 5
    assert resumed.final_slots["prng"].tolist() == [[0, 0]] * 5
    assert resumed.final_slots["adaptive_state"].tolist() == [8.0] * 5
    assert resumed.final_slots["batch_counter"].tolist() == [2] * 5
    assert resumed.final_slots["runtime_trace"].tolist() == [41.0] * 5


def test_mapped_execution_retains_metrics_and_checkpoint_axes_without_coordinate_arrays(
    tmp_path: Path,
) -> None:
    spec = _mapped_run_spec()
    emitter = RunEventEmitter(
        run_set_id="mapped",
        row_id="row",
        path=tmp_path / "events.jsonl",
        heartbeat_seconds=None,
    )
    try:
        result = execute_training_run_spec(
            spec,
            preparation=_valid_materialized_preparation(spec),
            registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
            manifest_root=tmp_path / "manifest",
            checkpoint_root=tmp_path / "checkpoint",
            run_id="mapped-evidence",
            run_event_emitter=emitter,
        )
    finally:
        emitter.close()

    history = result.history_events[0]
    assert "completed_batches" not in history["coordinate"]
    assert "train_loss" not in history["coordinate"]["metrics"]
    assert history["metrics"]["train_loss"]["value"] == [0.0] * 5
    assert result.manifest.summary_metrics["train_loss"]["axes"][0]["axis"] == "ensemble"
    checkpoint = result.checkpoint_writes[0].manifest
    assert checkpoint.completed_coordinate.completed_batches is None
    assert result.final_coordinate.completed_batches is None
    assert "train_loss" not in checkpoint.completed_coordinate.metrics
    assert all(slot.materialized_axes for slot in checkpoint.slots)
    events = RunEventReader(tmp_path / "events.jsonl").read_all()
    progress = next(event for event in events if event.type == "progress")
    assert "loss" not in progress.payload
    assert "train_loss" not in progress.payload["coordinate"]["metrics"]
    terminal = next(event for event in events if event.type == "complete")
    terminal_loss = terminal.payload["metrics"]["train_loss"]
    assert terminal_loss["value"] == [0.0] * 5
    assert terminal_loss["shape"] == [5]
    assert terminal_loss["dtype"] == "float32"
    assert [axis["axis"] for axis in terminal_loss["axes"]] == ["ensemble"]
    assert not isinstance(terminal_loss["value"], dict)

    loaded = load_latest_checkpoint(
        tmp_path / "checkpoint",
        expected_run_spec=spec,
        expected_phase_program=spec.worker_execution.method_contract.phase_program,
        expected_slots=result.final_slots,
    )
    assert loaded.slots["model"].shape == (5,)
    with pytest.raises(
        CheckpointCompatibilityError,
        match="structural ABI mismatch|does not materialize axis",
    ):
        load_latest_checkpoint(
            tmp_path / "checkpoint",
            expected_run_spec=spec,
            expected_phase_program=spec.worker_execution.method_contract.phase_program,
            expected_slots=result.final_slots,
            resume_slot_transform=lambda slots: {**slots, "model": slots["model"][:4]},
        )

    forked = fork_checkpoint_transaction(
        tmp_path / "checkpoint",
        tmp_path / "fork",
        target_run_spec=spec,
        expected_slots=result.final_slots,
    )
    assert all(
        slot.source_axes == slot.target_axes
        for slot in forked.manifest.fork_provenance.slots
    )

    def copy_blob(source, target):
        shutil.copy2(source, target)
        return "copy"

    copied = fork_checkpoint_transaction(
        tmp_path / "checkpoint",
        tmp_path / "fork-copy",
        target_run_spec=spec,
        expected_slots=result.final_slots,
        link_strategy=copy_blob,
    )
    assert set(copied.slot_transfer_modes.values()) == {"copy"}
    load_latest_checkpoint(
        copied.root,
        expected_run_spec=spec,
        expected_phase_program=spec.worker_execution.method_contract.phase_program,
        expected_slots=result.final_slots,
    )

    serialized = fork_checkpoint_transaction(
        tmp_path / "checkpoint",
        tmp_path / "fork-serialized",
        target_run_spec=spec,
        expected_slots=result.final_slots,
        source_slot_transforms={"model": lambda slots: dict(slots)},
        source_transform_metadata={
            "model": {"identity": "tests.identity", "parameters": {}}
        },
    )
    assert serialized.slot_transfer_modes["model"] == "serialized"
    model_provenance = next(
        item for item in serialized.manifest.fork_provenance.slots if item.slot == "model"
    )
    assert model_provenance.source_axes == model_provenance.target_axes
    load_latest_checkpoint(
        serialized.root,
        expected_run_spec=spec,
        expected_phase_program=spec.worker_execution.method_contract.phase_program,
        expected_slots=result.final_slots,
    )
    mismatched = spec.model_copy(deep=True)
    mismatched.worker_execution.method_contract.axes[-1].size = 4
    mismatched_slots = {
        name: value[:4] if hasattr(value, "shape") else value
        for name, value in result.final_slots.items()
    }
    with pytest.raises(CheckpointCompatibilityError, match="mapped-axis evidence mismatch"):
        load_latest_checkpoint(
            tmp_path / "checkpoint",
            expected_run_spec=mismatched,
            expected_phase_program=mismatched.worker_execution.method_contract.phase_program,
            expected_slots=mismatched_slots,
            allow_new_lineage_override=True,
        )
    with pytest.raises(CheckpointCompatibilityError, match="mapped axes differ"):
        fork_checkpoint_transaction(
            tmp_path / "checkpoint",
            tmp_path / "fork-mismatch",
            target_run_spec=mismatched,
            expected_slots=mismatched_slots,
        )

    add_state = spec.model_copy(deep=True)
    add_state.worker_execution.method_contract.state_slots.append(
        StateSlotSpec(
            name="adaptive_state",
            role="auxiliary",
            axis_bindings=[
                SlotAxisBindingSpec(axis="ensemble", mode="mapped", array_axis=0)
            ],
        )
    )
    add_state.worker_execution.method_contract.phase_program.checkpoint_barriers[0].slots.append(
        CheckpointSlotSpec(slot="adaptive_state")
    )
    adaptive = jnp.zeros((5,))
    added = fork_checkpoint_transaction(
        tmp_path / "checkpoint",
        tmp_path / "fork-add-state",
        target_run_spec=add_state,
        expected_slots={**result.final_slots, "adaptive_state": adaptive},
        target_slot_transform=lambda slots: {**slots, "adaptive_state": adaptive},
        target_transform_metadata={"identity": "tests.add_adaptive_state", "parameters": {}},
        target_transformed_slots=(),
        target_only_slots={"adaptive_state": {"reason": "sanctioned add-state"}},
    )
    adaptive_provenance = next(
        slot
        for slot in added.manifest.fork_provenance.slots
        if slot.slot == "adaptive_state"
    )
    assert adaptive_provenance.source_axes is None
    assert adaptive_provenance.target_axes[0].axis == "ensemble"

    extra = fork_checkpoint_transaction(
        tmp_path / "checkpoint",
        tmp_path / "fork-extra-corruption",
        target_run_spec=spec,
        expected_slots=result.final_slots,
    )
    corruptions = (
        (forked, lambda records: records.append(dict(records[0]))),
        (copied, lambda records: records[0].update({"source_relative_path": None})),
        (serialized, lambda records: records.pop()),
        (
            added,
            lambda records: next(
                item for item in records if item["slot"] == "adaptive_state"
            ).update(
                {
                    "source_axes": next(
                        item for item in records if item["slot"] == "adaptive_state"
                    )["target_axes"]
                }
            ),
        ),
        (
            extra,
            lambda records: records.append(
                {**records[0], "slot": "undeclared_extra_slot"}
            ),
        ),
    )
    for fork_result, corrupt in corruptions:
        payload = json.loads(fork_result.manifest_path.read_text())
        corrupt(payload["fork_provenance"]["slots"])
        fork_result.manifest_path.write_text(json.dumps(payload, sort_keys=True, indent=2))
        latest = json.loads(fork_result.latest_pointer_path.read_text())
        latest["manifest_sha256"] = sha256_bytes(fork_result.manifest_path.read_bytes())
        fork_result.latest_pointer_path.write_text(json.dumps(latest, sort_keys=True, indent=2))
        with pytest.raises(
            CheckpointCompatibilityError, match="fork provenance|target-only fork"
        ):
            load_latest_checkpoint(
                fork_result.root,
                expected_run_spec=(add_state if fork_result is added else spec),
                expected_phase_program=(
                    add_state.worker_execution.method_contract.phase_program
                    if fork_result is added
                    else spec.worker_execution.method_contract.phase_program
                ),
                expected_slots=(
                    {**result.final_slots, "adaptive_state": adaptive}
                    if fork_result is added
                    else result.final_slots
                ),
            )


@pytest.mark.parametrize(
    ("initial_batch", "expected_batches", "expected_steps"),
    [
        pytest.param(0, [3, 4], [3, 4], id="fresh"),
        pytest.param(10, [12, 14], [2, 4], id="continuation"),
    ],
)
def test_method_diagnostics_progress_observes_completed_batch_without_custody_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_batch: int,
    expected_batches: list[int],
    expected_steps: list[int],
) -> None:
    spec = _mapped_diagnostics_run_spec()
    spec.worker_execution.method_contract.phase_program.phases[0].max_steps = 4
    spec.training_config.n_batches = 4
    spec.checkpoint_progress.checkpoint_interval = 3
    spec.checkpoint_progress.progress_interval = 3
    callback_events: list[dict[str, object]] = []
    method_observations: list[dict[str, object]] = []
    observation_origins: list[int] = []
    build_diagnostics = training_executor._build_training_diagnostics

    def capture_method_observations(**kwargs):
        method_observations.extend(kwargs["method_observations"])
        observation_origins.append(kwargs["method_observation_origin_batch"])
        return build_diagnostics(**kwargs)

    monkeypatch.setattr(
        training_executor,
        "_build_training_diagnostics",
        capture_method_observations,
    )

    result = execute_training_run_spec(
        spec,
        preparation=_valid_materialized_preparation(spec, initial_batch=initial_batch),
        registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
        manifest_root=tmp_path / "manifest",
        checkpoint_root=tmp_path / "checkpoint",
        run_id="mapped-diagnostics-observation",
        progress_callback=callback_events.append,
    )

    assert [
        event["coordinate"]["completed_batches"] for event in result.history_events
    ] == expected_batches
    assert [
        event["coordinate"]["completed_batches"] for event in callback_events
    ] == expected_batches
    assert [
        event["coordinate"]["program_step"] for event in result.history_events
    ] == expected_steps
    assert [
        event["coordinate"]["completed_batches"] for event in method_observations
    ] == list(range(initial_batch + 1, initial_batch + 5))
    assert [event["coordinate"]["program_step"] for event in method_observations] == [
        1,
        2,
        3,
        4,
    ]
    assert observation_origins == [initial_batch]
    assert all(
        write.manifest.completed_coordinate.completed_batches is None
        for write in result.checkpoint_writes
    )
    assert result.final_coordinate.completed_batches is None


def test_method_observation_buffer_batches_host_materialization_and_preserves_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _mapped_diagnostics_run_spec()
    bindings = _valid_materialized_preparation(spec).slot_axis_bindings
    observations: list[dict[str, object]] = []
    buffer = training_executor._MethodObservationBuffer(
        observations,
        bindings,
        capacity=3,
    )
    coordinates = [
        phase_executor.ProgressCoordinate(
            run_id="batched-observations",
            phase="train_batch",
            program_step=batch,
            completed_batches=batch,
            metrics={
                "train_loss": {
                    "accepted": jnp.asarray(
                        [(batch + replica) % 2 == 0 for replica in range(5)]
                    ),
                    "score": jnp.arange(5, dtype=jnp.float32) + batch,
                    "nested": {
                        "count": jnp.arange(5, dtype=jnp.int32) + batch,
                    },
                },
                "shared_label": "ready",
            },
        )
        for batch in range(1, 8)
    ]
    expected = [training_executor._history_event(coordinate, bindings) for coordinate in coordinates]
    transfer_calls = 0
    materialized_leaves = 0
    jax_backed_asarray_calls = 0
    device_get = jax.device_get
    np_asarray = np.asarray

    def counted_device_get(value):
        nonlocal transfer_calls, materialized_leaves
        transfer_calls += 1
        materialized_leaves += sum(
            isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(value)
        )
        return device_get(value)

    def counted_asarray(value, *args, **kwargs):
        nonlocal jax_backed_asarray_calls
        if isinstance(value, jax.Array):
            jax_backed_asarray_calls += 1
        return np_asarray(value, *args, **kwargs)

    monkeypatch.setattr(training_executor.jax, "device_get", counted_device_get)
    monkeypatch.setattr(orchestration_events.np, "asarray", counted_asarray)

    for coordinate in coordinates:
        buffer.append(coordinate)
    buffer.flush()
    buffer.assert_empty()

    assert buffer.flush_sizes == [3, 3, 1]
    assert buffer.peak_buffered_updates == 3
    assert transfer_calls == 3
    assert materialized_leaves == 3 * 3
    assert jax_backed_asarray_calls == 0
    assert observations == expected


@pytest.mark.parametrize(
    ("checkpoint_interval", "progress_interval", "expected_chunks"),
    [
        pytest.param(None, None, [1, 1, 1, 1, 1], id="absent-intervals"),
        pytest.param(100, 100, [2, 2, 1], id="oversized-intervals"),
    ],
)
def test_method_observation_buffer_hard_cap_with_absent_or_oversized_intervals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_interval: int | None,
    progress_interval: int | None,
    expected_chunks: list[int],
) -> None:
    spec = _mapped_diagnostics_run_spec()
    spec.worker_execution.method_contract.phase_program.phases[0].max_steps = 5
    spec.training_config.n_batches = 5
    spec.checkpoint_progress.checkpoint_interval = checkpoint_interval
    spec.checkpoint_progress.progress_interval = progress_interval
    buffers: list[training_executor._MethodObservationBuffer] = []
    buffer_type = training_executor._MethodObservationBuffer

    def bounded_buffer(observations, bindings):
        buffer = buffer_type(observations, bindings, capacity=2)
        buffers.append(buffer)
        return buffer

    monkeypatch.setattr(training_executor, "_MethodObservationBuffer", bounded_buffer)
    result = execute_training_run_spec(
        spec,
        preparation=_valid_materialized_preparation(spec),
        registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
        manifest_root=tmp_path / "manifest",
        checkpoint_root=tmp_path / "checkpoint",
        run_id="bounded-observations",
    )

    assert len(buffers) == 1
    assert buffers[0].peak_buffered_updates <= 2
    assert buffers[0].flush_sizes == expected_chunks
    assert result.diagnostics.method_trace is not None
    assert len(result.diagnostics.method_trace.records) == 5 * 5


@pytest.mark.parametrize("invalid_kind", ["malformed", "nonfinite", "wrong-size"])
@pytest.mark.parametrize(
    ("invalid_batches", "expected_published_batches", "expected_checkpoint_count"),
    [
        pytest.param({1, 2, 3}, [], 0, id="early-chunk"),
        pytest.param({4}, [3], 1, id="terminal-remainder"),
    ],
)
def test_invalid_buffered_method_observation_fails_before_matching_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
    invalid_batches: set[int],
    expected_published_batches: list[int],
    expected_checkpoint_count: int,
) -> None:
    spec = _mapped_diagnostics_run_spec()
    spec.worker_execution.method_contract.phase_program.phases[0].max_steps = 4
    spec.training_config.n_batches = 4
    spec.checkpoint_progress.checkpoint_interval = 3
    spec.checkpoint_progress.progress_interval = 3
    buffer_type = training_executor._MethodObservationBuffer

    class CorruptingBuffer(buffer_type):
        def append(self, coordinate):
            completed_batches = coordinate.completed_batches
            if completed_batches in invalid_batches:
                if invalid_kind == "malformed":
                    invalid = {}
                elif invalid_kind == "nonfinite":
                    invalid = jnp.full((5,), jnp.nan)
                else:
                    invalid = {"score": jnp.arange(4, dtype=jnp.float32)}
                coordinate = coordinate.model_copy(
                    update={"metrics": {**coordinate.metrics, "train_loss": invalid}}
                )
            super().append(coordinate)

    monkeypatch.setattr(
        training_executor,
        "_MethodObservationBuffer",
        lambda observations, bindings: CorruptingBuffer(observations, bindings, capacity=3),
    )
    published: list[dict[str, object]] = []
    checkpoint_root = tmp_path / "checkpoint"

    with pytest.raises((TrainingRunExecutorError, ValueError)):
        execute_training_run_spec(
            spec,
            preparation=_valid_materialized_preparation(spec),
            registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
            manifest_root=tmp_path / "manifest",
            checkpoint_root=checkpoint_root,
            run_id=f"invalid-observation-{invalid_kind}",
            progress_callback=published.append,
        )

    assert [event["coordinate"]["completed_batches"] for event in published] == (
        expected_published_batches
    )
    assert len(list(checkpoint_root.glob("transactions/tx-*"))) == expected_checkpoint_count


@pytest.mark.parametrize("stop_mode", ["terminate", "stop", "stop_after_barrier"])
def test_buffered_method_observations_include_terminating_or_saved_barrier_update(
    tmp_path: Path,
    stop_mode: str,
) -> None:
    spec = _mapped_diagnostics_run_spec()
    spec.worker_execution.method_contract.phase_program.phases[0].max_steps = 7
    spec.training_config.n_batches = 7
    spec.checkpoint_progress.checkpoint_interval = 3
    spec.checkpoint_progress.progress_interval = 3
    kwargs: dict[str, object] = {}
    if stop_mode == "stop_after_barrier":
        kwargs["stop_after_barrier"] = "after_train_batch"
    else:
        decision = CancellationDecision(stop_mode, "test", 123.0)
        kwargs["cancellation_probe"] = (
            lambda coordinate: decision if coordinate.completed_batches == 3 else None
        )

    result = execute_training_run_spec(
        spec,
        preparation=_valid_materialized_preparation(spec),
        registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
        manifest_root=tmp_path / "manifest",
        checkpoint_root=tmp_path / "checkpoint",
        run_id=f"buffered-{stop_mode}",
        **kwargs,
    )

    trace = result.diagnostics.method_trace
    assert trace is not None
    assert [(row.completed_batch, row.replica_index) for row in trace.records] == [
        (batch, replica) for batch in range(1, 4) for replica in range(5)
    ]
    assert result.final_coordinate.program_step == 3
    assert len(result.checkpoint_writes) == (0 if stop_mode == "terminate" else 1)
    if stop_mode in {"terminate", "stop"}:
        assert result.status == "cancelled"
        assert result.manifest.provenance.metadata["cancellation"]["action"] == stop_mode


def test_buffered_method_observations_resume_from_partial_checkpoint(
    tmp_path: Path,
) -> None:
    spec = _mapped_diagnostics_run_spec()
    spec.worker_execution.method_contract.phase_program.phases[0].max_steps = 7
    spec.training_config.n_batches = 7
    spec.checkpoint_progress.checkpoint_interval = 3
    spec.checkpoint_progress.progress_interval = 3
    checkpoint_root = tmp_path / "checkpoint"
    partial = execute_training_run_spec(
        spec,
        preparation=_valid_materialized_preparation(spec),
        registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
        manifest_root=tmp_path / "partial-manifest",
        checkpoint_root=checkpoint_root,
        run_id="buffered-partial",
        stop_after_barrier="after_train_batch",
    )
    resumed = execute_training_run_spec(
        spec,
        preparation=_valid_materialized_preparation(spec),
        registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
        manifest_root=tmp_path / "resumed-manifest",
        checkpoint_root=checkpoint_root,
        run_id="buffered-resumed",
        resume=True,
    )

    partial_trace = partial.diagnostics.method_trace
    resumed_trace = resumed.diagnostics.method_trace
    assert partial_trace is not None
    assert resumed_trace is not None
    assert {row.completed_batch for row in partial_trace.records} == {1, 2, 3}
    assert {row.completed_batch for row in resumed_trace.records} == {4, 5, 6, 7}
    assert len(resumed_trace.records) == 4 * 5
    assert [write.manifest.completed_training_batches for write in resumed.checkpoint_writes] == [
        6,
        7,
    ]


def test_buffered_and_immediate_method_observation_contracts_are_equivalent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _mapped_diagnostics_run_spec()
    spec.worker_execution.method_contract.phase_program.phases[0].max_steps = 4
    spec.training_config.n_batches = 4
    spec.checkpoint_progress.checkpoint_interval = 3
    spec.checkpoint_progress.progress_interval = 3
    buffer_type = training_executor._MethodObservationBuffer

    def run_with_capacity(capacity: int, root: Path):
        monkeypatch.setattr(
            training_executor,
            "_MethodObservationBuffer",
            lambda observations, bindings: buffer_type(
                observations,
                bindings,
                capacity=capacity,
            ),
        )
        return execute_training_run_spec(
            spec,
            preparation=_valid_materialized_preparation(spec),
            registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
            manifest_root=root / "manifest",
            checkpoint_root=root / "checkpoint",
            run_id="observation-equivalence",
        )

    immediate = run_with_capacity(1, tmp_path / "immediate")
    buffered = run_with_capacity(500, tmp_path / "buffered")
    immediate_trace = immediate.diagnostics.method_trace
    buffered_trace = buffered.diagnostics.method_trace
    assert immediate_trace is not None
    assert buffered_trace is not None
    assert [record.model_dump(mode="json") for record in buffered_trace.records] == [
        record.model_dump(mode="json") for record in immediate_trace.records
    ]
    assert [
        (event["coordinate"], event["metrics"]) for event in buffered.history_events
    ] == [(event["coordinate"], event["metrics"]) for event in immediate.history_events]
    assert buffered.diagnostics.schema_version == immediate.diagnostics.schema_version
    assert buffered.manifest.schema_version == immediate.manifest.schema_version
    assert [
        write.manifest.completed_training_batches for write in buffered.checkpoint_writes
    ] == [write.manifest.completed_training_batches for write in immediate.checkpoint_writes]
    assert [
        [(slot.slot, slot.sha256) for slot in write.manifest.slots]
        for write in buffered.checkpoint_writes
    ] == [
        [(slot.slot, slot.sha256) for slot in write.manifest.slots]
        for write in immediate.checkpoint_writes
    ]


def test_declared_method_trace_is_durable_with_batch_replica_coordinates(tmp_path: Path) -> None:
    spec = _mapped_diagnostics_run_spec()
    spec.worker_execution.method_contract.phase_program.phases[0].max_steps = 4
    spec.training_config.n_batches = 4
    spec.checkpoint_progress.progress_interval = 3
    result = execute_training_run_spec(
        spec,
        preparation=_valid_materialized_preparation(spec, initial_batch=10),
        registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
        manifest_root=tmp_path / "manifest",
        checkpoint_root=tmp_path / "checkpoint",
        run_id="mapped-trace",
    )

    assert [event["coordinate"]["completed_batches"] for event in result.history_events] == [12, 14]
    trace = result.diagnostics.method_trace
    assert result.diagnostics.schema_version == "feedbax.manifest.training_diagnostics.v3"
    assert trace is not None
    assert trace.method_ref == spec.worker_execution.method_contract.method_ref
    assert trace.trace_schema_id == "feedbax.tests.method_trace"
    assert trace.measurement_basis == "kernel_input"
    assert [(row.completed_batch, row.replica_index) for row in trace.records] == [
        (batch, index) for batch in range(11, 15) for index in range(5)
    ]
    payload = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert payload["method_trace"]["records"] == [
        record.model_dump(mode="json") for record in trace.records
    ]
    diagnostics_ref = next(
        item for item in result.manifest.artifacts if item.role == "training_diagnostics"
    )
    assert diagnostics_ref.metadata["schema_version"].endswith(".v3")


def test_declared_method_trace_uses_cancelled_batch_authority(tmp_path: Path) -> None:
    spec = _mapped_diagnostics_run_spec()
    result = execute_training_run_spec(
        spec,
        preparation=_valid_materialized_preparation(spec, initial_batch=10),
        registry=_mapped_test_registry(spec, _mapped_scalar_kernel),
        manifest_root=tmp_path,
        run_id="cancelled-mapped-trace",
        cancellation_probe=lambda _coordinate: CancellationDecision("terminate", "test", 123.0),
    )

    trace = result.diagnostics.method_trace
    assert trace is not None
    assert result.status == result.manifest.status == "cancelled"
    assert result.diagnostics.cumulative_completed_batches == 11
    assert [(row.completed_batch, row.replica_index) for row in trace.records] == [
        (11, index) for index in range(5)
    ]


def test_method_trace_rejects_non_cartesian_history() -> None:
    declaration = _mapped_diagnostics_run_spec().worker_execution.method_contract.training_diagnostics
    assert declaration is not None
    axis = {
        "axis": "ensemble",
        "role": "replicate",
        "size": 2,
        "level": 0,
        "mode": "mapped",
        "array_axis": 0,
        "leaf_policy": "all_array_leaves",
    }

    def event(batch: int, values: list[float], *, size: int = 2) -> dict[str, object]:
        return {
            "coordinate": {"completed_batches": batch},
            "metrics": {
                "train_loss": {
                    "schema_id": "feedbax.manifest.mapped_metric_value",
                    "schema_version": "feedbax.manifest.mapped_metric_value.v1",
                    "axes": [{**axis, "size": size}],
                    "value": values,
                }
            },
        }

    with pytest.raises(TrainingRunExecutorError, match="duplicates completed batch"):
        _method_training_trace(
            declaration,
            method_ref="feedbax.tests.method",
            method_observations=[event(1, [1.0, 2.0]), event(1, [1.0, 2.0])],
            observation_origin_batch=0,
            completed_batches=1,
            replica_count=2,
        )
    with pytest.raises(TrainingRunExecutorError, match="incomplete batch coverage"):
        _method_training_trace(
            declaration,
            method_ref="feedbax.tests.method",
            method_observations=[event(1, [1.0, 2.0])],
            observation_origin_batch=0,
            completed_batches=2,
            replica_count=2,
        )
    with pytest.raises(TrainingRunExecutorError, match="outside the active segment"):
        _method_training_trace(
            declaration,
            method_ref="feedbax.tests.method",
            method_observations=[event(3, [1.0, 2.0])],
            observation_origin_batch=0,
            completed_batches=2,
            replica_count=2,
        )
    with pytest.raises(TrainingRunExecutorError, match="does not cover the replica axis"):
        _method_training_trace(
            declaration,
            method_ref="feedbax.tests.method",
            method_observations=[event(1, [1.0, 2.0], size=3)],
            observation_origin_batch=0,
            completed_batches=1,
            replica_count=2,
        )
    with pytest.raises(TrainingRunExecutorError, match="does not cover the replica axis"):
        _method_training_trace(
            declaration,
            method_ref="feedbax.tests.method",
            method_observations=[event(1, [1.0, 2.0])],
            observation_origin_batch=0,
            completed_batches=1,
            replica_count=5,
        )


def test_method_trace_transposes_named_mapped_values_by_replica() -> None:
    declaration = _mapped_diagnostics_run_spec().worker_execution.method_contract.training_diagnostics
    assert declaration is not None
    axis = {
        "axis": "ensemble",
        "role": "replicate",
        "size": 2,
        "level": 0,
        "mode": "mapped",
        "array_axis": 0,
        "leaf_policy": "all_array_leaves",
    }
    trace = _method_training_trace(
        declaration,
        method_ref="feedbax.tests.method",
        method_observations=[
            {
                "coordinate": {"completed_batches": 1},
                "metrics": {
                    "train_loss": {
                        "schema_id": "feedbax.manifest.structured_mapped_metric_value",
                        "schema_version": (
                            "feedbax.manifest.structured_mapped_metric_value.v1"
                        ),
                        "axes": [axis],
                        "value": {
                            "accepted": [True, False],
                            "nested": {"score": [[1.5, 2.5], [3.5, 4.5]]},
                        },
                    }
                },
            }
        ],
        observation_origin_batch=0,
        completed_batches=1,
        replica_count=2,
    )

    assert trace is not None
    assert [record.value for record in trace.records] == [
        {"accepted": True, "nested": {"score": [1.5, 2.5]}},
        {"accepted": False, "nested": {"score": [3.5, 4.5]}},
    ]


def test_mapped_learning_rate_trace_retains_every_instance_and_rejects_duplicates() -> None:
    axis = {
        "axis": "ensemble",
        "role": "replicate",
        "size": 2,
        "level": 0,
        "mode": "mapped",
        "array_axis": 0,
        "leaf_policy": "all_array_leaves",
    }
    event = {
        "coordinate": {"program_step": 3},
        "metrics": {
            "learning_rate": {
                "schema_id": "feedbax.manifest.mapped_metric_value",
                "schema_version": "feedbax.manifest.mapped_metric_value.v1",
                "value": [0.1, 0.2],
                "shape": [2],
                "dtype": "float64",
                "axes": [axis],
            }
        },
    }
    trace = _realized_lr_trace([event], declared=())
    assert [sample.learning_rate for sample in trace] == [0.1, 0.2]
    assert [sample.axis_coordinates[0].index for sample in trace] == [0, 1]
    with pytest.raises(TrainingRunExecutorError, match="duplicate coordinate/step"):
        _realized_lr_trace([event, event], declared=())


def test_mapped_preflight_fails_before_execution_and_batch_authority_must_sync() -> None:
    def tracer_unsafe(slots, coordinate, context):
        del coordinate, context
        return {"model": int(slots["model"])}

    executor, _ = _mapped_phase_executor(tracer_unsafe)
    with pytest.raises(WorkerContractValidationError, match="preflight failed"):
        executor.preflight(_mapped_slots(), run_id="mapped", context={})

    def divergent(slots, coordinate, context):
        updates = _mapped_scalar_kernel(slots, coordinate, context)
        updates["batch_counter"] = slots["batch_counter"] + slots["model"].astype(jnp.int32)
        return updates

    executor, _ = _mapped_phase_executor(divergent)
    with pytest.raises(WorkerContractValidationError, match="authorities diverged"):
        executor.run(_mapped_slots(), run_id="mapped")


@pytest.mark.parametrize("boundary", ["preflight", "runtime"])
@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("tree", "PyTree definition"),
        ("shape", "shape or dtype"),
        ("dtype", "shape or dtype"),
        ("category", "array/static leaf category"),
    ],
)
def test_mapped_write_rejects_full_destination_abi_drift(boundary, drift, message) -> None:
    def drifting(slots, coordinate, context):
        updates = _mapped_scalar_kernel(slots, coordinate, context)
        if drift == "tree":
            updates["model"] = {"value": slots["model"]}
        elif drift == "shape":
            updates["model"] = slots["model"][None]
        elif drift == "category":
            updates["model"] = 1.0
        else:
            updates["model"] = slots["model"].astype(jnp.int32)
        return updates

    executor, _ = _mapped_phase_executor(drifting)
    call = executor.preflight if boundary == "preflight" else executor.run
    with pytest.raises(WorkerContractValidationError, match=message):
        call(_mapped_slots(), run_id="mapped")


def test_mapped_optimizer_authority_uses_instance_slots_and_requires_equality() -> None:
    _, bindings = _mapped_phase_executor(_mapped_scalar_kernel)
    runtime = _mapped_slots()

    def extractor(_payload, slots):
        return int(slots["optimizer"])

    assert _synchronized_optimizer_step(extractor, None, runtime, bindings) == 0

    runtime["optimizer"] = jnp.arange(5)
    with pytest.raises(TrainingRunExecutorError, match="authorities diverged"):
        _synchronized_optimizer_step(extractor, None, runtime, bindings)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_slots": {"model": jnp.zeros((5,))}},
        {"preparation": ExecutionPreparationResult(initial_slots={})},
    ],
)
def test_mapped_executor_rejects_loose_prestacked_and_scalar_preparation(kwargs) -> None:
    with pytest.raises(
        TrainingRunExecutorError,
        match="active mapping levels require a Feedbax-materialized execution preparation",
    ):
        execute_training_run_spec(_mapped_run_spec(), **kwargs)


def test_mapped_executor_rejects_public_construction_and_forged_no_seal() -> None:
    with pytest.raises(TypeError):
        MaterializedExecutionPreparation()
    forged = object.__new__(MaterializedExecutionPreparation)

    with pytest.raises(TrainingRunExecutorError, match="lacks Feedbax provenance seal"):
        execute_training_run_spec(_mapped_run_spec(), preparation=forged)


def test_sealed_mapped_preparation_rejects_copy_and_mutated_nested_content() -> None:
    spec = _mapped_run_spec()
    preparation = _valid_materialized_preparation(spec)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy(preparation)
    with pytest.raises(TypeError, match="cannot be copied"):
        deepcopy(preparation)

    immutable_mapping = type(preparation.initial_slots)
    nested = _valid_materialized_preparation(spec)
    object.__setattr__(
        nested,
        "initial_slots",
        immutable_mapping({**nested.initial_slots, "model": {"value": jnp.zeros((5,))}}),
    )
    with pytest.raises(TrainingRunExecutorError, match="mutable mapping"):
        execute_training_run_spec(spec, preparation=nested)

    history = _valid_materialized_preparation(spec)
    object.__setattr__(
        history,
        "initial_slots",
        immutable_mapping(
            {**history.initial_slots, "model": BatchHistory(jnp.zeros((5,)))}
        ),
    )
    with pytest.raises(TrainingRunExecutorError, match="contains BatchHistory"):
        execute_training_run_spec(spec, preparation=history)

    wrong_shape = _valid_materialized_preparation(spec)
    object.__setattr__(
        wrong_shape,
        "initial_slots",
        immutable_mapping({**wrong_shape.initial_slots, "model": jnp.zeros((4,))}),
    )
    with pytest.raises(TrainingRunExecutorError, match="leading axis size 5"):
        execute_training_run_spec(spec, preparation=wrong_shape)


def test_materialized_preparation_digest_uses_canonical_training_spec_identity() -> None:
    spec = _mapped_run_spec()

    assert _run_spec_sha256(spec) == training_spec_sha256(
        spec.model_dump(mode="json", exclude_none=True)
    )


def test_mapped_executor_rejects_stale_fingerprint_and_run_spec_identity() -> None:
    spec = _mapped_run_spec()
    stale = _valid_materialized_preparation(spec)
    object.__setattr__(stale, "identity", replace(stale.identity, fingerprint="0" * 64))
    with pytest.raises(TrainingRunExecutorError, match="fingerprint is stale"):
        execute_training_run_spec(spec, preparation=stale)

    valid = _valid_materialized_preparation(spec)
    changed_spec = spec.model_copy(update={"metadata": {"identity_drift": True}})
    with pytest.raises(TrainingRunExecutorError, match="does not match TrainingRunSpec"):
        execute_training_run_spec(changed_spec, preparation=valid)


@pytest.mark.parametrize(
    ("identity_update", "message"),
    [
        (
            {"coordinate_order": ((AxisCoordinateSpec(axis="ensemble", index=0),),)},
            "coordinate identity mismatch",
        ),
        ({"rng_algorithm_version": "feedbax.preparation_rng_scope.fold_in.v0"}, "unsupported"),
        ({"provider_identity": ""}, "provider_identity"),
    ],
)
def test_mapped_executor_rejects_stale_coordinate_rng_and_provider_identity(
    identity_update,
    message: str,
) -> None:
    spec = _mapped_run_spec()
    preparation = _valid_materialized_preparation(spec)
    identity = replace(preparation.identity, **identity_update)
    identity = replace(identity, fingerprint=_identity_fingerprint(identity))
    object.__setattr__(preparation, "identity", identity)

    with pytest.raises(TrainingRunExecutorError, match=message):
        execute_training_run_spec(spec, preparation=preparation)


def _initial_slots(*, arrays: bool = False) -> dict[str, object]:
    if arrays:
        return {
            "model": jnp.array([0.0]),
            "optimizer": {"count": jnp.array([1.0])},
            "prng": jnp.array([0, 1], dtype=jnp.uint32),
            "batch_counter": jnp.array(0, dtype=jnp.int32),
        }
    return {
        "model": 0,
        "optimizer": {"count": 1},
        "prng": [0, 1],
        "batch_counter": 0,
    }


def _execution_context(
    *,
    collection_root: Path | None = None,
    planned_run_id: str = "feedbax-training-run:planned-row",
) -> NativeExecutionProducerContext:
    resolved_root = "b" * 64
    execution_hash = training_run_execution_hash(resolved_root, [])
    payload = _run_spec().model_dump(mode="json", exclude_none=True)
    payload_sha256 = sha256_bytes(training_spec_canonical_bytes(payload))
    artifact = {
        "schema_id": "feedbax.tests.native_execution",
        "schema_version": "feedbax.tests.native_execution.v1",
        "artifact_id": "artifact://tests/native-execution",
        "sha256": "a" * 64,
    }
    provenance = TrainingRowProvenance(
        row_id="row-a",
        row_index=2,
        planned_run_id=planned_run_id,
        authored_payload_hash="d" * 64,
        lowered_execution_payload_hash=payload_sha256,
        seed=7,
        axis_coordinates={"learning_rate": 3e-4},
        overrides=[{"path": "training.learning_rate", "value": 3e-4}],
        lowerer_identities=[
            RowLowererIdentity(
                lowerer_id="feedbax.tests.lowerer",
                lowerer_version="v3",
            )
        ],
    )
    execution = ExecutionIdentityEnvelope(
        payload=SchemaArtifactRef(
            **{
                **artifact,
                "schema_id": payload["schema_id"],
                "schema_version": payload["schema_version"],
                "artifact_id": f"artifact://sha256/{payload_sha256}",
                "sha256": payload_sha256,
            }
        ),
        authored_intent=AuthoredIntentRef(**artifact, intent_hash="c" * 64),
        resolved_snapshot=ResolvedSnapshotRef(**artifact, root_hash=resolved_root),
        execution_capsule=ExecutionCapsuleRef(
            **artifact,
            execution_hash=execution_hash,
        ),
        immutable_inputs=[],
        row_provenance=provenance,
    )
    schedule_context = ScheduleContextDiagnostic(
        schedule_origin_step=0,
        current_step=0,
        optimizer_count_at_current_step=0,
    )
    return NativeExecutionProducerContext(
        execution=execution,
        environment_fingerprint="environment:fixture",
        collection_root=(str(collection_root) if collection_root is not None else None),
        diagnostics=NativeTrainingDiagnosticsInput(
            seeds=[7],
            lr_trace=[LearningRateDiagnostic(step=1, learning_rate=3e-4)],
            resume_context=schedule_context,
            optimizer_build_context=schedule_context,
        ),
    )


def _chunked_registry(
    *,
    stop_after_program_step: int = 3,
    fail_on_program_step: int | None = None,
    barrier_artifact_sink: bool = False,
) -> tuple[TrainingMethodRegistry, object]:
    contract = standard_supervised_method_contract()
    program = contract.phase_program.model_copy(deep=True)
    phase_writes = list(program.phases[0].writes)
    step_writes = list(program.update_steps[0].writes)
    checkpoint_slots = list(program.checkpoint_barriers[0].slots)
    artifact_sinks = []
    state_slots = list(contract.state_slots)
    if barrier_artifact_sink:
        phase_writes.append("history_chunk")
        step_writes.append("history_chunk")
        artifact_sinks.append(
            BarrierArtifactSinkSpec(
                slot="history_chunk",
                role="training_history_chunk",
                logical_name="history_chunk.bin",
                media_type="application/octet-stream",
            )
        )
        state_slots.append(StateSlotSpec(name="history_chunk", role="auxiliary", required=False))
    phase = program.phases[0].model_copy(
        update={"legal_next": ["train_batch"], "writes": phase_writes}
    )
    update_step = program.update_steps[0].model_copy(update={"writes": step_writes})
    checkpoint_barrier = program.checkpoint_barriers[0].model_copy(
        update={"slots": checkpoint_slots, "artifact_sinks": artifact_sinks}
    )
    transition = PhaseTransitionSpec(
        source="train_batch",
        target="train_batch",
        barrier="after_train_batch",
        guard=MetricGuardSpec(
            predicate_ref="tests.continue_train_chunk",
            metric_slots=[],
        ),
    )
    program = program.model_copy(
        update={
            "phases": [phase],
            "transitions": [transition],
            "update_steps": [update_step],
            "checkpoint_barriers": [checkpoint_barrier],
        }
    )
    contract = contract.model_copy(update={"phase_program": program, "state_slots": state_slots})
    base_kernel = standard_supervised_update_kernels()[
        "feedbax.training.standard_supervised.gradient_update"
    ]

    def gradient_update(slots, coordinate, context):
        if fail_on_program_step is not None and coordinate.program_step >= fail_on_program_step:
            raise RuntimeError("simulated preemption after durable checkpoint")
        updates = dict(base_kernel(slots, coordinate, context))
        if barrier_artifact_sink:
            updates["history_chunk"] = bytes([0, 255, coordinate.program_step])
        return updates

    def continue_train_chunk(slots, coordinate, context):
        del slots, context
        return coordinate.program_step < stop_after_program_step

    registry = TrainingMethodRegistry()
    registry.register(
        TrainingMethodRegistration(
            method_ref="feedbax/standard_supervised/v1",
            payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=StandardSupervisedMethodPayload,
            contract_factory=lambda: contract,
            update_kernels_factory=lambda _payload: {
                "feedbax.training.standard_supervised.gradient_update": gradient_update
            },
            guard_predicates_factory=lambda _payload: {
                "tests.continue_train_chunk": continue_train_chunk
            },
            owner="tests.test_training_run_executor",
            package="feedbax",
        )
    )
    return registry, program


def _interval_registry(*, total_updates: int) -> tuple[TrainingMethodRegistry, object]:
    """Return one multi-update phase for cadence and mid-phase resume tests."""
    contract = standard_supervised_method_contract()
    program = contract.phase_program.model_copy(deep=True)
    phase = program.phases[0].model_copy(update={"max_steps": total_updates})
    program = program.model_copy(update={"phases": [phase]})
    contract = contract.model_copy(update={"phase_program": program})
    registry = TrainingMethodRegistry()
    registry.register(
        TrainingMethodRegistration(
            method_ref="feedbax/standard_supervised/v1",
            payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=StandardSupervisedMethodPayload,
            contract_factory=lambda: contract,
            update_kernels_factory=lambda _payload: standard_supervised_update_kernels(),
            owner="tests.test_training_run_executor",
            package="feedbax",
        )
    )
    return registry, program


def _history_registry(*, stop_after_program_step: int) -> tuple[TrainingMethodRegistry, object]:
    """Return a small registry whose checkpointed history is segment-local."""
    contract = standard_supervised_method_contract()
    program = contract.phase_program.model_copy(deep=True)
    phase = program.phases[0].model_copy(
        update={"writes": [*program.phases[0].writes, "batch_history"]}
    )
    update_step = program.update_steps[0].model_copy(
        update={"writes": [*program.update_steps[0].writes, "batch_history"]}
    )
    checkpoint_barrier = program.checkpoint_barriers[0].model_copy(
        update={
            "slots": [
                *program.checkpoint_barriers[0].slots,
                CheckpointSlotSpec(slot="batch_history"),
            ]
        }
    )
    transition = PhaseTransitionSpec(
        source="train_batch",
        target="train_batch",
        barrier="after_train_batch",
        guard=MetricGuardSpec(predicate_ref="tests.continue_history_segment", metric_slots=[]),
    )
    program = program.model_copy(
        update={
            "phases": [phase],
            "transitions": [transition],
            "update_steps": [update_step],
            "checkpoint_barriers": [checkpoint_barrier],
        }
    )
    contract = contract.model_copy(
        update={
            "phase_program": program,
            "state_slots": [
                *contract.state_slots,
                StateSlotSpec(name="batch_history", role="auxiliary"),
            ],
        }
    )
    base_kernel = standard_supervised_update_kernels()[
        "feedbax.training.standard_supervised.gradient_update"
    ]

    def gradient_update(slots, coordinate, context):
        updates = dict(base_kernel(slots, coordinate, context))
        history = slots["batch_history"]
        updates["batch_history"] = BatchHistory(
            jnp.append(history.value, updates["batch_counter"]),
            batch_axis=history.batch_axis,
            granularity=history.granularity,
        )
        return updates

    def continue_history_segment(slots, coordinate, context):
        del slots, context
        return coordinate.program_step < stop_after_program_step

    registry = TrainingMethodRegistry()
    registry.register(
        TrainingMethodRegistration(
            method_ref="feedbax/standard_supervised/v1",
            payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=StandardSupervisedMethodPayload,
            contract_factory=lambda: contract,
            update_kernels_factory=lambda _payload: {
                "feedbax.training.standard_supervised.gradient_update": gradient_update
            },
            guard_predicates_factory=lambda _payload: {
                "tests.continue_history_segment": continue_history_segment
            },
            owner="tests.test_training_run_executor",
            package="feedbax",
        )
    )
    return registry, program


def _nan_registry(
    *,
    nan_on_program_step: int,
    stop_after_program_step: int = 3,
) -> tuple[TrainingMethodRegistry, object]:
    registry, program = _chunked_registry(stop_after_program_step=stop_after_program_step)
    base_registration = registry.resolve(standard_supervised_method_ref(), path="/method_ref")
    base_kernel = standard_supervised_update_kernels()[
        "feedbax.training.standard_supervised.gradient_update"
    ]

    def gradient_update(slots, coordinate, context):
        updates = dict(base_kernel(slots, coordinate, context))
        if coordinate.program_step >= nan_on_program_step:
            updates["train_loss"] = jnp.array(float("nan"))
            updates["model"] = updates["model"] + jnp.array(float("nan"))
            updates["optimizer"] = {
                **dict(updates["optimizer"]),
                "count": updates["optimizer"]["count"] + jnp.array(float("nan")),
            }
        return updates

    runtime_registry = TrainingMethodRegistry()
    runtime_registry.register(
        TrainingMethodRegistration(
            method_ref="feedbax/standard_supervised/v1",
            payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=StandardSupervisedMethodPayload,
            contract_factory=base_registration.contract_factory,
            update_kernels_factory=lambda _payload: {
                "feedbax.training.standard_supervised.gradient_update": gradient_update
            },
            guard_predicates_factory=base_registration.guard_predicates_factory,
            owner="tests.test_training_run_executor",
            package="feedbax",
        )
    )
    return runtime_registry, program


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bad_rlrmp_payload() -> dict[str, object]:
    return {
        "schema_version": "rlrmp.cs_stochastic_gru.v1",
        "experiment": "flat_3e-5",
    }


def _corrected_rlrmp_payload() -> dict[str, object]:
    return {
        "schema_version": "rlrmp.run_spec.v2",
        "experiment": "flat_3e-5",
    }


class _RlrmpManifestProjectionValues(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    gru_postrun_candidate: bool


def _projection_registry() -> TrainingMethodRegistry:
    registry = default_training_method_registry()
    registry.register_manifest_metadata_projection(
        TrainingManifestMetadataProjectionRegistration(
            source_payload_kind="RLRMPRunSpec",
            source_payload_schema_id="rlrmp.run_spec",
            source_payload_schema_version="rlrmp.run_spec.v2",
            projection_schema_id="rlrmp.manifest_projection",
            projection_schema_version="rlrmp.manifest_projection.v1",
            values_model=_RlrmpManifestProjectionValues,
            owner="rlrmp.training_manifest_projection",
            package="rlrmp",
        )
    )
    return registry


def _manifest_projection(
    payload: dict[str, object],
    *,
    value: bool = True,
) -> TrainingManifestMetadataProjection:
    return TrainingManifestMetadataProjection(
        source_payload_kind="RLRMPRunSpec",
        source_payload_schema_id="rlrmp.run_spec",
        source_payload_schema_version="rlrmp.run_spec.v2",
        source_payload_sha256=sha256_bytes(training_spec_canonical_bytes(payload)),
        projection_schema_id="rlrmp.manifest_projection",
        projection_schema_version="rlrmp.manifest_projection.v1",
        values={"gru_postrun_candidate": value},
    )


def test_execute_training_run_spec_emits_native_manifest_and_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint-custody"
    result = execute_training_run_spec(
        _run_spec(),
        run_id="toy-run",
        initial_slots=_initial_slots(),
        manifest_root=tmp_path,
        checkpoint_root=checkpoint_root,
        training_spec_payload={"experiment": "rlrmp-demo", "variant": "a"},
        training_spec_payload_kind="RLRMPRunSpec",
        training_spec_payload_schema_id="rlrmp.spec.run",
        training_spec_payload_schema_version="rlrmp.spec.run.v1",
    )

    manifest = load_manifest(result.manifest_path)

    assert result.final_slots["model"] == 1
    assert result.final_slots["optimizer"]["count"] == 2
    assert result.final_slots["train_loss"] == 1.0
    assert result.checkpoint_writes[0].manifest.metadata["optimizer_step"] == 2
    assert result.history_events[0]["metrics"] == {"train_loss": 1.0}
    assert result.history_events[0]["coordinate"]["metrics"] == {"train_loss": 1.0}
    assert result.manifest_path.is_relative_to(tmp_path)
    assert manifest.training_spec.kind == "RLRMPRunSpec"
    assert manifest.training_spec.inline == {"experiment": "rlrmp-demo", "variant": "a"}
    assert manifest.training_spec.sha256 is not None
    assert manifest.graph_spec.kind == "GraphSpec"
    assert manifest.task_spec.kind == "TaskSpec"
    assert manifest.checkpoint_custody
    assert manifest.summary_metrics["train_loss"] == 1.0
    assert any(artifact.role == "training_history" for artifact in manifest.artifacts)

    emitted_ref = result.manifest.checkpoint_custody[0]
    loaded_ref = manifest.checkpoint_custody[0]
    assert loaded_ref == emitted_ref
    assert (
        emitted_ref.uri
        == result.checkpoint_writes[0].manifest_path.relative_to(checkpoint_root).as_posix()
    )
    assert not Path(emitted_ref.uri).is_absolute()
    staged_context = resolve_staged_execution_context(
        StagedExecutionDescriptor(
            schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
            schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
            artifact_providers={},
            checkpoint_custody={
                "training-checkpoints": StagedCheckpointCustodySpec(
                    backend="feedbax-checkpoint-transaction-tree"
                )
            },
        ),
        checkpoint_custody_bindings=[
            StagedCheckpointCustodyRootBinding("training-checkpoints", checkpoint_root)
        ],
    )
    resolved = staged_context.resolve_checkpoint_custody_ref(
        loaded_ref,
        binding_name="training-checkpoints",
        slot_names=["model"],
    )
    model_slot = next(slot for slot in resolved.manifest.slots if slot.slot == "model")

    assert resolved.parent_ref.kind == "TrainingCheckpointTransactionManifest"
    assert resolved.parent_ref == loaded_ref
    assert resolved.parent_ref.role == "training_checkpoint_custody"
    assert resolved.parent_ref.id == emitted_ref.id
    assert resolved.manifest_sha256 == emitted_ref.metadata["manifest_sha256"]
    assert resolved.manifest.kind == resolved.parent_ref.kind
    assert resolved.manifest.schema_id == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_ID
    assert resolved.manifest.schema_version == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION
    assert resolved.manifest.transaction_id == emitted_ref.id
    assert (
        resolved.manifest.content_integrity_digest.transaction_root_sha256
        == result.checkpoint_writes[0].manifest.content_integrity_digest.transaction_root_sha256
    )
    assert dict(resolved.slots) == {"model": result.final_slots["model"]}
    assert model_slot.sha256 == model_slot.content_digest.blob_sha256
    assert model_slot.content_digest.slot_root_sha256
    assert model_slot.structural_abi_fingerprint.schema_id
    assert model_slot.structural_abi_fingerprint.schema_version
    assert model_slot.structural_abi_fingerprint.fingerprint_sha256


def test_governed_manifest_metadata_projection_round_trips_deterministically(
    tmp_path: Path,
) -> None:
    payload = _corrected_rlrmp_payload()
    result = execute_training_run_spec(
        _run_spec(),
        run_id="projected",
        initial_slots=_initial_slots(),
        manifest_root=tmp_path,
        registry=_projection_registry(),
        training_spec_payload=payload,
        training_spec_payload_kind="RLRMPRunSpec",
        training_spec_payload_schema_id="rlrmp.run_spec",
        training_spec_payload_schema_version="rlrmp.run_spec.v2",
        manifest_metadata_projection=_manifest_projection(payload),
    )

    loaded = load_manifest(result.manifest_path)
    custody = loaded.metadata_projection_custody
    assert loaded.metadata["gru_postrun_candidate"] is True
    assert custody is not None
    assert custody.values == {"gru_postrun_candidate": True}
    assert custody.source_payload_sha256 == sha256_bytes(training_spec_canonical_bytes(payload))
    assert custody.registration_package == "rlrmp"
    assert loaded.provenance.metadata["manifest_metadata_projection"] == (
        custody.provenance_summary()
    )
    assert result.manifest_path.read_text(encoding="utf-8") == (
        loaded.model_dump_json(indent=2, exclude_none=True) + "\n"
    )


def test_descriptor_metadata_projection_is_the_default_custody_provider(tmp_path: Path) -> None:
    class MethodMetadata(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        optimizer_kind: str

    descriptor = replace(
        standard_supervised_method_descriptor(),
        metadata_projector=TrainingMethodMetadataProjector(
            schema_id="feedbax.tests.standard_metadata",
            schema_version="feedbax.tests.standard_metadata.v1",
            output_model=MethodMetadata,
            projector=lambda payload: {"optimizer_kind": payload.optimizer.type},
        ),
    )
    registry = TrainingMethodRegistry()
    registry.register_descriptor(descriptor)

    result = execute_training_run_spec(
        _run_spec(),
        run_id="descriptor-metadata",
        initial_slots=_initial_slots(),
        manifest_root=tmp_path,
        registry=registry,
    )

    custody = result.manifest.metadata_projection_custody
    assert custody is not None
    assert custody.values == {"optimizer_kind": "adamw"}
    assert custody.source_payload_kind == "TrainingRunSpec"
    assert custody.schema_version == "feedbax.manifest.training_metadata_projection_custody.v1"


def test_descriptor_metadata_projection_rejects_reserved_collision_before_output(
    tmp_path: Path,
) -> None:
    class CollisionMetadata(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        runtime_telemetry: bool

    descriptor = replace(
        standard_supervised_method_descriptor(),
        metadata_projector=TrainingMethodMetadataProjector(
            schema_id="feedbax.tests.collision_metadata",
            schema_version="feedbax.tests.collision_metadata.v1",
            output_model=CollisionMetadata,
            projector=lambda _payload: {"runtime_telemetry": True},
        ),
    )
    registry = TrainingMethodRegistry()
    registry.register_descriptor(descriptor)

    with pytest.raises(TrainingRunExecutorError, match="reserved.*collision"):
        execute_training_run_spec(
            _run_spec(),
            run_id="descriptor-collision",
            initial_slots=_initial_slots(),
            manifest_root=tmp_path,
            registry=registry,
        )
    assert not any(tmp_path.rglob("*"))


def test_projection_free_manifest_serialization_remains_absent(tmp_path: Path) -> None:
    result = execute_training_run_spec(
        _run_spec(),
        run_id="no-projection",
        initial_slots=_initial_slots(),
        manifest_root=tmp_path,
    )
    raw = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert "metadata_projection_custody" not in raw
    assert "manifest_metadata_projection" not in raw["provenance"]["metadata"]

    manifest = TrainingRunManifest(id="feedbax-training-run:serialization")
    explicit_none = TrainingRunManifest(
        id="feedbax-training-run:serialization",
        metadata_projection_custody=None,
    )
    assert manifest.model_dump_json(exclude_none=True) == explicit_none.model_dump_json(
        exclude_none=True
    )


def test_reserved_metadata_policy_matches_all_constructed_feedbax_root_keys(
    tmp_path: Path,
) -> None:
    context = _execution_context(collection_root=tmp_path / "row")
    result = execute_training_run_spec(
        _run_spec(),
        initial_slots=_initial_slots(),
        manifest_root=tmp_path / "runs",
        execution_context=context,
    )
    owned_policy = _feedbax_owned_training_manifest_metadata(
        training_run_spec_schema_version=_run_spec().schema_version,
        include_all_owned_keys=True,
    )
    assert set(result.manifest.metadata) == set(owned_policy)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"gru_postrun_candidate": True, "unregistered": True}, "extra_forbidden"),
        ({"gru_postrun_candidate": float("nan")}, "require finite numbers"),
        ({1: True}, "keys must all be strings"),
    ],
)
def test_manifest_metadata_projection_rejects_unregistered_or_noncanonical_values(
    tmp_path: Path,
    values: dict[object, object],
    match: str,
) -> None:
    payload = _corrected_rlrmp_payload()
    projection = _manifest_projection(payload).model_dump(mode="python")
    projection["values"] = values
    with pytest.raises(TrainingRunExecutorError, match=match):
        execute_training_run_spec(
            _run_spec(),
            run_id="bad-projection",
            initial_slots=_initial_slots(),
            manifest_root=tmp_path,
            registry=_projection_registry(),
            training_spec_payload=payload,
            training_spec_payload_kind="RLRMPRunSpec",
            training_spec_payload_schema_id="rlrmp.run_spec",
            training_spec_payload_schema_version="rlrmp.run_spec.v2",
            manifest_metadata_projection=projection,
        )
    assert not any(tmp_path.rglob("*"))


def test_manifest_metadata_projection_registration_requires_strict_values_model() -> None:
    class PermissiveValues(BaseModel):
        flag: bool

    with pytest.raises(ValueError, match="extra='forbid'"):
        TrainingMethodRegistry().register_manifest_metadata_projection(
            TrainingManifestMetadataProjectionRegistration(
                source_payload_kind="RLRMPRunSpec",
                source_payload_schema_id="rlrmp.run_spec",
                source_payload_schema_version="rlrmp.run_spec.v2",
                projection_schema_id="rlrmp.manifest_projection",
                projection_schema_version="rlrmp.manifest_projection.v1",
                values_model=PermissiveValues,
                owner="rlrmp",
                package="rlrmp",
            )
        )

    class NonStrictValues(BaseModel):
        model_config = ConfigDict(extra="forbid")

        flag: bool

    with pytest.raises(ValueError, match="strict=True"):
        TrainingMethodRegistry().register_manifest_metadata_projection(
            TrainingManifestMetadataProjectionRegistration(
                source_payload_kind="RLRMPRunSpec",
                source_payload_schema_id="rlrmp.run_spec",
                source_payload_schema_version="rlrmp.run_spec.v2",
                projection_schema_id="rlrmp.manifest_projection",
                projection_schema_version="rlrmp.manifest_projection.v1",
                values_model=NonStrictValues,
                owner="rlrmp",
                package="rlrmp",
            )
        )


def test_manifest_metadata_projection_rejects_reserved_collision_before_output(
    tmp_path: Path,
) -> None:
    class CollisionValues(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        runtime_telemetry: bool

    payload = _corrected_rlrmp_payload()
    registry = default_training_method_registry()
    registry.register_manifest_metadata_projection(
        TrainingManifestMetadataProjectionRegistration(
            source_payload_kind="RLRMPRunSpec",
            source_payload_schema_id="rlrmp.run_spec",
            source_payload_schema_version="rlrmp.run_spec.v2",
            projection_schema_id="rlrmp.manifest_projection",
            projection_schema_version="rlrmp.manifest_projection.v1",
            values_model=CollisionValues,
            owner="rlrmp",
            package="rlrmp",
        )
    )
    projection = _manifest_projection(payload).model_copy(
        update={"values": {"runtime_telemetry": True}}
    )
    with pytest.raises(TrainingRunExecutorError, match="reserved.*collision"):
        execute_training_run_spec(
            _run_spec(),
            initial_slots=_initial_slots(),
            manifest_root=tmp_path,
            registry=registry,
            training_spec_payload=payload,
            training_spec_payload_kind="RLRMPRunSpec",
            training_spec_payload_schema_id="rlrmp.run_spec",
            training_spec_payload_schema_version="rlrmp.run_spec.v2",
            manifest_metadata_projection=projection,
        )
    assert not any(tmp_path.rglob("*"))


def test_manifest_metadata_projection_rejects_reserved_feedbax_namespace(
    tmp_path: Path,
) -> None:
    class NamespacedValues(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        feedbax_downstream_marker: bool

    payload = _corrected_rlrmp_payload()
    registry = default_training_method_registry()
    registry.register_manifest_metadata_projection(
        TrainingManifestMetadataProjectionRegistration(
            source_payload_kind="RLRMPRunSpec",
            source_payload_schema_id="rlrmp.run_spec",
            source_payload_schema_version="rlrmp.run_spec.v2",
            projection_schema_id="rlrmp.manifest_projection",
            projection_schema_version="rlrmp.manifest_projection.v1",
            values_model=NamespacedValues,
            owner="rlrmp",
            package="rlrmp",
        )
    )
    projection = _manifest_projection(payload).model_copy(
        update={"values": {"feedbax_downstream_marker": True}}
    )
    with pytest.raises(TrainingRunExecutorError, match="reserved.*collision"):
        execute_training_run_spec(
            _run_spec(),
            initial_slots=_initial_slots(),
            manifest_root=tmp_path,
            registry=registry,
            training_spec_payload=payload,
            training_spec_payload_kind="RLRMPRunSpec",
            training_spec_payload_schema_id="rlrmp.run_spec",
            training_spec_payload_schema_version="rlrmp.run_spec.v2",
            manifest_metadata_projection=projection,
        )
    assert not any(tmp_path.rglob("*"))


def test_manifest_metadata_projection_rejects_source_hash_mismatch_before_output(
    tmp_path: Path,
) -> None:
    payload = _corrected_rlrmp_payload()
    projection = _manifest_projection(payload).model_copy(
        update={"source_payload_sha256": "0" * 64}
    )
    with pytest.raises(TrainingRunExecutorError, match="source payload sha256 mismatch"):
        execute_training_run_spec(
            _run_spec(),
            initial_slots=_initial_slots(),
            manifest_root=tmp_path,
            registry=_projection_registry(),
            training_spec_payload=payload,
            training_spec_payload_kind="RLRMPRunSpec",
            training_spec_payload_schema_id="rlrmp.run_spec",
            training_spec_payload_schema_version="rlrmp.run_spec.v2",
            manifest_metadata_projection=projection,
        )
    assert not any(tmp_path.rglob("*"))


def test_manifest_metadata_projection_rejects_unregistered_source_before_output(
    tmp_path: Path,
) -> None:
    payload = _corrected_rlrmp_payload()
    with pytest.raises(TrainingRunExecutorError, match="no manifest metadata projection"):
        execute_training_run_spec(
            _run_spec(),
            initial_slots=_initial_slots(),
            manifest_root=tmp_path,
            registry=default_training_method_registry(),
            training_spec_payload=payload,
            training_spec_payload_kind="RLRMPRunSpec",
            training_spec_payload_schema_id="rlrmp.run_spec",
            training_spec_payload_schema_version="rlrmp.run_spec.v2",
            manifest_metadata_projection=_manifest_projection(payload),
        )
    assert not any(tmp_path.rglob("*"))


def test_manifest_metadata_projection_registration_rejects_duplicates() -> None:
    registry = _projection_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register_manifest_metadata_projection(
            TrainingManifestMetadataProjectionRegistration(
                source_payload_kind="RLRMPRunSpec",
                source_payload_schema_id="rlrmp.run_spec",
                source_payload_schema_version="rlrmp.run_spec.v2",
                projection_schema_id="rlrmp.manifest_projection",
                projection_schema_version="rlrmp.manifest_projection.v1",
                values_model=_RlrmpManifestProjectionValues,
                owner="rlrmp",
                package="rlrmp",
            )
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "root_value",
        "custody_value",
        "digest",
        "provenance",
        "source_identity",
        "schema_version",
    ],
)
def test_manifest_metadata_projection_tampering_fails_on_load(
    tmp_path: Path,
    tamper: str,
) -> None:
    payload = _corrected_rlrmp_payload()
    result = execute_training_run_spec(
        _run_spec(),
        run_id=f"tamper-{tamper}",
        initial_slots=_initial_slots(),
        manifest_root=tmp_path,
        registry=_projection_registry(),
        training_spec_payload=payload,
        training_spec_payload_kind="RLRMPRunSpec",
        training_spec_payload_schema_id="rlrmp.run_spec",
        training_spec_payload_schema_version="rlrmp.run_spec.v2",
        manifest_metadata_projection=_manifest_projection(payload),
    )
    raw = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    custody = raw["metadata_projection_custody"]
    if tamper == "root_value":
        raw["metadata"]["gru_postrun_candidate"] = False
    elif tamper == "custody_value":
        custody["values"]["gru_postrun_candidate"] = False
    elif tamper == "digest":
        custody["values_sha256"] = "0" * 64
    elif tamper == "provenance":
        raw["provenance"]["metadata"]["manifest_metadata_projection"]["registration_owner"] = (
            "tampered"
        )
    elif tamper == "source_identity":
        custody["source_payload_kind"] = "TamperedRunSpec"
    else:
        custody["schema_version"] = "feedbax.manifest.training_metadata_projection_custody.v0"
    _write_json(result.manifest_path, raw)
    with pytest.raises(ValueError):
        load_manifest(result.manifest_path)


def test_same_manifest_identity_rejects_valid_alternate_projection(tmp_path: Path) -> None:
    payload = _corrected_rlrmp_payload()
    result = execute_training_run_spec(
        _run_spec(),
        run_id="projection-conflict",
        initial_slots=_initial_slots(),
        manifest_root=tmp_path,
        registry=_projection_registry(),
        training_spec_payload=payload,
        training_spec_payload_kind="RLRMPRunSpec",
        training_spec_payload_schema_id="rlrmp.run_spec",
        training_spec_payload_schema_version="rlrmp.run_spec.v2",
        manifest_metadata_projection=_manifest_projection(payload),
    )
    # This is a separately valid projection, not a partial-tamper test. Without
    # an external signature/custody anchor hashes cannot establish authorship;
    # the existing same-manifest-id conflict is the relevant protection here.
    altered = result.manifest.model_dump(mode="json", exclude_none=True)
    altered["metadata_projection_custody"]["values"] = {"gru_postrun_candidate": False}
    altered["metadata_projection_custody"]["values_sha256"] = sha256_bytes(
        training_spec_canonical_bytes({"gru_postrun_candidate": False})
    )
    altered["metadata"]["gru_postrun_candidate"] = False
    custody = altered["metadata_projection_custody"]
    altered["provenance"]["metadata"]["manifest_metadata_projection"] = {
        **altered["provenance"]["metadata"]["manifest_metadata_projection"],
        "values_sha256": custody["values_sha256"],
    }
    different = TrainingRunManifest.model_validate(altered)
    with pytest.raises(ManifestEmissionConflictError, match="different content"):
        _preflight_manifest_emission(
            different,
            root=tmp_path,
            conflict_policy="reuse-identical",
            path=result.manifest_path,
        )


def test_native_execution_context_emits_one_identity_manifest_and_typed_diagnostics(
    tmp_path: Path,
) -> None:
    manifest_root = tmp_path / "manifest-root"
    collection_root = tmp_path / "row"
    context = _execution_context(collection_root=collection_root)

    result = execute_training_run_spec(
        _run_spec(),
        initial_slots=_initial_slots(),
        manifest_root=manifest_root,
        execution_context=context,
    )

    assert result.manifest_path == collection_root / "manifest.json"
    assert result.diagnostics_path == collection_root / "training-diagnostics.json"
    assert not (manifest_root / "manifests" / "training_runs").exists()
    manifest = load_manifest(result.manifest_path)
    assert manifest.id == "feedbax-training-run:planned-row"
    assert manifest.intent_hash == context.execution.authored_intent.intent_hash
    assert manifest.execution_hash == context.execution.execution_capsule.execution_hash
    assert manifest.resolved_semantics_root_hash == context.execution.resolved_snapshot.root_hash
    assert manifest.input_data_identities == []
    assert manifest.training_spec.ref == context.execution.payload.artifact_id
    assert manifest.completed_batches == 1
    assert manifest.metadata["environment_fingerprint"] == "environment:fixture"
    provenance = manifest.metadata["training_row_provenance"]
    assert provenance["planned_run_id"] == manifest.id
    assert provenance["row_id"] == "row-a"
    assert provenance["row_index"] == 2
    assert provenance["axis_coordinates"] == {"learning_rate": 3e-4}
    assert provenance["lowerer_identities"] == [
        {
            "lowerer_id": "feedbax.tests.lowerer",
            "lowerer_version": "v3",
        }
    ]
    assert manifest.provenance.metadata["environment_fingerprint"] == ("environment:fixture")
    assert result.run_id == manifest.id
    assert result.final_coordinate.run_id == manifest.id
    assert result.checkpoint_writes[0].manifest.run_id == manifest.id

    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["kind"] == "TrainingDiagnostics"
    assert diagnostics["schema_version"] == "feedbax.manifest.training_diagnostics.v2"
    assert "method_trace" not in diagnostics
    assert "completed_batches" not in result.history_events[0]["coordinate"]
    assert diagnostics["manifest_id"] == manifest.id
    assert diagnostics["completed_batches"] == 1
    assert diagnostics["segment_completed_batches"] == 1
    assert diagnostics["cumulative_completed_batches"] == 1
    assert diagnostics["seeds"] == [7]
    assert diagnostics["lr_trace"] == [{"step": 1, "learning_rate": 3e-4}]
    assert diagnostics["checkpoint_coordinates"] == [1]
    assert diagnostics["checkpoint_transactions"][0]["completed_batches"] == 1
    assert diagnostics["checkpoint_transactions"][0]["cumulative_completed_batches"] == 1
    diagnostics_ref = next(
        artifact for artifact in manifest.artifacts if artifact.role == "training_diagnostics"
    )
    assert diagnostics_ref.uri == str(result.diagnostics_path)
    assert diagnostics_ref.sha256 == sha256_bytes(result.diagnostics_path.read_bytes())


def test_native_execution_context_and_diagnostics_reject_unknown_schema_versions() -> None:
    context_payload = _execution_context().model_dump(mode="json", exclude_none=True)
    context_payload["schema_version"] = "feedbax.spec.native_execution_context.v0"
    with pytest.raises(ValueError, match="native_execution_context.v1"):
        NativeExecutionProducerContext.model_validate(context_payload)

    diagnostics_payload = {
        "kind": "TrainingDiagnostics",
        "schema_id": "feedbax.manifest.training_diagnostics",
        "schema_version": "feedbax.manifest.training_diagnostics.v0",
        "manifest_id": "feedbax-training-run:test",
        "run_id": "test",
        "terminal_status": "completed",
        "completed_batches": 0,
        "segment_completed_batches": 0,
    }
    with pytest.raises(ValueError, match="training_diagnostics.v2"):
        TrainingDiagnostics.model_validate(diagnostics_payload)


@pytest.mark.parametrize("drift", ["schema", "sha256"])
def test_native_execution_rejects_payload_binding_drift_before_side_effects(
    tmp_path: Path,
    drift: str,
) -> None:
    collection_root = tmp_path / "row"
    context = _execution_context(collection_root=collection_root).model_dump(
        mode="json", exclude_none=True
    )
    if drift == "schema":
        context["execution"]["payload"]["schema_id"] = "feedbax.tests.wrong_payload"
    else:
        context["execution"]["payload"]["sha256"] = "e" * 64
        context["execution"]["row_provenance"]["lowered_execution_payload_hash"] = "e" * 64
    callback_called = False

    def observe_progress(_event: object) -> None:
        nonlocal callback_called
        callback_called = True

    with pytest.raises(TrainingRunExecutorError, match=drift):
        execute_training_run_spec(
            _run_spec(),
            initial_slots=_initial_slots(),
            manifest_root=tmp_path / "manifest-root",
            checkpoint_root=tmp_path / "checkpoint-root",
            execution_context=context,
            progress_callback=observe_progress,
        )

    assert callback_called is False
    assert not collection_root.exists()
    assert not (tmp_path / "manifest-root").exists()
    assert not (tmp_path / "checkpoint-root").exists()


def test_native_execution_rejects_explicit_payload_ref_drift_before_side_effects(
    tmp_path: Path,
) -> None:
    collection_root = tmp_path / "row"
    with pytest.raises(
        TrainingRunManifestPreflightError,
        match="disagrees with authoritative execution payload",
    ):
        execute_training_run_spec(
            _run_spec(),
            initial_slots=_initial_slots(),
            manifest_root=tmp_path / "manifest-root",
            checkpoint_root=tmp_path / "checkpoint-root",
            execution_context=_execution_context(collection_root=collection_root),
            training_spec_payload_ref="artifact://sha256/" + "f" * 64,
        )
    assert not collection_root.exists()
    assert not (tmp_path / "manifest-root").exists()
    assert not (tmp_path / "checkpoint-root").exists()


@pytest.mark.parametrize("invalid", [True, -1, 1.5, [1]])
def test_optimizer_step_extractor_rejects_invalid_output_before_checkpoint_write(
    tmp_path: Path,
    invalid: object,
) -> None:
    descriptor = replace(
        standard_supervised_method_descriptor(),
        optimizer_step_extractor=lambda _payload, _runtime: invalid,
    )
    registry = TrainingMethodRegistry()
    registry.register_descriptor(descriptor)
    checkpoint_root = tmp_path / "checkpoint-root"

    with pytest.raises(TrainingRunExecutorError, match="optimizer_step_extractor"):
        execute_training_run_spec(
            _run_spec(),
            initial_slots=_initial_slots(),
            manifest_root=tmp_path / "manifest-root",
            checkpoint_root=checkpoint_root,
            registry=registry,
        )
    assert not checkpoint_root.exists()


@pytest.mark.parametrize("case", ["missing", "tampered"])
def test_native_execution_rejects_invalid_local_payload_custody_before_side_effects(
    tmp_path: Path,
    case: str,
) -> None:
    collection_root = tmp_path / "row"
    context = _execution_context(collection_root=collection_root).model_dump(
        mode="json", exclude_none=True
    )
    custody_path = tmp_path / "custody" / "training-run-spec.json"
    if case == "tampered":
        custody_path.parent.mkdir()
        custody_path.write_text('{"tampered": true}\n', encoding="utf-8")
    context["execution"]["payload"]["uri"] = custody_path.as_uri()
    callback_called = False

    def observe_progress(_event: object) -> None:
        nonlocal callback_called
        callback_called = True

    with pytest.raises(TrainingRunExecutorError, match="custody"):
        execute_training_run_spec(
            _run_spec(),
            initial_slots=_initial_slots(),
            manifest_root=tmp_path / "manifest-root",
            checkpoint_root=tmp_path / "checkpoint-root",
            execution_context=context,
            progress_callback=observe_progress,
        )

    assert callback_called is False
    assert not collection_root.exists()
    assert not (tmp_path / "manifest-root").exists()
    assert not (tmp_path / "checkpoint-root").exists()
    if case == "tampered":
        assert custody_path.read_text(encoding="utf-8") == '{"tampered": true}\n'


def test_native_execution_allows_non_local_payload_custody_binding(tmp_path: Path) -> None:
    context = _execution_context(collection_root=tmp_path / "row").model_dump(
        mode="json", exclude_none=True
    )
    context["execution"]["payload"]["uri"] = "artifact://registry/execution-payload"

    result = execute_training_run_spec(
        _run_spec(),
        initial_slots=_initial_slots(),
        manifest_root=tmp_path / "manifest-root",
        checkpoint_root=tmp_path / "checkpoint-root",
        execution_context=context,
    )

    assert result.status == "completed"


def test_native_execution_rejects_non_planned_run_id_before_side_effects(
    tmp_path: Path,
) -> None:
    collection_root = tmp_path / "row"
    with pytest.raises(TrainingRunExecutorError, match="planned_run_id"):
        execute_training_run_spec(
            _run_spec(),
            run_id="row-label-is-not-execution-identity",
            initial_slots=_initial_slots(),
            manifest_root=tmp_path / "manifest-root",
            checkpoint_root=tmp_path / "checkpoint-root",
            execution_context=_execution_context(collection_root=collection_root),
        )

    assert not collection_root.exists()
    assert not (tmp_path / "manifest-root").exists()
    assert not (tmp_path / "checkpoint-root").exists()


def test_native_manifest_conflict_fails_before_partial_outputs(tmp_path: Path) -> None:
    collection_root = tmp_path / "row"
    collection_root.mkdir()
    manifest_path = collection_root / "manifest.json"
    manifest_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ManifestEmissionConflictError, match="already exists"):
        execute_training_run_spec(
            _run_spec(),
            initial_slots=_initial_slots(),
            manifest_root=tmp_path / "manifest-root",
            checkpoint_root=tmp_path / "checkpoint-root",
            execution_context=_execution_context(collection_root=collection_root),
        )

    assert manifest_path.read_text(encoding="utf-8") == "existing\n"
    assert list(collection_root.iterdir()) == [manifest_path]
    assert not (tmp_path / "manifest-root").exists()
    assert not (tmp_path / "checkpoint-root").exists()


def test_native_diagnostics_conflict_fails_before_training_or_checkpoint_side_effects(
    tmp_path: Path,
) -> None:
    collection_root = tmp_path / "row"
    collection_root.mkdir()
    diagnostics_path = collection_root / "training-diagnostics.json"
    diagnostics_path.write_text("existing diagnostics\n", encoding="utf-8")
    callback_called = False

    def observe_progress(_event: object) -> None:
        nonlocal callback_called
        callback_called = True

    with pytest.raises(DiagnosticsEmissionConflictError, match="before execution"):
        execute_training_run_spec(
            _run_spec(),
            initial_slots=_initial_slots(),
            manifest_root=tmp_path / "manifest-root",
            checkpoint_root=tmp_path / "checkpoint-root",
            execution_context=_execution_context(collection_root=collection_root),
            progress_callback=observe_progress,
        )

    assert callback_called is False
    assert diagnostics_path.read_text(encoding="utf-8") == "existing diagnostics\n"
    assert list(collection_root.iterdir()) == [diagnostics_path]
    assert not (tmp_path / "manifest-root").exists()
    assert not (tmp_path / "checkpoint-root").exists()


def test_orchestration_injects_canonical_context_only_for_native_commands() -> None:
    context = _execution_context()
    provenance = context.execution.row_provenance
    assert provenance is not None
    row = RunRowSpec(
        row_id=provenance.row_id,
        execution=context.execution,
        launch=RowLaunchSpec(
            command=["python", "-m", "feedbax", "execute-training-run-spec", "spec.json"]
        ),
    )

    command = inject_native_execution_context(
        row.launch.command,
        row=row,
        environment_fingerprint="environment:runtime",
        collection_root="/runtime/rows/row-a",
    )

    assert command[-2] == "--execution-context-json"
    payload = json.loads(command[-1])
    assert payload["execution"] == context.execution.model_dump(mode="json", exclude_none=True)
    assert payload["execution"]["row_provenance"] == provenance.model_dump(
        mode="json", exclude_none=True
    )
    assert payload["environment_fingerprint"] == "environment:runtime"
    assert payload["collection_root"] == "/runtime/rows/row-a"
    assert payload["execution"]["row_provenance"]["planned_run_id"] == (
        "feedbax-training-run:planned-row"
    )

    non_native = ["python", "worker.py"]
    assert (
        inject_native_execution_context(
            non_native,
            row=row,
            environment_fingerprint="environment:runtime",
            collection_root="/runtime/rows/row-a",
        )
        == non_native
    )

    with pytest.raises(NativeExecutionContextError, match="orchestration-owned"):
        inject_native_execution_context(
            [*row.launch.command, "--execution-context", "caller.json"],
            row=row,
            environment_fingerprint="environment:runtime",
            collection_root="/runtime/rows/row-a",
        )


def test_training_manifest_preflight_rejects_observed_rlrmp_schema_mismatch() -> None:
    with pytest.raises(TrainingRunManifestPreflightError) as excinfo:
        preflight_training_run_manifest_payloads(
            _run_spec(),
            training_spec_payload=_bad_rlrmp_payload(),
            training_spec_payload_kind="RLRMPRunSpec",
            training_spec_payload_schema_id="rlrmp.run_spec",
            training_spec_payload_schema_version="rlrmp.run_spec.v2",
            row_id="flat_3e-5",
            spec_path="specs/flat_3e-5.json",
        )

    message = str(excinfo.value)
    assert "row_id='flat_3e-5'" in message
    assert "spec_path='specs/flat_3e-5.json'" in message
    assert "Embedded SpecPayload schema version disagrees with inline payload" in message
    assert "kind='RLRMPRunSpec'" in message
    assert "schema_version='rlrmp.run_spec.v2'" in message
    assert "inline_schema_version='rlrmp.cs_stochastic_gru.v1'" in message


def test_training_manifest_preflight_accepts_corrected_rlrmp_payload() -> None:
    payloads = preflight_training_run_manifest_payloads(
        _run_spec(),
        training_spec_payload=_corrected_rlrmp_payload(),
        training_spec_payload_kind="RLRMPRunSpec",
        training_spec_payload_schema_id="rlrmp.run_spec",
        training_spec_payload_schema_version="rlrmp.run_spec.v2",
        row_id="flat_3e-5",
        spec_path="specs/flat_3e-5.json",
    )

    assert payloads.training_spec.kind == "RLRMPRunSpec"
    assert payloads.training_spec.schema_version == "rlrmp.run_spec.v2"
    assert payloads.training_spec.inline["schema_version"] == "rlrmp.run_spec.v2"
    assert payloads.training_spec.sha256 is not None


def test_executor_runs_manifest_payload_preflight_before_execution_setup() -> None:
    with pytest.raises(TrainingRunManifestPreflightError, match="RLRMPRunSpec"):
        execute_training_run_spec(
            _run_spec(),
            run_id="bad-preflight",
            initial_slots=None,
            training_spec_payload=_bad_rlrmp_payload(),
            training_spec_payload_kind="RLRMPRunSpec",
            training_spec_payload_schema_id="rlrmp.run_spec",
            training_spec_payload_schema_version="rlrmp.run_spec.v2",
        )


def test_manifest_writer_and_preflight_share_payload_builder() -> None:
    spec = _run_spec()
    preflight_payloads = preflight_training_run_manifest_payloads(
        spec,
        training_spec_payload=_corrected_rlrmp_payload(),
        training_spec_payload_kind="RLRMPRunSpec",
        training_spec_payload_schema_id="rlrmp.run_spec",
        training_spec_payload_schema_version="rlrmp.run_spec.v2",
    )
    writer_payloads = build_training_run_manifest_spec_payloads(
        spec,
        training_spec_payload=_corrected_rlrmp_payload(),
        training_spec_payload_kind="RLRMPRunSpec",
        training_spec_payload_schema_id="rlrmp.run_spec",
        training_spec_payload_schema_version="rlrmp.run_spec.v2",
    )

    assert writer_payloads.model_dump() == preflight_payloads.model_dump()


def test_execute_training_run_spec_kernel_context_reaches_kernels_and_guards(
    tmp_path: Path,
) -> None:
    registry, _program = _chunked_registry(stop_after_program_step=99)
    events: list[tuple[str, object]] = []

    base_registration = registry.resolve(standard_supervised_method_ref(), path="/method_ref")
    base_kernel = standard_supervised_update_kernels()[
        "feedbax.training.standard_supervised.gradient_update"
    ]

    def gradient_update(slots, coordinate, context):
        events.append(("kernel", context["runtime_token"]))
        updates = dict(base_kernel(slots, coordinate, context))
        updates["model"] = updates["model"] + context["model_increment"]
        return updates

    def continue_train_chunk(slots, coordinate, context):
        del slots
        events.append(("guard", context["runtime_token"]))
        return coordinate.program_step < context["stop_after_program_step"]

    registration = TrainingMethodRegistration(
        method_ref="feedbax/standard_supervised/v1",
        payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
        payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload_model=StandardSupervisedMethodPayload,
        contract_factory=base_registration.contract_factory,
        update_kernels_factory=lambda _payload: {
            "feedbax.training.standard_supervised.gradient_update": gradient_update
        },
        guard_predicates_factory=lambda _payload: {
            "tests.continue_train_chunk": continue_train_chunk
        },
        owner="tests.test_training_run_executor",
        package="feedbax",
    )
    runtime_registry = TrainingMethodRegistry()
    runtime_registry.register(registration)

    result = execute_training_run_spec(
        _run_spec(),
        run_id="runtime-context",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path,
        registry=runtime_registry,
        kernel_context={
            "runtime_token": "injected",
            "model_increment": jnp.array([10.0]),
            "stop_after_program_step": 1,
        },
    )

    assert result.final_slots["model"].tolist() == [11.0]
    assert ("kernel", "injected") in events
    assert ("guard", "injected") in events


@pytest.mark.parametrize("reserved_key", ["run_spec", "method_payload"])
def test_execute_training_run_spec_rejects_reserved_kernel_context_keys(
    tmp_path: Path,
    reserved_key: str,
) -> None:
    with pytest.raises(TrainingRunExecutorError, match="executor-reserved keys"):
        execute_training_run_spec(
            _run_spec(),
            run_id="reserved-context",
            initial_slots=_initial_slots(arrays=True),
            manifest_root=tmp_path,
            kernel_context={reserved_key: object()},
        )


def test_execute_training_run_spec_invokes_progress_callback_in_history_order(
    tmp_path: Path,
) -> None:
    registry, _program = _chunked_registry(stop_after_program_step=3)
    callback_events: list[dict[str, object]] = []

    result = execute_training_run_spec(
        _run_spec(),
        run_id="callback-run",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path,
        registry=registry,
        progress_callback=callback_events.append,
    )

    assert [event["coordinate"]["program_step"] for event in callback_events] == [1, 2, 3]
    assert [event["coordinate"]["phase"] for event in callback_events] == [
        "train_batch",
        "train_batch",
        "train_batch",
    ]
    assert [event["metrics"]["train_loss"] for event in callback_events] == [1.0, 2.0, 3.0]
    callback_events[0]["metrics"]["train_loss"] = 999.0
    assert result.history_events[0]["metrics"]["train_loss"] == 1.0


@pytest.mark.parametrize(
    (
        "total_updates",
        "checkpoint_interval",
        "progress_interval",
        "expected_checkpoints",
        "expected_progress",
    ),
    [
        (4, 2, 2, [2, 4], [2, 4]),
        (7, 3, 3, [3, 6, 7], [3, 6, 7]),
        (5, 2, 3, [2, 4, 5], [3, 5]),
    ],
)
def test_authored_intervals_drive_multi_update_phase_and_terminal_remainder(
    tmp_path: Path,
    total_updates: int,
    checkpoint_interval: int,
    progress_interval: int,
    expected_checkpoints: list[int],
    expected_progress: list[int],
) -> None:
    registry, _program = _interval_registry(total_updates=total_updates)
    base_spec = _run_spec()
    spec = base_spec.model_copy(
        update={
            "training_config": base_spec.training_config.model_copy(
                update={"n_batches": total_updates}
            ),
            "checkpoint_progress": base_spec.checkpoint_progress.model_copy(
                update={
                    "checkpoint_interval": checkpoint_interval,
                    "progress_interval": progress_interval,
                }
            ),
        }
    )
    callback_events: list[dict[str, object]] = []

    result = execute_training_run_spec(
        spec,
        run_id=f"interval-{total_updates}-{checkpoint_interval}-{progress_interval}",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "runs",
        checkpoint_root=tmp_path / "checkpoints",
        registry=registry,
        progress_callback=callback_events.append,
    )

    assert [
        write.manifest.completed_training_batches for write in result.checkpoint_writes
    ] == expected_checkpoints
    assert [
        write.manifest.completed_coordinate.program_step for write in result.checkpoint_writes
    ] == expected_checkpoints
    assert [
        write.manifest.metadata["barrier_visit_ordinal"] for write in result.checkpoint_writes
    ] == list(range(len(expected_checkpoints)))
    assert [event["coordinate"]["program_step"] for event in callback_events] == expected_progress
    assert [event["coordinate"]["program_step"] for event in result.history_events] == (
        expected_progress
    )
    assert result.final_coordinate.program_step == total_updates
    assert int(result.final_slots["batch_counter"]) == total_updates


def test_authored_cadence_uses_completed_batches_not_program_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _program = _interval_registry(total_updates=4)
    base_spec = _run_spec()
    spec = base_spec.model_copy(
        update={
            "checkpoint_progress": base_spec.checkpoint_progress.model_copy(
                update={"checkpoint_interval": 3, "progress_interval": 3}
            ),
        }
    )
    initial_slots = _initial_slots(arrays=True)
    initial_slots["batch_counter"] = jnp.array(10, dtype=jnp.int32)
    method_observations: list[dict[str, object]] = []
    observation_callbacks: list[object] = []
    build_diagnostics = training_executor._build_training_diagnostics
    run_phase_program = training_executor.PhaseProgramExecutor.run

    def capture_method_observations(**kwargs):
        method_observations.extend(kwargs["method_observations"])
        return build_diagnostics(**kwargs)

    def capture_observation_callback(executor, *args, **kwargs):
        observation_callbacks.append(kwargs["method_observation_callback"])
        return run_phase_program(executor, *args, **kwargs)

    monkeypatch.setattr(
        training_executor,
        "_build_training_diagnostics",
        capture_method_observations,
    )
    monkeypatch.setattr(
        training_executor.PhaseProgramExecutor,
        "run",
        capture_observation_callback,
    )

    result = execute_training_run_spec(
        spec,
        run_id="divergent-batch-progress",
        initial_slots=initial_slots,
        manifest_root=tmp_path / "runs",
        checkpoint_root=tmp_path / "checkpoints",
        registry=registry,
    )

    assert [
        (
            write.manifest.completed_training_batches,
            write.manifest.completed_coordinate.program_step,
        )
        for write in result.checkpoint_writes
    ] == [(12, 2), (14, 4)]
    assert [event["coordinate"]["program_step"] for event in result.history_events] == [2, 4]
    assert method_observations == []
    assert observation_callbacks == [None]
    assert int(result.final_slots["batch_counter"]) == 14


def test_interval_resume_preserves_inner_step_and_absolute_batch_cadence(
    tmp_path: Path,
) -> None:
    base_spec = _run_spec()
    spec = base_spec.model_copy(
        update={
            "training_config": base_spec.training_config.model_copy(update={"n_batches": 7}),
            "checkpoint_progress": base_spec.checkpoint_progress.model_copy(
                update={"checkpoint_interval": 3, "progress_interval": 3}
            ),
        }
    )
    checkpoint_root = tmp_path / "checkpoints"
    partial_registry, _program = _interval_registry(total_updates=7)
    partial = execute_training_run_spec(
        spec,
        run_id="interval-partial",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "partial-runs",
        checkpoint_root=checkpoint_root,
        registry=partial_registry,
        stop_after_barrier="after_train_batch",
    )
    assert partial.final_coordinate.program_step == 3
    assert partial.checkpoint_writes[-1].manifest.completed_training_batches == 3

    resume_events: list[dict[str, object]] = []
    resume_registry, _program = _interval_registry(total_updates=7)
    resumed = execute_training_run_spec(
        spec,
        run_id="interval-resumed",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "resumed-runs",
        checkpoint_root=checkpoint_root,
        registry=resume_registry,
        resume=True,
        progress_callback=resume_events.append,
    )
    full_registry, _program = _interval_registry(total_updates=7)
    full = execute_training_run_spec(
        spec,
        run_id="interval-full",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "full-runs",
        checkpoint_root=tmp_path / "full-checkpoints",
        registry=full_registry,
    )

    assert [write.manifest.completed_training_batches for write in resumed.checkpoint_writes] == [
        6,
        7,
    ]
    assert [
        write.manifest.metadata["barrier_visit_ordinal"] for write in resumed.checkpoint_writes
    ] == [1, 2]
    assert [event["coordinate"]["program_step"] for event in resume_events] == [6, 7]
    assert resumed.final_coordinate.program_step == 7
    assert resumed.final_slots["model"].tolist() == full.final_slots["model"].tolist()
    assert resumed.final_slots["optimizer"]["count"].tolist() == (
        full.final_slots["optimizer"]["count"].tolist()
    )
    assert resumed.final_slots["prng"].tolist() == full.final_slots["prng"].tolist()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "resume_from",
            {"phase": "train_batch", "completed_barrier": "after_train_batch"},
            "/checkpoint_progress/resume_from",
        ),
        (
            "checkpoint_slots",
            {
                "slots": [
                    {
                        "slot": "model",
                        "barrier": "after_train_batch",
                        "program_step": 1,
                    }
                ]
            },
            "/checkpoint_progress/checkpoint_slots",
        ),
    ],
)
def test_unconsumed_checkpoint_policy_fields_fail_loudly_before_execution(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _run_spec().model_dump(mode="json")
    payload["checkpoint_progress"][field] = value
    spec = TrainingRunSpec.model_validate(payload)

    with pytest.raises(NotImplementedError, match=error):
        execute_training_run_spec(
            spec,
            run_id=f"unsupported-{field}",
            initial_slots=_initial_slots(arrays=True),
            manifest_root=tmp_path,
        )

    assert not any(tmp_path.iterdir())


def test_execute_training_run_spec_emits_run_events_without_changing_manifest(
    tmp_path: Path,
) -> None:
    registry, _program = _chunked_registry(stop_after_program_step=3)
    events_path = tmp_path / "events" / "row-1.events.jsonl"
    emitter = RunEventEmitter(
        run_set_id="set-1",
        row_id="row-1",
        path=events_path,
        heartbeat_seconds=None,
    )
    try:
        result = execute_training_run_spec(
            _run_spec(),
            run_id="event-run",
            initial_slots=_initial_slots(arrays=True),
            manifest_root=tmp_path / "manifests",
            checkpoint_root=tmp_path / "checkpoint-custody",
            registry=registry,
            run_event_emitter=emitter,
        )
    finally:
        emitter.close()

    events = RunEventReader(events_path).read_all()
    event_types = [event.type for event in events]

    assert event_types[0] == "ready"
    assert event_types.count("complete") == 1
    assert event_types[-1] == "complete"
    assert [event.payload["program_step"] for event in events if event.type == "progress"] == [
        1,
        2,
        3,
    ]
    checkpoint_events = [event for event in events if event.type == "checkpoint_written"]
    assert [event.payload["program_step"] for event in checkpoint_events] == [1, 2, 3]
    assert all("batch" not in event.payload for event in events)
    assert all("coordinate" in event.payload for event in checkpoint_events)
    assert all("transaction_id" in event.payload for event in checkpoint_events)
    assert result.manifest_path.exists()
    assert result.manifest.checkpoint_custody


def test_execute_training_run_spec_stops_at_checkpoint_with_cancelled_provenance(
    tmp_path: Path,
) -> None:
    registry, _program = _chunked_registry(stop_after_program_step=3)
    events_path = tmp_path / "events" / "row-1.events.jsonl"
    decision = CancellationDecision(
        action="stop",
        source="test",
        requested_at_unix_seconds=123.0,
    )
    emitter = RunEventEmitter(
        run_set_id="set-1",
        row_id="row-1",
        path=events_path,
        heartbeat_seconds=None,
    )
    try:
        result = execute_training_run_spec(
            _run_spec(),
            run_id="cancelled-run",
            initial_slots=_initial_slots(arrays=True),
            manifest_root=tmp_path / "manifests",
            checkpoint_root=tmp_path / "checkpoint-custody",
            registry=registry,
            run_event_emitter=emitter,
            cancellation_probe=lambda coordinate: decision
            if coordinate.program_step == 1
            else None,
        )
    finally:
        emitter.close()

    assert result.status == "cancelled"
    assert result.final_coordinate.program_step == 1
    assert result.manifest.status == "cancelled"
    assert result.manifest.provenance.metadata["cancellation"] == decision.as_provenance()
    assert len(result.checkpoint_writes) == 1
    assert RunEventReader(events_path).read_all()[-1].payload["status"] == "cancelled"


def test_execute_training_run_spec_can_continue_after_an_interruption_decision(
    tmp_path: Path,
) -> None:
    registry, _program = _chunked_registry(stop_after_program_step=3)
    decision = CancellationDecision("continue", "test", 123.0)
    seen = False

    def cancellation_probe(coordinate):
        nonlocal seen
        if coordinate.program_step == 1 and not seen:
            seen = True
            return decision
        return None

    result = execute_training_run_spec(
        _run_spec(),
        run_id="continued-run",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path,
        registry=registry,
        cancellation_probe=cancellation_probe,
    )

    assert seen
    assert result.status == "completed"
    assert result.final_coordinate.program_step == 3


def test_execute_training_run_spec_can_terminate_with_cancelled_manifest(
    tmp_path: Path,
) -> None:
    registry, _program = _chunked_registry(stop_after_program_step=3)
    decision = CancellationDecision("terminate", "test", 123.0)

    result = execute_training_run_spec(
        _run_spec(),
        run_id="terminated-run",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path,
        registry=registry,
        cancellation_probe=lambda coordinate: decision if coordinate.program_step == 1 else None,
    )

    assert result.status == "cancelled"
    assert result.final_coordinate.program_step == 1
    assert not result.checkpoint_writes
    assert result.manifest.provenance.metadata["cancellation"] == decision.as_provenance()


def test_execute_training_run_spec_propagates_progress_callback_errors(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint-custody"
    callback_events: list[dict[str, object]] = []

    def fail_on_progress(event: dict[str, object]) -> None:
        assert event["type"] == "training_progress"
        callback_events.append(event)
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        execute_training_run_spec(
            _run_spec(),
            run_id="callback-failure",
            initial_slots=_initial_slots(),
            manifest_root=tmp_path,
            checkpoint_root=checkpoint_root,
            progress_callback=fail_on_progress,
        )

    assert [event["coordinate"]["program_step"] for event in callback_events] == [1]
    assert not list(checkpoint_root.glob("transactions/tx-*"))
    assert not list((tmp_path / "manifests" / "training_runs").glob("*.json"))


def test_execute_training_run_spec_resumes_through_checkpoint_custody(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint-custody"
    execute_training_run_spec(
        _run_spec(),
        run_id="interrupted",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path,
        checkpoint_root=checkpoint_root,
        stop_after_barrier="after_train_batch",
    )

    resumed = execute_training_run_spec(
        _run_spec(),
        run_id="resumed",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path,
        checkpoint_root=checkpoint_root,
        resume=True,
    )

    assert resumed.final_slots["model"].tolist() == [3.0]
    assert resumed.final_slots["optimizer"]["count"].tolist() == [3.0]
    assert resumed.final_coordinate.program_step == 2
    assert resumed.checkpoint_writes[0].manifest.parent_lineage


def test_same_row_resume_realizes_segment_lineage_and_cadence(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint-custody"
    source_context = _execution_context(collection_root=tmp_path / "source-row")
    source_context = source_context.model_copy(
        update={
            "diagnostics": source_context.diagnostics.model_copy(
                update={
                    "lr_trace": [
                        LearningRateDiagnostic(step=0, learning_rate=3e-4),
                        LearningRateDiagnostic(step=1, learning_rate=3e-4),
                    ]
                }
            )
        }
    )
    source = execute_training_run_spec(
        _run_spec(),
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "source-manifests",
        checkpoint_root=checkpoint_root,
        stop_after_barrier="after_train_batch",
        execution_context=source_context,
    )
    parent = source.checkpoint_writes[-1]
    assert parent.manifest.completed_training_batches == 1

    resume_context = source_context.model_copy(
        update={"collection_root": str(tmp_path / "resumed-row")}
    )
    resumed = execute_training_run_spec(
        _run_spec(),
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "resumed-manifests",
        checkpoint_root=checkpoint_root,
        resume=True,
        execution_context=resume_context,
    )

    child = resumed.checkpoint_writes[-1].manifest
    assert child.segment_lineage.parent_transaction_id == parent.manifest.transaction_id
    assert child.segment_lineage.start_batch == 1
    assert child.segment_lineage.segment_batch_count == 1
    assert resumed.diagnostics.segment_completed_batches == 1
    assert resumed.diagnostics.cumulative_completed_batches == 2
    assert resumed.diagnostics.checkpoint_coordinates == [1]
    assert resumed.diagnostics.resume_context == source_context.diagnostics.resume_context
    assert (
        resumed.diagnostics.optimizer_build_context
        == source_context.diagnostics.optimizer_build_context
    )
    assert resumed.diagnostics.lr_trace == source_context.diagnostics.lr_trace
    conformance_row = ConformanceRowArtifacts(
        row_id="row-a",
        execution=source_context.execution,
        manifest_path=resumed.manifest_path,
        training_diagnostics=resumed.diagnostics.model_dump(mode="json", exclude_none=True),
        bundle_row_spec={
            "training_config": {"n_batches": 2},
            "checkpoint_progress": {"checkpoint_interval": 1},
            "optimizer": {"type": "adamw", "params": {"learning_rate": 3e-4}},
        },
    )
    assert check_checkpoint_cadence(conformance_row).status == "pass"


def test_same_row_resume_progress_fails_closed_without_consistent_authority(
    tmp_path: Path,
) -> None:
    result = execute_training_run_spec(
        _run_spec(),
        run_id="source",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path,
    )
    manifest = result.checkpoint_writes[-1].manifest

    with pytest.raises(TrainingRunExecutorError, match="lacks authoritative"):
        _same_row_resume_start_batch(
            manifest.model_copy(update={"completed_training_batches": None})
        )

    inconsistent_lineage = manifest.segment_lineage.model_copy(
        update={"segment_batch_count": manifest.segment_lineage.segment_batch_count + 1}
    )
    with pytest.raises(TrainingRunExecutorError, match="disagrees with segment lineage"):
        _same_row_resume_start_batch(
            manifest.model_copy(update={"segment_lineage": inconsistent_lineage})
        )


@pytest.mark.parametrize("self_contained", [False, True])
def test_execute_training_run_spec_continuation_writes_segment_lineage_and_histories(
    tmp_path: Path,
    self_contained: bool,
) -> None:
    checkpoint_root = tmp_path / "checkpoint-custody"
    initial_slots = {
        **_initial_slots(arrays=True),
        "batch_history": BatchHistory(jnp.array([], dtype=jnp.int32)),
    }
    source_registry, _program = _history_registry(stop_after_program_step=1)
    source = execute_training_run_spec(
        _run_spec(),
        run_id="history-source",
        initial_slots=initial_slots,
        manifest_root=tmp_path / "source-runs",
        checkpoint_root=checkpoint_root,
        registry=source_registry,
    )
    parent = source.checkpoint_writes[-1]
    assert parent.manifest.completed_training_batches == 1

    continuation = CheckpointContinuationRequest(
        source_completed_batches=1,
        additional_batches=1,
        self_contained=self_contained,
    )
    continuation_spec = _run_spec().model_copy(
        update={
            "checkpoint_progress": _run_spec().checkpoint_progress.model_copy(
                update={"continuation": continuation}
            )
        }
    )
    resume_registry, resume_program = _history_registry(stop_after_program_step=2)
    resumed = execute_training_run_spec(
        continuation_spec,
        run_id="history-continuation",
        initial_slots=initial_slots,
        manifest_root=tmp_path / "continuation-runs",
        checkpoint_root=checkpoint_root,
        registry=resume_registry,
        resume=True,
    )

    child = resumed.checkpoint_writes[-1]
    assert child.manifest.segment_lineage.parent_transaction_id == parent.manifest.transaction_id
    assert child.manifest.segment_lineage.start_batch == 1
    assert child.manifest.segment_lineage.segment_batch_count == 1
    assert resumed.diagnostics.completed_batches == 2
    assert resumed.diagnostics.segment_completed_batches == 1
    assert resumed.diagnostics.cumulative_completed_batches == 2
    assert resumed.diagnostics.checkpoint_coordinates == [1]
    assert resumed.diagnostics.checkpoint_transactions[0].cumulative_completed_batches == 2
    expected_child_slots = {
        **initial_slots,
        "batch_history": BatchHistory(jnp.array([0], dtype=jnp.int32)),
    }
    loaded = load_latest_checkpoint(
        checkpoint_root,
        expected_run_spec=continuation_spec,
        expected_phase_program=resume_program,
        expected_slots=expected_child_slots,
    )
    assert loaded.slots["batch_history"].value.tolist() == [2]

    stitched = concatenate_checkpoint_histories(checkpoint_root, parent_roots={})
    assert stitched.completed_training_batches == 2
    assert stitched.histories["batch_history/"].value.tolist() == [1, 2]

    derived_path = checkpoint_root / "derived" / "history-continuation-stitched-histories.pkl"
    assert derived_path.exists() is self_contained
    if self_contained:
        with derived_path.open("rb") as stream:
            derived = pickle.load(stream)
        assert derived["schema_version"] == "feedbax.derived.checkpoint_histories.v1"
        assert derived["derived"] is True
        assert derived["resume_source"] is False
        assert derived["source_transaction_ids"] == stitched.transaction_ids
        assert derived["completed_training_batches"] == stitched.completed_training_batches
        assert derived["histories"]["batch_history/"].value.tolist() == [1, 2]


def test_execute_training_run_spec_applies_resume_slot_transform(
    tmp_path: Path,
) -> None:
    registry, _program = _chunked_registry(stop_after_program_step=2)
    checkpoint_root = tmp_path / "checkpoint-custody"
    execute_training_run_spec(
        _run_spec(),
        run_id="interrupted",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "runs",
        checkpoint_root=checkpoint_root,
        registry=registry,
        stop_after_barrier="after_train_batch",
    )

    def resize_model(slots):
        transformed = dict(slots)
        transformed["model"] = jnp.pad(transformed["model"], (0, 1))
        return transformed

    resized_initial_slots = _initial_slots(arrays=True)
    resized_initial_slots["model"] = jnp.array([0.0, 0.0])
    resumed = execute_training_run_spec(
        _run_spec(),
        run_id="resumed",
        initial_slots=resized_initial_slots,
        manifest_root=tmp_path / "resume-runs",
        checkpoint_root=checkpoint_root,
        registry=registry,
        resume=True,
        resume_slot_transform=resize_model,
    )

    assert resumed.final_slots["model"].shape == (2,)
    assert resumed.final_slots["model"].tolist() == [3.0, 2.0]
    assert resumed.final_coordinate.program_step == 2


def test_execute_training_run_spec_writes_checkpoint_before_later_failure(
    tmp_path: Path,
) -> None:
    registry, program = _chunked_registry(stop_after_program_step=3, fail_on_program_step=1)
    checkpoint_root = tmp_path / "checkpoint-custody"

    with pytest.raises(RuntimeError, match="simulated preemption"):
        execute_training_run_spec(
            _run_spec(),
            run_id="preempted",
            initial_slots=_initial_slots(arrays=True),
            manifest_root=tmp_path / "runs",
            checkpoint_root=checkpoint_root,
            registry=registry,
        )

    transactions = sorted((checkpoint_root / "transactions").glob("tx-*"))
    assert len(transactions) == 1
    loaded = load_latest_checkpoint(
        checkpoint_root,
        expected_run_spec=_run_spec(),
        expected_phase_program=program,
        expected_slots=_initial_slots(arrays=True),
    )
    assert loaded.manifest.barrier == "after_train_batch"
    assert loaded.manifest.completed_coordinate.program_step == 1
    assert loaded.manifest.metadata["barrier_visit_ordinal"] == 0
    assert loaded.slots["model"].tolist() == [1.0]


@pytest.mark.no_silent_substitution_contract
def test_execute_training_run_spec_raises_on_nan_with_program_coordinate(
    tmp_path: Path,
) -> None:
    registry, _program = _nan_registry(nan_on_program_step=1)

    with pytest.raises(FloatingPointError) as excinfo:
        execute_training_run_spec(
            _run_spec(),
            run_id="nan-raise",
            initial_slots=_initial_slots(arrays=True),
            manifest_root=tmp_path / "runs",
            checkpoint_root=tmp_path / "checkpoints",
            registry=registry,
        )

    message = str(excinfo.value)
    assert "NaN detected" in message
    assert "program_step 2" in message
    assert "inner_step 0" in message
    assert "train_loss" in message
    assert "on_nan='raise'" in message


@pytest.mark.no_silent_substitution_contract
def test_execute_training_run_spec_halts_and_restores_all_checkpoint_slots_on_nan(
    tmp_path: Path,
) -> None:
    registry, program = _nan_registry(nan_on_program_step=1)
    spec = _run_spec().model_copy(update={"on_nan": "halt_restore_checkpoint"})
    checkpoint_root = tmp_path / "checkpoints"

    result = execute_training_run_spec(
        spec,
        run_id="nan-restore",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "runs",
        checkpoint_root=checkpoint_root,
        registry=registry,
    )

    assert result.final_coordinate.program_step == 1
    assert result.final_slots["model"].tolist() == [1.0]
    assert result.final_slots["optimizer"]["count"].tolist() == [2.0]
    assert result.final_slots["prng"].tolist() == [0, 1]
    assert "train_loss" not in result.final_slots
    assert result.manifest.summary_metrics["train_loss"] == 1.0
    assert len(result.checkpoint_writes) == 1

    loaded = load_latest_checkpoint(
        checkpoint_root,
        expected_run_spec=spec,
        expected_phase_program=program,
        expected_slots=_initial_slots(arrays=True),
    )
    assert {slot.slot for slot in loaded.manifest.slots} == {
        "model",
        "optimizer",
        "prng",
        "batch_counter",
    }
    assert loaded.manifest.transaction_id == result.checkpoint_writes[0].manifest.transaction_id
    assert loaded.slots["model"].tolist() == result.final_slots["model"].tolist()
    assert loaded.slots["optimizer"]["count"].tolist() == (
        result.final_slots["optimizer"]["count"].tolist()
    )
    assert loaded.slots["prng"].tolist() == result.final_slots["prng"].tolist()


def test_repeated_barrier_visits_are_durable_and_latest_is_recoverable(
    tmp_path: Path,
) -> None:
    registry, program = _chunked_registry(stop_after_program_step=3)
    checkpoint_root = tmp_path / "checkpoint-custody"

    result = execute_training_run_spec(
        _run_spec(),
        run_id="chunked",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "runs",
        checkpoint_root=checkpoint_root,
        registry=registry,
    )

    assert [
        write.manifest.metadata["barrier_visit_ordinal"] for write in result.checkpoint_writes
    ] == [
        0,
        1,
        2,
    ]
    assert len({write.manifest.transaction_id for write in result.checkpoint_writes}) == 3
    assert len(list((checkpoint_root / "transactions").glob("tx-*"))) == 3

    loaded = load_latest_checkpoint(
        checkpoint_root,
        expected_run_spec=_run_spec(),
        expected_phase_program=program,
        expected_slots=_initial_slots(arrays=True),
    )
    assert loaded.manifest.transaction_id == result.checkpoint_writes[-1].manifest.transaction_id
    assert loaded.manifest.completed_coordinate.program_step == 3
    assert loaded.slots["model"].tolist() == result.final_slots["model"].tolist()


def test_repeated_barrier_visits_capture_binary_artifact_sinks(
    tmp_path: Path,
) -> None:
    registry, _program = _chunked_registry(
        stop_after_program_step=3,
        barrier_artifact_sink=True,
    )

    result = execute_training_run_spec(
        _run_spec(),
        run_id="chunked-sidecars",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "runs",
        checkpoint_root=tmp_path / "checkpoints",
        registry=registry,
    )

    artifacts = [
        artifact
        for artifact in result.manifest.artifacts
        if artifact.role == "training_history_chunk"
    ]
    assert len(artifacts) == 3
    assert [artifact.metadata["barrier_visit_ordinal"] for artifact in artifacts] == [0, 1, 2]
    assert [artifact.metadata["program_step"] for artifact in artifacts] == [1, 2, 3]
    assert all(artifact.metadata["barrier"] == "after_train_batch" for artifact in artifacts)
    assert all(artifact.metadata["slot"] == "history_chunk" for artifact in artifacts)
    assert all(artifact.metadata["checkpoint_transaction_id"] for artifact in artifacts)
    assert len({artifact.logical_name for artifact in artifacts}) == 3

    for index, artifact in enumerate(artifacts):
        payload = bytes([0, 255, index])
        assert artifact.media_type == "application/octet-stream"
        assert artifact.size_bytes == len(payload)
        assert artifact.sha256 == sha256_bytes(payload)
        assert Path(artifact.uri).read_bytes() == payload

    manifest = load_manifest(result.manifest_path)
    persisted_artifacts = [
        artifact for artifact in manifest.artifacts if artifact.role == "training_history_chunk"
    ]
    assert [artifact.sha256 for artifact in persisted_artifacts] == [
        artifact.sha256 for artifact in artifacts
    ]
    assert all(
        "history_chunk" not in {slot.slot for slot in write.manifest.slots}
        for write in result.checkpoint_writes
    )

    loaded = load_latest_checkpoint(
        tmp_path / "checkpoints",
        expected_run_spec=_run_spec(),
        expected_phase_program=_program,
        expected_slots=_initial_slots(arrays=True),
    )
    assert "history_chunk" not in loaded.slots


def test_partial_run_resume_matches_uninterrupted_chunked_execution(
    tmp_path: Path,
) -> None:
    full_registry, _program = _chunked_registry(stop_after_program_step=3)
    full = execute_training_run_spec(
        _run_spec(),
        run_id="chunked-full",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "full-runs",
        checkpoint_root=tmp_path / "full-checkpoints",
        registry=full_registry,
    )

    partial_registry, _program = _chunked_registry(stop_after_program_step=3)
    checkpoint_root = tmp_path / "partial-checkpoints"
    execute_training_run_spec(
        _run_spec(),
        run_id="chunked-partial",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "partial-runs",
        checkpoint_root=checkpoint_root,
        registry=partial_registry,
        stop_after_barrier="after_train_batch",
    )

    resume_registry, _program = _chunked_registry(stop_after_program_step=3)
    resumed = execute_training_run_spec(
        _run_spec(),
        run_id="chunked-resumed",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "resume-runs",
        checkpoint_root=checkpoint_root,
        registry=resume_registry,
        resume=True,
    )

    assert resumed.final_slots["model"].tolist() == full.final_slots["model"].tolist()
    assert resumed.final_slots["optimizer"]["count"].tolist() == (
        full.final_slots["optimizer"]["count"].tolist()
    )
    assert resumed.final_slots["prng"].tolist() == full.final_slots["prng"].tolist()
    assert resumed.final_coordinate.program_step == full.final_coordinate.program_step
    assert resumed.checkpoint_writes[0].manifest.metadata["barrier_visit_ordinal"] == 1
    assert resumed.checkpoint_writes[0].manifest.parent_lineage


def test_execute_training_run_spec_rejects_invalid_spec_before_launch_with_path() -> None:
    with pytest.raises(TrainingRunExecutorError, match="/initial_slots"):
        execute_training_run_spec(_run_spec())


def test_execute_training_run_spec_unknown_method_ref_reports_available_registry() -> None:
    payload = _run_spec().model_dump(mode="json")
    payload["method_ref"]["name"] = "unknown"

    with pytest.raises(TrainingRunExecutorError) as excinfo:
        execute_training_run_spec(payload, initial_slots=_initial_slots())

    message = str(excinfo.value)
    assert "/method_ref" in message
    assert "unknown method_ref 'feedbax/unknown/v1'" in message
    assert "feedbax/standard_supervised/v1" in message


def test_execute_training_run_spec_manifest_root_injection_and_conflict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "injected"
    execute_training_run_spec(
        _run_spec(),
        run_id="stable-id",
        initial_slots=_initial_slots(),
        manifest_root=root,
    )

    manifest_path = root / "manifests" / "training_runs" / "feedbax-training-run_stable-id.json"
    assert manifest_path.is_file()

    with pytest.raises(ManifestEmissionConflictError, match="already exists"):
        execute_training_run_spec(
            _run_spec(),
            run_id="stable-id",
            initial_slots={**_initial_slots(), "model": 5},
            manifest_root=root,
        )


def test_execute_training_run_spec_cli_smoke(tmp_path: Path) -> None:
    spec_path = tmp_path / "training-run-spec.json"
    slots_path = tmp_path / "initial-slots.json"
    context_path = tmp_path / "execution-context.json"
    row_dir = tmp_path / "row"
    events_dir = tmp_path / "events"
    cache_dir = tmp_path / "jax-cache"
    cache_dir.mkdir()
    (cache_dir / "preexisting-cache-entry").write_bytes(b"cache")
    _write_json(spec_path, _run_spec().model_dump(mode="json"))
    _write_json(slots_path, _initial_slots())
    _write_json(
        context_path,
        _execution_context(planned_run_id="feedbax-training-run:planned-cli").model_dump(
            mode="json", exclude_none=True
        ),
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "feedbax",
            "execute-training-run-spec",
            str(spec_path),
            "--manifest-root",
            str(tmp_path / "runs"),
            "--initial-slots",
            str(slots_path),
            "--run-id",
            "feedbax-training-run:planned-cli",
            "--execution-context",
            str(context_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
        env={
            **os.environ,
            "FEEDBAX_RUN_SET_ID": "telemetry-set",
            "FEEDBAX_ROW_ID": "telemetry-row",
            "FEEDBAX_ROW_DIR": str(row_dir),
            "FEEDBAX_RUN_EVENTS_DIR": str(events_dir),
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
        },
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["run_id"] == "feedbax-training-run:planned-cli"
    assert payload["status"] == "completed"
    assert Path(payload["manifest_path"]) == row_dir / "manifest.json"
    assert Path(payload["manifest_path"]).is_file()
    assert (row_dir / "training-diagnostics.json").is_file()
    assert payload["manifest_payload"]["id"] == "feedbax-training-run:planned-cli"
    assert "phase=train_batch" in proc.stderr
    assert "batch=1" in proc.stderr
    assert "loss=1" in proc.stderr
    assert "elapsed=" in proc.stderr
    telemetry = payload["manifest_payload"]["summary_metrics"]["runtime_telemetry"]
    assert telemetry["measurement_semantics"] == ("measurement_start_to_first_progress_callback")
    assert telemetry["measurement_start_semantics"] == "worker_command_entry"
    assert telemetry["start_to_first_progress_seconds"] >= 0
    assert telemetry["compile_time_estimate_seconds"] is None
    assert telemetry["persistent_cache_effectiveness_proxy"] in {
        "new_entries_observed",
        "preexisting_entries_no_new_entries_observed",
    }
    events = [
        json.loads(line)
        for line in (events_dir / "telemetry-row.events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    ready = next(event for event in events if event["type"] == "ready")
    assert ready["payload"]["runtime_telemetry"] == telemetry


def test_preflight_training_run_manifest_cli_reports_normalized_payload(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "training-run-spec.json"
    payload_path = tmp_path / "rlrmp-payload.json"
    _write_json(spec_path, _run_spec().model_dump(mode="json"))
    _write_json(payload_path, _corrected_rlrmp_payload())

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "feedbax",
            "preflight-training-run-manifest",
            str(spec_path),
            "--row-id",
            "flat_3e-5",
            "--training-payload",
            str(payload_path),
            "--training-payload-kind",
            "RLRMPRunSpec",
            "--training-payload-schema-id",
            "rlrmp.run_spec",
            "--training-payload-schema-version",
            "rlrmp.run_spec.v2",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["training_spec"]["kind"] == "RLRMPRunSpec"
    assert payload["training_spec"]["schema_version"] == "rlrmp.run_spec.v2"
    assert payload["training_spec"]["sha256"]
