from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import jax.numpy as jnp

from feedbax.contracts.manifest import load_manifest, sha256_bytes
from feedbax.contracts.training import (
    LossTermSpec,
    ObjectiveSlotSpec,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    StandardSupervisedMethodPayload,
    TaskSpec,
    TrainingConfig,
    TrainingMethodRegistration,
    TrainingMethodRegistry,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_update_kernels,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.contracts.worker import (
    BarrierArtifactSinkSpec,
    MetricGuardSpec,
    PhaseTransitionSpec,
    StateSlotSpec,
)
from feedbax.orchestration.events import RunEventEmitter, RunEventReader
from feedbax.training.checkpoint_custody import load_latest_checkpoint
from feedbax.training.executor import (
    ManifestEmissionConflictError,
    TrainingRunExecutorError,
    execute_training_run_spec,
)
from feedbax.training.manifest_preflight import (
    TrainingRunManifestPreflightError,
    build_training_run_manifest_spec_payloads,
    preflight_training_run_manifest_payloads,
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


def _initial_slots(*, arrays: bool = False) -> dict[str, object]:
    if arrays:
        return {
            "model": jnp.array([0.0]),
            "optimizer": {"count": jnp.array([1.0])},
            "prng": jnp.array([0, 1], dtype=jnp.uint32),
        }
    return {
        "model": 0,
        "optimizer": {"count": 1},
        "prng": [0, 1],
    }


def _chunked_registry(
    *,
    stop_after_global_step: int = 3,
    fail_on_global_step: int | None = None,
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
        if fail_on_global_step is not None and coordinate.global_step >= fail_on_global_step:
            raise RuntimeError("simulated preemption after durable checkpoint")
        updates = dict(base_kernel(slots, coordinate, context))
        if barrier_artifact_sink:
            updates["history_chunk"] = bytes([0, 255, coordinate.global_step])
        return updates

    def continue_train_chunk(slots, coordinate, context):
        del slots, context
        return coordinate.global_step < stop_after_global_step

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


def _nan_registry(
    *,
    nan_on_global_step: int,
    stop_after_global_step: int = 3,
) -> tuple[TrainingMethodRegistry, object]:
    registry, program = _chunked_registry(stop_after_global_step=stop_after_global_step)
    base_registration = registry.resolve(standard_supervised_method_ref(), path="/method_ref")
    base_kernel = standard_supervised_update_kernels()[
        "feedbax.training.standard_supervised.gradient_update"
    ]

    def gradient_update(slots, coordinate, context):
        updates = dict(base_kernel(slots, coordinate, context))
        if coordinate.global_step >= nan_on_global_step:
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


def test_execute_training_run_spec_emits_native_manifest_and_checkpoint(
    tmp_path: Path,
) -> None:
    result = execute_training_run_spec(
        _run_spec(),
        run_id="toy-run",
        initial_slots=_initial_slots(),
        manifest_root=tmp_path,
        training_spec_payload={"experiment": "rlrmp-demo", "variant": "a"},
        training_spec_payload_kind="RLRMPRunSpec",
        training_spec_payload_schema_id="rlrmp.spec.run",
        training_spec_payload_schema_version="rlrmp.spec.run.v1",
    )

    manifest = load_manifest(result.manifest_path)

    assert result.final_slots["model"] == 1
    assert result.final_slots["optimizer"]["count"] == 2
    assert result.final_slots["train_loss"] == 1.0
    assert result.history_events[0]["metrics"] == {"train_loss": 1.0}
    assert result.history_events[0]["coordinate"]["metrics"] == {"train_loss": 1.0}
    assert result.manifest_path.is_relative_to(tmp_path)
    assert manifest.training_spec.kind == "RLRMPRunSpec"
    assert manifest.training_spec.inline == {"experiment": "rlrmp-demo", "variant": "a"}
    assert manifest.training_spec.sha256 is not None
    assert manifest.graph_spec.kind == "GraphSpec"
    assert manifest.task_spec.kind == "TaskSpec"
    assert manifest.checkpoint_custody
    assert Path(manifest.checkpoint_custody[0].uri).is_file()
    assert manifest.summary_metrics["train_loss"] == 1.0
    assert any(artifact.role == "training_history" for artifact in manifest.artifacts)


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
    registry, _program = _chunked_registry(stop_after_global_step=99)
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
        return coordinate.global_step < context["stop_after_global_step"]

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
            "stop_after_global_step": 1,
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
    registry, _program = _chunked_registry(stop_after_global_step=3)
    callback_events: list[dict[str, object]] = []

    result = execute_training_run_spec(
        _run_spec(),
        run_id="callback-run",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path,
        registry=registry,
        progress_callback=callback_events.append,
    )

    assert [event["coordinate"]["global_step"] for event in callback_events] == [1, 2, 3]
    assert [event["coordinate"]["phase"] for event in callback_events] == [
        "train_batch",
        "train_batch",
        "train_batch",
    ]
    assert [event["metrics"]["train_loss"] for event in callback_events] == [1.0, 2.0, 3.0]
    callback_events[0]["metrics"]["train_loss"] = 999.0
    assert result.history_events[0]["metrics"]["train_loss"] == 1.0


def test_execute_training_run_spec_emits_run_events_without_changing_manifest(
    tmp_path: Path,
) -> None:
    registry, _program = _chunked_registry(stop_after_global_step=3)
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
    assert [event.payload["batch"] for event in events if event.type == "progress"] == [
        1,
        2,
        3,
    ]
    checkpoint_events = [event for event in events if event.type == "checkpoint_written"]
    assert [event.payload["batch"] for event in checkpoint_events] == [1, 2, 3]
    assert all("coordinate" in event.payload for event in checkpoint_events)
    assert all("transaction_id" in event.payload for event in checkpoint_events)
    assert result.manifest_path.exists()
    assert result.manifest.checkpoint_custody


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

    assert [event["coordinate"]["global_step"] for event in callback_events] == [1]
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
    assert resumed.final_coordinate.global_step == 2
    assert resumed.checkpoint_writes[0].manifest.parent_lineage


def test_execute_training_run_spec_applies_resume_slot_transform(
    tmp_path: Path,
) -> None:
    registry, _program = _chunked_registry(stop_after_global_step=2)
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
    assert resumed.final_coordinate.global_step == 2


def test_execute_training_run_spec_writes_checkpoint_before_later_failure(
    tmp_path: Path,
) -> None:
    registry, program = _chunked_registry(stop_after_global_step=3, fail_on_global_step=1)
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
    assert loaded.manifest.completed_coordinate.global_step == 1
    assert loaded.manifest.metadata["barrier_visit_ordinal"] == 0
    assert loaded.slots["model"].tolist() == [1.0]


@pytest.mark.no_silent_substitution_contract
def test_execute_training_run_spec_raises_on_nan_with_batch_and_step(
    tmp_path: Path,
) -> None:
    registry, _program = _nan_registry(nan_on_global_step=1)

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
    assert "batch 2" in message
    assert "step 0" in message
    assert "train_loss" in message
    assert "on_nan='raise'" in message


@pytest.mark.no_silent_substitution_contract
def test_execute_training_run_spec_halts_and_restores_all_checkpoint_slots_on_nan(
    tmp_path: Path,
) -> None:
    registry, program = _nan_registry(nan_on_global_step=1)
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

    assert result.final_coordinate.global_step == 1
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
    assert {slot.slot for slot in loaded.manifest.slots} == {"model", "optimizer", "prng"}
    assert loaded.manifest.transaction_id == result.checkpoint_writes[0].manifest.transaction_id
    assert loaded.slots["model"].tolist() == result.final_slots["model"].tolist()
    assert loaded.slots["optimizer"]["count"].tolist() == (
        result.final_slots["optimizer"]["count"].tolist()
    )
    assert loaded.slots["prng"].tolist() == result.final_slots["prng"].tolist()


def test_repeated_barrier_visits_are_durable_and_latest_is_recoverable(
    tmp_path: Path,
) -> None:
    registry, program = _chunked_registry(stop_after_global_step=3)
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
    assert loaded.manifest.completed_coordinate.global_step == 3
    assert loaded.slots["model"].tolist() == result.final_slots["model"].tolist()


def test_repeated_barrier_visits_capture_binary_artifact_sinks(
    tmp_path: Path,
) -> None:
    registry, _program = _chunked_registry(
        stop_after_global_step=3,
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
    assert [artifact.metadata["global_step"] for artifact in artifacts] == [1, 2, 3]
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
    full_registry, _program = _chunked_registry(stop_after_global_step=3)
    full = execute_training_run_spec(
        _run_spec(),
        run_id="chunked-full",
        initial_slots=_initial_slots(arrays=True),
        manifest_root=tmp_path / "full-runs",
        checkpoint_root=tmp_path / "full-checkpoints",
        registry=full_registry,
    )

    partial_registry, _program = _chunked_registry(stop_after_global_step=3)
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

    resume_registry, _program = _chunked_registry(stop_after_global_step=3)
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
    assert resumed.final_coordinate.global_step == full.final_coordinate.global_step
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

    with pytest.raises(ManifestEmissionConflictError, match="different content"):
        execute_training_run_spec(
            _run_spec(),
            run_id="stable-id",
            initial_slots={**_initial_slots(), "model": 5},
            manifest_root=root,
        )


def test_execute_training_run_spec_cli_smoke(tmp_path: Path) -> None:
    spec_path = tmp_path / "training-run-spec.json"
    slots_path = tmp_path / "initial-slots.json"
    _write_json(spec_path, _run_spec().model_dump(mode="json"))
    _write_json(slots_path, _initial_slots())

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
            "cli-toy",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["run_id"] == "cli-toy"
    assert payload["status"] == "completed"
    assert Path(payload["manifest_path"]).is_file()
    assert "batch=1" in proc.stderr
    assert "loss=1" in proc.stderr
    assert "elapsed=" in proc.stderr


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
