from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import jax.numpy as jnp

from feedbax.contracts.manifest import load_manifest
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
from feedbax.contracts.worker import MetricGuardSpec, PhaseTransitionSpec
from feedbax.training.checkpoint_custody import load_latest_checkpoint
from feedbax.training.executor import (
    ManifestEmissionConflictError,
    TrainingRunExecutorError,
    execute_training_run_spec,
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
) -> tuple[TrainingMethodRegistry, object]:
    contract = standard_supervised_method_contract()
    program = contract.phase_program.model_copy(deep=True)
    phase = program.phases[0].model_copy(update={"legal_next": ["train_batch"]})
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
        }
    )
    contract = contract.model_copy(update={"phase_program": program})
    base_kernel = standard_supervised_update_kernels()[
        "feedbax.training.standard_supervised.gradient_update"
    ]

    def gradient_update(slots, coordinate, context):
        if fail_on_global_step is not None and coordinate.global_step >= fail_on_global_step:
            raise RuntimeError("simulated preemption after durable checkpoint")
        return base_kernel(slots, coordinate, context)

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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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

    assert [write.manifest.metadata["barrier_visit_ordinal"] for write in result.checkpoint_writes] == [
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
