from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import optax
import pytest

from feedbax.contracts.checkpoints import CheckpointContinuationRequest
from feedbax.contracts.training import (
    StandardSupervisedMethodPayload,
    TrainingMethodRegistry,
    standard_supervised_method_descriptor,
)
from feedbax.contracts.worker import AxisCoordinateSpec, MethodTrainingDiagnosticsSpec
from feedbax.orchestration.drivers.runpod import (
    CommandResult,
    RunPodDriverConfig,
    RunPodDriverError,
    RunPodOrchestrationDriver,
    validate_runpod_repo_realization_plan,
)
from feedbax.orchestration.repo_snapshot import verify_repo_snapshot
from feedbax.orchestration.schedule_eval import compare_continuation_schedule_projections
from feedbax.orchestration.stages import (
    STAGE_PREFLIGHT,
    STAGE_PROVISION,
    StageEngine,
)
from feedbax.orchestration.state import RunSetStateStore
from feedbax.training.checkpoint_custody import (
    authenticated_run_contract_source_projection,
    load_checkpoint_custody_documents,
)
from feedbax.training.executor import execute_training_run_spec
from feedbax.training.optimizers import build_optimizer
from feedbax.training.preparation import (
    ExecutionPreparationRequest,
    _build_materialized_execution_preparation,
)
from feedbax.training.worker_validation import validate_worker_contract
from tests.test_runpod_orchestration_driver import (
    FakeRunPodTransport,
    _bundle,
    _layout_case,
)
from tests.test_training_run_executor import (
    _scheduled_continuation_spec,
    _write_mapped_schedule_checkpoint,
)


def test_layout_validation_composes_two_exact_sealed_repo_roots(tmp_path: Path) -> None:
    """Resolve live adjacency once, then validate only the keyed sealed roots."""
    lock_text = (
        'version = 1\n[[package]]\nname = "consumer"\n'
        'source = { editable = "../provider" }\n'
    )
    consumer = tmp_path / "consumer"
    provider = tmp_path / "provider"
    bundle, config = _layout_case(
        tmp_path,
        lock_text,
        local_repos={"consumer": consumer, "provider": provider},
        remote_repos={
            "consumer": "/workspace/consumer",
            "provider": "/workspace/provider",
        },
    )
    driver = RunPodOrchestrationDriver(config=config, transport=FakeRunPodTransport())
    plan = driver.seal_repo_realization_plan(bundle)
    sealed_error, sealed_observed = validate_runpod_repo_realization_plan(
        bundle, config, plan, driver._repo_snapshots
    )
    (consumer / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "missing"\n'
        'source = { editable = "../missing-sibling" }\n',
        encoding="utf-8",
    )
    after_live_mutation_error, after_live_mutation_observed = validate_runpod_repo_realization_plan(
        bundle, config, plan, driver._repo_snapshots
    )

    assert sealed_error is None
    assert after_live_mutation_error is None
    assert after_live_mutation_observed == sealed_observed
    assert plan.editable_source_resolutions[0].target_repo == "provider"
    assert plan.editable_source_resolutions[0].target_subpath == "."
    for name, snapshot in driver._repo_snapshots.snapshots.items():
        shipped = plan.snapshot_manifest.repos[name]
        assert snapshot.record.content_sha256 == shipped.content_sha256
        assert snapshot.staging_root.name == shipped.content_sha256
        verify_repo_snapshot(
            snapshot.staging_root,
            content_sha256=shipped.content_sha256,
            file_count=shipped.file_count,
        )


class _OwnedPodTransport(FakeRunPodTransport):
    def __init__(self, bundle_image: str) -> None:
        super().__init__()
        self.bundle_image = bundle_image
        self.owned_pods: set[str] = set()

    def runpodctl(
        self,
        *args: str,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        self.runpodctl_calls.append(args)
        self.runpodctl_timeouts.append(timeout_seconds)
        self.operations.append(f"runpodctl:{' '.join(args)}")
        if args == ("user", "--output", "json"):
            return CommandResult(0, '{"clientBalance": 20}')
        if args[:2] == ("pod", "create"):
            self.owned_pods.add("pod-composed")
            return CommandResult(0, '{"id": "pod-composed"}')
        if args[:3] == ("pod", "get", "pod-composed"):
            if "pod-composed" not in self.owned_pods:
                return CommandResult(1, "", "pod not found")
            return CommandResult(
                0,
                json.dumps(
                    {
                        "id": "pod-composed",
                        "status": "RUNNING",
                        "ssh": {"ip": "203.0.113.20", "port": 22},
                        "imageName": self.bundle_image,
                        "costPerHr": 0.5,
                        "createdAt": "2026-07-21T12:00:00Z",
                    }
                ),
            )
        if args == ("remove", "pod", "pod-composed"):
            self.owned_pods.discard("pod-composed")
            return CommandResult(0, "")
        if args == ("pod", "list", "--output", "json"):
            return CommandResult(0, json.dumps([{"id": pod} for pod in self.owned_pods]))
        return CommandResult(0, "{}")


def test_transfer_failure_tears_down_owned_pod_without_masking_primary(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, baseline=False)
    transport = _OwnedPodTransport(bundle.environment.image_id or "")
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            image=bundle.environment.image_id or "",
            local_repos={"feedbax": tmp_path},
            remote_repos={"feedbax": "/workspace/feedbax"},
            primary_repo="feedbax",
            api_key="fixture-key",
        ),
        transport=transport,
    )
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    engine = StageEngine(bundle=bundle, driver=driver, store=store)

    transport.rsync_result = CommandResult(23, "", "forced sealed snapshot transfer failure")
    with pytest.raises(
        RunPodDriverError,
        match="forced sealed snapshot transfer failure",
    ):
        engine.run()

    failed = store.load()
    assert failed.stage(STAGE_PREFLIGHT).status == "completed"
    assert failed.stage(STAGE_PROVISION).status == "completed"
    realize = failed.stage("REALIZE_ENV")
    teardown = failed.stage("TEARDOWN")
    assert realize.status == "failed"
    assert "forced sealed snapshot transfer failure" in (realize.error or "")
    assert teardown.status == "completed"
    assert teardown.outputs["pod_absence"]["verified"] is True
    assert teardown.outputs["pod_absence"]["pod_id"] == "pod-composed"
    assert teardown.outputs["failure_log_collection"]["status"] == "failed"
    assert transport.owned_pods == set()


def test_provider_free_continuation_preflight_matches_two_realized_updates(
    tmp_path: Path,
) -> None:
    spec = _scheduled_continuation_spec(origin={"kind": "run_start"})
    continuation = CheckpointContinuationRequest(
        source_completed_batches=12_000,
        additional_batches=2,
    )
    payload = StandardSupervisedMethodPayload.model_validate(spec.method_payload.payload)

    contract = spec.worker_execution.method_contract.model_copy(deep=True)
    program = contract.phase_program.model_copy(deep=True)
    program = program.model_copy(
        update={
            "phases": [program.phases[0].model_copy(update={"max_steps": 2})],
        }
    )
    contract = contract.model_copy(
        update={
            "phase_program": program,
            "training_diagnostics": MethodTrainingDiagnosticsSpec(
                trace_schema_id="feedbax.tests.continuation_lr_trace",
                trace_schema_version="feedbax.tests.continuation_lr_trace.v1",
                measurement_basis="kernel_output",
                metric_payload_slot="train_loss",
                replica_axis="ensemble",
            ),
        }
    )

    @eqx.filter_jit
    def transition(model, optimizer_state, optimizer):
        updates, next_state = optimizer.update(
            jnp.full_like(model, 10.0), optimizer_state, model
        )
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

    effective_phase = validate_worker_contract(
        contract,
        update_kernels={"feedbax.training.standard_supervised.gradient_update": kernel},
    )
    spec = spec.model_copy(
        update={
            "training_config": spec.training_config.model_copy(update={"n_batches": 2}),
            "worker_execution": spec.worker_execution.model_copy(
                update={"method_contract": contract, "effective_phase": effective_phase}
            ),
            "checkpoint_progress": spec.checkpoint_progress.model_copy(
                update={"continuation": continuation}
            ),
        }
    )

    def optimizer_step(_payload, runtime):
        return int(runtime["optimizer"][-1].count)

    descriptor = replace(
        standard_supervised_method_descriptor(),
        contract_compiler=lambda _payload: contract,
        optimizer_step_extractor=optimizer_step,
        update_kernels_factory=lambda _payload: {
            "feedbax.training.standard_supervised.gradient_update": kernel
        },
    )
    registry = TrainingMethodRegistry()
    registry.register_descriptor(descriptor)
    checkpoint_root = tmp_path / "checkpoint-custody"
    template_optimizer = _write_mapped_schedule_checkpoint(spec, checkpoint_root)
    source_manifest = load_checkpoint_custody_documents(checkpoint_root).manifest.document
    source_spec, _source_program = authenticated_run_contract_source_projection(source_manifest)
    for run_spec in (source_spec, spec):
        run_spec._resolved_method = registry.resolve_execution(
            run_spec.method_ref,
            run_spec.method_payload,
            worker_execution=run_spec.worker_execution,
        )
        run_spec._resolved_method_cache_key = run_spec._method_resolution_cache_key()
    failures, evidence = compare_continuation_schedule_projections(
        source_run_spec=source_spec,
        target_run_spec=spec,
        source_manifest=source_manifest,
        continuation=continuation,
    )

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

    prepared_slots: dict[str, Any] = {
        "model": jnp.zeros((5,)),
        "optimizer": template_optimizer,
        "prng": jnp.ones((5, 2), dtype=jnp.uint32),
        "objective": jnp.zeros((5,)),
        "train_loss": jnp.zeros((5,)),
        "batch_counter": jnp.zeros((5,), dtype=jnp.int32),
    }
    preparation = _build_materialized_execution_preparation(
        request=ExecutionPreparationRequest(run_spec=spec),
        provider_identity="tests.provider_free_continuation",
        initial_slots=prepared_slots,
        kernel_context={},
        loss_service=None,
        resume_slot_transform=None,
        restored_schedule_context_binder=bind,
        coordinate_order=tuple(
            (AxisCoordinateSpec(axis="ensemble", index=index),) for index in range(5)
        ),
    )

    assert failures == []
    assert evidence["coordinates"] == [12_000, 12_001, 12_002]
    result = execute_training_run_spec(
        spec,
        preparation=preparation,
        registry=registry,
        manifest_root=tmp_path / "continuation-runs",
        checkpoint_root=checkpoint_root,
        resume=True,
    )

    assert result.status == "completed"
    assert result.diagnostics.segment_completed_batches == 2
    assert result.diagnostics.method_trace is not None
    records = result.diagnostics.method_trace.records
    realized = [record.value for record in records if record.replica_index == 0]
    learning_rate = next(
        comparison
        for comparison in evidence["comparisons"]
        if comparison["schedule_id"] == "feedbax.learning_rate"
    )
    expected = learning_rate["target_values"]
    assert realized[:2] == pytest.approx(expected[1:3])
