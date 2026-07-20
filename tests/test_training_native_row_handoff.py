from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from feedbax.contracts.checkpoints import CheckpointContinuationRequest
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.manifest import ParentRef
from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot
from feedbax.contracts.run_matrix import (
    RowLowererIdentity,
    TrainingRowLoweringResult,
    TrainingRunMatrixSpec,
)
from feedbax.contracts.spec_storage import (
    canonicalize_immutable_input_identities,
    training_run_execution_hash,
    training_spec_canonical_bytes,
)
from feedbax.contracts.training import (
    CheckpointProgressPolicySpec,
    LossTermSpec,
    ObjectiveSlotSpec,
    OptimizerSpec,
    StandardSupervisedMethodPayload,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.orchestration.assembly import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    AssemblyInputDeclaration,
    CompilerIdentity,
    RunAssemblyRequest,
    assemble_run_bundle,
)
from feedbax.orchestration.bundle import (
    BudgetPolicy,
    DeploymentPolicy,
    EnvironmentDeclaration,
    ImmutableInputArtifactRef,
    ImmutableInputIdentity,
    InputCustodySource,
    InputFormatIdentity,
    CheckpointCustodyArchiveMaterializer,
    ResolvedAssemblyInput,
    RunBundle,
    SchemaArtifactRef,
)
from feedbax.orchestration.conformance import (
    ConformanceRowArtifacts,
    check_checkpoint_cadence,
    check_completed_batches,
    check_environment_fingerprint,
    check_execution_identity,
    check_lr_trace,
    check_seeds,
)
from feedbax.orchestration.drivers.local import LocalDriverError, LocalOrchestrationDriver
from feedbax.orchestration.drivers.native_execution import (
    NativeExecutionContextError,
    inject_native_execution_context,
    native_resume_checkpoint_role,
    seed_authenticated_checkpoint,
)
from feedbax.orchestration.drivers.runpod import (
    build_launch_row_command,
    build_native_resume_seed_command,
)
from feedbax.orchestration.input_materialization import InputProviderRootBinding
from feedbax.orchestration.state import RowState, RunSetState
from feedbax.orchestration.stages import (
    OrchestrationStageError,
    _verify_collected_native_checkpoint_custody,
)
from feedbax.training.diagnostics import (
    LearningRateDiagnostic,
    NativeExecutionProducerContext,
    NativeTrainingDiagnosticsInput,
    ScheduleContextDiagnostic,
)
from feedbax.training.executor import execute_training_run_spec
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.training.checkpoint_custody import produce_checkpoint_custody_archive
from feedbax.training.spec_storage import (
    TRAINING_RUN_MATRIX_COMPILER_ID,
    TRAINING_RUN_MATRIX_COMPILER_VERSION,
    TrainingRunIdentityAdapter,
    register_training_run_matrix_compiler,
)


def _minimal_graph(gain: int) -> dict[str, Any]:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": gain},
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


def _training_spec(
    gain: int,
    *,
    completed_batches: int = 1,
    continuation: CheckpointContinuationRequest | None = None,
) -> TrainingRunSpec:
    method_payload = standard_supervised_method_payload()
    method_payload.payload = StandardSupervisedMethodPayload(
        optimizer=OptimizerSpec(
            type="adamw",
            params={"learning_rate": 0.001},
        )
    ).model_dump(mode="json")
    return TrainingRunSpec(
        graph={"inline": _minimal_graph(gain)},
        task=TaskSpec(type="ToyTask", params={"n_steps": 1}),
        training_config=TrainingConfig(n_batches=completed_batches, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(
                type="target_state",
                label="target",
                selector="port:gain.output",
                target_value=[0.0],
            )
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=method_payload,
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
        checkpoint_progress=CheckpointProgressPolicySpec(
            checkpoint_interval=1,
            continuation=continuation,
        ),
    )


def _initial_slots() -> dict[str, Any]:
    return {
        "model": jnp.array([0.0]),
        "optimizer": {"count": jnp.array([1.0])},
        "prng": jnp.array([0, 1], dtype=jnp.uint32),
        "batch_counter": jnp.array(0, dtype=jnp.int32),
    }


def _assemble_lowered_bundle(
    root: Path,
    *,
    gain: int,
    completed_batches: int = 1,
    continuation: CheckpointContinuationRequest | None = None,
    input_kind: str = "registered-artifact",
) -> RunBundle:
    matrix = TrainingRunMatrixSpec.model_validate(
        {
            "schema_id": "feedbax.spec.training_run_matrix",
            "schema_version": "feedbax.spec.training_run_matrix.v4",
            "name": "native row handoff",
            "base": {"kind": "inline", "inline": {"compact": {"gain": gain}}},
            "rows": [{"row_id": "science-row", "seed": 7, "overrides": []}],
        }
    )
    authored = matrix.model_dump(mode="json", exclude_none=True)
    authored_bytes = training_spec_canonical_bytes(authored)
    authored_path = root / "authored-matrix.json"
    authored_path.parent.mkdir(parents=True, exist_ok=True)
    authored_path.write_bytes(authored_bytes)

    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id="feedbax.spec.training_run_matrix",
            schema_version="feedbax.spec.training_run_matrix.v4",
            artifact_id=f"fixture:{root.name}:authored-matrix",
            sha256=hashlib.sha256(authored_bytes).hexdigest(),
            uri=str(authored_path),
        ),
        compiler=CompilerIdentity(
            compiler_id=TRAINING_RUN_MATRIX_COMPILER_ID,
            compiler_version=TRAINING_RUN_MATRIX_COMPILER_VERSION,
        ),
        deployment_policy=DeploymentPolicy(
            driver="local",
            venue="local",
            cloud_authorized=False,
            review_required=False,
            review_authorized=False,
        ),
        inputs=[
            AssemblyInputDeclaration(
                role="dataset",
                kind=input_kind,
                locator="dataset:toy:v1",
            )
        ],
        environment=EnvironmentDeclaration(python_version="3.13"),
        budget=BudgetPolicy(max_wall_clock_seconds=30),
        orchestration_root=str(root / "orchestration"),
    )

    def lower(authored_row: Any) -> TrainingRowLoweringResult:
        lowered_gain = int(authored_row.payload["compact"]["gain"])
        return TrainingRowLoweringResult(
            execution_payload=_training_spec(
                lowered_gain,
                completed_batches=completed_batches,
                continuation=continuation,
            ).model_dump(mode="json", exclude_none=True),
            lowerer_identities=[
                RowLowererIdentity(
                    lowerer_id="feedbax.tests.science-normalizer",
                    lowerer_version="feedbax.tests.science-normalizer.v1",
                ),
                RowLowererIdentity(
                    lowerer_id="feedbax.tests.native-row-lowerer",
                    lowerer_version="feedbax.tests.native-row-lowerer.v1",
                ),
            ],
        )

    registry = AssemblyCompilerRegistry()
    register_training_run_matrix_compiler(
        registry,
        allow_inline_base=True,
        row_lowerer=lower,
    )

    def resolve_input(declaration: AssemblyInputDeclaration) -> ResolvedAssemblyInput:
        media_type = "application/vnd.feedbax.training-checkpoint-custody.v1+tar+gzip"
        return ResolvedAssemblyInput.model_validate({
            "identity": {"role": declaration.role, "kind": declaration.kind, "identifier": declaration.locator, "digest": {"value": "d" * 64}},
            "custody": {"target_role": declaration.role, "provider": {}, "provider_binding": "checkpoint.inputs",
                        "artifact": {"artifact_id": "artifact://sha256/" + "d" * 64, "sha256": "d" * 64,
                                     "size_bytes": 123, "media_type": media_type, "storage_backend": "feedbax-local"},
                        "format": {"format_id": "feedbax.archive.training_checkpoint_custody", "format_version": "feedbax.archive.training_checkpoint_custody.v1", "media_type": media_type},
                        "materializer": {"expected_parent_ref": {"kind": "TrainingCheckpointTransactionManifest",
                            "id": "tx-toy-v1", "role": "training_checkpoint_custody", "uri": "transactions/tx-toy-v1/manifest.json",
                            "metadata": {"manifest_sha256": "a" * 64}},
                            "expected_transaction_root_sha256": "b" * 64}}})

    input_document = root / "resolved-input.json"
    input_document.write_text(resolve_input(request.inputs[0]).model_dump_json(), encoding="utf-8")
    request.inputs[0].locator = str(input_document)
    return assemble_run_bundle(
        request,
        run_set_id=f"run-set-{root.name}",
        context=AssemblyContext(
            custody_root=root / "assembly-custody",
            repo_root=root,
            materializer_commit="feedbax-test-commit",
            dependency_lock_digest="e" * 64,
            environment_digest="f" * 64,
        ),
        registry=registry,
    )


def _native_context(
    bundle: RunBundle,
    *,
    collection_root: Path,
    current_step: int = 0,
) -> NativeExecutionProducerContext:
    row = bundle.rows[0]
    assert row.execution.row_provenance is not None
    schedule_context = ScheduleContextDiagnostic(
        schedule_origin_step=0,
        current_step=current_step,
        optimizer_count_at_current_step=current_step,
    )
    return NativeExecutionProducerContext(
        execution=row.execution,
        environment_fingerprint="environment:integrated-row",
        collection_root=str(collection_root),
        diagnostics=NativeTrainingDiagnosticsInput(
            seeds=[7],
            lr_trace=[
                LearningRateDiagnostic(step=step, learning_rate=0.001)
                for step in range(current_step, current_step + 3)
            ],
            resume_context=schedule_context,
            optimizer_build_context=schedule_context,
        ),
    )


def _write_authenticated_checkpoint_tree(
    source: Path,
    resolved: ResolvedAssemblyInput,
    *,
    slot_bytes: bytes = b"checkpoint-slot",
    additional_slots: dict[str, bytes] | None = None,
) -> ResolvedAssemblyInput:
    """Write a minimal custody tree and bind its exact authority to one input."""

    materializer = resolved.custody.materializer
    parent = materializer.expected_parent_ref
    manifest_path = source / parent.uri
    slot_payloads = {"model": slot_bytes, **(additional_slots or {})}
    slot_records = []
    for name, data in slot_payloads.items():
        relative_path = f"slots/{name}.bin"
        slot_path = manifest_path.parent / relative_path
        slot_path.parent.mkdir(parents=True, exist_ok=True)
        slot_path.write_bytes(data)
        slot_records.append(
            {
                "slot": name,
                "relative_path": relative_path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    manifest = {
        "kind": "TrainingCheckpointTransactionManifest",
        "transaction_id": parent.id,
        "content_integrity_digest": {
            "transaction_root_sha256": materializer.expected_transaction_root_sha256
        },
        "slots": slot_records,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    (source / "latest.json").write_text(
        json.dumps(
            {
                "transaction_id": parent.id,
                "manifest_relative_path": parent.uri,
                "manifest_sha256": manifest_sha256,
                "transaction_root_sha256": materializer.expected_transaction_root_sha256,
            }
        ),
        encoding="utf-8",
    )
    authenticated_parent = parent.model_copy(
        update={"metadata": {**parent.metadata, "manifest_sha256": manifest_sha256}}
    )
    return resolved.model_copy(
        update={
            "custody": resolved.custody.model_copy(
                update={
                    "materializer": materializer.model_copy(
                        update={"expected_parent_ref": authenticated_parent}
                    )
                }
            )
        }
    )


def _without_resolved_inputs(bundle: RunBundle) -> RunBundle:
    """Return a valid no-input bundle for tests focused only on driver command binding."""
    payload = bundle.model_dump(mode="json")
    payload["resolved_inputs"] = []
    for row in payload["rows"]:
        execution = row["execution"]
        execution["immutable_inputs"] = []
        execution["execution_capsule"]["execution_hash"] = training_run_execution_hash(
            execution["resolved_snapshot"]["root_hash"],
            [],
        )
    return RunBundle.model_validate(payload)


def test_authored_row_changes_propagate_through_assembly_identity_and_custody(
    tmp_path: Path,
) -> None:
    first = _assemble_lowered_bundle(tmp_path / "first", gain=2)
    second = _assemble_lowered_bundle(tmp_path / "second", gain=3)
    first_row = first.rows[0]
    second_row = second.rows[0]
    first_provenance = first_row.execution.row_provenance
    second_provenance = second_row.execution.row_provenance

    assert first_provenance is not None
    assert second_provenance is not None
    assert first_provenance.lowerer_identities == [
        RowLowererIdentity(
            lowerer_id="feedbax.tests.science-normalizer",
            lowerer_version="feedbax.tests.science-normalizer.v1",
        ),
        RowLowererIdentity(
            lowerer_id="feedbax.tests.native-row-lowerer",
            lowerer_version="feedbax.tests.native-row-lowerer.v1",
        ),
    ]
    assert first_provenance.axis_coordinates["run_id"] == (
        first_provenance.planned_run_id
    )
    assert first_provenance.authored_payload_hash != (
        second_provenance.authored_payload_hash
    )
    assert first_provenance.planned_run_id != second_provenance.planned_run_id
    assert first_row.execution.payload.sha256 != second_row.execution.payload.sha256
    assert first_row.execution.resolved_snapshot.root_hash != (
        second_row.execution.resolved_snapshot.root_hash
    )
    assert first_row.execution.execution_capsule.execution_hash != (
        second_row.execution.execution_capsule.execution_hash
    )

    lowered_payload = json.loads(Path(first_row.execution.payload.uri).read_text(encoding="utf-8"))
    assert lowered_payload["graph"]["inline"]["nodes"]["gain"]["params"]["gain"] == 2
    assert "compact" not in lowered_payload
    assert [item.identifier for item in first_row.execution.immutable_inputs] == [
        "dataset:toy:v1"
    ]
    assert first.resolved_inputs[0].identity == first_row.execution.immutable_inputs[0]

    snapshot = json.loads(
        Path(first_row.execution.resolved_snapshot.uri).read_text(encoding="utf-8")
    )
    decoded = decode_resolved_snapshot(snapshot)
    assert decoded["payload"] == lowered_payload
    assert decoded["planned_run_id"] == first_provenance.planned_run_id
    assert decoded["row_provenance"] == first_provenance.model_dump(
        mode="json", exclude_none=True
    )


def test_native_row_outputs_resume_and_collect_from_the_assembled_contract(
    tmp_path: Path,
) -> None:
    bundle = _assemble_lowered_bundle(tmp_path / "source", gain=2)
    row = bundle.rows[0]
    provenance = row.execution.row_provenance
    assert provenance is not None
    payload = json.loads(Path(row.execution.payload.uri).read_text(encoding="utf-8"))
    collection_root = bundle.run_set_dir / "rows" / row.row_id
    checkpoint_root = collection_root / "checkpoints"
    context = _native_context(bundle, collection_root=collection_root)

    result = execute_training_run_spec(
        payload,
        initial_slots=_initial_slots(),
        manifest_root=tmp_path / "manifest-root",
        checkpoint_root=checkpoint_root,
        execution_context=context,
    )

    assert result.run_id == provenance.planned_run_id
    assert result.manifest.id == provenance.planned_run_id
    assert result.manifest.intent_hash == row.execution.authored_intent.intent_hash
    assert result.manifest.execution_hash == row.execution.execution_capsule.execution_hash
    assert result.manifest.resolved_semantics_root_hash == (
        row.execution.resolved_snapshot.root_hash
    )
    assert result.manifest.input_data_identities == [
        identity.model_dump(mode="json", exclude_none=True)
        for identity in row.execution.immutable_inputs
    ]
    assert result.manifest.metadata["environment_fingerprint"] == (
        "environment:integrated-row"
    )
    assert result.manifest.metadata["training_row_provenance"] == (
        provenance.model_dump(mode="json", exclude_none=True)
    )
    assert result.manifest_path == collection_root / "manifest.json"
    assert result.diagnostics_path == collection_root / "training-diagnostics.json"
    assert not (tmp_path / "manifest-root" / "manifests" / "training_runs").exists()
    assert list(collection_root.glob("*manifest*.json")) == [result.manifest_path]

    diagnostics = result.diagnostics.model_dump(mode="json", exclude_none=True)
    assert diagnostics["schema_version"] == "feedbax.manifest.training_diagnostics.v2"
    assert diagnostics["manifest_id"] == result.manifest.id
    assert diagnostics["completed_batches"] == 1
    assert diagnostics["segment_completed_batches"] == 1
    assert diagnostics["seeds"] == [7]
    assert diagnostics["checkpoint_coordinates"] == [1]
    assert [sample["step"] for sample in diagnostics["lr_trace"]] == [0, 1, 2]
    assert result.checkpoint_writes[0].manifest.run_id == provenance.planned_run_id

    conformance_row = ConformanceRowArtifacts(
        row_id=row.row_id,
        execution=row.execution,
        execution_identity_adapter=TrainingRunIdentityAdapter(),
        manifest_path=result.manifest_path,
        training_diagnostics=diagnostics,
        recorded_environment_fingerprint=context.environment_fingerprint,
        bundle_row_spec={
            "training_config": {"n_batches": 1},
            "checkpoint_progress": {"checkpoint_interval": 1},
            "optimizer": {"type": "adamw", "params": {"learning_rate": 0.001}},
        },
    )
    assert check_execution_identity(conformance_row).status == "pass"
    assert check_environment_fingerprint(conformance_row).status == "pass"
    assert check_completed_batches(conformance_row).status == "pass"
    seed_check = check_seeds(conformance_row)
    assert seed_check.status == "pass"
    assert seed_check.expected == 7
    assert seed_check.observed == [7]
    assert check_checkpoint_cadence(conformance_row).status == "pass"
    assert check_lr_trace(conformance_row).status == "pass"

    conflicting_declarations = replace(
        conformance_row,
        bundle_row_spec={**conformance_row.bundle_row_spec, "seeds": [8]},
    )
    seed_check = check_seeds(conflicting_declarations)
    assert seed_check.status == "fail"
    assert seed_check.expected == {"bundle_row_spec": [8]}
    assert seed_check.observed == {"execution.row_provenance.seed": 7}
    assert seed_check.detail == (
        "declared seeds disagree between bundle row and execution provenance"
    )

    mismatched_seed = replace(
        conformance_row,
        training_diagnostics={**diagnostics, "seeds": [8]},
    )
    seed_check = check_seeds(mismatched_seed)
    assert seed_check.status == "fail"
    assert seed_check.expected == 7
    assert seed_check.observed == [8]

    missing_seed = replace(
        conformance_row,
        execution=row.execution.model_copy(update={"row_provenance": None}),
    )
    seed_check = check_seeds(missing_seed)
    assert seed_check.status == "fail"
    assert seed_check.detail == "missing required input: bundle_row_spec.seeds"

    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=[])
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={row.row_id: RowState(status="completed")},
    )
    collected = driver.collect(bundle, row, state)
    assert set(collected) == {
        "manifest.json",
        "training-diagnostics.json",
        "checkpoints",
    }
    assert json.loads(Path(collected["manifest.json"]).read_text(encoding="utf-8"))[
        "id"
    ] == result.manifest.id
    assert json.loads(
        Path(collected["training-diagnostics.json"]).read_text(encoding="utf-8")
    )["manifest_id"] == result.manifest.id
    collected_checkpoint_root = Path(collected["checkpoints"])
    assert (collected_checkpoint_root / "latest.json").is_file()
    assert (collected_checkpoint_root / "transactions").is_dir()
    verification = _verify_collected_native_checkpoint_custody(row, collected)
    assert verification["transaction_id"] == result.checkpoint_writes[-1].manifest.transaction_id
    assert set(verification["slot_names"]) == {
        "batch_counter",
        "model",
        "optimizer",
        "prng",
    }

    collected_manifest_path = Path(collected["manifest.json"])
    collected_manifest_bytes = collected_manifest_path.read_bytes()
    mismatched_manifest = json.loads(collected_manifest_bytes)
    mismatched_manifest["checkpoint_custody"][-1]["id"] = "tx-stale-input"
    collected_manifest_path.write_text(
        json.dumps(mismatched_manifest, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(
        OrchestrationStageError,
        match="not the terminal training manifest authority",
    ):
        _verify_collected_native_checkpoint_custody(row, collected)
    collected_manifest_path.write_bytes(collected_manifest_bytes)

    continuation = CheckpointContinuationRequest(
        source_completed_batches=1,
        additional_batches=1,
        self_contained=False,
    )
    resumed_bundle = _assemble_lowered_bundle(
        tmp_path / "resumed",
        gain=2,
        completed_batches=2,
        continuation=continuation,
    )
    resumed_row = resumed_bundle.rows[0]
    resumed_provenance = resumed_row.execution.row_provenance
    assert resumed_provenance is not None
    assert resumed_row.launch.command == [
        "python",
        "-m",
        "feedbax",
        "execute-training-run-spec",
        "--resume",
    ]
    assert resumed_row.launch.payload_routing == {
        "kind": "registered-execution-payload",
        "spec": "execution.payload",
        "manifest_root": "row-local",
        "checkpoint_root": "row-local",
    }
    resumed_payload = json.loads(
        Path(resumed_row.execution.payload.uri).read_text(encoding="utf-8")
    )
    resumed_collection_root = resumed_bundle.run_set_dir / "rows" / resumed_row.row_id
    resumed = execute_training_run_spec(
        resumed_payload,
        initial_slots=_initial_slots(),
        manifest_root=tmp_path / "resumed-manifest-root",
        checkpoint_root=checkpoint_root,
        resume=True,
        execution_context=_native_context(
            resumed_bundle,
            collection_root=resumed_collection_root,
            current_step=1,
        ),
    )

    assert resumed.run_id == resumed_provenance.planned_run_id
    assert resumed.manifest.id == resumed_provenance.planned_run_id
    assert resumed.manifest.id != result.manifest.id
    assert resumed.diagnostics.completed_batches == 2
    assert resumed.diagnostics.segment_completed_batches == 1
    assert resumed.diagnostics.cumulative_completed_batches == 2
    assert resumed.diagnostics.checkpoint_coordinates == [1]
    assert resumed.diagnostics.checkpoint_transactions[0].cumulative_completed_batches == 2
    assert resumed.checkpoint_writes[0].manifest.parent_lineage
    assert resumed.checkpoint_writes[0].manifest.run_id == resumed_provenance.planned_run_id
    assert resumed.manifest_path == resumed_collection_root / "manifest.json"
    assert resumed.diagnostics_path == resumed_collection_root / "training-diagnostics.json"


def test_local_and_runpod_drivers_inject_the_canonical_native_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from feedbax.orchestration.drivers import local as local_driver_module
    from feedbax.orchestration.drivers import runpod as runpod_driver_module

    bundle = _without_resolved_inputs(
        _assemble_lowered_bundle(tmp_path / "drivers", gain=2)
    )
    row = bundle.rows[0]
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        environment_fingerprint="environment:driver-route",
        rows={row.row_id: RowState()},
    )
    local_driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=[])
    local_driver.provision(bundle, state)
    local_driver.stage_inputs(bundle, state)
    local_capture: dict[str, Any] = {}

    class _Process:
        pid = 12345

        def poll(self) -> None:
            return None

    def capture_popen(command: list[str], **_kwargs: Any) -> _Process:
        local_capture["command"] = command
        return _Process()

    monkeypatch.setattr(local_driver_module.subprocess, "Popen", capture_popen)
    local_driver.launch_row(bundle, row, state)
    local_command = local_capture["command"]
    assert local_command[-2] == "--execution-context-json"
    local_context = json.loads(local_command[-1])
    expected_local_execution = row.execution.model_copy(
        update={
            "payload": row.execution.payload.model_copy(
                update={"uri": str(bundle.run_set_dir / "inputs" / f"{row.row_id}.json")}
            )
        }
    )
    assert local_context["execution"] == expected_local_execution.model_dump(
        mode="json", exclude_none=True
    )
    assert local_context["execution"]["row_provenance"] == (
        row.execution.row_provenance.model_dump(
            mode="json", exclude_none=True
        )
    )
    assert local_context["environment_fingerprint"] == "environment:driver-route"
    assert local_context["collection_root"] == str(
        bundle.run_set_dir / "rows" / row.row_id
    )

    runpod_capture: dict[str, Any] = {}

    def capture_runpod_context(command: list[str], **kwargs: Any) -> list[str]:
        injected = inject_native_execution_context(command, **kwargs)
        runpod_capture["command"] = injected
        runpod_capture.update(kwargs)
        return injected

    monkeypatch.setattr(
        runpod_driver_module,
        "inject_native_execution_context",
        capture_runpod_context,
    )
    build_launch_row_command(
        bundle=bundle,
        row=row,
        remote_run_dir="/remote/run-set",
        remote_sentinel_dir="/remote/sentinels",
        workdir="/remote/feedbax",
        env_fingerprint="environment:driver-route",
        jax_cache_dir="/remote/jax-cache",
    )
    runpod_command = runpod_capture["command"]
    assert runpod_command[-2] == "--execution-context-json"
    runpod_context = json.loads(runpod_command[-1])
    expected_runpod_execution = row.execution.model_copy(
        update={
            "payload": row.execution.payload.model_copy(
                update={"uri": "/remote/run-set/inputs/science-row.json"}
            )
        }
    )
    assert runpod_context["execution"] == expected_runpod_execution.model_dump(
        mode="json", exclude_none=True
    )
    assert runpod_context["execution"]["row_provenance"] == (
        row.execution.row_provenance.model_dump(
            mode="json", exclude_none=True
        )
    )
    assert runpod_context["environment_fingerprint"] == "environment:driver-route"
    assert runpod_context["collection_root"] == "/remote/run-set/rows/science-row"


def test_local_driver_executes_compiled_native_row_subprocess(tmp_path: Path) -> None:
    bundle = _without_resolved_inputs(
        _assemble_lowered_bundle(tmp_path / "subprocess", gain=2)
    )
    row = bundle.rows[0]
    slots_path = tmp_path / "initial-slots.json"
    slots_path.write_text(
        json.dumps(
            {
                "model": 0,
                "optimizer": {"count": 1},
                "prng": [0, 1],
                "batch_counter": 0,
            }
        ),
        encoding="utf-8",
    )
    row = row.model_copy(
        update={
            "launch": row.launch.model_copy(
                update={
                    "command": [
                        *row.launch.command,
                        "--initial-slots",
                        str(slots_path),
                        "--no-progress",
                    ]
                }
            )
        }
    )
    bundle = bundle.model_copy(update={"rows": [row]})
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        environment_fingerprint="environment:subprocess-route",
        rows={row.row_id: RowState()},
    )
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=[])
    driver.provision(bundle, state)
    staged = driver.stage_inputs(bundle, state)

    launched = driver.launch_row(bundle, row, state)
    for _ in range(400):
        probe = driver.probe(bundle, row, state)
        if probe.status != "running":
            break
        time.sleep(0.05)

    assert probe.status == "completed"
    command = launched["command"]
    command_index = command.index("execute-training-run-spec")
    staged_payload = Path(staged["payloads"][0]["target"])
    assert command[command_index + 1] == str(staged_payload)
    assert command[command.index("--checkpoint-root") + 1] == str(
        bundle.run_set_dir / "rows" / row.row_id / "checkpoints"
    )
    assert command[command.index("--run-id") + 1] == (
        row.execution.row_provenance.planned_run_id
    )
    row_dir = bundle.run_set_dir / "rows" / row.row_id
    assert (row_dir / "manifest.json").is_file()
    assert (row_dir / "training-diagnostics.json").is_file()


def test_native_resume_checkpoint_role_requires_one_authenticated_custody_input(
    tmp_path: Path,
) -> None:
    continuation = CheckpointContinuationRequest(
        source_completed_batches=1,
        additional_batches=1,
        self_contained=False,
    )
    bundle = _assemble_lowered_bundle(
        tmp_path / "resume-role",
        gain=2,
        completed_batches=2,
        continuation=continuation,
        input_kind="checkpoint-custody-archive",
    )
    row = bundle.rows[0]

    assert native_resume_checkpoint_role(bundle, row) == "dataset"

    no_checkpoint_bundle = _assemble_lowered_bundle(
        tmp_path / "missing-role",
        gain=2,
        completed_batches=2,
        continuation=continuation,
    )
    with pytest.raises(NativeExecutionContextError, match="exactly one immutable"):
        native_resume_checkpoint_role(no_checkpoint_bundle, no_checkpoint_bundle.rows[0])

    ambiguous = bundle.model_copy(
        update={"resolved_inputs": [bundle.resolved_inputs[0], bundle.resolved_inputs[0]]}
    )
    with pytest.raises(NativeExecutionContextError, match="exactly one resolved custody"):
        native_resume_checkpoint_role(ambiguous, row)


def test_local_native_resume_seeds_fresh_checkpoint_root_without_mutating_source(
    tmp_path: Path,
) -> None:
    from feedbax.orchestration.drivers import local as local_driver_module

    continuation = CheckpointContinuationRequest(
        source_completed_batches=1,
        additional_batches=1,
        self_contained=False,
    )
    bundle = _assemble_lowered_bundle(
        tmp_path / "local-resume",
        gain=2,
        completed_batches=2,
        continuation=continuation,
        input_kind="checkpoint-custody-archive",
    )
    row = bundle.rows[0]
    source = bundle.run_set_dir / "inputs" / "dataset"
    resolved = _write_authenticated_checkpoint_tree(source, bundle.resolved_inputs[0])
    row_dir = bundle.run_set_dir / "rows" / row.row_id
    row_dir.mkdir(parents=True)

    local_driver_module._seed_native_resume_checkpoint_root(source, row_dir, resolved)

    target = row_dir / "checkpoints"
    assert (target / "latest.json").read_text(encoding="utf-8") == (
        source / "latest.json"
    ).read_text(encoding="utf-8")
    assert (source / resolved.custody.materializer.expected_parent_ref.uri).is_file()
    assert not (target.parent / ".checkpoint-seed-attempt").exists()

    with pytest.raises(LocalDriverError, match="checkpoint target already exists"):
        local_driver_module._seed_native_resume_checkpoint_root(source, target.parent, resolved)

    target.rename(target.parent / "published-checkpoints")
    (target.parent / ".checkpoint-seed-attempt").mkdir()
    with pytest.raises(LocalDriverError, match="checkpoint attempt already exists"):
        local_driver_module._seed_native_resume_checkpoint_root(source, target.parent, resolved)


def test_runpod_native_resume_seeds_before_started_sentinel(tmp_path: Path) -> None:
    continuation = CheckpointContinuationRequest(
        source_completed_batches=1,
        additional_batches=1,
        self_contained=False,
    )
    bundle = _assemble_lowered_bundle(
        tmp_path / "runpod-resume",
        gain=2,
        completed_batches=2,
        continuation=continuation,
        input_kind="checkpoint-custody-archive",
    )
    row = bundle.rows[0]

    command = build_launch_row_command(
        bundle=bundle,
        row=row,
        remote_run_dir="/remote/run set",
        remote_sentinel_dir="/remote/run set/sentinels",
        workdir="/remote/feedbax",
        env_fingerprint="environment:resume-seed",
        jax_cache_dir="/remote/jax cache",
    )

    assert "/remote/run set/inputs/dataset" in command
    assert "/remote/run set/rows/science-row/.checkpoint-seed-attempt" in command
    assert "/remote/run set/rows/science-row/checkpoints" in command
    assert command.index(".checkpoint-seed-attempt") < command.rindex(".started")


@pytest.mark.parametrize("protocol", ["local", "remote"])
def test_secure_checkpoint_clone_rejects_symlink_entries(
    tmp_path: Path,
    protocol: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    (source / "latest.json").symlink_to(outside)
    attempt = tmp_path / f"{protocol}-attempt"
    target = tmp_path / f"{protocol}-target"
    bundle = _assemble_lowered_bundle(
        tmp_path / f"{protocol}-symlink-bundle",
        gain=2,
        completed_batches=2,
        continuation=CheckpointContinuationRequest(
            source_completed_batches=1, additional_batches=1, self_contained=False
        ),
        input_kind="checkpoint-custody-archive",
    )
    resolved = bundle.resolved_inputs[0]

    if protocol == "local":
        with pytest.raises(NativeExecutionContextError, match="authenticated checkpoint seed"):
            seed_authenticated_checkpoint(source, attempt, target, resolved)
    else:
        result = subprocess.run(
            ["bash", "-lc", build_native_resume_seed_command(
                str(source), str(attempt), str(target), resolved
            )],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
    assert not target.exists()


def test_remote_checkpoint_seed_protocol_clones_and_publishes(tmp_path: Path) -> None:
    source = tmp_path / "source with spaces"
    bundle = _assemble_lowered_bundle(
        tmp_path / "remote-seed-bundle",
        gain=2,
        completed_batches=2,
        continuation=CheckpointContinuationRequest(
            source_completed_batches=1, additional_batches=1, self_contained=False
        ),
        input_kind="checkpoint-custody-archive",
    )
    resolved = _write_authenticated_checkpoint_tree(source, bundle.resolved_inputs[0])
    attempt = tmp_path / "private attempt"
    target = tmp_path / "fresh checkpoints"

    subprocess.run(
        [
            "bash",
            "-lc",
            build_native_resume_seed_command(str(source), str(attempt), str(target), resolved),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (target / "latest.json").read_bytes() == (source / "latest.json").read_bytes()
    assert not attempt.exists()


def _invoke_checkpoint_seed_protocol(
    protocol: str,
    source: Path,
    attempt: Path,
    target: Path,
    resolved: ResolvedAssemblyInput,
) -> None:
    if protocol == "local":
        seed_authenticated_checkpoint(source, attempt, target, resolved)
        return
    subprocess.run(
        [
            "bash",
            "-lc",
            build_native_resume_seed_command(str(source), str(attempt), str(target), resolved),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("protocol", ["local", "remote"])
@pytest.mark.parametrize("replacement", [False, True], ids=["tampered", "replaced"])
def test_checkpoint_seed_reauthenticates_stable_post_stage_source(
    tmp_path: Path,
    protocol: str,
    replacement: bool,
) -> None:
    bundle = _assemble_lowered_bundle(
        tmp_path / f"{protocol}-post-stage-bundle-{replacement}",
        gain=2,
        completed_batches=2,
        continuation=CheckpointContinuationRequest(
            source_completed_batches=1, additional_batches=1, self_contained=False
        ),
        input_kind="checkpoint-custody-archive",
    )
    source = tmp_path / f"{protocol}-post-stage-source-{replacement}"
    resolved = _write_authenticated_checkpoint_tree(source, bundle.resolved_inputs[0])
    parent = resolved.custody.materializer.expected_parent_ref
    slot = source / Path(parent.uri).parent / "slots" / "model.bin"
    if replacement:
        source.rename(source.with_name(source.name + "-authenticated"))
        _write_authenticated_checkpoint_tree(
            source, bundle.resolved_inputs[0], slot_bytes=b"replacement-slot"
        )
    else:
        slot.write_bytes(b"tampered-slot!")
    attempt = tmp_path / f"{protocol}-post-stage-attempt-{replacement}"
    target = tmp_path / f"{protocol}-post-stage-target-{replacement}"

    with pytest.raises((NativeExecutionContextError, subprocess.CalledProcessError)):
        _invoke_checkpoint_seed_protocol(protocol, source, attempt, target, resolved)

    assert not target.exists()


@pytest.mark.parametrize("protocol", ["local", "remote"])
def test_checkpoint_seed_rejects_attempt_path_swap(
    tmp_path: Path,
    protocol: str,
) -> None:
    bundle = _assemble_lowered_bundle(
        tmp_path / f"{protocol}-attempt-swap-bundle",
        gain=2,
        completed_batches=2,
        continuation=CheckpointContinuationRequest(
            source_completed_batches=1, additional_batches=1, self_contained=False
        ),
        input_kind="checkpoint-custody-archive",
    )
    source = tmp_path / f"{protocol}-attempt-swap-source"
    resolved = _write_authenticated_checkpoint_tree(
        source,
        bundle.resolved_inputs[0],
        slot_bytes=b"x" * (64 * 1024 * 1024),
    )
    attempt = tmp_path / f"{protocol}-attempt-swap"
    target = tmp_path / f"{protocol}-attempt-swap-target"
    displaced = tmp_path / f"{protocol}-authenticated-attempt"
    swapped = threading.Event()

    def swap_attempt() -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (attempt / "latest.json").is_file():
                attempt.rename(displaced)
                attempt.mkdir()
                (attempt / "attacker.txt").write_text("replacement", encoding="utf-8")
                swapped.set()
                return
            time.sleep(0.001)

    thread = threading.Thread(target=swap_attempt)
    thread.start()
    try:
        with pytest.raises((NativeExecutionContextError, subprocess.CalledProcessError)):
            _invoke_checkpoint_seed_protocol(protocol, source, attempt, target, resolved)
    finally:
        thread.join()

    assert swapped.is_set()
    assert not target.exists()
    assert (attempt / "attacker.txt").is_file()


@pytest.mark.parametrize("protocol", ["local", "remote"])
def test_checkpoint_seed_rejects_governed_leaf_replacement_after_digest_read(
    tmp_path: Path,
    protocol: str,
) -> None:
    bundle = _assemble_lowered_bundle(
        tmp_path / f"{protocol}-leaf-swap-bundle",
        gain=2,
        completed_batches=2,
        continuation=CheckpointContinuationRequest(
            source_completed_batches=1, additional_batches=1, self_contained=False
        ),
        input_kind="checkpoint-custody-archive",
    )
    source = tmp_path / f"{protocol}-leaf-swap-source"
    resolved = _write_authenticated_checkpoint_tree(
        source,
        bundle.resolved_inputs[0],
        additional_slots={"zz_ballast": b"z" * (64 * 1024 * 1024)},
    )
    attempt = tmp_path / f"{protocol}-leaf-swap-attempt"
    target = tmp_path / f"{protocol}-leaf-swap-target"
    manifest_uri = resolved.custody.materializer.expected_parent_ref.uri
    governed_leaf = attempt / Path(manifest_uri).parent / "slots" / "model.bin"
    attacker_leaf = tmp_path / f"{protocol}-attacker-model.bin"
    attacker_leaf.write_bytes(b"attacker-slot!!")
    replaced = threading.Event()

    def replace_after_read() -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not governed_leaf.is_file():
            time.sleep(0.001)
        if not governed_leaf.is_file():
            return
        initial_atime = governed_leaf.stat().st_atime_ns
        while time.monotonic() < deadline:
            if governed_leaf.stat().st_atime_ns != initial_atime:
                os.replace(attacker_leaf, governed_leaf)
                replaced.set()
                return
            time.sleep(0.001)

    thread = threading.Thread(target=replace_after_read)
    thread.start()
    try:
        with pytest.raises((NativeExecutionContextError, subprocess.CalledProcessError)):
            _invoke_checkpoint_seed_protocol(protocol, source, attempt, target, resolved)
    finally:
        thread.join()

    assert replaced.is_set()
    assert not target.exists()


@pytest.mark.parametrize("protocol", ["local", "remote"])
def test_secure_checkpoint_clone_rejects_concurrent_source_mutation(
    tmp_path: Path,
    protocol: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    changing = source / "checkpoint.bin"
    changing.write_bytes(b"0" * (32 * 1024 * 1024))
    attempt = tmp_path / f"{protocol}-attempt"
    target = tmp_path / f"{protocol}-target"
    bundle = _assemble_lowered_bundle(
        tmp_path / f"{protocol}-mutation-bundle",
        gain=2,
        completed_batches=2,
        continuation=CheckpointContinuationRequest(
            source_completed_batches=1, additional_batches=1, self_contained=False
        ),
        input_kind="checkpoint-custody-archive",
    )
    resolved = bundle.resolved_inputs[0]
    stop = threading.Event()
    mutations = 0

    def mutate() -> None:
        nonlocal mutations
        value = 0
        while not stop.is_set():
            with changing.open("r+b", buffering=0) as handle:
                handle.write(bytes([value]))
                os.fsync(handle.fileno())
            value = 1 - value
            mutations += 1

    thread = threading.Thread(target=mutate)
    thread.start()
    try:
        if protocol == "local":
            with pytest.raises(NativeExecutionContextError, match="authenticated checkpoint seed"):
                seed_authenticated_checkpoint(source, attempt, target, resolved)
        else:
            result = subprocess.run(
                ["bash", "-lc", build_native_resume_seed_command(
                    str(source), str(attempt), str(target), resolved
                )],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
    finally:
        stop.set()
        thread.join()
    assert mutations > 0
    assert not target.exists()


def test_local_driver_executes_authenticated_custody_continuation_with_parent_lineage(
    tmp_path: Path,
) -> None:
    parent_bundle = _without_resolved_inputs(
        _assemble_lowered_bundle(tmp_path / "authenticated-parent", gain=2)
    )
    parent_row = parent_bundle.rows[0]
    parent_payload = json.loads(
        Path(parent_row.execution.payload.uri).read_text(encoding="utf-8")
    )
    parent_result = execute_training_run_spec(
        parent_payload,
        initial_slots=_initial_slots(),
        manifest_root=tmp_path / "parent-manifests",
        checkpoint_root=tmp_path / "parent-checkpoints",
        execution_context=_native_context(
            parent_bundle,
            collection_root=parent_bundle.run_set_dir / "rows" / parent_row.row_id,
        ),
    )
    parent_write = parent_result.checkpoint_writes[0]
    parent_ref = ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id=parent_write.manifest.transaction_id,
        role="training_checkpoint_custody",
        uri=parent_write.manifest_path.relative_to(parent_write.root).as_posix(),
        metadata={"manifest_sha256": parent_write.latest_pointer.manifest_sha256},
    )
    provider_root = tmp_path / "checkpoint-provider"
    artifact = produce_checkpoint_custody_archive(
        parent_ref,
        allowed_root=parent_write.root,
        artifact_provider=ImmutableArtifactBlobProvider(provider_root),
    ).artifact_ref
    identity = ImmutableInputIdentity(
        role="checkpoint",
        kind="checkpoint-custody-archive",
        identifier=parent_ref.id,
        digest={"value": artifact.sha256},
    )
    resolved = ResolvedAssemblyInput(
        identity=identity,
        custody=InputCustodySource(
            target_role="checkpoint",
            provider=ImmutableArtifactBlobProviderSpec(),
            provider_binding="checkpoint.inputs",
            artifact=ImmutableInputArtifactRef(
                **artifact.model_dump(
                    include={
                        "artifact_id",
                        "sha256",
                        "size_bytes",
                        "media_type",
                        "storage_backend",
                    }
                )
            ),
            format=InputFormatIdentity(
                format_id="feedbax.archive.training_checkpoint_custody",
                format_version="feedbax.archive.training_checkpoint_custody.v1",
                media_type="application/vnd.feedbax.training-checkpoint-custody.v1+tar+gzip",
            ),
            materializer=CheckpointCustodyArchiveMaterializer(
                expected_parent_ref=parent_ref,
                expected_transaction_root_sha256=(
                    parent_write.manifest.content_integrity_digest.transaction_root_sha256
                ),
            ),
        ),
    )
    continuation = CheckpointContinuationRequest(
        source_completed_batches=1,
        additional_batches=1,
        self_contained=False,
    )
    resumed_bundle = _assemble_lowered_bundle(
        tmp_path / "authenticated-child",
        gain=2,
        completed_batches=2,
        continuation=continuation,
        input_kind="checkpoint-custody-archive",
    )
    payload = resumed_bundle.model_dump(mode="json")
    canonical = canonicalize_immutable_input_identities([identity])
    execution = payload["rows"][0]["execution"]
    execution["immutable_inputs"] = canonical
    execution["execution_capsule"]["execution_hash"] = training_run_execution_hash(
        execution["resolved_snapshot"]["root_hash"], canonical
    )
    payload["resolved_inputs"] = [resolved.model_dump(mode="json")]
    resumed_bundle = RunBundle.model_validate(payload)
    resumed_row = resumed_bundle.rows[0]
    state = RunSetState(
        run_set_id=resumed_bundle.run_set_id,
        environment_fingerprint="environment:authenticated-resume",
        rows={resumed_row.row_id: RowState()},
    )
    driver = LocalOrchestrationDriver(
        cwd=tmp_path,
        freeze_lines=[],
        input_provider_bindings=[
            InputProviderRootBinding("checkpoint.inputs", provider_root)
        ],
    )
    driver.provision(resumed_bundle, state)
    driver.stage_inputs(resumed_bundle, state)
    row_dir = resumed_bundle.run_set_dir / "rows" / resumed_row.row_id
    row_dir.mkdir(parents=True)
    from feedbax.orchestration.drivers import local as local_driver_module

    local_driver_module._seed_native_resume_checkpoint_root(
        resumed_bundle.run_set_dir / "inputs" / "checkpoint",
        row_dir,
        resolved,
    )
    resumed_payload = json.loads(
        Path(resumed_row.execution.payload.uri).read_text(encoding="utf-8")
    )
    resumed_result = execute_training_run_spec(
        resumed_payload,
        initial_slots=_initial_slots(),
        manifest_root=tmp_path / "child-manifests",
        checkpoint_root=row_dir / "checkpoints",
        resume=True,
        execution_context=_native_context(
            resumed_bundle,
            collection_root=row_dir,
            current_step=1,
        ),
    )

    assert resumed_result.checkpoint_writes[0].manifest.parent_lineage[0].transaction_id == (
        parent_ref.id
    )
