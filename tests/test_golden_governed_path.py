"""One public, provider-free acceptance fixture for the governed launch path."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.checkpoints import CheckpointContinuationRequest
from feedbax.contracts.manifest import ParentRef
from feedbax.contracts.run_matrix import (
    TRAINING_ROW_LOWERER_REF_FIELD,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V3,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V4,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.shadow_launch import (
    SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID,
    SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION,
)
from feedbax.contracts.spec_storage import training_spec_sha256
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
    AssemblyInputDeclaration,
    CompilerIdentity,
    RunAssemblyRequest,
)
from feedbax.orchestration.bundle import (
    BudgetPolicy,
    CheckpointCustodyArchiveMaterializer,
    DeploymentPolicy,
    EnvironmentDeclaration,
    ImmutableInputArtifactRef,
    ImmutableInputIdentity,
    InputCustodySource,
    InputFormatIdentity,
    LaunchPolicy,
    ResolvedAssemblyInput,
    RunBundle,
    SchemaArtifactRef,
    canonical_run_bundle_sha256,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.training.checkpoint_custody import produce_checkpoint_custody_archive
from feedbax.training.spec_storage import (
    TRAINING_RUN_MATRIX_COMPILER_ID,
    TRAINING_RUN_MATRIX_COMPILER_VERSION,
)


METHOD_REF = "tests.golden/governed/v1"
METHOD_SCHEMA = "tests.golden.method"
METHOD_VERSION = "tests.golden.method.v1"
AUTHORED_SCHEMA = "tests.golden.authored_row"
AUTHORED_VERSION = "tests.golden.authored_row.v1"
LOWERER_ID = "tests.golden.lowerer"
LOWERER_VERSION = "tests.golden.lowerer.v1"


def _run_spec(*, batches: int, continuation: CheckpointContinuationRequest | None = None) -> dict:
    method_payload = StandardSupervisedMethodPayload(
        optimizer=OptimizerSpec(type="adamw", params={"learning_rate": 0.001})
    )
    spec = TrainingRunSpec(
        graph={
            "inline": {
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
        },
        task=TaskSpec(type="ToyTask", params={"n_steps": 1}),
        training_config=TrainingConfig(n_batches=batches, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(
                type="target_state", label="target", selector="port:gain.output", target_value=[0.0]
            )
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
        checkpoint_progress=CheckpointProgressPolicySpec(
            checkpoint_interval=1,
            continuation=continuation,
        ),
    ).model_dump(mode="json", exclude_none=True)
    spec["method_ref"] = {
        "package": "tests.golden",
        "name": "governed",
        "version": "v1",
    }
    spec["method_payload"] = {
        "schema_id": METHOD_SCHEMA,
        "schema_version": METHOD_VERSION,
        "payload": {
            "token": "golden",
            "optimizer": method_payload.optimizer.model_dump(mode="json"),
        },
    }
    spec["worker_execution"]["method_contract"]["method_ref"] = METHOD_REF
    spec["worker_execution"]["method_contract"]["method_payload_schema_version"] = METHOD_VERSION
    spec["worker_execution"]["effective_phase"]["method_ref"] = METHOD_REF
    return spec


def _write_plugin(path: Path) -> None:
    path.write_text(
        textwrap.dedent(f"""
        import jax.numpy as jnp
        from pydantic import BaseModel
        from feedbax.contracts.run_matrix import RowLowererIdentity, TrainingRowLoweringResult
        from feedbax.contracts.training import ScheduleProjection, TrainingMethodDescriptor, TrainingMethodScheduleProjector, standard_supervised_method_contract, standard_supervised_update_kernels
        from feedbax.training.preparation import ExecutionPreparationResult
        from feedbax.training.row_lowering import TrainingRowLowererRegistration, training_row_lowerer_implementation_sha256

        class Payload(BaseModel):
            token: str
            optimizer: dict

        def compile_contract(payload):
            return standard_supervised_method_contract().model_copy(update={{"method_ref": {METHOD_REF!r}, "method_payload_schema_version": {METHOD_VERSION!r}}})

        def kernels(payload):
            return standard_supervised_update_kernels(payload)

        def project_schedules(payload, coordinates):
            return ScheduleProjection(complete=True, schedules={{}})

        def prepare(request):
            assert request.method_payload.token == "golden"
            return ExecutionPreparationResult(initial_slots={{
                "model": jnp.array([0.0]), "optimizer": {{"count": jnp.array([1.0])}},
                "prng": jnp.array([0, 1], dtype=jnp.uint32), "batch_counter": jnp.array(0, dtype=jnp.int32),
            }})

        def lower(row, context):
            return TrainingRowLoweringResult(execution_payload=row.payload["execution_payload"], lowerer_identities=[RowLowererIdentity(lowerer_id={LOWERER_ID!r}, lowerer_version={LOWERER_VERSION!r})])

        LOWER_SHA256 = training_row_lowerer_implementation_sha256(lower)

        def register_feedbax_training_methods(registry):
            registry.register_descriptor(TrainingMethodDescriptor(method_ref={METHOD_REF!r}, payload_schema_id={METHOD_SCHEMA!r}, payload_schema_version={METHOD_VERSION!r}, payload_model=Payload, contract_compiler=compile_contract, update_kernels_factory=kernels, preparation_provider=prepare, schedule_projector=TrainingMethodScheduleProjector(projector_id="tests.golden.schedule_projection", projector_version="tests.golden.schedule_projection.v1", projector=project_schedules), optimizer_spec_projector=lambda payload: payload.optimizer, owner="golden", package="tests.golden"))

        def register_feedbax_training_row_lowerers(registry):
            registry.register(TrainingRowLowererRegistration(authored_schema_id={AUTHORED_SCHEMA!r}, authored_schema_version={AUTHORED_VERSION!r}, lowerer_id={LOWERER_ID!r}, lowerer_version={LOWERER_VERSION!r}, implementation_sha256=LOWER_SHA256, lower=lower, owner="golden"))
    """).lstrip(),
        encoding="utf-8",
    )


def _run(
    command: list[str],
    *,
    root: Path,
    env: dict[str, str],
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=root,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def _canonical_run_spec(
    payload: dict,
    *,
    root: Path,
    env: dict[str, str],
) -> dict:
    result = _run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "from feedbax.plugins import load_training_method_plugins; "
                "from feedbax.contracts.training import TrainingRunSpec; "
                "load_training_method_plugins(modules=['golden_plugin']); "
                "print(TrainingRunSpec.model_validate(json.load(sys.stdin))"
                ".model_dump_json(exclude_none=True))"
            ),
        ],
        root=root,
        env=env,
        input_text=json.dumps(payload),
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_golden_governed_path_restores_one_authenticated_continuation_batch(tmp_path: Path) -> None:
    """Exercise one local shadow transition and the same bundle's RunPod binding dry-run."""
    repo = Path(__file__).resolve().parents[1]
    plugin = tmp_path / "golden_plugin.py"
    _write_plugin(plugin)
    (tmp_path / "golden_plugin-1.0.dist-info").mkdir()
    (tmp_path / "golden_plugin-1.0.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: golden-plugin\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "golden_plugin-1.0.dist-info" / "entry_points.txt").write_text(
        "[feedbax.plugins]\ngolden = golden_plugin\n", encoding="utf-8"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(repo), os.environ.get("PYTHONPATH", "")]),
    }
    lower_sha = _run(
        [sys.executable, "-c", "import golden_plugin; print(golden_plugin.LOWER_SHA256)"],
        root=tmp_path,
        env=env,
    )
    assert lower_sha.returncode == 0, lower_sha.stderr

    parent_spec = tmp_path / "parent.json"
    parent_payload = _canonical_run_spec(_run_spec(batches=1), root=tmp_path, env=env)
    parent_spec.write_text(json.dumps(parent_payload, sort_keys=True), encoding="utf-8")
    parent = _run(
        [
            sys.executable,
            "-m",
            "feedbax",
            "execute-training-run-spec",
            str(parent_spec),
            "--plugin",
            "golden_plugin",
            "--checkpoint-root",
            str(tmp_path / "parent-checkpoints"),
            "--manifest-root",
            str(tmp_path / "parent-manifests"),
            "--run-id",
            "golden-parent",
            "--no-progress",
        ],
        root=tmp_path,
        env=env,
    )
    assert parent.returncode == 0, parent.stderr
    parent_output = json.loads(parent.stdout)
    latest = json.loads(
        (tmp_path / "parent-checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    parent_ref = ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id=latest["transaction_id"],
        role="training_checkpoint_custody",
        uri=latest["manifest_relative_path"],
        metadata={"manifest_sha256": latest["manifest_sha256"]},
    )
    provider_root = tmp_path / "provider"
    artifact = produce_checkpoint_custody_archive(
        parent_ref,
        allowed_root=tmp_path / "parent-checkpoints",
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
                    include={"artifact_id", "sha256", "size_bytes", "media_type", "storage_backend"}
                )
            ),
            format=InputFormatIdentity(
                format_id="feedbax.archive.training_checkpoint_custody",
                format_version="feedbax.archive.training_checkpoint_custody.v1",
                media_type="application/vnd.feedbax.training-checkpoint-custody.v1+tar+gzip",
            ),
            materializer=CheckpointCustodyArchiveMaterializer(
                expected_parent_ref=parent_ref,
                expected_transaction_root_sha256=latest["transaction_root_sha256"],
            ),
        ),
    )
    resolved_path = tmp_path / "checkpoint-input.json"
    resolved_path.write_text(resolved.model_dump_json(), encoding="utf-8")

    child_payload = _canonical_run_spec(
        _run_spec(
            batches=2,
            continuation=CheckpointContinuationRequest(
                source_completed_batches=1,
                additional_batches=1,
                self_contained=False,
            ),
        ),
        root=tmp_path,
        env=env,
    )
    authored = {
        "schema_id": AUTHORED_SCHEMA,
        "schema_version": AUTHORED_VERSION,
        TRAINING_ROW_LOWERER_REF_FIELD: {
            "schema_id": "feedbax.spec.training_row_lowerer_ref",
            "schema_version": "feedbax.spec.training_row_lowerer_ref.v2",
            "context_api_version": "feedbax.training_row_lowering_context.v1",
            "lowerer_id": LOWERER_ID,
            "lowerer_version": LOWERER_VERSION,
            "implementation_sha256": lower_sha.stdout.strip(),
        },
        "execution_payload": child_payload,
    }
    authored_path = tmp_path / "locked-row.json"
    authored_path.write_text(json.dumps(authored, sort_keys=True), encoding="utf-8")
    matrix = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V3,
        "name": "golden governed path",
        "base": {
            "kind": "authored_intent",
            "ref": authored_path.name,
            "content_hash": training_spec_sha256(authored),
        },
        "rows": [{"row_id": "golden-row", "seed": 7, "overrides": []}],
    }
    matrix_path = tmp_path / "locked-matrix-v3.json"
    matrix_bytes = json.dumps(matrix, sort_keys=True).encode()
    matrix_path.write_bytes(matrix_bytes)
    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            schema_version=TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V3,
            artifact_id="fixture:" + hashlib.sha256(matrix_bytes).hexdigest(),
            sha256=hashlib.sha256(matrix_bytes).hexdigest(),
            uri=str(matrix_path),
        ),
        compiler=CompilerIdentity(
            compiler_id=TRAINING_RUN_MATRIX_COMPILER_ID,
            compiler_version=TRAINING_RUN_MATRIX_COMPILER_VERSION,
        ),
        inputs=[
            AssemblyInputDeclaration(
                role="checkpoint", kind="checkpoint-custody-archive", locator=str(resolved_path)
            )
        ],
        deployment_policy=DeploymentPolicy(
            driver="local",
            venue="local",
            cloud_authorized=False,
            review_required=False,
            review_authorized=False,
        ),
        environment=EnvironmentDeclaration(python_version="3.13"),
        launch_policy=LaunchPolicy(max_parallel_rows=1),
        budget=BudgetPolicy(max_wall_clock_seconds=30),
        orchestration_root=str(tmp_path / "orchestration"),
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    launched = _run(
        [
            sys.executable,
            "-m",
            "feedbax.bin.orchestrate",
            "shadow-launch",
            "--assembly-request",
            str(request_path),
            "--input-provider",
            f"checkpoint.inputs={provider_root}",
        ],
        root=tmp_path,
        env=env,
    )
    assert launched.returncode == 0, launched.stderr
    shadow_evidence = json.loads(launched.stdout)
    assert shadow_evidence["schema_id"] == SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID
    assert shadow_evidence["schema_version"] == SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION
    assert shadow_evidence["provider_readiness"] == "not_evaluated"
    assert shadow_evidence["exercised_through_stage"] == "COLLECT"
    assert len(shadow_evidence["rows"]) == 1
    assert shadow_evidence["rows"][0]["segment_completed_batches"] == 1
    assert shadow_evidence["rows"][0]["payload_binding_status"] == "verified"
    bundle_path = next((tmp_path / "orchestration").glob("*/bundle.json"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    migrations = bundle["migration_evidence"]
    assert [
        (migration["source_schema_version"], migration["target_schema_version"])
        for migration in migrations
    ] == [
        (
            TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V3,
            TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V4,
        ),
        (
            TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V4,
            TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        ),
    ]
    assert shadow_evidence["bundle_sha256"] == canonical_run_bundle_sha256(
        RunBundle.model_validate(bundle)
    )
    row_dir = bundle_path.parent / "rows" / "golden-row"
    diagnostics = json.loads((row_dir / "training-diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["segment_completed_batches"] == 1
    assert diagnostics["cumulative_completed_batches"] == 2
    child_latest = json.loads(
        (row_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    child_manifest = json.loads(
        (row_dir / "checkpoints" / child_latest["manifest_relative_path"]).read_text(encoding="utf-8")
    )
    assert child_manifest["parent_lineage"][0]["transaction_id"] == latest["transaction_id"]

    driver_dry_run = _run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import json, sys",
                    "from feedbax.orchestration.bundle import RunBundle, canonical_run_bundle_sha256",
                    "from feedbax.orchestration.drivers import runpod",
                    "from feedbax.orchestration.input_materialization import InputProviderRootBinding",
                    "def unreachable(*args, **kwargs): raise AssertionError('RunPod provider/transport surface reached')",
                    "runpod.SubprocessRunPodTransport = unreachable",
                    "runpod.urllib.request.urlopen = unreachable",
                    "bundle = RunBundle.model_validate_json(open(sys.argv[1]).read())",
                    "commands = runpod.dry_run_launch_bundle(",
                    "    bundle,",
                    "    runpod.RunPodDriverConfig(remote_repo_root='/unreachable', remote_run_root='/unreachable'),",
                    "    (InputProviderRootBinding('checkpoint.inputs', sys.argv[2]),),",
                    ")",
                    "print(json.dumps({'bundle_sha256': canonical_run_bundle_sha256(bundle), 'command_count': len(commands)}))",
                ]
            ),
            str(bundle_path),
            str(provider_root),
        ],
        root=tmp_path,
        env=env,
    )
    assert driver_dry_run.returncode == 0, driver_dry_run.stderr
    assert json.loads(driver_dry_run.stdout) == {
        "bundle_sha256": shadow_evidence["bundle_sha256"],
        "command_count": 1,
    }
    assert parent_output["status"] == "completed"
