from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pytest

from feedbax.bin import orchestrate
from feedbax.contracts.evaluation_lifecycle import (
    EvaluationLifecycleEvidence,
    EvaluationLifecycleRowOutcome,
)
from feedbax.contracts.spec_storage import training_run_execution_hash
from feedbax.orchestration.bundle import (
    AuthoredIntentRef,
    BudgetPolicy,
    DeploymentPolicy,
    EnvironmentDeclaration,
    ExecutionCapsuleRef,
    ExecutionIdentityEnvelope,
    ResolvedSnapshotRef,
    RowLaunchSpec,
    RunBundle,
    RunRowSpec,
    SchemaArtifactRef,
)
from feedbax.orchestration.conformance import (
    ConformanceRowArtifacts,
    build_default_check_registry,
    check_evaluation_lifecycle,
)
from feedbax.orchestration.drivers.local import LocalOrchestrationDriver
from feedbax.orchestration.drivers.runpod import (
    RunPodDriverConfig,
    dry_run_launch_bundle,
)
from feedbax.orchestration.executor_family import (
    EVALUATION_COLLECTION_OUTPUTS,
    ExecutorFamilyError,
    executor_family_adapter,
)
from feedbax.contracts.manifest import canonical_json_bytes, sha256_bytes
from feedbax.orchestration.revision import resolve_feedbax_revision
from feedbax.orchestration.stages import StageEngine


def _execution(tmp_path: Path) -> ExecutionIdentityEnvelope:
    root_hash = "a" * 64
    execution_hash = training_run_execution_hash(root_hash, [])
    payload = tmp_path / "matrix.json"
    payload.write_text(
        json.dumps(
            {
                "schema_id": "feedbax.spec.evaluation_run_matrix",
                "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
            }
        ),
        encoding="utf-8",
    )
    payload_sha256 = hashlib.sha256(payload.read_bytes()).hexdigest()
    common = {
        "schema_id": "feedbax.test",
        "schema_version": "feedbax.test.v1",
        "artifact_id": "artifact:test",
        "sha256": payload_sha256,
    }
    return ExecutionIdentityEnvelope(
        payload=SchemaArtifactRef(**common, uri=str(payload)),
        authored_intent=AuthoredIntentRef(**common, intent_hash="c" * 64),
        resolved_snapshot=ResolvedSnapshotRef(**common, root_hash=root_hash),
        execution_capsule=ExecutionCapsuleRef(
            **common,
            execution_hash=execution_hash,
        ),
        immutable_inputs=[],
    )


def _bundle(tmp_path: Path, *, family: str = "evaluation-matrix") -> RunBundle:
    row = RunRowSpec(
        row_id="matrix",
        execution_family=family,
        execution=_execution(tmp_path),
        launch=RowLaunchSpec(
            command=["python", "-m", "feedbax", "matrix-harness"],
            collect=list(EVALUATION_COLLECTION_OUTPUTS),
            payload_routing={"kind": "registered-execution-payload"},
            metadata={
                "matrix_intent_hash": "c" * 64,
                "matrix_ordered_row_ids_sha256": sha256_bytes(
                    canonical_json_bytes(["gain-a", "gain-b"])
                ),
                "batch_plan": {
                    "schema_id": "feedbax.spec.evaluation_matrix_batch_plan",
                    "schema_version": "feedbax.spec.evaluation_matrix_batch_plan.v1",
                    "matrix_intent_hash": "c" * 64,
                    "batches": [
                        {
                            "batch_id": "whole-matrix",
                            "ordered_row_ids": ["gain-a", "gain-b"],
                        }
                    ],
                },
            },
        ),
    )
    return RunBundle(
        run_set_id="evaluation-lifecycle",
        feedbax_revision=resolve_feedbax_revision(),
        execution_family=family,
        deployment_policy=DeploymentPolicy(
            driver="local",
            venue="local",
            cloud_authorized=False,
            review_required=False,
            review_authorized=False,
        ),
        rows=[row],
        environment=EnvironmentDeclaration(python_version=platform.python_version()),
        budget=BudgetPolicy(max_wall_clock_seconds=60),
        orchestration_root=str(tmp_path),
    )


def test_bundle_and_rows_require_one_declared_execution_family(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    assert bundle.execution_family == "evaluation-matrix"
    assert bundle.rows[0].execution_family == "evaluation-matrix"

    payload = bundle.model_dump(mode="json")
    payload["execution_family"] = "native-training"
    with pytest.raises(ValueError, match="does not match the bundle"):
        RunBundle.model_validate(payload)


def test_evaluation_adapter_binds_only_public_whole_matrix_harness(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    row = bundle.rows[0]
    command, bound = executor_family_adapter("evaluation-matrix").bind_command(
        row.launch.command,
        bundle=bundle,
        row=row,
        payload_path="/run/inputs/matrix.json",
        collection_root="/run/rows/matrix",
        inputs_root="/run/inputs",
        environment_fingerprint="fingerprint",
    )

    assert command[:6] == [
        "python",
        "-m",
        "feedbax",
        "matrix-harness",
        "/run/inputs/matrix.json",
        "--manifest-root",
    ]
    assert "--batch" in command
    assert command[command.index("--orchestration-inputs-root") + 1] == "/run/inputs"
    assert bound.execution.payload.uri == "/run/inputs/matrix.json"

    with pytest.raises(ExecutorFamilyError, match="public `feedbax matrix-harness`"):
        executor_family_adapter("evaluation-matrix").bind_command(
            ["python", "copied-evaluator.py"],
            bundle=bundle,
            row=row,
            payload_path="/run/inputs/matrix.json",
            collection_root="/run/rows/matrix",
            inputs_root="/run/inputs",
            environment_fingerprint="fingerprint",
        )


def test_runpod_dry_run_uses_same_evaluation_family_adapter(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path).model_copy(
        update={
            "deployment_policy": DeploymentPolicy(
                driver="runpod",
                venue="remote",
                cloud_authorized=True,
                review_required=True,
                review_authorized=True,
            )
        }
    )
    command = dry_run_launch_bundle(bundle, RunPodDriverConfig())[0]

    assert "matrix-harness" in command
    assert "--orchestration-inputs-root" in command
    assert "--lifecycle-result" in command


def test_ordered_nested_outcomes_are_family_conformance_evidence(tmp_path: Path) -> None:
    evidence = EvaluationLifecycleEvidence(
        orchestration_row_id="matrix",
        ordered_row_ids=("gain-a", "gain-b"),
        outcomes=(
            EvaluationLifecycleRowOutcome(
                row_id="gain-a",
                manifest_id="feedbax-eval-run:a",
                manifest_path=str(tmp_path / "a.json"),
                diagnostic_schema_ids=("feedbax.diagnostic.a",),
            ),
            EvaluationLifecycleRowOutcome(
                row_id="gain-b",
                manifest_id="feedbax-eval-run:b",
                manifest_path=str(tmp_path / "b.json"),
                diagnostic_schema_ids=("feedbax.diagnostic.b",),
            ),
        ),
    )
    result = check_evaluation_lifecycle(
        ConformanceRowArtifacts(
            row_id="matrix",
            evaluation_lifecycle=evidence.model_dump(mode="json"),
        )
    )

    assert result.status == "pass"
    assert result.observed["ordered_row_ids"] == ["gain-a", "gain-b"]

    payload = evidence.model_dump(mode="json")
    payload["ordered_row_ids"] = ["gain-b", "gain-a"]
    with pytest.raises(ValueError, match="preserve ordered_row_ids"):
        EvaluationLifecycleEvidence.model_validate(payload)


def test_local_evaluation_family_traverses_production_lifecycle_to_teardown(
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "evaluation_lifecycle_plugin.py"
    plugin.write_text(
        """
from feedbax.analysis.evaluation import EvaluationRecipeResult, register_evaluation_recipe

def recipe(_spec, _root, _states_path, _context):
    return EvaluationRecipeResult(summary_metrics={"ok": 1.0})

def batch(items, _context):
    return [EvaluationRecipeResult(summary_metrics={"gain": item.spec.params["gain"]})
            for item in items]

register_evaluation_recipe("feedbax.test.lifecycle", recipe, batch_recipe=batch, replace=True)
""".strip(),
        encoding="utf-8",
    )
    matrix = {
        "schema_id": "feedbax.spec.evaluation_run_matrix",
        "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
        "base": {
            "evaluation_type": "feedbax.test.lifecycle",
            "params": {"gain": 0.0},
        },
        "rows": [
            {
                "row_id": "gain-a",
                "deltas": [{"path": "params.gain", "value": 1.0}],
            },
            {
                "row_id": "gain-b",
                "deltas": [{"path": "params.gain", "value": 2.0}],
            },
        ],
    }
    bundle = _bundle(tmp_path)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    row = bundle.rows[0]
    payload_ref = row.execution.payload.model_copy(
        update={
            "schema_id": matrix["schema_id"],
            "schema_version": matrix["schema_version"],
            "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            "uri": str(matrix_path),
        }
    )
    row = row.model_copy(
        update={
            "execution": row.execution.model_copy(update={"payload": payload_ref}),
            "launch": row.launch.model_copy(
                update={
                    "command": [
                        "python",
                        "-m",
                        "feedbax",
                        "matrix-harness",
                        "--plugin",
                        "evaluation_lifecycle_plugin",
                    ]
                }
            ),
        }
    )
    bundle = bundle.model_copy(update={"rows": [row]})
    state = StageEngine(
        bundle=bundle,
        driver=LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=[]),
        conformance_registry=build_default_check_registry(include_plugins=False),
        poll_interval_seconds=0.001,
    ).run(stop_after_stage="TEARDOWN")

    assert state.stage("CERTIFY").status == "completed"
    assert state.stage("TEARDOWN").status == "completed"
    assert state.stage("REGISTER").status == "pending"
    assert state.rows["matrix"].status == "completed"
    evidence = EvaluationLifecycleEvidence.model_validate_json(
        Path(state.rows["matrix"].collected_outputs["evaluation-matrix-result.json"]).read_text(
            encoding="utf-8"
        )
    )
    assert evidence.ordered_row_ids == ("gain-a", "gain-b")
    certificate = json.loads((bundle.run_set_dir / "conformance.json").read_text())
    checks = {item["check_id"]: item for item in certificate["rows"]["matrix"]["checks"]}
    assert checks["evaluation_lifecycle"]["status"] == "pass"
    assert checks["events_terminal"]["status"] == "pass"
    assert certificate["overall"] == "pass"
    shadow_evidence = orchestrate._shadow_launch_evidence(bundle, state)
    assert shadow_evidence.exercised_through_stage == "TEARDOWN"
    assert shadow_evidence.lifecycles == (evidence,)
    assert shadow_evidence.ordered_union.ordered_row_ids == ("gain-a", "gain-b")
    assert shadow_evidence.worker_topology.batch_count == 1
