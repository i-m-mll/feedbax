from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
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
    CertificateRow,
    CheckEntry,
    ConformanceRowArtifacts,
    RealizedDeploymentRecord,
    RunConformanceCertificate,
    build_default_check_registry,
    check_evaluation_lifecycle,
)
from feedbax.orchestration.drivers.local import LocalOrchestrationDriver
from feedbax.orchestration.drivers import local as local_driver
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
from feedbax.orchestration.state import StageState
from feedbax.orchestration.state import RowState, RunSetState


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
                    "schema_version": "feedbax.spec.evaluation_matrix_batch_plan.v4",
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


def _certificate(
    bundle: RunBundle,
    *,
    additional_check_status: str = "pass",
) -> RunConformanceCertificate:
    observed_at = datetime(2026, 7, 27, tzinfo=timezone.utc)
    realized = RealizedDeploymentRecord(
        driver="local",
        venue="local",
        provider="local",
        environment_fingerprint="fixture",
        provisioned_at=observed_at,
        row_started_at=observed_at,
        row_completed_at=observed_at,
        observed_at=observed_at,
        wall_time_seconds=0.0,
        hourly_rate=0.0,
        accrued_cost=0.0,
        currency="USD",
        cost_basis="local-not-billable",
        observation_basis={"fixture": "focused reclamation test"},
        unavailable={
            "gpu_model": "not applicable locally",
            "gpu_count": "not applicable locally",
            "region": "not applicable locally",
            "immutable_image_id": "not applicable locally",
            "billing_started_at": "not billable locally",
        },
    )
    realized_payload = realized.model_dump(mode="json")
    checks = [
        CheckEntry(
            check_id="realized_deployment",
            status="pass",
            observed=realized_payload,
        ),
        CheckEntry(check_id="focused_reclamation", status=additional_check_status),
    ]
    return RunConformanceCertificate(
        run_set_id=bundle.run_set_id,
        generated_at=observed_at,
        overall=additional_check_status,
        rows={
            bundle.rows[0].row_id: CertificateRow(
                checks=checks,
                realized_deployment=realized,
                realized_deployment_evidence=realized_payload,
            )
        },
    )


def test_bundle_and_rows_require_one_declared_execution_family(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    assert bundle.execution_family == "evaluation-matrix"
    assert bundle.rows[0].execution_family == "evaluation-matrix"
    assert "evaluation" not in bundle.rows[0].launch.collect

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
        repo_root="/governed/root",
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
    assert command[command.index("--repo-root") + 1] == "/governed/root"
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
            repo_root="/governed/root",
            environment_fingerprint="fingerprint",
        )

    with pytest.raises(ExecutorFamilyError, match="governed runtime repo root"):
        executor_family_adapter("evaluation-matrix").bind_command(
            row.launch.command,
            bundle=bundle,
            row=row,
            payload_path="/run/inputs/matrix.json",
            collection_root="/run/rows/matrix",
            inputs_root="/run/inputs",
            repo_root=None,
            environment_fingerprint="fingerprint",
        )

    with pytest.raises(ExecutorFamilyError, match="caller-supplied options"):
        executor_family_adapter("evaluation-matrix").bind_command(
            [*row.launch.command, "--repo-root", "/untrusted/root"],
            bundle=bundle,
            row=row,
            payload_path="/run/inputs/matrix.json",
            collection_root="/run/rows/matrix",
            inputs_root="/run/inputs",
            repo_root="/governed/root",
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


def test_successful_local_evaluation_reclamation_is_terminal_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    row = bundle.rows[0]
    row_root = bundle.run_set_dir / "rows" / row.row_id
    raw_store = row_root / "evaluation"
    compact_store = row_root / "evaluation-batch-compaction"
    raw_store.mkdir(parents=True)
    compact_store.mkdir()
    (raw_store / "states.bin").write_bytes(b"x" * 4096)
    (compact_store / "fragment").write_bytes(b"compact")
    collected_compact = bundle.run_set_dir / "collected" / row.row_id / compact_store.name
    collected_compact.mkdir(parents=True)
    (collected_compact / "fragment").write_bytes(b"compact")
    collected_outputs = {
        name: str(bundle.run_set_dir / "collected" / row.row_id / name)
        for name in EVALUATION_COLLECTION_OUTPUTS
    }
    collected_outputs["evaluation-batch-compaction"] = str(collected_compact)
    for name in EVALUATION_COLLECTION_OUTPUTS:
        if name != "evaluation-batch-compaction":
            Path(collected_outputs[name]).write_text("{}\n", encoding="utf-8")

    class VerifiedAdapter:
        def missing_collection_outputs(self, *_args: object) -> list[str]:
            return []

    monkeypatch.setattr(local_driver, "executor_family_adapter", lambda _family: VerifiedAdapter())
    union_path = bundle.run_set_dir / "evaluation-matrix-ordered-union.json"
    union = {
        "schema_id": "feedbax.orchestration.evaluation_matrix_ordered_union_evidence",
        "schema_version": "feedbax.orchestration.evaluation_matrix_ordered_union_evidence.v1",
        "matrix_intent_hash": "c" * 64,
        "ordered_row_ids_sha256": row.launch.metadata["matrix_ordered_row_ids_sha256"],
        "ordered_batch_ids": ["whole-matrix"],
        "ordered_row_ids": ["gain-a", "gain-b"],
    }
    union_path.parent.mkdir(parents=True, exist_ok=True)
    union_path.write_text(json.dumps(union), encoding="utf-8")
    certificate_path = bundle.run_set_dir / "conformance.json"
    certificate = _certificate(bundle)
    certificate_path.write_text(certificate.model_dump_json(indent=2) + "\n", encoding="utf-8")
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={
            row.row_id: RowState(
                status="completed",
                collected_outputs=collected_outputs,
            )
        },
        stages={
            "COLLECT": StageState(
                status="completed",
                outputs={
                    "rows": {row.row_id: collected_outputs},
                    "evaluation_matrix_ordered_union": {
                        **union,
                        "path": str(union_path),
                    },
                },
            ),
            "CERTIFY": StageState(
                status="completed",
                outputs={
                    "overall": "pass",
                    "certificate_ref": str(certificate_path),
                    "certificate_sha256": hashlib.sha256(
                        certificate_path.read_bytes()
                    ).hexdigest(),
                },
            ),
        },
        certificate_ref=str(certificate_path),
    )
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=[])

    durable_union = union_path.with_name("evaluation-matrix-ordered-union.actual.json")
    union_path.replace(durable_union)
    union_path.symlink_to(durable_union)
    with pytest.raises(
        local_driver.LocalDriverError,
        match="durable path is unsafe",
    ):
        driver.teardown(bundle, state)
    union_path.unlink()
    durable_union.replace(union_path)

    collected_row = bundle.run_set_dir / "collected" / row.row_id
    external_collected_row = tmp_path / "external-collected-row"
    collected_row.replace(external_collected_row)
    collected_row.symlink_to(external_collected_row, target_is_directory=True)
    with pytest.raises(
        local_driver.LocalDriverError,
        match="durable path is unsafe",
    ):
        driver.teardown(bundle, state)
    collected_row.unlink()
    external_collected_row.replace(collected_row)

    external_record = tmp_path / "external-reclamation-record.json"
    external_record.write_text(
        json.dumps(
            {
                "schema_version": (
                    "feedbax.orchestration.local_evaluation_store_reclamation.v1"
                ),
                "row_id": row.row_id,
                "source": str(raw_store),
                "reclaimed_bytes": 4096,
                "status": "deleting",
            }
        ),
        encoding="utf-8",
    )
    reclamation_record = row_root / ".evaluation-store-reclamation.json"
    reclamation_record.symlink_to(external_record)
    with pytest.raises(
        local_driver.LocalDriverError,
        match="reclamation record is unsafe",
    ):
        driver.teardown(bundle, state)
    reclamation_record.unlink()
    external_record.unlink()

    malformed_certificate = '{"overall":"pass"}\n'
    certificate_path.write_text(malformed_certificate, encoding="utf-8")
    malformed_state = state.with_stage(
        "CERTIFY",
        state.stage("CERTIFY").model_copy(
            update={
                "outputs": {
                    **state.stage("CERTIFY").outputs,
                    "certificate_sha256": hashlib.sha256(
                        malformed_certificate.encode("utf-8")
                    ).hexdigest(),
                }
            }
        ),
    )
    with pytest.raises(
        local_driver.LocalDriverError,
        match="valid passing certificate",
    ):
        driver.teardown(bundle, malformed_state)

    failing_certificate = _certificate(bundle, additional_check_status="fail")
    failing_bytes = (failing_certificate.model_dump_json(indent=2) + "\n").encode("utf-8")
    certificate_path.write_bytes(failing_bytes)
    relabeled_state = state.with_stage(
        "CERTIFY",
        state.stage("CERTIFY").model_copy(
            update={
                "outputs": {
                    **state.stage("CERTIFY").outputs,
                    "certificate_sha256": hashlib.sha256(failing_bytes).hexdigest(),
                }
            }
        ),
    )
    with pytest.raises(
        local_driver.LocalDriverError,
        match="valid passing certificate",
    ):
        driver.teardown(bundle, relabeled_state)
    certificate_path.write_text(certificate.model_dump_json(indent=2) + "\n", encoding="utf-8")

    pending = state.with_stage("CERTIFY", StageState(status="pending"))
    retained = driver.teardown(bundle, pending)["successful_evaluation_reclamation"]
    assert retained[0]["status"] == "retained"
    assert raw_store.is_dir()

    unexpected = row_root / ".evaluation.unexpectedly-missing"
    raw_store.rename(unexpected)
    with pytest.raises(
        local_driver.LocalDriverError,
        match="disappeared without reclamation authority",
    ):
        driver.teardown(bundle, state)
    unexpected.rename(raw_store)

    original_rmtree = local_driver.shutil.rmtree

    def interrupt_after_isolation(path: Path) -> None:
        assert path == row_root / ".evaluation.success-reclaiming"
        raise RuntimeError("simulated crash after isolation")

    monkeypatch.setattr(local_driver.shutil, "rmtree", interrupt_after_isolation)
    with pytest.raises(RuntimeError, match="simulated crash after isolation"):
        driver.teardown(bundle, state)
    deleting_record = json.loads(
        (row_root / ".evaluation-store-reclamation.json").read_text(encoding="utf-8")
    )
    assert deleting_record["status"] == "deleting"
    assert not raw_store.exists()
    assert (row_root / ".evaluation.success-reclaiming").is_dir()

    monkeypatch.setattr(local_driver.shutil, "rmtree", original_rmtree)
    reclaimed = driver.teardown(bundle, state)["successful_evaluation_reclamation"]
    assert reclaimed[0]["status"] == "reclaimed"
    assert reclaimed[0]["reclaimed_bytes"] >= 4096
    assert not raw_store.exists()
    record_path = row_root / ".evaluation-store-reclamation.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["reclaimed_bytes"] == reclaimed[0]["reclaimed_bytes"]
    assert (compact_store / "fragment").read_bytes() == b"compact"
    assert (collected_compact / "fragment").read_bytes() == b"compact"

    drifted_record = {**record, "source": str(row_root / "other-evaluation")}
    record_path.write_text(json.dumps(drifted_record), encoding="utf-8")
    with pytest.raises(
        local_driver.LocalDriverError,
        match="reclamation record drifted",
    ):
        driver.teardown(bundle, state)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    assert driver.teardown(bundle, state)["successful_evaluation_reclamation"] == [
        {
            "row_id": row.row_id,
            "status": "already-reclaimed",
            "reclaimed_bytes": 0,
        }
    ]


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
                    ],
                    # Simulate a durable bundle authored before raw evaluation
                    # stores were removed from the collection contract.
                    "collect": [*row.launch.collect, "evaluation"],
                }
            ),
        }
    )
    bundle = bundle.model_copy(update={"rows": [row]})
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=[])
    engine = StageEngine(
        bundle=bundle,
        driver=driver,
        conformance_registry=build_default_check_registry(include_plugins=False),
        poll_interval_seconds=0.001,
    )
    state = engine.run(stop_after_stage="CERTIFY")

    raw_evaluation = bundle.run_set_dir / "rows" / "matrix" / "evaluation"
    compact_store = bundle.run_set_dir / "rows" / "matrix" / "evaluation-batch-compaction"
    collected_compact_store = (
        bundle.run_set_dir / "collected" / "matrix" / "evaluation-batch-compaction"
    )
    assert raw_evaluation.is_dir()
    assert compact_store.is_dir()
    assert collected_compact_store.is_dir()
    assert "evaluation" not in state.rows["matrix"].collected_outputs
    assert not (bundle.run_set_dir / "collected" / "matrix" / "evaluation").exists()

    external_result = tmp_path / "external-evaluation-matrix-result.json"
    external_result.write_bytes(
        Path(state.rows["matrix"].collected_outputs["evaluation-matrix-result.json"]).read_bytes()
    )
    tampered_outputs = {
        **state.rows["matrix"].collected_outputs,
        "evaluation-matrix-result.json": str(external_result),
    }
    tampered_state = state.with_row(
        "matrix",
        state.rows["matrix"].model_copy(update={"collected_outputs": tampered_outputs}),
    ).with_stage(
        "COLLECT",
        state.stage("COLLECT").model_copy(
            update={
                "outputs": {
                    **state.stage("COLLECT").outputs,
                    "rows": {"matrix": tampered_outputs},
                }
            }
        ),
    )
    with pytest.raises(
        local_driver.LocalDriverError,
        match="collected output path is not run-owned",
    ):
        driver.teardown(bundle, tampered_state)
    assert raw_evaluation.is_dir()

    pre_barrier = state.with_stage("CERTIFY", StageState(status="pending"))
    retained = driver.teardown(bundle, pre_barrier)["successful_evaluation_reclamation"]
    assert retained == [
        {
            "row_id": "matrix",
            "status": "retained",
            "reclaimed_bytes": 0,
            "reason": "terminal-consumer-barrier-not-complete",
        }
    ]
    assert raw_evaluation.is_dir()

    reclaimed = driver.teardown(bundle, state)["successful_evaluation_reclamation"]
    assert reclaimed[0]["row_id"] == "matrix"
    assert reclaimed[0]["status"] == "reclaimed"
    assert reclaimed[0]["reclaimed_bytes"] > 0
    assert not raw_evaluation.exists()
    assert compact_store.is_dir()
    assert collected_compact_store.is_dir()

    state = engine.run(stop_after_stage="TEARDOWN")

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
    assert state.stage("TEARDOWN").outputs["successful_evaluation_reclamation"] == [
        {
            "row_id": "matrix",
            "status": "already-reclaimed",
            "reclaimed_bytes": 0,
        }
    ]
    assert compact_store.is_dir()
    assert collected_compact_store.is_dir()
    shadow_evidence = orchestrate._shadow_launch_evidence(bundle, state)
    assert shadow_evidence.exercised_through_stage == "TEARDOWN"
    assert shadow_evidence.lifecycles == (evidence,)
    assert shadow_evidence.ordered_union.ordered_row_ids == ("gain-a", "gain-b")
    assert shadow_evidence.worker_topology.batch_count == 1
