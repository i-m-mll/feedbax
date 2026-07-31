from __future__ import annotations

import os
import shutil
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import feedbax.orchestration.input_materialization as input_materialization
from feedbax.analysis.execution_context import resolve_staged_execution_context
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.evaluation_lifecycle import EVALUATION_COLLECTION_OUTPUTS
from feedbax.orchestration.bundle import RunBundle
from feedbax.orchestration.drivers import local as local_driver
from feedbax.orchestration.drivers.local import LocalOrchestrationDriver
from feedbax.orchestration.drivers.runpod import (
    RunPodDriverConfig,
    RunPodOrchestrationDriver,
)
from feedbax.orchestration.input_materialization import (
    InputMaterializationError,
    materialize_bundle_inputs,
    preflight_staged_root_bindings,
    reclaim_materialized_staged_roots,
    staged_execution_bindings_for_bundle,
)
from feedbax.orchestration.staged_root_custody import (
    StagedRootCustody,
    StagedRootCustodyError,
    StagedRootSnapshotBinding,
    StagedRootSourceBinding,
    seal_staged_root,
    verify_staged_root_snapshot,
)
from feedbax.orchestration.state import RowState, RunSetState
from feedbax.orchestration.conformance import build_core_check_registry
from feedbax.orchestration.stages import StageEngine
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from tests.test_evaluation_lifecycle import (
    _bundle as _evaluation_lifecycle_bundle,
    _certificate,
)
from tests.test_orchestration_core import _bundle
from tests.test_runpod_orchestration_driver import FakeRunPodTransport


def _sealed_bundle(
    tmp_path: Path,
) -> tuple[RunBundle, tuple[StagedRootSnapshotBinding, ...]]:
    sources = tmp_path / "sources"
    manifest_root = sources / "manifest"
    artifact_root = sources / "artifacts"
    checkpoint_root = sources / "checkpoints"
    manifest_path = manifest_root / "manifests" / "training_runs" / "run.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"id":"run"}\n', encoding="utf-8")
    artifact_root.mkdir(parents=True)
    ImmutableArtifactBlobProvider(artifact_root).store_bytes(
        b"artifact-bytes",
        role="fixture",
        logical_name="fixture",
    )
    checkpoint_path = checkpoint_root / "transactions" / "tx" / "model.eqx"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint-bytes")

    source_bindings = (
        StagedRootSourceBinding(
            "artifacts",
            "artifact-provider",
            artifact_root,
            ImmutableArtifactBlobProviderSpec(),
        ),
        StagedRootSourceBinding(
            "checkpoints",
            "checkpoint-custody",
            checkpoint_root,
        ),
        StagedRootSourceBinding("manifests", "manifest-store", manifest_root),
    )
    sealed = tuple(
        seal_staged_root(binding, snapshot_parent=tmp_path / "sealed")
        for binding in source_bindings
    )
    bundle = _bundle(tmp_path / "orchestration").model_copy(
        update={"staged_roots": [item.custody for item in sealed]}
    )
    return bundle, tuple(item.binding for item in sealed)


def _state(bundle: RunBundle, attempts: int = 1) -> RunSetState:
    return RunSetState(
        run_set_id=bundle.run_set_id,
        environment_fingerprint="fingerprint",
        stages={"STAGE_INPUTS": {"attempts": attempts}},
        rows={row.row_id: RowState() for row in bundle.rows},
    )


def _terminal_evaluation_state(bundle: RunBundle) -> RunSetState:
    row = bundle.rows[0]
    row_root = bundle.run_set_dir / "rows" / row.row_id
    raw_store = row_root / "evaluation"
    raw_store.mkdir(parents=True, exist_ok=True)
    (raw_store / "states.bin").write_bytes(b"raw")
    collected_outputs = {
        name: str(bundle.run_set_dir / "collected" / row.row_id / name)
        for name in EVALUATION_COLLECTION_OUTPUTS
    }
    for name, path in collected_outputs.items():
        output_path = Path(path)
        if name == "evaluation-batch-compaction":
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "fragment").write_bytes(b"compact")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("{}\n", encoding="utf-8")
    plan = row.launch.metadata["batch_plan"]
    ordered_row_ids = [row_id for batch in plan["batches"] for row_id in batch["ordered_row_ids"]]
    union = {
        "schema_id": "feedbax.orchestration.evaluation_matrix_ordered_union_evidence",
        "schema_version": ("feedbax.orchestration.evaluation_matrix_ordered_union_evidence.v1"),
        "matrix_intent_hash": plan["matrix_intent_hash"],
        "ordered_row_ids_sha256": row.launch.metadata["matrix_ordered_row_ids_sha256"],
        "ordered_batch_ids": [batch["batch_id"] for batch in plan["batches"]],
        "ordered_row_ids": ordered_row_ids,
    }
    union_path = bundle.run_set_dir / "evaluation-matrix-ordered-union.json"
    union_path.write_text(json.dumps(union), encoding="utf-8")
    certificate_path = bundle.run_set_dir / "conformance.json"
    certificate_path.write_text(
        _certificate(bundle).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return RunSetState(
        run_set_id=bundle.run_set_id,
        environment_fingerprint="fingerprint",
        stages={
            "STAGE_INPUTS": {"status": "completed"},
            "COLLECT": {
                "status": "completed",
                "outputs": {
                    "rows": {row.row_id: collected_outputs},
                    "evaluation_matrix_ordered_union": {
                        **union,
                        "path": str(union_path),
                    },
                },
            },
            "CERTIFY": {
                "status": "completed",
                "outputs": {
                    "overall": "pass",
                    "certificate_ref": str(certificate_path),
                    "certificate_sha256": hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
                },
            },
        },
        rows={
            row.row_id: RowState(
                status="completed",
                collected_outputs=collected_outputs,
            )
        },
        certificate_ref=str(certificate_path),
    )


def _make_writable(path: Path) -> None:
    path.parent.chmod(0o700)
    path.chmod(0o600)


def _regular_file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _production_evaluation_bundle(
    tmp_path: Path,
    *,
    run_set_id: str,
    matrix_intent_hash: str,
    gain_offset: float,
    staged_roots: list[StagedRootCustody],
) -> RunBundle:
    matrix = {
        "schema_id": "feedbax.spec.evaluation_run_matrix",
        "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
        "base": {
            "evaluation_type": "feedbax.test.staged_root_reclamation",
            "params": {"gain": gain_offset},
        },
        "rows": [
            {
                "row_id": "gain-a",
                "deltas": [{"path": "params.gain", "value": gain_offset + 1.0}],
            },
            {
                "row_id": "gain-b",
                "deltas": [{"path": "params.gain", "value": gain_offset + 2.0}],
            },
        ],
    }
    matrix_path = tmp_path / f"{run_set_id}.matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    bundle = _evaluation_lifecycle_bundle(tmp_path)
    row = bundle.rows[0]
    payload = row.execution.payload.model_copy(
        update={
            "schema_id": matrix["schema_id"],
            "schema_version": matrix["schema_version"],
            "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            "uri": str(matrix_path),
        }
    )
    batch_plan = {
        **row.launch.metadata["batch_plan"],
        "schema_version": "feedbax.spec.evaluation_matrix_batch_plan.v4",
        "matrix_intent_hash": matrix_intent_hash,
    }
    row = row.model_copy(
        update={
            "execution": row.execution.model_copy(
                update={
                    "payload": payload,
                    "authored_intent": row.execution.authored_intent.model_copy(
                        update={"intent_hash": matrix_intent_hash}
                    ),
                }
            ),
            "launch": row.launch.model_copy(
                update={
                    "command": [
                        "python",
                        "-m",
                        "feedbax",
                        "matrix-harness",
                        "--plugin",
                        "evaluation_reclamation_plugin",
                    ],
                    "metadata": {
                        **row.launch.metadata,
                        "matrix_intent_hash": matrix_intent_hash,
                        "batch_plan": batch_plan,
                    },
                }
            ),
        }
    )
    return RunBundle.model_validate(
        bundle.model_copy(
            update={
                "run_set_id": run_set_id,
                "rows": [row],
                "staged_roots": staged_roots,
            }
        ).model_dump(mode="json")
    )


def test_staged_root_contract_round_trip_and_schema_rejection(tmp_path: Path) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)

    payload = bundle.staged_roots[0].model_dump(mode="json")
    assert StagedRootCustody.model_validate(payload) == bundle.staged_roots[0]
    assert bundle == RunBundle.model_validate_json(bundle.model_dump_json())
    assert [record.relative_path for record in bundle.staged_roots[0].files] == sorted(
        record.relative_path for record in bundle.staged_roots[0].files
    )
    assert bundle.staged_roots[0].custody_ref.startswith("staged-root://sha256/")
    assert not preflight_staged_root_bindings(bundle, bindings)[0]

    payload["schema_version"] = "feedbax.orchestration.staged_root_custody.v0"
    with pytest.raises(ValidationError):
        StagedRootCustody.model_validate(payload)


def test_staged_roots_do_not_change_native_training_row_bytes(tmp_path: Path) -> None:
    native = _bundle(tmp_path / "native")
    staged, _bindings = _sealed_bundle(tmp_path / "staged")
    rebound = native.model_copy(update={"staged_roots": staged.staged_roots})

    assert rebound.rows == native.rows
    assert rebound.resolved_inputs == native.resolved_inputs


def test_sealing_rejects_symlinks_and_binding_name_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "real").write_text("bytes", encoding="utf-8")
    (source / "link").symlink_to(source / "real")
    with pytest.raises(StagedRootCustodyError, match="symlink or unsupported"):
        seal_staged_root(
            StagedRootSourceBinding("manifests", "manifest-store", source),
            snapshot_parent=tmp_path / "sealed",
        )

    bundle, bindings = _sealed_bundle(tmp_path / "drift")
    first = bindings[0]
    drifted = (
        StagedRootSnapshotBinding(
            "other",
            first.kind,
            first.root,
            first.expected_root_identity,
        ),
        *bindings[1:],
    )
    failures, _ = preflight_staged_root_bindings(bundle, drifted)
    assert any("missing=" in failure and "unexpected=" in failure for failure in failures)


def test_snapshot_drift_fails_local_preflight_before_provision(tmp_path: Path) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    mutated = Path(bindings[0].root) / bundle.staged_roots[0].files[0].relative_path
    _make_writable(mutated)
    mutated.write_bytes(b"mutated")

    driver = LocalOrchestrationDriver(
        freeze_lines=[],
        staged_root_bindings=bindings,
    )
    checks = driver.preflight_checks(bundle)

    assert checks[0].status == "fail"
    assert "differs from custody manifest" in (checks[0].detail or "")
    assert not bundle.run_set_dir.exists()


def test_byte_identical_snapshot_replacement_is_rejected(tmp_path: Path) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    replaced = Path(bindings[0].root)
    backup = replaced.with_name(replaced.name + ".original")
    replaced.rename(backup)
    shutil.copytree(backup, replaced)

    failures, _ = preflight_staged_root_bindings(bundle, bindings)

    assert any("binding was replaced" in failure for failure in failures)


def test_local_materialization_projects_exact_public_context(tmp_path: Path) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    staged = materialize_bundle_inputs(
        bundle,
        destination_root=bundle.run_set_dir / ".stage-attempts" / "focused",
        staged_root_bindings=bindings,
    )
    inputs_root = bundle.run_set_dir / ".stage-attempts" / "focused" / "inputs"

    assert len(staged) == 3
    projected = staged_execution_bindings_for_bundle(bundle, inputs_root=inputs_root)
    context = resolve_staged_execution_context(
        projected.descriptor,
        artifact_provider_bindings=projected.artifact_provider_bindings,
        manifest_root_bindings=projected.manifest_root_bindings,
        checkpoint_custody_bindings=projected.checkpoint_custody_bindings,
    )

    assert tuple(context.opened_artifact_providers) == ("artifacts",)
    assert tuple(context.checkpoint_custody_roots) == ("checkpoints",)
    assert tuple(context.manifest_roots) == ("manifests",)
    assert context.artifact_provider("artifacts").root == (
        inputs_root / "staged-roots" / "artifact-provider" / "artifacts"
    )


def test_staged_root_group_cleans_every_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    original = input_materialization.materialize_staged_root_snapshot
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StagedRootCustodyError("injected materialization failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        input_materialization,
        "materialize_staged_root_snapshot",
        fail_second,
    )
    destination_root = bundle.run_set_dir / ".stage-attempts" / "failure"
    with pytest.raises(InputMaterializationError, match="injected materialization failure"):
        materialize_bundle_inputs(
            bundle,
            destination_root=destination_root,
            staged_root_bindings=bindings,
        )

    inputs_root = destination_root / "inputs"
    assert not (inputs_root / "staged-roots").exists()
    assert not list(inputs_root.glob(".staged-roots-*"))


def test_sequential_local_lifecycles_reclaim_run_local_staged_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, bindings = _sealed_bundle(tmp_path)
    first = _production_evaluation_bundle(
        tmp_path,
        run_set_id="evaluation-reclamation-first",
        matrix_intent_hash="a" * 64,
        gain_offset=0.0,
        staged_roots=authority.staged_roots,
    )
    second = _production_evaluation_bundle(
        tmp_path,
        run_set_id="evaluation-reclamation-second",
        matrix_intent_hash="b" * 64,
        gain_offset=10.0,
        staged_roots=authority.staged_roots,
    )
    expected_bytes = sum(
        record.size_bytes for custody in first.staged_roots for record in custody.files
    )

    class VerifiedAdapter:
        def missing_collection_outputs(self, *_args: object) -> list[str]:
            return []

    monkeypatch.setattr(
        local_driver,
        "executor_family_adapter",
        lambda _family: VerifiedAdapter(),
    )
    driver = LocalOrchestrationDriver(
        freeze_lines=[],
        staged_root_bindings=bindings,
    )

    driver.provision(first, _state(first))
    driver.stage_inputs(first, _state(first))
    first_staged_roots = first.run_set_dir / "inputs" / "staged-roots"
    assert _regular_file_bytes(first_staged_roots) == expected_bytes
    first_terminal = _terminal_evaluation_state(first)
    first_collection = first.run_set_dir / "collected" / first.rows[0].row_id
    first_certificate = first.run_set_dir / "conformance.json"

    preterminal = first_terminal.with_stage(
        "CERTIFY",
        first_terminal.stage("CERTIFY").model_copy(update={"status": "pending"}),
    )
    retained = driver.teardown(first, preterminal)
    assert retained["staged_root_reclamation"]["status"] == "retained"
    assert first_staged_roots.is_dir()

    reclaimed = driver.teardown(first, first_terminal)
    assert reclaimed["staged_root_reclamation"] == {
        "status": "reclaimed",
        "custody_refs": [custody.custody_ref for custody in first.staged_roots],
        "reclaimed_bytes": expected_bytes,
    }
    assert not (first.run_set_dir / "inputs" / "staged-roots").exists()
    assert first_collection.is_dir()
    assert first_certificate.is_file()

    driver.provision(second, _state(second))
    driver.stage_inputs(second, _state(second))
    second_staged_roots = second.run_set_dir / "inputs" / "staged-roots"
    assert not first_staged_roots.exists()
    assert _regular_file_bytes(second_staged_roots) == expected_bytes
    assert first.run_set_id != second.run_set_id
    assert (
        first.rows[0].launch.metadata["matrix_intent_hash"]
        != second.rows[0].launch.metadata["matrix_intent_hash"]
    )

    second_terminal = _terminal_evaluation_state(second)
    second_reclaimed = driver.teardown(second, second_terminal)
    assert second_reclaimed["staged_root_reclamation"]["reclaimed_bytes"] == expected_bytes
    repeated = driver.teardown(second, second_terminal)
    assert repeated["staged_root_reclamation"]["status"] == "already-reclaimed"


def test_failed_local_lifecycle_reclaims_authenticated_staged_roots(
    tmp_path: Path,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    driver = LocalOrchestrationDriver(
        freeze_lines=[],
        staged_root_bindings=bindings,
    )
    driver.provision(bundle, _state(bundle))
    driver.stage_inputs(bundle, _state(bundle))
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        stages={
            "STAGE_INPUTS": {"status": "completed"},
            "COLLECT": {"status": "failed"},
        },
        rows={
            row.row_id: RowState(status="failed", error="fixture executor failure")
            for row in bundle.rows
        },
    )

    result = driver.teardown(bundle, state)

    assert result["staged_root_reclamation"]["status"] == "reclaimed"
    assert not (bundle.run_set_dir / "inputs" / "staged-roots").exists()
    assert (bundle.run_set_dir / "inputs" / ".staged-roots-reclaimed.json").is_file()
    assert driver.teardown(bundle, state)["staged_root_reclamation"]["status"] == (
        "already-reclaimed"
    )


def test_provider_free_sequential_lifecycles_bound_staged_root_peak(
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "evaluation_reclamation_plugin.py"
    plugin.write_text(
        """
from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.plugins.application import EVALUATION_RECIPES
from feedbax.plugins.bootstrap import FamilyRequirement, PluginDeclaration, PluginRegistration

def recipe(_spec, _root, _states_path, _context):
    return EvaluationRecipeResult(summary_metrics={"ok": 1.0})

def batch(items, _context):
    return [EvaluationRecipeResult(summary_metrics={"gain": item.spec.params["gain"]})
            for item in items]

def register(context):
    context.registry(EVALUATION_RECIPES).register(
        "feedbax.test.staged_root_reclamation",
        recipe,
        batch_recipe=batch,
    )

PLUGIN_REGISTRATION = PluginRegistration(
    PluginDeclaration(
        "feedbax.test.staged_root_reclamation",
        "1",
        families=(FamilyRequirement(EVALUATION_RECIPES.family),),
    ),
    register,
)
""".strip(),
        encoding="utf-8",
    )
    authority, bindings = _sealed_bundle(tmp_path / "authority")
    expected_bytes = sum(
        record.size_bytes for custody in authority.staged_roots for record in custody.files
    )
    bundles = [
        _production_evaluation_bundle(
            tmp_path,
            run_set_id=f"evaluation-reclamation-{index}",
            matrix_intent_hash=character * 64,
            gain_offset=float(index * 10),
            staged_roots=authority.staged_roots,
        )
        for index, character in ((1, "a"), (2, "b"))
    ]
    observed_peaks: list[int] = []

    class MeasuringLocalDriver(LocalOrchestrationDriver):
        def stage_inputs(
            self,
            bundle: RunBundle,
            state: RunSetState,
        ) -> dict[str, Any]:
            result = dict(super().stage_inputs(bundle, state))
            observed_peaks.append(
                _regular_file_bytes(bundle.run_set_dir / "inputs" / "staged-roots")
            )
            return result

    terminal_states = []
    for bundle in bundles:
        driver = MeasuringLocalDriver(
            cwd=tmp_path,
            freeze_lines=[],
            staged_root_bindings=bindings,
        )
        terminal = StageEngine(
            bundle=bundle,
            driver=driver,
            conformance_registry=build_core_check_registry(),
            poll_interval_seconds=0.001,
        ).run()
        terminal_states.append(terminal)
        assert terminal.stage("REGISTER").status == "completed"
        assert (
            terminal.stage("TEARDOWN").outputs["staged_root_reclamation"]["status"] == "reclaimed"
        )
        assert not (bundle.run_set_dir / "inputs" / "staged-roots").exists()
        assert (bundle.run_set_dir / "inputs" / ".staged-roots-reclaimed.json").is_file()
        assert (bundle.run_set_dir / "conformance.json").is_file()
        assert Path(
            terminal.rows["matrix"].collected_outputs["evaluation-matrix-result.json"]
        ).is_file()

        resumed = StageEngine(
            bundle=bundle,
            driver=MeasuringLocalDriver(
                cwd=tmp_path,
                freeze_lines=[],
                staged_root_bindings=bindings,
            ),
            conformance_registry=build_core_check_registry(),
            poll_interval_seconds=0.001,
        ).run()
        assert resumed.stage("TEARDOWN").outputs == terminal.stage("TEARDOWN").outputs

    assert observed_peaks == [expected_bytes, expected_bytes]
    assert bundles[0].run_set_id != bundles[1].run_set_id
    assert (
        bundles[0].rows[0].launch.metadata["matrix_intent_hash"]
        != bundles[1].rows[0].launch.metadata["matrix_intent_hash"]
    )
    assert all(state.stage("CERTIFY").outputs["overall"] == "pass" for state in terminal_states)
    assert (
        sum(
            (bundle.run_set_dir / "inputs" / ".staged-roots-reclaimed.json").stat().st_size
            for bundle in bundles
        )
        < 32 * 1024
    )


def test_staged_root_reclamation_resumes_after_atomic_isolation(
    tmp_path: Path,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    materialize_bundle_inputs(
        bundle,
        destination_root=bundle.run_set_dir,
        staged_root_bindings=bindings,
    )
    inputs_root = bundle.run_set_dir / "inputs"
    (inputs_root / "staged-roots").rename(inputs_root / ".staged-roots-reclaiming")

    result = reclaim_materialized_staged_roots(bundle, inputs_root=inputs_root)

    assert result.status == "reclaimed"
    assert not (inputs_root / "staged-roots").exists()
    assert not (inputs_root / ".staged-roots-reclaiming").exists()


def test_staged_root_reclamation_resumes_partial_authenticated_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    materialize_bundle_inputs(
        bundle,
        destination_root=bundle.run_set_dir,
        staged_root_bindings=bindings,
    )
    inputs_root = bundle.run_set_dir / "inputs"
    original_remove = input_materialization._remove_materialization_tree

    def interrupt_removal(
        root: Path,
        *,
        expected_identity: tuple[int, int] | None = None,
    ) -> None:
        assert expected_identity is not None
        first_file = next(path for path in root.rglob("*") if path.is_file())
        _make_writable(first_file)
        first_file.unlink()
        raise OSError("injected removal interruption")

    monkeypatch.setattr(
        input_materialization,
        "_remove_materialization_tree",
        interrupt_removal,
    )
    with pytest.raises(OSError, match="injected removal interruption"):
        reclaim_materialized_staged_roots(bundle, inputs_root=inputs_root)
    assert (inputs_root / ".staged-roots-reclaiming").is_dir()
    assert (inputs_root / ".staged-roots-reclaiming.json").is_file()

    monkeypatch.setattr(
        input_materialization,
        "_remove_materialization_tree",
        original_remove,
    )
    result = reclaim_materialized_staged_roots(bundle, inputs_root=inputs_root)

    assert result.status == "reclaimed"
    assert not (inputs_root / ".staged-roots-reclaiming").exists()
    assert not (inputs_root / ".staged-roots-reclaiming.json").exists()


def test_staged_root_reclamation_promotes_marker_after_completed_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    materialize_bundle_inputs(
        bundle,
        destination_root=bundle.run_set_dir,
        staged_root_bindings=bindings,
    )
    inputs_root = bundle.run_set_dir / "inputs"
    original_publish = input_materialization._publish_staged_root_reclamation_receipt

    def interrupt_receipt(_marker: Path, _receipt: Path) -> None:
        raise OSError("injected receipt interruption")

    monkeypatch.setattr(
        input_materialization,
        "_publish_staged_root_reclamation_receipt",
        interrupt_receipt,
    )
    with pytest.raises(OSError, match="injected receipt interruption"):
        reclaim_materialized_staged_roots(bundle, inputs_root=inputs_root)
    assert not (inputs_root / ".staged-roots-reclaiming").exists()
    assert (inputs_root / ".staged-roots-reclaiming.json").is_file()
    assert not (inputs_root / ".staged-roots-reclaimed.json").exists()

    monkeypatch.setattr(
        input_materialization,
        "_publish_staged_root_reclamation_receipt",
        original_publish,
    )
    result = reclaim_materialized_staged_roots(bundle, inputs_root=inputs_root)

    assert result.status == "already-reclaimed"
    assert not (inputs_root / ".staged-roots-reclaiming.json").exists()
    assert (inputs_root / ".staged-roots-reclaimed.json").is_file()


def test_staged_root_reclamation_rejects_missing_tree_without_receipt(
    tmp_path: Path,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    materialize_bundle_inputs(
        bundle,
        destination_root=bundle.run_set_dir,
        staged_root_bindings=bindings,
    )
    inputs_root = bundle.run_set_dir / "inputs"
    staged_roots = inputs_root / "staged-roots"
    staged_roots.rename(inputs_root / "externally-removed-staged-roots")

    with pytest.raises(
        StagedRootCustodyError,
        match="missing without a reclamation receipt",
    ):
        reclaim_materialized_staged_roots(bundle, inputs_root=inputs_root)

    assert not (inputs_root / ".staged-roots-reclaimed.json").exists()


def test_staged_root_reclamation_rejects_stale_restart_marker(
    tmp_path: Path,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    materialize_bundle_inputs(
        bundle,
        destination_root=bundle.run_set_dir,
        staged_root_bindings=bindings,
    )
    inputs_root = bundle.run_set_dir / "inputs"
    (inputs_root / "staged-roots").rename(inputs_root / ".staged-roots-reclaiming")
    marker = inputs_root / ".staged-roots-reclaiming.json"
    marker.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        StagedRootCustodyError,
        match="marker differs from bundle custody",
    ):
        reclaim_materialized_staged_roots(bundle, inputs_root=inputs_root)

    assert (inputs_root / ".staged-roots-reclaiming").is_dir()
    assert marker.is_file()


@pytest.mark.parametrize(
    "tamper",
    ["symlink", "unexpected-kind", "unexpected-binding", "missing-bytes"],
)
def test_staged_root_reclamation_fails_closed_on_unowned_materialization(
    tmp_path: Path,
    tamper: str,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    materialize_bundle_inputs(
        bundle,
        destination_root=bundle.run_set_dir,
        staged_root_bindings=bindings,
    )
    inputs_root = bundle.run_set_dir / "inputs"
    staged_roots = inputs_root / "staged-roots"
    if tamper == "symlink":
        original = inputs_root / "staged-roots-original"
        staged_roots.rename(original)
        staged_roots.symlink_to(original, target_is_directory=True)
    elif tamper == "unexpected-kind":
        (staged_roots / "unexpected-kind").mkdir()
    elif tamper == "unexpected-binding":
        unexpected = staged_roots / "manifest-store" / "unexpected"
        unexpected.mkdir()
    else:
        custody = bundle.staged_roots[0]
        missing = (
            staged_roots / custody.root_kind / custody.binding_name / custody.files[0].relative_path
        )
        _make_writable(missing)
        missing.unlink()

    with pytest.raises(StagedRootCustodyError):
        reclaim_materialized_staged_roots(bundle, inputs_root=inputs_root)

    assert os.path.lexists(staged_roots)


def test_staged_root_reclamation_rejects_parent_replacement_after_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    materialize_bundle_inputs(
        bundle,
        destination_root=bundle.run_set_dir,
        staged_root_bindings=bindings,
    )
    inputs_root = bundle.run_set_dir / "inputs"
    original_verify = input_materialization._verify_materialized_staged_root_group

    def replace_parent_after_verification(custodies, root):
        identity = original_verify(custodies, root)
        inputs_root.rename(bundle.run_set_dir / "inputs-original")
        replacement = bundle.run_set_dir / "inputs" / "staged-roots"
        replacement.mkdir(parents=True)
        (replacement / "unowned").write_text("unowned\n", encoding="utf-8")
        return identity

    monkeypatch.setattr(
        input_materialization,
        "_verify_materialized_staged_root_group",
        replace_parent_after_verification,
    )
    with pytest.raises(
        StagedRootCustodyError,
        match="materialization parent was replaced",
    ):
        reclaim_materialized_staged_roots(bundle, inputs_root=inputs_root)

    replacement = inputs_root / "staged-roots"
    assert (replacement / "unowned").read_text(encoding="utf-8") == "unowned\n"
    assert not (inputs_root / ".staged-roots-reclaiming").exists()


def test_staged_root_reclamation_rejects_symlinked_inputs_root(
    tmp_path: Path,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    materialize_bundle_inputs(
        bundle,
        destination_root=bundle.run_set_dir,
        staged_root_bindings=bindings,
    )
    inputs_root = bundle.run_set_dir / "inputs"
    alias = bundle.run_set_dir / "inputs-alias"
    alias.symlink_to(inputs_root, target_is_directory=True)

    with pytest.raises(StagedRootCustodyError, match="must not traverse symlinks"):
        reclaim_materialized_staged_roots(bundle, inputs_root=alias)

    assert (inputs_root / "staged-roots").is_dir()


def test_fd_bound_removal_does_not_delete_replacement_at_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated"
    root.mkdir()
    (root / "owned").write_text("owned\n", encoding="utf-8")
    identity = input_materialization._directory_identity(root)
    original_scandir = input_materialization.os.scandir
    replaced = False

    def replace_at_scandir(directory_descriptor):
        nonlocal replaced
        if not replaced:
            replaced = True
            root.rename(tmp_path / "authenticated-original")
            root.mkdir()
            (root / "unowned").write_text("unowned\n", encoding="utf-8")
        return original_scandir(directory_descriptor)

    monkeypatch.setattr(input_materialization.os, "scandir", replace_at_scandir)

    with pytest.raises(
        StagedRootCustodyError,
        match="removal target was replaced",
    ):
        input_materialization._remove_materialization_tree(
            root,
            expected_identity=identity,
        )

    assert (root / "unowned").read_text(encoding="utf-8") == "unowned\n"


def test_runpod_stage_inputs_transports_only_sealed_materialization(
    tmp_path: Path,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(remote_run_root="/remote/runs"),
        transport=transport,
        staged_root_bindings=bindings,
    )

    result = driver.stage_inputs(bundle, _state(bundle))

    assert result["input_count"] == 3
    assert len(transport.rsync_calls) == 1
    source, target, delete, _exclude = transport.rsync_calls[0]
    assert source.endswith("/inputs/")
    assert target.endswith("/inputs/")
    assert delete is True
    assert not transport.runpodctl_calls


def test_unexpected_member_is_rejected_without_replacing_custody(
    tmp_path: Path,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    root = Path(bindings[-1].root)
    root.chmod(0o700)
    unexpected = root / "unexpected.json"
    unexpected.write_text("{}", encoding="utf-8")

    with pytest.raises(StagedRootCustodyError, match="differs from custody manifest"):
        verify_staged_root_snapshot(bundle.staged_roots[-1], root)
    assert os.path.samefile(root, bindings[-1].root)
