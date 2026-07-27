from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.analysis.evaluation_orchestration import (
    EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_COMPILER_ID,
    EVALUATION_RUN_MATRIX_COMPILER_VERSION,
    EvaluationMatrixExecutionCapsule,
    evaluation_matrix_intent_hash,
)
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.evaluation_lifecycle import (
    EvaluationMatrixBatchPlan,
    EvaluationMatrixBatchUnit,
)
from feedbax.contracts.evaluation_preflight import EvaluationOutputPreflightPolicy
from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot
from feedbax.orchestration import (
    AssemblyContext,
    BudgetPolicy,
    CompilerIdentity,
    DeploymentPolicy,
    EnvironmentDeclaration,
    LaunchPolicy,
    RunAssemblyRequest,
    SchemaArtifactRef,
    StageEngine,
    assemble_run_bundle,
    build_default_assembly_registry,
)
from feedbax.orchestration.assembly import _prepare_run_assembly
from feedbax.orchestration.staged_root_custody import (
    StagedRootSourceBinding,
    seal_staged_root,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def _matrix() -> dict[str, Any]:
    return {
        "schema_id": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "base": {
            "evaluation_type": "tests.evaluation_orchestration",
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


def _sized_matrix(count: int) -> dict[str, Any]:
    matrix = _matrix()
    matrix["rows"] = [
        {
            "row_id": f"gain-{index:02d}",
            "deltas": [{"path": "params.gain", "value": float(index)}],
        }
        for index in range(count)
    ]
    return matrix


def _write_json(path: Path, payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return data


def _request(
    tmp_path: Path,
    authored: dict[str, Any],
    *,
    driver: str = "local",
    batch_plan: EvaluationMatrixBatchPlan | None = None,
    output_preflight: EvaluationOutputPreflightPolicy | None = None,
) -> RunAssemblyRequest:
    authored_path = tmp_path / "matrix.json"
    authored_bytes = _write_json(authored_path, authored)
    return RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=str(authored["schema_id"]),
            schema_version=str(authored["schema_version"]),
            artifact_id=f"fixture:{hashlib.sha256(authored_bytes).hexdigest()}",
            sha256=hashlib.sha256(authored_bytes).hexdigest(),
            uri=str(authored_path),
        ),
        compiler=CompilerIdentity(
            compiler_id=EVALUATION_RUN_MATRIX_COMPILER_ID,
            compiler_version=EVALUATION_RUN_MATRIX_COMPILER_VERSION,
        ),
        deployment_policy=DeploymentPolicy(
            driver=driver,
            venue="local" if driver == "local" else "remote",
            cloud_authorized=driver == "runpod",
            review_required=driver == "runpod",
            review_authorized=driver == "runpod",
        ),
        environment=EnvironmentDeclaration(python_version=platform.python_version()),
        launch_policy=LaunchPolicy(max_parallel_rows=1),
        evaluation_batch_plan=batch_plan,
        evaluation_output_preflight=output_preflight,
        budget=BudgetPolicy(max_wall_clock_seconds=60),
        orchestration_root=str(tmp_path / "orchestration"),
    )


def _output_preflight(
    *,
    expected_rows: int,
    bytes_per_row: int = 100,
    repetitions: int = 2,
    reserve_bytes: int = 50,
) -> EvaluationOutputPreflightPolicy:
    return EvaluationOutputPreflightPolicy(
        expected_resolved_row_count=expected_rows,
        retained_bytes_per_resolved_row=bytes_per_row,
        retained_bytes_per_resolved_row_source="measured representative row fixture",
        planned_repetitions=repetitions,
        required_free_space_reserve_bytes=reserve_bytes,
    )


@pytest.mark.parametrize(
    "field",
    [
        "expected_resolved_row_count",
        "retained_bytes_per_resolved_row",
        "planned_repetitions",
        "required_free_space_reserve_bytes",
    ],
)
def test_evaluation_output_preflight_policy_rejects_boolean_integers(field: str) -> None:
    payload = _output_preflight(expected_rows=2).model_dump(mode="json")
    payload[field] = True

    with pytest.raises(ValidationError, match=field):
        EvaluationOutputPreflightPolicy.model_validate(payload)


def _assemble(request: RunAssemblyRequest, tmp_path: Path, *, run_set_id: str = "run"):
    return assemble_run_bundle(
        request,
        run_set_id=run_set_id,
        context=AssemblyContext(
            custody_root=tmp_path / "custody",
            repo_root=tmp_path,
            materializer_commit="d9e62cfd" + "0" * 32,
        ),
        registry=build_default_assembly_registry(),
    )


def test_authored_matrix_compiles_ordered_batches_under_one_matrix_identity(
    tmp_path: Path,
) -> None:
    matrix = _matrix()
    intent_hash = evaluation_matrix_intent_hash(matrix)
    plan = EvaluationMatrixBatchPlan(
        matrix_intent_hash=intent_hash,
        batches=(
            EvaluationMatrixBatchUnit(batch_id="0000", ordered_row_ids=("gain-a",)),
            EvaluationMatrixBatchUnit(batch_id="0001", ordered_row_ids=("gain-b",)),
        ),
    )
    request = _request(tmp_path, matrix, batch_plan=plan).model_copy(
        update={"launch_policy": LaunchPolicy(max_parallel_rows=4)}
    )
    bundle = _assemble(request, tmp_path)

    assert bundle.execution_family == "evaluation-matrix"
    assert len(bundle.rows) == 1
    assert bundle.launch_policy.max_parallel_rows == 4
    row = bundle.rows[0]
    assert row.row_id == f"matrix-{intent_hash[:16]}"
    assert row.launch.metadata["batch_plan"]["batches"] == [
        {"batch_id": "0000", "ordered_row_ids": ["gain-a"]},
        {"batch_id": "0001", "ordered_row_ids": ["gain-b"]},
    ]
    assert row.launch.metadata["matrix_intent_hash"] == intent_hash
    assert row.launch.command == ["python", "-m", "feedbax", "matrix-harness"]
    assert row.launch.payload_routing["kind"] == "registered-execution-payload"
    assert row.execution.row_provenance is None

    capsule = EvaluationMatrixExecutionCapsule.model_validate_json(
        Path(row.execution.execution_capsule.uri).read_text(encoding="utf-8")
    )
    assert capsule.schema_id == EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_ID
    assert capsule.intent_hash == evaluation_matrix_intent_hash(matrix)
    assert capsule.execution_hash == row.execution.execution_capsule.execution_hash


def test_batch_plan_drift_is_rejected_during_assembly_before_launch(
    tmp_path: Path,
) -> None:
    matrix = _matrix()
    plan = EvaluationMatrixBatchPlan(
        matrix_intent_hash=evaluation_matrix_intent_hash(matrix),
        batches=(
            EvaluationMatrixBatchUnit(
                batch_id="drifted",
                ordered_row_ids=("gain-b", "gain-a"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="ordered union"):
        _assemble(_request(tmp_path, matrix, batch_plan=plan), tmp_path)


@pytest.mark.parametrize("driver", ["local", "runpod"])
def test_explicit_authored_rows_defeat_inert_subset_expectation_before_outputs(
    tmp_path: Path,
    driver: str,
) -> None:
    request = _request(
        tmp_path,
        _matrix(),
        driver=driver,
        output_preflight=_output_preflight(expected_rows=1),
    )
    output_root = Path(request.orchestration_root) / "cardinality"

    with pytest.raises(
        ValueError,
        match=r"resolved row count.*expected=1 resolved=2",
    ):
        _assemble(request, tmp_path, run_set_id="cardinality")

    assert not output_root.exists()
    assert not (tmp_path / "custody").exists()


@pytest.mark.parametrize("driver", ["local", "runpod"])
def test_insufficient_disk_fails_before_run_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    driver: str,
) -> None:
    from feedbax.orchestration import assembly as assembly_module

    request = _request(
        tmp_path,
        _matrix(),
        driver=driver,
        output_preflight=_output_preflight(
            expected_rows=2,
            bytes_per_row=100,
            repetitions=3,
            reserve_bytes=50,
        ),
    )
    disk_paths: list[Path] = []

    def disk_usage(path: Path) -> SimpleNamespace:
        disk_paths.append(Path(path))
        return SimpleNamespace(free=649)

    monkeypatch.setattr(assembly_module.shutil, "disk_usage", disk_usage)
    run_set_id = "disk-refusal"
    output_root = Path(request.orchestration_root) / run_set_id
    engine = StageEngine.from_request(
        request,
        context=AssemblyContext(
            custody_root=tmp_path / "custody",
            repo_root=tmp_path,
            materializer_commit="d9e62cfd" + "0" * 32,
        ),
        registry=build_default_assembly_registry(),
        driver_factory=lambda _bundle: pytest.fail("driver constructed before disk refusal"),
        run_set_id=run_set_id,
    )

    with pytest.raises(
        ValueError,
        match=r"required_free_bytes=650 observed_free_bytes=649",
    ):
        engine.run()

    assert not output_root.exists()
    assert not (tmp_path / "custody").exists()
    assert disk_paths == [tmp_path.resolve()]


def test_successful_pre_output_pass_is_pure_before_the_run_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from feedbax.orchestration import assembly as assembly_module

    request = _request(
        tmp_path,
        _matrix(),
        output_preflight=_output_preflight(expected_rows=2),
    )
    monkeypatch.setattr(
        assembly_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=10_000),
    )

    prepared = _prepare_run_assembly(
        request,
        run_set_id="pure-preflight",
        context=AssemblyContext(
            custody_root=tmp_path / "custody",
            repo_root=tmp_path,
            materializer_commit="d9e62cfd" + "0" * 32,
        ),
        registry=build_default_assembly_registry(),
    )

    assert prepared.evaluation_output_preflight is not None
    assert not (Path(request.orchestration_root) / "pure-preflight").exists()
    assert not (tmp_path / "custody").exists()


def test_stage_engine_reuses_the_pre_root_capacity_decision_inside_the_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from feedbax.orchestration import assembly as assembly_module

    request = _request(
        tmp_path,
        _matrix(),
        output_preflight=_output_preflight(expected_rows=2),
    )
    disk_paths: list[Path] = []

    def disk_usage(path: Path) -> SimpleNamespace:
        disk_paths.append(Path(path))
        if len(disk_paths) > 1:
            raise AssertionError("disk capacity was re-observed after the output root existed")
        return SimpleNamespace(free=10_000)

    monkeypatch.setattr(assembly_module.shutil, "disk_usage", disk_usage)
    run_set_id = "single-pre-root-decision"
    engine = StageEngine.from_request(
        request,
        context=AssemblyContext(
            custody_root=tmp_path / "custody",
            repo_root=tmp_path,
            materializer_commit="d9e62cfd" + "0" * 32,
        ),
        registry=build_default_assembly_registry(),
        driver_factory=lambda _bundle: SimpleNamespace(),
        run_set_id=run_set_id,
    )

    state = engine.run(stop_after_stage="ASSEMBLE")

    assert state.stage("ASSEMBLE").status == "completed"
    assert disk_paths == [tmp_path.resolve()]
    assert (Path(request.orchestration_root) / run_set_id / "bundle.json").is_file()


@pytest.mark.parametrize("driver", ["local", "runpod"])
def test_sufficient_disk_budget_retains_exact_preflight_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    driver: str,
) -> None:
    from feedbax.orchestration import assembly as assembly_module

    policy = _output_preflight(
        expected_rows=2,
        bytes_per_row=100,
        repetitions=3,
        reserve_bytes=50,
    )
    request = _request(
        tmp_path,
        _matrix(),
        driver=driver,
        output_preflight=policy,
    )
    disk_paths: list[Path] = []

    def disk_usage(path: Path) -> SimpleNamespace:
        disk_paths.append(Path(path))
        return SimpleNamespace(free=651)

    monkeypatch.setattr(assembly_module.shutil, "disk_usage", disk_usage)

    bundle = _assemble(request, tmp_path, run_set_id="disk-pass")

    evidence = bundle.evaluation_output_preflight
    assert evidence is not None
    assert evidence.expected_resolved_row_count == 2
    assert evidence.resolved_row_count == 2
    assert evidence.retained_bytes_per_resolved_row == 100
    assert evidence.retained_bytes_per_resolved_row_source == (
        "measured representative row fixture"
    )
    assert evidence.planned_repetitions == 3
    assert evidence.estimated_retained_bytes == 600
    assert evidence.required_free_space_reserve_bytes == 50
    assert evidence.required_free_bytes == 650
    assert evidence.observed_free_bytes == 651
    assert evidence.output_root == str(Path(request.orchestration_root) / "disk-pass")
    assert evidence.observed_filesystem_path == str(tmp_path.resolve())
    assert evidence.observed_filesystem_device == tmp_path.stat().st_dev
    assert disk_paths == [tmp_path.resolve()]
    assert bundle.rows[0].launch.metadata["batch_plan"]["batches"] == [
        {"batch_id": "whole-matrix", "ordered_row_ids": ["gain-a", "gain-b"]}
    ]


def test_content_pinned_delta_keeps_authored_payload_and_ordered_resolved_identity(
    tmp_path: Path,
) -> None:
    parent = _matrix()
    parent_path = tmp_path / "parent.json"
    _write_json(parent_path, parent)
    delta = {
        "schema_id": EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
        "schema_version": EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
        "parent": {
            "ref": parent_path.name,
            "sha256": sha256_bytes(canonical_json_bytes(parent)),
        },
        "deltas": [
            {
                "layer_id": "gain-shift",
                "patches": [{"path": "base.params.gain", "value": 0.5}],
            }
        ],
    }
    bundle = _assemble(_request(tmp_path, delta), tmp_path, run_set_id="delta")
    row = bundle.rows[0]
    payload = json.loads(Path(row.execution.payload.uri).read_text(encoding="utf-8"))
    snapshot = decode_resolved_snapshot(
        json.loads(Path(row.execution.resolved_snapshot.uri).read_text(encoding="utf-8"))
    )

    assert payload["schema_id"] == EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID
    assert snapshot["ordered_row_ids"] == ["gain-a", "gain-b"]
    assert snapshot["composition"]["layers"][0]["layer_ids"] == ["gain-shift"]


def test_matrix_staged_parents_require_and_bind_exact_governed_root_custody(
    tmp_path: Path,
) -> None:
    def parent(identifier: str, *, checkpoint: bool = False) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": "a" * 64,
            "size_bytes": 1,
        }
        if checkpoint:
            metadata["checkpoint_custody_binding"] = "checkpoints"
        return {
            "kind": "EvaluationRunManifest",
            "id": identifier,
            "role": "evaluation_run",
            "metadata": metadata,
        }

    local = {
        "parent": parent("local", checkpoint=True),
        "artifact_provider": None,
    }
    remote = {
        "parent": parent("remote"),
        "artifact_provider": "artifacts",
    }
    matrix = _matrix()
    matrix["base"]["params"]["staged_prerequisites"] = {
        "local": local,
        "remote": remote,
    }
    matrix["staged_parents"] = {"local": local, "remote": remote}
    request = _request(tmp_path, matrix)
    with pytest.raises(ValueError, match="lack governed staged-root custody"):
        _assemble(request, tmp_path)

    roots = tmp_path / "roots"
    manifest_root = roots / "manifests"
    artifact_root = roots / "artifacts"
    checkpoint_root = roots / "checkpoints"
    for root in (manifest_root, artifact_root, checkpoint_root):
        root.mkdir(parents=True)
        (root / "fixture").write_text("x", encoding="utf-8")
    sealed = [
        seal_staged_root(
            StagedRootSourceBinding(
                "artifacts",
                "artifact-provider",
                artifact_root,
                ImmutableArtifactBlobProviderSpec(),
            ),
            snapshot_parent=tmp_path / "sealed",
        ),
        seal_staged_root(
            StagedRootSourceBinding(
                "checkpoints",
                "checkpoint-custody",
                checkpoint_root,
            ),
            snapshot_parent=tmp_path / "sealed",
        ),
        seal_staged_root(
            StagedRootSourceBinding(
                "local",
                "manifest-store",
                manifest_root,
            ),
            snapshot_parent=tmp_path / "sealed",
        ),
    ]
    governed = request.model_copy(update={"staged_roots": [item.custody for item in sealed]})
    bundle = _assemble(governed, tmp_path, run_set_id="governed")

    assert [(item.root_kind, item.binding_name) for item in bundle.staged_roots] == [
        ("artifact-provider", "artifacts"),
        ("checkpoint-custody", "checkpoints"),
        ("manifest-store", "local"),
    ]


def test_provider_free_cli_shadow_reaches_terminal_collection_in_fresh_process(
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "evaluation_plugin.py"
    plugin.write_text(
        """
from feedbax.analysis.evaluation import EvaluationRecipeResult, register_evaluation_recipe

def _recipe(_spec, _root, _states_path, _context):
    return EvaluationRecipeResult(summary_metrics={"ok": 1.0})

def _batch(items, _context):
    return [
        EvaluationRecipeResult(
            summary_metrics={"gain": item.spec.params["gain"]},
            metadata={"states_schema": "tests.diagnostic.evaluation.v1"},
        )
        for item in items
    ]

def register_feedbax_analysis_recipes():
    register_evaluation_recipe(
        "tests.evaluation_orchestration",
        _recipe,
        batch_recipe=_batch,
        replace=True,
    )
""".strip(),
        encoding="utf-8",
    )
    dist_info = tmp_path / "evaluation_plugin-0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Name: evaluation-plugin\nVersion: 0.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[feedbax.plugins]\nevaluation-test = evaluation_plugin\n",
        encoding="utf-8",
    )
    manifest_root = tmp_path / "shadow-roots" / "manifests"
    artifact_root = tmp_path / "shadow-roots" / "artifacts"
    checkpoint_root = tmp_path / "shadow-roots" / "checkpoints"
    manifest_root.mkdir(parents=True)
    (manifest_root / "fixture.json").write_text("{}\n", encoding="utf-8")
    artifact_root.mkdir(parents=True)
    ImmutableArtifactBlobProvider(artifact_root).store_bytes(
        b"artifact",
        role="shadow",
        logical_name="shadow",
    )
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "fixture.eqx").write_bytes(b"checkpoint")
    sealed = [
        seal_staged_root(
            StagedRootSourceBinding(
                "artifacts",
                "artifact-provider",
                artifact_root,
                ImmutableArtifactBlobProviderSpec(),
            ),
            snapshot_parent=tmp_path / "shadow-sealed",
        ),
        seal_staged_root(
            StagedRootSourceBinding(
                "checkpoints",
                "checkpoint-custody",
                checkpoint_root,
            ),
            snapshot_parent=tmp_path / "shadow-sealed",
        ),
        seal_staged_root(
            StagedRootSourceBinding(
                "manifests",
                "manifest-store",
                manifest_root,
            ),
            snapshot_parent=tmp_path / "shadow-sealed",
        ),
    ]
    shadow_matrix = _sized_matrix(8)
    request = _request(tmp_path, shadow_matrix).model_copy(
        update={
            "staged_roots": [item.custody for item in sealed],
            "evaluation_batch_plan": EvaluationMatrixBatchPlan(
                matrix_intent_hash=evaluation_matrix_intent_hash(shadow_matrix),
                batches=tuple(
                    EvaluationMatrixBatchUnit(
                        batch_id=f"{index:04d}",
                        ordered_row_ids=(f"gain-{index:02d}",),
                    )
                    for index in range(8)
                ),
            ),
            "launch_policy": LaunchPolicy(max_parallel_rows=4),
        }
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(repo_root)))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "feedbax.bin.orchestrate",
            "shadow-launch",
            "--assembly-request",
            str(request_path),
            *[
                option
                for item in sealed
                for option in (
                    "--staged-root",
                    f"{item.custody.root_kind}:{item.custody.binding_name}={item.staging_root}",
                )
            ],
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(completed.stdout)
    assert evidence["exercised_through_stage"] == "TEARDOWN"
    assert evidence["provider_readiness"] == "not_evaluated"
    assert (
        evidence["ordered_union"]["schema_version"]
        == "feedbax.orchestration.evaluation_matrix_ordered_union_evidence.v1"
    )
    assert evidence["ordered_union"]["ordered_row_ids"] == [
        f"gain-{index:02d}" for index in range(8)
    ]
    assert evidence["ordered_union"]["ordered_batch_ids"] == [f"{index:04d}" for index in range(8)]
    topology = evidence["worker_topology"]
    assert topology["requested_worker_count"] == 4
    assert topology["batch_count"] == 8
    assert len(topology["processes"]) == 4
    assert len({item["pid"] for item in topology["processes"]}) == 4
    assert all(len(item["ordered_batch_ids"]) == 2 for item in topology["processes"])
    assert [
        outcome["diagnostic_schema_ids"]
        for lifecycle in evidence["lifecycles"]
        for outcome in lifecycle["outcomes"]
    ] == [["tests.diagnostic.evaluation.v1"] for _ in range(8)]


def test_runpod_dry_run_keeps_the_same_public_matrix_executor(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, _matrix(), driver="runpod")
    bundle = _assemble(request, tmp_path)

    from feedbax.orchestration.drivers.runpod import (
        RunPodDriverConfig,
        dry_run_launch_bundle,
    )

    command = dry_run_launch_bundle(bundle, RunPodDriverConfig())[0]
    assert "matrix-harness" in command
    assert "--batch" in command
    assert "--orchestration-inputs-root" in command
    assert "--lifecycle-result" in command
