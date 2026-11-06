from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.analysis.evaluation_compaction import activate_evaluation_payload_collection
from feedbax.analysis.evaluation_orchestration import (
    EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_COMPILER_ID,
    EVALUATION_RUN_MATRIX_COMPILER_VERSION,
    EvaluationMatrixExecutionCapsule,
    evaluation_matrix_intent_hash,
)
from feedbax.contracts.artifact_custody import (
    ContentBinding,
    ContentObjectRef,
    ImmutableArtifactBlobProviderSpec,
)
from feedbax.contracts.evaluation_lifecycle import (
    EvaluationBatchCompactionEvidence,
    EvaluationBatchConsumerDeclaration,
    EvaluationMatrixBatchPlan,
    EvaluationMatrixBatchUnit,
    EvaluationPayloadRetentionPolicy,
)
from feedbax.contracts.evaluation_preflight import EvaluationOutputPreflightPolicy
from feedbax.contracts.base import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    ArtifactRef,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
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
from feedbax.orchestration.bundle import RunBundle
from feedbax.orchestration.drivers.capabilities import DriverConstructionContext
from feedbax.orchestration.executor_family import (
    EVALUATION_COLLECTION_OUTPUTS,
    executor_family_adapter,
)
from feedbax.orchestration.staged_root_custody import (
    StagedRootSourceBinding,
    migrate_staged_root_v1,
    seal_staged_root,
)
from feedbax.persistence import content_retention
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins.application import ApplicationRegistryBundle, new_application_registry_bundle
from feedbax.orchestration.revision import resolve_feedbax_revision


def _assembly_registry():
    registries = new_application_registry_bundle(local_component_source=None)
    return build_default_assembly_registry(
        method_registry=registries.training_programs,
        row_lowerer_registry=registries.row_lowerers,
        evaluation_registry=registries.evaluation_recipes,
    )


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
    output_preflight: EvaluationOutputPreflightPolicy | None,
    driver: str = "local",
    batch_plan: EvaluationMatrixBatchPlan | None = None,
) -> RunAssemblyRequest:
    authored_path = tmp_path / "matrix.json"
    authored_bytes = _write_json(authored_path, authored)
    return RunAssemblyRequest(
        feedbax_revision=resolve_feedbax_revision(),
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
        storage_mode="retain_all",
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


def test_evaluation_output_preflight_policy_requires_explicit_storage_mode() -> None:
    payload = _output_preflight(expected_rows=2).model_dump(mode="json")
    del payload["storage_mode"]

    with pytest.raises(ValidationError, match="storage_mode"):
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
        registry=_assembly_registry(),
    )


def _grouped_worker_tasks(
    tmp_path: Path,
    ordered_batches: tuple[tuple[str, ...], ...],
    *,
    matrix: dict[str, Any] | None = None,
    content_bindings: tuple[ContentBinding, ...] = (),
    consumers: tuple[EvaluationBatchConsumerDeclaration, ...] = (),
) -> tuple[RunBundle, tuple[dict[str, Any], ...]]:
    if matrix is None:
        matrix = _sized_matrix(sum(map(len, ordered_batches)))
    plan = EvaluationMatrixBatchPlan(
        matrix_intent_hash=evaluation_matrix_intent_hash(matrix),
        batches=tuple(
            EvaluationMatrixBatchUnit(
                batch_id=f"{index:04d}",
                ordered_row_ids=row_ids,
                required_leaf_ids=tuple(consumer.leaf_id for consumer in consumers) or None,
            )
            for index, row_ids in enumerate(ordered_batches)
        ),
        consumers=consumers,
    )
    request = _request(
        tmp_path,
        matrix,
        batch_plan=plan,
        output_preflight=_output_preflight(expected_rows=len(matrix["rows"])),
    ).model_copy(update={"content_bindings": list(content_bindings)})
    bundle = _assemble(request, tmp_path)
    inputs_root = tmp_path / "inputs"
    inputs_root.mkdir()
    bundle_path = inputs_root / "run-bundle.json"
    bundle_path.write_text(bundle.model_dump_json(exclude_none=True), encoding="utf-8")
    payload_path = inputs_root / f"{bundle.rows[0].row_id}.json"
    payload_uri = bundle.rows[0].execution.payload.uri
    assert payload_uri is not None
    shutil.copyfile(payload_uri, payload_path)
    common = {
        "bundle_path": str(bundle_path),
        "bundle_sha256": sha256_bytes(bundle_path.read_bytes()),
        "orchestration_row_id": bundle.rows[0].row_id,
        "inputs_root": str(inputs_root),
        "payload_path": str(payload_path),
        "manifest_root": str(tmp_path / "evaluations"),
        "repo_root": str(tmp_path),
        "compaction_root": str(tmp_path / "compaction"),
        "timing_origin_ns": time.monotonic_ns(),
    }
    return bundle, tuple({**common, "batch_index": index} for index in range(len(plan.batches)))


def _register_grouped_worker_recipe(monkeypatch, batch_recipe) -> ApplicationRegistryBundle:
    from feedbax.plugins import composition

    registries = new_application_registry_bundle(local_component_source=None)
    registries.evaluation_recipes.register(
        "tests.evaluation_orchestration",
        lambda *_args: pytest.fail("expected the batch recipe"),
        batch_recipe=batch_recipe,
    )

    async def compose(*, modules):
        return SimpleNamespace(bundle=registries)

    monkeypatch.setattr(composition, "compose_application", compose)
    return registries


@pytest.mark.parametrize("accepts_fresh_states", [False, True], ids=["cache", "fresh"])
def test_grouped_worker_preserves_order_outputs_and_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepts_fresh_states: bool
) -> None:
    from feedbax.analysis import evaluation
    from feedbax.analysis.evaluation_compaction import EvaluationBatchFragment
    from feedbax.analysis.harness import (
        _execute_evaluation_batch_partition_group,
        resolve_manifest_regeneration,
    )
    from feedbax.contracts.base import ArtifactRef
    from feedbax.contracts.manifest import EvaluationRunManifest

    ordered_batches = (("gain-00",), ("gain-01", "gain-02", "gain-03"))
    declaration = EvaluationBatchConsumerDeclaration(
        leaf_id="gains",
        consumer_id="tests.projector",
        consumer_version="v1",
        terminal_analysis_type="tests.gains.analysis",
        accepted_evaluation_state_schema_ids=("tests.diagnostic.evaluation.v1",),
        compact_product_schema_id="tests.gains.product",
        compact_product_schema_version="tests.gains.product.v1",
        compact_product_role="gains_result",
        merge_state_schema_id="tests.gains.merge",
        merge_state_schema_version="tests.gains.merge.v1",
    )
    _, tasks = _grouped_worker_tasks(tmp_path, ordered_batches, consumers=(declaration,))

    class CacheDecodedState(dict):
        def __reduce__(self):
            return dict, ({**self, "decoded": True},)

    state_type = dict if accepts_fresh_states else CacheDecodedState
    registries = _register_grouped_worker_recipe(
        monkeypatch,
        lambda items, _context: [
            evaluation.EvaluationRecipeResult(
                states=state_type(gain=item.spec.params["gain"]),
                summary_metrics={"gain": item.spec.params["gain"]},
                metadata={"states_schema": "tests.diagnostic.evaluation.v1"},
            )
            for item in items
        ],
    )

    def compact(value):
        assert all(
            states.get("decoded", False) is not accepts_fresh_states for states in value.states
        )
        return EvaluationBatchFragment(
            payload=[
                [outcome.row_id, states["gain"]]
                for outcome, states in zip(value.outcomes, value.states, strict=True)
            ],
            schema_id=declaration.compact_product_schema_id,
            schema_version=declaration.compact_product_schema_version,
            role=declaration.compact_product_role,
        )

    def unexpected_terminal_call(_value):
        pytest.fail("the batch worker only compacts fragments")

    registries.evaluation_batch_consumers.register(
        declaration.consumer_id,
        declaration.consumer_version,
        compact=compact,
        merge=unexpected_terminal_call,
        finalize=unexpected_terminal_call,
        **({"accepts_fresh_states": True} if accepts_fresh_states else {}),
    )

    loaded_manifest_ids = []
    load_cache = evaluation.load_evaluation_states_cache

    def record_cache_load(path, *, manifest):
        loaded_manifest_ids.append(manifest.id)
        return load_cache(path, manifest=manifest)

    monkeypatch.setattr(evaluation, "load_evaluation_states_cache", record_cache_load)
    provider = ImmutableArtifactBlobProvider(Path(tasks[0]["compaction_root"]))
    materialized_counts = []
    materialize = evaluation.materialize_matrix_rows

    def record_materialization(*args, **kwargs):
        rows = materialize(*args, **kwargs)
        materialized_counts.append(len(rows))
        return rows

    monkeypatch.setattr(evaluation, "materialize_matrix_rows", record_materialization)

    def snapshot(completed):
        rows = []
        (fragment_value,) = completed["fragments"]
        fragment = ArtifactRef.model_validate(fragment_value)
        fragment_bytes = provider.get_bytes(fragment)
        assert fragment.metadata["parent_authorities"] == completed["parent_authorities"]
        assert json.loads(fragment_bytes) == [
            [outcome["row_id"], float(outcome["row_id"].removeprefix("gain-"))]
            for outcome in completed["outcomes"]
        ]
        for outcome, authority in zip(
            completed["outcomes"], completed["parent_authorities"], strict=True
        ):
            raw = Path(outcome["manifest_path"]).read_bytes()
            assert authority["id"] == outcome["manifest_id"]
            assert authority["metadata"]["manifest_sha256"] == sha256_bytes(raw)
            assert authority["metadata"]["size_bytes"] == len(raw)
            regeneration = resolve_manifest_regeneration(
                EvaluationRunManifest.model_validate_json(raw),
                root=Path(tasks[0]["manifest_root"]) / completed["batch_id"],
            )
            assert regeneration.parameters["row_id"] == outcome["row_id"]
            assert regeneration.parameters["batch_execution"] == {
                "mode": "governed_ordered_partition",
                "batch_id": completed["batch_id"],
                "ordered_row_ids": [item["row_id"] for item in completed["outcomes"]],
            }
            manifest = json.loads(raw)
            del manifest["created_at"]
            cache = manifest["metadata"]["cache"]
            cache_bytes = Path(cache["states_path"]).read_bytes()
            assert sha256_bytes(cache_bytes) == cache["states_cache_sha256"]
            rows.append((outcome, manifest, cache_bytes))
        return rows

    initial = _execute_evaluation_batch_partition_group(tasks, ())
    grouped = tuple(map(snapshot, initial))
    expected_loads = (
        []
        if accepts_fresh_states
        else [outcome["manifest_id"] for batch in grouped for outcome, _, _ in batch]
    )
    assert loaded_manifest_ids == expected_loads
    cached = _execute_evaluation_batch_partition_group(tasks, ())
    assert all(batch["reused_verified_fragments"] for batch in cached)
    assert [batch["fragments"] for batch in cached] == [batch["fragments"] for batch in initial]
    assert loaded_manifest_ids == expected_loads
    loaded_manifest_ids.clear()
    assert materialized_counts == [4]
    shutil.rmtree(tasks[0]["manifest_root"])
    shutil.rmtree(tasks[0]["compaction_root"])
    (first,) = _execute_evaluation_batch_partition_group(tasks[:1], ())
    manifest_path = Path(first["outcomes"][0]["manifest_path"])
    manifest_bytes = manifest_path.read_bytes()
    manifest_path.write_bytes(manifest_bytes + b" ")
    with pytest.raises(ValueError, match="checkpoint parent authority drifted"):
        _execute_evaluation_batch_partition_group(tasks, ())
    assert not (Path(tasks[0]["manifest_root"]) / "0001").exists()
    manifest_path.write_bytes(manifest_bytes)

    resumed = _execute_evaluation_batch_partition_group(tasks, ())
    assert [batch["reused_verified_fragments"] for batch in resumed] == [True, False]
    assert loaded_manifest_ids == expected_loads
    assert materialized_counts == [4, 4, 4]

    assert tuple(tuple(outcome["row_id"] for outcome, _, _ in batch) for batch in grouped) == (
        ordered_batches
    )
    assert [
        manifest["summary_metrics"]["gain"] for batch in grouped for _, manifest, _ in batch
    ] == [0.0, 1.0, 2.0, 3.0]
    assert grouped == tuple(map(snapshot, resumed))

    shutil.rmtree(tasks[0]["manifest_root"])
    shutil.rmtree(tasks[0]["compaction_root"])
    execute = evaluation._execute_evaluation_run_plan

    def execute_and_corrupt_cache(*args, **kwargs):
        result = execute(*args, **kwargs)
        cache_path = Path(result.rows[0].result.metadata["cache"]["states_path"])
        cache_path.write_bytes(cache_path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(evaluation, "_execute_evaluation_run_plan", execute_and_corrupt_cache)
    with pytest.raises(ValueError, match="matching state cache"):
        _execute_evaluation_batch_partition_group(tasks[:1], ())
    assert loaded_manifest_ids == expected_loads + expected_loads[:1]
    assert not (Path(tasks[0]["compaction_root"]) / "fragment-checkpoints").exists()


def test_grouped_worker_reauthenticates_staged_content_before_next_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from feedbax.analysis.evaluation import EvaluationRecipeResult
    from feedbax.analysis.harness import _execute_evaluation_batch_partition_group
    from feedbax.orchestration.input_materialization import (
        InputMaterializationError,
        compile_content_input_projection,
        materialize_content_input_projection,
    )

    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.json").write_bytes(b"{}")
    sealed = seal_staged_root(
        StagedRootSourceBinding("manifests", "manifest-store", source),
        snapshot_parent=tmp_path / "sealed",
    )
    content_store = tmp_path / "content-store"
    binding, *_ = migrate_staged_root_v1(
        sealed.custody, source_root=sealed.staging_root, destination_store=content_store
    )
    bundle, tasks = _grouped_worker_tasks(
        tmp_path, (("gain-00",), ("gain-01",)), content_bindings=(binding,)
    )
    provider = ImmutableArtifactBlobProvider(content_store)
    (staged,) = materialize_content_input_projection(
        compile_content_input_projection(bundle.content_bindings, provider),
        provider,
        destination_root=Path(tasks[0]["inputs_root"]) / "content" / "trees",
    )
    executed = []

    def batch_recipe(items, _context):
        executed.extend(item.row_id for item in items)
        (staged.destination / "fixture.json").write_bytes(b"[]")
        return [EvaluationRecipeResult() for _ in items]

    _register_grouped_worker_recipe(monkeypatch, batch_recipe)

    with pytest.raises(InputMaterializationError, match="materialized content object differs"):
        _execute_evaluation_batch_partition_group(tasks, ())

    assert executed == ["gain-00"]
    assert (Path(tasks[0]["manifest_root"]) / "0000").is_dir()
    assert not (Path(tasks[0]["manifest_root"]) / "0001").exists()
    checkpoints = Path(tasks[0]["compaction_root"]) / "fragment-checkpoints"
    assert sorted(path.name for path in checkpoints.iterdir()) == ["0000.json"]


@pytest.mark.parametrize("drift_before_first_batch", [True, False])
def test_grouped_worker_rejects_changed_source_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift_before_first_batch: bool
) -> None:
    from feedbax.analysis.evaluation import EvaluationRecipeResult
    from feedbax.analysis.harness import _execute_evaluation_batch_partition_group

    source = tmp_path / "source.json"
    _write_json(source, {"value": 1})
    matrix = _sized_matrix(2)
    matrix["sources"] = [{"alias": "source", "kind": "json", "uri": source.name}]
    matrix["derivations"] = [
        {"output_path": "params.derived", "query": {"item": "source", "path": "value"}}
    ]
    _, tasks = _grouped_worker_tasks(tmp_path, (("gain-00",), ("gain-01",)), matrix=matrix)
    executed = []

    def batch_recipe(items, _context):
        executed.extend((item.row_id, item.spec.params["derived"]) for item in items)
        _write_json(source, {"value": 2})
        return [EvaluationRecipeResult() for _ in items]

    _register_grouped_worker_recipe(monkeypatch, batch_recipe)
    if drift_before_first_batch:
        _write_json(source, {"value": 2})
        match = "resolved matrix authority drifted"
    else:
        match = "authoring input changed after it was read"

    with pytest.raises(ValueError, match=match):
        _execute_evaluation_batch_partition_group(tasks, ())

    assert executed == ([] if drift_before_first_batch else [("gain-00", 1)])
    manifest_root = Path(tasks[0]["manifest_root"])
    assert (manifest_root / "0000").is_dir() == (not drift_before_first_batch)
    assert not (manifest_root / "0001").exists()


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
    request = _request(
        tmp_path,
        matrix,
        batch_plan=plan,
        output_preflight=_output_preflight(expected_rows=2),
    ).model_copy(update={"launch_policy": LaunchPolicy(max_parallel_rows=4)})
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
        _assemble(
            _request(
                tmp_path,
                matrix,
                batch_plan=plan,
                output_preflight=_output_preflight(expected_rows=2),
            ),
            tmp_path,
        )


def test_evaluation_matrix_requires_authored_output_adequacy_before_outputs(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, _matrix(), output_preflight=None)
    output_root = Path(request.orchestration_root) / "missing-output-adequacy"

    with pytest.raises(
        ValueError,
        match=r"evaluation_output_preflight.*explicit storage_mode choice",
    ):
        _assemble(request, tmp_path, run_set_id="missing-output-adequacy")

    assert not output_root.exists()
    assert not (tmp_path / "custody").exists()


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

    monkeypatch.setattr(
        assembly_module,
        "shutil",
        SimpleNamespace(disk_usage=disk_usage),
    )
    run_set_id = "disk-refusal"
    output_root = Path(request.orchestration_root) / run_set_id
    registries = new_application_registry_bundle(local_component_source=None)
    engine = StageEngine.from_request(
        request,
        context=AssemblyContext(
            custody_root=tmp_path / "custody",
            repo_root=tmp_path,
            materializer_commit="d9e62cfd" + "0" * 32,
        ),
        registry=_assembly_registry(),
        driver_registry=registries.drivers,
        driver_context=lambda _bundle: pytest.fail("driver constructed before disk refusal"),
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
        registry=_assembly_registry(),
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

    monkeypatch.setattr(
        assembly_module,
        "shutil",
        SimpleNamespace(disk_usage=disk_usage),
    )
    run_set_id = "single-pre-root-decision"
    registries = new_application_registry_bundle(local_component_source=None)
    engine = StageEngine.from_request(
        request,
        context=AssemblyContext(
            custody_root=tmp_path / "custody",
            repo_root=tmp_path,
            materializer_commit="d9e62cfd" + "0" * 32,
        ),
        registry=_assembly_registry(),
        driver_registry=registries.drivers,
        driver_context=lambda bundle: DriverConstructionContext(configuration={"bundle": bundle}),
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


@pytest.mark.parametrize("driver", ["local", "runpod"])
def test_batch_reclamation_preflight_bounds_peak_before_any_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    driver: str,
) -> None:
    from feedbax.orchestration import assembly as assembly_module

    matrix = _sized_matrix(100)
    consumer = EvaluationBatchConsumerDeclaration(
        leaf_id="velocity",
        consumer_id="tests.velocity",
        consumer_version="v1",
        terminal_analysis_type="tests.velocity.analysis",
        parameters={"projection": "velocity"},
        accepted_evaluation_state_schema_ids=("tests.states.v1",),
        compact_product_schema_id="tests.velocity.product",
        compact_product_schema_version="tests.velocity.product.v1",
        compact_product_role="velocity_result",
        merge_state_schema_id="tests.velocity.merge",
        merge_state_schema_version="tests.velocity.merge.v1",
    )
    plan = EvaluationMatrixBatchPlan(
        matrix_intent_hash=evaluation_matrix_intent_hash(matrix),
        batches=tuple(
            EvaluationMatrixBatchUnit(
                batch_id=f"{index:04d}",
                ordered_row_ids=tuple(
                    f"gain-{row:02d}" for row in range(index * 10, (index + 1) * 10)
                ),
                required_leaf_ids=("velocity",),
            )
            for index in range(10)
        ),
        consumers=(consumer,),
    )
    policy = EvaluationOutputPreflightPolicy(
        expected_resolved_row_count=100,
        retained_bytes_per_resolved_row=10,
        retained_bytes_per_resolved_row_source="fixture",
        planned_repetitions=2,
        storage_mode="batch_reclamation",
        estimated_compact_retained_bytes=100,
        required_free_space_reserve_bytes=50,
    )
    request = _request(
        tmp_path,
        matrix,
        driver=driver,
        batch_plan=plan,
        output_preflight=policy,
    ).model_copy(update={"launch_policy": LaunchPolicy(max_parallel_rows=4)})
    monkeypatch.setattr(
        assembly_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=949),
    )
    output_root = Path(request.orchestration_root) / "bounded"
    with pytest.raises(ValueError, match="required_free_bytes=950"):
        _assemble(request, tmp_path, run_set_id="bounded")
    assert not output_root.exists()
    assert not (tmp_path / "custody").exists()

    monkeypatch.setattr(
        assembly_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=950),
    )
    bundle = _assemble(request, tmp_path, run_set_id="bounded-pass")
    evidence = bundle.evaluation_output_preflight
    assert evidence is not None
    assert evidence.storage_mode == "batch_reclamation"
    assert evidence.active_batch_count == 4
    assert evidence.max_rows_per_active_batch == 10
    assert evidence.estimated_retained_bytes == 900
    assert evidence.required_free_bytes == 950
    assert evidence.estimated_retained_bytes < 100 * 10 * 2
    assert bundle.rows[0].launch.metadata["batch_plan"]["consumers"][0]["leaf_id"] == "velocity"


@pytest.mark.parametrize(
    ("batch_plan", "message"),
    [
        (None, "evaluation_batch_plan is required"),
        ("without-consumers", "declared terminal consumers"),
    ],
)
def test_batch_reclamation_requires_authored_plan_and_terminal_consumers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_plan: EvaluationMatrixBatchPlan | str | None,
    message: str,
) -> None:
    from feedbax.orchestration import assembly as assembly_module

    matrix = _matrix()
    if batch_plan == "without-consumers":
        batch_plan = EvaluationMatrixBatchPlan(
            matrix_intent_hash=evaluation_matrix_intent_hash(matrix),
            batches=(
                EvaluationMatrixBatchUnit(
                    batch_id="whole-matrix",
                    ordered_row_ids=("gain-a", "gain-b"),
                ),
            ),
        )
    policy = EvaluationOutputPreflightPolicy(
        expected_resolved_row_count=2,
        retained_bytes_per_resolved_row=100,
        retained_bytes_per_resolved_row_source="fixture",
        planned_repetitions=1,
        storage_mode="batch_reclamation",
        required_free_space_reserve_bytes=0,
    )
    monkeypatch.setattr(
        assembly_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=10_000),
    )

    with pytest.raises(ValueError, match=message):
        _assemble(
            _request(
                tmp_path,
                matrix,
                batch_plan=batch_plan,
                output_preflight=policy,
            ),
            tmp_path,
        )


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
    bundle = _assemble(
        _request(
            tmp_path,
            delta,
            output_preflight=_output_preflight(expected_rows=2),
        ),
        tmp_path,
        run_set_id="delta",
    )
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
    request = _request(
        tmp_path,
        matrix,
        output_preflight=_output_preflight(expected_rows=2),
    )
    legacy_request = request.model_dump(mode="json")
    legacy_request["staged_roots"] = []
    with pytest.raises(ValueError, match="RunAssemblyRequest v9 does not admit staged_roots"):
        RunAssemblyRequest.model_validate(legacy_request)
    with pytest.raises(ValueError, match="lack canonical content bindings"):
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
    content_store = tmp_path / "content-store"
    bindings = [
        migrate_staged_root_v1(
            item.custody,
            source_root=item.staging_root,
            destination_store=content_store,
        )[0]
        for item in sealed
    ]
    governed = request.model_copy(update={"content_bindings": bindings})
    bundle = _assemble(governed, tmp_path, run_set_id="governed")
    legacy_bundle = bundle.model_dump(mode="json")
    legacy_bundle["staged_roots"] = []
    with pytest.raises(ValueError, match="RunBundle v14 does not admit staged_roots"):
        type(bundle).model_validate(legacy_bundle)

    assert [(item.root_kind, item.binding_name) for item in bundle.content_bindings] == [
        ("artifact-provider", "artifacts"),
        ("checkpoint-custody", "checkpoints"),
        ("manifest-store", "local"),
    ]


def _write_evaluation_plugin(tmp_path: Path) -> None:
    plugin = tmp_path / "evaluation_plugin.py"
    plugin.write_text(
        """
import os
from pathlib import Path

from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.base import ArtifactRef
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.analysis import (
    EvaluationBatchFragment,
    EvaluationBatchFinalizeInput,
    EvaluationBatchMergeState,
)
from feedbax.plugins import (
    EVALUATION_BATCH_CONSUMERS,
    EVALUATION_RECIPES,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
)

def _recipe(_spec, _root, _states_path, _context):
    return EvaluationRecipeResult(summary_metrics={"ok": 1.0})

def _batch(items, context):
    probe = Path(context.repo_root) / "retry-probe"
    if probe.is_dir():
        with (probe / "recipe-calls").open("a") as calls:
            for item in items:
                calls.write(item.row_id + "\\n")
    return [
        EvaluationRecipeResult(
            states={"gain": item.spec.params["gain"]},
            summary_metrics={"gain": item.spec.params["gain"]},
            metadata={"states_schema": "tests.diagnostic.evaluation.v1"},
        )
        for item in items
    ]

def _compact(value):
    leaf = value.parameters["leaf"]
    repo_root = Path(value.execution_context.repo_root)
    observed_roots = repo_root / "observed-worker-roots"
    observed_roots.mkdir(exist_ok=True)
    (observed_roots / str(os.getpid())).write_text(str(repo_root), encoding="utf-8")
    probe = repo_root / "retry-probe"
    if probe.is_dir():
        assert [state["gain"] for state in value.states] == [0.0, 1.0]
        with (probe / "compact-calls").open("a") as calls:
            calls.write(",".join(value.batch.ordered_row_ids) + "\\n")
        interrupt = probe / "interrupt"
        if interrupt.is_file():
            interrupt.unlink()
            raise RuntimeError("consumer interrupted after row publication")
    else:
        assert value.execution_context.artifact_provider("artifacts").root.is_dir()
    assert len(value.parent_authorities) == len(value.outcomes)
    assert all(
        parent.metadata["ref_schema_version"] == "feedbax.ref.authenticated_manifest.v1"
        for parent in value.parent_authorities
    )
    if value.parameters.get("durable_probe") and leaf == "trajectory":
        (state,) = value.states
        (row_id,) = value.batch.ordered_row_ids
        assert state["gain"] == float(row_id.removeprefix("gain-"))
        (manifest,) = value.manifests
        (artifact,) = (
            ArtifactRef.model_validate(item) for item in manifest["artifacts"]
            if item["role"] == "evaluation_states"
        )
        batch_root = Path(value.outcomes[0].manifest_path).parents[2]
        provider = ImmutableArtifactBlobProvider(batch_root)
        state_bytes = provider.get_bytes(artifact.artifact_id, size_bytes=artifact.size_bytes)
        if row_id == "gain-00":
            target = ContentRetentionTarget(ref_kind="object", object=provider.store_object(state_bytes))
            provider.register_root(
                ContentRootRecord(root_id="test:scientific-pin", root_kind="logical-ref", targets=(target,))
            )
        if row_id == "gain-07":
            # Both copies deliberately have the same digest as a disposable state.
            ImmutableArtifactBlobProvider(batch_root.parent).store_object(state_bytes)
            ImmutableArtifactBlobProvider(batch_root.with_name("0003")).store_object(state_bytes)
    return EvaluationBatchFragment(
        payload={"leaf": leaf, "rows": list(value.batch.ordered_row_ids)},
        schema_id=f"tests.{leaf}.product",
        schema_version=f"tests.{leaf}.product.v1",
        role=f"{leaf}_result",
    )

def _register(context):
    context.registry(EVALUATION_RECIPES).register(
        "tests.evaluation_orchestration",
        _recipe,
        batch_recipe=_batch,
    )
    context.registry(EVALUATION_BATCH_CONSUMERS).register(
        "tests.projector",
        "v1",
        compact=_compact,
        merge=lambda value: EvaluationBatchMergeState(
            payload={
                "leaf": value.parameters["leaf"],
                "rows": [
                    *(
                        []
                        if value.prior_merge_state is None
                        else value.prior_merge_state["rows"]
                    ),
                    *value.fragment["rows"],
                ],
            },
            schema_id=f"tests.{value.parameters['leaf']}.merge",
            schema_version=f"tests.{value.parameters['leaf']}.merge.v1",
        ),
        finalize=lambda value: EvaluationBatchFragment(
            payload=value.terminal_merge_state,
            schema_id=f"tests.{value.parameters['leaf']}.product",
            schema_version=f"tests.{value.parameters['leaf']}.product.v1",
            role=f"{value.parameters['leaf']}_result",
        ),
    )

PLUGIN_REGISTRATION = PluginRegistration(
    declaration=PluginDeclaration(
        plugin_id="tests.evaluation_plugin",
        version="1",
        downstream_protocol_version=1,
        families=(
            FamilyRequirement(EVALUATION_RECIPES.family),
            FamilyRequirement(EVALUATION_BATCH_CONSUMERS.family),
        ),
    ),
    register=_register,
)
""".strip(),
        encoding="utf-8",
    )


def test_governed_harness_retries_compaction_after_row_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from feedbax.analysis.harness import main
    from feedbax.contracts.artifact_custody import ArtifactBlobIntegrityError
    from feedbax.contracts.base import ArtifactRef

    leaf = "trajectory"
    declaration = EvaluationBatchConsumerDeclaration(
        leaf_id=leaf,
        consumer_id="tests.projector",
        consumer_version="v1",
        terminal_analysis_type=f"tests.{leaf}.analysis",
        accepted_evaluation_state_schema_ids=("tests.diagnostic.evaluation.v1",),
        compact_product_schema_id=f"tests.{leaf}.product",
        compact_product_schema_version=f"tests.{leaf}.product.v1",
        compact_product_role=f"{leaf}_result",
        merge_state_schema_id=f"tests.{leaf}.merge",
        merge_state_schema_version=f"tests.{leaf}.merge.v1",
        parameters={"leaf": leaf},
    )
    rows = ("gain-00", "gain-01")
    _, (task,) = _grouped_worker_tasks(tmp_path, (rows,), consumers=(declaration,))
    _write_evaluation_plugin(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    probe = tmp_path / "retry-probe"
    probe.mkdir()
    (probe / "interrupt").touch()
    lifecycle = tmp_path / "lifecycle.json"
    compaction = tmp_path / "evaluation-batch-compaction"
    completion = compaction / "fragment-checkpoints" / "0000.json"
    argv = [
        task["payload_path"],
        "--batch",
        "--plugin",
        "evaluation_plugin",
        "--manifest-root",
        task["manifest_root"],
        "--repo-root",
        task["repo_root"],
        "--orchestration-bundle",
        task["bundle_path"],
        "--orchestration-inputs-root",
        task["inputs_root"],
        "--orchestration-row-id",
        task["orchestration_row_id"],
        "--lifecycle-result",
        str(lifecycle),
    ]

    with pytest.raises(RuntimeError, match="consumer interrupted after row publication"):
        main(argv)

    assert not completion.exists()
    assert not lifecycle.exists()
    store = Path(task["manifest_root"]) / "0000"
    row_paths = sorted((store / "manifests").glob("*/*.json"))
    assert len(row_paths) == len(rows)
    row_digests = {path: sha256_bytes(path.read_bytes()) for path in row_paths}
    matrix = Path(task["payload_path"])
    matrix_digest = sha256_bytes(matrix.read_bytes())
    assert (probe / "recipe-calls").read_text().splitlines() == list(rows)
    assert (probe / "compact-calls").read_text().splitlines() == [",".join(rows)]

    row_manifest = json.loads(row_paths[0].read_bytes())
    (resolved_row,) = (
        item for item in row_manifest["artifacts"] if item["role"] == "resolved_row_spec"
    )
    blob = Path(resolved_row["uri"])
    original_blob = blob.rename(tmp_path / "original-row-spec")
    blob.write_bytes(b" " * original_blob.stat().st_size)
    with pytest.raises(ArtifactBlobIntegrityError):
        main(argv)
    original_blob.replace(blob)

    cache = Path(row_manifest["metadata"]["cache"]["states_path"])
    cache_bytes = cache.read_bytes()
    cache.write_bytes(cache_bytes + b" ")
    with pytest.raises(ValueError, match=r"state cache|states cache"):
        main(argv)
    assert not completion.exists()
    assert not lifecycle.exists()
    assert (probe / "recipe-calls").read_text().splitlines() == list(rows)
    assert (probe / "compact-calls").read_text().splitlines() == [",".join(rows)]
    cache.write_bytes(cache_bytes)

    start_retry = Barrier(2, timeout=10)

    def retry() -> int:
        start_retry.wait()
        return main(argv)

    with ThreadPoolExecutor(max_workers=2) as pool:
        retries = [pool.submit(retry) for _ in range(2)]
        assert [result.result(timeout=30) for result in retries] == [0, 0]

    assert (probe / "recipe-calls").read_text().splitlines() == list(rows)
    assert (probe / "compact-calls").read_text().splitlines() == [",".join(rows)] * 2
    assert not cache.exists()
    assert {path: sha256_bytes(path.read_bytes()) for path in row_paths} == row_digests
    assert sha256_bytes(matrix.read_bytes()) == matrix_digest
    receipt = json.loads(completion.read_bytes())
    assert receipt["outcomes"] == json.loads(lifecycle.read_bytes())["outcomes"]
    assert [outcome["row_id"] for outcome in receipt["outcomes"]] == list(rows)
    assert {
        Path(outcome["manifest_path"]): authority["metadata"]["manifest_sha256"]
        for outcome, authority in zip(
            receipt["outcomes"], receipt["parent_authorities"], strict=True
        )
    } == row_digests
    (fragment,) = receipt["fragments"]
    assert json.loads(
        ImmutableArtifactBlobProvider(compaction).get_bytes(ArtifactRef.model_validate(fragment))
    ) == {"leaf": leaf, "rows": list(rows)}


@pytest.mark.parametrize("disposition", ["immediate", "grace"])
def test_provider_free_cli_shadow_reaches_terminal_collection_in_fresh_process(
    tmp_path: Path,
    subprocess_dirty_tolerance,
    disposition: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_evaluation_plugin(tmp_path)
    dist_info = tmp_path / "evaluation_plugin-0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Name: evaluation-plugin\nVersion: 0.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[feedbax.plugins]\nevaluation-test = evaluation_plugin:PLUGIN_REGISTRATION\n",
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
    content_pinned_base = {
        "evaluation_type": "tests.evaluation_orchestration",
        "params": {"gain": 0.0, "states_custody": "durable"},
    }
    base_path = tmp_path / "content-pinned-base.json"
    base_path.write_text(json.dumps(content_pinned_base), encoding="utf-8")
    shadow_matrix = {
        "schema_id": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "base": {
            "ref": base_path.name,
            "sha256": sha256_bytes(canonical_json_bytes(content_pinned_base)),
        },
        "axes": [
            {
                "id": "gain",
                "values": [
                    {
                        "id": f"{index:02d}",
                        "deltas": [{"path": "params.gain", "value": float(index)}],
                    }
                    for index in range(8)
                ],
            }
        ],
    }
    consumers = tuple(
        EvaluationBatchConsumerDeclaration(
            leaf_id=leaf,
            consumer_id="tests.projector",
            consumer_version="v1",
            terminal_analysis_type=f"tests.{leaf}.analysis",
            parameters={"leaf": leaf, "durable_probe": True},
            accepted_evaluation_state_schema_ids=("tests.diagnostic.evaluation.v1",),
            compact_product_schema_id=f"tests.{leaf}.product",
            compact_product_schema_version=f"tests.{leaf}.product.v1",
            compact_product_role=f"{leaf}_result",
            merge_state_schema_id=f"tests.{leaf}.merge",
            merge_state_schema_version=f"tests.{leaf}.merge.v1",
        )
        for leaf in ("trajectory", "velocity")
    )
    content_store = tmp_path / "shadow-content-store"
    bindings = [
        migrate_staged_root_v1(
            item.custody,
            source_root=item.staging_root,
            destination_store=content_store,
        )[0]
        for item in sealed
    ]
    request = _request(
        tmp_path,
        shadow_matrix,
        output_preflight=_output_preflight(expected_rows=8),
    ).model_copy(
        update={
            "content_bindings": bindings,
            "evaluation_batch_plan": EvaluationMatrixBatchPlan(
                matrix_intent_hash=evaluation_matrix_intent_hash(shadow_matrix),
                batches=tuple(
                    EvaluationMatrixBatchUnit(
                        batch_id=f"{index:04d}",
                        ordered_row_ids=(f"gain-{index:02d}",),
                        required_leaf_ids=("trajectory", "velocity"),
                    )
                    for index in range(8)
                ),
                consumers=consumers,
                retention=EvaluationPayloadRetentionPolicy(
                    intermediate_disposition=disposition,
                    intermediate_grace_seconds=60 if disposition == "grace" else None,
                ),
            ),
            "launch_policy": LaunchPolicy(max_parallel_rows=4),
        }
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    feedbax_source_root = Path(__file__).resolve().parents[1]
    subprocess_dirty_tolerance(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(feedbax_source_root)))

    command = [
        sys.executable,
        "-m",
        "feedbax.bin.orchestrate",
        "shadow-launch",
        "--assembly-request",
        str(request_path),
        "--content-store-root",
        str(content_store),
        "--content-lease-seconds",
        "3600",
    ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
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
    assert all(item["ordered_batch_ids"] for item in topology["processes"])
    assert sum(len(item["ordered_batch_ids"]) for item in topology["processes"]) == 8
    assert all(
        [timing["batch_id"] for timing in item["batch_timings"]] == item["ordered_batch_ids"]
        for item in topology["processes"]
    )
    assert all(
        timing["completed_offset_ns"] >= timing["started_offset_ns"] >= 0
        and timing["duration_ns"] == timing["completed_offset_ns"] - timing["started_offset_ns"]
        and not timing["reused_verified_fragments"]
        for item in topology["processes"]
        for timing in item["batch_timings"]
    )
    observed_roots = tmp_path / "observed-worker-roots"
    assert len(list(observed_roots.iterdir())) == 4
    assert {path.read_text(encoding="utf-8") for path in observed_roots.iterdir()} == {
        str(tmp_path)
    }
    assert [
        outcome["diagnostic_schema_ids"]
        for lifecycle in evidence["lifecycles"]
        for outcome in lifecycle["outcomes"]
    ] == [["tests.diagnostic.evaluation.v1"] for _ in range(8)]
    run_root = Path(request.orchestration_root) / evidence["run_set_id"]
    compaction_paths = list(run_root.rglob("evaluation-batch-compaction.json"))
    assert compaction_paths
    assert len({path.read_bytes() for path in compaction_paths}) == 1
    compaction = json.loads(compaction_paths[0].read_text())
    assert (
        compaction["schema_version"]
        == "feedbax.orchestration.evaluation_batch_compaction_evidence.v4"
    )
    assert compaction["declared_leaf_ids"] == ["trajectory", "velocity"]
    assert [item["batch_id"] for item in compaction["reclamations"]] == [
        f"{index:04d}" for index in range(8)
    ]
    assert [item["metadata"]["analysis_type"] for item in compaction["terminal_products"]] == [
        "tests.trajectory.analysis",
        "tests.velocity.analysis",
    ]
    assert compaction["retention"]["retained_output_ids"] == ["trajectory", "velocity"]
    assert compaction["retention"]["disposable_output_ids"] == []
    assert compaction["retention"]["temporary_lease_ids"]
    row_id = evidence["lifecycles"][0]["orchestration_row_id"]
    source_root = run_root / "rows" / row_id
    evaluation_root = source_root / "evaluation"
    collected_store = run_root / "collected" / row_id / "evaluation-batch-compaction"
    terminal = EvaluationBatchCompactionEvidence.model_validate(compaction)
    outcomes = evidence["lifecycles"][0]["outcomes"]

    def read_release_lease(batch):
        provider_root = evaluation_root / batch.batch_id
        (lease_id,) = batch.retention_record_ids
        index = json.loads((provider_root / "retention" / "index.json").read_bytes())
        return next(lease for lease in index["leases"] if lease["lease_id"] == lease_id)

    if disposition == "grace":
        ready_path = source_root / "evaluation-payload-retention-ready"
        assert ready_path.is_file()
        for batch in terminal.reclamations:
            provider_root = evaluation_root / batch.batch_id
            (job_path,) = (provider_root / "retention" / "automatic-collections").glob("*.json")
            job = json.loads(job_path.read_bytes())
            assert job["root"] == str(provider_root)
            assert job["lease_ids"] == list(batch.retention_record_ids)
            assert job["ready_path"] == str(ready_path)
            lease = read_release_lease(batch)
            assert not lease["released"]
            (target,) = lease["targets"]
            assert ImmutableArtifactBlobProvider(provider_root).get_object(
                ContentObjectRef.model_validate(target["object"])
            )

        after_grace = datetime.now(timezone.utc) + timedelta(minutes=2)

        class PostGraceClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return after_grace.astimezone(tz)

        with monkeypatch.context() as clock:
            clock.setattr(content_retention, "datetime", PostGraceClock)
            for job_path in run_root.glob("**/retention/automatic-collections/*.json"):
                job = json.loads(job_path.read_bytes())
                content_retention.run_scheduled_lease_collection(
                    Path(job["root"]),
                    tuple(job["lease_ids"]),
                    ready_path=Path(job["ready_path"]) if job["ready_path"] else None,
                    fallback_at=(
                        datetime.fromisoformat(job["fallback_at"]) if job["fallback_at"] else None
                    ),
                )

    if disposition == "immediate":
        source_store = source_root / "evaluation-batch-compaction"
        wrong_provider = tmp_path / "wrong-provider"
        shutil.copytree(evaluation_root / "0000", wrong_provider / "0000")
        with pytest.raises(ValueError, match="different batch provider"):
            activate_evaluation_payload_collection(
                terminal,
                compaction_root=source_store,
                evaluation_root=wrong_provider,
            )

        first, second, *others = terminal.reclamations
        (first_lease,) = first.retention_record_ids
        (second_lease,) = second.retention_record_ids
        swapped_ids = {
            f"evaluation/{first_lease}": f"evaluation/{second_lease}",
            f"evaluation/{second_lease}": f"evaluation/{first_lease}",
        }
        assert terminal.retention is not None
        swapped = terminal.model_copy(
            update={
                "reclamations": (
                    first.model_copy(update={"retention_record_ids": second.retention_record_ids}),
                    second.model_copy(update={"retention_record_ids": first.retention_record_ids}),
                    *others,
                ),
                "retention": terminal.retention.model_copy(
                    update={
                        "temporary_lease_ids": tuple(
                            swapped_ids.get(lease_id, lease_id)
                            for lease_id in terminal.retention.temporary_lease_ids
                        )
                    }
                ),
            }
        )
        with pytest.raises(ValueError, match="does not belong to its batch provider"):
            activate_evaluation_payload_collection(
                swapped,
                compaction_root=source_store,
                evaluation_root=evaluation_root,
            )

    artifacts = []
    for batch, outcome in zip(terminal.reclamations, outcomes, strict=True):
        lease = read_release_lease(batch)
        assert lease["released"]
        manifest = json.loads(Path(outcome["manifest_path"]).read_bytes())
        (artifact,) = (
            ArtifactRef.model_validate(item)
            for item in manifest["artifacts"]
            if item["role"] == "evaluation_states"
        )
        assert [target["object"]["sha256"] for target in lease["targets"]] == [artifact.sha256]
        provider = ImmutableArtifactBlobProvider(evaluation_root / batch.batch_id)
        if batch.batch_id == "0000":
            assert provider.get_bytes(artifact.artifact_id, size_bytes=artifact.size_bytes)
        else:
            with pytest.raises(FileNotFoundError, match="artifact blob is missing"):
                provider.get_bytes(artifact.artifact_id, size_bytes=artifact.size_bytes)
        artifacts.append(artifact)

    copied = artifacts[-1]
    for root in (evaluation_root, evaluation_root / "0003"):
        assert ImmutableArtifactBlobProvider(root).get_bytes(
            copied.artifact_id, size_bytes=copied.size_bytes
        )
    if disposition == "immediate":
        index = json.loads((collected_store / "retention" / "index.json").read_bytes())
        compact_leases = [
            lease for lease in index["leases"] if lease["owner"] == "feedbax.evaluation.compaction"
        ]
        assert compact_leases
        for lease in compact_leases:
            assert lease["released"]
            for target in lease["targets"]:
                digest = target["object"]["sha256"]
                assert not (collected_store / "artifacts" / "sha256" / digest[:2] / digest).exists()

    for manifest_ref in terminal.terminal_products:
        manifest = AnalysisRunManifest.model_validate_json(
            ImmutableArtifactBlobProvider(collected_store).get_bytes(manifest_ref)
        )
        (product,) = manifest.produced_data
        (product_artifact,) = product.artifacts
        assert json.loads(
            ImmutableArtifactBlobProvider(collected_store / "analysis").get_bytes(product_artifact)
        ) == {"leaf": product.logical_name, "rows": [outcome["row_id"] for outcome in outcomes]}
        assert [parent.manifest_hash for parent in product.parent_manifests] == [
            sha256_bytes(Path(outcome["manifest_path"]).read_bytes()) for outcome in outcomes
        ]

    replay = compaction["retention"]["replay"]
    assert replay["canonical_executor"] == "python -m feedbax matrix-harness"
    run_bundle = json.loads((run_root / "bundle.json").read_text(encoding="utf-8"))
    assert replay["execution_envelope"] == run_bundle["rows"][0]["execution"]
    bundle = RunBundle.model_validate(run_bundle)
    assert executor_family_adapter("evaluation-matrix").missing_collection_outputs(
        bundle,
        bundle.rows[0],
        {name: str(collected_store.with_name(name)) for name in EVALUATION_COLLECTION_OUTPUTS},
    ) == []


def test_runpod_dry_run_keeps_the_same_public_matrix_executor(
    tmp_path: Path,
) -> None:
    matrix = _matrix()
    consumer = EvaluationBatchConsumerDeclaration(
        leaf_id="velocity",
        consumer_id="tests.velocity",
        consumer_version="v1",
        terminal_analysis_type="tests.velocity.analysis",
        parameters={"projection": "velocity"},
        accepted_evaluation_state_schema_ids=("tests.states.v1",),
        compact_product_schema_id="tests.velocity.product",
        compact_product_schema_version="tests.velocity.product.v1",
        compact_product_role="velocity_result",
        merge_state_schema_id="tests.velocity.merge",
        merge_state_schema_version="tests.velocity.merge.v1",
    )
    plan = EvaluationMatrixBatchPlan(
        matrix_intent_hash=evaluation_matrix_intent_hash(matrix),
        batches=(
            EvaluationMatrixBatchUnit(
                batch_id="whole-matrix",
                ordered_row_ids=("gain-a", "gain-b"),
                required_leaf_ids=("velocity",),
            ),
        ),
        consumers=(consumer,),
    )
    request = _request(
        tmp_path,
        matrix,
        driver="runpod",
        batch_plan=plan,
        output_preflight=_output_preflight(expected_rows=2),
    )
    bundle = _assemble(request, tmp_path)
    (tmp_path / "local").mkdir()
    local_bundle = _assemble(
        _request(
            tmp_path / "local",
            matrix,
            batch_plan=plan,
            output_preflight=_output_preflight(expected_rows=2),
        ),
        tmp_path / "local",
    )

    from feedbax.orchestration.drivers.runpod import (
        RunPodDriverConfig,
        dry_run_launch_bundle,
    )

    command = dry_run_launch_bundle(bundle, RunPodDriverConfig())[0]
    assert "matrix-harness" in command
    assert "--batch" in command
    assert "--orchestration-inputs-root" in command
    assert "--lifecycle-result" in command
    assert (
        bundle.rows[0].launch.metadata["batch_plan"]
        == local_bundle.rows[0].launch.metadata["batch_plan"]
    )
    assert bundle.rows[0].launch.metadata["batch_plan"]["consumers"][0]["parameters"] == {
        "projection": "velocity"
    }
    assert (
        bundle.rows[0].launch.metadata["batch_plan"]["consumers"][0]["terminal_analysis_type"]
        == "tests.velocity.analysis"
    )
    assert bundle.rows[0].launch.collect == local_bundle.rows[0].launch.collect
