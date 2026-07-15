from __future__ import annotations

from functools import partial
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

import feedbax.contracts.training as training_contracts
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.run_matrix import (
    AuthoredTrainingRow,
    RowLowererIdentity,
    TrainingRowLoweringResult,
    TrainingRunMatrixSpec,
)
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.training import (
    ArtifactPolicySpec,
    ExecutionPolicySpec,
    MethodExtensionsSpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    RiskAggregationSpec,
    RUN_CONTROL_SPEC_SCHEMA_ID,
    RUN_CONTROL_SPEC_SCHEMA_VERSION,
    RunControlSpec,
    TrainingConfig,
    TrainingMethodAuthoringContribution,
    TrainingMethodAuthoringHook,
    TrainingMethodDescriptor,
    TrainingMethodRegistration,
    TrainingMethodRegistry,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_method_contract,
    standard_supervised_update_kernels,
)
from feedbax.training.authoring import (
    TRAINING_METHOD_AUTHORING_LOWERER_IDENTITY,
    TrainingMethodAuthoringError,
    TrainingMethodAuthoringProjectors,
    compile_training_method_authoring,
)
from feedbax.training.run_matrix import materialize_adapted_run_matrix


METHOD_REF = "example/typed/v1"
PAYLOAD_SCHEMA_ID = "example.spec.typed_method"
PAYLOAD_SCHEMA_VERSION = "example.spec.typed_method.v1"


class TypedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    gain: int
    task_name: str


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


def _run_control(**updates: Any) -> RunControlSpec:
    values = {
        "n_batches": 4,
        "batch_size": 2,
        "checkpoint_interval": 2,
        "progress_interval": 1,
    }
    values.update(updates)
    return RunControlSpec(**values)


def _authored_row(payload: dict[str, Any] | None = None) -> AuthoredTrainingRow:
    compact = payload or {"gain": 3, "task_name": "reach"}
    return AuthoredTrainingRow(
        row_id="typed-row",
        row_index=0,
        payload=compact,
        payload_hash=training_spec_sha256(compact),
        seed=7,
        axis_coordinates={"gain": compact.get("gain")},
    )


def _method_contract():
    return standard_supervised_method_contract().model_copy(
        update={
            "method_ref": METHOD_REF,
            "method_payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        }
    )


def _contribution() -> TrainingMethodAuthoringContribution:
    return TrainingMethodAuthoringContribution(
        training_config=TrainingConfig(
            n_batches=999,
            batch_size=999,
            learning_rate=0.025,
            hidden_dim=32,
        ),
        method_extensions=MethodExtensionsSpec(metadata={"method_family": "typed-toy"}),
    )


def _projectors(calls: dict[str, int] | None = None) -> TrainingMethodAuthoringProjectors:
    counters = calls if calls is not None else {}

    def mark(name: str) -> None:
        counters[name] = counters.get(name, 0) + 1

    def graph(payload: TypedPayload) -> dict[str, Any]:
        mark("graph")
        return {"inline": _minimal_graph(), "metadata": {"gain": payload.gain}}

    def task(payload: TypedPayload) -> dict[str, Any]:
        mark("task")
        return {"type": payload.task_name, "params": {"n_steps": 4}}

    def objective(payload: TypedPayload) -> dict[str, Any]:
        mark("objective")
        return {
            "loss": {
                "type": "target_state",
                "label": f"target-{payload.gain}",
                "selector": "port:gain.output",
            }
        }

    def domain(payload: TypedPayload) -> dict[str, Any]:
        mark("domain")
        return {"domain_family": "toy", "authored_gain": payload.gain}

    return TrainingMethodAuthoringProjectors(
        graph=graph,
        task=task,
        objective=objective,
        domain=domain,
    )


@pytest.fixture
def authoring_registry(monkeypatch: pytest.MonkeyPatch):
    registry = TrainingMethodRegistry()
    default_result = object()
    holder: dict[str, Any] = {
        "authoring_calls": 0,
        "row_compiler_calls": 0,
        "mutate_payload": False,
        "return_value": default_result,
    }

    def low_level_lower(_row: AuthoredTrainingRow) -> TrainingRowLoweringResult:
        holder["row_compiler_calls"] += 1
        return TrainingRowLoweringResult(
            execution_payload={"adapter": "low-level-only"},
            lowerer_identities=[
                RowLowererIdentity(
                    lowerer_id="example.typed_method.low_level",
                    lowerer_version="example.typed_method.low_level.v1",
                )
            ],
        )

    def author(payload: TypedPayload) -> TrainingMethodAuthoringContribution:
        holder["authoring_calls"] += 1
        if holder["mutate_payload"]:
            payload.gain += 1
        result = holder["return_value"]
        return _contribution() if result is default_result else result

    registry.register_descriptor(
        TrainingMethodDescriptor(
            method_ref=METHOD_REF,
            payload_schema_id=PAYLOAD_SCHEMA_ID,
            payload_schema_version=PAYLOAD_SCHEMA_VERSION,
            payload_model=TypedPayload,
            contract_compiler=lambda _payload: _method_contract(),
            update_kernels_factory=standard_supervised_update_kernels,
            row_compiler=low_level_lower,
            authoring_hook=TrainingMethodAuthoringHook(
                lowerer_id="example.typed_method.authoring",
                lowerer_version="example.typed_method.authoring.v1",
                compile=author,
            ),
            owner="tests.test_training_authoring",
            package="example",
        )
    )
    monkeypatch.setattr(training_contracts, "DEFAULT_TRAINING_METHOD_REGISTRY", registry)
    envelope = MethodPayloadEnvelope(
        schema_id=PAYLOAD_SCHEMA_ID,
        schema_version=PAYLOAD_SCHEMA_VERSION,
        payload={"gain": 3, "task_name": "reach"},
    )
    resolved = registry.resolve_execution(
        MethodRefSpec(package="example", name="typed", version="v1"),
        envelope,
    )
    holder["worker"] = WorkerExecutionSpec(
        method_contract=resolved.contract,
        effective_phase=resolved.effective_phase,
    )
    holder["hook_identity"] = RowLowererIdentity(
        lowerer_id="example.typed_method.authoring",
        lowerer_version="example.typed_method.authoring.v1",
    )
    return registry, holder


def test_run_control_v1_is_strict_and_continuation_is_segment_local() -> None:
    control = _run_control()

    assert control.schema_id == RUN_CONTROL_SPEC_SCHEMA_ID
    assert control.schema_version == RUN_CONTROL_SPEC_SCHEMA_VERSION
    with pytest.raises(ValueError, match="valid integer"):
        _run_control(n_batches="4")
    with pytest.raises(ValueError, match="greater than 0"):
        _run_control(progress_interval=0)
    with pytest.raises(ValueError, match="additional_batches"):
        _run_control(
            continuation={
                "source_completed_batches": 5,
                "additional_batches": 3,
            }
        )


def test_run_control_registry_accepts_current_and_rejects_old_unknown_versions() -> None:
    payload = _run_control().model_dump(mode="json", exclude_none=True)
    result = default_spec_registry.migrate("RunControlSpec", payload)

    assert result.payload == payload
    assert result.migration_records == []
    for version in (
        "feedbax.spec.training.run_control.v0",
        "feedbax.spec.training.run_control.v99",
    ):
        with pytest.raises(UnsupportedSpecVersion):
            default_spec_registry.migrate(
                "RunControlSpec",
                {**payload, "schema_version": version},
            )


def test_compile_authoring_projects_once_and_returns_canonical_contracts(
    authoring_registry,
) -> None:
    _registry, holder = authoring_registry
    calls: dict[str, int] = {}
    row = _authored_row()
    before = row.model_dump(mode="python")

    compiled = compile_training_method_authoring(
        row,
        method_ref=METHOD_REF,
        run_control=_run_control(),
        projectors=_projectors(calls),
    )

    assert calls == {"graph": 1, "task": 1, "objective": 1, "domain": 1}
    assert holder["authoring_calls"] == 1
    assert holder["row_compiler_calls"] == 0
    assert row.model_dump(mode="python") == before
    assert compiled.run_spec.schema_version == "feedbax.spec.training_run.v3"
    assert compiled.worker_execution == holder["worker"]
    assert compiled.run_spec.metadata == {"domain_family": "toy", "authored_gain": 3}
    assert compiled.run_spec.training_config.n_batches == 4
    assert compiled.run_spec.training_config.batch_size == 2
    assert compiled.run_spec.training_config.learning_rate == 0.025
    assert compiled.run_spec.training_config.hidden_dim == 32
    assert compiled.run_spec.method_extensions == MethodExtensionsSpec(
        metadata={"method_family": "typed-toy"}
    )
    assert set(TrainingMethodAuthoringContribution.model_fields) == {
        "training_config",
        "method_extensions",
    }
    assert compiled.lowering_result.execution_payload == compiled.run_spec.model_dump(
        mode="json", exclude_none=True
    )
    assert compiled.lowering_result.lowerer_identities == [
        TRAINING_METHOD_AUTHORING_LOWERER_IDENTITY,
        holder["hook_identity"],
    ]
    assert TrainingRowLoweringResult.model_validate(compiled) == compiled.lowering_result


def test_compile_authoring_owns_default_and_explicit_policies(authoring_registry) -> None:
    defaulted = compile_training_method_authoring(
        _authored_row(),
        method_ref=METHOD_REF,
        run_control=_run_control(),
        projectors=_projectors(),
    )

    assert defaulted.run_spec.execution == ExecutionPolicySpec()
    assert defaulted.run_spec.artifacts == ArtifactPolicySpec()
    assert defaulted.run_spec.risk_aggregation == RiskAggregationSpec()
    assert defaulted.run_spec.checkpoint_progress.checkpoint_interval == 2
    assert defaulted.run_spec.checkpoint_progress.progress_interval == 1

    execution = ExecutionPolicySpec(mode="remote", allow_cloud=True)
    artifacts = ArtifactPolicySpec(custody="mandible", artifact_root="artifacts")
    risk = RiskAggregationSpec(realization="mean", replicate="max")
    explicit = compile_training_method_authoring(
        _authored_row(),
        method_ref=METHOD_REF,
        run_control=_run_control(),
        projectors=_projectors(),
        execution=execution,
        artifacts=artifacts,
        risk_aggregation=risk,
    )

    assert explicit.run_spec.execution == execution
    assert explicit.run_spec.artifacts == artifacts
    assert explicit.run_spec.risk_aggregation == risk

    with pytest.raises(TrainingMethodAuthoringError, match="exact ExecutionPolicySpec"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
            execution={"mode": "dry_run"},  # type: ignore[arg-type]
        )


def test_compiler_result_is_consumed_as_an_ordinary_row_lowerer(
    authoring_registry,
    tmp_path,
) -> None:
    matrix = TrainingRunMatrixSpec.model_validate(
        {
            "name": "typed authoring",
            "base": {
                "kind": "inline",
                "inline": {"gain": 3, "task_name": "reach"},
            },
            "rows": [{"row_id": "typed", "overrides": []}],
        }
    )
    lowerer = partial(
        compile_training_method_authoring,
        method_ref=METHOD_REF,
        run_control=_run_control(),
        projectors=_projectors(),
    )

    materialized = materialize_adapted_run_matrix(
        matrix,
        repo_root=tmp_path,
        row_lowerer=lowerer,
        row_validator=lambda payload, _row_id: TrainingRunSpec.model_validate(payload),
    )

    row = materialized.rows[0]
    assert row.spec is not None
    assert row.spec.method_ref.key == METHOD_REF
    assert row.provenance.lowerer_identities[0] == TRAINING_METHOD_AUTHORING_LOWERER_IDENTITY


def test_authoring_rejects_missing_schema_low_level_and_missing_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low_level = TrainingMethodRegistry()
    low_level.register(
        TrainingMethodRegistration(
            method_ref=METHOD_REF,
            payload_schema_id=PAYLOAD_SCHEMA_ID,
            payload_schema_version=PAYLOAD_SCHEMA_VERSION,
            payload_model=TypedPayload,
            contract_factory=_method_contract,
            update_kernels_factory=standard_supervised_update_kernels,
        )
    )
    monkeypatch.setattr(training_contracts, "DEFAULT_TRAINING_METHOD_REGISTRY", low_level)
    with pytest.raises(TrainingMethodAuthoringError, match="low-level-only"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
        )

    row_compiler_only = TrainingMethodRegistry()
    row_compiler_calls = 0

    def low_level_row_compiler(_row: AuthoredTrainingRow) -> TrainingRowLoweringResult:
        nonlocal row_compiler_calls
        row_compiler_calls += 1
        return TrainingRowLoweringResult(
            execution_payload={"adapter": True},
            lowerer_identities=[
                RowLowererIdentity(lowerer_id="example.low", lowerer_version="example.low.v1")
            ],
        )

    row_compiler_only.register_descriptor(
        TrainingMethodDescriptor(
            method_ref=METHOD_REF,
            payload_schema_id=PAYLOAD_SCHEMA_ID,
            payload_schema_version=PAYLOAD_SCHEMA_VERSION,
            payload_model=TypedPayload,
            contract_compiler=lambda _payload: _method_contract(),
            update_kernels_factory=standard_supervised_update_kernels,
            row_compiler=low_level_row_compiler,
        )
    )
    monkeypatch.setattr(
        training_contracts,
        "DEFAULT_TRAINING_METHOD_REGISTRY",
        row_compiler_only,
    )
    with pytest.raises(TrainingMethodAuthoringError, match="has no authoring_hook"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
        )
    assert row_compiler_calls == 0

    with pytest.raises(TrainingMethodAuthoringError, match="explicit schema_id"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control={"n_batches": 4, "batch_size": 2},
            projectors=_projectors(),
        )


def test_authoring_rejects_projector_and_reserved_domain_errors(authoring_registry) -> None:
    projectors = _projectors()
    broken = TrainingMethodAuthoringProjectors(
        graph=projectors.graph,
        task=projectors.task,
        objective=projectors.objective,
        domain=lambda _payload: {"method_ref": "collision"},
    )
    with pytest.raises(TrainingMethodAuthoringError, match="reserved metadata"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=broken,
        )

    broken = TrainingMethodAuthoringProjectors(
        graph=lambda _payload: {"inline": {"nodes": "invalid"}},
        task=projectors.task,
        objective=projectors.objective,
        domain=projectors.domain,
    )
    with pytest.raises(TrainingMethodAuthoringError, match="graph projector returned invalid"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=broken,
        )


def test_authoring_rejects_hook_mutation_and_untyped_contribution(
    authoring_registry,
) -> None:
    _registry, holder = authoring_registry
    holder["mutate_payload"] = True
    with pytest.raises(TrainingMethodAuthoringError, match="mutated"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
        )
    assert holder["authoring_calls"] == 1
    assert holder["row_compiler_calls"] == 0

    holder["mutate_payload"] = False
    holder["return_value"] = {
        "training_config": TrainingConfig(),
        "method_extensions": MethodExtensionsSpec(),
    }
    with pytest.raises(TrainingMethodAuthoringError, match="exact.*Contribution"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
        )
    assert holder["authoring_calls"] == 2
    assert holder["row_compiler_calls"] == 0


def test_invalid_contribution_fails_before_matrix_materialization_writes(
    authoring_registry,
    tmp_path,
) -> None:
    _registry, holder = authoring_registry
    holder["return_value"] = {"training_config": {}}
    matrix = TrainingRunMatrixSpec.model_validate(
        {
            "name": "invalid typed authoring",
            "base": {
                "kind": "inline",
                "inline": {"gain": 3, "task_name": "reach"},
            },
            "rows": [{"row_id": "typed", "overrides": []}],
        }
    )
    lowerer = partial(
        compile_training_method_authoring,
        method_ref=METHOD_REF,
        run_control=_run_control(),
        projectors=_projectors(),
    )

    with pytest.raises(TrainingMethodAuthoringError, match="exact.*Contribution"):
        materialize_adapted_run_matrix(
            matrix,
            repo_root=tmp_path,
            row_lowerer=lowerer,
            row_validator=lambda payload, _row_id: TrainingRunSpec.model_validate(payload),
        )

    assert list(tmp_path.iterdir()) == []


def test_authoring_rejects_invalid_method_payload_before_projectors(authoring_registry) -> None:
    calls: dict[str, int] = {}
    with pytest.raises(TrainingMethodAuthoringError, match="method payload schema"):
        compile_training_method_authoring(
            _authored_row({"gain": "3", "task_name": "reach"}),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(calls),
        )

    assert calls == {}


def test_authoring_rejects_noncanonical_authored_payload_hash(authoring_registry) -> None:
    row = _authored_row().model_copy(update={"payload_hash": "0" * 64})

    with pytest.raises(TrainingMethodAuthoringError, match="payload_hash"):
        compile_training_method_authoring(
            row,
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
        )


def test_authoring_rejects_reserved_duplicate_hook_identity_before_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def author(_payload: TypedPayload) -> TrainingMethodAuthoringContribution:
        nonlocal calls
        calls += 1
        return _contribution()

    registry = TrainingMethodRegistry()
    registry.register_descriptor(
        TrainingMethodDescriptor(
            method_ref=METHOD_REF,
            payload_schema_id=PAYLOAD_SCHEMA_ID,
            payload_schema_version=PAYLOAD_SCHEMA_VERSION,
            payload_model=TypedPayload,
            contract_compiler=lambda _payload: _method_contract(),
            update_kernels_factory=standard_supervised_update_kernels,
            authoring_hook=TrainingMethodAuthoringHook(
                lowerer_id=TRAINING_METHOD_AUTHORING_LOWERER_IDENTITY.lowerer_id,
                lowerer_version=TRAINING_METHOD_AUTHORING_LOWERER_IDENTITY.lowerer_version,
                compile=author,
            ),
            owner="tests.test_training_authoring",
            package="example",
        )
    )
    monkeypatch.setattr(training_contracts, "DEFAULT_TRAINING_METHOD_REGISTRY", registry)

    with pytest.raises(
        TrainingMethodAuthoringError,
        match="duplicates the reserved authoring compiler identity",
    ):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
        )
    assert calls == 0


@pytest.mark.parametrize("field", ["lowerer_id", "lowerer_version"])
def test_authoring_hook_rejects_empty_identity(field: str) -> None:
    values = {
        "lowerer_id": "example.typed_method.authoring",
        "lowerer_version": "example.typed_method.authoring.v1",
        "compile": lambda _payload: _contribution(),
    }
    values[field] = ""
    hook = TrainingMethodAuthoringHook(**values)
    descriptor = TrainingMethodDescriptor(
        method_ref=METHOD_REF,
        payload_schema_id=PAYLOAD_SCHEMA_ID,
        payload_schema_version=PAYLOAD_SCHEMA_VERSION,
        payload_model=TypedPayload,
        contract_compiler=lambda _payload: _method_contract(),
        update_kernels_factory=standard_supervised_update_kernels,
        authoring_hook=hook,
    )

    with pytest.raises(ValueError, match="identity must not be empty"):
        TrainingMethodRegistry().register_descriptor(descriptor)


def test_authoring_identity_and_canonical_payload_are_deterministic(
    authoring_registry,
) -> None:
    first = compile_training_method_authoring(
        _authored_row(),
        method_ref=METHOD_REF,
        run_control=_run_control(),
        projectors=_projectors(),
    )
    second = compile_training_method_authoring(
        _authored_row(),
        method_ref={"package": "example", "name": "typed", "version": "v1"},
        run_control=_run_control(),
        projectors=_projectors(),
    )

    assert first.lowering_result == second.lowering_result
    assert training_spec_sha256(first.lowering_result.execution_payload) == (
        training_spec_sha256(second.lowering_result.execution_payload)
    )
