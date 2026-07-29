from __future__ import annotations

from dataclasses import replace
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

import feedbax.contracts.training as training_contracts
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.run_matrix import (
    AuthoredTrainingRow,
    RowLowererIdentity,
    TRAINING_ROW_LOWERER_REF_FIELD,
    TrainingRowLowererRef,
    TrainingRowLoweringResult,
    TrainingRunMatrixSpec,
)
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.training import (
    ArtifactPolicySpec,
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
from feedbax.contracts.worker import AxisSpec, MappingLevelSpec, SlotAxisBindingSpec
from feedbax.training.authoring import (
    TRAINING_METHOD_AUTHORING_LOWERER_IDENTITY,
    TrainingMethodAuthoringError,
    compile_training_method_authoring,
    training_method_authoring_implementation_sha256,
)
from feedbax.plugins.discovery import load_training_method_plugins
from feedbax.training.preparation import ExecutionPreparationProviderRegistry
from feedbax.training.row_lowering import (
    TrainingRowLowererRegistry,
    training_row_lowerer_implementation_sha256,
)
from feedbax.training.run_matrix import materialize_adapted_run_matrix
from feedbax.training.spec_storage import TrainingRunMatrixCompiler
from feedbax.training.worker_validation import (
    WorkerContractValidationError,
    resolve_execution_mapping,
)


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


def _mapped_method_contract():
    contract = _method_contract()
    contract.axes.append(AxisSpec(name="ensemble", role="replicate", size=5))
    for slot in contract.state_slots:
        slot.axis_bindings = [SlotAxisBindingSpec(axis="ensemble", mode="mapped", array_axis=0)]
    for step in contract.phase_program.update_steps:
        step.axes.append("ensemble")
    return contract


def _contribution(
    *, mapping_levels: list[MappingLevelSpec] | None = None
) -> TrainingMethodAuthoringContribution:
    return TrainingMethodAuthoringContribution(
        training_config=TrainingConfig(
            n_batches=4,
            batch_size=2,
            learning_rate=0.025,
            hidden_dim=32,
        ),
        checkpoint_interval=2,
        progress_interval=1,
        method_extensions=MethodExtensionsSpec(metadata={"method_family": "typed-toy"}),
        mapping_levels=mapping_levels,
    )


def _projectors(calls: dict[str, int] | None = None) -> dict[str, object]:
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

    return {"graph": graph, "task": task, "objective": objective, "domain": domain}


@pytest.fixture
def authoring_registry(monkeypatch: pytest.MonkeyPatch):
    registry = TrainingMethodRegistry()
    default_result = object()
    holder: dict[str, Any] = {
        "authoring_calls": 0,
        "row_compiler_calls": 0,
        "mutate_payload": False,
        "return_value": default_result,
        "contract": _method_contract(),
        "projector_calls": {},
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
            contract_compiler=lambda _payload: holder["contract"].model_copy(deep=True),
            update_kernels_factory=standard_supervised_update_kernels,
            row_compiler=low_level_lower,
            authoring_hook=TrainingMethodAuthoringHook(
                lowerer_id="example.typed_method.authoring",
                lowerer_version="example.typed_method.authoring.v1",
                compile=author,
                **_projectors(holder["projector_calls"]),
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
    row = _authored_row()
    before = row.model_dump(mode="python")

    compiled = compile_training_method_authoring(
        row,
        method_ref=METHOD_REF,
    )

    assert holder["projector_calls"] == {
        "graph": 1,
        "task": 1,
        "objective": 1,
        "domain": 1,
    }
    assert holder["authoring_calls"] == 1
    assert holder["row_compiler_calls"] == 0
    assert row.model_dump(mode="python") == before
    assert compiled.run_spec.schema_version == "feedbax.spec.training_run.v4"
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
        "checkpoint_interval",
        "progress_interval",
        "method_extensions",
        "mapping_levels",
    }
    assert compiled.worker_execution.mapping_levels is None
    assert "mapping_levels" not in compiled.lowering_result.execution_payload["worker_execution"]
    assert compiled.lowering_result.execution_payload == compiled.run_spec.model_dump(
        mode="json", exclude_none=True
    )
    assert compiled.lowering_result.lowerer_identities == [
        TRAINING_METHOD_AUTHORING_LOWERER_IDENTITY,
        holder["hook_identity"],
    ]
    assert TrainingRowLoweringResult.model_validate(compiled) == compiled.lowering_result


def test_compile_authoring_carries_one_mapping_level_without_row_compiler(
    authoring_registry,
) -> None:
    _registry, holder = authoring_registry
    holder["contract"] = _mapped_method_contract()
    mapping_levels = [MappingLevelSpec(axis="ensemble")]
    holder["return_value"] = _contribution(mapping_levels=mapping_levels)

    compiled = compile_training_method_authoring(
        _authored_row(),
        method_ref=METHOD_REF,
    )

    assert compiled.worker_execution.mapping_levels == mapping_levels
    assert compiled.lowering_result.execution_payload["worker_execution"]["mapping_levels"] == [
        {"axis": "ensemble"}
    ]
    levels, _bindings = resolve_execution_mapping(compiled.worker_execution)
    assert [(level.axis, level.role, level.size) for level in levels] == [
        ("ensemble", "replicate", 5)
    ]
    assert holder["row_compiler_calls"] == 0


def test_compile_authoring_preserves_empty_mapping_as_scalar(authoring_registry) -> None:
    _registry, holder = authoring_registry
    holder["return_value"] = _contribution(mapping_levels=[])

    compiled = compile_training_method_authoring(
        _authored_row(),
        method_ref=METHOD_REF,
    )

    assert compiled.worker_execution.mapping_levels == []
    assert resolve_execution_mapping(compiled.worker_execution) == ((), {})
    assert holder["row_compiler_calls"] == 0


def test_compile_authoring_leaves_mapping_validation_to_worker_contract(
    authoring_registry,
) -> None:
    _registry, holder = authoring_registry
    holder["contract"] = _mapped_method_contract()
    holder["return_value"] = _contribution(
        mapping_levels=[
            MappingLevelSpec(axis="ensemble"),
            MappingLevelSpec(axis="ensemble"),
        ]
    )

    compiled = compile_training_method_authoring(
        _authored_row(),
        method_ref=METHOD_REF,
    )

    with pytest.raises(WorkerContractValidationError, match="exactly one mapping level"):
        resolve_execution_mapping(compiled.worker_execution)
    assert holder["row_compiler_calls"] == 0


def test_compile_authoring_owns_scientific_policies_only(authoring_registry) -> None:
    defaulted = compile_training_method_authoring(
        _authored_row(),
        method_ref=METHOD_REF,
    )

    assert "execution" not in TrainingRunSpec.model_fields
    assert defaulted.run_spec.artifacts == ArtifactPolicySpec()
    assert defaulted.run_spec.risk_aggregation == RiskAggregationSpec()
    assert defaulted.run_spec.checkpoint_progress.checkpoint_interval == 2
    assert defaulted.run_spec.checkpoint_progress.progress_interval == 1

    artifacts = ArtifactPolicySpec(custody="mandible", artifact_root="artifacts")
    risk = RiskAggregationSpec(realization="mean", replicate="max")
    explicit = compile_training_method_authoring(
        _authored_row(),
        method_ref=METHOD_REF,
        artifacts=artifacts,
        risk_aggregation=risk,
    )

    assert explicit.run_spec.artifacts == artifacts
    assert explicit.run_spec.risk_aggregation == risk
    assert "execution" not in explicit.lowering_result.execution_payload


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
        )
    assert row_compiler_calls == 0


def test_authoring_rejects_projector_and_reserved_domain_errors(
    authoring_registry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _holder = authoring_registry
    descriptor = registry.descriptor(METHOD_REF)
    assert descriptor is not None and descriptor.authoring_hook is not None

    def install(**updates: object) -> None:
        candidate = TrainingMethodRegistry()
        candidate.register_descriptor(
            replace(
                descriptor,
                authoring_hook=replace(descriptor.authoring_hook, **updates),
            )
        )
        monkeypatch.setattr(training_contracts, "DEFAULT_TRAINING_METHOD_REGISTRY", candidate)

    install(domain=lambda _payload: {"method_ref": "collision"})
    with pytest.raises(TrainingMethodAuthoringError, match="reserved metadata"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
        )

    install(graph=lambda _payload: {"inline": {"nodes": "invalid"}})
    with pytest.raises(TrainingMethodAuthoringError, match="graph projector returned invalid"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
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
    _registry, holder = authoring_registry
    with pytest.raises(TrainingMethodAuthoringError, match="method payload schema"):
        compile_training_method_authoring(
            _authored_row({"gain": "3", "task_name": "reach"}),
            method_ref=METHOD_REF,
        )

    assert holder["projector_calls"] == {}


def test_authoring_rejects_continuation_batch_drift(authoring_registry) -> None:
    with pytest.raises(TrainingMethodAuthoringError, match="additional_batches"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            continuation={"source_completed_batches": 4, "additional_batches": 3},
        )


def test_authoring_accepts_matching_continuation(authoring_registry) -> None:
    compiled = compile_training_method_authoring(
        _authored_row(),
        method_ref=METHOD_REF,
        continuation={"source_completed_batches": 6, "additional_batches": 4},
    )

    assert compiled.run_spec.checkpoint_progress.continuation is not None
    assert (
        compiled.run_spec.checkpoint_progress.continuation.source_completed_batches
        == 6
    )
    assert compiled.run_spec.checkpoint_progress.continuation.additional_batches == 4


def test_authoring_rejects_noncanonical_authored_payload_hash(authoring_registry) -> None:
    row = _authored_row().model_copy(update={"payload_hash": "0" * 64})

    with pytest.raises(TrainingMethodAuthoringError, match="payload_hash"):
        compile_training_method_authoring(
            row,
            method_ref=METHOD_REF,
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
                **_projectors(),
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
        )
    assert calls == 0


@pytest.mark.parametrize("field", ["lowerer_id", "lowerer_version"])
def test_authoring_hook_rejects_empty_identity(field: str) -> None:
    values = {
        "lowerer_id": "example.typed_method.authoring",
        "lowerer_version": "example.typed_method.authoring.v1",
        "compile": lambda _payload: _contribution(),
        **_projectors(),
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
    )
    second = compile_training_method_authoring(
        _authored_row(),
        method_ref={"package": "example", "name": "typed", "version": "v1"},
    )

    assert first.lowering_result == second.lowering_result
    assert training_spec_sha256(first.lowering_result.execution_payload) == (
        training_spec_sha256(second.lowering_result.execution_payload)
    )


def _descriptor_plugin(descriptor: TrainingMethodDescriptor[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        register_feedbax_training_methods=lambda registry: registry.register_descriptor(descriptor)
    )


def _derived_authority(
    descriptor: TrainingMethodDescriptor[Any],
) -> dict[str, Any]:
    return TrainingRowLowererRef(
        lowerer_id="example.typed_method.authoring",
        lowerer_version="example.typed_method.authoring.v1",
        implementation_sha256=training_method_authoring_implementation_sha256(descriptor),
    ).model_dump(mode="json")


def test_plugin_derives_authoring_lowerer_and_replay_is_idempotent(
    authoring_registry,
) -> None:
    source_registry, _holder = authoring_registry
    descriptor = source_registry.descriptor(METHOD_REF)
    assert descriptor is not None
    lowerers = TrainingRowLowererRegistry()
    plugin = _descriptor_plugin(descriptor)
    entry_point = SimpleNamespace(name="typed", load=lambda: plugin)

    for _ in range(2):
        load_training_method_plugins(
            registry=TrainingMethodRegistry(),
            preparation_registry=ExecutionPreparationProviderRegistry(),
            row_lowerer_registry=lowerers,
            entry_points=[entry_point],
            fail_on_load_error=True,
        )

    assert lowerers.available_keys() == (
        (
            PAYLOAD_SCHEMA_ID,
            PAYLOAD_SCHEMA_VERSION,
            "example.typed_method.authoring",
            "example.typed_method.authoring.v1",
        ),
    )


def test_plugin_skips_descriptor_without_authoring_hook(authoring_registry) -> None:
    source_registry, _holder = authoring_registry
    descriptor = source_registry.descriptor(METHOD_REF)
    assert descriptor is not None
    lowerers = TrainingRowLowererRegistry()

    load_training_method_plugins(
        registry=TrainingMethodRegistry(),
        preparation_registry=ExecutionPreparationProviderRegistry(),
        row_lowerer_registry=lowerers,
        entry_points=[
            SimpleNamespace(
                name="typed",
                load=lambda: _descriptor_plugin(replace(descriptor, authoring_hook=None)),
            )
        ],
        fail_on_load_error=True,
    )

    assert lowerers.available_keys() == ()


def test_plugin_derivation_preserves_explicit_conflicts(authoring_registry) -> None:
    source_registry, _holder = authoring_registry
    descriptor = source_registry.descriptor(METHOD_REF)
    assert descriptor is not None
    lowerers = TrainingRowLowererRegistry()
    plugin = _descriptor_plugin(descriptor)
    entry_point = SimpleNamespace(name="typed", load=lambda: plugin)
    load_training_method_plugins(
        registry=TrainingMethodRegistry(),
        preparation_registry=ExecutionPreparationProviderRegistry(),
        row_lowerer_registry=lowerers,
        entry_points=[entry_point],
        fail_on_load_error=True,
    )

    with pytest.raises(RuntimeError, match="ambiguous.*owners"):
        load_training_method_plugins(
            registry=TrainingMethodRegistry(),
            preparation_registry=ExecutionPreparationProviderRegistry(),
            row_lowerer_registry=lowerers,
            entry_points=[
                SimpleNamespace(
                    name="other",
                    load=lambda: _descriptor_plugin(replace(descriptor, owner="other")),
                )
            ],
            fail_on_load_error=True,
        )


def test_descriptor_authoring_digest_binds_projector_implementation(
    authoring_registry,
) -> None:
    source_registry, _holder = authoring_registry
    descriptor = source_registry.descriptor(METHOD_REF)
    assert descriptor is not None and descriptor.authoring_hook is not None
    drifted = replace(
        descriptor,
        authoring_hook=replace(descriptor.authoring_hook, graph=Path.exists),
    )

    assert training_method_authoring_implementation_sha256(descriptor) != (
        training_method_authoring_implementation_sha256(drifted)
    )


def test_composite_digest_binds_same_module_callable_identity_and_order() -> None:
    original = training_row_lowerer_implementation_sha256(
        (_minimal_graph, _method_contract)
    )

    assert original != training_row_lowerer_implementation_sha256(
        (_minimal_graph, _mapped_method_contract)
    )
    assert original != training_row_lowerer_implementation_sha256(
        (_method_contract, _minimal_graph)
    )


def test_distinct_methods_cannot_alias_one_derived_lowerer_authority(
    authoring_registry,
) -> None:
    source_registry, _holder = authoring_registry
    descriptor = source_registry.descriptor(METHOD_REF)
    assert descriptor is not None
    other = replace(descriptor, method_ref="example/other/v1")

    with pytest.raises(RuntimeError, match="ambiguous.*implementation sha256"):
        load_training_method_plugins(
            registry=TrainingMethodRegistry(),
            preparation_registry=ExecutionPreparationProviderRegistry(),
            row_lowerer_registry=TrainingRowLowererRegistry(),
            entry_points=[
                SimpleNamespace(
                    name="colliding-methods",
                    load=lambda: SimpleNamespace(
                        register_feedbax_training_methods=lambda registry: (
                            registry.register_descriptor(descriptor),
                            registry.register_descriptor(other),
                        )
                    ),
                )
            ],
            fail_on_load_error=True,
        )


def test_default_matrix_compiler_reaches_descriptor_authoring(
    authoring_registry,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source_registry, _holder = authoring_registry
    descriptor = source_registry.descriptor(METHOD_REF)
    assert descriptor is not None
    methods = TrainingMethodRegistry()
    lowerers = TrainingRowLowererRegistry()
    load_training_method_plugins(
        registry=methods,
        preparation_registry=ExecutionPreparationProviderRegistry(),
        row_lowerer_registry=lowerers,
        entry_points=[
            SimpleNamespace(
                name="typed",
                load=lambda: _descriptor_plugin(descriptor),
            )
        ],
        fail_on_load_error=True,
    )
    monkeypatch.setattr(training_contracts, "DEFAULT_TRAINING_METHOD_REGISTRY", methods)
    monkeypatch.setattr(
        "feedbax.training.row_lowering.DEFAULT_TRAINING_ROW_LOWERER_REGISTRY",
        lowerers,
    )
    payload = {
        "schema_id": PAYLOAD_SCHEMA_ID,
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "gain": 3,
        "task_name": "reach",
        TRAINING_ROW_LOWERER_REF_FIELD: _derived_authority(descriptor),
    }
    matrix = TrainingRunMatrixSpec.model_validate(
        {
            "name": "descriptor-authored",
            "base": {"kind": "inline", "inline": payload},
            "rows": [{"row_id": "typed", "overrides": []}],
        }
    )

    compiled = TrainingRunMatrixCompiler(allow_inline_base=True).compile(
        authored=matrix,
        run_set_id="descriptor-authored",
        context=SimpleNamespace(
            repo_root=tmp_path,
            resolved_inputs=(),
            training_row_lowering_context=None,
        ),
    )

    assert compiled.rows[0].payload["method_ref"] == {
        "package": "example",
        "name": "typed",
        "version": "v1",
    }
    execution_method_payload = compiled.rows[0].payload["method_payload"]["payload"]
    assert execution_method_payload == {"gain": 3, "task_name": "reach"}
    assert {
        "schema_id",
        "schema_version",
        TRAINING_ROW_LOWERER_REF_FIELD,
    }.isdisjoint(execution_method_payload)
    assert compiled.rows[0].provenance.authored_payload_hash == training_spec_sha256(
        payload
    )
    assert (
        compiled.rows[0].provenance.lowered_execution_payload_hash
        == training_spec_sha256(compiled.rows[0].payload)
    )
    assert compiled.rows[0].provenance.lowerer_identities == [
        RowLowererIdentity(
            lowerer_id="example.typed_method.authoring",
            lowerer_version="example.typed_method.authoring.v1",
        )
    ]


@pytest.mark.parametrize(
    ("field", "observed"),
    [
        ("schema_id", "example.spec.other"),
        ("schema_version", "example.spec.typed_method.v2"),
    ],
)
def test_descriptor_authoring_rejects_dispatch_schema_mismatch(
    authoring_registry,
    field: str,
    observed: str,
) -> None:
    registry, _holder = authoring_registry
    descriptor = registry.descriptor(METHOD_REF)
    assert descriptor is not None
    payload = {
        "schema_id": PAYLOAD_SCHEMA_ID,
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "gain": 3,
        "task_name": "reach",
        TRAINING_ROW_LOWERER_REF_FIELD: _derived_authority(descriptor),
    }
    payload[field] = observed

    with pytest.raises(
        TrainingMethodAuthoringError,
        match=rf"{field} does not match bound descriptor authority",
    ):
        compile_training_method_authoring(
            _authored_row(payload),
            method_ref=METHOD_REF,
        )


def test_fresh_process_plugin_derives_same_authoring_lowerer(tmp_path: Path) -> None:
    plugin_path = tmp_path / "descriptor_authoring_plugin.py"
    plugin_path.write_text(
        """
from pydantic import BaseModel, ConfigDict

from feedbax.contracts.training import (
    TrainingConfig,
    TrainingMethodAuthoringContribution,
    TrainingMethodAuthoringHook,
    TrainingMethodDescriptor,
    standard_supervised_method_contract,
    standard_supervised_update_kernels,
)


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_id: str = "tests.spec.fresh_method"
    schema_version: str = "tests.spec.fresh_method.v1"
    value: int


def register_feedbax_training_methods(registry):
    registry.register_descriptor(TrainingMethodDescriptor(
        method_ref="tests/fresh/v1",
        payload_schema_id="tests.spec.fresh_method",
        payload_schema_version="tests.spec.fresh_method.v1",
        payload_model=Payload,
        contract_compiler=lambda _payload: standard_supervised_method_contract(),
        update_kernels_factory=standard_supervised_update_kernels,
        authoring_hook=TrainingMethodAuthoringHook(
            lowerer_id="tests.fresh.authoring",
            lowerer_version="tests.fresh.authoring.v1",
            compile=lambda _payload: TrainingMethodAuthoringContribution(
                training_config=TrainingConfig(n_batches=1, batch_size=1)
            ),
            graph=lambda _payload: {},
            task=lambda _payload: {},
            objective=lambda _payload: {},
            domain=lambda _payload: {},
        ),
        owner="tests.fresh",
        package="tests",
    ))
""".lstrip(),
        encoding="utf-8",
    )
    dist_info = tmp_path / "descriptor_authoring_plugin-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "entry_points.txt").write_text(
        "[tests.feedbax.plugins]\ndescriptor-authoring = descriptor_authoring_plugin\n",
        encoding="utf-8",
    )
    script = """
import json
from feedbax.contracts.training import DEFAULT_TRAINING_METHOD_REGISTRY
from feedbax.plugins.discovery import load_training_method_plugins
from feedbax.training.row_lowering import DEFAULT_TRAINING_ROW_LOWERER_REGISTRY

load_training_method_plugins(
    entry_point_group="tests.feedbax.plugins",
    fail_on_load_error=True,
)
print(json.dumps({
    "methods": DEFAULT_TRAINING_METHOD_REGISTRY.descriptor_keys(),
    "lowerers": DEFAULT_TRAINING_ROW_LOWERER_REGISTRY.available_keys(),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path(__file__).parents[1])])
    observed = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        observed.append(json.loads(result.stdout))

    assert (
        observed[0]
        == observed[1]
        == {
            "methods": ["feedbax/standard_supervised/v1", "tests/fresh/v1"],
            "lowerers": [
                [
                    "tests.spec.fresh_method",
                    "tests.spec.fresh_method.v1",
                    "tests.fresh.authoring",
                    "tests.fresh.authoring.v1",
                ]
            ],
        }
    )
