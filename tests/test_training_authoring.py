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
    CheckpointProgressPolicySpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    RUN_CONTROL_SPEC_SCHEMA_ID,
    RUN_CONTROL_SPEC_SCHEMA_VERSION,
    RunControlSpec,
    TrainingConfig,
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
    holder: dict[str, Any] = {
        "calls": 0,
        "duplicate_identity": False,
        "mutate_row": False,
        "mutate_execution": None,
    }
    contract = standard_supervised_method_contract().model_copy(
        update={
            "method_ref": METHOD_REF,
            "method_payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        }
    )

    def lower(row: AuthoredTrainingRow) -> TrainingRowLoweringResult:
        holder["calls"] += 1
        assert row.payload_hash == training_spec_sha256(row.payload)
        if holder["mutate_row"]:
            row.payload["domain"]["mutated"] = True
        compact = row.payload
        control = RunControlSpec.model_validate(compact["run_control"])
        spec = TrainingRunSpec(
            graph=compact["graph"],
            task=compact["task"],
            training_config=TrainingConfig(
                n_batches=control.n_batches,
                batch_size=control.batch_size,
            ),
            objective=compact["objective"],
            method_ref=compact["method_ref"],
            method_payload=compact["method_payload"],
            worker_execution=holder["worker"],
            checkpoint_progress=CheckpointProgressPolicySpec(
                checkpoint_interval=control.checkpoint_interval,
                progress_interval=control.progress_interval,
                continuation=control.continuation,
            ),
            metadata=compact["domain"],
        )
        execution = spec.model_dump(mode="json", exclude_none=True)
        mutator = holder["mutate_execution"]
        if mutator is not None:
            mutator(execution)
        identities = [
            RowLowererIdentity(
                lowerer_id="example.typed_method",
                lowerer_version="example.typed_method.v1",
            )
        ]
        if holder["duplicate_identity"]:
            identities.append(identities[0].model_copy())
        return TrainingRowLoweringResult(
            execution_payload=execution,
            lowerer_identities=identities,
        )

    registry.register_descriptor(
        TrainingMethodDescriptor(
            method_ref=METHOD_REF,
            payload_schema_id=PAYLOAD_SCHEMA_ID,
            payload_schema_version=PAYLOAD_SCHEMA_VERSION,
            payload_model=TypedPayload,
            contract_compiler=lambda _payload: contract,
            update_kernels_factory=standard_supervised_update_kernels,
            row_compiler=lower,
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
    assert holder["calls"] == 1
    assert row.model_dump(mode="python") == before
    assert compiled.run_spec.schema_version == "feedbax.spec.training_run.v2"
    assert compiled.worker_execution == holder["worker"]
    assert compiled.run_spec.metadata == {"domain_family": "toy", "authored_gain": 3}
    assert compiled.lowering_result.execution_payload == compiled.run_spec.model_dump(
        mode="json", exclude_none=True
    )
    assert compiled.lowering_result.lowerer_identities[0] == (
        TRAINING_METHOD_AUTHORING_LOWERER_IDENTITY
    )
    assert TrainingRowLoweringResult.model_validate(compiled) == compiled.lowering_result


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


def test_authoring_rejects_missing_schema_low_level_and_missing_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low_level = TrainingMethodRegistry()
    contract = standard_supervised_method_contract().model_copy(
        update={
            "method_ref": METHOD_REF,
            "method_payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        }
    )
    low_level.register(
        TrainingMethodRegistration(
            method_ref=METHOD_REF,
            payload_schema_id=PAYLOAD_SCHEMA_ID,
            payload_schema_version=PAYLOAD_SCHEMA_VERSION,
            payload_model=TypedPayload,
            contract_factory=lambda: contract,
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

    descriptor_only = TrainingMethodRegistry()
    descriptor_only.register_descriptor(
        TrainingMethodDescriptor(
            method_ref=METHOD_REF,
            payload_schema_id=PAYLOAD_SCHEMA_ID,
            payload_schema_version=PAYLOAD_SCHEMA_VERSION,
            payload_model=TypedPayload,
            contract_compiler=lambda _payload: contract,
            update_kernels_factory=standard_supervised_update_kernels,
        )
    )
    monkeypatch.setattr(
        training_contracts,
        "DEFAULT_TRAINING_METHOD_REGISTRY",
        descriptor_only,
    )
    with pytest.raises(TrainingMethodAuthoringError, match="has no row_compiler"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
        )

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


def test_authoring_rejects_row_mutation_noncanonical_output_and_projection_drift(
    authoring_registry,
) -> None:
    _registry, holder = authoring_registry
    holder["mutate_row"] = True
    with pytest.raises(TrainingMethodAuthoringError, match="mutated"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
        )

    holder["mutate_row"] = False
    holder["mutate_execution"] = lambda payload: payload["artifacts"].update(
        {"manifest_root": None}
    )
    with pytest.raises(TrainingMethodAuthoringError, match="not the canonical"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
        )

    def drift(payload: dict[str, Any]) -> None:
        payload["task"]["type"] = "drifted"

    holder["mutate_execution"] = drift
    with pytest.raises(TrainingMethodAuthoringError, match="/task"):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
        )


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


def test_authoring_rejects_duplicate_descriptor_lowerer_identities(
    authoring_registry,
) -> None:
    _registry, holder = authoring_registry
    holder["duplicate_identity"] = True

    with pytest.raises(
        TrainingMethodAuthoringError,
        match=r"/row_compiler/lowerer_identities/1 duplicates identity at index 0",
    ):
        compile_training_method_authoring(
            _authored_row(),
            method_ref=METHOD_REF,
            run_control=_run_control(),
            projectors=_projectors(),
        )
