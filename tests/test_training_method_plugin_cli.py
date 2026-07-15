from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

import feedbax.__main__ as cli_module
from feedbax.contracts.training import (
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    LossTermSpec,
    ObjectiveSlotSpec,
    StandardSupervisedMethodPayload,
    TaskSpec,
    TrainingConfig,
    TrainingMethodDescriptor,
    TrainingMethodMetadataProjector,
    TrainingMethodRegistration,
    TrainingMethodRegistry,
    TrainingRunSpec,
    WorkerExecutionSpec,
    default_training_method_registry,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
    standard_supervised_update_kernels,
)
from feedbax.contracts.worker import AxisCoordinateSpec
from feedbax.plugins.discovery import load_training_method_plugins
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.training.preparation import (
    ExecutionPreparationError,
    ExecutionPreparationPlan,
    ExecutionPreparationProviderRegistry,
    ExecutionPreparationRegistration,
    ExecutionPreparationRequest,
    ExecutionPreparationResult,
    PREPARATION_RNG_ALGORITHM_VERSION,
    ScalarInstancePreparationResult,
    derive_preparation_rng_scope,
    lower_zero_level_preparation_plan,
    preparation_rng_token,
    require_execution_preparation_provider,
)
from feedbax.training.worker_validation import WorkerContractValidationError


DUMMY_METHOD_REF = "dummy/custom/v1"
DUMMY_SCHEMA_ID = "dummy.spec.training_method"
DUMMY_SCHEMA_VERSION = "dummy.spec.training_method.v1"


class DummyPayload(BaseModel):
    token: str = "typed"


class DummyMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    token_length: int


class _NumpyModule(eqx.Module):
    value: np.ndarray


@jax.tree_util.register_pytree_node_class
class _NumpyCustomTree:
    def __init__(self, value):
        self.value = value

    def tree_flatten(self):
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(children[0])


def _dummy_descriptor(**hooks: object) -> TrainingMethodDescriptor[DummyPayload]:
    return TrainingMethodDescriptor(
        method_ref=DUMMY_METHOD_REF,
        payload_schema_id=DUMMY_SCHEMA_ID,
        payload_schema_version=DUMMY_SCHEMA_VERSION,
        payload_model=DummyPayload,
        contract_compiler=lambda _payload: standard_supervised_method_contract().model_copy(
            update={
                "method_ref": DUMMY_METHOD_REF,
                "method_payload_schema_version": DUMMY_SCHEMA_VERSION,
            }
        ),
        update_kernels_factory=standard_supervised_update_kernels,
        **hooks,
    )


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


def _standard_run_spec_payload() -> dict[str, object]:
    spec = TrainingRunSpec(
        graph={"inline": _minimal_graph()},
        task=TaskSpec(type="ToyTask", params={"n_steps": 1}),
        training_config=TrainingConfig(n_batches=1, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(
                type="target_state",
                label="target",
                selector="port:gain.output",
                target_value=[0.0],
            )
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
    )
    return spec.model_dump(mode="json", exclude_none=True)


def _dummy_run_spec_payload() -> dict[str, object]:
    payload = _standard_run_spec_payload()
    payload["method_ref"] = {"package": "dummy", "name": "custom", "version": "v1"}
    payload["method_payload"] = {
        "schema_id": DUMMY_SCHEMA_ID,
        "schema_version": DUMMY_SCHEMA_VERSION,
        "payload": {"token": "typed"},
    }
    worker_execution = payload["worker_execution"]
    assert isinstance(worker_execution, dict)
    method_contract = worker_execution["method_contract"]
    effective_phase = worker_execution["effective_phase"]
    assert isinstance(method_contract, dict)
    assert isinstance(effective_phase, dict)
    method_contract["method_ref"] = DUMMY_METHOD_REF
    method_contract["method_payload_schema_version"] = DUMMY_SCHEMA_VERSION
    effective_phase["method_ref"] = DUMMY_METHOD_REF
    return payload


def _register_dummy_training_method(registry) -> None:
    contract = standard_supervised_method_contract().model_copy(
        update={
            "method_ref": DUMMY_METHOD_REF,
            "method_payload_schema_version": DUMMY_SCHEMA_VERSION,
        }
    )
    registry.register(
        TrainingMethodRegistration(
            method_ref=DUMMY_METHOD_REF,
            payload_schema_id=DUMMY_SCHEMA_ID,
            payload_schema_version=DUMMY_SCHEMA_VERSION,
            payload_model=DummyPayload,
            contract_factory=lambda: contract,
            update_kernels_factory=standard_supervised_update_kernels,
            owner="tests.test_training_method_plugin_cli",
            package="dummy",
        )
    )


def test_entry_point_can_register_training_method() -> None:
    registry = default_training_method_registry()
    plugin = SimpleNamespace(register_feedbax_training_methods=_register_dummy_training_method)
    entry_point = SimpleNamespace(name="dummy-training-method", load=lambda: plugin)

    load_training_method_plugins(registry=registry, entry_points=[entry_point])

    assert DUMMY_METHOD_REF in registry.available_keys()


def test_preparation_rng_algorithm_has_frozen_tokens_and_key_vectors() -> None:
    assert preparation_rng_token("algorithm", PREPARATION_RNG_ALGORITHM_VERSION) == 3959945493
    assert preparation_rng_token("root", "model") == 2366028346
    assert preparation_rng_token("axis", "ensemble") == 3470389890

    cases = (
        (0, "model", 0, (4211853719, 2725690202)),
        (0, "model", 4, (1509863984, 1304489815)),
        (17, "runtime", 2, (3131609803, 682786736)),
    )
    for seed, root, index, expected in cases:
        scope = derive_preparation_rng_scope(
            {root: jax.random.key(seed)},
            (AxisCoordinateSpec(axis="ensemble", index=index),),
        )
        assert tuple(map(int, jax.random.key_data(scope.keys[root]))) == expected


def test_preparation_plan_defensively_freezes_containers_and_numpy_storage() -> None:
    source = np.arange(3, dtype=np.float32)
    plan = ExecutionPreparationPlan(
        shared_slots={"array": source, "nested": [{1: "integer-key"}]},
        kernel_context={},
        loss_service=None,
        resume_slot_transform=None,
        rng_roots={"model": jax.random.key(0)},
        materialize_instance=lambda _request: ScalarInstancePreparationResult(mapped_slots={}),
    )

    source[0] = 99
    frozen = plan.shared_slots["array"]
    assert frozen.tolist() == [0.0, 1.0, 2.0]
    assert not frozen.flags.writeable
    assert isinstance(plan.shared_slots["nested"], tuple)
    assert plan.shared_slots["nested"][0] == {1: "integer-key"}
    with pytest.raises(TypeError, match="immutable Feedbax preparation mapping"):
        plan.shared_slots["new"] = object()
    with pytest.raises(ValueError, match="read-only"):
        frozen[0] = 5
    with pytest.raises(ExecutionPreparationError, match="object-backed"):
        ExecutionPreparationPlan(
            shared_slots={"bad": np.array([object()], dtype=object)},
            kernel_context={},
            loss_service=None,
            resume_slot_transform=None,
            rng_roots={"model": jax.random.key(0)},
            materialize_instance=lambda _request: ScalarInstancePreparationResult(
                mapped_slots={}
            ),
        )


def test_preparation_freezes_numpy_leaves_in_custom_and_equinox_pytrees() -> None:
    custom_source = np.arange(2, dtype=np.float32)
    module_source = np.arange(3, dtype=np.float32)
    plan = ExecutionPreparationPlan(
        shared_slots={
            "custom": _NumpyCustomTree(custom_source),
            "module": _NumpyModule(module_source),
        },
        kernel_context={},
        loss_service=None,
        resume_slot_transform=None,
        rng_roots={"model": jax.random.key(0)},
        materialize_instance=lambda _request: ScalarInstancePreparationResult(mapped_slots={}),
    )
    custom_source[:] = -1
    module_source[:] = -1

    for frozen, expected in (
        (plan.shared_slots["custom"].value, [0.0, 1.0]),
        (plan.shared_slots["module"].value, [0.0, 1.0, 2.0]),
    ):
        assert frozen.tolist() == expected
        assert not frozen.flags.writeable
    with pytest.raises(ExecutionPreparationError, match="must be a string identifier"):
        ExecutionPreparationPlan(
            shared_slots={1: "invalid-root-name"},
            kernel_context={},
            loss_service=None,
            resume_slot_transform=None,
            rng_roots={"model": jax.random.key(0)},
            materialize_instance=lambda _request: ScalarInstancePreparationResult(
                mapped_slots={}
            ),
        )


def test_zero_level_plan_materializes_once_through_scalar_compatibility() -> None:
    requests = []

    def materialize(request):
        requests.append(request)
        return ScalarInstancePreparationResult(mapped_slots={"model": np.array([2.0])})

    plan = ExecutionPreparationPlan(
        shared_slots={"objective": "shared"},
        kernel_context={"token": "context"},
        loss_service=None,
        resume_slot_transform=None,
        rng_roots={"model": jax.random.key(0)},
        materialize_instance=materialize,
    )

    default_result = lower_zero_level_preparation_plan(plan)
    assert requests[-1].resume_template is False
    result = lower_zero_level_preparation_plan(plan, resume_template=True)

    assert len(requests) == 2
    assert requests[-1].axis_coordinates == ()
    assert requests[-1].rng.axis_coordinates == ()
    assert requests[-1].resume_template is True
    assert default_result.initial_slots.keys() == result.initial_slots.keys()
    assert set(result.initial_slots) == {"model", "objective"}
    assert result.kernel_context == {"token": "context"}


def test_execute_cli_routes_resume_to_zero_level_plan(monkeypatch, tmp_path, capsys) -> None:
    plan = ExecutionPreparationPlan(
        shared_slots={},
        kernel_context={},
        loss_service=None,
        resume_slot_transform=None,
        rng_roots={"model": jax.random.key(0)},
        materialize_instance=lambda _request: ScalarInstancePreparationResult(mapped_slots={}),
    )
    registration = SimpleNamespace(owner="tests.cli", requires_execution_preparation=True)
    resolved = SimpleNamespace(
        registration=SimpleNamespace(requires_execution_preparation=True),
        payload=None,
        contract=None,
        effective_phase=None,
    )
    run_spec = SimpleNamespace(
        method_ref=SimpleNamespace(key="tests/zero-level/v1"),
        resolved_method=resolved,
        worker_execution=object(),
    )
    registry = SimpleNamespace(
        get=lambda _key: registration,
        prepare=lambda _request: plan,
    )
    routed = []

    monkeypatch.setattr(cli_module, "_load_training_method_plugins", lambda _plugins: None)
    monkeypatch.setattr(cli_module, "_read_json", lambda _path: {})
    monkeypatch.setattr(cli_module, "validate_training_run_spec", lambda _payload: run_spec)
    monkeypatch.setattr(cli_module, "resolve_execution_mapping", lambda _worker: ((), {}))
    monkeypatch.setattr(cli_module, "require_execution_preparation_provider", lambda **_kw: None)
    monkeypatch.setattr(cli_module, "DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY", registry)
    monkeypatch.setattr(
        cli_module,
        "lower_zero_level_preparation_plan",
        lambda _plan, *, resume_template=False: (
            routed.append(resume_template) or ExecutionPreparationResult(initial_slots={})
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "execute_training_run_spec",
        lambda *_args, **_kwargs: SimpleNamespace(
            run_id="run", status="completed", manifest_path=tmp_path / "manifest.json",
            manifest=SimpleNamespace(model_dump=lambda **_kwargs: {}),
        ),
    )
    monkeypatch.setattr(cli_module.RunEventEmitter, "from_env", lambda **_kwargs: None)

    assert cli_module.main(["execute-training-run-spec", "spec.json", "--resume"]) == 0
    assert routed == [True]
    capsys.readouterr()


def test_low_level_registration_requires_exactly_one_contract_producer() -> None:
    contract = standard_supervised_method_contract()
    common = {
        "method_ref": DUMMY_METHOD_REF,
        "payload_schema_id": DUMMY_SCHEMA_ID,
        "payload_schema_version": DUMMY_SCHEMA_VERSION,
        "payload_model": DummyPayload,
        "update_kernels_factory": standard_supervised_update_kernels,
    }

    with pytest.raises(ValueError, match="exactly one contract producer"):
        TrainingMethodRegistry().register(
            TrainingMethodRegistration(contract_factory=None, **common)
        )
    with pytest.raises(ValueError, match="exactly one contract producer"):
        TrainingMethodRegistry().register(
            TrainingMethodRegistration(
                contract_factory=lambda: contract,
                contract_compiler=lambda _payload: contract,
                **common,
            )
        )


def test_entry_point_descriptor_derives_method_and_preparation_from_one_hook() -> None:
    method_registry = default_training_method_registry()
    preparation_registry = ExecutionPreparationProviderRegistry()
    contract = standard_supervised_method_contract().model_copy(
        update={
            "method_ref": DUMMY_METHOD_REF,
            "method_payload_schema_version": DUMMY_SCHEMA_VERSION,
        }
    )

    def register(registry) -> None:
        registry.register_descriptor(
            TrainingMethodDescriptor(
                method_ref=DUMMY_METHOD_REF,
                payload_schema_id=DUMMY_SCHEMA_ID,
                payload_schema_version=DUMMY_SCHEMA_VERSION,
                payload_model=DummyPayload,
                contract_compiler=lambda payload: contract,
                update_kernels_factory=standard_supervised_update_kernels,
                preparation_provider=lambda request: ExecutionPreparationResult(initial_slots={}),
                owner="dummy-descriptor",
                package="dummy",
            )
        )

    load_training_method_plugins(
        registry=method_registry,
        preparation_registry=preparation_registry,
        entry_points=[
            SimpleNamespace(
                name="descriptor",
                load=lambda: SimpleNamespace(register_feedbax_training_methods=register),
            )
        ],
    )

    assert method_registry.descriptor_keys() == (
        "dummy/custom/v1",
        "feedbax/standard_supervised/v1",
    )
    assert preparation_registry.available_keys() == (DUMMY_METHOD_REF,)


def test_descriptor_binds_existing_row_compiler_boundary() -> None:
    def lower_row(row):
        return row

    descriptor = _dummy_descriptor(row_compiler=lower_row)

    registry = TrainingMethodRegistry()
    registry.register_descriptor(descriptor)

    assert registry.descriptor(DUMMY_METHOD_REF).row_compiler is lower_row


def test_metadata_projector_validates_stable_identity_and_output() -> None:
    projector = TrainingMethodMetadataProjector[DummyPayload](
        schema_id="dummy.spec.training_metadata",
        schema_version="dummy.spec.training_metadata.v1",
        output_model=DummyMetadata,
        projector=lambda payload: {"token_length": len(payload.token)},
    )
    descriptor = _dummy_descriptor(metadata_projector=projector)

    registry = TrainingMethodRegistry()
    registry.register_descriptor(descriptor)

    assert projector.project(DummyPayload(token="typed")) == DummyMetadata(token_length=5)
    assert registry.descriptor(DUMMY_METHOD_REF).metadata_projector is projector


@pytest.mark.parametrize("field", ["schema_id", "schema_version"])
def test_metadata_projector_rejects_empty_identity(field: str) -> None:
    kwargs = {
        "schema_id": "dummy.spec.training_metadata",
        "schema_version": "dummy.spec.training_metadata.v1",
        "output_model": DummyMetadata,
        "projector": lambda payload: {"token_length": len(payload.token)},
    }
    kwargs[field] = " "
    projector = TrainingMethodMetadataProjector[DummyPayload](**kwargs)

    with pytest.raises(ValueError, match="identity must not be empty"):
        TrainingMethodRegistry().register_descriptor(
            _dummy_descriptor(metadata_projector=projector)
        )


def test_metadata_projector_rejects_invalid_model_callable_and_output() -> None:
    with pytest.raises(TypeError, match="output_model must extend BaseModel"):
        TrainingMethodRegistry().register_descriptor(
            _dummy_descriptor(
                metadata_projector=TrainingMethodMetadataProjector(
                    schema_id="dummy.spec.training_metadata",
                    schema_version="dummy.spec.training_metadata.v1",
                    output_model=dict,
                    projector=lambda _payload: {},
                )
            )
        )

    with pytest.raises(TypeError, match="projector must be callable"):
        TrainingMethodRegistry().register_descriptor(
            _dummy_descriptor(
                metadata_projector=TrainingMethodMetadataProjector(
                    schema_id="dummy.spec.training_metadata",
                    schema_version="dummy.spec.training_metadata.v1",
                    output_model=DummyMetadata,
                    projector=None,
                )
            )
        )

    projector = TrainingMethodMetadataProjector[DummyPayload](
        schema_id="dummy.spec.training_metadata",
        schema_version="dummy.spec.training_metadata.v1",
        output_model=DummyMetadata,
        projector=lambda _payload: {"token_length": "not-an-integer"},
    )
    TrainingMethodRegistry().register_descriptor(_dummy_descriptor(metadata_projector=projector))
    with pytest.raises(ValueError, match="token_length"):
        projector.project(DummyPayload())


def test_metadata_projector_requires_strict_output_model() -> None:
    class PermissiveMetadata(BaseModel):
        token_length: int

    with pytest.raises(ValueError, match="extra='forbid'"):
        TrainingMethodRegistry().register_descriptor(
            _dummy_descriptor(
                metadata_projector=TrainingMethodMetadataProjector(
                    schema_id="dummy.spec.training_metadata",
                    schema_version="dummy.spec.training_metadata.v1",
                    output_model=PermissiveMetadata,
                    projector=lambda payload: {"token_length": len(payload.token)},
                )
            )
        )

    class NonStrictMetadata(BaseModel):
        model_config = ConfigDict(extra="forbid")

        token_length: int

    with pytest.raises(ValueError, match="strict=True"):
        TrainingMethodRegistry().register_descriptor(
            _dummy_descriptor(
                metadata_projector=TrainingMethodMetadataProjector(
                    schema_id="dummy.spec.training_metadata",
                    schema_version="dummy.spec.training_metadata.v1",
                    output_model=NonStrictMetadata,
                    projector=lambda payload: {"token_length": len(payload.token)},
                )
            )
        )


def test_descriptor_optimizer_hooks_are_callable_and_standard_is_explicit() -> None:
    registry = default_training_method_registry()
    standard = registry.descriptor("feedbax/standard_supervised/v1")
    assert standard is not None
    assert standard.optimizer_spec_projector is not None
    assert standard.optimizer_step_extractor is not None

    for hook in ("optimizer_spec_projector", "optimizer_step_extractor"):
        with pytest.raises(TypeError, match="non-callable hooks"):
            TrainingMethodRegistry().register_descriptor(
                _dummy_descriptor(**{hook: "not-callable"})
            )


@pytest.mark.parametrize("invalid_mapping", ["kernel", "guard"])
def test_descriptor_rejects_invalid_runtime_mapping_before_preparation(
    tmp_path: Path,
    invalid_mapping: str,
) -> None:
    method_registry = TrainingMethodRegistry()
    preparation_registry = ExecutionPreparationProviderRegistry()
    sentinel = tmp_path / "preparation-ran"

    def prepare(_request: ExecutionPreparationRequest) -> ExecutionPreparationResult:
        sentinel.write_text("unexpected", encoding="utf-8")
        return ExecutionPreparationResult(initial_slots={})

    def register(registry: TrainingMethodRegistry) -> None:
        registry.register_descriptor(
            TrainingMethodDescriptor(
                method_ref="feedbax/standard_supervised/v1",
                payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
                payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
                payload_model=StandardSupervisedMethodPayload,
                contract_compiler=lambda _payload: standard_supervised_method_contract(),
                update_kernels_factory=(
                    (
                        lambda _payload: {
                            "feedbax.training.standard_supervised.gradient_update": lambda slots: {}
                        }
                    )
                    if invalid_mapping == "kernel"
                    else standard_supervised_update_kernels
                ),
                guard_predicates_factory=(
                    (lambda _payload: {"tests.invalid_guard": lambda slots: True})
                    if invalid_mapping == "guard"
                    else (lambda _payload: {})
                ),
                preparation_provider=prepare,
                owner="invalid-descriptor-test",
                package="feedbax",
            )
        )

    load_training_method_plugins(
        registry=method_registry,
        preparation_registry=preparation_registry,
        entry_points=[
            SimpleNamespace(
                name="invalid-descriptor",
                load=lambda: SimpleNamespace(register_feedbax_training_methods=register),
            )
        ],
    )
    run_spec = TrainingRunSpec.model_validate(_standard_run_spec_payload())

    with pytest.raises(WorkerContractValidationError, match="must have signature"):
        resolved = method_registry.resolve_execution(
            run_spec.method_ref,
            run_spec.method_payload,
            worker_execution=run_spec.worker_execution,
        )
        preparation_registry.prepare(
            ExecutionPreparationRequest(
                run_spec=run_spec,
                method_payload=resolved.payload,
                method_contract=resolved.contract,
                effective_phase=resolved.effective_phase,
            )
        )

    assert not sentinel.exists()
    assert not any(tmp_path.iterdir())


def test_entry_point_can_register_execution_preparation() -> None:
    method_registry = default_training_method_registry()
    preparation_registry = ExecutionPreparationProviderRegistry()
    plugin = SimpleNamespace(
        register_feedbax_training_methods=_register_dummy_training_method,
        register_feedbax_execution_preparations=lambda registry: registry.register(
            ExecutionPreparationRegistration(
                method_ref=DUMMY_METHOD_REF,
                provider=lambda _request: ExecutionPreparationResult(initial_slots={"model": 0}),
                owner="dummy-plugin",
            )
        ),
    )

    load_training_method_plugins(
        registry=method_registry,
        preparation_registry=preparation_registry,
        entry_points=[SimpleNamespace(name="dummy", load=lambda: plugin)],
    )

    assert preparation_registry.available_keys() == (DUMMY_METHOD_REF,)


def test_execution_preparation_registry_rejects_duplicate_and_mismatched_providers() -> None:
    preparation_registry = ExecutionPreparationProviderRegistry()
    registration = ExecutionPreparationRegistration(
        method_ref=DUMMY_METHOD_REF,
        provider=lambda _request: ExecutionPreparationResult(initial_slots={}),
        owner="first",
    )
    preparation_registry.register(registration)
    with pytest.raises(ValueError, match="already registered.*first"):
        preparation_registry.register(registration)

    with pytest.raises(RuntimeError, match="do not match registered training methods"):
        load_training_method_plugins(
            registry=default_training_method_registry(),
            preparation_registry=ExecutionPreparationProviderRegistry(),
            entry_points=[
                SimpleNamespace(
                    name="mismatched",
                    load=lambda: SimpleNamespace(
                        register_execution_preparations=lambda registry: registry.register(
                            registration
                        )
                    ),
                )
            ],
        )


def test_execution_preparation_fails_closed_for_missing_mutating_and_failing_provider() -> None:
    run_spec = TrainingRunSpec.model_validate(_standard_run_spec_payload())
    registry = ExecutionPreparationProviderRegistry()
    with pytest.raises(ExecutionPreparationError, match="no execution-preparation provider"):
        registry.prepare(ExecutionPreparationRequest(run_spec=run_spec))
    with pytest.raises(ExecutionPreparationError, match="requires.*but none"):
        require_execution_preparation_provider(
            method_ref=run_spec.method_ref.key,
            preparation_registry=registry,
        )

    def mutate(request: ExecutionPreparationRequest) -> ExecutionPreparationResult:
        request.run_spec.metadata["mutated"] = True
        return ExecutionPreparationResult(initial_slots={})

    registry.register(
        ExecutionPreparationRegistration(
            method_ref=run_spec.method_ref.key,
            provider=mutate,
            owner="mutator",
        )
    )
    with pytest.raises(ExecutionPreparationError, match="mutated TrainingRunSpec"):
        registry.prepare(ExecutionPreparationRequest(run_spec=run_spec))
    assert "mutated" not in run_spec.metadata

    failing_registry = ExecutionPreparationProviderRegistry()

    def fail(_request: ExecutionPreparationRequest) -> ExecutionPreparationResult:
        raise RuntimeError("provider exploded")

    failing_registry.register(
        ExecutionPreparationRegistration(
            method_ref=run_spec.method_ref.key,
            provider=fail,
            owner="failure-test",
        )
    )
    with pytest.raises(ExecutionPreparationError, match="failure-test.*provider exploded"):
        failing_registry.prepare(ExecutionPreparationRequest(run_spec=run_spec))


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (lambda _request: {"initial_slots": {}}, "expected ExecutionPreparationResult"),
        (
            lambda _request: ExecutionPreparationResult(
                initial_slots={},
                kernel_context={"run_spec": object()},
            ),
            "reserved kernel_context keys.*run_spec",
        ),
    ],
)
def test_execution_preparation_rejects_invalid_provider_results(provider, message: str) -> None:
    run_spec = TrainingRunSpec.model_validate(_standard_run_spec_payload())
    registry = ExecutionPreparationProviderRegistry()
    registry.register(
        ExecutionPreparationRegistration(
            method_ref=run_spec.method_ref.key,
            provider=provider,
            owner="invalid-result-test",
        )
    )

    with pytest.raises(ExecutionPreparationError, match=message):
        registry.prepare(ExecutionPreparationRequest(run_spec=run_spec))


def test_unknown_method_ref_guides_plugin_registration() -> None:
    payload = _dummy_run_spec_payload()

    try:
        TrainingRunSpec.model_validate(payload)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("dummy method unexpectedly validated without a plugin")

    assert "unknown method_ref 'dummy/custom/v1'" in message
    assert "feedbax.plugins training-method hook" in message
    assert "--plugin <module>" in message


def test_checkpoint_fork_validates_plugin_registered_training_method(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "dummy-run-spec.json"
    spec_path.write_text(
        json.dumps(_dummy_run_spec_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plugin_path = tmp_path / "dummy_training_plugin.py"
    plugin_path.write_text(
        textwrap.dedent(
            f"""
            from pydantic import BaseModel

            from feedbax.contracts.training import (
                TrainingMethodRegistration,
                standard_supervised_method_contract,
                standard_supervised_update_kernels,
            )


            class DummyPayload(BaseModel):
                pass


            def register_feedbax_training_methods(registry):
                contract = standard_supervised_method_contract().model_copy(
                    update={{
                        "method_ref": {DUMMY_METHOD_REF!r},
                        "method_payload_schema_version": {DUMMY_SCHEMA_VERSION!r},
                    }}
                )
                registry.register(
                    TrainingMethodRegistration(
                        method_ref={DUMMY_METHOD_REF!r},
                        payload_schema_id={DUMMY_SCHEMA_ID!r},
                        payload_schema_version={DUMMY_SCHEMA_VERSION!r},
                        payload_model=DummyPayload,
                        contract_factory=lambda: contract,
                        update_kernels_factory=standard_supervised_update_kernels,
                        owner="dummy_training_plugin",
                        package="dummy",
                    )
                )
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(repo_root), env.get("PYTHONPATH", "")])
    command = [
        sys.executable,
        "-m",
        "feedbax",
        "checkpoint",
        "fork",
        "--plugin",
        "dummy_training_plugin",
        "--source",
        str(tmp_path / "missing-source"),
        "--target",
        f"{spec_path}:{tmp_path / 'target-checkpoint'}",
    ]

    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    output = json.loads(completed.stdout)
    error = output["targets"][0]["error"]
    assert "unknown method_ref" not in error


def test_checkpoint_fork_without_plugin_reports_unknown_method(tmp_path: Path) -> None:
    spec_path = tmp_path / "dummy-run-spec.json"
    spec_path.write_text(
        json.dumps(_dummy_run_spec_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "feedbax",
        "checkpoint",
        "fork",
        "--source",
        str(tmp_path / "missing-source"),
        "--target",
        f"{spec_path}:{tmp_path / 'target-checkpoint'}",
    ]

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    output = json.loads(completed.stdout)
    error = output["targets"][0]["error"]
    assert "unknown method_ref 'dummy/custom/v1'" in error
    assert "--plugin <module>" in error


def test_execute_cli_descriptor_plugin_prepares_typed_runtime_objects(tmp_path: Path) -> None:
    spec_path = tmp_path / "dummy-run-spec.json"
    spec_path.write_text(
        json.dumps(_dummy_run_spec_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plugin_path = tmp_path / "dummy_execution_plugin.py"
    plugin_path.write_text(
        textwrap.dedent(
            f"""
            import jax.numpy as jnp
            from pydantic import BaseModel

            from feedbax.contracts.training import (
                TrainingMethodDescriptor,
                standard_supervised_method_contract,
                standard_supervised_update_kernels,
            )
            from feedbax.training.preparation import (
                ExecutionPreparationResult,
            )


            MARKER = object()


            class DummyPayload(BaseModel):
                token: str


            def compile_contract(payload):
                assert isinstance(payload, DummyPayload)
                assert payload.token == "typed"
                return standard_supervised_method_contract().model_copy(
                    update={{
                        "method_ref": {DUMMY_METHOD_REF!r},
                        "method_payload_schema_version": {DUMMY_SCHEMA_VERSION!r},
                    }}
                )


            def update_kernels(payload):
                assert isinstance(payload, DummyPayload)
                assert payload.token == "typed"
                base = standard_supervised_update_kernels(payload)[
                    "feedbax.training.standard_supervised.gradient_update"
                ]

                def gradient_update(slots, coordinate, context):
                    if context.get("plugin_marker") is not MARKER:
                        raise RuntimeError("prepared kernel context was not delivered")
                    return base(slots, coordinate, context)

                return {{"feedbax.training.standard_supervised.gradient_update": gradient_update}}


            def register_feedbax_training_methods(registry):
                def prepare(request):
                    assert request.run_id == "prepared-cli"
                    assert isinstance(request.method_payload, DummyPayload)
                    assert request.method_payload.token == "typed"
                    assert request.method_contract == request.run_spec.worker_execution.method_contract
                    assert request.effective_phase == request.run_spec.worker_execution.effective_phase
                    return ExecutionPreparationResult(
                        initial_slots={{
                            "model": jnp.array([0.0]),
                            "optimizer": {{"count": jnp.array([1.0])}},
                            "prng": jnp.array([0, 1], dtype=jnp.uint32),
                            "batch_counter": jnp.array(0, dtype=jnp.int32),
                        }},
                        kernel_context={{"plugin_marker": MARKER}},
                    )

                registry.register_descriptor(
                    TrainingMethodDescriptor(
                        method_ref={DUMMY_METHOD_REF!r},
                        payload_schema_id={DUMMY_SCHEMA_ID!r},
                        payload_schema_version={DUMMY_SCHEMA_VERSION!r},
                        payload_model=DummyPayload,
                        contract_compiler=compile_contract,
                        update_kernels_factory=update_kernels,
                        preparation_provider=prepare,
                        owner="dummy_execution_plugin",
                        package="dummy",
                    )
                )
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(repo_root), env.get("PYTHONPATH", "")])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "feedbax",
            "execute-training-run-spec",
            str(spec_path),
            "--plugin",
            "dummy_execution_plugin",
            "--manifest-root",
            str(tmp_path / "runs"),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--run-id",
            "prepared-cli",
            "--no-progress",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["run_id"] == "prepared-cli"
    assert output["status"] == "completed"
    assert Path(output["manifest_path"]).is_file()

    slots_path = tmp_path / "explicit-slots.json"
    slots_path.write_text("{}", encoding="utf-8")
    ambiguous = subprocess.run(
        [
            *completed.args,
            "--initial-slots",
            str(slots_path),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert ambiguous.returncode != 0
    assert "--initial-slots cannot be combined" in ambiguous.stderr

    missing_plugin_path = tmp_path / "missing_preparation_plugin.py"
    missing_plugin_path.write_text(
        textwrap.dedent(
            f"""
            from dummy_execution_plugin import DummyPayload, compile_contract, update_kernels
            from feedbax.contracts.training import TrainingMethodRegistration


            def register_feedbax_training_methods(registry):
                registry.register(
                    TrainingMethodRegistration(
                        method_ref={DUMMY_METHOD_REF!r},
                        payload_schema_id={DUMMY_SCHEMA_ID!r},
                        payload_schema_version={DUMMY_SCHEMA_VERSION!r},
                        payload_model=DummyPayload,
                        contract_factory=None,
                        contract_compiler=compile_contract,
                        update_kernels_factory=update_kernels,
                        requires_execution_preparation=True,
                        owner="missing_preparation_plugin",
                        package="dummy",
                    )
                )
            """
        ),
        encoding="utf-8",
    )
    missing = subprocess.run(
        [
            sys.executable,
            "-m",
            "feedbax",
            "execute-training-run-spec",
            str(spec_path),
            "--plugin",
            "missing_preparation_plugin",
            "--manifest-root",
            str(tmp_path / "missing-runs"),
            "--run-id",
            "missing-prep",
            "--no-progress",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert missing.returncode != 0
    assert "requires an execution-preparation provider, but none is registered" in missing.stderr
    assert not (tmp_path / "missing-runs").exists()


def test_native_cli_plugin_projects_governed_training_manifest_metadata(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "standard-run-spec.json"
    slots_path = tmp_path / "initial-slots.json"
    payload_path = tmp_path / "external-training-payload.json"
    projection_path = tmp_path / "manifest-metadata-projection.json"
    plugin_path = tmp_path / "projection_plugin.py"
    spec_path.write_text(
        json.dumps(_standard_run_spec_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    slots_path.write_text(
        json.dumps(
            {
                "model": 0,
                "optimizer": {"count": 1},
                "prng": [0, 1],
                "batch_counter": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    external_payload = {
        "schema_version": "rlrmp.run_spec.v2",
        "experiment": "projection-cli",
    }
    payload_path.write_text(
        json.dumps(external_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    projection_path.write_text(
        json.dumps(
            {
                "schema_id": "feedbax.spec.training_manifest_metadata_projection",
                "schema_version": "feedbax.spec.training_manifest_metadata_projection.v1",
                "source_payload_kind": "RLRMPRunSpec",
                "source_payload_schema_id": "rlrmp.run_spec",
                "source_payload_schema_version": "rlrmp.run_spec.v2",
                "source_payload_sha256": training_spec_sha256(external_payload),
                "projection_schema_id": "rlrmp.manifest_projection",
                "projection_schema_version": "rlrmp.manifest_projection.v1",
                "values": {"gru_postrun_candidate": True},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    plugin_path.write_text(
        textwrap.dedent(
            """
            from pydantic import BaseModel, ConfigDict

            from feedbax.contracts.training import (
                TrainingManifestMetadataProjectionRegistration,
            )


            class ProjectionValues(BaseModel):
                model_config = ConfigDict(extra="forbid", strict=True)

                gru_postrun_candidate: bool


            def register_feedbax_training_methods(registry):
                registry.register_manifest_metadata_projection(
                    TrainingManifestMetadataProjectionRegistration(
                        source_payload_kind="RLRMPRunSpec",
                        source_payload_schema_id="rlrmp.run_spec",
                        source_payload_schema_version="rlrmp.run_spec.v2",
                        projection_schema_id="rlrmp.manifest_projection",
                        projection_schema_version="rlrmp.manifest_projection.v1",
                        values_model=ProjectionValues,
                        owner="projection_plugin",
                        package="rlrmp",
                    )
                )
            """
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), str(repo_root), os.environ.get("PYTHONPATH", "")]
        ),
    }
    shared_args = [
        "--plugin",
        "projection_plugin",
        "--training-payload",
        str(payload_path),
        "--training-payload-kind",
        "RLRMPRunSpec",
        "--training-payload-schema-id",
        "rlrmp.run_spec",
        "--training-payload-schema-version",
        "rlrmp.run_spec.v2",
        "--manifest-metadata-projection",
        str(projection_path),
    ]

    preflight = subprocess.run(
        [
            sys.executable,
            "-m",
            "feedbax",
            "preflight-training-run-manifest",
            str(spec_path),
            *shared_args,
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert preflight.returncode == 0, preflight.stderr
    preflight_output = json.loads(preflight.stdout)
    assert preflight_output["metadata_projection_custody"]["values"] == {
        "gru_postrun_candidate": True
    }

    executed = subprocess.run(
        [
            sys.executable,
            "-m",
            "feedbax",
            "execute-training-run-spec",
            str(spec_path),
            "--initial-slots",
            str(slots_path),
            "--manifest-root",
            str(tmp_path / "runs"),
            "--run-id",
            "projection-cli",
            "--no-progress",
            *shared_args,
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert executed.returncode == 0, executed.stderr
    manifest = json.loads(executed.stdout)["manifest_payload"]
    assert manifest["metadata"]["gru_postrun_candidate"] is True
    assert manifest["metadata_projection_custody"]["registration_owner"] == ("projection_plugin")
