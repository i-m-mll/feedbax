from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from feedbax.contracts.training import (
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingMethodRegistration,
    TrainingRunSpec,
    WorkerExecutionSpec,
    default_training_method_registry,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
    standard_supervised_update_kernels,
)
from feedbax.plugins.discovery import load_training_method_plugins
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.training.preparation import (
    ExecutionPreparationError,
    ExecutionPreparationProviderRegistry,
    ExecutionPreparationRegistration,
    ExecutionPreparationRequest,
    ExecutionPreparationResult,
    require_execution_preparation_provider,
)


DUMMY_METHOD_REF = "dummy/custom/v1"
DUMMY_SCHEMA_ID = "dummy.spec.training_method"
DUMMY_SCHEMA_VERSION = "dummy.spec.training_method.v1"


class DummyPayload(BaseModel):
    pass


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
        "payload": {},
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


def test_execute_cli_plugin_prepares_non_json_slots_and_kernel_context(tmp_path: Path) -> None:
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
                TrainingMethodRegistration,
                standard_supervised_method_contract,
                standard_supervised_update_kernels,
            )
            from feedbax.training.preparation import (
                ExecutionPreparationRegistration,
                ExecutionPreparationResult,
            )


            MARKER = object()


            class DummyPayload(BaseModel):
                pass


            def update_kernels(payload):
                base = standard_supervised_update_kernels(payload)[
                    "feedbax.training.standard_supervised.gradient_update"
                ]

                def gradient_update(slots, coordinate, context):
                    if context.get("plugin_marker") is not MARKER:
                        raise RuntimeError("prepared kernel context was not delivered")
                    return base(slots, coordinate, context)

                return {{"feedbax.training.standard_supervised.gradient_update": gradient_update}}


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
                        update_kernels_factory=update_kernels,
                        owner="dummy_execution_plugin",
                        package="dummy",
                        requires_execution_preparation=True,
                    )
                )


            def register_feedbax_execution_preparations(registry):
                def prepare(request):
                    assert request.run_id == "prepared-cli"
                    return ExecutionPreparationResult(
                        initial_slots={{
                            "model": jnp.array([0.0]),
                            "optimizer": {{"count": jnp.array([1.0])}},
                            "prng": jnp.array([0, 1], dtype=jnp.uint32),
                            "batch_counter": jnp.array(0, dtype=jnp.int32),
                        }},
                        kernel_context={{"plugin_marker": MARKER}},
                    )

                registry.register(
                    ExecutionPreparationRegistration(
                        method_ref={DUMMY_METHOD_REF!r},
                        provider=prepare,
                        owner="dummy_execution_plugin",
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
        "from dummy_execution_plugin import register_feedbax_training_methods\n",
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
