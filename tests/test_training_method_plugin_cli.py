from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

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
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(repo_root), env.get("PYTHONPATH", "")]
    )
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
