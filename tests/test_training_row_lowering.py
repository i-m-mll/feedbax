from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from feedbax.bin import orchestrate
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.run_matrix import (
    AuthoredTrainingRow,
    RowLowererIdentity,
    TRAINING_ROW_LOWERER_REF_FIELD,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    TrainingRowLowererRef,
    TrainingRowLoweringResult,
)
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.training import (
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.orchestration import (
    BudgetPolicy,
    CompilerIdentity,
    DeploymentPolicy,
    EnvironmentDeclaration,
    LaunchPolicy,
    RunAssemblyRequest,
    SchemaArtifactRef,
)
from feedbax.training.row_lowering import (
    TrainingRowLowererRegistration,
    TrainingRowLowererRegistry,
    TrainingRowLowererRegistryError,
    training_row_lowerer_implementation_sha256,
)
from feedbax.training.spec_storage import (
    TRAINING_RUN_MATRIX_COMPILER_ID,
    TRAINING_RUN_MATRIX_COMPILER_VERSION,
)


_AUTHORED_SCHEMA_ID = "tests.spec.downstream_authored_row"
_AUTHORED_SCHEMA_VERSION = f"{_AUTHORED_SCHEMA_ID}.v1"
_LOWERER_ID = "tests.downstream-row"
_LOWERER_VERSION = f"{_LOWERER_ID}.v1"


def _execution_payload() -> dict[str, object]:
    method_payload = standard_supervised_method_payload().model_copy(deep=True)
    method_payload.payload["optimizer"]["params"]["learning_rate"] = 0.01
    return TrainingRunSpec(
        graph={
            "inline": {
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
        },
        task=TaskSpec(type="ToyTask", params={"n_steps": 1}),
        training_config=TrainingConfig(n_batches=1, batch_size=1, learning_rate=0.01),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(
                type="target_state",
                label="target",
                selector="port:gain.output",
                target_value=[0.0],
            )
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=method_payload,
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
    ).model_dump(mode="json", exclude_none=True)


def _authored_payload(implementation_sha256: str | None = None) -> dict[str, object]:
    return {
        "schema_id": _AUTHORED_SCHEMA_ID,
        "schema_version": _AUTHORED_SCHEMA_VERSION,
        TRAINING_ROW_LOWERER_REF_FIELD: {
            "schema_id": "feedbax.spec.training_row_lowerer_ref",
            "schema_version": "feedbax.spec.training_row_lowerer_ref.v1",
            "lowerer_id": _LOWERER_ID,
            "lowerer_version": _LOWERER_VERSION,
            "implementation_sha256": implementation_sha256
            or _registration().implementation_sha256,
        },
        "execution_payload": _execution_payload(),
    }


def _registration(*, broken: bool = False) -> TrainingRowLowererRegistration:
    def lower(row: AuthoredTrainingRow) -> TrainingRowLoweringResult:
        if broken:
            raise RuntimeError("broken by design")
        return TrainingRowLoweringResult(
            execution_payload=row.payload["execution_payload"],
            lowerer_identities=[
                RowLowererIdentity(
                    lowerer_id=_LOWERER_ID,
                    lowerer_version=_LOWERER_VERSION,
                )
            ],
        )

    return TrainingRowLowererRegistration(
        authored_schema_id=_AUTHORED_SCHEMA_ID,
        authored_schema_version=_AUTHORED_SCHEMA_VERSION,
        lowerer_id=_LOWERER_ID,
        lowerer_version=_LOWERER_VERSION,
        implementation_sha256=training_row_lowerer_implementation_sha256(lower),
        lower=lower,
        owner="tests",
    )


def _write_request(
    tmp_path: Path, *, orchestration_name: str, implementation_sha256: str | None = None
) -> Path:
    authored_payload = _authored_payload(implementation_sha256)
    base_path = tmp_path / "downstream-row.json"
    base_path.write_text(json.dumps(authored_payload), encoding="utf-8")
    matrix = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "downstream authored matrix",
        "base": {
            "kind": "authored_intent",
            "ref": base_path.name,
            "content_hash": training_spec_sha256(authored_payload),
        },
        "rows": [{"row_id": "downstream-row", "seed": 7}],
    }
    matrix_bytes = json.dumps(matrix, sort_keys=True).encode("utf-8")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_bytes(matrix_bytes)
    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            schema_version=TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            artifact_id=f"fixture:{hashlib.sha256(matrix_bytes).hexdigest()}",
            sha256=hashlib.sha256(matrix_bytes).hexdigest(),
            uri=str(matrix_path),
        ),
        compiler=CompilerIdentity(
            compiler_id=TRAINING_RUN_MATRIX_COMPILER_ID,
            compiler_version=TRAINING_RUN_MATRIX_COMPILER_VERSION,
        ),
        deployment_policy=DeploymentPolicy(
            driver="local",
            venue="local",
            cloud_authorized=False,
            review_required=False,
            review_authorized=False,
        ),
        environment=EnvironmentDeclaration(python_version="3.12"),
        launch_policy=LaunchPolicy(max_parallel_rows=1),
        budget=BudgetPolicy(max_wall_clock_seconds=30.0),
        orchestration_root=str(tmp_path / orchestration_name),
    )
    request_path = tmp_path / f"{orchestration_name}.request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    return request_path


def test_orchestration_cli_discovers_and_lowers_downstream_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = TrainingRowLowererRegistry()
    plugin = SimpleNamespace(
        register_feedbax_training_row_lowerers=lambda target: target.register(
            _registration()
        )
    )
    monkeypatch.setattr(
        "feedbax.training.row_lowering.DEFAULT_TRAINING_ROW_LOWERER_REGISTRY",
        registry,
    )
    monkeypatch.setattr(
        "feedbax.plugins.discovery.feedbax_plugin_entry_points",
        lambda _group: [SimpleNamespace(name="downstream", load=lambda: plugin)],
    )
    monkeypatch.chdir(tmp_path)
    request_path = _write_request(tmp_path, orchestration_name="cli")

    result = orchestrate.main(["preflight", "--assembly-request", str(request_path)])

    assert result == 0
    bundle_path = next((tmp_path / "cli").glob("*/bundle.json"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    provenance = bundle["rows"][0]["execution"]["row_provenance"]
    assert provenance["lowerer_identities"] == [
        {"lowerer_id": _LOWERER_ID, "lowerer_version": _LOWERER_VERSION}
    ]
    assert provenance["authored_payload_hash"] == training_spec_sha256(
        _authored_payload()
    )
    assert provenance["lowered_execution_payload_hash"] == bundle["rows"][0][
        "execution"
    ]["payload"]["sha256"]


def test_registry_fails_closed_on_missing_broken_ambiguous_and_drifted_lowerers() -> None:
    row = AuthoredTrainingRow(
        row_id="row",
        row_index=0,
        payload=_authored_payload(),
        payload_hash=training_spec_sha256(_authored_payload()),
        axis_coordinates={},
    )
    with pytest.raises(TrainingRowLowererRegistryError, match="no exact"):
        TrainingRowLowererRegistry().lower(row)

    broken = TrainingRowLowererRegistry()
    broken.register(_registration(broken=True))
    with pytest.raises(TrainingRowLowererRegistryError, match="broken by design"):
        broken.lower(row)

    ambiguous = TrainingRowLowererRegistry()
    ambiguous.register(_registration())
    with pytest.raises(TrainingRowLowererRegistryError, match="ambiguous"):
        ambiguous.register(_registration())

    drifted = TrainingRowLowererRegistry()
    registration = _registration()
    drifted.register(
        TrainingRowLowererRegistration(
            **{
                **registration.__dict__,
                "lower": lambda authored: TrainingRowLoweringResult(
                    execution_payload=authored.payload["execution_payload"],
                    lowerer_identities=[
                        RowLowererIdentity(
                            lowerer_id=_LOWERER_ID,
                            lowerer_version=f"{_LOWERER_ID}.v2",
                        )
                    ],
                ),
            }
        )
    )
    with pytest.raises(TrainingRowLowererRegistryError, match="drifted identity"):
        drifted.lower(row)


def test_lowerer_reference_rejects_unsupported_versions() -> None:
    payload = _authored_payload()[TRAINING_ROW_LOWERER_REF_FIELD]
    assert isinstance(payload, dict)
    with pytest.raises(ValidationError, match="schema_version"):
        TrainingRowLowererRef.model_validate(
            {**payload, "schema_version": "feedbax.spec.training_row_lowerer_ref.v0"}
        )
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            "TrainingRowLowererRef",
            {"schema_version": "feedbax.spec.training_row_lowerer_ref.v0"},
        )


def test_installed_plugin_replays_identical_rows_across_fresh_processes(
    tmp_path: Path,
) -> None:
    plugin_path = tmp_path / "downstream_row_plugin.py"
    plugin_path.write_text(
        """
from feedbax.contracts.run_matrix import RowLowererIdentity, TrainingRowLoweringResult
from feedbax.training.row_lowering import TrainingRowLowererRegistration, training_row_lowerer_implementation_sha256

def register_feedbax_training_row_lowerers(registry):
    def lower(row):
        return TrainingRowLoweringResult(
            execution_payload=row.payload["execution_payload"],
            lowerer_identities=[RowLowererIdentity(
                lowerer_id="tests.downstream-row",
                lowerer_version="tests.downstream-row.v1",
            )],
        )
    registry.register(TrainingRowLowererRegistration(
        authored_schema_id="tests.spec.downstream_authored_row",
        authored_schema_version="tests.spec.downstream_authored_row.v1",
        lowerer_id="tests.downstream-row",
        lowerer_version="tests.downstream-row.v1",
        implementation_sha256=training_row_lowerer_implementation_sha256(lower),
        lower=lower,
        owner="fresh-process-plugin",
    ))
""".lstrip(),
        encoding="utf-8",
    )
    request_path = _write_request(
        tmp_path,
        orchestration_name="fresh",
        implementation_sha256=hashlib.sha256(plugin_path.read_bytes()).hexdigest(),
    )
    dist_info = tmp_path / "downstream_row_plugin-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "entry_points.txt").write_text(
        "[feedbax.plugins]\ndownstream-row = downstream_row_plugin\n",
        encoding="utf-8",
    )
    script = """
import json
from pathlib import Path
import sys
from feedbax.bin.orchestrate import _load_assembly_request
from feedbax.orchestration import AssemblyContext, assemble_run_bundle, build_default_assembly_registry
from feedbax.plugins import load_training_method_plugins

load_training_method_plugins(fail_on_load_error=True)
request = _load_assembly_request(sys.argv[1])
bundle = assemble_run_bundle(
    request,
    run_set_id="fresh-process",
    context=AssemblyContext(custody_root=Path(sys.argv[2]), repo_root=Path(sys.argv[3])),
    registry=build_default_assembly_registry(),
)
row = bundle.rows[0]
print(json.dumps({
    "payload": row.execution.payload.sha256,
    "planned": row.execution.row_provenance.planned_run_id,
    "provenance": row.execution.row_provenance.model_dump(mode="json"),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(Path(__file__).parents[1])]
    )
    observed = []
    for index in range(2):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(request_path),
                str(tmp_path / f"custody-{index}"),
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        observed.append(json.loads(result.stdout))

    assert observed[0] == observed[1]
    plugin_path.write_text(plugin_path.read_text(encoding="utf-8") + "# drift\n")
    with pytest.raises(subprocess.CalledProcessError) as error:
        subprocess.run(
            [sys.executable, "-c", script, str(request_path), str(tmp_path / "drift"), str(tmp_path)],
            check=True, capture_output=True, text=True, env=environment,
        )
    assert "implementation drifted" in error.value.stderr


def test_explicit_training_run_rows_are_not_attributed_to_lowerers() -> None:
    registry = TrainingRowLowererRegistry()
    row = AuthoredTrainingRow(
        row_id="explicit",
        row_index=0,
        payload=_execution_payload(),
        payload_hash=training_spec_sha256(_execution_payload()),
        axis_coordinates={},
    )

    assert registry.lower(row) is None
