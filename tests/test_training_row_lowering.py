from __future__ import annotations

import asyncio
from dataclasses import replace
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
from feedbax.contracts.run_composition import (
    AuthoredIntentParent,
    CompositionNode,
    ResolvedOutputParent,
    authored_envelope_hash,
    flatten_composition,
)
from feedbax.contracts.spec_storage import (
    training_spec_canonical_bytes,
    training_spec_sha256,
)
from feedbax.contracts.training import (
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    default_training_method_registry,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.orchestration import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    BudgetPolicy,
    CompilerIdentity,
    DeploymentPolicy,
    EnvironmentDeclaration,
    GovernedTrainingRowParentDeclaration,
    LaunchPolicy,
    RunAssemblyRequest,
    SchemaArtifactRef,
    assemble_run_bundle,
    run_authority_preflight_checks,
)
from feedbax.plugins import (
    ROW_LOWERERS,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
    bootstrap_application,
    new_registration_context,
)
import feedbax.plugins.bootstrap as plugin_bootstrap
from feedbax.training.row_lowering import (
    TrainingRowLowererRegistration,
    TrainingRowLowererRegistry,
    TrainingRowLowererRegistryError,
    TrainingRowLoweringContext,
    training_row_lowerer_implementation_sha256,
)
from feedbax.training.spec_storage import (
    TRAINING_RUN_MATRIX_COMPILER_ID,
    TRAINING_RUN_MATRIX_COMPILER_VERSION,
    register_training_run_matrix_compiler,
)
from feedbax.orchestration.revision import resolve_feedbax_revision


_AUTHORED_SCHEMA_ID = "tests.spec.downstream_authored_row"
_AUTHORED_SCHEMA_VERSION = f"{_AUTHORED_SCHEMA_ID}.v1"
_LOWERER_ID = "tests.downstream-row"
_LOWERER_VERSION = f"{_LOWERER_ID}.v1"


def _lowerer_plugin(plugin_id: str = "tests.downstream") -> PluginRegistration:
    return PluginRegistration(
        PluginDeclaration(
            plugin_id,
            "1",
            1,
            families=(FamilyRequirement("row_lowerers"),),
        ),
        lambda context: context.registry(ROW_LOWERERS).register(_registration()),
    )


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
            "schema_version": "feedbax.spec.training_row_lowerer_ref.v2",
            "context_api_version": "feedbax.training_row_lowering_context.v1",
            "lowerer_id": _LOWERER_ID,
            "lowerer_version": _LOWERER_VERSION,
            "implementation_sha256": implementation_sha256 or _registration().implementation_sha256,
        },
        "execution_payload": _execution_payload(),
    }


def _registration(*, broken: bool = False) -> TrainingRowLowererRegistration:
    def lower(row: AuthoredTrainingRow, _context: object) -> TrainingRowLoweringResult:
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
        feedbax_revision=resolve_feedbax_revision(),
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


def _write_parent_request(
    tmp_path: Path,
    *,
    orchestration_name: str,
    implementation_sha256: str,
) -> Path:
    terminal = _execution_payload()
    terminal_parent = ResolvedOutputParent(
        ref="terminal",
        resolved_root_hash=training_spec_sha256(terminal),
    )
    authored_parent = CompositionNode(name="parent", parent=terminal_parent)
    child_parent = AuthoredIntentParent(
        ref="parent",
        content_hash=authored_envelope_hash(authored_parent),
    )
    authored_payload = _authored_payload(implementation_sha256)
    authored_payload.pop("execution_payload")
    authored_payload["composition"] = CompositionNode(
        name="child",
        parent=child_parent,
    ).model_dump(mode="json", exclude_none=True)
    authored_path = tmp_path / "parent-aware-row.json"
    authored_path.write_bytes(training_spec_canonical_bytes(authored_payload))
    matrix = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "parent-aware authored matrix",
        "base": {
            "kind": "authored_intent",
            "ref": authored_path.name,
            "content_hash": training_spec_sha256(authored_payload),
        },
        "rows": [{"row_id": "parent-aware-row", "seed": 7}],
    }
    documents = {
        "matrix": matrix,
        "parent": authored_parent.model_dump(mode="json", exclude_none=True),
        "terminal": terminal,
    }
    refs = {}
    for name, document in documents.items():
        data = training_spec_canonical_bytes(document)
        path = tmp_path / f"{name}.json"
        path.write_bytes(data)
        refs[name] = SchemaArtifactRef(
            schema_id=str(document["schema_id"]),
            schema_version=str(document["schema_version"]),
            artifact_id=f"fixture:{name}",
            sha256=hashlib.sha256(data).hexdigest(),
            uri=str(path),
        )
    request = RunAssemblyRequest(
        feedbax_revision=resolve_feedbax_revision(),
        authored=refs["matrix"],
        compiler=CompilerIdentity(
            compiler_id=TRAINING_RUN_MATRIX_COMPILER_ID,
            compiler_version=TRAINING_RUN_MATRIX_COMPILER_VERSION,
        ),
        training_row_parents=[
            GovernedTrainingRowParentDeclaration(
                role="parent",
                parent=child_parent,
                artifact=refs["parent"],
            ),
            GovernedTrainingRowParentDeclaration(
                role="terminal",
                parent=terminal_parent,
                artifact=refs["terminal"],
            ),
        ],
        deployment_policy=DeploymentPolicy(
            driver="local",
            venue="local",
            cloud_authorized=False,
            review_required=False,
            review_authorized=False,
        ),
        environment=EnvironmentDeclaration(python_version="3.13"),
        launch_policy=LaunchPolicy(max_parallel_rows=1),
        budget=BudgetPolicy(max_wall_clock_seconds=30),
        orchestration_root=str(tmp_path / orchestration_name),
    )
    request_path = tmp_path / f"{orchestration_name}.request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    return request_path


def test_orchestration_cli_discovers_and_lowers_downstream_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        plugin_bootstrap,
        "_installed_entry_points",
        lambda _group: (
            SimpleNamespace(
                name="downstream",
                value="tests:PLUGIN_REGISTRATION",
                load=_lowerer_plugin,
            ),
        ),
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
    assert provenance["authored_payload_hash"] == training_spec_sha256(_authored_payload())
    assert (
        provenance["lowered_execution_payload_hash"]
        == bundle["rows"][0]["execution"]["payload"]["sha256"]
    )


def test_registry_exact_registration_replay_is_a_collision() -> None:
    registry = TrainingRowLowererRegistry()
    registration = _registration()

    registry.register(registration)
    with pytest.raises(TrainingRowLowererRegistryError, match="ambiguous"):
        registry.register(_registration())


def test_plugin_registration_replays_in_isolated_contexts() -> None:
    states = tuple(
        asyncio.run(
            bootstrap_application(
                new_registration_context(local_component_source=None),
                registrations=(_lowerer_plugin(f"tests.downstream.{index}"),),
            )
        )
        for index in range(2)
    )

    expected = (
        (
            _AUTHORED_SCHEMA_ID,
            _AUTHORED_SCHEMA_VERSION,
            _LOWERER_ID,
            _LOWERER_VERSION,
        ),
    )
    assert all(state.bundle.row_lowerers.available_keys() == expected for state in states)


def test_registry_rejects_conflicting_registration_owner() -> None:
    registry = TrainingRowLowererRegistry()
    registration = _registration()
    registry.register(registration)

    with pytest.raises(
        TrainingRowLowererRegistryError,
        match="ambiguous.*owners 'tests' and 'other'",
    ):
        registry.register(replace(registration, owner="other"))

    assert registry.available_keys() == (
        (
            _AUTHORED_SCHEMA_ID,
            _AUTHORED_SCHEMA_VERSION,
            _LOWERER_ID,
            _LOWERER_VERSION,
        ),
    )


def test_registry_rejects_conflicting_registration_implementation() -> None:
    registry = TrainingRowLowererRegistry()
    registration = _registration()
    registry.register(registration)

    with pytest.raises(
        TrainingRowLowererRegistryError,
        match="ambiguous.*implementation sha256 values",
    ):
        registry.register(replace(registration, implementation_sha256="0" * 64))

    assert registry.available_keys() == (
        (
            _AUTHORED_SCHEMA_ID,
            _AUTHORED_SCHEMA_VERSION,
            _LOWERER_ID,
            _LOWERER_VERSION,
        ),
    )


def test_registry_fails_closed_on_missing_broken_and_drifted_lowerers() -> None:
    row = AuthoredTrainingRow(
        row_id="row",
        row_index=0,
        payload=_authored_payload(),
        payload_hash=training_spec_sha256(_authored_payload()),
        axis_coordinates={},
    )
    with pytest.raises(TrainingRowLowererRegistryError, match="no exact"):
        TrainingRowLowererRegistry().lower(row, TrainingRowLoweringContext())

    broken = TrainingRowLowererRegistry()
    broken.register(_registration(broken=True))
    with pytest.raises(TrainingRowLowererRegistryError, match="broken by design"):
        broken.lower(row, TrainingRowLoweringContext())

    drifted = TrainingRowLowererRegistry()
    registration = _registration()
    drifted.register(
        TrainingRowLowererRegistration(
            **{
                **registration.__dict__,
                "lower": lambda authored, _context: TrainingRowLoweringResult(
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
        drifted.lower(row, TrainingRowLoweringContext())


def test_assembly_supplies_exact_composition_parents_and_binds_provenance(
    tmp_path: Path,
) -> None:
    terminal = _execution_payload()
    terminal_parent = ResolvedOutputParent(
        ref="terminal",
        resolved_root_hash=training_spec_sha256(terminal),
    )
    authored_parent = CompositionNode(name="parent", parent=terminal_parent)
    child_parent = AuthoredIntentParent(
        ref="parent",
        content_hash=authored_envelope_hash(authored_parent),
    )
    child = CompositionNode(name="child", parent=child_parent)

    def lower(
        row: AuthoredTrainingRow,
        context: TrainingRowLoweringContext,
    ) -> TrainingRowLoweringResult:
        copy_one = context.resolve_parent(child_parent)
        assert isinstance(copy_one, CompositionNode)
        copy_one.name = "mutated"
        assert context.resolve_parent(child_parent).name == "parent"
        flattened = flatten_composition(
            CompositionNode.model_validate(row.payload["composition"]),
            context.resolve_parent,
        )
        return TrainingRowLoweringResult(
            execution_payload=flattened.payload,
            lowerer_identities=[
                RowLowererIdentity(
                    lowerer_id=_LOWERER_ID,
                    lowerer_version=_LOWERER_VERSION,
                )
            ],
        )

    registration = TrainingRowLowererRegistration(
        authored_schema_id=_AUTHORED_SCHEMA_ID,
        authored_schema_version=_AUTHORED_SCHEMA_VERSION,
        lowerer_id=_LOWERER_ID,
        lowerer_version=_LOWERER_VERSION,
        implementation_sha256=training_row_lowerer_implementation_sha256(lower),
        lower=lower,
        owner="tests",
    )
    lowerers = TrainingRowLowererRegistry()
    lowerers.register(registration)
    authored_row = {
        **_authored_payload(registration.implementation_sha256),
        "composition": child.model_dump(mode="json", exclude_none=True),
    }
    authored_row.pop("execution_payload")
    authored_path = tmp_path / "authored-row.json"
    authored_path.write_bytes(training_spec_canonical_bytes(authored_row))
    matrix = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "declared parents",
        "base": {
            "kind": "authored_intent",
            "ref": authored_path.name,
            "content_hash": training_spec_sha256(authored_row),
        },
        "rows": [{"row_id": "row", "seed": 7}],
    }
    payloads = {
        "matrix": training_spec_canonical_bytes(matrix),
        "parent": training_spec_canonical_bytes(
            authored_parent.model_dump(mode="json", exclude_none=True)
        ),
        "terminal": training_spec_canonical_bytes(terminal),
    }

    def artifact_ref(name: str, payload: dict[str, object]) -> SchemaArtifactRef:
        return SchemaArtifactRef(
            schema_id=str(payload["schema_id"]),
            schema_version=str(payload["schema_version"]),
            artifact_id=name,
            sha256=hashlib.sha256(payloads[name]).hexdigest(),
        )

    request = RunAssemblyRequest(
        feedbax_revision=resolve_feedbax_revision(),
        authored=artifact_ref("matrix", matrix),
        compiler=CompilerIdentity(
            compiler_id=TRAINING_RUN_MATRIX_COMPILER_ID,
            compiler_version=TRAINING_RUN_MATRIX_COMPILER_VERSION,
        ),
        training_row_parents=[
            GovernedTrainingRowParentDeclaration(
                role="parent",
                parent=child_parent,
                artifact=artifact_ref(
                    "parent",
                    authored_parent.model_dump(mode="json", exclude_none=True),
                ),
            ),
            GovernedTrainingRowParentDeclaration(
                role="terminal",
                parent=terminal_parent,
                artifact=artifact_ref("terminal", terminal),
            ),
        ],
        deployment_policy=DeploymentPolicy(
            driver="local",
            venue="local",
            cloud_authorized=False,
            review_required=False,
            review_authorized=False,
        ),
        environment=EnvironmentDeclaration(python_version="3.13"),
        launch_policy=LaunchPolicy(max_parallel_rows=1),
        budget=BudgetPolicy(max_wall_clock_seconds=30),
    )
    registry = AssemblyCompilerRegistry()
    register_training_run_matrix_compiler(
        registry,
        method_registry=default_training_method_registry(),
        row_lowerer=lowerers.lower,
        row_validator=lambda payload, _row_id: TrainingRunSpec.model_validate(payload),
    )

    def assemble(candidate: RunAssemblyRequest, name: str):
        return assemble_run_bundle(
            candidate,
            run_set_id=name,
            context=AssemblyContext(
                custody_root=tmp_path / name,
                repo_root=tmp_path,
                artifact_resolver=lambda ref: bytes(payloads[ref.artifact_id]),
            ),
            registry=registry,
        )

    bundle = assemble(request, "valid")
    provenance = bundle.rows[0].execution.row_provenance
    assert provenance is not None
    assert [(item.parent_kind, item.ref) for item in provenance.parent_inputs] == [
        ("authored_intent", "parent"),
        ("resolved_output", "terminal"),
    ]
    assert all(check.status == "pass" for check in run_authority_preflight_checks(bundle))

    with pytest.raises(TrainingRowLowererRegistryError, match="missing or undeclared"):
        assemble(
            request.model_copy(update={"training_row_parents": request.training_row_parents[:1]}),
            "missing",
        )
    duplicated = request.model_dump(mode="json")
    duplicated["training_row_parents"].append(duplicated["training_row_parents"][0])
    with pytest.raises(ValidationError, match="ambiguous"):
        RunAssemblyRequest.model_validate(duplicated)
    payloads["terminal"] = training_spec_canonical_bytes(
        {**terminal, "metadata": {"tampered": True}}
    )
    with pytest.raises(ValueError, match="artifact byte digest mismatch"):
        assemble(request, "tampered")


def test_lowerer_reference_rejects_unsupported_versions() -> None:
    payload = _authored_payload()[TRAINING_ROW_LOWERER_REF_FIELD]
    assert isinstance(payload, dict)
    for version in (
        "feedbax.spec.training_row_lowerer_ref.v0",
        "feedbax.spec.training_row_lowerer_ref.v1",
    ):
        with pytest.raises(ValidationError, match="schema_version"):
            TrainingRowLowererRef.model_validate({**payload, "schema_version": version})
        with pytest.raises(
            UnsupportedSpecVersion,
            match="migration_intentionally_absent=yes",
        ):
            default_spec_registry.migrate(
                "TrainingRowLowererRef",
                {"schema_version": version},
            )


def test_installed_plugin_replays_identical_rows_across_fresh_processes(
    tmp_path: Path,
) -> None:
    plugin_path = tmp_path / "downstream_row_plugin.py"
    plugin_path.write_text(
        """
from feedbax.contracts.run_composition import CompositionNode, flatten_composition
from feedbax.contracts.run_matrix import RowLowererIdentity, TrainingRowLoweringResult
from feedbax.plugins import FamilyRequirement, PluginDeclaration, PluginRegistration, ROW_LOWERERS
from feedbax.training.row_lowering import TrainingRowLowererRegistration, training_row_lowerer_implementation_sha256

def register(context):
    def lower(row, context):
        flattened = flatten_composition(
            CompositionNode.model_validate(row.payload["composition"]),
            context.resolve_parent,
        )
        return TrainingRowLoweringResult(
            execution_payload=flattened.payload,
            lowerer_identities=[RowLowererIdentity(
                lowerer_id="tests.downstream-row",
                lowerer_version="tests.downstream-row.v1",
            )],
        )
    context.registry(ROW_LOWERERS).register(TrainingRowLowererRegistration(
        authored_schema_id="tests.spec.downstream_authored_row",
        authored_schema_version="tests.spec.downstream_authored_row.v1",
        lowerer_id="tests.downstream-row",
        lowerer_version="tests.downstream-row.v1",
        implementation_sha256=training_row_lowerer_implementation_sha256(lower),
        lower=lower,
        owner="fresh-process-plugin",
    ))

PLUGIN_REGISTRATION = PluginRegistration(
    PluginDeclaration(
        "tests.downstream_row",
        "1",
        1,
        families=(FamilyRequirement("row_lowerers"),),
    ),
    register,
)
""".lstrip(),
        encoding="utf-8",
    )
    request_path = _write_parent_request(
        tmp_path,
        orchestration_name="fresh",
        implementation_sha256=hashlib.sha256(plugin_path.read_bytes()).hexdigest(),
    )
    dist_info = tmp_path / "downstream_row_plugin-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "entry_points.txt").write_text(
        "[feedbax.plugins]\ndownstream-row = downstream_row_plugin:PLUGIN_REGISTRATION\n",
        encoding="utf-8",
    )
    script = """
import json
import asyncio
from pathlib import Path
import sys
from feedbax.bin.orchestrate import _load_assembly_request
from feedbax.orchestration import AssemblyContext, assemble_run_bundle, build_default_assembly_registry
from feedbax.plugins.composition import compose_application

state = asyncio.run(compose_application(local_component_source=None))
request = _load_assembly_request(sys.argv[1])
bundle = assemble_run_bundle(
    request,
    run_set_id="fresh-process",
    context=AssemblyContext(custody_root=Path(sys.argv[2]), repo_root=Path(sys.argv[3])),
    registry=build_default_assembly_registry(
        method_registry=state.bundle.training_methods,
        row_lowerer_registry=state.bundle.row_lowerers,
        evaluation_registry=state.bundle.evaluation_recipes,
    ),
)
row = bundle.rows[0]
print(json.dumps({
    "payload": row.execution.payload.sha256,
    "planned": row.execution.row_provenance.planned_run_id,
    "provenance": row.execution.row_provenance.model_dump(mode="json"),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path(__file__).parents[1])])
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
    assert [
        (item["parent_kind"], item["ref"]) for item in observed[0]["provenance"]["parent_inputs"]
    ] == [("authored_intent", "parent"), ("resolved_output", "terminal")]
    plugin_path.write_text(plugin_path.read_text(encoding="utf-8") + "# drift\n")
    with pytest.raises(subprocess.CalledProcessError) as error:
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(request_path),
                str(tmp_path / "drift"),
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
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

    assert registry.lower(row, TrainingRowLoweringContext()) is None
