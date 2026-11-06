"""Exercise a matrix seed through native CLI preparation and checkpoint publication."""

from __future__ import annotations

import asyncio
from pathlib import Path

import jax.random as jr
import numpy as np
import pytest

from feedbax import __main__ as feedbax_main
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.run_matrix import (
    TRAINING_ROW_LOWERER_REF_FIELD,
    AuthoredTrainingRow,
    RowLowererIdentity,
    TrainingRowLowererRef,
    TrainingRowLoweringResult,
    TrainingRunMatrixSpec,
)
from feedbax.contracts.spec_storage import training_spec_canonical_bytes, training_spec_sha256
from feedbax.contracts.training import (
    CheckpointProgressPolicySpec,
    LossTermSpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    ObjectiveSlotSpec,
    OptimizerSpec,
    StandardSupervisedMethodPayload,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    evolve_training_program,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
)
from feedbax.orchestration.assembly import (
    AssemblyContext,
    CompilerIdentity,
    RunAssemblyRequest,
    assemble_run_bundle,
    build_default_assembly_registry,
)
from feedbax.orchestration.bundle import (
    BudgetPolicy,
    DeploymentPolicy,
    EnvironmentDeclaration,
    SchemaArtifactRef,
)
from feedbax.orchestration.drivers.native_execution import (
    bind_native_execution_command,
    inject_native_execution_context,
)
from feedbax.orchestration.revision import resolve_feedbax_revision
from feedbax.plugins import (
    TRAINING_PROGRAMS,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
)
from feedbax.plugins.composition import compose_application
from feedbax.training.checkpoint_custody import (
    authenticate_published_checkpoint_custody,
    resolve_checkpoint_custody_ref,
)
from feedbax.training.preparation import ExecutionPreparationRequest, ExecutionPreparationResult
from feedbax.training.row_lowering import (
    TrainingRowLowererRegistration,
    TrainingRowLoweringContext,
    training_row_lowerer_implementation_sha256,
)
from feedbax.training.run_matrix import RunMatrixError
from feedbax.training.spec_storage import (
    TRAINING_RUN_MATRIX_COMPILER_ID,
    TRAINING_RUN_MATRIX_COMPILER_VERSION,
)
from tests.support.training_fixtures import (
    initial_training_slots,
    synthetic_supervised_training_program,
)


METHOD_REF = "tests/matrix_seed/v1"
METHOD_SCHEMA = "tests.spec.matrix_seed"
METHOD_VERSION = "tests.spec.matrix_seed.v1"
AUTHORED_SCHEMA = "tests.spec.matrix_seed_row"
AUTHORED_VERSION = "tests.spec.matrix_seed_row.v1"
LOWERER = RowLowererIdentity(lowerer_id="tests.matrix_seed", lowerer_version="v1")


class SeededPayload(StandardSupervisedMethodPayload):
    random_seed: int


def _contract(_payload: SeededPayload):
    return standard_supervised_method_contract().model_copy(
        update={"method_ref": METHOD_REF, "method_payload_schema_version": METHOD_VERSION}
    )


def _prepare(request: ExecutionPreparationRequest) -> ExecutionPreparationResult:
    assert isinstance(request.method_payload, SeededPayload)
    slots = initial_training_slots()
    slots["prng"] = jr.key_data(jr.split(jr.key(request.method_payload.random_seed))[0])
    return ExecutionPreparationResult(initial_slots=slots)


def _lower(row: AuthoredTrainingRow, _context: TrainingRowLoweringContext):
    spec = row.payload["run"]
    bound = row.payload["bind_seed"]
    if bound:
        spec["method_payload"]["payload"]["random_seed"] = row.seed
    return TrainingRowLoweringResult(
        execution_payload=spec,
        lowerer_identities=[LOWERER],
        execution_seed_path="method_payload.payload.random_seed" if bound else None,
    )


def _register(context) -> None:
    context.registry(TRAINING_PROGRAMS).register_program(
        evolve_training_program(
            synthetic_supervised_training_program(),
            method_ref=METHOD_REF,
            payload_schema_id=METHOD_SCHEMA,
            payload_schema_version=METHOD_VERSION,
            payload_model=SeededPayload,
            contract_compiler=_contract,
            preparation_provider=_prepare,
            row_lowerers=(
                TrainingRowLowererRegistration(
                    authored_schema_id=AUTHORED_SCHEMA,
                    authored_schema_version=AUTHORED_VERSION,
                    lowerer_id=LOWERER.lowerer_id,
                    lowerer_version=LOWERER.lowerer_version,
                    implementation_sha256=training_row_lowerer_implementation_sha256(_lower),
                    lower=_lower,
                ),
            ),
            package="tests",
        )
    )


PLUGIN_REGISTRATION = PluginRegistration(
    PluginDeclaration(
        "tests.matrix_seed", "1", 1, families=(FamilyRequirement("training_programs"),)
    ),
    _register,
)


def _base(bind_seed: bool) -> dict:
    payload = SeededPayload(
        random_seed=0, optimizer=OptimizerSpec(type="adamw", params={"learning_rate": 0.001})
    )
    run_spec = TrainingRunSpec(
        graph={
            "inline": {
                "nodes": {
                    "gain": {
                        "type": "Gain",
                        "params": {"gain": 1},
                        "input_ports": ["input"],
                        "output_ports": ["output"],
                    },
                },
                "wires": [],
                "input_ports": ["input"],
                "output_ports": ["output"],
                "input_bindings": {"input": ["gain", "input"]},
                "output_bindings": {"output": ["gain", "output"]},
            },
        },
        task=TaskSpec(type="ToyTask", params={"n_steps": 1}),
        training_config=TrainingConfig(n_batches=1, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(
                type="target_state", label="target", selector="port:gain.output", target_value=[0.0]
            )
        ),
        method_ref=MethodRefSpec(package="tests", name="matrix_seed", version="v1"),
        method_payload=MethodPayloadEnvelope(
            schema_id=METHOD_SCHEMA,
            schema_version=METHOD_VERSION,
            payload=payload.model_dump(mode="json"),
        ),
        worker_execution=WorkerExecutionSpec(
            method_contract=_contract(payload),
            effective_phase=standard_supervised_effective_phase_spec().model_copy(
                update={"method_ref": METHOD_REF}
            ),
        ),
        checkpoint_progress=CheckpointProgressPolicySpec(checkpoint_interval=1),
    )
    return {
        "schema_id": AUTHORED_SCHEMA,
        "schema_version": AUTHORED_VERSION,
        TRAINING_ROW_LOWERER_REF_FIELD: TrainingRowLowererRef(
            **LOWERER.model_dump(),
            implementation_sha256=training_row_lowerer_implementation_sha256(_lower),
        ).model_dump(mode="json"),
        "run": run_spec.model_dump(mode="json", exclude_none=True),
        "bind_seed": bind_seed,
    }


def _assemble(root: Path, *, base: dict, **matrix_fields):
    base_path = root / "base.json"
    base_path.write_bytes(training_spec_canonical_bytes(base))
    matrix = TrainingRunMatrixSpec(
        name="native matrix RNG",
        base={
            "kind": "authored_intent",
            "ref": base_path.name,
            "content_hash": training_spec_sha256(base),
        },
        **matrix_fields,
    )
    authored = training_spec_canonical_bytes(matrix.model_dump(mode="json", exclude_none=True))
    authored_path = root / "matrix.json"
    authored_path.write_bytes(authored)
    methods = asyncio.run(compose_application(modules=(__name__,))).bundle
    registry = build_default_assembly_registry(
        method_registry=methods.training_programs,
        row_lowerer_registry=methods.row_lowerers,
        evaluation_registry=methods.evaluation_recipes,
    )
    return assemble_run_bundle(
        RunAssemblyRequest(
            feedbax_revision=resolve_feedbax_revision(),
            authored=SchemaArtifactRef(
                schema_id=matrix.schema_id,
                schema_version=matrix.schema_version,
                artifact_id="fixture:matrix-seed",
                sha256=sha256_bytes(authored),
                uri=str(authored_path),
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
            environment=EnvironmentDeclaration(),
            budget=BudgetPolicy(max_wall_clock_seconds=30),
        ),
        run_set_id="matrix-seed",
        context=AssemblyContext(custody_root=root / "custody", repo_root=root),
        registry=registry,
    )


def test_matrix_seed_controls_native_initial_prng(tmp_path: Path) -> None:
    bundle = _assemble(
        tmp_path,
        base=_base(bind_seed=True),
        axes=[{"id": "replica", "path": "seed", "variation": {"values": [7, 8, 7]}}],
    )
    keys = []
    for row in bundle.rows:
        row_root = tmp_path / row.row_id
        command, bound_row = bind_native_execution_command(
            [*row.launch.command, "--plugin", __name__],
            row=row,
            payload_path=row.execution.payload.uri,
            collection_root=row_root,
        )
        command = inject_native_execution_context(
            command, row=bound_row, environment_fingerprint="matrix-seed", collection_root=row_root
        )
        assert feedbax_main.main(command[3:]) == 0
        checkpoint_root = row_root / "checkpoints"
        checkpoint = authenticate_published_checkpoint_custody(checkpoint_root)
        keys.append(
            resolve_checkpoint_custody_ref(
                checkpoint.parent_ref, allowed_root=checkpoint_root, slot_names=("prng",)
            ).slots["prng"]
        )

    first, different, repeat = keys
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize("passthrough", [True, False], ids=["native", "lowered"])
@pytest.mark.parametrize(
    ("matrix_fields", "row_id"),
    [
        ({"rows": [{"row_id": "unbound", "seed": 7}]}, "unbound"),
        ({"axes": [{"id": "replica", "path": "seed", "variation": {"values": [7]}}]}, "row-0000"),
    ],
    ids=["explicit-row", "sweep"],
)
def test_unbound_matrix_seed_fails_before_publishing(
    tmp_path: Path, matrix_fields: dict, row_id: str, passthrough: bool
) -> None:
    base = _base(bind_seed=False)
    if passthrough:
        base = base["run"]
    with pytest.raises(RunMatrixError, match=rf"{row_id}.*execution_seed_path"):
        _assemble(tmp_path, base=base, **matrix_fields)
    assert not (tmp_path / "custody").exists()
