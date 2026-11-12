"""Governed authored contract for Studio worker training launches."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

from pydantic import Field, field_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.checkpoints import CheckpointLatestPointer
from feedbax.contracts.spec_storage import (
    TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION,
    TrainingRunExecutionCapsule,
    training_spec_sha256,
)
from feedbax.contracts.training import MethodRefSpec, OptimizerSpec, TrainingRunSpec

if TYPE_CHECKING:
    from feedbax.orchestration.assembly import (
        AssemblyContext,
        CompiledExecutionRow,
        CompiledRunSet,
        RowIdentities,
    )


STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID = "feedbax.spec.studio.training_assembly"
STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION = (
    f"{STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID}.v1"
)
STUDIO_TRAINING_COMPILER_ID = "feedbax.studio.worker"
STUDIO_TRAINING_COMPILER_VERSION = "feedbax.studio.worker.v1"
STUDIO_GRAPH_METHOD = MethodRefSpec(package="feedbax", name="studio_graph", version="v1")
STUDIO_GRAPH_METHOD_PAYLOAD_ID = "feedbax.spec.training_method.studio_graph"
STUDIO_GRAPH_METHOD_PAYLOAD_VERSION = f"{STUDIO_GRAPH_METHOD_PAYLOAD_ID}.v1"
STUDIO_INDEPENDENT_KEY_SCHEDULE = "independent_v1"


class StudioGraphMethodPayload(StrictModel):
    """Numerical settings for one checkpointable Studio graph training run."""

    optimizer: OptimizerSpec
    target_total: int = Field(gt=0)
    seed: int = Field(default=0, ge=0, lt=2**32, strict=True)
    key_schedule: Literal["independent_v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    checkpoint_initialization: Literal["initialize_from"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    def require_supported_launch(self, *, resume: bool) -> None:
        # A resume takes its model and carried update key from checkpoint custody.
        # An unstarted legacy seed-zero request would silently change numerics.
        if not resume and self.key_schedule is None and self.seed == 0:
            raise ValueError("Create a new Studio request to start training with seed 0")

    @field_validator("optimizer")
    @classmethod
    def _require_constant_schedule(cls, optimizer: OptimizerSpec) -> OptimizerSpec:
        if optimizer.lr_schedule is not None:
            raise ValueError("checkpointable Studio graph training requires a constant learning rate")
        return optimizer


class PublishedTrainingCheckpoint(StrictModel):
    checkpoint_root: Path
    completed_updates: int = Field(ge=0)
    pointer: CheckpointLatestPointer


class StudioTrainingJobInput(StrictModel):
    spec: TrainingRunSpec
    project_id: str
    request_id: str
    manifest_root: Path
    bootstrap_modules: tuple[str, ...] = ()
    resume: PublishedTrainingCheckpoint | None = None


class StudioTrainingAssemblySpec(StrictModel):
    """Normalized, durable Studio ``/start`` request authored before ASSEMBLE.

    Orchestration identities (``job_id`` and ``run_set_id``) are deliberately
    absent. ASSEMBLE mints those identities and the worker driver adds them only
    when routing the validated executable payload to the worker.
    """

    schema_id: Literal["feedbax.spec.studio.training_assembly"] = (
        STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.studio.training_assembly.v1"] = (
        STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION
    )
    total_batches: int = Field(gt=0)
    snapshot_interval: int = Field(default=100, gt=0)
    training_config: dict[str, Any] | None = None
    training_spec: dict[str, Any] | None = None
    task_spec: dict[str, Any] | None = None
    task_binding_spec: dict[str, Any] | None = None
    graph_spec: dict[str, Any] | None = None

    @field_validator(
        "training_config",
        "training_spec",
        "task_spec",
        "task_binding_spec",
        "graph_spec",
        mode="before",
    )
    @classmethod
    def _copy_mapping(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("Studio worker request spec fields must be JSON objects")
        return dict(value)

    def worker_payload(self) -> dict[str, Any]:
        """Return the governed executable payload, without launch identities."""
        return self.model_dump(mode="json", exclude_none=True)


class StudioTrainingCompiler:
    """Compile one governed Studio request into one generic execution row."""

    def compile(
        self,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: "AssemblyContext",
    ) -> "CompiledRunSet":
        del context
        from feedbax.orchestration.assembly import CompiledExecutionRow, CompiledRunSet
        from feedbax.orchestration.bundle import RowLaunchSpec

        spec = StudioTrainingAssemblySpec.model_validate(authored)
        payload = spec.worker_payload()
        return CompiledRunSet(
            rows=[
                CompiledExecutionRow(
                    row_id=f"{run_set_id}-studio",
                    payload=payload,
                    resolved_semantics=payload,
                    immutable_inputs=[],
                    launch=RowLaunchSpec(
                        command=["worker-http"],
                        collect=[f"../events/{run_set_id}-studio.events.jsonl"],
                        payload_routing={"driver": "worker-http"},
                    ),
                )
            ]
        )


class StudioTrainingIdentityAdapter:
    """Bind a Studio worker payload to the standard training execution capsule."""

    def intent_hash(self, authored: Mapping[str, Any]) -> str:
        spec = StudioTrainingAssemblySpec.model_validate(authored)
        return training_spec_sha256(spec.worker_payload())

    def build_capsule(
        self,
        row: "CompiledExecutionRow",
        *,
        identities: "RowIdentities",
        context: "AssemblyContext",
    ) -> Mapping[str, Any]:
        return TrainingRunExecutionCapsule(
            materializer_commit=context.materializer_commit,
            relevant_schema_versions={
                "studio_training_assembly": STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
                "execution_capsule": TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION,
                "executable_payload": str(row.payload["schema_version"]),
            },
            dependency_lock_digest=context.dependency_lock_digest,
            environment_digest=context.environment_digest,
            input_data_identities=identities.immutable_inputs,
            intent_hash=identities.intent_hash,
            resolved_root_hash=identities.resolved_root_hash,
            execution_hash=identities.execution_hash,
        ).model_dump(mode="json", exclude_none=True)

    def capsule_identities(self, capsule: Mapping[str, Any]) -> "RowIdentities":
        from feedbax.orchestration.assembly import RowIdentities

        validated = TrainingRunExecutionCapsule.model_validate(capsule)
        assert validated.execution_hash is not None
        return RowIdentities(
            intent_hash=validated.intent_hash,
            resolved_root_hash=validated.resolved_root_hash,
            immutable_inputs=validated.input_data_identities,
            execution_hash=validated.execution_hash,
        )


def register_studio_training_compiler(registry: Any) -> None:
    """Register the exact Studio compiler/adapter pair with an assembly registry."""
    registry.register(
        schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        compiler_id=STUDIO_TRAINING_COMPILER_ID,
        compiler_version=STUDIO_TRAINING_COMPILER_VERSION,
        compiler=StudioTrainingCompiler(),
        identity_adapter=StudioTrainingIdentityAdapter(),
    )
