"""Lower an admitted Studio continuation to the registered checkpoint fork."""

from pathlib import Path

from feedbax.contracts.base import ParentRef
from feedbax.contracts.checkpoints import (
    CheckpointForkHistoryPolicy,
    CheckpointForkPlan,
    CheckpointForkSourcePreparation,
    CheckpointForkTarget,
    CheckpointForkTransformRecord,
    CheckpointForkTransformStep,
    CheckpointLatestPointer,
)
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD, StudioGraphMethodPayload
from feedbax.contracts.training import (
    TrainingProgramRegistry,
    TrainingRunSpec,
    resolve_training_run_spec,
)
from feedbax.orchestration.schedule_eval import compare_continuation_schedule_projections
from feedbax.training.checkpoint_custody import (
    CheckpointCompatibilityError,
    CheckpointForkPlanBindings,
    authenticate_checkpoint_custody_ref,
    authenticated_run_contract_source_projection,
    checkpoint_slot_names,
    derive_checkpoint_fork_compatibility_projection,
)
from feedbax.training.checkpoint_transform import (
    SEMANTIC_CHECKPOINT_TRANSFORM_IDENTITY,
    SEMANTIC_CHECKPOINT_TRANSFORM_VERSION,
    semantic_checkpoint_transform_implementation_sha256,
)
from feedbax.training.model_checkpoint import MODEL_TREE_MEDIA_TYPE, SemanticModelWriter
from feedbax.training.standard_train_checkpoint import (
    STANDARD_TRAIN_SLOTS,
    STANDARD_TRAIN_STATE_MEDIA_TYPE,
)
from feedbax.training.preparation import ExecutionPreparationResult, _thaw_runtime_value


def plan_studio_checkpoint_fork(
    *,
    source_root: Path,
    pointer: CheckpointLatestPointer,
    source: ParentRef,
    target_root: Path,
    job_id: str,
    spec: TrainingRunSpec,
    method: StudioGraphMethodPayload,
    preparation: ExecutionPreparationResult,
    registry: TrainingProgramRegistry,
    writer: SemanticModelWriter,
) -> tuple[CheckpointForkPlan, CheckpointForkPlanBindings]:
    policy = method.continuation
    assert policy is not None
    if pointer.completed_training_batches != policy.progress.source_completed_batches:
        raise CheckpointCompatibilityError(
            "/checkpoint_initialization/progress/source_completed_batches: "
            "authored boundary differs from the pinned Train checkpoint"
        )
    custody = authenticate_checkpoint_custody_ref(
        source.model_copy(update={"uri": pointer.manifest_relative_path}),
        allowed_root=source_root,
    )
    manifest = custody.manifest
    expected_media = dict.fromkeys(STANDARD_TRAIN_SLOTS, STANDARD_TRAIN_STATE_MEDIA_TYPE)
    expected_media["model"] = MODEL_TREE_MEDIA_TYPE
    actual_media = {slot.slot: slot.media_type for slot in manifest.slots}
    for name in sorted(expected_media.keys() | actual_media.keys()):
        if actual_media.get(name) != expected_media.get(name):
            raise CheckpointCompatibilityError(
                f"/slots/{name}: no typed standard Train state binding for the source slot"
            )
    source_spec, _ = authenticated_run_contract_source_projection(manifest)
    if source_spec.method_ref != STUDIO_GRAPH_METHOD:
        raise CheckpointCompatibilityError("/method_ref: source is outside standard Studio Train")
    source_method = StudioGraphMethodPayload.model_validate(source_spec.method_payload.payload)
    _require_compatible_graphs(
        GraphSpec.model_validate(source_spec.graph.inline),
        GraphSpec.model_validate(spec.graph.inline),
        preserve_optimizer=policy.optimizer == "preserve",
    )
    if policy.optimizer == "preserve":
        old, new = source_method.optimizer, method.optimizer
        if (
            old.type != new.type
            or (old.lr_schedule is None) != (new.lr_schedule is None)
            or {k: v for k, v in old.params.items() if k != "learning_rate"}
            != {k: v for k, v in new.params.items() if k != "learning_rate"}
            or source_spec.training_config.grad_clip != spec.training_config.grad_clip
        ):
            raise CheckpointCompatibilityError(
                "/slots/optimizer: optimizer change requires an explicit reset"
            )
    source_method_runtime = resolve_training_run_spec(source_spec, registry)
    target_method_runtime = resolve_training_run_spec(spec, registry)
    failures, _ = compare_continuation_schedule_projections(
        source_run_spec=source_spec,
        target_run_spec=spec,
        source_manifest=manifest,
        continuation=policy.progress,
        source_resolved_method=source_method_runtime,
        target_resolved_method=target_method_runtime,
    )
    if failures:
        raise CheckpointCompatibilityError(
            "/checkpoint_initialization/schedule: " + ", ".join(failures)
        )
    program = target_method_runtime.effective_phase.phase_program
    values = _thaw_runtime_value(preparation.initial_slots)
    slots = {name: values[name] for name in checkpoint_slot_names(program, manifest.barrier)}
    origin = policy.progress.source_completed_batches if policy.schedule == "restart" else 0
    coordinate = manifest.completed_coordinate.model_copy(
        update={
            "run_id": job_id,
            "schedule_origin_step": origin,
        }
    )
    transformed_slots = ["model", "optimizer"]
    if policy.rng == "reset":
        transformed_slots.append("prng")
    implementation = semantic_checkpoint_transform_implementation_sha256()
    parameters = {
        "added_axes": [],
        "optimizer_policy": policy.optimizer,
        "learning_rate_schedule_policy": policy.schedule,
        "rng_policy": policy.rng,
    }
    step = CheckpointForkTransformStep(
        step_id="studio_stateful_successor",
        stage="target_post",
        records=[
            CheckpointForkTransformRecord(
                slot=name,
                identity=SEMANTIC_CHECKPOINT_TRANSFORM_IDENTITY,
                version=SEMANTIC_CHECKPOINT_TRANSFORM_VERSION,
                implementation_sha256=implementation,
                parameters=parameters,
            )
            for name in transformed_slots
        ],
    )
    plan = CheckpointForkPlan(
        source=CheckpointForkSourcePreparation(
            checkpoint_root_ref="source",
            expected_transaction_id=pointer.transaction_id,
            expected_transaction_root_sha256=pointer.transaction_root_sha256,
        ),
        targets=[
            CheckpointForkTarget(
                target_id="successor",
                checkpoint_root_ref="target",
                run_spec_ref="run",
                slot_template_ref="slots",
                compatibility=derive_checkpoint_fork_compatibility_projection(spec, program, slots),
                history_policy=CheckpointForkHistoryPolicy(
                    mode="prepare_continuation",
                    continuation_request=policy.progress,
                ),
                target_coordinate=coordinate,
                transforms=[step],
            )
        ],
    )
    return plan, CheckpointForkPlanBindings(
        checkpoint_roots={"source": source_root, "target": target_root},
        run_specs={"run": spec},
        slot_templates={"slots": slots},
        semantic_model_writers={"successor": writer},
    )


def _require_compatible_graphs(
    source: GraphSpec,
    target: GraphSpec,
    *,
    preserve_optimizer: bool,
) -> None:
    if source.model_dump(exclude={"nodes"}) != target.model_dump(exclude={"nodes"}):
        raise CheckpointCompatibilityError("/graph: Studio successor requires the same graph")
    for name in sorted(source.nodes.keys() | target.nodes.keys()):
        old, new = source.nodes.get(name), target.nodes.get(name)
        if old == new:
            continue
        path = f"/graph/nodes/{name}"
        if (
            old is None
            or new is None
            or old.model_dump(exclude={"params"})
            != new.model_dump(
                exclude={"params"},
            )
            or (
                {k: v for k, v in old.params.items() if k != "trainable"}
                != {k: v for k, v in new.params.items() if k != "trainable"}
            )
        ):
            raise CheckpointCompatibilityError(f"{path}: incompatible Studio successor node")
        if preserve_optimizer:
            raise CheckpointCompatibilityError(
                f"/slots/optimizer: {path}/params/trainable change requires an explicit reset"
            )
