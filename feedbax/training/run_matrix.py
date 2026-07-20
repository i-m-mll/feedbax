"""Materialization and checkpoint forking for training run matrices."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Protocol

import equinox as eqx
import jax.tree as jt
from feedbax.contracts.expressions import evaluate_query
from feedbax.contracts.extraction import load_expression_context, set_dotted_path
from feedbax.contracts.matrix_core import ordered_index_product
from feedbax.contracts.manifest import (
    Provenance,
    SpecPayload,
    TrainingRunAxisCoordinate,
    TrainingRunSetAxes,
    TrainingRunSetManifest,
    TrainingSweepAxis,
    TrainingSweepAxisGroup,
    TrainingSweepAxisVariation,
    TrainingSweepCombinationSpec,
    canonical_json_bytes,
    planned_training_run_manifest_id,
    planned_training_run_set_manifest_id,
    sha256_bytes,
    spec_payload,
)
from feedbax.contracts.run_matrix import (
    AuthoredTrainingRow,
    AuthoredIntentMatrixBaseSpec,
    ForkFromSelectedCheckpoint,
    InlineMatrixBaseSpec,
    RowLowererIdentity,
    RUN_MATRIX_MATERIALIZATION_SCHEMA_ID,
    RUN_MATRIX_MATERIALIZATION_SCHEMA_VERSION,
    ResolvedOutputMatrixBaseSpec,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    TrainingRowLoweringResult,
    TrainingRowPlanningProvenance,
    TrainingRowProvenance,
    TrainingRunMatrixSpec,
    TaskIdentityGate,
    apply_composition_deltas,
    apply_override_patches,
)
from feedbax.contracts.migrations import default_spec_registry, migrate_graph_spec
from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot
from feedbax.contracts.spec_storage import training_spec_canonical_bytes, training_spec_sha256
from feedbax.contracts.checkpoints import (
    CheckpointForkBarrierMapping,
    CheckpointForkPlan,
    CheckpointSegmentLineage,
)
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    LrScheduleSpec,
    OptimizerSpec,
    TrainingMethodRegistry,
    TrainingRunSpec,
)
from feedbax.training.checkpoint_custody import (
    CheckpointCompatibilityError,
    CheckpointForkPlanBindings,
    CheckpointForkTransformRegistry,
    ResumeSlotTransform,
    checkpoint_fork_plan_sha256,
    fork_checkpoint_plan,
    fork_checkpoint_transaction,
    run_contract_binding,
    validate_checkpoint_fork_execution_dependencies,
    _load_latest_checkpoint_transaction,
)
from feedbax.training.optimizers import learning_rate_at_step
from feedbax.training.schedule_clocks import resolve_schedule_window


RUN_MATRIX_FORK_PARITY_SCHEMA_VERSION = "feedbax.run_matrix_fork_parity.v1"


class RunMatrixError(ValueError):
    """Raised when a run matrix cannot be materialized."""


class ForkParityError(RunMatrixError):
    """Raised when forked checkpoint slot parity fails."""


class _LoadedSourceManifest(dict[str, Any]):
    """Ephemeral manifest view carrying verified decoded source slots."""

    def __init__(self, manifest: Mapping[str, Any], slots: Mapping[str, Any]) -> None:
        super().__init__(manifest)
        self.slots = dict(slots)


@dataclass(frozen=True)
class MaterializedMatrixRow:
    """One concrete row materialized from a matrix document."""

    row_id: str
    planned_run_id: str
    spec: TrainingRunSpec | None
    authored_payload: dict[str, Any]
    payload: dict[str, Any]
    provenance: TrainingRowProvenance
    coordinate: TrainingRunAxisCoordinate | None
    overrides: list[Any]
    seed: int | None = None


@dataclass(frozen=True)
class MaterializedRunMatrix:
    """Concrete run set produced from a matrix document."""

    matrix_spec_sha256: str
    run_set_id: str
    base_payload: dict[str, Any]
    rows: list[MaterializedMatrixRow]
    run_set_manifest: TrainingRunSetManifest


class LrContinuationReporter(Protocol):
    """Protocol for method-specific learning-rate continuation reporting."""

    def points(
        self,
        *,
        source_manifest: Mapping[str, Any],
        row_payload: Mapping[str, Any],
        row_spec: TrainingRunSpec,
        declared_mode: str,
    ) -> list[dict[str, Any]]:
        """Return reportable LR points for one target row."""


RowPayloadValidator = Callable[[dict[str, Any], str], TrainingRunSpec | None]


def _validate_matrix_checkpoint_fork_plan(
    spec: TrainingRunMatrixSpec,
    materialized: MaterializedRunMatrix,
    plan: CheckpointForkPlan,
    bindings: CheckpointForkPlanBindings,
    source_checkpoint_root: Path,
) -> None:
    """Reject matrix/plan identity drift before any checkpoint publication."""
    if spec.fork is None:
        raise RunMatrixError("matrix spec has no fork block")
    materialized_ids = [row.row_id for row in materialized.rows]
    if len(materialized_ids) != len(set(materialized_ids)):
        raise RunMatrixError("materialized matrix row identities are not unique")
    if any(target.row_id is None for target in plan.targets):
        raise RunMatrixError(
            "matrix checkpoint fork plan targets require explicit row_id; "
            "target_id fallback is ambiguous"
        )
    plan_ids = [target.row_id for target in plan.targets]
    if len(plan_ids) != len(set(plan_ids)):
        raise RunMatrixError("checkpoint fork plan target row identities are not unique")
    unknown = sorted(set(plan_ids) - set(materialized_ids))
    missing = sorted(set(materialized_ids) - set(plan_ids))
    if unknown:
        raise RunMatrixError(f"checkpoint fork plan contains unknown matrix rows {unknown!r}")
    if missing:
        raise RunMatrixError(f"checkpoint fork plan is missing matrix rows {missing!r}")

    try:
        bound_source = bindings.checkpoint_roots[plan.source.checkpoint_root_ref]
    except KeyError as exc:
        raise RunMatrixError(
            f"checkpoint fork plan source root ref {plan.source.checkpoint_root_ref!r} "
            "has no runtime binding"
        ) from exc
    if Path(bound_source).resolve() != source_checkpoint_root.resolve():
        raise RunMatrixError(
            "matrix source_checkpoint_root does not match checkpoint fork plan source binding; "
            f"matrix={str(source_checkpoint_root.resolve())!r} "
            f"plan={str(Path(bound_source).resolve())!r}"
        )

    rows = {row.row_id: row for row in materialized.rows}
    resolved_roots = [Path(bound_source).resolve()]
    for target in plan.targets:
        assert target.row_id is not None
        row = rows[target.row_id]
        if row.spec is None:
            raise RunMatrixError(
                f"materialized row {target.row_id!r} has no canonical TrainingRunSpec"
            )
        try:
            bound_spec = bindings.run_specs[target.run_spec_ref]
            bound_target_root = bindings.checkpoint_roots[target.checkpoint_root_ref]
            bound_templates = bindings.slot_templates[target.slot_template_ref]
            if target.history_policy.segment_history_template_ref is not None:
                bindings.segment_history_templates[
                    target.history_policy.segment_history_template_ref
                ]
            if target.population_member_ids_ref is not None:
                bindings.population_member_ids[target.population_member_ids_ref]
        except KeyError as exc:
            raise RunMatrixError(
                f"checkpoint fork plan target {target.target_id!r} has an unresolved "
                f"runtime binding {exc.args[0]!r}"
            ) from exc
        if not isinstance(bound_spec, TrainingRunSpec):
            raise RunMatrixError(
                f"checkpoint fork plan target {target.target_id!r} run-spec binding "
                "is not a TrainingRunSpec"
            )
        if not isinstance(bound_templates, Mapping):
            raise RunMatrixError(
                f"checkpoint fork plan target {target.target_id!r} slot-template "
                "binding is not a mapping"
            )
        resolved_roots.append(Path(bound_target_root).resolve())
        bound_program = bound_spec.worker_execution.method_contract.phase_program
        row_program = row.spec.worker_execution.method_contract.phase_program
        bound_hash = run_contract_binding(bound_spec, bound_program).canonical_projection_sha256
        row_hash = run_contract_binding(row.spec, row_program).canonical_projection_sha256
        if bound_hash != row_hash:
            raise RunMatrixError(
                f"checkpoint fork plan target {target.target_id!r} runtime run spec does "
                f"not match materialized row {target.row_id!r}; "
                f"bound={bound_hash!r} materialized={row_hash!r}"
            )
    if len(resolved_roots) != len(set(resolved_roots)):
        raise RunMatrixError(
            "checkpoint fork plan source and target root mappings must be distinct"
        )
    _validate_typed_checkpoint_dependencies(spec, materialized, plan)


def _validate_typed_checkpoint_dependencies(
    spec: TrainingRunMatrixSpec,
    materialized: MaterializedRunMatrix,
    plan: CheckpointForkPlan,
) -> None:
    try:
        validate_checkpoint_fork_execution_dependencies(plan, spec.execution_dependencies, allow_task_identity=True)
    except CheckpointCompatibilityError as exc:
        raise RunMatrixError(str(exc)) from exc
    for gate in (
        item for item in spec.execution_dependencies if isinstance(item, TaskIdentityGate)
    ):
        if gate.identity_kind == "dataset":
            raise RunMatrixError(
                "dataset task-identity gates require an explicit dataset identity resolver"
            )
        actual = {
            training_spec_sha256(
                (
                    row.spec
                    if gate.identity_kind == "training_run_spec"
                    else getattr(row.spec, gate.identity_kind)
                ).model_dump(mode="json", exclude_none=True)
            )
            for row in materialized.rows
            if row.spec is not None
        }
        if actual != {gate.expected_identity_hash}:
            raise RunMatrixError(
                f"checkpoint fork {gate.identity_kind} task-identity gate mismatch; "
                f"declared={gate.expected_identity_hash!r} actual={sorted(actual)!r}"
            )


class TrainingRowLowerer(Protocol):
    """Public typed boundary from authored row intent to execution payload."""

    def __call__(self, row: AuthoredTrainingRow) -> TrainingRowLoweringResult:
        """Lower one axis-patched authored row without mutating the input."""


class StandardLrContinuationReporter:
    """Generic LR reporter for constant and declarative schedule optimizer specs."""

    def __init__(self, registry: TrainingMethodRegistry = DEFAULT_TRAINING_METHOD_REGISTRY) -> None:
        self.registry = registry

    def points(
        self,
        *,
        source_manifest: Mapping[str, Any],
        row_payload: Mapping[str, Any],
        row_spec: TrainingRunSpec,
        declared_mode: str,
    ) -> list[dict[str, Any]]:
        del row_payload
        optimizer = _project_optimizer_spec(row_spec, registry=self.registry)
        if optimizer is None:
            raise RunMatrixError(
                "scheduled LR continuation requires the method descriptor to define "
                "optimizer_spec_projector, or the caller to supply an explicit lr_reporter"
            )
        segment_start = _source_completed_step(source_manifest, row_spec)
        current_step = segment_start
        recorded_optimizer_step = _recorded_optimizer_step(
            row_spec,
            source_manifest,
            registry=self.registry,
        )
        schedule = optimizer.lr_schedule
        if schedule is None:
            if "learning_rate" in optimizer.params:
                return [
                    {
                        "step": current_step,
                        "lr": optimizer.params["learning_rate"],
                        "mode": declared_mode,
                        "recorded_optimizer_step": recorded_optimizer_step,
                    }
                ]
            return [
                {
                    "step": current_step,
                    "lr": None,
                    "mode": declared_mode,
                    "recorded_optimizer_step": recorded_optimizer_step,
                }
            ]
        schedule_spec = LrScheduleSpec.model_validate(schedule)
        lineage = CheckpointSegmentLineage(
            start_batch=segment_start,
            segment_batch_count=0,
            parent_transaction_id=(
                None
                if segment_start == 0
                else str(source_manifest.get("transaction_id") or "source")
            ),
        )
        window = resolve_schedule_window(
            schedule_spec.origin,
            lineage=lineage,
            duration=schedule_spec.total_steps,
            allow_inert=schedule_spec.allow_inert,
        )
        lr = learning_rate_at_step(
            schedule_spec,
            current_step=current_step,
            schedule_origin_step=window.start_batch,
        )
        return [
            {
                "step": current_step,
                "lr": float(lr),
                "mode": declared_mode,
                "recorded_optimizer_step": recorded_optimizer_step,
            }
        ]


def materialize_run_matrix(
    spec: TrainingRunMatrixSpec | Mapping[str, Any],
    *,
    repo_root: Path,
    method_registry: TrainingMethodRegistry = DEFAULT_TRAINING_METHOD_REGISTRY,
) -> MaterializedRunMatrix:
    """Materialize a ``TrainingRunMatrixSpec`` into validated row specs."""
    return _materialize_run_matrix(
        spec,
        repo_root=repo_root,
        row_validator=lambda payload, row_id: _validate_training_payload(
            payload,
            row_id=row_id,
            method_registry=method_registry,
        ),
    )


def materialize_adapted_run_matrix(
    spec: TrainingRunMatrixSpec | Mapping[str, Any],
    *,
    repo_root: Path,
    row_validator: RowPayloadValidator | None = None,
    row_lowerer: TrainingRowLowerer | None = None,
) -> MaterializedRunMatrix:
    """Materialize rows through an optional lowerer and validation-only adapter.

    When supplied, ``row_lowerer`` receives the axis-patched authored payload and
    returns the authoritative execution payload plus its declared implementation
    identity. ``row_validator`` always receives an isolated copy of the payload
    that will execute, so legacy validation-only callbacks cannot mutate custody.
    """
    if row_validator is None and row_lowerer is None:
        raise ValueError("adapted run-matrix materialization requires a validator or lowerer")
    return _materialize_run_matrix(
        spec,
        repo_root=repo_root,
        row_validator=row_validator,
        row_lowerer=row_lowerer,
    )


def _materialize_run_matrix(
    spec: TrainingRunMatrixSpec | Mapping[str, Any],
    *,
    repo_root: Path,
    row_validator: RowPayloadValidator | None,
    row_lowerer: TrainingRowLowerer | None = None,
) -> MaterializedRunMatrix:
    if isinstance(spec, TrainingRunMatrixSpec):
        matrix = spec
    else:
        migrated = default_spec_registry.migrate("TrainingRunMatrixSpec", spec)
        matrix = TrainingRunMatrixSpec.model_validate(migrated.payload)
    base_payload = _resolve_base_payload(matrix, repo_root=repo_root)
    if matrix.derivations:
        ctx = load_expression_context(matrix.sources, repo_root)
        for derivation in matrix.derivations:
            set_dotted_path(
                base_payload,
                derivation.output_path,
                evaluate_query(derivation.query, ctx),
            )

    if matrix.rows:
        rows, axes_block = _materialize_explicit_rows(
            matrix,
            base_payload=base_payload,
            row_validator=row_validator,
            row_lowerer=row_lowerer,
        )
        axes_identity = {
            "mode": "explicit_rows",
            "rows": axes_block.metadata.get("explicit_rows", []),
        }
    else:
        rows, axes_block = _materialize_sweep_rows(
            matrix,
            base_payload=base_payload,
            row_validator=row_validator,
            row_lowerer=row_lowerer,
        )
        axes_identity = axes_block.model_dump(mode="json", exclude_none=True)

    run_set_id = planned_training_run_set_manifest_id(
        graph_spec=_identity_graph_spec(base_payload),
        base_training_spec=_identity_training_spec(base_payload),
        task_spec=_identity_task_spec(base_payload),
        task_binding_spec=_identity_task_binding_spec(base_payload),
        axes=axes_identity,
    )
    run_set_manifest = TrainingRunSetManifest(
        id=run_set_id,
        name=matrix.name,
        run_ids=[row.planned_run_id for row in rows],
        graph_spec=_manifest_graph_payload(base_payload),
        axes=axes_block,
        tags=list(matrix.tags),
        provenance=Provenance(issues=[matrix.issue] if matrix.issue else []),
        metadata={
            "matrix_spec_sha256": _matrix_sha256(matrix),
            "matrix_schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            "explicit_rows": bool(matrix.rows),
            **copy.deepcopy(matrix.metadata),
        },
    )
    return MaterializedRunMatrix(
        matrix_spec_sha256=_matrix_sha256(matrix),
        run_set_id=run_set_id,
        base_payload=base_payload,
        rows=rows,
        run_set_manifest=run_set_manifest,
    )


def write_materialized_matrix(
    materialized: MaterializedRunMatrix,
    out_dir: Path,
    *,
    wrap_key: str | None = None,
) -> dict[str, Any]:
    """Write row payloads and a deterministic materialization manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    row_records: list[dict[str, Any]] = []
    for row in materialized.rows:
        row_payload: dict[str, Any]
        if wrap_key is None:
            row_payload = row.payload
        else:
            row_payload = {wrap_key: row.payload}
        row_bytes = canonical_json_bytes(row_payload)
        row_path = out_dir / f"{row.row_id}.json"
        row_path.write_bytes(row_bytes + b"\n")
        row_records.append(
            {
                "row_id": row.row_id,
                "planned_run_id": row.planned_run_id,
                "payload_path": row_path.name,
                "payload_sha256": sha256_bytes(row_bytes),
                "row_provenance": row.provenance.model_dump(
                    mode="json", exclude_none=True
                ),
            }
        )
    manifest = {
        "schema_id": RUN_MATRIX_MATERIALIZATION_SCHEMA_ID,
        "schema_version": RUN_MATRIX_MATERIALIZATION_SCHEMA_VERSION,
        "matrix_spec_sha256": materialized.matrix_spec_sha256,
        "run_set_id": materialized.run_set_id,
        "rows": row_records,
    }
    (out_dir / "matrix_materialization.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def render_spec_lock_table(
    spec: TrainingRunMatrixSpec,
    materialized: MaterializedRunMatrix,
    *,
    segment_lineages: Mapping[str, CheckpointSegmentLineage] | None = None,
    method_registry: TrainingMethodRegistry = DEFAULT_TRAINING_METHOD_REGISTRY,
) -> str:
    """Render a Markdown spec-lock summary for reviewable launch plans."""
    override_paths = sorted(
        {
            override.path
            for row in materialized.rows
            for override in row.overrides
            if hasattr(override, "path")
        }
    )
    schedule_lines = _resolved_schedule_lines(
        materialized,
        segment_lineages or {},
        method_registry=method_registry,
    )
    header = [
        f"Matrix: {spec.name}",
        f"Issue: {spec.issue or ''}",
        f"Base ref: {getattr(spec.base, 'ref', None) or '<inline>'}",
        "Base content hash: "
        + str(
            getattr(spec.base, "content_hash", None)
            or getattr(spec.base, "resolved_root_hash", "")
        ),
        "Fork source: "
        + next(
            (
                dependency.source_row_id
                for dependency in spec.execution_dependencies
                if isinstance(dependency, ForkFromSelectedCheckpoint)
            ),
            "",
        ),
        (f"LR continuation schedule: {spec.fork.lr_continuation if spec.fork else ''}"),
        f"Row count: {len(materialized.rows)}",
        *schedule_lines,
        "",
    ]
    columns = ["row_id", "seed", "planned_run_id", *override_paths]
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in materialized.rows:
        value_by_path = {
            override.path: _render_value(getattr(override, "value", ""))
            for override in row.overrides
            if hasattr(override, "path")
        }
        cells = [
            row.row_id,
            "" if row.seed is None else str(row.seed),
            row.planned_run_id[:32],
            *(value_by_path.get(path, "") for path in override_paths),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([*header, *rows])


def _resolved_schedule_lines(
    materialized: MaterializedRunMatrix,
    segment_lineages: Mapping[str, CheckpointSegmentLineage],
    *,
    method_registry: TrainingMethodRegistry,
) -> list[str]:
    lines: list[str] = []
    for row in materialized.rows:
        if row.spec is None:
            continue
        optimizer = _project_optimizer_spec(row.spec, registry=method_registry)
        if optimizer is None or optimizer.lr_schedule is None:
            continue
        schedule = optimizer.lr_schedule
        lineage = segment_lineages.get(row.row_id)
        if lineage is None:
            continuation = row.spec.checkpoint_progress.continuation if row.spec else None
            start_batch = 0 if continuation is None else continuation.source_completed_batches
            lineage = CheckpointSegmentLineage(
                start_batch=start_batch,
                segment_batch_count=0,
                parent_transaction_id=None if start_batch == 0 else "prelaunch-source",
            )
        window = resolve_schedule_window(
            schedule.origin,
            lineage=lineage,
            duration=schedule.total_steps,
            allow_inert=schedule.allow_inert,
        )
        end = "ongoing" if window.end_batch is None else f"{window.end_batch:,}"
        lines.append(
            f"{row.row_id} LR schedule: batches {window.start_batch:,} -> {end}"
        )
    return lines


def fork_matrix_checkpoints(
    spec: TrainingRunMatrixSpec,
    materialized: MaterializedRunMatrix,
    *,
    source_checkpoint_root: Path,
    parity_output_path: Path,
    target_checkpoint_roots: Mapping[str, Path] | None = None,
    fork_plan: CheckpointForkPlan | Mapping[str, Any] | None = None,
    fork_plan_bindings: CheckpointForkPlanBindings | None = None,
    fork_transform_registry: CheckpointForkTransformRegistry | None = None,
    target_slot_templates: Mapping[str, Mapping[str, Any]] | None = None,
    row_slot_transforms: Mapping[str, Mapping[str, ResumeSlotTransform]] | None = None,
    row_transform_metadata: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    row_segment_history_templates: Mapping[str, Mapping[str, Any]] | None = None,
    row_target_slot_transforms: Mapping[str, ResumeSlotTransform] | None = None,
    row_target_transform_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    row_target_transformed_slots: Mapping[str, Sequence[str]] | None = None,
    row_target_only_slots: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    row_barrier_mappings: Mapping[
        str,
        CheckpointForkBarrierMapping | Mapping[str, Any],
    ]
    | None = None,
    skip_fork: bool = False,
    lr_reporter: LrContinuationReporter | None = None,
    method_registry: TrainingMethodRegistry = DEFAULT_TRAINING_METHOD_REGISTRY,
    tool_version: str = "feedbax.run_matrix_fork.v1",
    _preflight_lr_points: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Fork a source checkpoint to all matrix rows and write a parity table.

    ``target_slot_templates`` describes the final target topology. Rows that
    both continue and change topology must also provide
    ``row_segment_history_templates`` for the raw pre-topology segment logs.
    ``row_slot_transforms`` run before segment allocation; the explicit target/post
    transform family runs after it and must declare changed source slots and
    newly initialized target-only slots.

    ``row_barrier_mappings`` is the only route to a distinct target checkpoint
    barrier. It is caller-owned and passed unchanged to custody, which verifies
    the actual source barrier and target coordinate before publishing a fork.
    """
    legacy_values = (
        target_checkpoint_roots,
        target_slot_templates,
        row_slot_transforms,
        row_transform_metadata,
        row_segment_history_templates,
        row_target_slot_transforms,
        row_target_transform_metadata,
        row_target_transformed_slots,
        row_target_only_slots,
        row_barrier_mappings,
    )
    if fork_plan is not None:
        if any(value is not None for value in legacy_values) or skip_fork:
            raise RunMatrixError(
                "fork_plan cannot be combined with legacy fork mappings or skip_fork"
            )
        if fork_plan_bindings is None:
            raise RunMatrixError("fork_plan requires fork_plan_bindings")
        resolved_plan = (
            fork_plan
            if isinstance(fork_plan, CheckpointForkPlan)
            else CheckpointForkPlan.model_validate(
                default_spec_registry.migrate("CheckpointForkPlan", fork_plan).payload
            )
        )
        _validate_matrix_checkpoint_fork_plan(
            spec,
            materialized,
            resolved_plan,
            fork_plan_bindings,
            source_checkpoint_root,
        )
        reporter = lr_reporter or StandardLrContinuationReporter(method_registry)
        cached_lr_points = _preflight_lr_continuation_points(
            spec,
            materialized,
            source_manifest=_read_latest_manifest(source_checkpoint_root),
            source_checkpoint_root=source_checkpoint_root,
            reporter=reporter,
        )
        results = fork_checkpoint_plan(
            resolved_plan,
            fork_plan_bindings,
            transform_registry=fork_transform_registry,
            tool_version=tool_version,
        )
        roots: dict[str, Path] = {}
        transformed: dict[str, list[str]] = {}
        target_only: dict[str, Mapping[str, Mapping[str, Any]]] = {}
        transform_meta: dict[str, Mapping[str, Any]] = {}
        common_slots = {
            record.slot
            for step in resolved_plan.source.transforms
            for record in step.records
        }
        plan_sha256 = checkpoint_fork_plan_sha256(resolved_plan)
        for target in resolved_plan.targets:
            row_id = target.row_id or target.target_id
            roots[row_id] = results[target.target_id].root
            target_only[row_id] = {
                slot: declaration
                for step in target.transforms
                for slot, declaration in step.target_only_slots.items()
            }
            changed = common_slots | {
                record.slot for step in target.transforms for record in step.records
            }
            transformed[row_id] = sorted(changed - set(target_only[row_id]))
            if changed:
                transform_meta[row_id] = {
                    "identity": "feedbax.training_checkpoint.plan_materialization.v1",
                    "parameters": {"plan_sha256": plan_sha256},
                }
        return fork_matrix_checkpoints(
            spec,
            materialized,
            source_checkpoint_root=source_checkpoint_root,
            target_checkpoint_roots=roots,
            parity_output_path=parity_output_path,
            row_target_transform_metadata=transform_meta or None,
            row_target_transformed_slots=transformed or None,
            row_target_only_slots=target_only or None,
            skip_fork=True,
            lr_reporter=lr_reporter,
            method_registry=method_registry,
            tool_version=tool_version,
            _preflight_lr_points=cached_lr_points,
        )
    if fork_plan_bindings is not None or fork_transform_registry is not None:
        raise RunMatrixError("fork plan bindings/registry require fork_plan")
    if target_checkpoint_roots is None:
        raise RunMatrixError("target_checkpoint_roots is required for legacy fork execution")
    if spec.fork is None:
        raise RunMatrixError("matrix spec has no fork block")
    row_ids = {row.row_id for row in materialized.rows}
    unexpected_targets = sorted(set(target_checkpoint_roots) - row_ids)
    if unexpected_targets:
        raise RunMatrixError(f"target checkpoint roots contain unknown rows {unexpected_targets!r}")
    unexpected_templates = sorted(set(target_slot_templates or {}) - row_ids)
    if unexpected_templates:
        raise RunMatrixError(f"target slot templates contain unknown rows {unexpected_templates!r}")
    unexpected_transforms = sorted(set(row_slot_transforms or {}) - row_ids)
    if unexpected_transforms:
        raise RunMatrixError(f"row slot transforms contain unknown rows {unexpected_transforms!r}")
    unexpected_transform_metadata = sorted(set(row_transform_metadata or {}) - row_ids)
    if unexpected_transform_metadata:
        raise RunMatrixError(
            f"row transform metadata contains unknown rows {unexpected_transform_metadata!r}"
        )
    for label, values in (
        ("row segment history templates", row_segment_history_templates),
        ("row target slot transforms", row_target_slot_transforms),
        ("row target transform metadata", row_target_transform_metadata),
        ("row target transformed slots", row_target_transformed_slots),
        ("row target-only slots", row_target_only_slots),
        ("row barrier mappings", row_barrier_mappings),
    ):
        unexpected = sorted(set(values or {}) - row_ids)
        if unexpected:
            raise RunMatrixError(f"{label} contain unknown rows {unexpected!r}")
    reporter = lr_reporter or StandardLrContinuationReporter(method_registry)
    cached_lr_points = (
        {
            row_id: [dict(point) for point in points]
            for row_id, points in _preflight_lr_points.items()
        }
        if _preflight_lr_points is not None
        else _preflight_lr_continuation_points(
            spec,
            materialized,
            source_manifest=_read_latest_manifest(source_checkpoint_root),
            source_checkpoint_root=source_checkpoint_root,
            reporter=reporter,
        )
    )
    parity_rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for row in materialized.rows:
        if row.spec is None:
            raise RunMatrixError(f"row {row.row_id!r} does not contain a canonical TrainingRunSpec")
        target_root = target_checkpoint_roots.get(row.row_id)
        if target_root is None:
            raise RunMatrixError(f"missing target checkpoint root for row {row.row_id!r}")
        if skip_fork:
            source_manifest, target_manifest = _latest_manifest_pair(
                source_checkpoint_root,
                target_root,
            )
            transaction_id = target_manifest.get("transaction_id")
        else:
            continuation = row.spec.checkpoint_progress.continuation
            expected_slots = (target_slot_templates or {}).get(row.row_id)
            if continuation is not None and expected_slots is None:
                raise RunMatrixError(
                    "row declares checkpoint continuation but has no target slot template; "
                    f"row={row.row_id!r} contract=checkpoint_progress.continuation"
                )
            target_transform = (row_target_slot_transforms or {}).get(row.row_id)
            continuation_templates = (row_segment_history_templates or {}).get(row.row_id)
            if (
                continuation is not None
                and target_transform is not None
                and continuation_templates is None
            ):
                raise RunMatrixError(
                    "topology-changing continuation row has no raw segment history slot "
                    f"template; row={row.row_id!r} "
                    "contract=row_segment_history_templates"
                )
            result = fork_checkpoint_transaction(
                source_checkpoint_root,
                target_root,
                target_run_spec=row.spec,
                target_phase_program=row.spec.worker_execution.method_contract.phase_program,
                expected_slots=expected_slots,
                slot_transforms=(row_slot_transforms or {}).get(row.row_id),
                transform_metadata=(row_transform_metadata or {}).get(row.row_id),
                segment_history_templates=continuation_templates,
                target_slot_transform=target_transform,
                target_transform_metadata=(row_target_transform_metadata or {}).get(row.row_id),
                target_transformed_slots=(row_target_transformed_slots or {}).get(row.row_id),
                target_only_slots=(row_target_only_slots or {}).get(row.row_id),
                barrier_mapping=(row_barrier_mappings or {}).get(row.row_id),
                continuation_request=continuation,
                tool_version=tool_version,
                metadata={
                    "matrix_spec_sha256": materialized.matrix_spec_sha256,
                    "matrix_row_id": row.row_id,
                    "planned_run_id": row.planned_run_id,
                },
            )
            # Fork provenance records source identity and slot digests, while
            # the complete source manifest remains the authority for
            # continuation arithmetic. In particular, program_step is not a
            # training-batch total and must never become an LR fallback.
            source_manifest = _read_latest_manifest(source_checkpoint_root)
            target_manifest = result.manifest.model_dump(mode="json", exclude_none=True)
            transaction_id = result.manifest.transaction_id
        slot_rows, row_mismatches = _parity_rows(
            row_id=row.row_id,
            planned_run_id=row.planned_run_id,
            source_manifest=source_manifest,
            target_manifest=target_manifest,
            expected_slots=spec.fork.expected_slots,
            source_transformed_slots=tuple(
                (row_slot_transforms or {}).get(row.row_id, {})
            ),
            source_transform_metadata={
                slot: dict(metadata)
                for slot, metadata in (row_transform_metadata or {}).get(row.row_id, {}).items()
            },
            target_transform_metadata=(row_target_transform_metadata or {}).get(row.row_id),
            target_transformed_slots=(row_target_transformed_slots or {}).get(row.row_id, ()),
            target_only_slots=(row_target_only_slots or {}).get(row.row_id, {}),
        )
        parity_rows.extend(slot_rows)
        mismatches.extend(row_mismatches)
        parity_rows.extend(
            {
                "row_id": row.row_id,
                "planned_run_id": row.planned_run_id,
                "kind": "lr_continuation",
                "transaction_id": transaction_id,
                "source_row_id": next(
                    (
                        dependency.source_row_id
                        for dependency in spec.execution_dependencies
                        if isinstance(dependency, ForkFromSelectedCheckpoint)
                    ),
                    None,
                ),
                "target_run_id": row.planned_run_id,
                "source_transaction_id": source_manifest.get("transaction_id"),
                "target_transaction_id": transaction_id,
                "source_completed_batches": source_manifest.get("completed_training_batches"),
                "target_completed_batches": target_manifest.get("completed_training_batches"),
                "source_segment_lineage": source_manifest.get("segment_lineage"),
                "target_segment_lineage": target_manifest.get("segment_lineage"),
                "declared_mode": spec.fork.lr_continuation,
                **point,
            }
            for point in cached_lr_points[row.row_id]
        )
    table = {
        "schema_version": RUN_MATRIX_FORK_PARITY_SCHEMA_VERSION,
        "matrix_spec_sha256": materialized.matrix_spec_sha256,
        "ok": not mismatches,
        "rows": parity_rows,
    }
    parity_output_path.parent.mkdir(parents=True, exist_ok=True)
    parity_output_path.write_bytes(canonical_json_bytes(table) + b"\n")
    if mismatches and spec.fork.parity == "require":
        raise ForkParityError("; ".join(mismatches))
    return table


def expand_sweep_coordinates(
    axes: list[TrainingSweepAxis],
    combination: TrainingSweepCombinationSpec,
) -> list[dict[str, int]]:
    """Expand typed sweep axes into coordinate value indices."""
    _validate_group_axes(axes, combination)
    return _expand_coordinates(axes, combination)


def expected_ordered_matrix_row_ids(matrix: TrainingRunMatrixSpec) -> tuple[str, ...]:
    """Return the complete authored or expanded row identity in canonical order."""
    if matrix.rows:
        return tuple(row.row_id for row in matrix.rows)
    coordinates = expand_sweep_coordinates(matrix.axes, matrix.combination)
    return tuple(f"row-{index:04d}" for index in range(len(coordinates)))


def variation_values(variation: TrainingSweepAxisVariation) -> list[Any]:
    """Return the authored concrete values for a sweep variation."""
    if variation.kind == "explicit":
        return list(variation.values)
    if variation.kind == "linspace":
        assert variation.min is not None and variation.max is not None and variation.n is not None
        if variation.n == 1:
            return [variation.min]
        step = (variation.max - variation.min) / (variation.n - 1)
        return [variation.min + step * index for index in range(variation.n)]
    if variation.kind == "logspace":
        assert variation.min is not None and variation.max is not None and variation.n is not None
        if variation.n == 1:
            return [variation.min]
        start = math.log10(variation.min)
        stop = math.log10(variation.max)
        step = (stop - start) / (variation.n - 1)
        return [10 ** (start + step * index) for index in range(variation.n)]
    assert variation.sampler is not None and variation.n is not None
    rng = random.Random(variation.seed)
    if variation.sampler == "uniform":
        low = float(variation.params.get("min", 0.0))
        high = float(variation.params.get("max", 1.0))
        return [rng.uniform(low, high) for _ in range(variation.n)]
    if variation.sampler == "log_uniform":
        low = float(variation.params.get("min"))
        high = float(variation.params.get("max"))
        if low <= 0 or high <= 0:
            raise RunMatrixError("log_uniform sampler requires positive min and max")
        return [10 ** rng.uniform(math.log10(low), math.log10(high)) for _ in range(variation.n)]
    if variation.sampler == "normal":
        mean = float(variation.params.get("mean", 0.0))
        std = float(variation.params.get("std", 1.0))
        return [rng.gauss(mean, std) for _ in range(variation.n)]
    raise RunMatrixError(f"unsupported sweep sampler {variation.sampler!r}")


def _resolve_base_payload(spec: TrainingRunMatrixSpec, *, repo_root: Path) -> dict[str, Any]:
    resolved, _ = resolve_base_payload_with_attribution(spec, repo_root=repo_root)
    return resolved


def resolve_base_payload_with_attribution(
    spec: TrainingRunMatrixSpec, *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve composed intent and retain the last-writing layer for each patched path."""
    resolved, attribution, _ = _resolve_composed_base(
        spec, repo_root=repo_root, resolving=set()
    )
    graph_source = resolved.get("graph")
    if isinstance(graph_source, Mapping) and isinstance(graph_source.get("inline"), Mapping):
        migrated_graph = migrate_graph_spec(graph_source["inline"], path="graph.inline")
        resolved["graph"] = {**graph_source, "inline": migrated_graph.payload}
    graph_spec = resolved.get("graph_spec")
    if isinstance(graph_spec, Mapping):
        resolved["graph_spec"] = migrate_graph_spec(graph_spec, path="graph_spec").payload
    return resolved, attribution


def _resolve_composed_base(
    spec: TrainingRunMatrixSpec,
    *,
    repo_root: Path,
    resolving: set[Path],
) -> tuple[dict[str, Any], dict[str, str], set[str]]:
    if isinstance(spec.base, InlineMatrixBaseSpec):
        document = copy.deepcopy(spec.base.inline)
        payload_path = None
    elif isinstance(spec.base, AuthoredIntentMatrixBaseSpec):
        path = repo_root / spec.base.ref
        canonical_path = path.resolve()
        if canonical_path in resolving:
            raise RunMatrixError(f"/base/ref authored composition cycle: {spec.base.ref}")
        data = path.read_bytes()
        document = json.loads(data.decode("utf-8"))
        if spec.base.pin_algorithm == "legacy_raw_sha256":
            actual_hash = sha256_bytes(data)
            mismatch_name = "legacy raw sha256"
        else:
            actual_hash = training_spec_sha256(document)
            mismatch_name = "canonical content hash"
        if actual_hash != spec.base.content_hash:
            raise RunMatrixError(f"/base/ref {mismatch_name} mismatch: {spec.base.ref}")
        if isinstance(document, Mapping) and _is_authored_matrix_document(document):
            migrated = default_spec_registry.migrate("TrainingRunMatrixSpec", document).payload
            parent = TrainingRunMatrixSpec.model_validate(migrated)
            resolving.add(canonical_path)
            try:
                parent_payload, attribution, written = _resolve_composed_base(
                    parent, repo_root=repo_root, resolving=resolving
                )
            finally:
                resolving.remove(canonical_path)
            resolved, local_attribution, written = apply_composition_deltas(
                parent_payload,
                spec.deltas,
                ancestor_written_paths=written,
            )
            attribution.update(local_attribution)
            return resolved, attribution, written
        else:
            payload_path = spec.base.payload_path
    else:
        assert isinstance(spec.base, ResolvedOutputMatrixBaseSpec)
        path = repo_root / spec.base.ref
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot.get("root_hash") != spec.base.resolved_root_hash:
            raise RunMatrixError(f"/base/ref resolved root hash mismatch: {spec.base.ref}")
        document = decode_resolved_snapshot(snapshot)
        payload_path = spec.base.payload_path
    payload = _get_dotted(document, payload_path)
    if not isinstance(payload, dict):
        raise RunMatrixError("/base payload must resolve to an object")
    resolved = copy.deepcopy(payload)
    resolved, attribution, written = apply_composition_deltas(resolved, spec.deltas)
    return resolved, attribution, written


def _is_authored_matrix_document(document: Mapping[str, Any]) -> bool:
    schema_id = document.get("schema_id")
    schema_version = document.get("schema_version")
    return schema_id == "feedbax.spec.training_run_matrix" or (
        isinstance(schema_version, str)
        and schema_version.startswith("feedbax.spec.training_run_matrix.v")
    )


def _materialize_explicit_rows(
    matrix: TrainingRunMatrixSpec,
    *,
    base_payload: dict[str, Any],
    row_validator: RowPayloadValidator | None,
    row_lowerer: TrainingRowLowerer | None,
) -> tuple[list[MaterializedMatrixRow], TrainingRunSetAxes]:
    rows: list[MaterializedMatrixRow] = []
    explicit_records: list[dict[str, Any]] = []
    coordinates: list[TrainingRunAxisCoordinate] = []
    for index, row in enumerate(matrix.rows):
        authored_payload = apply_override_patches(base_payload, row.overrides)
        axis_coordinates = {
            "row_id": row.row_id,
            "overrides": [
                patch.model_dump(mode="json", exclude_none=True) for patch in row.overrides
            ],
        }
        (
            payload,
            spec,
            lowerer_identities,
            authored_payload_hash,
            lowered_execution_payload_hash,
        ) = _lower_authored_row(
            row_id=row.row_id,
            row_index=index,
            authored_payload=authored_payload,
            seed=row.seed,
            axis_coordinates=axis_coordinates,
            overrides=axis_coordinates["overrides"],
            row_validator=row_validator,
            row_lowerer=row_lowerer,
        )
        run_id = _planned_run_id(
            payload,
            seed=row.seed,
            axis_coordinates=axis_coordinates,
            authored_payload_hash=authored_payload_hash,
            lowered_execution_payload_hash=lowered_execution_payload_hash,
            lowerer_identities=lowerer_identities,
        )
        coordinate = TrainingRunAxisCoordinate(
            run_id=run_id,
            index=index,
            value_indices={"row": index},
            values=axis_coordinates,
            label=row.label or row.row_id,
        )
        coordinates.append(coordinate)
        explicit_records.append(
            {
                "row_id": row.row_id,
                "label": row.label,
                "seed": row.seed,
                "overrides": axis_coordinates["overrides"],
                "notes": row.notes,
                "metadata": row.metadata,
            }
        )
        rows.append(
            MaterializedMatrixRow(
                row_id=row.row_id,
                planned_run_id=run_id,
                spec=spec,
                authored_payload=authored_payload,
                payload=payload,
                provenance=TrainingRowProvenance(
                    row_id=row.row_id,
                    row_index=index,
                    planned_run_id=run_id,
                    authored_payload_hash=authored_payload_hash,
                    lowered_execution_payload_hash=lowered_execution_payload_hash,
                    seed=row.seed,
                    axis_coordinates=coordinate.model_dump(mode="json", exclude_none=True),
                    overrides=axis_coordinates["overrides"],
                    lowerer_identities=lowerer_identities,
                ),
                coordinate=None,
                overrides=list(row.overrides),
                seed=row.seed,
            )
        )
    axes = TrainingRunSetAxes(
        runs=coordinates,
        metadata={
            "mode": "explicit_rows",
            "row_count": len(rows),
            "explicit_rows": explicit_records,
        },
    )
    return rows, axes


def _materialize_sweep_rows(
    matrix: TrainingRunMatrixSpec,
    *,
    base_payload: dict[str, Any],
    row_validator: RowPayloadValidator | None,
    row_lowerer: TrainingRowLowerer | None,
) -> tuple[list[MaterializedMatrixRow], TrainingRunSetAxes]:
    axes_with_values = [
        axis.model_copy(update={"values": variation_values(axis.variation)}) for axis in matrix.axes
    ]
    axis_by_id = {axis.id: axis for axis in axes_with_values}
    indexed_coordinates = expand_sweep_coordinates(axes_with_values, matrix.combination)
    expected_row_ids = expected_ordered_matrix_row_ids(matrix)
    run_set_axes = TrainingRunSetAxes(
        axes=axes_with_values,
        combination=matrix.combination,
        metadata={"axis_count": len(axes_with_values), "run_count": len(indexed_coordinates)},
    )
    rows: list[MaterializedMatrixRow] = []
    for index, value_indices in enumerate(indexed_coordinates):
        values = {
            axis_id: axis_by_id[axis_id].values[value_index]
            for axis_id, value_index in value_indices.items()
        }
        authored_payload = copy.deepcopy(base_payload)
        seed = None
        patches = []
        for axis_id, value in values.items():
            axis = axis_by_id[axis_id]
            if axis.path in {"seed", "master_prng_key", "prng_key"}:
                seed = value
                studio_training_spec = authored_payload.get("training_spec")
                if isinstance(studio_training_spec, dict):
                    studio_training_spec["seed"] = value
            else:
                patch = _patch_object({"path": axis.path, "value": value, "op": "replace"})
                patches.append(patch)
                authored_payload = apply_override_patches(authored_payload, [patch])
        row_id = expected_row_ids[index]
        override_payloads = [
            patch.model_dump(mode="json", exclude_none=True) for patch in patches
        ]
        (
            payload,
            spec,
            lowerer_identities,
            authored_payload_hash,
            lowered_execution_payload_hash,
        ) = _lower_authored_row(
            row_id=row_id,
            row_index=index,
            authored_payload=authored_payload,
            seed=seed if isinstance(seed, int) else None,
            axis_coordinates=values,
            overrides=override_payloads,
            row_validator=row_validator,
            row_lowerer=row_lowerer,
        )
        run_id = _planned_run_id(
            payload,
            seed=seed,
            axis_coordinates=values,
            authored_payload_hash=authored_payload_hash,
            lowered_execution_payload_hash=lowered_execution_payload_hash,
            lowerer_identities=lowerer_identities,
        )
        coordinate = TrainingRunAxisCoordinate(
            run_id=run_id,
            index=index,
            value_indices=value_indices,
            values=values,
            label=_coordinate_label(axis_by_id, values),
        )
        rows.append(
            MaterializedMatrixRow(
                row_id=row_id,
                planned_run_id=run_id,
                spec=spec,
                authored_payload=authored_payload,
                payload=payload,
                provenance=TrainingRowProvenance(
                    row_id=row_id,
                    row_index=index,
                    planned_run_id=run_id,
                    authored_payload_hash=authored_payload_hash,
                    lowered_execution_payload_hash=lowered_execution_payload_hash,
                    seed=seed if isinstance(seed, int) else None,
                    axis_coordinates=coordinate.model_dump(mode="json", exclude_none=True),
                    overrides=override_payloads,
                    lowerer_identities=lowerer_identities,
                ),
                coordinate=coordinate,
                overrides=patches,
                seed=seed if isinstance(seed, int) else None,
            )
        )
    run_set_axes.runs = [row.coordinate for row in rows if row.coordinate is not None]
    return rows, run_set_axes


def _lower_authored_row(
    *,
    row_id: str,
    row_index: int,
    authored_payload: dict[str, Any],
    seed: int | None,
    axis_coordinates: dict[str, Any],
    overrides: list[dict[str, Any]],
    row_validator: RowPayloadValidator | None,
    row_lowerer: TrainingRowLowerer | None,
) -> tuple[
    dict[str, Any],
    TrainingRunSpec | None,
    list[RowLowererIdentity],
    str,
    str,
]:
    """Lower and validate one row while isolating callback-owned mutations."""
    authored_copy = copy.deepcopy(authored_payload)
    authored_payload_hash = training_spec_sha256(authored_copy)
    lowerer_identities: list[RowLowererIdentity] = []
    if row_lowerer is None:
        execution_payload = copy.deepcopy(authored_copy)
    else:
        authored_row = AuthoredTrainingRow(
            row_id=row_id,
            row_index=row_index,
            payload=authored_copy,
            payload_hash=authored_payload_hash,
            seed=seed,
            axis_coordinates=copy.deepcopy(axis_coordinates),
            overrides=copy.deepcopy(overrides),
        )
        raw_result = row_lowerer(authored_row.model_copy(deep=True))
        result = (
            raw_result
            if isinstance(raw_result, TrainingRowLoweringResult)
            else TrainingRowLoweringResult.model_validate(raw_result)
        )
        execution_payload = copy.deepcopy(result.execution_payload)
        lowerer_identities = list(result.lowerer_identities)
    spec = (
        None
        if row_validator is None
        else row_validator(copy.deepcopy(execution_payload), row_id)
    )
    lowered_execution_payload_hash = training_spec_sha256(execution_payload)
    return (
        execution_payload,
        spec,
        lowerer_identities,
        authored_payload_hash,
        lowered_execution_payload_hash,
    )


def _validate_training_payload(
    payload: dict[str, Any],
    *,
    row_id: str,
    method_registry: TrainingMethodRegistry,
) -> TrainingRunSpec:
    try:
        spec = TrainingRunSpec.model_validate(payload)
        method_registry.validate_payload(
            spec.method_ref,
            spec.method_payload,
            path="/method_payload",
        )
        return spec
    except Exception as exc:
        raise RunMatrixError(f"row {row_id!r} TrainingRunSpec validation failed: {exc}") from exc


def _planned_run_id(
    payload: Mapping[str, Any],
    *,
    seed: Any | None,
    axis_coordinates: dict[str, Any],
    authored_payload_hash: str,
    lowered_execution_payload_hash: str,
    lowerer_identities: list[RowLowererIdentity],
) -> str:
    custody_payload = json.loads(training_spec_canonical_bytes(payload))
    custody_axis_coordinates = json.loads(
        training_spec_canonical_bytes(axis_coordinates)
    )
    provenance_identity = TrainingRowPlanningProvenance(
        authored_payload_hash=authored_payload_hash,
        lowered_execution_payload_hash=lowered_execution_payload_hash,
        lowerer_identities=lowerer_identities,
    )
    return planned_training_run_manifest_id(
        graph_spec=_identity_graph_spec(custody_payload),
        training_spec=_identity_training_spec(custody_payload),
        task_spec=_identity_task_spec(custody_payload),
        task_binding_spec=_identity_task_binding_spec(custody_payload),
        seed=seed,
        axis_coordinates=custody_axis_coordinates,
        row_provenance_identity=provenance_identity.model_dump(
            mode="json", exclude_none=True
        ),
    )


def _identity_graph_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    studio_graph = payload.get("graph_spec")
    if isinstance(studio_graph, dict):
        return copy.deepcopy(studio_graph)
    graph = payload.get("graph")
    return copy.deepcopy(graph) if isinstance(graph, dict) else {}


def _identity_training_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    studio_training = payload.get("training_spec")
    if isinstance(studio_training, dict):
        return copy.deepcopy(studio_training)
    return {
        key: copy.deepcopy(payload[key])
        for key in (
            "training_config",
            "objective",
            "risk_aggregation",
            "method_ref",
            "method_payload",
            "method_extensions",
            "worker_execution",
            "artifacts",
            "checkpoint_progress",
            "on_nan",
        )
        if key in payload
    }


def _identity_task_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    studio_task = payload.get("task_spec")
    if isinstance(studio_task, dict):
        return copy.deepcopy(studio_task)
    task = payload.get("task")
    return copy.deepcopy(task) if isinstance(task, dict) else {}


def _identity_task_binding_spec(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    task_binding = payload.get("task_binding_spec")
    return copy.deepcopy(task_binding) if isinstance(task_binding, dict) else None


def _manifest_graph_payload(payload: Mapping[str, Any]) -> SpecPayload | None:
    studio_graph = payload.get("graph_spec")
    if isinstance(studio_graph, dict):
        return spec_payload("GraphSpec", studio_graph)
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        return None
    inline = graph.get("inline")
    if isinstance(inline, dict):
        return spec_payload("GraphSpec", inline)
    return SpecPayload(kind="GraphSpec", inline=copy.deepcopy(graph))


def _matrix_sha256(spec: TrainingRunMatrixSpec) -> str:
    return sha256_bytes(canonical_json_bytes(spec.model_dump(mode="json", exclude_none=True)))


def _validate_group_axes(
    axes: list[TrainingSweepAxis],
    combination: TrainingSweepCombinationSpec,
) -> None:
    axis_ids = {axis.id for axis in axes}
    used: set[str] = set()
    for group in combination.groups:
        unknown = [axis_id for axis_id in group.axes if axis_id not in axis_ids]
        if unknown:
            raise RunMatrixError(f"sweep group {group.id!r} references unknown axes {unknown!r}")
        overlap = used.intersection(group.axes)
        if overlap:
            raise RunMatrixError(f"sweep axes {sorted(overlap)!r} appear in more than one group")
        used.update(group.axes)
    if combination.groups:
        missing = sorted(axis_ids - used)
        if missing:
            raise RunMatrixError(
                f"sweep matrix groups must cover every declared axis; missing axes {missing!r}"
            )


def _expand_coordinates(
    axes: list[TrainingSweepAxis],
    combination: TrainingSweepCombinationSpec,
) -> list[dict[str, int]]:
    axis_lengths = {axis.id: len(variation_values(axis.variation)) for axis in axes}
    if combination.mode == "manual":
        if not combination.manual_coordinates:
            raise RunMatrixError("manual sweep matrix requires manual_coordinates")
        return [
            _validate_manual_coordinate(raw, axis_lengths, index)
            for index, raw in enumerate(combination.manual_coordinates)
        ]
    groups = combination.groups or [
        TrainingSweepAxisGroup(
            id="all",
            axes=[axis.id for axis in axes],
            mode="zip" if combination.mode == "zip" else "cross",
        )
    ]
    out: list[dict[str, int]] = []
    for parts in _product(*[_expand_group(group, axis_lengths) for group in groups]):
        coordinate: dict[str, int] = {}
        for part in parts:
            coordinate.update(part)
        out.append(coordinate)
    return out


def _expand_group(
    group: TrainingSweepAxisGroup,
    axis_lengths: Mapping[str, int],
) -> list[dict[str, int]]:
    if group.mode == "zip":
        lengths = {axis_lengths[axis_id] for axis_id in group.axes}
        if len(lengths) != 1:
            raise RunMatrixError(
                f"zip sweep group {group.id!r} has mismatched lengths {sorted(lengths)!r}"
            )
        return [{axis_id: index for axis_id in group.axes} for index in range(next(iter(lengths)))]
    return ordered_index_product(
        [(axis_id, axis_lengths[axis_id]) for axis_id in group.axes]
    )


def _validate_manual_coordinate(
    raw: Mapping[str, Any],
    axis_lengths: Mapping[str, int],
    index: int,
) -> dict[str, int]:
    coordinate: dict[str, int] = {}
    for axis_id, length in axis_lengths.items():
        value = raw.get(axis_id)
        if not isinstance(value, int):
            raise RunMatrixError(
                f"manual sweep coordinate {index} must include integer index for {axis_id!r}"
            )
        if value < 0 or value >= length:
            raise RunMatrixError(
                f"manual sweep coordinate {index} index for {axis_id!r} is out of range"
            )
        coordinate[axis_id] = value
    return coordinate


def _product(*iterables: Any) -> Any:
    from itertools import product

    return product(*iterables)


def _coordinate_label(
    axis_by_id: Mapping[str, TrainingSweepAxis], values: Mapping[str, Any]
) -> str:
    return ", ".join(
        f"{axis_by_id[axis_id].label or axis_by_id[axis_id].path}={values[axis_id]!r}"
        for axis_id in sorted(values)
    )


def _get_dotted(document: Any, path: str | None) -> Any:
    current = document
    if path is None:
        return current
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        raise RunMatrixError(f"payload_path segment is missing: {path!r}")
    return current


def _patch_object(patch: Mapping[str, Any]) -> Any:
    from feedbax.contracts.manifest import OverridePatch

    return OverridePatch.model_validate(patch)


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _latest_manifest_pair(
    source_root: Path, target_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _read_latest_manifest(source_root), _read_latest_manifest(target_root)


def _read_latest_manifest(root: Path) -> dict[str, Any]:
    latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    manifest_path = root / latest["manifest_relative_path"]
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _slot_digest_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    integrity = manifest.get("content_integrity_digest")
    if isinstance(integrity, Mapping):
        slots = integrity.get("slots", [])
    else:
        slots = manifest.get("slot_content_digests", [])
    out: dict[str, str] = {}
    if isinstance(slots, list):
        for slot in slots:
            if isinstance(slot, Mapping) and "slot" in slot:
                digest = slot.get("slot_root_sha256") or slot.get("blob_sha256")
                if digest is not None:
                    out[str(slot["slot"])] = str(digest)
    return out


def _slot_blob_digest_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    slots = manifest.get("slots", [])
    if not isinstance(slots, list):
        return {}
    return {
        str(slot["slot"]): str(slot["sha256"])
        for slot in slots
        if isinstance(slot, Mapping) and "slot" in slot and "sha256" in slot
    }


def _fork_slot_provenance(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    fork = manifest.get("fork_provenance")
    if not isinstance(fork, Mapping):
        return {}
    slots = fork.get("slots", [])
    if not isinstance(slots, list):
        return {}
    return {
        str(slot["slot"]): slot for slot in slots if isinstance(slot, Mapping) and "slot" in slot
    }


def _transform_provenance_mismatches(
    *,
    row_id: str,
    slot: str,
    provenance: Mapping[str, Any] | None,
    expected_metadata: Mapping[str, Any] | None,
    expected_target_only_declaration: Mapping[str, Any] | None = None,
) -> list[str]:
    context = f"row={row_id} slot={slot}"
    if provenance is None:
        return [f"{context} missing fork provenance"]
    transform = provenance.get("transform")
    if not isinstance(transform, Mapping):
        return [f"{context} missing transform provenance"]
    mismatches: list[str] = []
    if transform.get("slot") != slot:
        mismatches.append(f"{context} transform provenance names wrong slot")
    if expected_metadata is not None:
        identity = expected_metadata.get("identity")
        if identity is not None and transform.get("identity") != identity:
            mismatches.append(f"{context} transform identity mismatch")
        parameters = expected_metadata.get("parameters", {})
        if transform.get("parameters", {}) != parameters:
            mismatches.append(f"{context} transform parameters mismatch")
    metadata = transform.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    if expected_target_only_declaration is not None:
        if metadata.get("target_only_declaration") != dict(expected_target_only_declaration):
            mismatches.append(f"{context} target-only declaration mismatch")
        stages = metadata.get("stages")
        if not isinstance(stages, list) or not stages:
            mismatches.append(f"{context} missing ordered transform stages")
        elif not isinstance(stages[-1], Mapping) or stages[-1].get("stage") != "target_post":
            mismatches.append(f"{context} target-only provenance is not target_post")
    return mismatches


def _parity_rows(
    *,
    row_id: str,
    planned_run_id: str,
    source_manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    expected_slots: list[str],
    source_transformed_slots: Sequence[str] = (),
    source_transform_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    target_transform_metadata: Mapping[str, Any] | None = None,
    target_transformed_slots: Sequence[str] = (),
    target_only_slots: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    source_digests = _slot_digest_map(source_manifest)
    target_digests = _slot_digest_map(target_manifest)
    comparable_slots = set(expected_slots) if expected_slots else set(source_digests)
    declared_target_only = dict(target_only_slots or {})
    declared_transformed = set(source_transformed_slots) | set(target_transformed_slots)
    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    transaction_id = target_manifest.get("transaction_id")
    if len(comparable_slots) != len(expected_slots) and expected_slots:
        mismatches.append(f"row={row_id} expected_slots contains duplicates")
    if set(source_digests) != comparable_slots:
        mismatches.append(
            f"row={row_id} source_slots={sorted(source_digests)!r} "
            f"expected_comparable={sorted(comparable_slots)!r}"
        )
    expected_target_slots = comparable_slots | set(declared_target_only)
    if set(target_digests) != expected_target_slots:
        mismatches.append(
            f"row={row_id} target_slots={sorted(target_digests)!r} "
            f"expected_topology={sorted(expected_target_slots)!r}"
        )
    undeclared_transformed = declared_transformed - comparable_slots
    if undeclared_transformed:
        mismatches.append(
            f"row={row_id} transformed slots are not source-comparable; "
            f"slots={sorted(undeclared_transformed)!r}"
        )
    if (declared_target_only or target_transformed_slots) and target_transform_metadata is None:
        mismatches.append(f"row={row_id} missing target transform declaration")

    provenance = _fork_slot_provenance(target_manifest)
    source_blob_digests = _slot_blob_digest_map(source_manifest)
    target_blob_digests = _slot_blob_digest_map(target_manifest)
    for slot in sorted(comparable_slots):
        source = source_digests.get(slot)
        target = target_digests.get(slot)
        transformed = slot in declared_transformed
        slot_mismatches: list[str] = []
        if source is None or target is None:
            slot_mismatches.append(f"row={row_id} slot={slot} missing comparable digest")
        elif transformed:
            expected_metadata = (
                target_transform_metadata
                if slot in target_transformed_slots
                else (source_transform_metadata or {}).get(slot)
            )
            slot_mismatches.extend(
                _transform_provenance_mismatches(
                    row_id=row_id,
                    slot=slot,
                    provenance=provenance.get(slot),
                    expected_metadata=expected_metadata,
                )
            )
        elif source != target:
            slot_mismatches.append(f"row={row_id} slot={slot}")
        slot_provenance = provenance.get(slot)
        if transformed and slot_provenance is not None:
            if slot_provenance.get("transfer_mode") != "serialized":
                slot_mismatches.append(
                    f"row={row_id} slot={slot} transformed provenance is not serialized"
                )
            if source_blob_digests and slot_provenance.get("source_sha256") != (
                source_blob_digests.get(slot)
            ):
                slot_mismatches.append(f"row={row_id} slot={slot} source provenance mismatch")
            if target_blob_digests and slot_provenance.get("target_sha256") != (
                target_blob_digests.get(slot)
            ):
                slot_mismatches.append(f"row={row_id} slot={slot} target provenance mismatch")
        ok = not slot_mismatches
        mismatches.extend(slot_mismatches)
        rows.append(
            {
                "kind": "slot_parity",
                "row_id": row_id,
                "planned_run_id": planned_run_id,
                "transaction_id": transaction_id,
                "slot": slot,
                "source_digest": source,
                "target_digest": target,
                "parity_mode": "transformed" if transformed else "preserved",
                "ok": ok,
            }
        )
    for slot, declaration in sorted(declared_target_only.items()):
        slot_provenance = provenance.get(slot)
        slot_mismatches = _transform_provenance_mismatches(
            row_id=row_id,
            slot=slot,
            provenance=slot_provenance,
            expected_metadata=target_transform_metadata,
            expected_target_only_declaration=declaration,
        )
        if slot_provenance is not None and slot_provenance.get("source_sha256") is not None:
            slot_mismatches.append(f"row={row_id} slot={slot} target-only source must be absent")
        if slot_provenance is not None and slot_provenance.get("transfer_mode") != "serialized":
            slot_mismatches.append(
                f"row={row_id} slot={slot} target-only provenance is not serialized"
            )
        if (
            target_blob_digests
            and slot_provenance is not None
            and (slot_provenance.get("target_sha256") != target_blob_digests.get(slot))
        ):
            slot_mismatches.append(f"row={row_id} slot={slot} target provenance mismatch")
        if slot not in target_digests:
            slot_mismatches.append(f"row={row_id} slot={slot} missing target-only digest")
        mismatches.extend(slot_mismatches)
        rows.append(
            {
                "kind": "target_only_provenance",
                "row_id": row_id,
                "planned_run_id": planned_run_id,
                "transaction_id": transaction_id,
                "slot": slot,
                "target_digest": target_digests.get(slot),
                "declaration": dict(declaration),
                "ok": not slot_mismatches,
            }
        )
    return rows, mismatches


def _preflight_lr_continuation_points(
    spec: TrainingRunMatrixSpec,
    materialized: MaterializedRunMatrix,
    *,
    source_manifest: Mapping[str, Any],
    source_checkpoint_root: Path,
    reporter: LrContinuationReporter,
) -> dict[str, list[dict[str, Any]]]:
    """Resolve and validate all continuation report points before fork writes."""
    if spec.fork is None:
        raise RunMatrixError("matrix spec has no fork block")
    if source_manifest.get("schema_id"):
        loaded = _load_latest_checkpoint_transaction(source_checkpoint_root)
        source_manifest = _LoadedSourceManifest(source_manifest, loaded.slots)
    cached: dict[str, list[dict[str, Any]]] = {}
    for row in materialized.rows:
        if row.spec is None:
            raise RunMatrixError(
                f"row {row.row_id!r} does not contain a canonical TrainingRunSpec"
            )
        try:
            points = reporter.points(
                source_manifest=source_manifest,
                row_payload=row.payload,
                row_spec=row.spec,
                declared_mode=spec.fork.lr_continuation,
            )
            if not isinstance(points, list) or any(
                not isinstance(point, Mapping) for point in points
            ):
                raise TypeError("lr_reporter.points must return a list of mappings")
            normalized = [dict(point) for point in points]
            canonical_json_bytes(normalized)
        except RunMatrixError:
            raise
        except Exception as exc:
            raise RunMatrixError(
                f"LR continuation reporter preflight failed for row={row.row_id!r}: {exc}"
            ) from exc
        cached[row.row_id] = normalized
    return cached


def _project_optimizer_spec(
    row_spec: TrainingRunSpec,
    *,
    registry: TrainingMethodRegistry,
) -> OptimizerSpec | None:
    """Project optimizer intent through the method descriptor only."""
    descriptor = registry.descriptor(row_spec.method_ref)
    payload = registry.validate_payload(
        row_spec.method_ref,
        row_spec.method_payload,
        path="/method_payload",
    )
    projector = descriptor.optimizer_spec_projector if descriptor is not None else None
    if projector is None:
        return None
    try:
        projected = projector(payload)
        if not isinstance(projected, (OptimizerSpec, Mapping)):
            raise TypeError("optimizer_spec_projector must return OptimizerSpec or a mapping")
        if isinstance(projected, Mapping):
            unknown = sorted(set(projected) - {"type", "params", "lr_schedule"})
            if unknown:
                raise ValueError(f"optimizer projection contains unknown keys={unknown!r}")
        return OptimizerSpec.model_validate(projected)
    except Exception as exc:
        raise RunMatrixError(
            f"method {row_spec.method_ref.key!r} optimizer projection failed: {exc}"
        ) from exc


def _recorded_optimizer_step(
    row_spec: TrainingRunSpec,
    source_manifest: Mapping[str, Any],
    *,
    registry: TrainingMethodRegistry,
) -> int | None:
    """Read an explicitly recorded checkpoint step through the descriptor hook."""
    metadata = source_manifest.get("metadata")
    if not isinstance(metadata, Mapping) or "optimizer_step" not in metadata:
        return None
    descriptor = registry.descriptor(row_spec.method_ref)
    payload = registry.validate_payload(
        row_spec.method_ref,
        row_spec.method_payload,
        path="/method_payload",
    )
    extractor = descriptor.optimizer_step_extractor if descriptor is not None else None
    if extractor is None:
        return None
    slots = getattr(source_manifest, "slots", None)
    if not isinstance(slots, Mapping):
        if source_manifest.get("schema_id"):
            raise ForkParityError("optimizer parity requires verified decoded source slots")
        slots = source_manifest
    try:
        values = [
            extractor(payload, instance)
            for instance in _loaded_slot_instances(source_manifest, slots)
        ]
    except Exception as exc:
        raise ForkParityError(
            f"method {row_spec.method_ref.key!r} optimizer step extraction failed: {exc}"
        ) from exc
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ForkParityError(
            "optimizer_step_extractor must return a non-bool non-negative integer; "
            f"observed={values!r}"
        )
    if any(value != values[0] for value in values[1:]):
        raise ForkParityError(f"mapped optimizer steps diverge: {values!r}")
    value = values[0]
    if value != metadata["optimizer_step"]:
        raise ForkParityError(
            "recorded optimizer step disagrees with descriptor extraction; "
            f"metadata={metadata['optimizer_step']!r}, extracted={value!r}"
        )
    return value


def _source_completed_step(
    source_manifest: Mapping[str, Any],
    row_spec: TrainingRunSpec,
) -> int:
    slots = getattr(source_manifest, "slots", None)
    authority = row_spec.worker_execution.method_contract.phase_program.batch_progress
    if isinstance(slots, Mapping) and authority is not None and authority.slot in slots:
        values = []
        for instance in _loaded_slot_instances(source_manifest, slots):
            value = instance[authority.slot]
            for segment in authority.field_path:
                value = value[segment]
            values.append(int(value))
        if any(item != values[0] for item in values[1:]):
            raise ForkParityError(f"mapped batch authorities diverge: {values!r}")
        recorded = source_manifest.get("completed_training_batches")
        if recorded != values[0]:
            raise ForkParityError(
                "recorded batch authority disagrees with loaded state: "
                f"metadata={recorded!r}, loaded={values[0]!r}"
            )
        return values[0]
    value = source_manifest.get("completed_training_batches")
    if isinstance(value, int) and value >= 0:
        return value
    raise ForkParityError(
        "source checkpoint manifest is missing a non-negative "
        "/completed_training_batches authority; a program coordinate cannot be "
        "used for LR-continuation arithmetic"
    )


def _loaded_slot_instances(
    manifest: Mapping[str, Any],
    slots: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = {
        str(record["slot"]): record.get("materialized_axes")
        for record in manifest.get("slots", ())
        if isinstance(record, Mapping) and "slot" in record
    }
    mapped = [axes[0] for axes in records.values() if axes]
    if not mapped:
        return [dict(slots)]
    return [
        {
            name: (
                jt.map(lambda leaf: leaf[index] if eqx.is_array(leaf) else leaf, value)
                if records.get(name) and records[name][0]["mode"] == "mapped"
                else value
            )
            for name, value in slots.items()
        }
        for index in range(int(mapped[0]["size"]))
    ]


def _load_spec(path: Path) -> TrainingRunMatrixSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TrainingRunMatrixSpec.model_validate(
        default_spec_registry.migrate("TrainingRunMatrixSpec", payload).payload
    )


def _parse_fork_target(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("target must be ROW=CHECKPOINT_ROOT")
    row_id, raw_path = value.split("=", 1)
    if not row_id:
        raise argparse.ArgumentTypeError("target row id must not be empty")
    if not raw_path:
        raise argparse.ArgumentTypeError("target checkpoint root must not be empty")
    return row_id, Path(raw_path)


def _fork_target_roots(targets: list[tuple[str, Path]]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for row_id, root in targets:
        if row_id in roots:
            raise RunMatrixError(f"duplicate target row id {row_id!r}")
        roots[row_id] = root
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize", "render", "fork"):
        child = subparsers.add_parser(name)
        child.add_argument("spec", type=Path)
        child.add_argument("--repo-root", type=Path, default=Path.cwd())
    materialize_parser = subparsers.choices["materialize"]
    materialize_parser.add_argument("--out-dir", type=Path, required=True)
    materialize_parser.add_argument("--wrap-key")
    materialize_parser.add_argument(
        "--plugin",
        action="append",
        help=(
            "Import a module that registers Feedbax training methods before "
            "run-matrix validation; may be repeated."
        ),
    )
    fork_parser = subparsers.choices["fork"]
    fork_parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    fork_parser.add_argument(
        "--target",
        action="append",
        type=_parse_fork_target,
        required=True,
        help="Target checkpoint root as ROW=CHECKPOINT_ROOT; may be repeated.",
    )
    fork_parser.add_argument("--parity-output", type=Path, required=True)
    fork_parser.add_argument("--skip-fork", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "materialize":
        from feedbax.plugins import load_training_method_plugins

        load_training_method_plugins(modules=args.plugin)
    spec = _load_spec(args.spec)
    materialized = materialize_run_matrix(spec, repo_root=args.repo_root)
    if args.command == "render":
        print(render_spec_lock_table(spec, materialized))
        return 0
    if args.command == "fork":
        try:
            fork_matrix_checkpoints(
                spec,
                materialized,
                source_checkpoint_root=args.source_checkpoint_root,
                target_checkpoint_roots=_fork_target_roots(args.target),
                parity_output_path=args.parity_output,
                skip_fork=args.skip_fork,
            )
        except ForkParityError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0
    write_materialized_matrix(materialized, args.out_dir, wrap_key=args.wrap_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
