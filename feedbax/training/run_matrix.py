"""Materialization and checkpoint forking for training run matrices."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Protocol

from feedbax.contracts.expressions import evaluate_query
from feedbax.contracts.extraction import load_expression_context, set_dotted_path
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
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    TrainingRunMatrixSpec,
    apply_override_patches,
)
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    LrScheduleSpec,
    TrainingMethodRegistry,
    TrainingRunSpec,
)
from feedbax.training.checkpoint_custody import fork_checkpoint_transaction
from feedbax.training.optimizers import learning_rate_at_step


RUN_MATRIX_FORK_PARITY_SCHEMA_VERSION = "feedbax.run_matrix_fork_parity.v1"


class RunMatrixError(ValueError):
    """Raised when a run matrix cannot be materialized."""


class ForkParityError(RunMatrixError):
    """Raised when forked checkpoint slot parity fails."""


@dataclass(frozen=True)
class MaterializedMatrixRow:
    """One concrete row materialized from a matrix document."""

    row_id: str
    planned_run_id: str
    spec: TrainingRunSpec | None
    payload: dict[str, Any]
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


class StandardLrContinuationReporter:
    """Generic LR reporter for constant and declarative schedule optimizer specs."""

    def points(
        self,
        *,
        source_manifest: Mapping[str, Any],
        row_payload: Mapping[str, Any],
        row_spec: TrainingRunSpec,
        declared_mode: str,
    ) -> list[dict[str, Any]]:
        optimizer = _find_optimizer_spec(row_payload)
        if optimizer is None:
            return []
        current_step = _source_completed_step(source_manifest)
        if declared_mode == "restart":
            current_step = 0
        schedule = optimizer.get("lr_schedule")
        if schedule is None:
            params = optimizer.get("params")
            if isinstance(params, Mapping) and "learning_rate" in params:
                return [{"step": current_step, "lr": params["learning_rate"], "mode": declared_mode}]
            training_config = row_payload.get("training_config")
            if isinstance(training_config, Mapping) and "learning_rate" in training_config:
                return [
                    {
                        "step": current_step,
                        "lr": training_config["learning_rate"],
                        "mode": declared_mode,
                    }
                ]
            return []
        lr = learning_rate_at_step(
            LrScheduleSpec.model_validate(schedule),
            current_step=current_step,
            schedule_origin_step=0,
        )
        return [{"step": current_step, "lr": float(lr), "mode": declared_mode}]


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
    row_validator: RowPayloadValidator,
) -> MaterializedRunMatrix:
    """Materialize an adapter payload after validating every expanded row."""
    return _materialize_run_matrix(
        spec,
        repo_root=repo_root,
        row_validator=row_validator,
    )


def _materialize_run_matrix(
    spec: TrainingRunMatrixSpec | Mapping[str, Any],
    *,
    repo_root: Path,
    row_validator: RowPayloadValidator,
) -> MaterializedRunMatrix:
    matrix = (
        spec
        if isinstance(spec, TrainingRunMatrixSpec)
        else TrainingRunMatrixSpec.model_validate(spec)
    )
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
            }
        )
    manifest = {
        "schema_version": "feedbax.run_matrix_materialization.v1",
        "matrix_spec_sha256": materialized.matrix_spec_sha256,
        "run_set_id": materialized.run_set_id,
        "rows": row_records,
    }
    (out_dir / "matrix_materialization.json").write_bytes(
        canonical_json_bytes(manifest) + b"\n"
    )
    return manifest


def render_spec_lock_table(
    spec: TrainingRunMatrixSpec,
    materialized: MaterializedRunMatrix,
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
    header = [
        f"Matrix: {spec.name}",
        f"Issue: {spec.issue or ''}",
        f"Base ref: {spec.base.ref or '<inline>'}",
        f"Base sha256: {spec.base.sha256 or ''}",
        f"Fork source: {spec.fork.source_run_id if spec.fork else ''}",
        (
            "LR continuation schedule: "
            f"{spec.fork.lr_continuation if spec.fork else ''}"
        ),
        f"Row count: {len(materialized.rows)}",
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


def fork_matrix_checkpoints(
    spec: TrainingRunMatrixSpec,
    materialized: MaterializedRunMatrix,
    *,
    source_checkpoint_root: Path,
    target_checkpoint_roots: Mapping[str, Path],
    parity_output_path: Path,
    skip_fork: bool = False,
    lr_reporter: LrContinuationReporter | None = None,
    tool_version: str = "feedbax.run_matrix_fork.v1",
) -> dict[str, Any]:
    """Fork a source checkpoint to all matrix rows and write a parity table."""
    if spec.fork is None:
        raise RunMatrixError("matrix spec has no fork block")
    row_ids = {row.row_id for row in materialized.rows}
    unexpected_targets = sorted(set(target_checkpoint_roots) - row_ids)
    if unexpected_targets:
        raise RunMatrixError(
            f"target checkpoint roots contain unknown rows {unexpected_targets!r}"
        )
    reporter = lr_reporter or StandardLrContinuationReporter()
    parity_rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for row in materialized.rows:
        if row.spec is None:
            raise RunMatrixError(
                f"row {row.row_id!r} does not contain a canonical TrainingRunSpec"
            )
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
            result = fork_checkpoint_transaction(
                source_checkpoint_root,
                target_root,
                target_run_spec=row.spec,
                target_phase_program=row.spec.worker_execution.method_contract.phase_program,
                tool_version=tool_version,
                metadata={
                    "matrix_spec_sha256": materialized.matrix_spec_sha256,
                    "matrix_row_id": row.row_id,
                    "planned_run_id": row.planned_run_id,
                },
            )
            source_manifest = result.manifest.fork_provenance.source.model_dump(
                mode="json",
                exclude_none=True,
            )
            target_manifest = result.manifest.model_dump(mode="json", exclude_none=True)
            transaction_id = result.manifest.transaction_id
        slot_rows, row_mismatches = _parity_rows(
            row_id=row.row_id,
            planned_run_id=row.planned_run_id,
            source_manifest=source_manifest,
            target_manifest=target_manifest,
            expected_slots=spec.fork.expected_slots,
        )
        parity_rows.extend(slot_rows)
        mismatches.extend(row_mismatches)
        parity_rows.extend(
            {
                "row_id": row.row_id,
                "planned_run_id": row.planned_run_id,
                "kind": "lr_continuation",
                "transaction_id": transaction_id,
                **point,
            }
            for point in reporter.points(
                source_manifest=source_manifest,
                row_payload=row.payload,
                row_spec=row.spec,
                declared_mode=spec.fork.lr_continuation,
            )
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
    if spec.base.inline is not None:
        document = copy.deepcopy(spec.base.inline)
    else:
        assert spec.base.ref is not None
        path = repo_root / spec.base.ref
        data = path.read_bytes()
        if spec.base.sha256 is not None and sha256_bytes(data) != spec.base.sha256:
            raise RunMatrixError(f"/base/ref sha256 mismatch: {spec.base.ref}")
        document = json.loads(data.decode("utf-8"))
    payload = _get_dotted(document, spec.base.payload_path)
    if not isinstance(payload, dict):
        raise RunMatrixError("/base payload must resolve to an object")
    return copy.deepcopy(payload)


def _materialize_explicit_rows(
    matrix: TrainingRunMatrixSpec,
    *,
    base_payload: dict[str, Any],
    row_validator: RowPayloadValidator,
) -> tuple[list[MaterializedMatrixRow], TrainingRunSetAxes]:
    rows: list[MaterializedMatrixRow] = []
    explicit_records: list[dict[str, Any]] = []
    coordinates: list[TrainingRunAxisCoordinate] = []
    for index, row in enumerate(matrix.rows):
        payload = apply_override_patches(base_payload, row.overrides)
        spec = row_validator(payload, row.row_id)
        axis_coordinates = {
            "row_id": row.row_id,
            "overrides": [
                patch.model_dump(mode="json", exclude_none=True)
                for patch in row.overrides
            ],
        }
        run_id = _planned_run_id(payload, seed=row.seed, axis_coordinates=axis_coordinates)
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
                payload=payload,
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
    row_validator: RowPayloadValidator,
) -> tuple[list[MaterializedMatrixRow], TrainingRunSetAxes]:
    axes_with_values = [
        axis.model_copy(update={"values": variation_values(axis.variation)})
        for axis in matrix.axes
    ]
    axis_by_id = {axis.id: axis for axis in axes_with_values}
    indexed_coordinates = expand_sweep_coordinates(axes_with_values, matrix.combination)
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
        payload = copy.deepcopy(base_payload)
        seed = None
        patches = []
        for axis_id, value in values.items():
            axis = axis_by_id[axis_id]
            if axis.path in {"seed", "master_prng_key", "prng_key"}:
                seed = value
                studio_training_spec = payload.get("training_spec")
                if isinstance(studio_training_spec, dict):
                    studio_training_spec["seed"] = value
            else:
                patch = _patch_object({"path": axis.path, "value": value, "op": "replace"})
                patches.append(patch)
                payload = apply_override_patches(payload, [patch])
        spec = row_validator(payload, f"sweep-{index}")
        run_id = _planned_run_id(payload, seed=seed, axis_coordinates=values)
        coordinate = TrainingRunAxisCoordinate(
            run_id=run_id,
            index=index,
            value_indices=value_indices,
            values=values,
            label=_coordinate_label(axis_by_id, values),
        )
        rows.append(
            MaterializedMatrixRow(
                row_id=f"row-{index:04d}",
                planned_run_id=run_id,
                spec=spec,
                payload=payload,
                coordinate=coordinate,
                overrides=patches,
                seed=seed if isinstance(seed, int) else None,
            )
        )
    run_set_axes.runs = [row.coordinate for row in rows if row.coordinate is not None]
    return rows, run_set_axes


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
) -> str:
    return planned_training_run_manifest_id(
        graph_spec=_identity_graph_spec(payload),
        training_spec=_identity_training_spec(payload),
        task_spec=_identity_task_spec(payload),
        task_binding_spec=_identity_task_binding_spec(payload),
        seed=seed,
        axis_coordinates=axis_coordinates,
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
            "execution",
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
                "sweep matrix groups must cover every declared axis; "
                f"missing axes {missing!r}"
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
    return [
        dict(zip(group.axes, indices))
        for indices in _product(*(range(axis_lengths[axis_id]) for axis_id in group.axes))
    ]


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


def _coordinate_label(axis_by_id: Mapping[str, TrainingSweepAxis], values: Mapping[str, Any]) -> str:
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


def _latest_manifest_pair(source_root: Path, target_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
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


def _parity_rows(
    *,
    row_id: str,
    planned_run_id: str,
    source_manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    expected_slots: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    source_digests = _slot_digest_map(source_manifest)
    target_digests = _slot_digest_map(target_manifest)
    slots = sorted(set(source_digests) | set(target_digests))
    if expected_slots:
        if sorted(expected_slots) != slots:
            return [], [f"row={row_id} slots={slots!r} expected={sorted(expected_slots)!r}"]
        slots = sorted(expected_slots)
    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    transaction_id = target_manifest.get("transaction_id")
    for slot in slots:
        source = source_digests.get(slot)
        target = target_digests.get(slot)
        ok = source == target
        if not ok:
            mismatches.append(f"row={row_id} slot={slot}")
        rows.append(
            {
                "kind": "slot_parity",
                "row_id": row_id,
                "planned_run_id": planned_run_id,
                "transaction_id": transaction_id,
                "slot": slot,
                "source_digest": source,
                "target_digest": target,
                "ok": ok,
            }
        )
    return rows, mismatches


def _find_optimizer_spec(row_payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    method_payload = row_payload.get("method_payload")
    if isinstance(method_payload, Mapping):
        payload = method_payload.get("payload")
        if isinstance(payload, Mapping):
            optimizer = payload.get("optimizer")
            if isinstance(optimizer, Mapping):
                return optimizer
    training_config = row_payload.get("training_config")
    if isinstance(training_config, Mapping) and "learning_rate" in training_config:
        return {"params": {"learning_rate": training_config["learning_rate"]}}
    return None


def _source_completed_step(source_manifest: Mapping[str, Any]) -> int:
    value = source_manifest.get("completed_training_batches")
    if isinstance(value, int) and value >= 0:
        return value
    raise ForkParityError(
        "source checkpoint manifest is missing a non-negative "
        "/completed_training_batches authority; a program coordinate cannot be "
        "used for LR-continuation arithmetic"
    )


def _load_spec(path: Path) -> TrainingRunMatrixSpec:
    return TrainingRunMatrixSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))


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
