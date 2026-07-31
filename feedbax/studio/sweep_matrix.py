"""Training sweep matrix expansion for Studio staging."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from feedbax.contracts.manifest import (
    TrainingRunAxisCoordinate,
    TrainingRunSetAxes,
    TrainingSweepAxis,
    TrainingSweepAxisVariation,
    TrainingSweepCombinationSpec,
)
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1,
    TrainingRunMatrixSpec,
)
from feedbax.training.run_matrix import (
    MaterializedRunMatrix,
    RunMatrixError,
    expand_sweep_coordinates,
    materialize_adapted_run_matrix,
    materialize_run_matrix,
    variation_values,
)
from feedbax.contracts.training import TrainingMethodRegistry
from feedbax.training.row_lowering import TrainingRowLowererRegistry


class SweepMatrixError(ValueError):
    """Raised when a Studio-authored sweep matrix cannot be expanded."""


class ExpandedSweepRun:
    """One concrete run produced by matrix expansion."""

    def __init__(
        self,
        *,
        run_id: str,
        graph_spec: dict[str, Any],
        training_spec: dict[str, Any],
        task_spec: dict[str, Any],
        task_binding_spec: dict[str, Any] | None,
        coordinate: TrainingRunAxisCoordinate,
        overrides: list[dict[str, Any]],
    ) -> None:
        self.run_id = run_id
        self.graph_spec = graph_spec
        self.training_spec = training_spec
        self.task_spec = task_spec
        self.task_binding_spec = task_binding_spec
        self.coordinate = coordinate
        self.overrides = overrides


class ExpandedSweepMatrix:
    """Concrete run set produced by matrix expansion."""

    def __init__(
        self,
        *,
        run_set_id: str,
        name: str,
        axes: TrainingRunSetAxes,
        runs: list[ExpandedSweepRun],
    ) -> None:
        self.run_set_id = run_set_id
        self.name = name
        self.axes = axes
        self.runs = runs


def matrix_spec_from_selection(selection_spec: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the authored matrix spec from a train stage selection, if present."""
    for key in ("matrix", "sweep_matrix", "run_matrix"):
        value = selection_spec.get(key)
        if isinstance(value, Mapping):
            if value.get("schema_id") == TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID:
                try:
                    TrainingRunMatrixSpec.model_validate(value)
                except ValidationError as exc:
                    raise SweepMatrixError(f"invalid training run matrix spec: {exc}") from exc
            return value
    return None


def expand_sweep_matrix(
    matrix_spec: Mapping[str, Any],
    *,
    graph_spec: dict[str, Any],
    training_spec: dict[str, Any],
    task_spec: dict[str, Any],
    task_binding_spec: dict[str, Any] | None,
    default_name: str,
    method_registry: TrainingMethodRegistry,
    row_lowerer_registry: TrainingRowLowererRegistry,
) -> ExpandedSweepMatrix:
    """Adapt the shared materializer result to the legacy Studio return shape."""
    materialized = materialize_sweep_matrix(
        matrix_spec,
        graph_spec=graph_spec,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=task_binding_spec,
        default_name=default_name,
        method_registry=method_registry,
        row_lowerer_registry=row_lowerer_registry,
    )
    runs: list[ExpandedSweepRun] = []
    for row in materialized.rows:
        graph, training, task, task_binding = _studio_payload_parts(row.payload)
        coordinate = row.coordinate
        if coordinate is None:
            raise SweepMatrixError("legacy Studio sweep adapter requires sweep coordinates")
        runs.append(
            ExpandedSweepRun(
                run_id=row.planned_run_id,
                graph_spec=graph,
                training_spec=training,
                task_spec=task,
                task_binding_spec=task_binding,
                coordinate=coordinate,
                overrides=[
                    override.model_dump(mode="json", exclude_none=True)
                    for override in row.overrides
                ],
            )
        )
    return ExpandedSweepMatrix(
        run_set_id=materialized.run_set_id,
        name=materialized.run_set_manifest.name,
        axes=materialized.run_set_manifest.axes,
        runs=runs,
    )


def materialize_sweep_matrix(
    matrix_spec: Mapping[str, Any],
    *,
    graph_spec: dict[str, Any],
    training_spec: dict[str, Any],
    task_spec: dict[str, Any],
    task_binding_spec: dict[str, Any] | None,
    default_name: str,
    method_registry: TrainingMethodRegistry,
    row_lowerer_registry: TrainingRowLowererRegistry,
    repo_root: Path | None = None,
) -> MaterializedRunMatrix:
    """Route governed and legacy Studio matrix documents through one materializer."""
    if matrix_spec.get("schema_id") == TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID:
        governed = TrainingRunMatrixSpec.model_validate(matrix_spec)
        try:
            return materialize_run_matrix(
                governed,
                repo_root=Path.cwd() if repo_root is None else repo_root,
                method_registry=method_registry,
                row_lowerer=row_lowerer_registry.lower,
            )
        except ValueError as exc:
            raise SweepMatrixError(str(exc)) from exc

    axes = _parse_axes(matrix_spec)
    combination = _parse_combination(matrix_spec)
    _validate_group_axes(axes, combination)
    governed = TrainingRunMatrixSpec.model_validate(
        {
            "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            "name": str(matrix_spec.get("name") or matrix_spec.get("label") or default_name),
            "base": {
                "kind": "inline",
                "inline": {
                    "graph_spec": graph_spec,
                    "training_spec": training_spec,
                    "task_spec": task_spec,
                    "task_binding_spec": task_binding_spec,
                },
            },
            "axes": [axis.model_dump(mode="json", exclude_none=True) for axis in axes],
            "combination": combination.model_dump(mode="json", exclude_none=True),
            "metadata": {
                "matrix_schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1,
                "studio_legacy_adapter": True,
            },
        }
    )
    try:
        return materialize_adapted_run_matrix(
            governed,
            repo_root=Path.cwd() if repo_root is None else repo_root,
            row_lowerer=row_lowerer_registry.lower,
            row_validator=_validate_studio_row_payload,
        )
    except ValueError as exc:
        raise SweepMatrixError(str(exc)) from exc


def _validate_studio_row_payload(
    payload: dict[str, Any],
    row_id: str,
) -> None:
    try:
        _studio_payload_parts(payload)
    except SweepMatrixError as exc:
        raise SweepMatrixError(f"row {row_id!r}: {exc}") from exc
    return None


def _studio_payload_parts(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    graph = payload.get("graph_spec")
    training = payload.get("training_spec")
    task = payload.get("task_spec")
    task_binding = payload.get("task_binding_spec")
    if isinstance(graph, dict) and isinstance(training, dict) and isinstance(task, dict):
        return (
            graph,
            training,
            task,
            task_binding if isinstance(task_binding, dict) else None,
        )
    graph_source = payload.get("graph")
    canonical_task = payload.get("task")
    if isinstance(graph_source, dict) and isinstance(canonical_task, dict):
        graph_inline = graph_source.get("inline")
        if isinstance(graph_inline, dict):
            return graph_inline, dict(payload), canonical_task, None
    raise SweepMatrixError("Studio matrix payload lost its spec envelope")


def _parse_axes(matrix_spec: Mapping[str, Any]) -> list[TrainingSweepAxis]:
    raw_axes = matrix_spec.get("axes")
    if not isinstance(raw_axes, list) or not raw_axes:
        raise SweepMatrixError("sweep matrix requires a non-empty axes list")
    axes: list[TrainingSweepAxis] = []
    seen: set[str] = set()
    for index, raw_axis in enumerate(raw_axes):
        if not isinstance(raw_axis, Mapping):
            raise SweepMatrixError(f"sweep axis {index} must be an object")
        axis_data = dict(raw_axis)
        if "variation" not in axis_data:
            if "values" in axis_data:
                axis_data["variation"] = {
                    "kind": "explicit",
                    "values": axis_data.pop("values"),
                }
            else:
                variation_keys = {"kind", "min", "max", "n", "sampler", "seed", "params"}
                variation = {
                    key: axis_data.pop(key) for key in list(axis_data) if key in variation_keys
                }
                if variation:
                    axis_data["variation"] = variation
        try:
            axis = TrainingSweepAxis.model_validate(axis_data)
        except ValidationError as exc:
            raise SweepMatrixError(f"invalid sweep axis {index}: {exc}") from exc
        if axis.id in seen:
            raise SweepMatrixError(f"duplicate sweep axis id {axis.id!r}")
        seen.add(axis.id)
        axes.append(axis)
    return axes


def _parse_combination(matrix_spec: Mapping[str, Any]) -> TrainingSweepCombinationSpec:
    raw_combination = matrix_spec.get("combination")
    if isinstance(raw_combination, Mapping):
        data = dict(raw_combination)
    else:
        data = {"mode": matrix_spec.get("mode", "cross")}
    try:
        return TrainingSweepCombinationSpec.model_validate(data)
    except ValidationError as exc:
        raise SweepMatrixError(f"invalid sweep matrix combination: {exc}") from exc


def _validate_group_axes(
    axes: list[TrainingSweepAxis],
    combination: TrainingSweepCombinationSpec,
) -> None:
    axis_ids = {axis.id for axis in axes}
    used: set[str] = set()
    for group in combination.groups:
        unknown = [axis_id for axis_id in group.axes if axis_id not in axis_ids]
        if unknown:
            raise SweepMatrixError(f"sweep group {group.id!r} references unknown axes {unknown!r}")
        overlap = used.intersection(group.axes)
        if overlap:
            raise SweepMatrixError(f"sweep axes {sorted(overlap)!r} appear in more than one group")
        used.update(group.axes)
    if combination.groups:
        missing = sorted(axis_ids - used)
        if missing:
            raise SweepMatrixError(
                f"sweep matrix groups must cover every declared axis; missing axes {missing!r}"
            )


def _expand_coordinates(
    axes: list[TrainingSweepAxis],
    combination: TrainingSweepCombinationSpec,
) -> list[dict[str, int]]:
    try:
        return expand_sweep_coordinates(axes, combination)
    except RunMatrixError as exc:
        raise SweepMatrixError(str(exc)) from exc


def _variation_values(variation: TrainingSweepAxisVariation) -> list[Any]:
    try:
        return variation_values(variation)
    except RunMatrixError as exc:
        raise SweepMatrixError(str(exc)) from exc


def _coordinate_label(
    axis_by_id: Mapping[str, TrainingSweepAxis],
    values: Mapping[str, Any],
) -> str:
    return ", ".join(
        f"{axis_by_id[axis_id].label or axis_by_id[axis_id].path}={values[axis_id]!r}"
        for axis_id in sorted(values)
    )
