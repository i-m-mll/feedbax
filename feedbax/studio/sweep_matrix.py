"""Training sweep matrix expansion for Studio staging."""

from __future__ import annotations

import copy
import math
import random
from itertools import product
from typing import Any, Mapping

from pydantic import ValidationError

from feedbax.contracts.manifest import (
    TrainingRunAxisCoordinate,
    TrainingRunSetAxes,
    TrainingSweepAxis,
    TrainingSweepAxisGroup,
    TrainingSweepAxisVariation,
    TrainingSweepCombinationSpec,
    planned_training_run_manifest_id,
    planned_training_run_set_manifest_id,
)


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
) -> ExpandedSweepMatrix:
    """Expand one Studio matrix spec into concrete training run specs."""
    axes = _parse_axes(matrix_spec)
    combination = _parse_combination(matrix_spec)
    _validate_group_axes(axes, combination)
    indexed_coordinates = _expand_coordinates(axes, combination)
    axes_with_values = [
        axis.model_copy(update={"values": _variation_values(axis.variation)})
        for axis in axes
    ]
    axis_by_id = {axis.id: axis for axis in axes_with_values}
    run_set_axes = TrainingRunSetAxes(
        axes=axes_with_values,
        combination=combination,
        metadata={"axis_count": len(axes), "run_count": len(indexed_coordinates)},
    )
    run_set_id = planned_training_run_set_manifest_id(
        graph_spec=graph_spec,
        base_training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=task_binding_spec,
        axes=run_set_axes.model_dump(mode="json", exclude_none=True),
    )
    runs: list[ExpandedSweepRun] = []
    for index, value_indices in enumerate(indexed_coordinates):
        values = {
            axis_id: axis_by_id[axis_id].values[value_index]
            for axis_id, value_index in value_indices.items()
        }
        next_graph_spec = copy.deepcopy(graph_spec)
        next_training_spec = copy.deepcopy(training_spec)
        next_task_spec = copy.deepcopy(task_spec)
        next_task_binding_spec = copy.deepcopy(task_binding_spec)
        for axis_id, value in values.items():
            _set_axis_value(
                axis_by_id[axis_id].path,
                value,
                graph_spec=next_graph_spec,
                training_spec=next_training_spec,
                task_spec=next_task_spec,
                task_binding_spec=next_task_binding_spec,
            )
        run_id = planned_training_run_manifest_id(
            graph_spec=next_graph_spec,
            training_spec=next_training_spec,
            task_spec=next_task_spec,
            task_binding_spec=next_task_binding_spec,
            seed=_training_seed(next_training_spec),
            axis_coordinates=values,
        )
        overrides = [
            {"path": axis_by_id[axis_id].path, "value": value, "op": "replace"}
            for axis_id, value in values.items()
        ]
        coordinate = TrainingRunAxisCoordinate(
            run_id=run_id,
            index=index,
            value_indices=value_indices,
            values=values,
            label=_coordinate_label(axis_by_id, values),
        )
        runs.append(
            ExpandedSweepRun(
                run_id=run_id,
                graph_spec=next_graph_spec,
                training_spec=next_training_spec,
                task_spec=next_task_spec,
                task_binding_spec=next_task_binding_spec,
                coordinate=coordinate,
                overrides=overrides,
            )
        )
    run_set_axes.runs = [run.coordinate for run in runs]
    return ExpandedSweepMatrix(
        run_set_id=run_set_id,
        name=str(matrix_spec.get("name") or matrix_spec.get("label") or default_name),
        axes=run_set_axes,
        runs=runs,
    )


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
                    key: axis_data.pop(key)
                    for key in list(axis_data)
                    if key in variation_keys
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
            raise SweepMatrixError(
                f"sweep group {group.id!r} references unknown axes {unknown!r}"
            )
        overlap = used.intersection(group.axes)
        if overlap:
            raise SweepMatrixError(
                f"sweep axes {sorted(overlap)!r} appear in more than one group"
            )
        used.update(group.axes)
    if combination.groups:
        missing = sorted(axis_ids - used)
        if missing:
            raise SweepMatrixError(
                "sweep matrix groups must cover every declared axis; "
                f"missing axes {missing!r}"
            )


def _expand_coordinates(
    axes: list[TrainingSweepAxis],
    combination: TrainingSweepCombinationSpec,
) -> list[dict[str, int]]:
    axis_lengths = {axis.id: len(_variation_values(axis.variation)) for axis in axes}
    if combination.mode == "manual":
        if not combination.manual_coordinates:
            raise SweepMatrixError("manual sweep matrix requires manual_coordinates")
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
    grouped_coordinates = [
        _expand_group(group, axis_lengths)
        for group in groups
    ]
    out: list[dict[str, int]] = []
    for parts in product(*grouped_coordinates):
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
            raise SweepMatrixError(
                f"zip sweep group {group.id!r} has mismatched lengths {sorted(lengths)!r}"
            )
        return [
            {axis_id: index for axis_id in group.axes}
            for index in range(next(iter(lengths)))
        ]
    return [
        dict(zip(group.axes, indices))
        for indices in product(*(range(axis_lengths[axis_id]) for axis_id in group.axes))
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
            raise SweepMatrixError(
                f"manual sweep coordinate {index} must include integer index for {axis_id!r}"
            )
        if value < 0 or value >= length:
            raise SweepMatrixError(
                f"manual sweep coordinate {index} index for {axis_id!r} is out of range"
            )
        coordinate[axis_id] = value
    return coordinate


def _variation_values(variation: TrainingSweepAxisVariation) -> list[Any]:
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
            raise SweepMatrixError("log_uniform sampler requires positive min and max")
        return [10 ** rng.uniform(math.log10(low), math.log10(high)) for _ in range(variation.n)]
    if variation.sampler == "normal":
        mean = float(variation.params.get("mean", 0.0))
        std = float(variation.params.get("std", 1.0))
        return [rng.gauss(mean, std) for _ in range(variation.n)]
    raise SweepMatrixError(f"unsupported sweep sampler {variation.sampler!r}")


def _set_axis_value(
    path: str,
    value: Any,
    *,
    graph_spec: dict[str, Any],
    training_spec: dict[str, Any],
    task_spec: dict[str, Any],
    task_binding_spec: dict[str, Any] | None,
) -> None:
    root_name, parts = _split_axis_path(path)
    if root_name == "seed":
        training_spec["seed"] = value
        return
    roots: dict[str, dict[str, Any] | None] = {
        "graph_spec": graph_spec,
        "training_spec": training_spec,
        "task_spec": task_spec,
        "task_binding_spec": task_binding_spec,
    }
    root = roots.get(root_name)
    if root is None:
        raise SweepMatrixError(f"sweep axis path {path!r} does not resolve to a spec root")
    _set_nested(root, parts, value, path=path)


def _split_axis_path(path: str) -> tuple[str, list[str]]:
    if path in {"seed", "master_prng_key", "prng_key"}:
        return "seed", []
    if path.startswith("/"):
        parts = [part for part in path.split("/") if part]
    else:
        parts = [part for part in path.split(".") if part]
    if not parts:
        raise SweepMatrixError("sweep axis path must be non-empty")
    root = parts[0]
    if root == "seed":
        return "seed", []
    if root == "master_prng_key":
        return "seed", []
    return root, parts[1:]


def _set_nested(root: dict[str, Any], parts: list[str], value: Any, *, path: str) -> None:
    if not parts:
        raise SweepMatrixError(f"sweep axis path {path!r} must include a field after the root")
    current: Any = root
    for part in parts[:-1]:
        if isinstance(current, dict):
            if part not in current:
                raise SweepMatrixError(
                    f"sweep axis path {path!r} cannot traverse missing field {part!r}"
                )
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index < 0 or index >= len(current):
                raise SweepMatrixError(
                    f"sweep axis path {path!r} list index {index} is out of range"
                )
            current = current[index]
            continue
        raise SweepMatrixError(f"sweep axis path {path!r} cannot traverse {part!r}")
    leaf = parts[-1]
    if isinstance(current, dict):
        if leaf not in current:
            raise SweepMatrixError(
                f"sweep axis path {path!r} cannot set missing field {leaf!r}"
            )
        current[leaf] = value
        return
    if isinstance(current, list) and leaf.isdigit():
        index = int(leaf)
        if index < 0 or index >= len(current):
            raise SweepMatrixError(
                f"sweep axis path {path!r} list index {index} is out of range"
            )
        current[index] = value
        return
    raise SweepMatrixError(f"sweep axis path {path!r} cannot set {leaf!r}")


def _coordinate_label(
    axis_by_id: Mapping[str, TrainingSweepAxis],
    values: Mapping[str, Any],
) -> str:
    return ", ".join(
        f"{axis_by_id[axis_id].label or axis_by_id[axis_id].path}={values[axis_id]!r}"
        for axis_id in sorted(values)
    )


def _training_seed(training_spec: Mapping[str, Any]) -> Any | None:
    if "seed" in training_spec:
        return training_spec["seed"]
    params = training_spec.get("params")
    if isinstance(params, Mapping) and "seed" in params:
        return params["seed"]
    return None
