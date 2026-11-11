"""Authored sweep grammar and static inheritance for project steps."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import product
import math
import re
from typing import Annotated, Literal, TypeVar

from pydantic import Field, JsonValue, model_validator

from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project_inputs import ProjectName, RowKey
from feedbax.contracts.run_matrix import _paths_overlap

type PatchPath = Annotated[str, Field(pattern=r"^[^.]+(?:\.[^.]+)*$")]


class SetValue(StrictModel):
    value: JsonValue
    op: Literal["add", "replace"] = "replace"


class RemoveValue(StrictModel):
    op: Literal["remove"]


type PatchValue = SetValue | RemoveValue


def sweep_patches(patches: Mapping[str, PatchValue | None]) -> list[OverridePatch]:
    """Adapt authored path-keyed patches to the common patch engine."""
    return [
        OverridePatch(path=path, op=patch.op, value=patch.value)
        if isinstance(patch, SetValue)
        else OverridePatch(path=path, op="remove")
        for path, patch in patches.items()
        if patch is not None
    ]


class AxisValue(StrictModel):
    label: str | None = None
    patches: dict[PatchPath, PatchValue | None] = Field(default_factory=dict)


class Row(AxisValue):
    source: ProjectName | None = Field(default=None, alias="from")


class NamedAxis(StrictModel):
    values: dict[ProjectName, AxisValue | None]


class InputAxis(StrictModel):
    input: ProjectName


class ListedAxis(StrictModel):
    path: PatchPath
    values: list[JsonValue] = Field(min_length=1)


class RangeValues(StrictModel):
    start: float
    stop: float
    count: int = Field(ge=1)
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def _check_log_range(self) -> RangeValues:
        if self.scale == "log" and min(self.start, self.stop) <= 0:
            raise ValueError("a logarithmic range requires positive bounds")
        return self


class RangeAxis(StrictModel):
    path: PatchPath
    range: RangeValues


class SampleValues(StrictModel):
    distribution: ProjectName
    count: int = Field(ge=1)
    seed: int


class SampledAxis(StrictModel):
    path: PatchPath
    sampler: SampleValues


type Axis = NamedAxis | InputAxis | ListedAxis | RangeAxis | SampledAxis


class ManualCombination(StrictModel):
    manual: list[dict[ProjectName, RowKey]]


class AxisGroup(StrictModel):
    axes: list[ProjectName] = Field(min_length=1)
    mode: Literal["cross", "zip"]


class GroupCombination(StrictModel):
    groups: dict[ProjectName, AxisGroup]


type Combination = Literal["cross", "zip"] | ManualCombination | GroupCombination


class Sweep(StrictModel):
    axes: dict[ProjectName, Axis | None] = Field(default_factory=dict)
    rows: dict[ProjectName, Row | None] = Field(default_factory=dict)
    combination: Combination | None = None
    inherit_rows: bool = True


_T = TypeVar("_T")


def merge_named(parent: Mapping[str, _T], child: Mapping[str, _T | None]) -> dict[str, _T]:
    """Apply ordered whole-value changes to a name-keyed collection."""
    merged = dict(parent)
    for name, entry in child.items():
        if entry is None:
            merged.pop(name, None)
        else:
            merged[name] = entry
    return merged


def _merge_value(parent: AxisValue | None, child: AxisValue) -> AxisValue:
    if parent is None:
        return child.model_copy(update={"patches": merge_named({}, child.patches)})
    return child.model_copy(
        update={
            "label": child.label if "label" in child.model_fields_set else parent.label,
            "patches": merge_named(
                {path: patch for path, patch in parent.patches.items() if patch is not None},
                child.patches,
            ),
        }
    )


def _derive_row(parent: Row, child: Row) -> AxisValue:
    """Compose a source row's writes with a derived row's writes at the same path."""
    combined = _merge_value(parent, child)
    for path, write in child.patches.items():
        previous = parent.patches.get(path)
        if isinstance(previous, SetValue) and previous.op == "add":
            if isinstance(write, SetValue):
                combined.patches[path] = write.model_copy(update={"op": "add"})
            elif isinstance(write, RemoveValue):
                combined.patches.pop(path, None)
        elif isinstance(previous, RemoveValue) and isinstance(write, SetValue):
            combined.patches[path] = write.model_copy(update={"op": "replace"})
    return combined


class SweepInheritanceError(ValueError):
    def __init__(self, row: str, source: str):
        self.row = row
        self.source = source
        super().__init__(f"row {row!r} derives from missing or later row {source!r}")


def inherit_sweep(parent: Sweep | None, child: Sweep | None) -> Sweep | None:
    """Resolve by-id refinements and earlier row sources without expanding axes."""
    if child is None:
        return parent
    previous = parent or Sweep()
    axes = merge_named(
        {name: axis for name, axis in previous.axes.items() if axis is not None}, child.axes
    )
    for name, axis in child.axes.items():
        if isinstance(axis, NamedAxis):
            prior = previous.axes.get(name)
            values = (
                {key: value for key, value in prior.values.items() if value is not None}
                if isinstance(prior, NamedAxis)
                else {}
            )
            for key, value in axis.values.items():
                if value is None:
                    values.pop(key, None)
                else:
                    values[key] = _merge_value(values.get(key), value)
            axes[name] = NamedAxis(values=values)

    prior_rows = {name: row for name, row in previous.rows.items() if row is not None}
    rows = dict(prior_rows) if child.inherit_rows else {}
    sources = dict(prior_rows)
    for name, row in child.rows.items():
        if row is None:
            rows.pop(name, None)
            sources.pop(name, None)
            continue
        source = row.source
        if source is not None and source not in sources:
            raise SweepInheritanceError(name, source)
        combined = (
            _derive_row(sources[source], row)
            if source is not None
            else _merge_value(rows.get(name), row)
        )
        resolved = Row(label=combined.label, patches=combined.patches)
        rows[name] = sources[name] = resolved

    return Sweep(
        axes=axes,
        rows=rows,
        combination=(
            child.combination if "combination" in child.model_fields_set else previous.combination
        ),
    )


class SweepExpansionCode(StrEnum):
    INVALID_COMBINATION = "invalid_combination"
    UNKNOWN_VALUE = "unknown_value"
    EMPTY_AXIS = "empty_axis"
    UNKNOWN_CARDINALITY = "unknown_cardinality"
    INPUT_CONFLICT = "input_conflict"
    UNRESOLVED_SAMPLER = "unresolved_sampler"
    INVALID_GENERATION = "invalid_generation"
    PATCH_CONFLICT = "patch_conflict"
    COORDINATE_CONFLICT = "coordinate_conflict"
    DUPLICATE_ROW = "duplicate_row"
    INVALID_ROW_SOURCE = "invalid_row_source"
    ROW_BUDGET = "row_budget"
    AXIS_BUDGET = "axis_budget"


class SweepExpansionError(ValueError):
    """A sweep refusal with a path suitable for an authored diagnostic."""

    def __init__(self, code: SweepExpansionCode, path: str, detail: str):
        self.code = code
        self.path = path
        self.detail = detail
        super().__init__(f"{path}: {detail}")


@dataclass(frozen=True)
class SweepCoordinate:
    value_id: str
    label: str | None = None
    value: JsonValue = None


@dataclass(frozen=True)
class SweepInputMember:
    """A member whose slot and key are known at plan time."""

    key: str
    coordinates: Mapping[str, SweepCoordinate] = field(default_factory=dict)
    label: str | None = None


type SweepInputs = Mapping[str, Sequence[SweepInputMember] | None]
type SamplerResolver = Callable[[str, int, int], Sequence[JsonValue] | None]


@dataclass(frozen=True)
class ExpandedRow:
    id: str
    label: str | None
    coordinates: Mapping[str, SweepCoordinate]
    patches: Mapping[str, PatchValue]
    input_keys: Mapping[str, str]


DEFAULT_SWEEP_ROW = "default"
DEFAULT_SWEEP_LIMIT = 10_000


@dataclass(frozen=True)
class _AxisChoice:
    key: str
    coordinate: SweepCoordinate
    patches: Mapping[str, PatchValue] = field(default_factory=dict)
    member: SweepInputMember | None = None


@dataclass(frozen=True)
class _SweepPlan:
    axes: Mapping[str, Axis]
    rows: Mapping[str, Row]
    sizes: Mapping[str, int]
    dimensions: tuple[tuple[tuple[str, ...], int], ...] = ()
    manual: tuple[Mapping[str, str], ...] | None = None

    @property
    def generated_count(self) -> int:
        if self.manual is not None:
            return len(self.manual)
        return math.prod(size for _, size in self.dimensions) if self.axes else 0

    @property
    def row_count(self) -> int:
        return len(self.rows) + self.generated_count if self.rows or self.axes else 1

    def selections(
        self, choices: Mapping[str, Sequence[_AxisChoice]]
    ) -> Iterator[dict[str, _AxisChoice]]:
        if self.manual is not None:
            lookup = {
                name: {choice.key: choice for choice in values} for name, values in choices.items()
            }
            for index, selection in enumerate(self.manual):
                resolved = {}
                for name, key in selection.items():
                    if key not in lookup[name]:
                        raise SweepExpansionError(
                            SweepExpansionCode.UNKNOWN_VALUE,
                            f"sweep.combination.manual.{index}.{name}",
                            f"axis {name!r} has no value {key!r}",
                        )
                    resolved[name] = lookup[name][key]
                yield resolved
            return
        for indices in product(*(range(size) for _, size in self.dimensions)):
            yield {
                name: choices[name][index]
                for (names, _), index in zip(self.dimensions, indices, strict=True)
                for name in names
            }


def _patches(value: AxisValue) -> dict[str, PatchValue]:
    return {path: patch for path, patch in value.patches.items() if patch is not None}


def _axis_size(name: str, axis: Axis, inputs: SweepInputs) -> int:
    match axis:
        case InputAxis(input=role):
            members = inputs.get(role)
            if members is None:
                raise SweepExpansionError(
                    SweepExpansionCode.UNKNOWN_CARDINALITY,
                    f"sweep.axes.{name}.input",
                    f"input {role!r} has no known member count",
                )
            return len(members)
        case NamedAxis(values=values):
            size = sum(value is not None for value in values.values())
            if size == 0:
                raise SweepExpansionError(
                    SweepExpansionCode.EMPTY_AXIS, f"sweep.axes.{name}", "axis has no values"
                )
            return size
        case ListedAxis(values=values):
            return len(values)
        case RangeAxis(range=range_values):
            return range_values.count
        case SampledAxis(sampler=sampler):
            return sampler.count


def _zip_dimension(
    path: str, names: Sequence[str], sizes: Mapping[str, int]
) -> tuple[tuple[str, ...], int]:
    lengths = {name: sizes[name] for name in names}
    if len(set(lengths.values())) != 1:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_COMBINATION,
            path,
            f"zip requires equal axis lengths, got {lengths}",
        )
    return tuple(names), next(iter(lengths.values()))


def _dimensions(
    sizes: Mapping[str, int], combination: Literal["cross", "zip"] | GroupCombination
) -> tuple[tuple[tuple[str, ...], int], ...]:
    if not sizes and isinstance(combination, str):
        return ()
    if combination == "cross":
        return tuple(((name,), size) for name, size in sizes.items())
    if combination == "zip":
        return (_zip_dimension("sweep.combination", tuple(sizes), sizes),)

    dimensions = []
    used: set[str] = set()
    for group_name, group in combination.groups.items():
        path = f"sweep.combination.groups.{group_name}.axes"
        for name in group.axes:
            if name not in sizes:
                raise SweepExpansionError(
                    SweepExpansionCode.INVALID_COMBINATION, path, f"unknown axis {name!r}"
                )
            if name in used:
                raise SweepExpansionError(
                    SweepExpansionCode.INVALID_COMBINATION,
                    path,
                    f"axis {name!r} occurs more than once",
                )
            used.add(name)
        if group.mode == "zip":
            dimensions.append(_zip_dimension(path, group.axes, sizes))
        else:
            dimensions.extend(((name,), sizes[name]) for name in group.axes)
    if missing := sizes.keys() - used:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_COMBINATION,
            "sweep.combination.groups",
            f"groups must cover every axis, missing {sorted(missing)}",
        )
    if not dimensions:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_COMBINATION, "sweep.combination.groups", "no axis groups"
        )
    return tuple(dimensions)


def _plan(sweep: Sweep | None, inputs: SweepInputs) -> _SweepPlan:
    try:
        resolved = inherit_sweep(None, sweep) or Sweep()
    except SweepInheritanceError as error:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_ROW_SOURCE, f"sweep.rows.{error.row}.from", str(error)
        ) from error
    axes = {name: axis for name, axis in resolved.axes.items() if axis is not None}
    rows = {name: row for name, row in resolved.rows.items() if row is not None}
    sizes = {name: _axis_size(name, axis, inputs) for name, axis in axes.items()}
    bound: set[str] = set()
    for name, axis in axes.items():
        if isinstance(axis, InputAxis):
            if axis.input in bound:
                raise SweepExpansionError(
                    SweepExpansionCode.INPUT_CONFLICT,
                    f"sweep.axes.{name}.input",
                    f"input {axis.input!r} is already bound by another axis",
                )
            bound.add(axis.input)

    combination = resolved.combination or "cross"
    if not isinstance(combination, ManualCombination):
        return _SweepPlan(axes, rows, sizes, dimensions=_dimensions(sizes, combination))
    if not sizes or not combination.manual:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_COMBINATION,
            "sweep.combination.manual",
            "manual combination requires axes and at least one coordinate",
        )
    for index, coordinate in enumerate(combination.manual):
        if coordinate.keys() != sizes.keys():
            raise SweepExpansionError(
                SweepExpansionCode.INVALID_COMBINATION,
                f"sweep.combination.manual.{index}",
                f"expected axes {list(sizes)}, got {list(coordinate)}",
            )
    return _SweepPlan(axes, rows, sizes, manual=tuple(combination.manual))


def count_sweep(sweep: Sweep | None, *, inputs: SweepInputs | None = None) -> int:
    """Count planned rows from declarations, without sampling or visiting rows."""
    return _plan(sweep, inputs or {}).row_count


def _range_values(spec: RangeValues) -> Iterator[float]:
    start, stop = spec.start, spec.stop
    if not math.isfinite(start) or not math.isfinite(stop):
        raise ValueError("range bounds must be finite")
    yield start
    if spec.count == 1:
        return
    if spec.scale == "log":
        first, last = math.log10(start), math.log10(stop)
        for index in range(1, spec.count - 1):
            yield 10 ** (first + (last - first) * index / (spec.count - 1))
    else:
        for index in range(1, spec.count - 1):
            fraction = index / (spec.count - 1)
            yield start * (1 - fraction) + stop * fraction
    yield stop


def _generated_key_value(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return ["list", [_generated_key_value(item) for item in value]]
    if isinstance(value, dict):
        return ["object", {key: _generated_key_value(item) for key, item in value.items()}]
    if type(value) is int and abs(value) > (1 << 53) - 1:
        return ["wide_integer", str(value)]
    return value


def sweep_value_identity_bytes(value: JsonValue) -> bytes:
    """Compare authored axis values under the generated-coordinate identity rule."""
    return canonical_json_v2_bytes(_generated_key_value(value))


def _generated_choices(
    path: str, values: Sequence[JsonValue] | Iterator[JsonValue]
) -> Iterator[_AxisChoice]:
    for value in values:
        key = "value-" + sha256_bytes(sweep_value_identity_bytes(value))
        yield _AxisChoice(
            key, SweepCoordinate(key, str(value), value), {path: SetValue(value=value)}
        )


def _choices(
    name: str, axis: Axis, inputs: SweepInputs, sampler: SamplerResolver | None
) -> tuple[_AxisChoice, ...]:
    path = f"sweep.axes.{name}"
    try:
        match axis:
            case NamedAxis(values=values):
                choices = tuple(
                    _AxisChoice(key, SweepCoordinate(key, value.label), _patches(value))
                    for key, value in values.items()
                    if value is not None
                )
            case InputAxis(input=role):
                members = inputs[role]
                assert members is not None
                choices = tuple(
                    _AxisChoice(
                        member.key, SweepCoordinate(member.key, member.label), member=member
                    )
                    for member in members
                )
                if any(not choice.key for choice in choices):
                    raise SweepExpansionError(
                        SweepExpansionCode.INPUT_CONFLICT,
                        path,
                        f"input {role!r} has an empty member key",
                    )
            case ListedAxis(path=target, values=listed):
                choices = tuple(_generated_choices(target, listed))
            case RangeAxis(path=target, range=range_values):
                choices = tuple(_generated_choices(target, _range_values(range_values)))
            case SampledAxis(path=target, sampler=sample):
                values = (
                    None
                    if sampler is None
                    else sampler(sample.distribution, sample.count, sample.seed)
                )
                if values is None:
                    raise SweepExpansionError(
                        SweepExpansionCode.UNRESOLVED_SAMPLER,
                        f"{path}.sampler.distribution",
                        f"no deterministic resolver for {sample.distribution!r}",
                    )
                if len(values) != sample.count:
                    raise ValueError(
                        f"sampler returned {len(values)} values, expected {sample.count}"
                    )
                choices = tuple(_generated_choices(target, values))
    except SweepExpansionError:
        raise
    except (ValueError, OverflowError, TypeError) as error:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_GENERATION, path, str(error)
        ) from error
    if len({choice.key for choice in choices}) != len(choices):
        raise SweepExpansionError(
            SweepExpansionCode.DUPLICATE_ROW, path, "duplicate axis value or member key"
        )
    return choices


_PLAIN_VALUE_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]*")


def _row_key(key: str) -> str:
    if _PLAIN_VALUE_ID.fullmatch(key):
        return key
    return "=" + sha256_bytes(canonical_json_v2_bytes(key))


def _expanded(axes: Mapping[str, Axis], selection: Mapping[str, _AxisChoice]) -> ExpandedRow:
    row_id = "--".join(f"{name}={_row_key(selection[name].key)}" for name in sorted(axes))
    coordinates: dict[str, SweepCoordinate] = {}
    patches: dict[str, PatchValue] = {}
    owners: dict[str, str] = {}
    bindings: dict[str, str] = {}
    labels = []
    for name, axis in axes.items():
        choice = selection[name]
        labels.append(f"{name}={choice.coordinate.label or choice.key}")
        member = choice.member
        if isinstance(axis, InputAxis):
            assert member is not None
            bindings[axis.input] = member.key
            carried = member.coordinates
        else:
            carried = {name: choice.coordinate}
        for axis_id, coordinate in carried.items():
            existing = coordinates.get(axis_id)
            if existing is not None and (
                existing.value_id != coordinate.value_id
                or canonical_json_v2_bytes(_generated_key_value(existing.value))
                != canonical_json_v2_bytes(_generated_key_value(coordinate.value))
            ):
                raise SweepExpansionError(
                    SweepExpansionCode.COORDINATE_CONFLICT,
                    f"sweep.axes.{name}",
                    f"row {row_id!r} carries conflicting values for axis {axis_id!r}",
                )
            coordinates.setdefault(axis_id, coordinate)

        for path, patch in choice.patches.items():
            for prior_path, prior_axis in owners.items():
                if prior_axis != name and _paths_overlap(path, prior_path):
                    raise SweepExpansionError(
                        SweepExpansionCode.PATCH_CONFLICT,
                        f"sweep.axes.{name}",
                        f"row {row_id!r} patches {path!r}, overlapping {prior_path!r} from axis {prior_axis!r}",
                    )
            owners[path] = name
            patches[path] = patch
    return ExpandedRow(row_id, ", ".join(labels), coordinates, patches, bindings)


def expand_sweep(
    sweep: Sweep | None,
    *,
    inputs: SweepInputs | None = None,
    sampler: SamplerResolver | None = None,
    max_rows: int = DEFAULT_SWEEP_LIMIT,
    max_axis_values: int = DEFAULT_SWEEP_LIMIT,
) -> tuple[ExpandedRow, ...]:
    """Expand explicit rows and axis combinations without compiling payloads.

    Pass a resolved step's sweep, or a project's standalone sweep. An input of
    None has an unknown count, while an empty member sequence has a known count
    of zero. The sampler resolver returns deterministic, ordered JSON values.
    """
    known_inputs = inputs or {}
    plan = _plan(sweep, known_inputs)
    if plan.row_count > max_rows:
        raise SweepExpansionError(
            SweepExpansionCode.ROW_BUDGET,
            "sweep",
            f"{plan.row_count} rows exceeds the limit of {max_rows}",
        )
    if not plan.axes and not plan.rows:
        return (ExpandedRow(DEFAULT_SWEEP_ROW, None, {}, {}, {}),)
    rows = [ExpandedRow(name, row.label, {}, _patches(row), {}) for name, row in plan.rows.items()]
    if not plan.generated_count:
        return tuple(rows)
    for name, size in plan.sizes.items():
        if size > max_axis_values:
            raise SweepExpansionError(
                SweepExpansionCode.AXIS_BUDGET,
                f"sweep.axes.{name}",
                f"{size} values exceeds the axis limit of {max_axis_values}",
            )
    choices = {
        name: _choices(name, axis, known_inputs, sampler) for name, axis in plan.axes.items()
    }
    ids = set(plan.rows)
    for selection in plan.selections(choices):
        row = _expanded(plan.axes, selection)
        if row.id in ids:
            raise SweepExpansionError(
                SweepExpansionCode.DUPLICATE_ROW, "sweep", f"duplicate row id {row.id!r}"
            )
        ids.add(row.id)
        rows.append(row)
    return tuple(rows)
