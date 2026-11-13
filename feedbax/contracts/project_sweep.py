"""Authored sweep grammar and static inheritance for project steps."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from fractions import Fraction
from functools import lru_cache
from itertools import product
import math
import re
from typing import Annotated, Literal, TypeVar

from pydantic import Field, JsonValue, PrivateAttr, TypeAdapter, model_validator

from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.expression_values import DrawValue
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.model_address import FieldElement, ModelAddress
from feedbax.contracts.model_attachment import SetField
from feedbax.contracts.request_refs import UpdateCadence, UpdateRule
from feedbax.contracts.project_inputs import (
    CheckpointRule,
    NonemptyText,
    ProjectName,
    StepReference,
    SubsetSelectionError,
    SubsetTerms,
    select_static_rows,
)
type PatchPath = Annotated[str, Field(pattern=r"^[^.]+(?:\.[^.]+)*$")]
type SweepTarget = Annotated[str, Field(pattern=r"^(?:(?:steps\.[A-Za-z0-9_][A-Za-z0-9_-]*\.)?\{[\s\S]+\}|[^.]+(?:\.[^.]+)*)$")]
type IntegerSite = Callable[[str], bool]


class SetValue(StrictModel):
    value: JsonValue
    op: Literal["add", "replace"] = "replace"


class RemoveValue(StrictModel):
    op: Literal["remove"]


type PatchValue = SetValue | RemoveValue


type CheckpointRulePath = Literal['{"field":"checkpoint","input":"subject"}']
CHECKPOINT_RULE_TARGET: CheckpointRulePath = '{"field":"checkpoint","input":"subject"}'
_CHECKPOINT_RULE = TypeAdapter(CheckpointRule)


def checkpoint_rule_write(subject: object, write: PatchValue) -> StepReference:
    if not isinstance(subject, StepReference):
        raise ValueError("A checkpoint rule edit requires a Train step reference")
    if not isinstance(write, SetValue):
        raise ValueError("A checkpoint rule edit requires a rule")
    return subject.model_copy(update={"checkpoint": _CHECKPOINT_RULE.validate_python(write.value)})


@dataclass(frozen=True)
class ModelFieldTarget:
    address: ModelAddress
    step: str | None = None


@lru_cache(maxsize=4096)
def model_field_target(path: str) -> ModelFieldTarget | None:
    if path == CHECKPOINT_RULE_TARGET:
        return None
    step = None
    if path.startswith("steps.") and ".{" in path:
        scope, _, path = path.partition(".{")
        step = scope.removeprefix("steps.")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", step):
            raise ValueError("A scoped Model field target must name one step")
        path = "{" + path
    elif not path.startswith("{"):
        return None
    address = ModelAddress.from_text(path)
    if not isinstance(address.element, FieldElement) or any(
        not isinstance(part, str) for part in address.element.path
    ):
        raise ValueError("A Model sweep target must name a field using named parts")
    return ModelFieldTarget(address, step)


def _target_prefix(parent: str, child: str) -> bool:
    if parent == child:
        return True
    try:
        owner = model_field_target(parent)
        target = model_field_target(child)
    except ValueError:
        return False  # The addressed write reports its own invalid target.
    if owner is None:
        return target is None and child.startswith(parent + ".")
    if target is None or owner.step != target.step:
        return False
    owner, target = owner.address, target.address
    return (
        owner.document == target.document and owner.composites == target.composites
        and owner.element.parent == target.element.parent
        and owner.element.path == target.element.path[:len(owner.element.path)]
    )


def _paths_overlap(left: str, right: str) -> bool:
    return _target_prefix(left, right) or _target_prefix(right, left)


def sweep_patch(path: str, patch: PatchValue) -> OverridePatch | SetField:
    if path == CHECKPOINT_RULE_TARGET:
        raise ValueError("Checkpoint rule writes require a step-level Evaluate subject")
    target = model_field_target(path)
    if target is not None:
        if not isinstance(patch, SetValue):
            raise ValueError("A Model field sweep requires a value")
        return SetField(op="set_field", address=target.address, value=patch.value)
    if isinstance(patch, SetValue):
        return OverridePatch(path=path, op=patch.op, value=patch.value)
    return OverridePatch(path=path, op="remove")


def sweep_patches(patches: Mapping[str, PatchValue | None]) -> list[OverridePatch]:
    """Convert payload-only patches for callers without a Model resolver."""
    writes = []
    for path, value in patches.items():
        if value is None:
            continue
        patch = sweep_patch(path, value)
        if isinstance(patch, SetField):
            raise ValueError("Addressed Model patches require a Model resolver")
        writes.append(patch)
    return writes


class PatchSet(StrictModel):
    patches: dict[SweepTarget, PatchValue | None] = Field(default_factory=dict)
    _patch_authors: dict[str, str] = PrivateAttr(default_factory=dict)


class AxisValue(PatchSet):
    label: str | None = None
    value: JsonValue = None


class Row(AxisValue):
    source: ProjectName | None = Field(default=None, alias="from")
    _patch_sources: dict[str, str] = PrivateAttr(default_factory=dict)


class SweepException(PatchSet):
    where: SubsetTerms | None = None
    exclude: bool = False


class NamedAxis(StrictModel):
    values: dict[ProjectName, AxisValue | None]


class InputAxis(StrictModel):
    input: ProjectName


class GeneratedAxis(StrictModel):
    path: SweepTarget
    _author: str | None = PrivateAttr(default=None)


class ListedAxis(GeneratedAxis):
    values: list[JsonValue] = Field(min_length=1)


class RangeValues(StrictModel):
    start: int | float
    stop: int | float
    count: int = Field(ge=1)
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def _check_log_range(self) -> RangeValues:
        if self.scale == "log" and min(self.start, self.stop) <= 0:
            raise ValueError("a logarithmic range requires positive bounds")
        return self


class RangeAxis(GeneratedAxis):
    range: RangeValues


class SampleValues(StrictModel):
    distribution: DrawValue | ProjectName
    count: int = Field(ge=1)
    seed: int


class SampledAxis(GeneratedAxis):
    sampler: SampleValues
    draws: dict[ProjectName, JsonValue] | None = None


class CheckpointRuleAxis(StrictModel):
    path: CheckpointRulePath = CHECKPOINT_RULE_TARGET
    checkpoint_rule: UpdateRule
    _author: str | None = PrivateAttr(default=None)


type Axis = NamedAxis | InputAxis | ListedAxis | RangeAxis | SampledAxis | CheckpointRuleAxis


class ManualCombination(StrictModel):
    manual: list[dict[ProjectName, NonemptyText]]


class AxisGroup(StrictModel):
    axes: list[ProjectName] = Field(min_length=1)
    mode: Literal["cross", "zip"]


class GroupCombination(StrictModel):
    groups: dict[ProjectName, AxisGroup]


type Combination = Literal["cross", "zip"] | ManualCombination | GroupCombination


class Sweep(StrictModel):
    axes: dict[ProjectName, Axis | None] = Field(default_factory=dict)
    rows: dict[ProjectName, Row | None] = Field(default_factory=dict)
    exceptions: dict[ProjectName, SweepException | None] = Field(default_factory=dict)
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


def _merge_patchset[T: PatchSet](parent: PatchSet | None, child: T, author: str | None) -> T:
    patches = merge_named(_patches(parent) if parent is not None else {}, child.patches)
    authors = dict(parent._patch_authors) if parent is not None else {}
    for path, patch in child.patches.items():
        writer = author if author is not None else child._patch_authors.get(path)
        if patch is not None and writer is not None:
            authors[path] = writer
        else:
            authors.pop(path, None)
    result = child.model_copy(update={"patches": patches})
    result._patch_authors = authors
    return result


def _merge_value[T: AxisValue](parent: AxisValue | None, child: T, author: str | None) -> T:
    merged = _merge_patchset(parent, child, author)
    if parent is not None:
        for name in ("label", "value"):
            if name not in child.model_fields_set:
                setattr(merged, name, getattr(parent, name))
    return merged


def _compose_writes(
    parent: Mapping[str, PatchValue], child: Mapping[str, PatchValue]
) -> dict[str, PatchValue]:
    """Represent two ordered write sets as one patch per path."""
    combined = dict(parent)
    for path, write in child.items():
        previous = combined.pop(path, None)
        for descendant in tuple(combined):
            if _target_prefix(path, descendant):
                del combined[descendant]
        if isinstance(previous, SetValue) and previous.op == "add":
            if isinstance(write, RemoveValue):
                continue
            write = write.model_copy(update={"op": "add"})
        elif isinstance(previous, RemoveValue) and isinstance(write, SetValue):
            write = write.model_copy(update={"op": "replace"})
        combined[path] = write
    return combined


class SweepInheritanceError(ValueError):
    def __init__(self, row: str, source: str):
        self.row = row
        self.source = source
        super().__init__(f"row {row!r} derives from missing or later row {source!r}")


def inherit_sweep(
    parent: Sweep | None, child: Sweep | None, *, author: str | None = None
) -> Sweep | None:
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
                    values[key] = _merge_value(values.get(key), value, author)
            axes[name] = NamedAxis(values=values)
        elif isinstance(axis, (GeneratedAxis, CheckpointRuleAxis)) and author is not None:
            generated = axis.model_copy()
            generated._author = author
            axes[name] = generated

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
        prior = sources[source] if source is not None else rows.get(name)
        combined = _merge_value(prior, row, author)
        if source is not None and prior is not None:
            inherited = _patches(prior)
            for path, patch in row.patches.items():
                if patch is None:
                    inherited.pop(path, None)
            combined.patches = _compose_writes(inherited, _patches(row))
        resolved = combined.model_copy(update={"source": None})
        origins = (
            {path: prior._patch_sources.get(path, source or name) for path in prior.patches}
            if prior is not None
            else {}
        )
        for path in row.patches:
            origins[path] = row._patch_sources.get(path, name)
        resolved._patch_sources = {path: origins.get(path, name) for path in resolved.patches}
        resolved._patch_authors = {
            path: writer
            for path, writer in resolved._patch_authors.items()
            if path in resolved.patches
        }
        rows[name] = sources[name] = resolved

    exceptions = {name: item for name, item in previous.exceptions.items() if item is not None}
    for name, item in child.exceptions.items():
        if item is None:
            exceptions.pop(name, None)
            continue
        prior = exceptions.get(name)
        merged = _merge_patchset(prior, item, author)
        if prior is not None:
            for field_name in ("where", "exclude"):
                if field_name not in item.model_fields_set:
                    setattr(merged, field_name, getattr(prior, field_name))
        exceptions[name] = merged

    return Sweep(
        axes=axes,
        rows=rows,
        exceptions=exceptions,
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
    MISSING_DRAWS = "missing_draws"
    INVALID_GENERATION = "invalid_generation"
    PATCH_CONFLICT = "patch_conflict"
    COORDINATE_CONFLICT = "coordinate_conflict"
    DUPLICATE_ROW = "duplicate_row"
    INVALID_ROW_SOURCE = "invalid_row_source"
    INVALID_EXCEPTION = "invalid_exception"
    ROW_BUDGET = "row_budget"
    AXIS_BUDGET = "axis_budget"


class SweepExpansionError(ValueError):
    """A sweep refusal with a path suitable for an authored diagnostic."""

    def __init__(self, code: SweepExpansionCode, path: str, detail: str, *, target: str | None = None):
        self.code = code
        self.path = path
        self.detail = detail
        self.target = target
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
    checkpoint_horizon: int | None = None


type SweepInputs = Mapping[str, Sequence[SweepInputMember] | None]
type PatchOrigin = tuple[Literal["sweep_value"], str, str] | tuple[Literal["exception", "row"], str]


@dataclass(frozen=True)
class ExpandedRow:
    id: str
    label: str | None
    coordinates: Mapping[str, SweepCoordinate]
    patches: Mapping[str, PatchValue]
    input_keys: Mapping[str, str]
    patch_origins: Mapping[str, PatchOrigin] = field(default_factory=dict)
    patch_authors: Mapping[str, str] = field(default_factory=dict)
    axis_value_ids: Mapping[str, str] = field(default_factory=dict)

    def inherited_id(self, axes: tuple[str, ...]) -> str | None:
        """Project generated choices onto the inherited axes, or keep an authored row ID."""
        if not self.axis_value_ids:
            return self.id
        if not axes or any(name not in self.axis_value_ids for name in axes):
            return None
        return _generated_row_id({name: self.axis_value_ids[name] for name in axes})


DEFAULT_SWEEP_ROW = "default"
DEFAULT_SWEEP_LIMIT = 10_000


@dataclass(frozen=True)
class _AxisChoice:
    key: str
    coordinate: SweepCoordinate
    patches: Mapping[str, PatchValue] = field(default_factory=dict)
    member: SweepInputMember | None = None
    patch_authors: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _SweepPlan:
    axes: Mapping[str, Axis]
    rows: Mapping[str, Row]
    sizes: Mapping[str, int]
    exceptions: Mapping[str, SweepException]
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


def _patches(value: PatchSet) -> dict[str, PatchValue]:
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
        case SampledAxis():
            return len(_recorded_draws(name, axis))
        case CheckpointRuleAxis():
            raise SweepExpansionError(
                SweepExpansionCode.INVALID_GENERATION, f"sweep.axes.{name}",
                "checkpoint rule axes require a selected Evaluate subject",
            )


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


def _resolved_sweep(sweep: Sweep | None) -> Sweep:
    try:
        return inherit_sweep(None, sweep) or Sweep()
    except SweepInheritanceError as error:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_ROW_SOURCE, f"sweep.rows.{error.row}.from", str(error)
        ) from error


def _plan(sweep: Sweep | None, inputs: SweepInputs) -> _SweepPlan:
    resolved = _resolved_sweep(sweep)
    axes = {name: axis for name, axis in resolved.axes.items() if axis is not None}
    rows = {name: row for name, row in resolved.rows.items() if row is not None}
    combination = resolved.combination or "cross"
    empty_manual = isinstance(combination, ManualCombination) and not combination.manual
    sizes = {
        name: 0 if empty_manual and isinstance(axis, InputAxis) else _axis_size(name, axis, inputs)
        for name, axis in axes.items()
    }
    exceptions = {name: item for name, item in resolved.exceptions.items() if item is not None}
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

    if not isinstance(combination, ManualCombination):
        return _SweepPlan(axes, rows, sizes, exceptions, dimensions=_dimensions(sizes, combination))
    if not sizes:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_COMBINATION,
            "sweep.combination.manual",
            "manual combination requires at least one axis",
        )
    for index, coordinate in enumerate(combination.manual):
        if coordinate.keys() != sizes.keys():
            raise SweepExpansionError(
                SweepExpansionCode.INVALID_COMBINATION,
                f"sweep.combination.manual.{index}",
                f"expected axes {list(sizes)}, got {list(coordinate)}",
            )
    return _SweepPlan(axes, rows, sizes, exceptions, manual=tuple(combination.manual))


def count_sweep(
    sweep: Sweep | None,
    *,
    inputs: SweepInputs | None = None,
    max_rows: int = DEFAULT_SWEEP_LIMIT,
    max_axis_values: int = DEFAULT_SWEEP_LIMIT,
    integer_site: IntegerSite | None = None,
) -> int:
    """Count final rows, using bounded expansion when exceptions select rows."""
    known_inputs = inputs or {}
    plan = _plan(sweep, known_inputs)
    if plan.exceptions:
        return len(_expand_plan(plan, known_inputs, max_rows, max_axis_values, integer_site))
    return plan.row_count


def _recorded_draws(name: str, axis: SampledAxis) -> Mapping[str, JsonValue]:
    path = f"sweep.axes.{name}.draws"
    if axis.draws is None:
        raise SweepExpansionError(
            SweepExpansionCode.MISSING_DRAWS,
            path,
            f"sampler {axis.sampler.distribution!r} has no recorded draws",
        )
    if len(axis.draws) != axis.sampler.count:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_GENERATION,
            path,
            f"expected {axis.sampler.count} recorded draws, got {len(axis.draws)}",
        )
    return axis.draws


def _range_values(spec: RangeValues) -> Iterator[float]:
    start, stop = float(spec.start), float(spec.stop)
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


def _integer_range(spec: RangeValues) -> tuple[int, ...]:
    if spec.scale == "linear":
        start, stop = Fraction(str(spec.start)), Fraction(str(spec.stop))
        if spec.count == 1:
            values = [start]
        else:
            stride = (stop - start) / (spec.count - 1)
            values = [start + index * stride for index in range(spec.count)]
        nonwhole = [str(value) for value in values if value.denominator != 1]
        members = tuple(int(value) for value in values)
    else:
        values = list(_range_values(spec))
        members = tuple(round(value) for value in values)
        nonwhole = [
            str(value) for value, whole in zip(values, members, strict=True)
            if not math.isclose(value, whole, rel_tol=1e-9)
        ]
    if nonwhole:
        raise ValueError("Integer range has nonwhole members: " + ", ".join(nonwhole))
    return members


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
    axis: GeneratedAxis, values: Iterator[tuple[str | None, JsonValue]]
) -> Iterator[_AxisChoice]:
    authors = {axis.path: axis._author} if axis._author is not None else {}
    for key, value in values:
        if key is None:
            key = "value-" + sha256_bytes(sweep_value_identity_bytes(value))
        yield _AxisChoice(
            key,
            SweepCoordinate(key, str(value), value),
            {axis.path: SetValue(value=value)},
            patch_authors=authors,
        )


def _integer_member(value: JsonValue) -> int:
    if type(value) is int:
        return value
    if type(value) is float and value.is_integer():
        return int(value)
    raise ValueError(f"Integer sweep has a nonwhole member: {value!r}")


def _choices(
    name: str, axis: Axis, inputs: SweepInputs, integer_site: IntegerSite | None = None,
) -> tuple[_AxisChoice, ...]:
    path = f"sweep.axes.{name}"
    target = None
    try:
        match axis:
            case NamedAxis(values=values):
                choices = tuple(
                    _AxisChoice(
                        key,
                        SweepCoordinate(key, value.label, value.value),
                        _patches(value),
                        patch_authors=value._patch_authors,
                    )
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
            case ListedAxis(values=listed):
                target = axis.path
                values = map(_integer_member, listed) if integer_site and integer_site(target) else listed
                choices = tuple(_generated_choices(axis, ((None, value) for value in values)))
            case RangeAxis(range=range_values):
                target = axis.path
                if integer_site and integer_site(axis.path):
                    values = _integer_range(range_values)
                else:
                    values = _range_values(range_values)
                choices = tuple(
                    _generated_choices(axis, ((None, value) for value in values))
                )
            case SampledAxis():
                target = axis.path
                draws = _recorded_draws(name, axis)
                values = draws.items()
                if integer_site and integer_site(target):
                    values = ((key, _integer_member(value)) for key, value in values)
                choices = tuple(_generated_choices(axis, iter(values)))
    except SweepExpansionError:
        raise
    except (ValueError, OverflowError, TypeError) as error:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_GENERATION, path, str(error), target=target,
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


def _generated_row_id(values: Mapping[str, str]) -> str:
    return "--".join(f"{name}={_row_key(values[name])}" for name in sorted(values))


def _expanded(axes: Mapping[str, Axis], selection: Mapping[str, _AxisChoice]) -> ExpandedRow:
    axis_value_ids = {name: choice.key for name, choice in selection.items()}
    row_id = _generated_row_id(axis_value_ids)
    coordinates: dict[str, SweepCoordinate] = {}
    patches: dict[str, PatchValue] = {}
    origins: dict[str, PatchOrigin] = {}
    authors: dict[str, str] = {}
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
            for prior_path, origin in origins.items():
                prior_axis = origin[1]
                if prior_axis != name and _paths_overlap(path, prior_path):
                    raise SweepExpansionError(
                        SweepExpansionCode.PATCH_CONFLICT,
                        f"sweep.axes.{name}",
                        f"row {row_id!r} patches {path!r}, overlapping {prior_path!r} from axis {prior_axis!r}",
                    )
            origins[path] = ("sweep_value", name, choice.key)
            patches[path] = patch
        authors.update(choice.patch_authors)
    return ExpandedRow(
        row_id, ", ".join(labels), coordinates, patches, bindings, origins, authors, axis_value_ids
    )


def _select_exception_rows(
    name: str, exception: SweepException, rows: tuple[ExpandedRow, ...]
) -> tuple[ExpandedRow, ...]:
    path = f"sweep.exceptions.{name}"
    if exception.where is None:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_EXCEPTION, f"{path}.where", "exception needs a selector"
        )
    if exception.exclude and exception.patches:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_EXCEPTION, path, "exclusion cannot also patch a row"
        )
    try:
        return select_static_rows(
            exception.where,
            rows,
            key=lambda row: row.id,
            coordinates=lambda row: {
                axis: value.value_id for axis, value in row.coordinates.items()
            },
        )
    except SubsetSelectionError as error:
        term = f".{error.path}" if error.path else ""
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_EXCEPTION, f"{path}.where{term}", error.detail
        ) from error


def _apply_exceptions(
    rows: tuple[ExpandedRow, ...], exceptions: Mapping[str, SweepException]
) -> tuple[ExpandedRow, ...]:
    if not exceptions:
        return rows
    kept = {row.id: row for row in rows}
    for name, exception in exceptions.items():
        path = f"sweep.exceptions.{name}"
        selected = _select_exception_rows(name, exception, rows)
        writes = _patches(exception)
        for original in selected:
            row = kept.get(original.id)
            if row is None:
                continue
            if exception.exclude:
                del kept[row.id]
                continue
            origins = dict(row.patch_origins)
            authors = dict(row.patch_authors)
            for target in writes:
                for prior, origin in row.patch_origins.items():
                    if origin[0] == "sweep_value" and _paths_overlap(prior, target):
                        raise SweepExpansionError(
                            SweepExpansionCode.PATCH_CONFLICT,
                            f"{path}.patches.{target}",
                            f"row {row.id!r} already writes {prior!r} from sweep value {origin[1]}={origin[2]}",
                        )
                origins[target] = ("exception", name)
                if target in exception._patch_authors:
                    authors[target] = exception._patch_authors[target]
                else:
                    authors.pop(target, None)
            patches = _compose_writes(row.patches, writes)
            kept[row.id] = replace(
                row,
                patches=patches,
                patch_origins={target: origins[target] for target in patches},
                patch_authors={
                    target: writer for target, writer in authors.items() if target in patches
                },
            )
    return tuple(kept.values())


def _check_row_budget(plan: _SweepPlan, max_rows: int) -> None:
    if plan.row_count > max_rows:
        raise SweepExpansionError(
            SweepExpansionCode.ROW_BUDGET,
            "sweep",
            f"{plan.row_count} rows exceeds the limit of {max_rows}",
        )


def _check_axis_budget(name: str, size: int, limit: int) -> None:
    if size > limit:
        raise SweepExpansionError(
            SweepExpansionCode.AXIS_BUDGET,
            f"sweep.axes.{name}",
            f"{size} values exceeds the axis limit of {limit}",
        )


def _planned_choices(
    plan: _SweepPlan, inputs: SweepInputs, max_axis_values: int, integer_site: IntegerSite | None,
) -> dict[str, tuple[_AxisChoice, ...]]:
    for name, size in plan.sizes.items():
        _check_axis_budget(name, size, max_axis_values)
    return {name: _choices(name, axis, inputs, integer_site) for name, axis in plan.axes.items()}


def _explicit_rows(rows: Mapping[str, Row]) -> Iterator[ExpandedRow]:
    for name, row in rows.items():
        patches = _patches(row)
        origins: dict[str, PatchOrigin] = {
            path: ("row", row._patch_sources[path]) for path in patches
        }
        yield ExpandedRow(name, row.label, {}, patches, {}, origins, row._patch_authors)


def _generated_rows(
    plan: _SweepPlan, inputs: SweepInputs, max_axis_values: int, integer_site: IntegerSite | None = None,
) -> Iterator[tuple[dict[str, _AxisChoice], ExpandedRow]]:
    if not plan.generated_count:
        return
    choices = _planned_choices(plan, inputs, max_axis_values, integer_site)
    ids = set(plan.rows)
    for selection in plan.selections(choices):
        row = _expanded(plan.axes, selection)
        if row.id in ids:
            raise SweepExpansionError(
                SweepExpansionCode.DUPLICATE_ROW, "sweep", f"duplicate row id {row.id!r}"
            )
        ids.add(row.id)
        yield selection, row


def _expand_plan(
    plan: _SweepPlan, inputs: SweepInputs, max_rows: int, max_axis_values: int,
    integer_site: IntegerSite | None = None,
) -> tuple[ExpandedRow, ...]:
    _check_row_budget(plan, max_rows)
    rows = list(_explicit_rows(plan.rows))
    if not plan.axes and not plan.rows:
        rows.append(ExpandedRow(DEFAULT_SWEEP_ROW, None, {}, {}, {}))
    rows.extend(row for _, row in _generated_rows(plan, inputs, max_axis_values, integer_site))
    return _apply_exceptions(tuple(rows), plan.exceptions)


def expand_sweep(
    sweep: Sweep | None,
    *,
    inputs: SweepInputs | None = None,
    max_rows: int = DEFAULT_SWEEP_LIMIT,
    max_axis_values: int = DEFAULT_SWEEP_LIMIT,
    integer_site: IntegerSite | None = None,
    checkpoint_subject: bool = False,
) -> tuple[ExpandedRow, ...]:
    """Expand the declared sweep using recorded draws and known input members.

    None has an unknown input member count. An empty sequence is a known count
    of zero. Expansion never draws new samples or mutates an authored rule.
    """
    known_inputs = inputs or {}
    if checkpoint_subject and sweep is not None:
        resolved = _resolved_sweep(sweep)
        checkpoints = [(name, axis) for name, axis in resolved.axes.items() if isinstance(axis, CheckpointRuleAxis)]
        if checkpoints:
            if len(checkpoints) != 1:
                raise SweepExpansionError(
                    SweepExpansionCode.INVALID_GENERATION, "sweep.axes", "an Evaluate step has one checkpoint rule",
                )
            (name, axis), = checkpoints
            return _expand_checkpoint_axis(resolved, name, axis, known_inputs, max_rows, max_axis_values, integer_site)
    return _expand_plan(_plan(sweep, known_inputs), known_inputs, max_rows, max_axis_values, integer_site)


def _expand_checkpoint_axis(
    sweep: Sweep, name: str, axis: CheckpointRuleAxis, inputs: SweepInputs,
    max_rows: int, max_axis_values: int, integer_site: IntegerSite | None,
) -> tuple[ExpandedRow, ...]:
    from feedbax.checkpoint_coordinates import finite_coordinates

    path = f"sweep.axes.{name}"
    if sweep.combination not in (None, "cross") or any(sweep.rows.values()):
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_COMBINATION, path, "checkpoint rules cross generated rows",
        )
    base = sweep.model_copy(update={
        "axes": {key: value for key, value in sweep.axes.items() if key != name}, "exceptions": {},
    })
    rows = _expand_plan(_plan(base, inputs), inputs, max_rows, max_axis_values, integer_site)
    subjects = {member.key: member for member in inputs.get("subject") or ()}
    only_subject = next(iter(subjects.values())) if len(subjects) == 1 else None
    choices = {}
    rows_and_updates = []
    count = 0
    for row in rows:
        subject = subjects.get(row.input_keys.get("subject")) if "subject" in row.input_keys else only_subject
        if subject is None:
            raise SweepExpansionError(
                SweepExpansionCode.INPUT_CONFLICT, path, "select one Train row or bind the subject axis",
            )
        if subject.key not in choices:
            rule = axis.checkpoint_rule
            try:
                if isinstance(rule, UpdateCadence):
                    if subject.checkpoint_horizon is None:
                        raise ValueError("an update cadence requires a planned Train definition")
                    updates = rule.coordinates(subject.checkpoint_horizon)
                else:
                    updates = finite_coordinates(tuple(point.at for point in rule))
            except ValueError as error:
                raise SweepExpansionError(SweepExpansionCode.INVALID_GENERATION, path, str(error)) from error
            _check_axis_budget(name, len(updates), max_axis_values)
            choices[subject.key] = updates
        updates = choices[subject.key]
        if name in row.coordinates:
            raise SweepExpansionError(
                SweepExpansionCode.COORDINATE_CONFLICT, path, "the subject already carries this axis name",
            )
        if axis.path in row.patches:
            raise SweepExpansionError(
                SweepExpansionCode.PATCH_CONFLICT, path, "another axis writes the checkpoint rule",
            )
        count += len(updates)
        rows_and_updates.append((row, updates))
    if count > max_rows:
        raise SweepExpansionError(
            SweepExpansionCode.ROW_BUDGET, path, f"{count} rows exceeds the limit of {max_rows}",
        )
    result = []
    for row, updates in rows_and_updates:
        for update in updates:
            key = str(update)
            keys = {**row.axis_value_ids, name: key}
            authors = dict(row.patch_authors)
            if axis._author is not None:
                authors[axis.path] = axis._author
            result.append(replace(
                row, id=_generated_row_id(keys), axis_value_ids=keys,
                label=", ".join(part for part in (row.label, f"{name}={key}") if part),
                coordinates={**row.coordinates, name: SweepCoordinate(key, key, update)},
                patches={**row.patches, axis.path: SetValue(value=[{"at": update}])},
                patch_origins={**row.patch_origins, axis.path: ("sweep_value", name, key)},
                patch_authors=authors,
            ))
    return _apply_exceptions(tuple(result), {key: value for key, value in sweep.exceptions.items() if value is not None})


def authored_row(sweep: Sweep | None, row_id: str) -> ExpandedRow | None:
    """Select one explicitly authored row, applying the enclosing sweep's exceptions."""
    resolved = _resolved_sweep(sweep)
    rows = {name: row for name, row in resolved.rows.items() if row is not None}
    if row_id not in rows:
        return None
    exceptions = {name: item for name, item in resolved.exceptions.items() if item is not None}
    needs_generated = any(
        where.axes or any(name not in rows for name in where.rows or ())
        for item in exceptions.values()
        if (where := item.where) is not None
    )
    expanded = (
        expand_sweep(resolved)
        if needs_generated
        else _apply_exceptions(tuple(_explicit_rows(rows)), exceptions)
    )
    return next((row for row in expanded if row.id == row_id), None)


def axis_to_list(
    axis: ListedAxis | RangeAxis | SampledAxis, *, max_axis_values: int = DEFAULT_SWEEP_LIMIT,
    integer_site: IntegerSite | None = None,
) -> NamedAxis:
    """Materialize an authored rule as an editable, identity-preserving value map."""
    _check_axis_budget("values", _axis_size("values", axis, {}), max_axis_values)
    values = {}
    for choice in _choices("values", axis, {}, integer_site):
        value = AxisValue(
            label=choice.coordinate.label,
            value=choice.coordinate.value,
            patches=dict(choice.patches),
        )
        value._patch_authors = dict(choice.patch_authors)
        values[choice.key] = value
    return NamedAxis(values=values)


def sweep_to_rows(
    sweep: Sweep,
    *,
    inputs: SweepInputs | None = None,
    max_rows: int = DEFAULT_SWEEP_LIMIT,
    max_axis_values: int = DEFAULT_SWEEP_LIMIT,
    integer_site: IntegerSite | None = None,
) -> Sweep:
    """Pin surviving combinations by axis value ID, preserving named patch exceptions."""
    known_inputs = inputs or {}
    plan = _plan(sweep, known_inputs)
    _check_row_budget(plan, max_rows)
    if not plan.axes:
        return sweep
    explicit = tuple(_explicit_rows(plan.rows))
    generated = tuple(_generated_rows(plan, known_inputs, max_axis_values, integer_site))
    candidates = explicit + tuple(row for _, row in generated)
    survivors = {row.id for row in _apply_exceptions(candidates, plan.exceptions)}
    retained = [(selection, row) for selection, row in generated if row.id in survivors]
    retained_candidates = explicit + tuple(row for _, row in retained)
    retained_ids = {row.id for row in retained_candidates}

    exceptions = dict(sweep.exceptions)
    for name, exception in plan.exceptions.items():
        try:
            _select_exception_rows(name, exception, retained_candidates)
        except SweepExpansionError:
            members = _select_exception_rows(name, exception, candidates)
            exceptions[name] = exception.model_copy(
                update={
                    "where": SubsetTerms(
                        rows=[row.id for row in members if row.id in retained_ids]
                    ),
                }
            )

    manual = [{name: selection[name].key for name in plan.axes} for selection, _ in retained]
    return sweep.model_copy(
        update={
            "combination": ManualCombination(manual=manual),
            "exceptions": exceptions,
        }
    )
