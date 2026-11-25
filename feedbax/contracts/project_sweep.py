"""Authored sweep grammar and static inheritance for project steps."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from fractions import Fraction
from functools import lru_cache
from graphlib import CycleError, TopologicalSorter
from itertools import product
import math
import re
from typing import Annotated, Literal, TypeVar

from pydantic import Field, JsonValue, PrivateAttr, TypeAdapter, model_validator

from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.expression_values import DrawValue, Expression, TextSpan, WeightPart
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.model_address import FieldElement, ModelAddress
from feedbax.contracts.model_attachment import SetField
from feedbax.contracts.request_refs import UpdateCadence, UpdateRule, axis_values_id
from feedbax.contracts.project_inputs import (
    CheckpointRule,
    NonemptyText,
    ProjectName,
    RowKey,
    StepReference,
    SweepExceptionTerms,
    select_sweep_exception_rows,
)
from feedbax.contracts.run_rule import RuleRefusal, rule_names
from feedbax.contracts.task_document import TaskAdditions
type PatchPath = Annotated[str, Field(pattern=r"^[^.]+(?:\.[^.]+)*$")]
type SweepTarget = Annotated[str, Field(pattern=r"^(?:(?:(?:steps|documents)\.\??[A-Za-z0-9_][A-Za-z0-9_-]*\.)?\{[\s\S]+\}|[^.]+(?:\.[^.]+)*)$")]
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


def checkpoint_sweep_refusal(subject: object, sweep: Sweep | None, *, project_swept: bool) -> str | None:
    """Why an authored checkpoint rule cannot be moved to an update axis."""
    if project_swept:
        return "Checkpoint sweep edits belong to the step definition."
    if not isinstance(subject, StepReference) or not isinstance(subject.checkpoint, (tuple, UpdateCadence)):
        return "Choose a locally authored checkpoint list or cadence."
    if sweep is not None:
        if sweep.combination not in (None, "cross") or any(sweep.rows.values()):
            return "To Sweep requires a cross of generated rows."
        if sweep.axes.get("update") is not None:
            return "This step already has an update axis."
    return None


@dataclass(frozen=True)
class ModelFieldTarget:
    address: ModelAddress
    scope: tuple[Literal["steps", "documents"], str] | None = None

    @property
    def step(self) -> str | None:
        return self.scope[1] if self.scope and self.scope[0] == "steps" else None

    @property
    def document(self) -> str | None:
        return self.scope[1] if self.scope and self.scope[0] == "documents" else None


@lru_cache(maxsize=4096)
def model_field_target(path: str) -> ModelFieldTarget | None:
    if path == CHECKPOINT_RULE_TARGET:
        return None
    if path.startswith(("steps.?", "documents.?")):
        return None
    scope = None
    if path.startswith(("steps.", "documents.")) and ".{" in path:
        prefix, _, path = path.partition(".{")
        match = re.fullmatch(r"(steps|documents)\.([A-Za-z0-9_][A-Za-z0-9_-]*)", prefix)
        if match is None:
            raise ValueError("A scoped Model field target must name one step or document")
        group, name = match.groups()
        scope = ("steps" if group == "steps" else "documents", name)
        path = "{" + path
    elif not path.startswith("{"):
        return None
    address = ModelAddress.from_text(path)
    if not isinstance(address.element, FieldElement):
        raise ValueError("A Model sweep target must name a field")
    return ModelFieldTarget(address, scope)


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
    if target is None or owner.scope != target.scope:
        return False
    owner, target = owner.address, target.address
    return (
        owner.composites == target.composites
        and owner.element.parent == target.element.parent
        and owner.element.path == target.element.path[:len(owner.element.path)]
    )


def _paths_overlap(left: str, right: str) -> bool:
    return _target_prefix(left, right) or _target_prefix(right, left)


def sweep_patch(path: str, patch: PatchValue) -> OverridePatch | SetField:
    if path.startswith(("steps.?", "documents.?")):
        raise ValueError(f"Unresolved target {path.replace('.?', '.', 1)!r}.")
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


class AxisValue(PatchSet, TaskAdditions):
    label: str | None = None
    value: JsonValue = None


class Row(AxisValue):
    source: ProjectName | None = Field(default=None, alias="from")
    _patch_sources: dict[str, str] = PrivateAttr(default_factory=dict)


class SweepException(PatchSet):
    where: SweepExceptionTerms | None = None
    exclude: bool = False


class AxisCondition(StrictModel):
    """The rows an axis applies to: those a run rule on the step's other axes chooses, among ``rows`` when given.

    ``rows`` names the rows the step has without this axis; an axis conditioned on rows divides its rows after every
    conditioned axis authored before it.
    """

    rule: str = ""
    rows: Annotated[list[RowKey], Field(min_length=1)] | None = Field(default=None, exclude_if=lambda rows: rows is None)

    @model_validator(mode="after")
    def _has_terms(self) -> AxisCondition:
        if not self.rule.strip() and self.rows is None:
            raise ValueError("a condition has a rule or rows")
        return self


class NamedAxis(StrictModel):
    values: dict[ProjectName, AxisValue | None]
    where: AxisCondition | None = None
    optional: bool = Field(default=False, exclude_if=lambda optional: not optional)
    """A row an optional axis applies to is kept as it is, with no value on the axis, and one row per value is added
    beside it."""


class InputAxis(StrictModel):
    input: ProjectName


class GeneratedAxis(StrictModel):
    path: SweepTarget
    where: AxisCondition | None = None
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


class SetAxis(StrictModel):
    """One value per entry of the named Set document, applied to the Model document the step reads."""

    set: ProjectName
    where: AxisCondition | None = None
    optional: bool = Field(default=False, exclude_if=lambda optional: not optional)


class ListedSetTrials(StrictModel):
    kind: Literal["listed"]


class WeightedSetTrials(StrictModel):
    kind: Literal["weighted"]
    weights: dict[ProjectName, WeightPart] = Field(default_factory=dict)
    none: WeightPart = 1.0
    """Unspecified entries receive one part, including entries added to the Set later."""


class SetEntryParts(StrictModel):
    on: WeightPart = 1.0
    off: WeightPart = 1.0


class CooccurringSetTrials(StrictModel):
    kind: Literal["cooccurring"]
    weights: dict[ProjectName, SetEntryParts] = Field(default_factory=dict)


type SetTrialSampling = Annotated[
    ListedSetTrials | WeightedSetTrials | CooccurringSetTrials, Field(discriminator="kind"),
]


class TrialSetField(StrictModel):
    """One sweep axis placed in the step's trials."""

    axis: SetAxis
    sampling: SetTrialSampling
    condition: Expression | None = Field(default=None, exclude_if=lambda value: value is None)


type Axis = NamedAxis | InputAxis | ListedAxis | RangeAxis | SampledAxis | CheckpointRuleAxis | SetAxis


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
            kept = {
                field: getattr(axis if field in axis.model_fields_set or not isinstance(prior, NamedAxis) else prior, field)
                for field in ("where", "optional")
            }
            axes[name] = NamedAxis(values=values, **kept)
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
    INVALID_CONDITION = "invalid_condition"
    ROW_BUDGET = "row_budget"
    AXIS_BUDGET = "axis_budget"
    INVALID_SET = "invalid_set"


class SweepExpansionError(ValueError):
    """A sweep refusal with a path suitable for an authored diagnostic."""

    def __init__(
        self, code: SweepExpansionCode, path: str, detail: str, *,
        target: str | None = None, span: TextSpan | None = None,
    ):
        self.code = code
        self.path = path
        self.detail = detail
        self.target = target
        self.span = span
        super().__init__(f"{path}: {detail}")


@dataclass(frozen=True)
class SweepCoordinate:
    value_id: str
    label: str | None = None
    value: JsonValue = None


@dataclass(frozen=True)
class SweepInputCoordinate(SweepCoordinate):
    """An input axis's value: a member, which carries its own coordinates into the row."""

    carries: tuple[str, ...] = ()


@dataclass(frozen=True)
class SweepInputMember:
    """A member whose slot and key are known at plan time."""

    key: str
    coordinates: Mapping[str, SweepCoordinate] = field(default_factory=dict)
    label: str | None = None
    checkpoint_horizon: int | None = None


type SweepInputs = Mapping[str, Sequence[SweepInputMember] | None]
type SweepSets = Mapping[str, NamedAxis]
"""Each set axis's entries as the inline axis they convert to, by axis name."""
type PatchOrigin = (
    tuple[Literal["sweep_value"], str, str]
    | tuple[Literal["set_entry"], str, str, str]
    | tuple[Literal["exception", "row"], str]
)


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
    input_axes: Mapping[str, SweepInputCoordinate] = field(default_factory=dict)
    """Each input axis's value on this row, its member, which ``coordinates`` holds only as the member's own
    coordinates."""
    task_additions: tuple[TaskAdditions, ...] = ()

    def inherited_id(self, axes: tuple[str, ...]) -> str | None:
        """Project generated choices onto the inherited axes, or keep an authored row ID.

        An inherited axis this row has no value on is absent from the ID, as it is from the ID of a parent row the
        axis does not apply to.
        """
        if not self.axis_value_ids:
            return self.id
        values = {name: self.axis_value_ids[name] for name in axes if name in self.axis_value_ids}
        return axis_values_id(values) if values else None


DEFAULT_SWEEP_ROW = "default"
DEFAULT_SWEEP_LIMIT = 10_000


@dataclass(frozen=True)
class _AxisChoice:
    key: str
    coordinate: SweepCoordinate
    patches: Mapping[str, PatchValue] = field(default_factory=dict)
    member: SweepInputMember | None = None
    patch_authors: Mapping[str, str] = field(default_factory=dict)
    origin: PatchOrigin | None = None
    task_additions: TaskAdditions | None = None


@dataclass(frozen=True)
class _SweepPlan:
    axes: Mapping[str, Axis]
    rows: Mapping[str, Row]
    sizes: Mapping[str, int]
    exceptions: Mapping[str, SweepException]
    sets: SweepSets = field(default_factory=dict)
    dimensions: tuple[_Dimension, ...] = ()
    manual: tuple[Mapping[str, str], ...] | None = None
    conditional: tuple[str, ...] = ()
    """The axes with a condition, each after the axes it is conditioned on. The combination is of the others."""

    @property
    def generated_count(self) -> int:
        """The rows the combination generates, before the conditional axes divide some of them."""
        if self.manual is not None:
            return len(self.manual)
        return math.prod(len(steps) for _, steps in self.dimensions) if self.axes else 0

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
        for steps in product(*(steps for _, steps in self.dimensions)):
            yield {
                name: choices[name][index]
                for (names, _), indices in zip(self.dimensions, steps, strict=True)
                for name, index in zip(names, indices, strict=True)
            }


def _patches(value: PatchSet) -> dict[str, PatchValue]:
    return {path: patch for path, patch in value.patches.items() if patch is not None}


def _axis_size(name: str, axis: Axis, inputs: SweepInputs, sets: SweepSets) -> int:
    match axis:
        case SetAxis(set=document):
            if name not in sets:
                raise SweepExpansionError(
                    SweepExpansionCode.INVALID_SET, f"sweep.axes.{name}.set",
                    f"set {document!r} applies to the Model document of a Train or Evaluate step",
                )
            return _axis_size(name, sets[name], inputs, sets)
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


type _Dimension = tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]
"""Axes that vary together, and the value index of each axis at each step."""


def _label_pairs(path: str, names: Sequence[str], members: Sequence[Sequence[SweepInputMember]]) -> tuple[tuple[int, ...], ...]:
    """Pair the members of zipped inputs whose shared varying labels are equal, one to one."""
    every = [member for group in members for member in group]
    shared = [
        label for label in every[0].coordinates if all(label in member.coordinates for member in every)
        and not all(len({member.coordinates[label].value_id for member in group}) <= 1 for group in members)
    ]
    if not shared and any(len(group) > 1 for group in members):
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_COMBINATION, path, f"zipped inputs {list(names)} share no varying label",
        )
    indices = []
    for name, group in zip(names, members, strict=True):
        keys: dict[tuple[str, ...], int] = {}
        for index, member in enumerate(group):
            key = tuple(member.coordinates[label].value_id for label in shared)
            if keys.setdefault(key, index) != index:
                raise SweepExpansionError(
                    SweepExpansionCode.INVALID_COMBINATION, path,
                    f"input axis {name!r} has several members labelled {dict(zip(shared, key, strict=True))}",
                )
        indices.append(keys)
    if any(keys.keys() != indices[0].keys() for keys in indices[1:]):
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_COMBINATION, path,
            f"zipped inputs {list(names)} do not pair one to one by {shared}",
        )
    return tuple(tuple(keys[key] for keys in indices) for key in indices[0])


def _zip_dimension(
    path: str, names: Sequence[str], sizes: Mapping[str, int], axes: Mapping[str, Axis], inputs: SweepInputs,
) -> _Dimension:
    roles = [axes[name].input for name in names if isinstance(axes[name], InputAxis)]
    if roles:
        if len(roles) != len(names):
            raise SweepExpansionError(
                SweepExpansionCode.INVALID_COMBINATION, path, "zip pairs input axes by label, with input axes only",
            )
        return tuple(names), _label_pairs(path, names, [inputs[role] or () for role in roles])
    lengths = {name: sizes[name] for name in names}
    if len(set(lengths.values())) != 1:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_COMBINATION,
            path,
            f"zip requires equal axis lengths, got {lengths}",
        )
    return tuple(names), tuple((index,) * len(names) for index in range(next(iter(lengths.values()))))


def _cross_dimension(name: str, size: int) -> _Dimension:
    return (name,), tuple((index,) for index in range(size))


def _dimensions(
    sizes: Mapping[str, int], combination: Literal["cross", "zip"] | GroupCombination,
    axes: Mapping[str, Axis], inputs: SweepInputs,
) -> tuple[_Dimension, ...]:
    if not sizes and isinstance(combination, str):
        return ()
    if combination == "cross":
        return tuple(_cross_dimension(name, size) for name, size in sizes.items())
    if combination == "zip":
        return (_zip_dimension("sweep.combination", tuple(sizes), sizes, axes, inputs),)

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
            dimensions.append(_zip_dimension(path, group.axes, sizes, axes, inputs))
        else:
            dimensions.extend(_cross_dimension(name, sizes[name]) for name in group.axes)
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


def _condition(axis: Axis) -> AxisCondition | None:
    return axis.where if isinstance(axis, NamedAxis | GeneratedAxis | SetAxis) else None


def _optional(axis: Axis) -> bool:
    return isinstance(axis, NamedAxis | SetAxis) and axis.optional


def _divides(axis: Axis) -> bool:
    """Whether the axis adds rows to the combination of the others, rather than being one of its factors."""
    return _condition(axis) is not None or _optional(axis)


def _conditional_order(axes: Mapping[str, Axis]) -> tuple[str, ...]:
    """The conditioned and optional axes, each after those it depends on; a condition names other axes only.

    An optional axis without a condition applies to every row, so it follows every such axis authored before it.
    """
    graph: dict[str, set[str]] = {}
    earlier: list[str] = []
    for name, axis in axes.items():
        if not _divides(axis):
            continue
        if (where := _condition(axis)) is None:
            graph[name] = set(earlier)
            earlier.append(name)
            continue
        try:
            names = rule_names(where.rule)
        except RuleRefusal as error:
            raise SweepExpansionError(
                SweepExpansionCode.INVALID_CONDITION, f"sweep.axes.{name}.where.rule", error.detail, span=error.span,
            ) from error
        if name in names:
            raise SweepExpansionError(
                SweepExpansionCode.INVALID_CONDITION, f"sweep.axes.{name}.where.rule",
                f"axis {name!r} is conditioned on itself",
            )
        graph[name] = {other for other in names if other in axes and _divides(axes[other])}
        if where.rows is not None:
            graph[name] |= set(earlier)
        earlier.append(name)
    try:
        return tuple(TopologicalSorter(graph).static_order())
    except CycleError as error:
        cycle = list(dict.fromkeys(error.args[1]))
        named = ", ".join(repr(name) for name in cycle[:-1]) + f" and {cycle[-1]!r}"
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_CONDITION, f"sweep.axes.{cycle[0]}.where",
            f"axes {named} are conditioned on each other",
        ) from error


def _plan(sweep: Sweep | None, inputs: SweepInputs, sets: SweepSets | None = None) -> _SweepPlan:
    resolved = _resolved_sweep(sweep)
    axes = {name: axis for name, axis in resolved.axes.items() if axis is not None}
    rows = {name: row for name, row in resolved.rows.items() if row is not None}
    combination = resolved.combination or "cross"
    empty_manual = isinstance(combination, ManualCombination) and not combination.manual
    conditional = _conditional_order(axes)
    every = {
        name: 0 if empty_manual and isinstance(axis, InputAxis) else _axis_size(name, axis, inputs, sets or {})
        for name, axis in axes.items()
    }
    sizes = {name: size for name, size in every.items() if name not in conditional}
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
        return _SweepPlan(
            axes, rows, every, exceptions, sets or {}, dimensions=_dimensions(sizes, combination, axes, inputs),
            conditional=conditional,
        )
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
    return _SweepPlan(
        axes, rows, every, exceptions, sets or {}, manual=tuple(combination.manual), conditional=conditional,
    )


def count_sweep(
    sweep: Sweep | None,
    *,
    inputs: SweepInputs | None = None,
    max_rows: int = DEFAULT_SWEEP_LIMIT,
    max_axis_values: int = DEFAULT_SWEEP_LIMIT,
    integer_site: IntegerSite | None = None,
    sets: SweepSets | None = None,
) -> int:
    """Count final rows, using bounded expansion when exceptions select rows."""
    known_inputs = inputs or {}
    plan = _plan(sweep, known_inputs, sets)
    if plan.exceptions or plan.conditional:
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
    sets: SweepSets | None = None,
) -> tuple[_AxisChoice, ...]:
    path = f"sweep.axes.{name}"
    target = None
    try:
        match axis:
            case SetAxis(set=document):
                choices = tuple(
                    replace(choice, origin=("set_entry", name, choice.key, document))
                    for choice in _choices(name, (sets or {})[name], inputs)
                )
            case NamedAxis(values=values):
                choices = tuple(
                    _AxisChoice(
                        key,
                        SweepCoordinate(key, value.label, value.value),
                        _patches(value),
                        patch_authors=value._patch_authors,
                        task_additions=_task_additions(value),
                    )
                    for key, value in values.items()
                    if value is not None
                )
            case InputAxis(input=role):
                members = inputs[role]
                assert members is not None
                choices = tuple(
                    _AxisChoice(
                        member.key,
                        SweepInputCoordinate(member.key, member.label, carries=tuple(member.coordinates)),
                        member=member,
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


def axis_coordinates(
    coordinates: Mapping[str, SweepCoordinate], input_axes: Mapping[str, SweepCoordinate],
) -> dict[str, SweepCoordinate]:
    """Each axis a row is chosen by: its coordinates, and its input axes' members."""
    return {**coordinates, **input_axes}


def coordinate_text(coordinate: SweepCoordinate) -> str:
    """A coordinate as drawn: its label, else its single value, else its name."""
    if coordinate.label is not None:
        return coordinate.label
    value = coordinate.value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    return coordinate.value_id


def _expanded(axes: Mapping[str, Axis], selection: Mapping[str, _AxisChoice]) -> ExpandedRow:
    """The row of one value per axis in ``selection``; an axis it has no value on is absent from its ID and label."""
    axis_value_ids = {name: choice.key for name, choice in selection.items()}
    row_id = axis_values_id(axis_value_ids) if axis_value_ids else DEFAULT_SWEEP_ROW
    coordinates: dict[str, SweepCoordinate] = {}
    patches: dict[str, PatchValue] = {}
    origins: dict[str, PatchOrigin] = {}
    authors: dict[str, str] = {}
    bindings: dict[str, str] = {}
    input_axes: dict[str, SweepInputCoordinate] = {}
    labels = []
    for name, axis in axes.items():
        if (choice := selection.get(name)) is None:
            continue
        labels.append(f"{name}={coordinate_text(choice.coordinate)}")
        member = choice.member
        if isinstance(axis, InputAxis):
            assert member is not None
            bindings[axis.input] = member.key
            assert isinstance(choice.coordinate, SweepInputCoordinate)
            input_axes[name] = choice.coordinate
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
            origins[path] = choice.origin or ("sweep_value", name, choice.key)
            patches[path] = patch
        authors.update(choice.patch_authors)
    return ExpandedRow(
        row_id, ", ".join(labels) or None, coordinates, patches, bindings, origins, authors, axis_value_ids,
        input_axes, tuple(choice.task_additions for choice in selection.values() if choice.task_additions is not None),
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
        return select_sweep_exception_rows(
            exception.where,
            rows,
            key=lambda row: row.id,
            coordinates=lambda row: axis_coordinates(row.coordinates, row.input_axes), producer="this step",
        )
    except RuleRefusal as error:
        term = f".{error.path}" if error.path else ""
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_EXCEPTION, f"{path}.where{term}", error.detail, span=error.span,
        ) from error


def _apply_exceptions(
    rows: tuple[ExpandedRow, ...], exceptions: Mapping[str, SweepException],
    matches: dict[str, tuple[str, ...]] | None = None,
) -> tuple[ExpandedRow, ...]:
    if not exceptions:
        return rows
    kept = {row.id: row for row in rows}
    for name, exception in exceptions.items():
        path = f"sweep.exceptions.{name}"
        selected = _select_exception_rows(name, exception, rows)
        if matches is not None:
            matches[name] = tuple(row.id for row in selected)
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
                    if origin[0] in ("sweep_value", "set_entry") and _paths_overlap(prior, target):
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
    return {name: _choices(name, axis, inputs, integer_site, plan.sets) for name, axis in plan.axes.items()}


def _task_additions(value: AxisValue) -> TaskAdditions | None:
    if value.task_variables or value.task_links:
        return TaskAdditions(task_variables=value.task_variables, task_links=value.task_links)
    return None


def _explicit_rows(rows: Mapping[str, Row]) -> Iterator[ExpandedRow]:
    for name, row in rows.items():
        patches = _patches(row)
        origins: dict[str, PatchOrigin] = {
            path: ("row", row._patch_sources[path]) for path in patches
        }
        additions = _task_additions(row)
        yield ExpandedRow(
            name, row.label, {}, patches, {}, origins, row._patch_authors,
            task_additions=(additions,) if additions is not None else (),
        )


def _condition_picks(name: str, where: AxisCondition, rows: Sequence[ExpandedRow]) -> tuple[bool, ...]:
    try:
        picked = {
            id(row) for row in select_sweep_exception_rows(
                SweepExceptionTerms.model_validate(where.model_dump(exclude_none=True)), rows, key=lambda row: row.id,
                coordinates=lambda row: axis_coordinates(row.coordinates, row.input_axes), producer="this step",
            )
        }
    except RuleRefusal as error:
        term = f".{error.path}" if error.path else ""
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_CONDITION, f"sweep.axes.{name}.where{term}", error.detail, span=error.span,
        ) from error
    return tuple(id(row) in picked for row in rows)


def _conditioned(
    plan: _SweepPlan, choices: Mapping[str, Sequence[_AxisChoice]], max_rows: int,
) -> list[tuple[dict[str, _AxisChoice], ExpandedRow]]:
    """The combination's rows, each row a conditional axis picks divided into one row per value of the axis.

    A row an optional axis picks is kept, and one row per value of the axis is added after it.
    """
    pairs = [(selection, _expanded(plan.axes, selection)) for selection in plan.selections(choices)]
    for name in plan.conditional:
        axis = plan.axes[name]
        where = _condition(axis)
        picks = (True,) * len(pairs) if where is None else _condition_picks(name, where, [row for _, row in pairs])
        divided = []
        for (selection, row), picked in zip(pairs, picks, strict=True):
            if not picked or _optional(axis):
                divided.append((selection, row))
            if not picked:
                continue
            for choice in choices[name]:
                chosen = selection | {name: choice}
                divided.append((chosen, _expanded(plan.axes, chosen)))
        if len(plan.rows) + len(divided) > max_rows:
            raise SweepExpansionError(
                SweepExpansionCode.ROW_BUDGET, "sweep",
                f"{len(plan.rows) + len(divided)} rows exceeds the limit of {max_rows}",
            )
        pairs = divided
    return pairs


def _generated_rows(
    plan: _SweepPlan, inputs: SweepInputs, max_axis_values: int, integer_site: IntegerSite | None = None,
    max_rows: int = DEFAULT_SWEEP_LIMIT,
) -> Iterator[tuple[dict[str, _AxisChoice], ExpandedRow]]:
    if not plan.generated_count:
        return
    choices = _planned_choices(plan, inputs, max_axis_values, integer_site)
    ids = set(plan.rows)
    for selection, row in _conditioned(plan, choices, max_rows):
        if row.id in ids:
            raise SweepExpansionError(
                SweepExpansionCode.DUPLICATE_ROW, "sweep", f"duplicate row id {row.id!r}"
            )
        ids.add(row.id)
        yield selection, row


def _expand_plan(
    plan: _SweepPlan, inputs: SweepInputs, max_rows: int, max_axis_values: int,
    integer_site: IntegerSite | None = None,
    exception_matches: dict[str, tuple[str, ...]] | None = None,
) -> tuple[ExpandedRow, ...]:
    _check_row_budget(plan, max_rows)
    rows = list(_explicit_rows(plan.rows))
    if not plan.axes and not plan.rows:
        rows.append(ExpandedRow(DEFAULT_SWEEP_ROW, None, {}, {}, {}))
    rows.extend(row for _, row in _generated_rows(plan, inputs, max_axis_values, integer_site, max_rows))
    return _apply_exceptions(tuple(rows), plan.exceptions, exception_matches)


def expand_sweep(
    sweep: Sweep | None,
    *,
    inputs: SweepInputs | None = None,
    max_rows: int = DEFAULT_SWEEP_LIMIT,
    max_axis_values: int = DEFAULT_SWEEP_LIMIT,
    integer_site: IntegerSite | None = None,
    checkpoint_subject: bool = False,
    exception_matches: dict[str, tuple[str, ...]] | None = None,
    sets: SweepSets | None = None,
) -> tuple[ExpandedRow, ...]:
    """Expand the declared sweep using recorded draws, known input members and resolved sets.

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
            return _expand_checkpoint_axis(
                resolved, name, axis, known_inputs, max_rows, max_axis_values, integer_site, exception_matches,
                sets,
            )
    return _expand_plan(
        _plan(sweep, known_inputs, sets), known_inputs, max_rows, max_axis_values, integer_site, exception_matches,
    )


def _expand_checkpoint_axis(
    sweep: Sweep, name: str, axis: CheckpointRuleAxis, inputs: SweepInputs,
    max_rows: int, max_axis_values: int, integer_site: IntegerSite | None,
    exception_matches: dict[str, tuple[str, ...]] | None, sets: SweepSets | None,
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
    rows = _expand_plan(_plan(base, inputs, sets), inputs, max_rows, max_axis_values, integer_site)
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
                row, id=axis_values_id(keys), axis_value_ids=keys,
                label=", ".join(part for part in (row.label, f"{name}={key}") if part),
                coordinates={**row.coordinates, name: SweepCoordinate(key, key, update)},
                patches={**row.patches, axis.path: SetValue(value=[{"at": update}])},
                patch_origins={**row.patch_origins, axis.path: ("sweep_value", name, key)},
                patch_authors=authors,
            ))
    return _apply_exceptions(tuple(result), {key: value for key, value in sweep.exceptions.items() if value is not None},
                             exception_matches)


def authored_row(sweep: Sweep | None, row_id: str) -> ExpandedRow | None:
    """Select one explicitly authored row, applying the enclosing sweep's exceptions."""
    resolved = _resolved_sweep(sweep)
    rows = {name: row for name, row in resolved.rows.items() if row is not None}
    if row_id not in rows:
        return None
    exceptions = {name: item for name, item in resolved.exceptions.items() if item is not None}
    needs_generated = any(
        where.rule.strip() or any(name not in rows for name in where.rows or ())
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
    _check_axis_budget("values", _axis_size("values", axis, {}, {}), max_axis_values)
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
    if plan.conditional:
        raise SweepExpansionError(
            SweepExpansionCode.INVALID_COMBINATION, f"sweep.axes.{plan.conditional[0]}.where",
            "a manual combination names every axis in every row; a conditioned axis is not on every row",
        )
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
                    "where": SweepExceptionTerms(
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
