"""Shared expansion of authored Objective, Set, and validation-trial families."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import islice
from math import prod
from string import Formatter
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer, StringConstraints

from feedbax.contracts.combinations import (
    DEFAULT_COMBINATION_LIMIT, cross_dimension, select_combinations, zip_dimension, zip_length,
)
from feedbax.contracts.expression_storage import AuthoredExpressionText, EXPRESSION_MEMBER_RESERVED_KEYS, StoredExpression
from feedbax.contracts.expression_text import parse_sequence_rule, print_stored_expression
from feedbax.contracts.expression_sequence import sequence_definition
from feedbax.contracts.expression_values import (
    DrawValue, Expression, NameValue, SequenceValue, UnresolvedValue, ValueRecord, children,
)


type IndexName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z_0-9]*$")]


class NamedIndexValue(ValueRecord):
    value: StoredExpression
    label: str | None = Field(default=None, min_length=1)


def _read_index_sequence(value: object) -> SequenceValue | UnresolvedValue:
    if isinstance(value, (SequenceValue, UnresolvedValue)):
        return value
    if not isinstance(value, str):
        raise ValueError("A sequence index requires rule text.")
    parsed = parse_sequence_rule(value)
    return UnresolvedValue(text=value) if isinstance(parsed, tuple) else parsed.value


def _write_index_sequence(value: SequenceValue | UnresolvedValue) -> str:
    printed = print_stored_expression(value)
    assert isinstance(printed, str)
    return printed


type StoredIndexSequence = Annotated[
    SequenceValue | UnresolvedValue,
    BeforeValidator(_read_index_sequence, json_schema_input_type=AuthoredExpressionText),
    PlainSerializer(_write_index_sequence, return_type=AuthoredExpressionText),
]

type IndexRule = (
    StoredIndexSequence | Annotated[dict[IndexName, StoredExpression | NamedIndexValue], Field(min_length=1)]
)


class StoredFamilyMember(ValueRecord):
    over: Annotated[dict[IndexName, IndexRule], Field(min_length=1)] | None = None
    combination: Literal["cross", "zip"] = "cross"


@dataclass(frozen=True)
class ExpandedFamilyMember[Member: StoredFamilyMember]:
    key: str
    source_key: str
    value: Member
    coordinates: Mapping[str, str]
    indices: Mapping[str, Expression] = field(default_factory=dict)
    coordinate_names: Mapping[str, str] = field(default_factory=dict)


class FamilyProblem(ValueError):
    def __init__(
        self, source_key: str, path: tuple[str, ...], detail: str, coordinates: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.source_key = source_key
        self.path = path
        self.coordinates = {} if coordinates is None else dict(coordinates)


@dataclass(frozen=True)
class _IndexValue:
    key: str
    value: Expression
    label: str


def _constant(value: Expression) -> bool:
    return not isinstance(value, (NameValue, DrawValue, UnresolvedValue)) and all(
        _constant(child) for _, child in children(value)
    )


def named_index_name(key: str, value: Expression | NamedIndexValue) -> str:
    return value.label or key if isinstance(value, NamedIndexValue) else key


def index_rule_count(rule: IndexRule) -> int:
    """Number of index values, without materializing the sequence."""
    return _index_values(rule)[0]


def _index_values(rule: IndexRule) -> tuple[int, Iterator[_IndexValue]]:
    if isinstance(rule, UnresolvedValue):
        parsed = parse_sequence_rule(rule.text)
        assert isinstance(parsed, tuple)
        raise ValueError(parsed[0].reason)
    if isinstance(rule, SequenceValue):
        sequence = sequence_definition(rule)

        def sequence_values():
            for position, member in enumerate(sequence.members()):
                number = int(member.value) if sequence.rule == "span" else position
                key = str(number) if number >= 0 else f"m{-number}"
                yield _IndexValue(key, member, str(print_stored_expression(member)))

        return sequence.count, sequence_values()

    def named_values():
        for key, item in rule.items():
            value = item.value if isinstance(item, NamedIndexValue) else item
            if not _constant(value):
                raise ValueError(f"Index value {key!r} must be a constant expression.")
            yield _IndexValue(key, value, named_index_name(key, item))

    return len(rule), named_values()


def _bind(value, bindings: Mapping[str, Expression]):
    if isinstance(value, NameValue):
        return bindings.get(value.reference, value)
    if isinstance(value, BaseModel):
        updates = {}
        for name in type(value).model_fields:
            old = getattr(value, name)
            new = _bind(old, bindings)
            if old is not new:
                updates[name] = new
        return value.model_copy(update=updates) if updates else value
    if isinstance(value, dict):
        changed = {key: _bind(item, bindings) for key, item in value.items()}
        return value if all(value[key] is item for key, item in changed.items()) else changed
    if isinstance(value, tuple):
        changed = tuple(_bind(item, bindings) for item in value)
        return value if all(old is new for old, new in zip(value, changed, strict=True)) else changed
    return value


def _label(template: str, selection: Mapping[str, _IndexValue], *, placeholders: bool) -> str:
    parts = []
    for text, name, spec, conversion in Formatter().parse(template):
        parts.append(text)
        if name is not None:
            if not placeholders:
                raise ValueError("An Objective family has one label and takes no placeholders.")
            if name not in selection or spec or conversion:
                raise ValueError(f"No family index {name!r} for the label.")
            parts.append(selection[name].label)
    return "".join(parts)


def _family_rules(
    key: str, member: StoredFamilyMember,
) -> tuple[dict[str, tuple[int, Iterator[_IndexValue]]], int]:
    if member.over is None:
        if member.combination != "cross":
            raise FamilyProblem(key, ("combination",), "A combination needs an index rule.")
        return {}, 1

    rules = {}
    definitions = getattr(member, "task_variables", {})
    for name, rule in member.over.items():
        if name in EXPRESSION_MEMBER_RESERVED_KEYS or name in definitions:
            raise FamilyProblem(key, ("over", name), f"{name!r} is reserved or defined by this member.")
        try:
            rules[name] = _index_values(rule)
        except ValueError as error:
            raise FamilyProblem(key, ("over", name), str(error)) from error
    sizes = {name: size for name, (size, _) in rules.items()}
    zipped = member.combination == "zip"
    try:
        count = zip_length(sizes) if zipped else prod(sizes.values())
    except ValueError as error:
        raise FamilyProblem(key, ("combination",), str(error)) from error
    if count > DEFAULT_COMBINATION_LIMIT:
        raise FamilyProblem(key, ("over",), f"A family may generate at most {DEFAULT_COMBINATION_LIMIT:,} members.")
    return rules, count


def family_member_count(key: str, member: StoredFamilyMember) -> int:
    return _family_rules(key, member)[1]


def expand_family[Member: StoredFamilyMember](
    key: str, member: Member, *, first: bool = False, label_placeholders: bool = True, unique_labels: bool = False,
) -> Iterator[ExpandedFamilyMember[Member]]:
    """Substitute one ordered family, retaining its authored key and selection in every generated member."""
    rules, _ = _family_rules(key, member)
    if not rules:
        yield ExpandedFamilyMember(key, key, member, {})
        return

    choices = {}
    for name, (_, values) in rules.items():
        try:
            choices[name] = tuple(islice(values, 1) if first else values)
        except ValueError as error:
            raise FamilyProblem(key, ("over", name), str(error)) from error
    loaded = {name: len(values) for name, values in choices.items()}
    dimensions = (zip_dimension(loaded),) if member.combination == "zip" else tuple(
        cross_dimension(name, size) for name, size in loaded.items()
    )

    labels = set()
    for selection in select_combinations(dimensions, choices):
        coordinates = {name: selected.key for name, selected in selection.items()}
        generated_key = "-".join((key, *coordinates.values()))
        indices = {name: selected.value for name, selected in selection.items()}
        coordinate_names = {name: selected.label for name, selected in selection.items()}
        bound = _bind(member.model_copy(update={"over": None, "combination": "cross"}), indices)
        if (label := getattr(bound, "label", None)) is not None:
            try:
                label = _label(label, selection, placeholders=label_placeholders)
                if unique_labels and label in labels:
                    raise ValueError(f"Several family members have the label {label!r}.")
                labels.add(label)
            except ValueError as error:
                raise FamilyProblem(key, ("label",), str(error), coordinate_names) from error
            bound = bound.model_copy(update={"label": label})
        yield ExpandedFamilyMember(generated_key, key, bound, coordinates, indices, coordinate_names)


def expand_members[Member: StoredFamilyMember](
    members: Mapping[str, Member], *, first: bool = False, label_placeholders: bool = True, unique_labels: bool = False,
) -> Iterator[ExpandedFamilyMember[Member]]:
    keys = set(members)
    for key, member in members.items():
        for generated in expand_family(
            key, member, first=first, label_placeholders=label_placeholders, unique_labels=unique_labels,
        ):
            if generated.coordinates:
                if generated.key in keys:
                    raise FamilyProblem(
                        key, ("over",), f"Generated member key {generated.key!r} already exists.", generated.coordinate_names,
                    )
                keys.add(generated.key)
            yield generated
