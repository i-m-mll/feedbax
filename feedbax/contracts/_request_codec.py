"""Model-driven codec for the kernel's minimal request documents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from enum import Enum
from functools import cache
from types import UnionType
from typing import Annotated, Any, ForwardRef, Literal, TypeAliasType, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, TypeAdapter

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.array_values import (
    ARRAY_VALUE_SCHEMA_ID,
    ARRAY_VALUE_SCHEMA_VERSION,
    ConstantArrayValueSpec,
    SparseCooArrayValueSpec,
    _parse_array_value_payload,
)
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.graph import (
    AnalysisInputRequirement,
    ComponentSpec,
    GraphSpec,
    RetainedObservableSpec,
)
from feedbax.contracts.objective_document import ObjectiveTerm
from feedbax.contracts.request_identity_guard import IdentityAbsent, IdentityWideInteger
from feedbax.contracts.request_refs import ReferenceValue


_SAFE_INT = (1 << 53) - 1
_WIRE_FIELDS = frozenset({"schema_id", "schema_version"})
_FAMILY = "$family"
_OPAQUE = "$json"
_PRESENTATION = {(GraphSpec, "metadata"), (AcausalGraphSpec, "metadata"), (ObjectiveTerm, "name")}
_REQUIRED_STORED = {(RetainedObservableSpec, "id"), (AnalysisInputRequirement, "id")}


def _annotation(annotation: Any) -> Any:
    while True:
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
        elif get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        else:
            return annotation


@cache
def _model_choices(annotation: Any) -> tuple[type[BaseModel], ...]:
    annotation = _annotation(annotation)
    if get_origin(annotation) in (Union, UnionType):
        return tuple(model for part in get_args(annotation) for model in _model_choices(part))
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return (annotation,)
    return ()


def _opaque_mapping(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, UnionType) and any(
        part is Any or get_origin(part) is dict for part in get_args(annotation)
    )


def _tagged(annotation: Any) -> bool:
    choices = _model_choices(annotation)
    return len(choices) > 1 or bool(choices) and _opaque_mapping(annotation)


def _wire(model: type[BaseModel], name: str) -> bool:
    field = model.model_fields[name]
    return name in _WIRE_FIELDS and isinstance(field.default, str)


def _omitted(model: type[BaseModel], name: str) -> bool:
    return _wire(model, name) or (model, name) in _PRESENTATION


def _model_key(model: type[BaseModel]) -> str:
    return f"{model.__module__}.{model.__qualname__}"


def _family(model: type[BaseModel]) -> str:
    if issubclass(model, ReferenceValue):
        if name := getattr(model, "identity_family", None):
            return name
        if kind := model.model_fields.get("kind"):
            return kind.default
    field = model.model_fields.get("schema_id")
    return (
        field.default if field is not None and isinstance(field.default, str) else _model_key(model)
    )


def _sequence_annotation(annotation: Any, index: int) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is tuple and args and args[-1] is not Ellipsis:
        return args[index] if index < len(args) else Any
    return args[0] if origin in (list, tuple) and args else Any


def _mapping_annotation(annotation: Any) -> Any:
    args = get_args(annotation)
    return args[1] if get_origin(annotation) is dict and args else Any


def _container_annotation(annotation: Any, origin: type) -> Any:
    if get_origin(annotation) in (Union, UnionType):
        choices = [part for part in get_args(annotation) if get_origin(part) is origin]
        if len(choices) == 1:
            return choices[0]
    return annotation


def _array_parameters(value: Any, *, decode: bool) -> Any:
    if isinstance(value, list):
        return [_array_parameters(item, decode=decode) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("schema_id") != ARRAY_VALUE_SCHEMA_ID:
        return {key: _array_parameters(item, decode=decode) for key, item in value.items()}

    if decode:
        encoded = {**value, "schema_version": ARRAY_VALUE_SCHEMA_VERSION}
        if encoded["dtype"] in {"int64", "uint64"}:
            if encoded["encoding"] == "constant":
                encoded["value"] = int(encoded["value"])
            else:
                encoded["fill"] = int(encoded["fill"])
                encoded["entries"] = [
                    {**entry, "value": int(entry["value"])} for entry in encoded["entries"]
                ]
        return encoded

    array = _parse_array_value_payload(value)
    assert array is not None
    result = array.model_dump(mode="json", exclude={"schema_version"})
    if array.dtype in {"int64", "uint64"}:
        if isinstance(array, ConstantArrayValueSpec):
            result["value"] = str(array.value)
        elif isinstance(array, SparseCooArrayValueSpec):
            result["fill"] = str(array.fill)
            result["entries"] = [
                {**entry, "value": str(entry["value"])} for entry in result["entries"]
            ]
    return result


def _wide_integer(model: type[BaseModel], name: str) -> bool:
    return any(isinstance(item, IdentityWideInteger) for item in model.model_fields[name].metadata)


@cache
def _field_annotation(model: type[BaseModel], name: str) -> Any:
    annotation = model.model_fields[name].annotation
    return get_type_hints(model, include_extras=True)[name] if isinstance(annotation, ForwardRef) else annotation


def _encode_field(
    model: type[BaseModel], name: str, value: Any, pins: Mapping | None, *, explicit: bool = False
) -> Any:
    if value is not None and _wide_integer(model, name):
        return str(value) if abs(value) > _SAFE_INT else value
    if model is ComponentSpec and name == "params":
        return _array_parameters(value, decode=False)
    return encode(value, _field_annotation(model, name), pins, explicit=explicit)


def encode(
    value: Any,
    annotation: Any,
    pins: Mapping | None = None,
    *,
    explicit: bool = False,
    projected_fields: Mapping[str, Any] | None = None,
) -> Any:
    """Project a typed value into JSON. With pins, omit equal absent values."""
    annotation = _annotation(annotation)
    if isinstance(value, BaseModel):
        model = type(value)
        rows = {} if pins is None else pins[_model_key(model)]
        result = {}
        if _tagged(annotation):
            result[_FAMILY] = _family(model)
        for name in model.model_fields:
            item = getattr(value, name)
            if _wire(model, name) and item != model.model_fields[name].default:
                raise ValueError(
                    f"Normalize the wire version before encoding {model.__name__}.{name}"
                )
            if _omitted(model, name) or (explicit and name not in value.model_fields_set):
                continue
            row = rows.get(name, {})
            projected = (
                projected_fields[name]
                if projected_fields is not None and name in projected_fields
                else _encode_field(model, name, item, pins, explicit=explicit)
            )
            if "absent" in row and canonical_json_v2_bytes(projected) == canonical_json_v2_bytes(
                _project_absent(model, name, row["absent"], pins)
            ):
                continue
            result[name] = projected
        return result
    if isinstance(value, dict):
        if _model_choices(annotation) and _opaque_mapping(annotation):
            return {_FAMILY: _OPAQUE, "value": encode(value, Any, pins, explicit=explicit)}
        member = _mapping_annotation(_container_annotation(annotation, dict))
        return {key: encode(item, member, pins, explicit=explicit) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        shape = _container_annotation(annotation, type(value))
        return [
            encode(item, _sequence_annotation(shape, index), pins, explicit=explicit)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Enum):
        return value.value
    return value


def _project_absent(model: type[BaseModel], name: str, value: Any, pins: Mapping) -> Any:
    annotation = _field_annotation(model, name)
    if not isinstance(value, (dict, list)) or not value and not _model_choices(annotation):
        return value
    restored = decode(value, annotation, pins)
    validated = TypeAdapter(annotation).validate_json(json.dumps(restored))
    return _encode_field(model, name, validated, pins)


def _decode_model(model: type[BaseModel], value: dict, pins: Mapping) -> dict:
    rows = pins[_model_key(model)]
    fields = model.model_fields
    if unexpected := value.keys() - rows.keys():
        raise ValueError(f"Unknown stored fields for {model.__name__}: {sorted(unexpected)}")
    result = {}
    for name, row in rows.items():
        if name in value:
            item = value[name]
        elif "absent" in row:
            item = row["absent"]
        else:
            raise ValueError(f"Missing required stored field: {model.__name__}.{name}")
        if isinstance(item, str) and _wide_integer(model, name):
            item = int(item)
        if model is ComponentSpec and name == "params":
            item = _array_parameters(item, decode=True)
        result[fields[name].alias or name] = decode(item, _field_annotation(model, name), pins)
    for name in fields:
        if _wire(model, name):
            result[fields[name].alias or name] = fields[name].default
    return result


def decode(value: Any, annotation: Any, pins: Mapping) -> Any:
    """Fill pinned values recursively before model validation can apply author defaults."""
    annotation = _annotation(annotation)
    choices = _model_choices(annotation)
    if isinstance(value, dict) and choices:
        if not _tagged(annotation):
            return _decode_model(choices[0], value, pins)
        family = value.get(_FAMILY)
        if family == _OPAQUE and _opaque_mapping(annotation):
            if value.keys() != {_FAMILY, "value"} or not isinstance(value["value"], dict):
                raise ValueError("Invalid opaque value in stored request")
            return value["value"]
        for model in choices:
            if _family(model) == family:
                return _decode_model(
                    model, {key: item for key, item in value.items() if key != _FAMILY}, pins
                )
        raise ValueError(f"Unknown stored model family: {family!r}")
    if isinstance(value, dict):
        member = _mapping_annotation(_container_annotation(annotation, dict))
        return {key: decode(item, member, pins) for key, item in value.items()}
    if isinstance(value, list):
        shape = _container_annotation(annotation, list)
        if get_origin(shape) is not list:
            shape = _container_annotation(annotation, tuple)
        return [
            decode(item, _sequence_annotation(shape, index), pins)
            for index, item in enumerate(value)
        ]
    return value


def _enumerated_values(annotation: Any) -> list[Any]:
    annotation = _annotation(annotation)
    if get_origin(annotation) is Literal:
        return list(get_args(annotation))
    if get_origin(annotation) in (Union, UnionType):
        return [value for part in get_args(annotation) for value in _enumerated_values(part)]
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return [member.value for member in annotation]
    return []


def _union_families(annotation: Any, aliases: frozenset[Any] = frozenset()) -> set[str]:
    if isinstance(annotation, TypeAliasType):
        if annotation in aliases:
            return set()
        aliases = aliases | {annotation}
    annotation = _annotation(annotation)
    choices = _model_choices(annotation)
    if _tagged(annotation):
        names = {_family(model) for model in choices}
        if len(names) != len(choices) or _OPAQUE in names:
            raise ValueError(f"Ambiguous model families in request annotation: {annotation}")
        if _opaque_mapping(annotation):
            names.add(_OPAQUE)
        return names
    return {family for child in get_args(annotation) for family in _union_families(child, aliases)}


def describe_models(roots: Iterable[type[BaseModel]]) -> dict[str, Any]:
    """Generate typed absent values and enumerated spellings for reachable models."""
    descriptions: dict[str, Any] = {}
    visited_aliases: set[TypeAliasType] = set()

    def visit(annotation: Any) -> None:
        if isinstance(annotation, TypeAliasType):
            if annotation in visited_aliases:
                return
            visited_aliases.add(annotation)
        annotation = _annotation(annotation)
        if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
            if get_origin(annotation) is not Literal:
                for member in get_args(annotation):
                    visit(member)
            return
        key = _model_key(annotation)
        if key in descriptions:
            return
        fields: dict[str, Any] = {}
        descriptions[key] = fields
        for name, field in annotation.model_fields.items():
            member = _field_annotation(annotation, name)
            if _omitted(annotation, name):
                continue
            row: dict[str, Any] = {}
            if families := _union_families(member):
                row["families"] = sorted(families)
            if values := _enumerated_values(member):
                row["values"] = sorted(values, key=canonical_json_v2_bytes)
            if not field.is_required() and (annotation, name) not in _REQUIRED_STORED:
                absence = next(
                    (item for item in field.metadata if isinstance(item, IdentityAbsent)), None
                )
                if absence is None:
                    default = field.get_default(call_default_factory=True)
                    second = field.get_default(call_default_factory=True)
                    if canonical_json_v2_bytes(_encode_field(annotation, name, default, None)) != (
                        canonical_json_v2_bytes(_encode_field(annotation, name, second, None))
                    ):
                        raise ValueError(
                            f"Non-deterministic default needs a pinned absent value: {key}.{name}"
                        )
                else:
                    default = absence.value
                row["absent"] = _encode_field(annotation, name, default, None, explicit=True)
            fields[name] = row
            visit(member)

    for root in roots:
        visit(root)
    return descriptions
