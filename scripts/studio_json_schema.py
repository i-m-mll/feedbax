"""Project the structural JSON Schema of authored Studio documents into Zod."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, NamedTuple

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema


class _Projection(NamedTuple):
    typescript: str
    zod: str


_ANNOTATIONS = {"title", "description", "default"}
_SCALARS = {
    "string": _Projection("string", "z.string()"),
    "number": _Projection("number", "z.number().finite()"),
    "integer": _Projection("number", "z.number().int().safe()"),
    "boolean": _Projection("boolean", "z.boolean()"),
    "null": _Projection("null", "z.null()"),
}
_JSON = _Projection("JsonValue", "jsonValueSchema")
_REF_PREFIX = "#/$defs/"


def _project(schema: dict[str, Any], path: str) -> _Projection:
    def check(*keywords: str) -> None:
        unknown = schema.keys() - _ANNOTATIONS - set(keywords)
        if unknown:
            raise ValueError(
                f"Unsupported Studio JSON Schema keywords at {path}: {sorted(unknown)}"
            )

    if not schema.keys() - _ANNOTATIONS:
        return _JSON

    if "$ref" in schema:
        check("$ref")
        ref = schema["$ref"]
        if not ref.startswith(_REF_PREFIX):
            raise ValueError(f"Unsupported Studio JSON Schema reference at {path}: {ref}")
        name = ref.removeprefix(_REF_PREFIX)
        return _JSON if name == "JsonValue" else _Projection(name, f"{name}Schema")

    if "const" in schema or "enum" in schema:
        check("const", "enum", "type")
        values = schema["enum"] if "enum" in schema else [schema["const"]]
        literals = [json.dumps(value, ensure_ascii=False) for value in values]
        return _union([_Projection(value, f"z.literal({value})") for value in literals])

    for key in ("anyOf", "oneOf"):
        if key in schema:
            check(key, "discriminator")
            if key == "oneOf" and "discriminator" not in schema:
                raise ValueError(f"Undiscriminated Studio JSON Schema oneOf at {path}")
            return _union(
                [
                    _project(option, f"{path}.{key}[{index}]")
                    for index, option in enumerate(schema[key])
                ]
            )

    kind = schema.get("type")
    if kind == "object":
        if "properties" in schema:
            check("type", "properties", "required", "additionalProperties")
            if schema.get("additionalProperties") is not False:
                raise ValueError(f"Unsealed Studio object at {path}")
            required = set(schema.get("required", []))
            members: list[str] = []
            validators: list[str] = []
            for name, definition in schema["properties"].items():
                item = _project(definition, f"{path}.{name}")
                key = json.dumps(name)
                optional = name not in required
                members.append(f"  {key}{'?' if optional else ''}: {item.typescript};")
                validators.append(f"  {key}: {item.zod}{'.optional()' if optional else ''},")
            return _Projection(
                "{\n" + "\n".join(members) + "\n}",
                "z.object({\n" + "\n".join(validators) + "\n}).strict()",
            )

        check("type", "patternProperties", "propertyNames", "additionalProperties")
        names = schema.get("propertyNames", {"type": "string"})
        values = schema.get("additionalProperties", True)
        if "patternProperties" in schema:
            patterns = schema["patternProperties"]
            if len(patterns) != 1 or "propertyNames" in schema or "additionalProperties" in schema:
                raise ValueError(f"Ambiguous Studio record at {path}")
            pattern, values = next(iter(patterns.items()))
            names = {"type": "string", "pattern": pattern}
        key = _project(names, f"{path}.key")
        value = _JSON if values is True else _project(values, f"{path}.*")
        return _Projection(
            f"Record<string, {value.typescript}>", f"z.record({key.zod}, {value.zod})"
        )

    if kind == "array":
        if "prefixItems" in schema:
            check("type", "prefixItems", "minItems", "maxItems")
            definitions = schema["prefixItems"]
            if schema.get("minItems") != len(definitions) or schema.get("maxItems") != len(definitions):
                raise ValueError(f"Unbounded Studio tuple at {path}")
            items = [_project(item, f"{path}[{index}]") for index, item in enumerate(definitions)]
            return _Projection(
                "[" + ", ".join(item.typescript for item in items) + "]",
                "z.tuple([" + ", ".join(item.zod for item in items) + "])",
            )
        check("type", "items", "minItems", "maxItems")
        item = _project(schema["items"], f"{path}[]")
        zod = _bounds(f"z.array({item.zod})", schema, {"minItems": "min", "maxItems": "max"})
        return _Projection(f"Array<{item.typescript}>", zod)

    if kind in _SCALARS:
        constraints = {
            "string": {"minLength": "min", "maxLength": "max"},
            "number": {
                "minimum": "min",
                "exclusiveMinimum": "gt",
                "maximum": "max",
                "exclusiveMaximum": "lt",
            },
            "integer": {
                "minimum": "min",
                "exclusiveMinimum": "gt",
                "maximum": "max",
                "exclusiveMaximum": "lt",
            },
        }.get(kind, {})
        check("type", *constraints, *(("pattern",) if kind == "string" else ()))
        typescript, zod = _SCALARS[kind]
        zod = _bounds(zod, schema, constraints)
        if "pattern" in schema:
            zod += f".regex(new RegExp({json.dumps(schema['pattern'])}))"
        return _Projection(typescript, zod)

    raise ValueError(f"Unsupported Studio JSON Schema at {path}: {schema!r}")


def _bounds(zod: str, schema: dict[str, Any], constraints: dict[str, str]) -> str:
    return zod + "".join(
        f".{method}({schema[key]})" for key, method in constraints.items() if key in schema
    )


def _union(items: list[_Projection]) -> _Projection:
    if len(items) == 1:
        return items[0]
    return _Projection(
        " | ".join(dict.fromkeys(item.typescript for item in items)),
        f"z.union([{', '.join(item.zod for item in items)}])",
    )


def emit_authored_contracts(
    models: Sequence[type[BaseModel]],
    existing: Sequence[type[BaseModel]],
) -> tuple[str, str]:
    """Emit wire types and structural validators, leaving semantic admission to Python."""
    _, schema = models_json_schema([(model, "validation") for model in models])
    definitions = schema["$defs"]
    shared = {model.__name__: model for model in existing}
    interfaces, validators = [], []
    for name, definition in definitions.items():
        if name == "JsonValue" and definition == {}:
            continue
        if name in shared:
            previous = shared[name].model_json_schema()
            previous_definitions = previous.pop("$defs", {})
            if previous == {"$ref": f"#/$defs/{name}"}:
                previous = previous_definitions[name]
            if definition != previous:
                raise ValueError(f"Conflicting Studio JSON Schema definition: {name}")
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise ValueError(f"Invalid Studio contract name: {name}")
        typescript, zod = _project(definition, name)
        interfaces.append(f"export type {name} = {typescript};")
        validators.append(
            f"export const {name}Schema: z.ZodType<{name}> = z.lazy(() =>\n"
            f"  {zod.replace(chr(10), chr(10) + '  ')}\n"
            f") as unknown as z.ZodType<{name}>;"
        )
    return "\n\n".join(interfaces), "\n\n".join(validators)
