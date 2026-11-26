"""Resolve and normalize the settings of one installed Train method."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import JsonValue

if TYPE_CHECKING:
    from pydantic import BaseModel

    from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
    from feedbax.training.methods import RegisteredTrainingMethod, TrainingMethodRegistry


def normalize_train_method(
    request: DocumentTrainingRequestSpec, registry: TrainingMethodRegistry,
) -> tuple[DocumentTrainingRequestSpec, RegisteredTrainingMethod, BaseModel]:
    try:
        method = registry.resolve(request.method)
    except KeyError as error:
        raise ValueError(f"/method: {error.args[0]}") from error
    settings = method.settings_model.model_validate(request.settings, extra="forbid")
    return request.model_copy(update={
        "method": method.key,
        "seed": 0 if request.seed is None else request.seed,
        "settings": settings.model_dump(mode="json"),
    }), method, settings


def setting_schema_node(schema: dict, node: dict, value: JsonValue) -> dict:
    """The schema that governs ``value``: references followed, an optional's value member, a tagged union's chosen class."""
    while "$ref" in node:
        node = schema["$defs"][node["$ref"].rsplit("/", 1)[-1]] | {k: v for k, v in node.items() if k != "$ref"}
    if "anyOf" in node:
        members = [member for member in node["anyOf"] if member.get("type") != "null"]
        rest = {k: v for k, v in node.items() if k != "anyOf"}
        return setting_schema_node(schema, rest | members[0], value) if len(members) == 1 else rest
    if "discriminator" in node:
        tag = node["discriminator"]["propertyName"]
        chosen = node["discriminator"]["mapping"].get(value.get(tag)) if isinstance(value, dict) else None
        return setting_schema_node(schema, {"$ref": chosen}, value) if isinstance(chosen, str) else {}
    return node


def setting_field_title(schema: dict, path: tuple[str, ...], value: JsonValue) -> str | None:
    """The name Train authoring gives a field, including a wrapped numeric setting."""
    node = schema
    title = None
    owner_title = None
    for part in path:
        declared = node.get("properties", {}).get(part)
        if not isinstance(declared, dict):
            return None
        title = declared.get("title")
        node = setting_schema_node(schema, declared, value.get(part) if isinstance(value, dict) else None)
        value = value.get(part) if isinstance(value, dict) else None
        if part != "value":
            owner_title = title
    return owner_title if path and path[-1] == "value" and owner_title is not None else title
