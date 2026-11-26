"""Compare project reads by deleting authored keys one at a time."""

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

from pydantic import BaseModel, RootModel

from feedbax.application.stored_regions import StoredRegionRegistries
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.project import InlineBase, RootDocument, Step
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_io import TextProject, parse_text_project_file
from feedbax.contracts.project_storage import StoredPath, validated_stored_payload
from feedbax.contracts.project_storage_dependencies import protected_project_fields
from feedbax.contracts.project_sweep import AxisValue, NamedAxis, Row, Sweep, SweepException, sweep_refines_parent
from feedbax.contracts.store_check import RestatedDefaultAdvice
from feedbax.contracts.stored_regions import StoredRegion, step_inline_model, typed_region
from feedbax.contracts.strict_json import strict_json_loads

type AuthoredKeyPath = tuple[str | int, ...]


def _refinement_fields(value: object, path: AuthoredKeyPath = ()) -> Iterator[tuple[AuthoredKeyPath, tuple[str, ...]]]:
    if isinstance(value, BaseModel):
        if isinstance(value, (Sweep, NamedAxis, AxisValue, Row, SweepException)):
            yield path, tuple(sorted(value.model_fields_set))
        for field in type(value).model_fields:
            child = getattr(value, field)
            if isinstance(value, Step) and field == "sweep" and isinstance(child, Sweep) and not sweep_refines_parent(child):
                continue
            yield from _refinement_fields(child, (*path, field))
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _refinement_fields(child, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _refinement_fields(child, (*path, index))


def _effective_payload(value, encoded, registries):
    """Interpret stored values with registered types and defaults before comparing reads."""
    if isinstance(value, RootModel):
        return _effective_payload(value.root, encoded, registries)
    if isinstance(value, RootDocument):
        payload = validated_stored_payload(value)
        if payload is None:
            return encoded
        return encoded | {"payload": _effective_payload(payload, payload.model_dump(mode="json", by_alias=True), registries)}
    if isinstance(value, BaseModel):
        if not isinstance(encoded, dict):
            # Stored expression and time aliases encode typed nodes as scalars or text.
            return value.model_dump(mode="json", by_alias=True)

        def read_field(name, field):
            key = field.serialization_alias or field.alias or name
            child = getattr(value, name)
            if isinstance(value, Step) and isinstance(child, InlineBase):
                region = StoredRegion(step_inline_model(value.kind, child.inline))
                return {"inline": read_region(region, child.inline, inline=True)}
            if isinstance(encoded[key], dict):
                try:
                    region = registries.region(value, name)
                    if region is not None:
                        return read_region(region, encoded[key])
                except ValueError:
                    # An incomplete installed declaration may be a saved draft.
                    return encoded[key]
            return _effective_payload(child, encoded[key], registries)

        def read_region(region, raw, *, inline=False):
            if region.model is None:
                return raw
            context = {
                key: region.lowered_expressions.get(key, item)
                for key, item in raw.items() if key in region.contextual
            }
            typed, values, complete = typed_region(
                region, {key: item for key, item in raw.items() if key not in context}, inline=inline,
            )
            if not complete and not inline:
                return raw
            return _effective_payload(typed, values, registries) | context

        for name, field in type(value).model_fields.items():
            key = field.serialization_alias or field.alias or name
            if key in encoded:
                if (isinstance(value, Step) and name == "sweep" and isinstance(value.sweep, Sweep)
                    and not sweep_refines_parent(value.sweep)):
                    encoded[key] = None
                else:
                    encoded[key] = read_field(name, field)
        return encoded
    if isinstance(value, dict) and isinstance(encoded, dict):
        return {key: _effective_payload(value.get(key), child, registries) for key, child in encoded.items()}
    if isinstance(value, (tuple, list)) and isinstance(encoded, list):
        return [_effective_payload(item, child, registries) for item, child in zip(value, encoded, strict=True)]
    return encoded


def project_file_meaning(model: BaseModel, registries: StoredRegionRegistries) -> tuple[bytes, tuple]:
    """Read values with defaults filled, retaining which refinement fields the author set."""
    encoded = model.model_dump(mode="json", by_alias=True)
    value = _effective_payload(model, encoded, registries)
    return canonical_json_v2_bytes(value), tuple(_refinement_fields(model))


def authored_key_paths(value: object, prefix: AuthoredKeyPath = ()) -> Iterator[AuthoredKeyPath]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, key)
            yield path
            yield from authored_key_paths(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from authored_key_paths(child, (*prefix, index))


def without_authored_key(value: dict, path: AuthoredKeyPath) -> dict:
    edited = deepcopy(value)
    parent = edited
    for part in path[:-1]:
        parent = parent[part]
    del parent[path[-1]]
    return edited


def restated_default_advice(project: TextProject) -> tuple[RestatedDefaultAdvice, ...]:
    """Report each removable field at its file and pointer using the normal file reader."""
    regions = StoredRegionRegistries()
    protected = protected_project_fields(project.project, registries=regions)
    advice = []
    for name, data in project.files.items():
        source = Path(name)
        raw = strict_json_loads(data, ref=name)
        protected_paths: set[StoredPath] = {
            path[:length]
            for path in protected.get((source.parent.name, source.stem), ())
            for length in range(1, len(path) + 1)
        }
        meaning = project_file_meaning(parse_text_project_file(source, raw), regions)
        removable = set()
        for path in authored_key_paths(raw):
            parts = tuple(map(str, path))
            if parts in protected_paths or any(path[:length] in removable for length in range(1, len(path))):
                continue
            try:
                edited = parse_text_project_file(source, without_authored_key(raw, path))
                if project_file_meaning(edited, regions) != meaning:
                    continue
            except ValueError:
                continue
            removable.add(path)
            advice.append(RestatedDefaultAdvice(
                code="restated_default", address=name, path=pointer(parts),
                detail="This key can be omitted because it reads the same as its default.",
            ))
    return tuple(advice)
