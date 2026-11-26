"""Canonical JSON for authored project files, kept separate from request identity."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from functools import cache, lru_cache
from types import MappingProxyType, UnionType
from typing import Annotated, Any, TYPE_CHECKING, TypeAliasType, Union, get_args, get_origin

import rfc8785
from pydantic import BaseModel, JsonValue, RootModel, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python

from feedbax.contracts.authored_arrays import check_authored_array_budget
from feedbax.contracts.canonical_json import validate_canonical_json_v2
from feedbax.contracts.expression_storage import StoredExpression, StoredTime
from feedbax.contracts.expression_values import LiteralValue
from feedbax.contracts.figure_document import STORED_FIGURE_DOCUMENT_SCHEMA_ID, StoredFigureSource
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import ExpressionSiteKind
from feedbax.contracts.model_attachment import SetField
from feedbax.contracts.objective_document import ObjectiveTerm, StoredObjectiveTerm
from feedbax.contracts.project import ExternalBase, InlineBase, RootDocument, SetDocumentValue, Step
from feedbax.contracts.project_sweep import AxisValue, NamedAxis, SetValue, Sweep, SweepException
from feedbax.contracts.rollout import ROLLOUT_TYPE
from feedbax.contracts.stored_members import StoredAddressedMembers, StoredMembers


if TYPE_CHECKING:
    from feedbax.contracts.stored_regions import StoredRegionResolver


type StoredPath = tuple[str, ...]

_REFINEMENTS = (Sweep, NamedAxis, AxisValue, SweepException)
_REAL_TOKEN_VALUES = (SetDocumentValue, OverridePatch, SetValue, SetField, LiteralValue)
_WIDTH = 100


class StoredFigurePayload(RootModel[StoredFigureSource | ExternalBase]):
    """The authored Figure, template selection, or external pin in a Figure document."""


@cache
def stored_document_models() -> Mapping[str, type[BaseModel] | None]:
    """The payload models used by the authored writer."""
    from feedbax.contracts.analysis_document import StoredAnalysisPayload
    from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID
    from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument
    from feedbax.contracts.objective_document import STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID, StoredObjectiveDocument
    from feedbax.contracts.project import LoadedModelDocument
    from feedbax.contracts.project_inputs import SubsetTerms
    from feedbax.contracts.report_document import ReportDocument
    from feedbax.contracts.result_address import ANALYSIS_SCHEMA
    from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID, SetDocument
    from feedbax.contracts.task_document import (
        PYTHON_TASK_SCHEMA_ID, STORED_TASK_DOCUMENT_SCHEMA_ID, PythonTaskSelection, StoredTaskDocument,
    )
    from feedbax.contracts.training_document import TRAINING_DOCUMENT_SCHEMA_ID, TrainingDocument

    return MappingProxyType({
        STORED_MODEL_DOCUMENT_SCHEMA_ID: StoredModelDocument,
        STORED_TASK_DOCUMENT_SCHEMA_ID: StoredTaskDocument,
        STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID: StoredObjectiveDocument,
        TRAINING_DOCUMENT_SCHEMA_ID: TrainingDocument,
        PYTHON_TASK_SCHEMA_ID: PythonTaskSelection,
        LOADED_MODEL_SCHEMA_ID: LoadedModelDocument,
        SET_DOCUMENT_SCHEMA_ID: SetDocument,
        ANALYSIS_SCHEMA: StoredAnalysisPayload,
        STORED_FIGURE_DOCUMENT_SCHEMA_ID: StoredFigurePayload,
        ReportDocument.model_fields["schema_version"].default: ReportDocument,
        "subset": SubsetTerms,
    })


@lru_cache(maxsize=4096)
def stored_keyed_collection(schema_id: str, path: StoredPath) -> bool:
    """Whether a stored pointer names a typed member map whose authored order matters."""
    model = stored_document_models().get(schema_id)
    return model is not None and _keyed_collection_at(model, path)


def _keyed_collection_at(annotation: Any, path: StoredPath) -> bool:
    if annotation is JsonValue:
        return False
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (StoredMembers, StoredAddressedMembers):
        return not path or _keyed_collection_at(arguments[0], path[1:])
    if origin is dict and len(arguments) == 2:
        return bool(path) and _keyed_collection_at(arguments[1], path[1:])
    if origin is Annotated:
        return _keyed_collection_at(arguments[0], path)
    if origin in (Union, UnionType):
        return any(_keyed_collection_at(member, path) for member in arguments)
    if isinstance(annotation, TypeAliasType):
        return _keyed_collection_at(annotation.__value__, path)
    if isinstance(annotation, type) and issubclass(annotation, RootModel):
        return _keyed_collection_at(annotation.model_fields["root"].annotation, path)
    if path and isinstance(annotation, type) and issubclass(annotation, BaseModel):
        annotation.model_rebuild()
        for name, field in annotation.model_fields.items():
            if _key(name, field) == path[0]:
                return _keyed_collection_at(field.annotation, path[1:])
    return False


@lru_cache(maxsize=4096)
def stored_expression_site(schema_id: str, path: StoredPath) -> ExpressionSiteKind | None:
    """Identify a declared scalar expression or time, without inspecting its content."""
    model = stored_document_models().get(schema_id)
    return _expression_site_at(model, path)


def _expression_site_at(annotation: Any, path: StoredPath) -> ExpressionSiteKind | None:
    if annotation is StoredExpression or annotation is StoredTime:
        return None if path else "expression" if annotation is StoredExpression else "time"
    if annotation is JsonValue:
        return None
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (StoredMembers, StoredAddressedMembers):
        return _expression_site_at(arguments[0], path[1:]) if path else None
    if origin is dict and len(arguments) == 2:
        return _expression_site_at(arguments[1], path[1:]) if path else None
    if origin is Annotated:
        return _expression_site_at(arguments[0], path)
    if origin in (Union, UnionType):
        return next((site for arg in arguments if (site := _expression_site_at(arg, path)) is not None), None)
    if isinstance(annotation, TypeAliasType):
        return _expression_site_at(annotation.__value__, path)
    if path and isinstance(annotation, type) and issubclass(annotation, BaseModel):
        annotation.model_rebuild()
        for name, field in annotation.model_fields.items():
            if _key(name, field) == path[0]:
                return _expression_site_at(field.annotation, path[1:])
    return None


@lru_cache(maxsize=4096)
def stored_step_expression_site(kind: str, path: StoredPath) -> ExpressionSiteKind | None:
    """Resolve an expression in a step's authored payload using its declared document roles."""
    from feedbax.contracts.objective_document import STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID
    from feedbax.contracts.result_address import ANALYSIS_SCHEMA
    from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID
    from feedbax.contracts.task_document import STORED_TASK_DOCUMENT_SCHEMA_ID

    if kind == "analyze":
        return stored_expression_site(ANALYSIS_SCHEMA, path)
    if kind == "synthesize":
        from feedbax.contracts.synthesis import SynthesisStepPayload

        return _expression_site_at(SynthesisStepPayload, path)
    if kind in {"train", "evaluate"} and len(path) > 2 and path[0] == "trial_documents":
        schema = {
            "model": STORED_MODEL_DOCUMENT_SCHEMA_ID,
            "task": STORED_TASK_DOCUMENT_SCHEMA_ID,
            "objective": STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID,
        }.get(path[1])
        return stored_expression_site(schema, path[2:]) if schema is not None else None
    return None


def stored_project_expression_site(
    project: Mapping[str, Any], path: StoredPath,
) -> ExpressionSiteKind | None:
    """Resolve declarations for variants, whose paths start at the authored project."""
    if len(path) < 3:
        return None
    family, name, *member_path = path
    if family not in {"documents", "steps"}:
        return None
    member = project.get(family, {}).get(name, {})
    if family == "documents" and member_path[0] == "payload":
        return stored_expression_site(member.get("schema", ""), tuple(member_path[1:]))
    if family == "steps":
        if member_path[:2] == ["base", "inline"]:
            return stored_step_expression_site(member.get("kind", ""), tuple(member_path[2:]))
        return _expression_site_at(Step, tuple(member_path))
    return None


@cache
def _schema_inventory(extra_models: tuple[Any, ...] = ()) -> tuple[tuple[type[BaseModel], ...], dict[type[BaseModel], set[str]]]:
    """Reachable stored models and the tag fields that discriminate their actual unions."""
    from feedbax.contracts.project import BasedDocument, Project, Step
    from feedbax.contracts.project_io import StudioProjectFile, TextProjectManifest
    from feedbax.contracts.studio_project import PresentationDocument

    roots = (RootDocument, BasedDocument, Project, Step, TextProjectManifest, StudioProjectFile, PresentationDocument)
    models: set[type[BaseModel]] = set()
    seen: set[Any] = set()
    tags: dict[type[BaseModel], set[str]] = defaultdict(set)

    def variants(annotation: Any) -> Iterator[type[BaseModel]]:
        if isinstance(annotation, TypeAliasType):
            yield from variants(annotation.__value__)
        elif get_origin(annotation) is Annotated:
            yield from variants(get_args(annotation)[0])
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield annotation
        else:
            for item in get_args(annotation):
                yield from variants(item)

    def visit(annotation: Any, discriminator: str | None = None) -> None:
        if discriminator is not None:
            for model in variants(annotation):
                tags[model].add(discriminator)
        if annotation in seen:
            return
        seen.add(annotation)
        if isinstance(annotation, TypeAliasType):
            visit(annotation.__value__)
        elif get_origin(annotation) is Annotated:
            inner, *metadata = get_args(annotation)
            tag = next((item.discriminator for item in metadata if isinstance(item, FieldInfo)
                        and isinstance(item.discriminator, str)), None)
            visit(inner, tag)
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            annotation.model_rebuild()
            models.add(annotation)
            for field in annotation.model_fields.values():
                visit(field.rebuild_annotation(), field.discriminator if isinstance(field.discriminator, str) else None)
        else:
            for item in get_args(annotation):
                visit(item)

    for model in (*roots, *stored_document_models().values(), *extra_models):
        if model is not None:
            visit(model)
    return tuple(sorted(models, key=_model_name)), tags


def _model_name(model: type[BaseModel]) -> str:
    return f"{model.__module__}.{model.__qualname__}"


@cache
def _field_default(model: type[BaseModel], name: str) -> JsonValue:
    field = model.model_fields[name]
    if field.default_factory_takes_validated_data:
        raise ValueError(f"{_model_name(model)}.{name} has a data-dependent stored default")
    value = field.get_default(call_default_factory=True)
    if _expression_site_at(field.annotation, ()) is not None:
        codec = TypeAdapter(field.rebuild_annotation())
        return codec.dump_python(codec.validate_python(value), mode="json", by_alias=True)
    return to_jsonable_python(value, by_alias=True)


def _key(name: str, field: FieldInfo) -> str:
    return field.serialization_alias or field.alias or name


def stored_defaults_manifest(installed: Iterable[type[BaseModel]]) -> dict[str, JsonValue]:
    """Describe constant authored defaults from declared and supplied model roots."""
    from feedbax.contracts.stored_regions import inline_models

    models, tags = _schema_inventory((*inline_models(), *installed))
    return {
        _model_name(model): {
            _key(name, field): _field_default(model, name)
            for name, field in model.model_fields.items()
            if not field.is_required() and not field.exclude and name not in tags.get(model, ())
        }
        for model in models
    }


def _payload_model(document: RootDocument) -> BaseModel | None:
    model = stored_document_models().get(document.schema_id)
    if model is None:
        return None
    validate_canonical_json_v2(document.payload)
    try:
        return model.model_validate(document.payload)
    except ValidationError:
        # These payloads can be saved in Studio with planner Problems.
        from feedbax.contracts.result_address import ANALYSIS_SCHEMA
        from feedbax.contracts.report_document import ReportDocument

        if document.schema_id in (
            ANALYSIS_SCHEMA, STORED_FIGURE_DOCUMENT_SCHEMA_ID, ReportDocument.model_fields["schema_version"].default,
            "subset",
        ):
            return None
        raise


def _same_stored_value(left: JsonValue, right: JsonValue, *, real_site: bool) -> bool:
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return False
        if left.get("kind") == right.get("kind") == "literal" and not real_site:
            if type(left.get("value")) is not type(right.get("value")):
                return False
        return all(_same_stored_value(value, right[key], real_site=real_site) for key, value in left.items())
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            _same_stored_value(a, b, real_site=real_site) for a, b in zip(left, right)
        )
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    return type(left) is type(right) and left == right


def _projected_file(
    model: BaseModel, exclude: set[str] | None, protected: Iterable[StoredPath],
    *, keep_explicit: bool = False, registries: StoredRegionResolver | None = None,
) -> tuple[JsonValue, set[StoredPath]]:
    from feedbax.contracts.stored_regions import StoredRegion, declared_stored_region, step_inline_model, typed_region

    resolve_region = registries.region if registries is not None else declared_stored_region
    real_tokens: set[StoredPath] = set()
    prefixes = {path[:end] for path in protected for end in range(1, len(path) + 1)}
    _, base_tags = _schema_inventory()
    tags = {model: set(fields) for model, fields in base_tags.items()}
    registered_here: set[type[BaseModel]] = set()

    def project_region(region: StoredRegion, child: dict, path: StoredPath, *, inline=False) -> JsonValue:
        context = {key: value for key, value in child.items() if key in region.contextual}
        declared = {key: value for key, value in child.items() if key not in context}
        typed, encoded, complete = typed_region(region, declared, inline=inline)
        if not complete and not inline:
            authored = region.authored_values(child)
            remember_reals(authored, path)
            return authored
        cls = type(typed)
        if cls not in registered_here:
            registered_here.add(cls)
            _, found = _schema_inventory((cls,))
            for owner, fields in found.items():
                tags.setdefault(owner, set()).update(fields)
        projected = project(typed, encoded, path) | context
        if inline:
            known = {_key(name, field) for name, field in cls.model_fields.items()}
            return projected | {key: value for key, value in encoded.items() if key not in known}
        return region.authored_values(projected)

    def remember_reals(value: JsonValue, path: StoredPath) -> None:
        if type(value) is float:
            real_tokens.add(path)
        elif isinstance(value, dict):
            for key, child in value.items():
                remember_reals(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                remember_reals(child, (*path, str(index)))

    def remember_variation(value: JsonValue, path: StoredPath) -> None:
        params = value.get("params") if isinstance(value, dict) else None
        edits = params.get("variation") if isinstance(params, dict) else None
        entries = edits.items() if isinstance(edits, dict) else enumerate(edits) if isinstance(edits, list) else ()
        for key, edit in entries:
            if isinstance(edit, dict) and isinstance(edit.get("path"), str) and "value" in edit:
                remember_reals(edit["value"], (*path, "params", "variation", str(key), "value"))

    def project(value: Any, encoded: JsonValue, path: StoredPath) -> JsonValue:
        if isinstance(value, RootModel):
            return project(value.root, encoded, path)
        if isinstance(value, BaseModel) and isinstance(encoded, dict):
            result = {}
            cls = type(value)
            for name, field in cls.model_fields.items():
                if field.exclude or not path and exclude and name in exclude:
                    continue
                key = _key(name, field)
                child_path = (*path, key)
                explicit = name in value.model_fields_set
                tagged = name in tags.get(cls, ())
                required = field.is_required() or tagged
                observed = explicit and (
                    child_path in prefixes or isinstance(value, _REFINEMENTS)
                    or isinstance(value, OverridePatch) and name == "value"
                    or keep_explicit and key in encoded
                )
                if key not in encoded and not observed:
                    continue
                raw = getattr(value, name)
                child = encoded[key] if key in encoded else to_jsonable_python(raw, by_alias=True)
                region = resolve_region(value, name) if isinstance(child, dict) and (explicit or child) else None
                if region is not None:
                    if region.model is None:
                        remember_reals(child, child_path)
                        result[key] = child
                        continue
                    if not (isinstance(child.get("document"), str) and child.keys() <= {"document", "pointer", "unresolved"}):
                        child = project_region(region, child, child_path)
                        raw = child
                if not (required or observed) and (
                    isinstance(value, _REFINEMENTS) or _same_stored_value(
                        child, _field_default(cls, name),
                        real_site=isinstance(value, (ObjectiveTerm, StoredObjectiveTerm)) and name == "weight",
                    )
                ):
                    continue
                if isinstance(value, RootDocument) and name == "payload":
                    if isinstance(child, dict) and child.get("evaluation_type") == ROLLOUT_TYPE:
                        remember_variation(child, child_path)
                    typed = _payload_model(value)
                    if typed is not None:
                        raw = typed
                        child = typed.model_dump(mode="json", by_alias=True)
                if isinstance(value, _REAL_TOKEN_VALUES) and name == "value":
                    remember_reals(child, child_path)
                elif isinstance(value, Step) and name == "base" and isinstance(raw, InlineBase):
                    body_path = (*child_path, "inline")
                    if value.kind == "evaluate":
                        remember_variation(raw.inline, body_path)
                    result[key] = {"inline": project_region(
                        StoredRegion(step_inline_model(value.kind, raw.inline)), raw.inline, body_path, inline=True,
                    )}
                    continue
                result[key] = project(raw, child, child_path)
            return result
        if isinstance(value, (tuple, list)) and isinstance(encoded, list):
            return [project(raw, item, (*path, str(index))) for index, (raw, item) in enumerate(zip(value, encoded, strict=True))]
        if isinstance(value, dict) and isinstance(encoded, dict):
            return {key: project(value.get(key), child, (*path, key)) for key, child in encoded.items()}
        return encoded

    return project(model, model.model_dump(mode="json", by_alias=True), ()), real_tokens


def stored_project_value(
    model: BaseModel, *, exclude: set[str] | None = None, protected: Iterable[StoredPath] = (),
    keep_explicit: bool = False, registries: StoredRegionResolver | None = None,
) -> JsonValue:
    """Project typed stored fields, retaining explicit defaults when a patch view needs their presence."""
    return _projected_file(model, exclude, protected, keep_explicit=keep_explicit, registries=registries)[0]


def _scalar(value: JsonValue, *, keep_real: bool = False) -> str:
    text = rfc8785.dumps(value).decode("utf-8")
    if type(value) is float and "e" not in text and "." not in text:
        if abs(value) >= 2**53:
            return repr(value)
        if keep_real:
            return text + ".0"
    return text


class _Layout:
    def __init__(self, real_tokens: set[StoredPath]):
        self.real_tokens = real_tokens

    def scalar(self, value: JsonValue, path: StoredPath) -> str:
        return _scalar(value, keep_real=path in self.real_tokens)

    def tokens(self, value: JsonValue, path: StoredPath) -> Iterator[str]:
        if isinstance(value, dict):
            yield "{"
            for index, (key, child) in enumerate(value.items()):
                if index:
                    yield ", "
                yield _scalar(key)
                yield ": "
                yield from self.tokens(child, (*path, key))
            yield "}"
        elif isinstance(value, list):
            yield "["
            for index, child in enumerate(value):
                if index:
                    yield ", "
                yield from self.tokens(child, (*path, str(index)))
            yield "]"
        else:
            yield self.scalar(value, path)

    def flat(self, value: JsonValue, width: int, path: StoredPath) -> str | None:
        parts = []
        for part in self.tokens(value, path):
            width -= len(part)
            if width < 0:
                return None
            parts.append(part)
        return "".join(parts)

    def filled(self, values: list[JsonValue], indent: str, path: StoredPath) -> Iterator[str]:
        line = indent
        for index, value in enumerate(values):
            item = self.scalar(value, (*path, str(index))) + ("," if index + 1 < len(values) else "")
            if len(line) > len(indent):
                if len(line) + 1 + len(item) <= _WIDTH:
                    line += " "
                else:
                    yield line
                    line = indent
            line += item
        if values:
            yield line

    def lines(self, value: JsonValue, level: int, key: str, suffix: str, path: StoredPath) -> Iterator[str]:
        head = " " * level + key
        if not isinstance(value, (dict, list)):
            yield head + self.scalar(value, path) + suffix
            return
        flat = self.flat(value, _WIDTH - len(head) - len(suffix), path) if level or not value else None
        if flat is not None:
            yield head + flat + suffix
            return
        if isinstance(value, dict):
            yield head + "{"
            for index, (name, child) in enumerate(value.items()):
                comma = "," if index + 1 < len(value) else ""
                yield from self.lines(child, level + 2, _scalar(name) + ": ", comma, (*path, name))
            yield " " * level + "}" + suffix
        else:
            yield head + "["
            if all(not isinstance(item, (dict, list)) for item in value):
                yield from self.filled(value, " " * (level + 2), path)
            else:
                for index, child in enumerate(value):
                    comma = "," if index + 1 < len(value) else ""
                    yield from self.lines(child, level + 2, "", comma, (*path, str(index)))
            yield " " * level + "]" + suffix


def write_project_file(
    model: BaseModel, *, exclude: set[str] | None = None, protected: Iterable[StoredPath] = (),
    registries: StoredRegionResolver | None = None, source: str | None = None,
) -> bytes:
    """Encode one typed authored file with canonical numbers and 100-column layout."""
    value, real_tokens = _projected_file(model, exclude, protected, registries=registries)
    validate_canonical_json_v2(value)
    check_authored_array_budget(value, source=source)
    return ("\n".join(_Layout(real_tokens).lines(value, 0, "", "", ())) + "\n").encode("utf-8")
