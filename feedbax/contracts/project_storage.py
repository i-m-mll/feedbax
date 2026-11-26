"""Canonical JSON for authored project files, kept separate from request identity."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from functools import cache
from types import MappingProxyType
from typing import Annotated, Any, TypeAliasType, get_args, get_origin

import rfc8785
from pydantic import BaseModel, JsonValue, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python

from feedbax.contracts.canonical_json import validate_canonical_json_v2
from feedbax.contracts.expression_values import LiteralValue
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.objective_document import ObjectiveTerm
from feedbax.contracts.project import RootDocument
from feedbax.contracts.project_sweep import AxisValue, NamedAxis, Sweep, SweepException


type StoredPath = tuple[str, ...]

_REFINEMENTS = (Sweep, NamedAxis, AxisValue, SweepException)
_WIDTH = 100


@cache
def stored_document_models() -> Mapping[str, type[BaseModel] | None]:
    """The payload models used by the authored writer, including the opaque Figure kind."""
    from feedbax.contracts.analysis_document import AnalysisPayload
    from feedbax.contracts.figure_document import FIGURE_DOCUMENT_SCHEMA
    from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID
    from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
    from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument
    from feedbax.contracts.project import LoadedModelDocument
    from feedbax.contracts.project_inputs import SubsetTerms
    from feedbax.contracts.report_document import ReportDocument
    from feedbax.contracts.result_address import ANALYSIS_SCHEMA
    from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID, SetDocument
    from feedbax.contracts.task_document import PYTHON_TASK_SCHEMA_ID, TASK_DOCUMENT_SCHEMA_ID, PythonTaskSelection, TaskDocument
    from feedbax.contracts.training_document import TRAINING_DOCUMENT_SCHEMA_ID, TrainingDocument

    return MappingProxyType({
        MODEL_DOCUMENT_SCHEMA_ID: ModelDocument,
        TASK_DOCUMENT_SCHEMA_ID: TaskDocument,
        OBJECTIVE_DOCUMENT_SCHEMA_ID: ObjectiveDocument,
        TRAINING_DOCUMENT_SCHEMA_ID: TrainingDocument,
        PYTHON_TASK_SCHEMA_ID: PythonTaskSelection,
        LOADED_MODEL_SCHEMA_ID: LoadedModelDocument,
        SET_DOCUMENT_SCHEMA_ID: SetDocument,
        ANALYSIS_SCHEMA: AnalysisPayload[JsonValue],
        FIGURE_DOCUMENT_SCHEMA: None,
        ReportDocument.model_fields["schema_version"].default: ReportDocument,
        "subset": SubsetTerms,
    })


@cache
def _schema_inventory() -> tuple[tuple[type[BaseModel], ...], dict[type[BaseModel], set[str]]]:
    """Reachable stored models and the tag fields that discriminate their actual unions."""
    from feedbax.contracts.project import BasedDocument, Project, Step
    from feedbax.contracts.project_io import StudioProjectFile, TextProjectManifest
    from feedbax.contracts.studio_project import PresentationDocument

    roots = (RootDocument, BasedDocument, Project, Step, TextProjectManifest, StudioProjectFile, PresentationDocument)
    models: set[type[BaseModel]] = set()
    seen: set[Any] = set()
    tags: dict[type[BaseModel], set[str]] = defaultdict(set, {OverridePatch: {"op"}})

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

    for model in (*roots, *stored_document_models().values()):
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
    return to_jsonable_python(value, by_alias=True)


def _key(name: str, field: FieldInfo) -> str:
    return field.serialization_alias or field.alias or name


def stored_defaults_manifest() -> dict[str, JsonValue]:
    """Declare every constant default reachable through a stored model."""
    models, tags = _schema_inventory()
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
        # Analysis, Report and Subset payloads can be saved in Studio with planner Problems.
        from feedbax.contracts.result_address import ANALYSIS_SCHEMA
        from feedbax.contracts.report_document import ReportDocument

        if document.schema_id in (ANALYSIS_SCHEMA, ReportDocument.model_fields["schema_version"].default, "subset"):
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
) -> tuple[JsonValue, set[StoredPath]]:
    real_literals: set[StoredPath] = set()
    prefixes = {path[:end] for path in protected for end in range(1, len(path) + 1)}
    _, tags = _schema_inventory()

    def project(value: Any, encoded: JsonValue, path: StoredPath) -> JsonValue:
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
                )
                raw = getattr(value, name)
                if key in encoded:
                    child = encoded[key]
                elif observed:
                    child = to_jsonable_python(raw, by_alias=True)
                else:
                    continue
                if not (required or observed) and (
                    isinstance(value, _REFINEMENTS) or _same_stored_value(
                        child, _field_default(cls, name), real_site=isinstance(value, ObjectiveTerm) and name == "weight",
                    )
                ):
                    continue
                if isinstance(value, RootDocument) and name == "payload":
                    typed = _payload_model(value)
                    if typed is not None:
                        raw = typed
                        child = typed.model_dump(mode="json", by_alias=True)
                if isinstance(value, LiteralValue) and name == "value" and type(child) is float:
                    real_literals.add(child_path)
                result[key] = project(raw, child, child_path)
            return result
        if isinstance(value, (tuple, list)) and isinstance(encoded, list):
            return [project(raw, item, (*path, str(index))) for index, (raw, item) in enumerate(zip(value, encoded, strict=True))]
        if isinstance(value, dict) and isinstance(encoded, dict):
            return {key: project(value.get(key), child, (*path, key)) for key, child in encoded.items()}
        return encoded

    return project(model, model.model_dump(mode="json", by_alias=True), ()), real_literals


def stored_project_value(
    model: BaseModel, *, exclude: set[str] | None = None, protected: Iterable[StoredPath] = (),
) -> JsonValue:
    """Project typed stored fields, retaining authored keys that a strict operation observes."""
    return _projected_file(model, exclude, protected)[0]


def _scalar(value: JsonValue, *, keep_real: bool = False) -> str:
    text = rfc8785.dumps(value).decode("utf-8")
    if type(value) is float and "e" not in text and "." not in text:
        if abs(value) >= 2**53:
            return repr(value)
        if keep_real:
            return text + ".0"
    return text


class _Layout:
    def __init__(self, real_literals: set[StoredPath]):
        self.real_literals = real_literals

    def scalar(self, value: JsonValue, path: StoredPath) -> str:
        return _scalar(value, keep_real=path in self.real_literals)

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
) -> bytes:
    """Encode one typed authored file with canonical numbers and 100-column layout."""
    value, real_literals = _projected_file(model, exclude, protected)
    validate_canonical_json_v2(value)
    return ("\n".join(_Layout(real_literals).lines(value, 0, "", "", ())) + "\n").encode("utf-8")
