"""Declared contracts for parameter objects embedded in durable specs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, JsonValue, StrictBool, StrictFloat, StrictInt, StrictStr
from pydantic import TypeAdapter, model_validator

from feedbax.contracts.manifest import (
    ArtifactRef,
    ParentRef,
    authenticated_manifest_ref_profile,
)


ANALYSIS_PARAMS_SCHEMA_ID = "feedbax.spec.params.analysis"
ANALYSIS_PARAMS_SCHEMA_VERSION = f"{ANALYSIS_PARAMS_SCHEMA_ID}.v1"
ANALYSIS_BUNDLE_PARAMS_SCHEMA_ID = "feedbax.spec.params.analysis_bundle"
ANALYSIS_BUNDLE_PARAMS_SCHEMA_VERSION = f"{ANALYSIS_BUNDLE_PARAMS_SCHEMA_ID}.v1"
FIGURE_ASSEMBLER_PARAMS_SCHEMA_ID = "feedbax.spec.params.figure_assembler"
FIGURE_ASSEMBLER_PARAMS_SCHEMA_VERSION = f"{FIGURE_ASSEMBLER_PARAMS_SCHEMA_ID}.v1"
FIGURE_TRACE_PARAMS_SCHEMA_ID = "feedbax.spec.params.figure_trace"
FIGURE_TRACE_PARAMS_SCHEMA_VERSION = f"{FIGURE_TRACE_PARAMS_SCHEMA_ID}.v1"

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_ARTIFACT_AUTHORITY_FIELDS = frozenset(
    {"artifact_id", "sha256", "media_type", "size_bytes", "storage_backend", "uri"}
)


class ParameterContractError(ValueError):
    """One addressed parameter object failed its declared schema."""

    def __init__(self, path: str, schema: "ParameterSchema", cause: Exception):
        self.path = path
        self.schema = schema
        self.cause = cause
        super().__init__(
            f"{path} does not satisfy parameter schema {schema.schema_version!r}: {cause}"
        )


def _validate_schema_pairs(value: Mapping[str, Any], *, path: str) -> None:
    stems: set[str] = set()
    for key in value:
        if key == "schema_id" or key == "schema_version":
            stems.add("")
        elif key.endswith("_schema_id"):
            stems.add(key[: -len("_schema_id")])
        elif key.endswith("_schema_version"):
            stems.add(key[: -len("_schema_version")])
    for stem in stems:
        id_key = "schema_id" if not stem else f"{stem}_schema_id"
        version_key = "schema_version" if not stem else f"{stem}_schema_version"
        schema_id = value.get(id_key)
        schema_version = value.get(version_key)
        if not isinstance(schema_id, str) or not schema_id:
            raise ValueError(f"{path} states {version_key!r} without a nonempty {id_key!r}")
        if not isinstance(schema_version, str) or not schema_version:
            raise ValueError(f"{path} states {id_key!r} without a nonempty {version_key!r}")
        if not schema_version.startswith(f"{schema_id}.v"):
            raise ValueError(
                f"{path} schema version {schema_version!r} is not versioned under {schema_id!r}"
            )


def _validate_durable_structures(value: Any, *, path: str = "params") -> None:
    if isinstance(value, Mapping):
        _validate_schema_pairs(value, path=path)
        authority_fields = _ARTIFACT_AUTHORITY_FIELDS.intersection(value)
        if {"sha256", "size_bytes", "media_type"}.issubset(authority_fields):
            try:
                ArtifactRef.model_validate(value)
            except Exception as exc:
                raise ValueError(
                    f"{path} looks like an artifact reference but does not satisfy the "
                    "declared ArtifactRef contract"
                ) from exc
        for key, item in value.items():
            if key.endswith("artifact_authority"):
                try:
                    ArtifactRef.model_validate(item)
                except Exception as exc:
                    raise ValueError(
                        f"{path}.{key} must use the declared ArtifactRef contract"
                    ) from exc
            if key.endswith("manifest_authority"):
                try:
                    authority = ParentRef.model_validate(item)
                except Exception as exc:
                    raise ValueError(
                        f"{path}.{key} must use the declared ParentRef contract"
                    ) from exc
                if authenticated_manifest_ref_profile(authority) is None:
                    raise ValueError(
                        f"{path}.{key} must carry authenticated manifest_sha256 and "
                        "size_bytes metadata"
                    )
            if key == "checkpoint_transaction":
                try:
                    ParentRef.model_validate(item)
                except Exception as exc:
                    raise ValueError(
                        f"{path}.{key} must use the declared ParentRef contract"
                    ) from exc
            _validate_durable_structures(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_durable_structures(item, path=f"{path}[{index}]")


class _OpenParams(BaseModel):
    """JSON parameters with deliberate extension fields and guarded durable structures."""

    model_config = ConfigDict(extra="allow")
    schema_id: ClassVar[str]
    schema_version: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def _validate_object(cls, value: Any) -> Any:
        parsed = _JSON_OBJECT.validate_python(value)
        _validate_durable_structures(parsed)
        return parsed


class AnalysisParams(_OpenParams):
    """Open scientific parameters identified by the containing ``analysis_type``."""

    schema_id = ANALYSIS_PARAMS_SCHEMA_ID
    schema_version = ANALYSIS_PARAMS_SCHEMA_VERSION
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"$id": ANALYSIS_PARAMS_SCHEMA_VERSION},
    )


class AnalysisBundleParams(_OpenParams):
    """Open shared bundle parameters identified by the bundle schema and path."""

    schema_id = ANALYSIS_BUNDLE_PARAMS_SCHEMA_ID
    schema_version = ANALYSIS_BUNDLE_PARAMS_SCHEMA_VERSION
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"$id": ANALYSIS_BUNDLE_PARAMS_SCHEMA_VERSION},
    )


class FigureAssemblerParams(_OpenParams):
    """Open assembler vocabulary identified by the figure schema and field role."""

    schema_id = FIGURE_ASSEMBLER_PARAMS_SCHEMA_ID
    schema_version = FIGURE_ASSEMBLER_PARAMS_SCHEMA_VERSION
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"$id": FIGURE_ASSEMBLER_PARAMS_SCHEMA_VERSION},
    )


class FigureTraceParams(_OpenParams):
    """Open, typed styling vocabulary shared by all durable figure trace locations."""

    schema_id = FIGURE_TRACE_PARAMS_SCHEMA_ID
    schema_version = FIGURE_TRACE_PARAMS_SCHEMA_VERSION
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"$id": FIGURE_TRACE_PARAMS_SCHEMA_VERSION},
    )

    color: StrictStr | None = None
    label: StrictStr | None = None
    line_dash: StrictStr | None = None
    line_width: StrictFloat | StrictInt | None = None
    show_band: StrictBool | None = None
    showlegend: StrictBool | None = None
    opacity: StrictFloat | StrictInt | None = None
    show_mean: StrictBool | None = None
    start_marker: JsonValue | None = None
    marker_size: StrictFloat | StrictInt | None = None
    marker_symbol: StrictStr | None = None
    error_bar_thickness: StrictFloat | StrictInt | None = None
    error_bar_width: StrictFloat | StrictInt | None = None


@dataclass(frozen=True)
class ParameterSchema:
    """One publishable parameter schema used by the compile-time authority."""

    schema_id: str
    schema_version: str
    model_ref: tuple[str, str]

    def model(self) -> type[BaseModel]:
        from importlib import import_module

        module, attribute = self.model_ref
        return getattr(import_module(module), attribute)


@dataclass(frozen=True)
class ParameterBinding:
    """Address parameter objects and select their one declared schema."""

    paths: tuple[str, ...]
    schemas: Mapping[str, ParameterSchema]
    discriminator: str | None = None
    default_schema: ParameterSchema | None = None

    def schema_for(self, document: Mapping[str, Any]) -> ParameterSchema | None:
        if self.discriminator is None:
            return self.default_schema
        return self.schemas.get(str(document.get(self.discriminator)), self.default_schema)

    def objects(self, document: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
        found: list[tuple[str, Any]] = []
        for path in self.paths:
            found.extend(_objects_at_path(document, path.split(".")))
        return tuple(found)


def _objects_at_path(value: Any, parts: list[str], path: str = "") -> list[tuple[str, Any]]:
    if not parts:
        return [(path, value)]
    part, *rest = parts
    if part == "*":
        if isinstance(value, Mapping):
            items = value.items()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = enumerate(value)
        else:
            return []
        return [
            item
            for key, child in items
            for item in _objects_at_path(child, rest, f"{path}.{key}".lstrip("."))
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            item
            for index, child in enumerate(value)
            for item in _objects_at_path(child, parts, f"{path}.{index}".lstrip("."))
        ]
    if not isinstance(value, Mapping) or part not in value:
        return []
    return _objects_at_path(value[part], rest, f"{path}.{part}".lstrip("."))


ANALYSIS_PARAMS_SCHEMA = ParameterSchema(
    ANALYSIS_PARAMS_SCHEMA_ID,
    ANALYSIS_PARAMS_SCHEMA_VERSION,
    (__name__, "AnalysisParams"),
)
ANALYSIS_BUNDLE_PARAMS_SCHEMA = ParameterSchema(
    ANALYSIS_BUNDLE_PARAMS_SCHEMA_ID,
    ANALYSIS_BUNDLE_PARAMS_SCHEMA_VERSION,
    (__name__, "AnalysisBundleParams"),
)
FIGURE_ASSEMBLER_PARAMS_SCHEMA = ParameterSchema(
    FIGURE_ASSEMBLER_PARAMS_SCHEMA_ID,
    FIGURE_ASSEMBLER_PARAMS_SCHEMA_VERSION,
    (__name__, "FigureAssemblerParams"),
)
FIGURE_TRACE_PARAMS_SCHEMA = ParameterSchema(
    FIGURE_TRACE_PARAMS_SCHEMA_ID,
    FIGURE_TRACE_PARAMS_SCHEMA_VERSION,
    (__name__, "FigureTraceParams"),
)


__all__ = [
    "ANALYSIS_BUNDLE_PARAMS_SCHEMA",
    "ANALYSIS_BUNDLE_PARAMS_SCHEMA_ID",
    "ANALYSIS_BUNDLE_PARAMS_SCHEMA_VERSION",
    "ANALYSIS_PARAMS_SCHEMA",
    "ANALYSIS_PARAMS_SCHEMA_ID",
    "ANALYSIS_PARAMS_SCHEMA_VERSION",
    "FIGURE_ASSEMBLER_PARAMS_SCHEMA",
    "FIGURE_ASSEMBLER_PARAMS_SCHEMA_ID",
    "FIGURE_ASSEMBLER_PARAMS_SCHEMA_VERSION",
    "FIGURE_TRACE_PARAMS_SCHEMA",
    "FIGURE_TRACE_PARAMS_SCHEMA_ID",
    "FIGURE_TRACE_PARAMS_SCHEMA_VERSION",
    "AnalysisBundleParams",
    "AnalysisParams",
    "FigureAssemblerParams",
    "FigureTraceParams",
    "ParameterBinding",
    "ParameterContractError",
    "ParameterSchema",
]
