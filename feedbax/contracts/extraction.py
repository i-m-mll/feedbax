"""Declarative extraction specs for governed analysis data products.

Extraction source files are loaded from the filesystem before their payloads are
bound into the expression context. That filesystem-existence check is separate
from ``Compare(op="exists")``, which only checks paths inside already-bound
payloads.

Additive changelog, 2026-07-05: v1 accepts structural value expressions
(``ValueQuery`` or ``Coalesce``) in field mappings and source payload queries.
Optional source placeholders and residual passthrough are additive fields in the
same v1 schema family.
"""

from __future__ import annotations

from copy import deepcopy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from feedbax.contracts.expressions import (
    Coalesce,
    Compare,
    ContextItem,
    ExpressionContext,
    ExpressionPathMissing,
    Select,
    ValueExpr,
    ValueQuery,
    evaluate_query,
)
from feedbax.contracts.manifest import AnalysisDataProduct, StrictModel


EXTRACTION_PRODUCT_SPEC_SCHEMA_ID = "feedbax.spec.extraction_product"
EXTRACTION_PRODUCT_SPEC_SCHEMA_VERSION = "feedbax.spec.extraction_product.v1"


class DataProductDrift(ValueError):
    """Raised when a persisted data product no longer matches extracted sources."""

    def __init__(
        self,
        *,
        output_path: str,
        source_uri: str | None,
        persisted_value: Any,
        source_value: Any,
    ) -> None:
        self.output_path = output_path
        self.source_uri = source_uri
        self.persisted_value = persisted_value
        self.source_value = source_value
        super().__init__(
            "Analysis data product drift: "
            f"output_path={output_path!r}, source_uri={source_uri!r}, "
            f"persisted_value={persisted_value!r}, source_value={source_value!r}"
        )


class ExtractionProductIdentityMismatch(ValueError):
    """Raised when an extracted product does not match its pinned identity."""


class SourceBinding(StrictModel):
    """Source JSON bound into the extraction expression context."""

    alias: str
    kind: str
    uri: str
    payload_query: ValueExpr | None = None
    adoption_note: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> "SourceBinding":
        if not self.alias.strip():
            raise ValueError("SourceBinding alias must not be empty")
        if not self.kind.strip():
            raise ValueError("SourceBinding kind must not be empty")
        if not self.uri.strip():
            raise ValueError("SourceBinding uri must not be empty")
        if Path(self.uri).is_absolute():
            raise ValueError("SourceBinding uri must be repo-relative")
        return self


class FieldMapping(StrictModel):
    """One destination parameter path populated by a value query."""

    output_path: str
    query: ValueExpr
    adopts_source_field: str | None = None

    @model_validator(mode="after")
    def _validate_output_path(self) -> "FieldMapping":
        if not self.output_path.strip():
            raise ValueError("FieldMapping output_path must not be empty")
        if any(not part for part in self.output_path.split(".")):
            raise ValueError(
                "FieldMapping output_path is not dotted-path-like: "
                f"{self.output_path!r}"
            )
        return self


class ExtractionProductSpec(StrictModel):
    """Declarative source-to-product extraction specification."""

    schema_id: str = EXTRACTION_PRODUCT_SPEC_SCHEMA_ID
    schema_version: str = EXTRACTION_PRODUCT_SPEC_SCHEMA_VERSION
    product_schema_id: str
    product_schema_version: str
    role: str
    logical_name: str
    producer_manifest_id: str
    sources: list[SourceBinding]
    fields: list[FieldMapping]
    static_parameters: dict[str, Any] = Field(default_factory=dict)
    materialization: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_identity_hash: str | None = None

    @model_validator(mode="after")
    def _validate_spec(self) -> "ExtractionProductSpec":
        if self.schema_id != EXTRACTION_PRODUCT_SPEC_SCHEMA_ID:
            raise ValueError(
                "unsupported ExtractionProductSpec schema_id: "
                f"{self.schema_id!r}, expected {EXTRACTION_PRODUCT_SPEC_SCHEMA_ID!r}"
            )
        if self.schema_version != EXTRACTION_PRODUCT_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "unsupported ExtractionProductSpec schema_version: "
                f"{self.schema_version!r}, expected {EXTRACTION_PRODUCT_SPEC_SCHEMA_VERSION!r}"
            )
        aliases = [source.alias for source in self.sources]
        if len(set(aliases)) != len(aliases):
            raise ValueError("ExtractionProductSpec sources must have unique aliases")
        source_aliases = set(aliases)
        for field in self.fields:
            undeclared = _query_items(field.query) - source_aliases
            if undeclared:
                raise ValueError(
                    "FieldMapping query item is not a declared source alias: "
                    f"output_path={field.output_path!r}, items={sorted(undeclared)!r}"
                )
        return self


def materialize_extraction_product(
    spec: ExtractionProductSpec,
    repo_root: Path | str,
) -> AnalysisDataProduct:
    """Build an :class:`AnalysisDataProduct` from declarative field mappings."""

    repo_root = Path(repo_root)
    ctx = _load_expression_context(spec, repo_root)
    parameters = deepcopy(spec.static_parameters)

    for field in spec.fields:
        value = evaluate_query(field.query, ctx)
        _set_path(parameters, field.output_path, value)

    _add_adoption_records(spec, ctx, parameters)

    product = AnalysisDataProduct(
        product_schema_id=spec.product_schema_id,
        product_schema_version=spec.product_schema_version,
        role=spec.role,
        logical_name=spec.logical_name,
        producer_manifest_id=spec.producer_manifest_id,
        parameters=parameters,
        materialization=deepcopy(spec.materialization),
        metadata=deepcopy(spec.metadata),
    )
    if (
        spec.expected_identity_hash is not None
        and product.product_identity_hash != spec.expected_identity_hash
    ):
        raise ExtractionProductIdentityMismatch(
            "Extraction product identity mismatch: "
            f"expected={spec.expected_identity_hash!r}, "
            f"actual={product.product_identity_hash!r}"
        )
    return product


def verify_extraction_product(
    spec: ExtractionProductSpec,
    persisted_product: AnalysisDataProduct | Mapping[str, Any],
    repo_root: Path | str,
) -> AnalysisDataProduct:
    """Re-extract a product and fail closed if persisted parameters drift."""

    persisted = (
        persisted_product
        if isinstance(persisted_product, AnalysisDataProduct)
        else AnalysisDataProduct.model_validate(persisted_product)
    )
    source_product = materialize_extraction_product(spec, repo_root)
    if persisted.parameters != source_product.parameters:
        source_by_output_path = _source_uri_by_output_path(spec)
        path, persisted_value, source_value = _first_difference(
            persisted.parameters,
            source_product.parameters,
            path="parameters",
        )
        output_path = path.removeprefix("parameters.")
        raise DataProductDrift(
            output_path=output_path,
            source_uri=_source_uri_for_output_path(output_path, source_by_output_path),
            persisted_value=persisted_value,
            source_value=source_value,
        )
    return persisted


def _load_expression_context(spec: ExtractionProductSpec, repo_root: Path) -> ExpressionContext:
    items: dict[str, ContextItem] = {}
    for source in spec.sources:
        payload = _read_source_payload(repo_root, source.uri)
        if source.payload_query is not None:
            payload = evaluate_query(
                source.payload_query,
                ExpressionContext(
                    items={"source": ContextItem(kind=source.kind, payload=payload)}
                ),
            )
        items[source.alias] = ContextItem(kind=source.kind, payload=payload)
    return ExpressionContext(items=items)


def _read_source_payload(repo_root: Path, uri: str) -> Any:
    path = repo_root / uri
    if not path.is_file():
        raise FileNotFoundError(f"Extraction source is missing: {uri}")
    return json.loads(path.read_text(encoding="utf-8"))


def _add_adoption_records(
    spec: ExtractionProductSpec,
    ctx: ExpressionContext,
    parameters: dict[str, Any],
) -> None:
    source_by_alias = {source.alias: source for source in spec.sources}
    groups: dict[str, list[FieldMapping]] = {}
    for field in spec.fields:
        if field.adopts_source_field is None:
            continue
        parent_path = field.output_path.rsplit(".", 1)[0]
        groups.setdefault(parent_path, []).append(field)

    for parent_path, fields in groups.items():
        source_aliases = set().union(*(_query_items(field.query) for field in fields))
        if len(source_aliases) != 1:
            raise ValueError(
                "adoption records require one source alias per output object: "
                f"output_path={parent_path!r}, aliases={sorted(source_aliases)!r}"
            )
        source = source_by_alias[next(iter(source_aliases))]
        adopted_values = deepcopy(_get_parameter_path(parameters, parent_path))
        adopted_values.pop("adoption", None)
        adoption = {
            "source_manifest": source.uri,
            "source_pointer": _source_pointer(source),
        }
        try:
            adoption["source_factor_key"] = evaluate_query(
                ValueQuery(item=source.alias, path="factor_key"),
                ctx,
            )
        except ExpressionPathMissing:
            pass
        adoption["adopted_fields"] = {
            field.output_path.rsplit(".", 1)[1]: field.adopts_source_field
            for field in fields
        }
        adoption["adopted_values"] = adopted_values
        if source.adoption_note is not None:
            adoption["note"] = source.adoption_note
        _set_path(parameters, f"{parent_path}.adoption", adoption)


def _source_pointer(source: SourceBinding) -> str:
    query = source.payload_query
    if query is None:
        return ""
    if isinstance(query, Coalesce):
        if len(query.queries) != 1:
            return ""
        query = query.queries[0]
    if query.select is None:
        return query.path if query is not None else ""
    selector = _select_pointer(query.select)
    if selector is None:
        return query.path
    return f"{query.path}[{selector}]"


def _select_pointer(select: Select) -> str | None:
    where = select.where
    if not isinstance(where, Compare):
        return None
    if where.item != "entry" or where.op not in {"eq", "approx_eq"}:
        return None
    return f"{where.path}={where.value}"


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = current.get(part)
        if next_value is None:
            next_value = {}
            current[part] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"cannot set nested output_path through non-dict segment: {path!r}")
        current = next_value
    current[parts[-1]] = value


def _get_parameter_path(root: Mapping[str, Any], path: str) -> dict[str, Any]:
    current: Any = root
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"adoption parent path is missing: {path!r}")
        current = current[part]
    if not isinstance(current, dict):
        raise ValueError(f"adoption parent path is not an object: {path!r}")
    return current


def _source_uri_by_output_path(spec: ExtractionProductSpec) -> dict[str, str]:
    source_by_alias = {source.alias: source for source in spec.sources}
    result: dict[str, str] = {}
    for field in spec.fields:
        source_uris = [
            source_by_alias[item].uri
            for item in sorted(_query_items(field.query))
            if item in source_by_alias
        ]
        if source_uris:
            result[field.output_path] = source_uris[0]
        if field.adopts_source_field is not None:
            parent_path = field.output_path.rsplit(".", 1)[0]
            if source_uris:
                result.setdefault(parent_path, source_uris[0])
    return result


def _query_items(query: ValueExpr) -> set[str]:
    if isinstance(query, ValueQuery):
        return {query.item}
    if isinstance(query, Coalesce):
        return set().union(*(_query_items(child) for child in query.queries))
    raise TypeError(f"unsupported value query type: {type(query).__name__}")


def _source_uri_for_output_path(
    output_path: str,
    source_by_output_path: Mapping[str, str],
) -> str | None:
    if output_path in source_by_output_path:
        return source_by_output_path[output_path]
    for known_path in sorted(source_by_output_path, key=len, reverse=True):
        if output_path.startswith(f"{known_path}."):
            return source_by_output_path[known_path]
    return None


def _first_difference(left: Any, right: Any, *, path: str) -> tuple[str, Any, Any]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in left:
            if key not in right:
                return f"{path}.{key}", left[key], None
            child_path, left_value, right_value = _first_difference(
                left[key],
                right[key],
                path=f"{path}.{key}",
            )
            if left_value != right_value:
                return child_path, left_value, right_value
        for key in right:
            if key not in left:
                return f"{path}.{key}", None, right[key]
        return path, left, right
    if isinstance(left, list) and isinstance(right, list):
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            child_path, left_value, right_value = _first_difference(
                left_item,
                right_item,
                path=f"{path}[{index}]",
            )
            if left_value != right_value:
                return child_path, left_value, right_value
        if len(left) != len(right):
            index = min(len(left), len(right))
            return (
                f"{path}[{index}]",
                left[index] if index < len(left) else None,
                right[index] if index < len(right) else None,
            )
        return path, left, right
    return path, left, right


__all__ = [
    "EXTRACTION_PRODUCT_SPEC_SCHEMA_ID",
    "EXTRACTION_PRODUCT_SPEC_SCHEMA_VERSION",
    "DataProductDrift",
    "ExtractionProductIdentityMismatch",
    "ExtractionProductSpec",
    "FieldMapping",
    "SourceBinding",
    "materialize_extraction_product",
    "verify_extraction_product",
]
