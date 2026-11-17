"""Source documents for authored expression contexts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import model_validator

from feedbax.contracts._authoring_inputs import AuthoringTextReader, read_authoring_text
from feedbax.contracts.base import StrictModel
from feedbax.contracts.expressions import ContextItem, ExpressionContext, ValueExpr, evaluate_query
from feedbax.contracts.strict_json import strict_json_loads


class SourceBinding(StrictModel):
    """Source JSON bound into the extraction expression context."""

    alias: str
    kind: str
    uri: str
    payload_query: ValueExpr | None = None
    optional: bool = False
    missing_payload: Any | None = None

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
        has_missing_payload = "missing_payload" in self.model_fields_set
        if self.optional and not has_missing_payload:
            raise ValueError("SourceBinding missing_payload is required when optional=True")
        if not self.optional and has_missing_payload:
            raise ValueError("SourceBinding missing_payload is only allowed when optional=True")
        return self


def load_expression_context(
    sources: list[SourceBinding],
    repo_root: Path | str,
    *,
    read_text: AuthoringTextReader = read_authoring_text,
) -> ExpressionContext:
    """Load source bindings into an expression context from repo-relative JSON files."""
    repo_root = Path(repo_root)
    items: dict[str, ContextItem] = {}
    for source in sources:
        try:
            payload = _read_source_payload(repo_root, source.uri, read_text=read_text)
        except FileNotFoundError:
            if not source.optional:
                raise
            items[source.alias] = ContextItem(kind=source.kind, payload=deepcopy(source.missing_payload))
            continue
        if source.payload_query is not None:
            payload = evaluate_query(
                source.payload_query,
                ExpressionContext(
                    items={"source": ContextItem(kind=source.kind, payload=payload)}
                ),
            )
        items[source.alias] = ContextItem(kind=source.kind, payload=payload)
    return ExpressionContext(items=items)


def _read_source_payload(repo_root: Path, uri: str, *, read_text: AuthoringTextReader) -> Any:
    path = repo_root / uri
    try:
        text = read_text(path, require_file=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Extraction source is missing: {uri}") from exc
    payload = strict_json_loads(text, ref=uri)
    # Resolve declared content-pinned sub-document inheritance into the effective
    # document before any dotted-path query evaluation. Imported locally because
    # matrix_core imports this module at top level; a top-level import here would
    # create a cycle. Documents without the reserved key pass through unchanged.
    from feedbax.contracts.matrix_core import materialize_inherited_document

    return materialize_inherited_document(payload, repo_root=repo_root, read_text=read_text)


def set_dotted_path(root: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path in a nested mapping, creating missing mappings."""
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
