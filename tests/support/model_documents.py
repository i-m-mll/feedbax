"""Author a Model change for one step as the document form holds it: a document based on the one it reads."""

from typing import Any

from feedbax.compiler.model_attachment import resolve_effective_model
from feedbax.contracts.model_attachment import MODEL_DIFFERENCE_BATCHES
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.project import Project, diff
from feedbax.contracts.project_plan import _builtin_model_registry
from feedbax.contracts.project_resolution import compose_document


def changed_model(payload: dict[str, Any], records: list[dict[str, Any]], components=None) -> dict[str, Any]:
    """``payload`` with the addressed Model ``records`` applied, in its normalized spelling."""
    differences = MODEL_DIFFERENCE_BATCHES.validate_python({"records": records})["records"]
    return resolve_effective_model(
        ModelDocument.model_validate(payload), components or _builtin_model_registry(), tuple(differences),
    ).model_dump(mode="json")


def read_changed_model(
    raw: dict[str, Any], step: str, records: list[dict[str, Any]], *, base: str | None = None,
    name: str | None = None, components=None,
) -> str:
    """Point ``step`` at a new document based on the Model it reads, holding ``records``; return its name.

    ``base`` is the document the step reads, its own ``trial_documents.model`` when not given.
    """
    inline = raw["steps"][step]["base"]["inline"]
    slots = inline.setdefault("trial_documents", {})
    base = base or slots["model"]["document"]
    payload = compose_document(Project.model_validate({"documents": raw["documents"]}).documents.get, base).payload
    normalized = ModelDocument.model_validate(payload).model_dump(mode="json")
    name = name or f"{base}-{step}"
    raw["documents"][name] = {
        "schema": MODEL_DOCUMENT_SCHEMA_ID, "base": {"document": base},
        "differences": {
            path: record.model_dump(mode="json")
            for path, record in diff(normalized, changed_model(payload, records, components)).items()
        },
    }
    slots["model"] = {"document": name}
    return name
