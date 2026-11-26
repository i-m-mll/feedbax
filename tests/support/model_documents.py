"""Author a Model change for one step as the document form holds it: a document based on the one it reads."""

from typing import Any

from feedbax.compiler.model_attachment import resolve_effective_model
from feedbax.contracts.model_attachment import MODEL_DIFFERENCE_BATCHES
from feedbax.contracts.project import Project, diff
from feedbax.planner.project_plan import _builtin_model_registry
from feedbax.contracts.project_resolution import compose_document
from feedbax.contracts.stored_model import (
    STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument, model_keyed_collection,
)


def changed_model(payload: dict[str, Any], records: list[dict[str, Any]], components=None) -> dict[str, Any]:
    """``payload`` with the addressed Model ``records`` applied, in its normalized spelling."""
    differences = MODEL_DIFFERENCE_BATCHES.validate_python({"records": records})["records"]
    return StoredModelDocument.from_lowered(resolve_effective_model(
        StoredModelDocument.model_validate(payload).lower(), components or _builtin_model_registry(), tuple(differences),
    )).model_dump(mode="json", exclude_defaults=True)


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
    name = name or f"{base}-{step}"
    raw["documents"][name] = {
        "label": "Document",
        "schema": STORED_MODEL_DOCUMENT_SCHEMA_ID, "base": {"document": base},
        "differences": {
            path: record.model_dump(mode="json")
            for path, record in diff(
                payload, changed_model(payload, records, components), keyed_collection=model_keyed_collection,
            ).items()
        },
    }
    slots["model"] = {"document": name}
    return name
