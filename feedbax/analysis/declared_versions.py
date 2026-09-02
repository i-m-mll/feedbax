"""One declared-schema-version resolution path for authored analysis documents.

Authored analysis documents — analysis runs, evaluation run matrices, and
analysis bundles — are durable artifacts. A document written against an older
schema must migrate through an explicit registered rule or fail closed with an
actionable error; a version is never inferred and never retried through a
compatibility shim.

The three authoring resolvers previously each called the structured spec
migration registry with their own argument set, so the same skeleton carried
three different migration behaviours. This module is the single place where an
authored document's own ``schema_version`` declaration is read and handed to the
registry, so the resolvers cannot drift again.

Versionless documents fail closed. The loaders consume durable authored material;
they do not stamp new current payloads and therefore must never infer a version.
"""

from __future__ import annotations

from typing import Any, Mapping

from feedbax.contracts.migrations import (
    SpecMigrationResult,
    SpecSchemaRegistry,
    UnsupportedSpecVersion,
    default_spec_registry,
    migrate_structured_spec_payload,
)

__all__ = [
    "migrate_authored_document",
]


def migrate_authored_document(
    kind: str,
    document: Mapping[str, Any],
    *,
    path: str = "spec",
    registry: SpecSchemaRegistry | None = None,
) -> SpecMigrationResult:
    """Accept, migrate, or reject one authored document from its own declaration.

    The document's declared ``schema_version`` is always the migration source.
    A supported older version migrates through its registered rules; an
    explicitly rejected or unknown version fails closed with the registry's
    actionable diagnostic. A declaration that is present but is not a non-empty
    string is malformed and fails closed here, because a malformed declaration
    must not fall through to a version-free code path.

    Presence is decided by the key, not by its value: a document carrying
    ``"schema_version": null`` has declared a version and declared it malformed.
    A document with no ``schema_version`` key at all is versionless and also fails
    closed. Reading the declaration with a value test would collapse those two
    distinct invalid cases.

    Args:
        kind: Registered structured spec family, e.g. ``"AnalysisRunSpec"``.
        document: The authored document payload.
        path: Spec path recorded on emitted migration records.
        registry: Structured spec registry; defaults to the process registry.

    Returns:
        The migration result whose ``payload`` is at the family's current version.

    Raises:
        UnsupportedSpecVersion: The declared version is malformed, explicitly
            unsupported, or has no registered migration path to the current
            version; or the document is versionless.
    """
    active_registry = registry or default_spec_registry
    is_declared = "schema_version" in document
    declared = document.get("schema_version")
    if is_declared and not (isinstance(declared, str) and declared):
        family = active_registry.resolve(kind)
        raise UnsupportedSpecVersion(
            "Authored Feedbax document declares a malformed schema_version: "
            f"family={family.kind!r}, schema_id={family.identity!r}, path={path!r}, "
            f"schema_version={declared!r} of type {type(declared).__name__!r}, "
            f"current_version={family.current_version!r}; declare the version string "
            "the document was authored against so it can migrate or be rejected."
        )
    return migrate_structured_spec_payload(
        kind,
        document,
        source_version=declared,
        assume_current=False,
        path=path,
        registry=active_registry,
    )
