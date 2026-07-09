"""Governed schema identity namespaces for Feedbax payload families.

Stable schema identities are part of Feedbax's durable contract. New emitted
families must use one of these namespaces instead of inventing flat
``feedbax.*`` names:

- ``feedbax.spec.*`` for request/spec payloads and reusable nested specs.
- ``feedbax.manifest.*`` for durable execution, artifact, provider, and manifest
  records.
- ``feedbax.run_event`` for the canonical training-run event stream envelope.
- ``feedbax.component.<component>.params`` for globally named component
  parameter payload schemas.

Component parameter version strings such as ``"1"`` remain component-local and
are migrated by ``ComponentRegistry``/``ComponentMigration``. They become global
schema identities only when an owner intentionally exports a reusable parameter
payload schema under ``feedbax.component.<component>.params``.
"""

from __future__ import annotations

from enum import Enum


class SchemaNamespaceError(ValueError):
    """Raised when a Feedbax schema identity violates the taxonomy."""


class SchemaNamespaceKind(str, Enum):
    """Top-level governed namespaces for Feedbax schema-bearing payloads."""

    SPEC = "spec"
    MANIFEST = "manifest"
    RUN_EVENT = "run_event"
    COMPONENT_PARAMS = "component_params"
    EXTERNAL = "external"


def classify_schema_identity(schema_id: str) -> SchemaNamespaceKind:
    """Classify a stable schema identity under the Feedbax namespace taxonomy."""
    # The run-conformance certificate identity is fixed by the lifecycle
    # contract that consumes it, even though it predates the governed prefixes.
    if schema_id == "feedbax.run_conformance":
        return SchemaNamespaceKind.MANIFEST
    if schema_id.startswith("feedbax.spec."):
        return SchemaNamespaceKind.SPEC
    if schema_id.startswith("feedbax.manifest."):
        return SchemaNamespaceKind.MANIFEST
    if schema_id == "feedbax.run_event":
        return SchemaNamespaceKind.RUN_EVENT
    if schema_id.startswith("feedbax.component.") and schema_id.endswith(".params"):
        return SchemaNamespaceKind.COMPONENT_PARAMS
    if schema_id.startswith("feedbax."):
        raise SchemaNamespaceError(
            "Feedbax schema identity must use a governed namespace: "
            f"schema_id={schema_id!r}, expected_prefixes=('feedbax.spec.', "
            "'feedbax.manifest.', 'feedbax.run_event', "
            "'feedbax.component.<component>.params')"
        )
    return SchemaNamespaceKind.EXTERNAL


def validate_schema_identity(schema_id: str, *, family: str) -> SchemaNamespaceKind:
    """Validate and classify one schema identity for a registered family."""
    if not schema_id:
        raise SchemaNamespaceError(f"{family} schema_id must be non-empty")
    try:
        return classify_schema_identity(schema_id)
    except SchemaNamespaceError as exc:
        raise SchemaNamespaceError(f"{family}: {exc}") from exc


def validate_schema_version(version: str, *, family: str) -> None:
    """Validate a current Feedbax schema version string.

    Historical versions may appear in explicit migration or rejection policy
    rows, but current Feedbax versions must not introduce new flat
    ``feedbax.*`` names.
    """
    if not version:
        raise SchemaNamespaceError(f"{family} current_version must be non-empty")
    if not version.startswith("feedbax."):
        return
    if version == "feedbax.run_conformance.v1":
        return
    if (
        version.startswith("feedbax.spec.")
        or version.startswith("feedbax.manifest.")
        or version.startswith("feedbax.run_event.")
        or version.startswith("feedbax.component.")
    ):
        return
    raise SchemaNamespaceError(
        "Feedbax current schema version must use a governed namespace: "
        f"family={family!r}, current_version={version!r}"
    )
