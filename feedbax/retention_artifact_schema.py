"""Schema identities for training retention artifacts."""

from __future__ import annotations

from typing import Any


RETENTION_PLAN_SCHEMA_ID = "feedbax.manifest.training.retention_plan"
RETENTION_PLAN_SCHEMA_VERSION = "feedbax.manifest.training.retention_plan.v1"
RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_ID = (
    "feedbax.manifest.training.retained_observables"
)
RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_VERSION = (
    "feedbax.manifest.training.retained_observables.v1"
)

RETAINED_OBSERVABLE_PLAN_SCHEMA_ID = (
    "feedbax.manifest.training.retained_observable_plan"
)
RETAINED_OBSERVABLE_PLAN_SCHEMA_VERSION = (
    "feedbax.manifest.training.retained_observable_plan.v1"
)
RETENTION_POLICY_PLAN_SCHEMA_ID = "feedbax.manifest.training.retention_policy_plan"
RETENTION_POLICY_PLAN_SCHEMA_VERSION = (
    "feedbax.manifest.training.retention_policy_plan.v1"
)
LOSS_TERM_PLAN_SCHEMA_ID = "feedbax.manifest.training.loss_term_plan"
LOSS_TERM_PLAN_SCHEMA_VERSION = "feedbax.manifest.training.loss_term_plan.v1"

RETENTION_ARTIFACT_ROLE_SCHEMAS: dict[str, tuple[str, str, str]] = {
    "retention_plan": (
        "RetentionPlan",
        RETENTION_PLAN_SCHEMA_ID,
        RETENTION_PLAN_SCHEMA_VERSION,
    ),
    "retained_observables": (
        "RetainedObservablesArtifact",
        RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_ID,
        RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_VERSION,
    ),
}


def retention_artifact_metadata(role: str) -> dict[str, str]:
    """Return schema metadata for a retention artifact role."""
    _kind, schema_id, schema_version = retention_artifact_schema(role)
    return {"schema_id": schema_id, "schema_version": schema_version}


def retention_artifact_schema(role: str) -> tuple[str, str, str]:
    """Return ``(kind, schema_id, schema_version)`` for a retention artifact role."""
    try:
        return RETENTION_ARTIFACT_ROLE_SCHEMAS[role]
    except KeyError as exc:
        known = ", ".join(sorted(RETENTION_ARTIFACT_ROLE_SCHEMAS))
        raise ValueError(f"Unknown retention artifact role {role!r}; known roles: {known}") from exc


def retained_observables_to_json(observables: Any) -> dict[str, Any]:
    """Wrap retained-observable data as a governed artifact payload."""
    return {
        "schema_id": RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_ID,
        "schema_version": RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_VERSION,
        "observables": observables,
    }
