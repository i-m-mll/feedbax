"""Schema identities shared by execution producers and governed readers."""

from typing import Final

MAPPED_METRIC_VALUE_SCHEMA_ID = "feedbax.manifest.mapped_metric_value"
MAPPED_METRIC_VALUE_SCHEMA_VERSION = "feedbax.manifest.mapped_metric_value.v1"
STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_ID = "feedbax.manifest.structured_mapped_metric_value"
STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_VERSION = "feedbax.manifest.structured_mapped_metric_value.v1"

EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID = "feedbax.spec.execution_identity_envelope"
EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION_V1 = "feedbax.spec.execution_identity_envelope.v1"
EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION = "feedbax.spec.execution_identity_envelope.v2"

REPO_REALIZATION_PLAN_SCHEMA_ID = "feedbax.orchestration.repo_realization_plan"
REPO_REALIZATION_PLAN_SCHEMA_VERSION = (
    "feedbax.orchestration.repo_realization_plan.v1"
)

REPO_SNAPSHOT_MANIFEST_SCHEMA_ID: Final = "feedbax.orchestration.repo_snapshot_manifest"
REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION_V1: Final = "feedbax.orchestration.repo_snapshot_manifest.v1"
REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION_V2: Final = "feedbax.orchestration.repo_snapshot_manifest.v2"
REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION: Final = "feedbax.orchestration.repo_snapshot_manifest.v3"
