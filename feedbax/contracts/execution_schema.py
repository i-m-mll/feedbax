"""Schema identities shared by execution producers and governed readers."""

from typing import Final

REPO_REALIZATION_PLAN_SCHEMA_ID = "feedbax.orchestration.repo_realization_plan"
REPO_REALIZATION_PLAN_SCHEMA_VERSION = (
    "feedbax.orchestration.repo_realization_plan.v1"
)

REPO_SNAPSHOT_MANIFEST_SCHEMA_ID: Final = "feedbax.orchestration.repo_snapshot_manifest"
REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION_V1: Final = "feedbax.orchestration.repo_snapshot_manifest.v1"
REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION_V2: Final = "feedbax.orchestration.repo_snapshot_manifest.v2"
REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION: Final = "feedbax.orchestration.repo_snapshot_manifest.v3"
