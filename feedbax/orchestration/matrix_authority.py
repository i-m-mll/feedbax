"""Provider-neutral authentication for governed training-run matrices."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TrainingRunMatrixArtifactBinding,
    TrainingRunMatrixAuthority,
    TrainingRunMatrixCodeAuthority,
    TrainingRunMatrixInputCustodyBinding,
    TrainingRunMatrixMonitorBinding,
    TrainingRunMatrixRowPreflightBinding,
    TrainingRunMatrixSpec,
    TrainingRowProvenance,
)
from feedbax.contracts.spec_storage import (
    TrainingRunExecutionCapsule,
    canonicalize_immutable_input_identities,
    training_run_execution_hash,
    training_run_intent_hash,
    training_spec_sha256,
)
from feedbax.orchestration.bundle import (
    ResolvedAssemblyInput,
    RunBundle,
    RunRowSpec,
    SchemaArtifactRef,
    canonical_run_bundle_sha256,
)
from feedbax.orchestration.native_execution import is_native_training_command
from feedbax.training.run_matrix import _planned_run_id, expected_ordered_matrix_row_ids


class MatrixAuthorityError(ValueError):
    """Raised when local matrix authority cannot be authenticated."""


def is_training_matrix_bundle(bundle: RunBundle) -> bool:
    """Return whether the bundle carries governed matrix intent."""
    return any(
        row.execution.authored_intent.schema_id == TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID
        for row in bundle.rows
    )


def build_training_run_matrix_authority(
    bundle: RunBundle,
    *,
    local_repos: Mapping[str, str | Path],
    protected_refs: Mapping[str, str],
    expected_bundle_sha256: str | None = None,
    expected_run_set_id: str | None = None,
) -> TrainingRunMatrixAuthority:
    """Authenticate provider-unverified authority for an exact assembled matrix."""
    bundle_sha256 = canonical_run_bundle_sha256(bundle)
    if expected_bundle_sha256 is not None and expected_bundle_sha256 != bundle_sha256:
        raise MatrixAuthorityError("run bundle identity does not match its expected SHA-256")
    if expected_run_set_id is not None and expected_run_set_id != bundle.run_set_id:
        raise MatrixAuthorityError("run bundle identity does not match its expected run-set id")
    if not is_training_matrix_bundle(bundle):
        raise MatrixAuthorityError("run bundle is not a governed training matrix")
    first = bundle.rows[0].execution.authored_intent
    payload = _read_artifact(first, "training matrix")
    matrix = TrainingRunMatrixSpec.model_validate(payload)
    if tuple(row.row_id for row in bundle.rows) != expected_ordered_matrix_row_ids(matrix):
        raise MatrixAuthorityError("matrix bundle rows are incomplete or out of order")
    if training_run_intent_hash(payload) != first.intent_hash:
        raise MatrixAuthorityError("training matrix intent identity does not match custody")
    if any(row.execution.authored_intent != first for row in bundle.rows):
        raise MatrixAuthorityError("matrix rows do not share one authored matrix identity")
    return TrainingRunMatrixAuthority(
        matrix=TrainingRunMatrixArtifactBinding(
            schema_id=first.schema_id,
            schema_version=first.schema_version,
            artifact_id=first.artifact_id,
            artifact_sha256=first.sha256,
            canonical_sha256=training_spec_sha256(
                matrix.model_dump(mode="json", exclude_none=True)
            ),
            intent_hash=first.intent_hash,
        ),
        rows=[_authenticate_row(row, bundle.run_set_id) for row in bundle.rows],
        resolved_inputs=[_input_binding(item) for item in bundle.resolved_inputs],
        code_authorities=_code_authorities(bundle, local_repos, protected_refs),
        monitor=TrainingRunMatrixMonitorBinding(
            event_paths=[f"events/{row.row_id}.events.jsonl" for row in bundle.rows]
        ),
        bundle_sha256=bundle_sha256,
    )


def _read_artifact(ref: SchemaArtifactRef, label: str) -> dict[str, Any]:
    if ref.uri is None:
        raise MatrixAuthorityError(f"{label} is not locally materialized")
    try:
        data = Path(ref.uri).read_bytes()
        payload = json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatrixAuthorityError(f"{label} is unavailable or invalid JSON") from exc
    if hashlib.sha256(data).hexdigest() != ref.sha256:
        raise MatrixAuthorityError(f"{label} custody digest mismatch")
    if not isinstance(payload, dict):
        raise MatrixAuthorityError(f"{label} must contain a JSON object")
    if (payload.get("schema_id"), payload.get("schema_version")) != (
        ref.schema_id,
        ref.schema_version,
    ):
        raise MatrixAuthorityError(f"{label} schema identity does not match custody")
    return payload


def _authenticate_row(row: RunRowSpec, run_set_id: str) -> TrainingRunMatrixRowPreflightBinding:
    execution = row.execution
    payload = _read_artifact(execution.payload, f"matrix row {row.row_id!r} runtime payload")
    snapshot = _read_artifact(
        execution.resolved_snapshot, f"matrix row {row.row_id!r} resolved snapshot"
    )
    resolved = decode_resolved_snapshot(snapshot)
    capsule = TrainingRunExecutionCapsule.model_validate(
        _read_artifact(execution.execution_capsule, f"matrix row {row.row_id!r} execution capsule")
    )
    stored = resolved.get("row_provenance") if isinstance(resolved, Mapping) else None
    if resolved.get("payload") != payload or not isinstance(stored, Mapping):
        raise MatrixAuthorityError(f"matrix row {row.row_id!r} resolved authority is mismatched")
    governed = TrainingRowProvenance.model_validate(stored)
    if execution.row_provenance is None or governed != execution.row_provenance:
        raise MatrixAuthorityError(f"matrix row {row.row_id!r} bundle provenance is synthetic")
    coordinates = governed.axis_coordinates.get("values")
    if not isinstance(coordinates, dict):
        raise MatrixAuthorityError(f"matrix row {row.row_id!r} lacks planning coordinates")
    planned = _planned_run_id(
        payload,
        seed=governed.seed,
        axis_coordinates=coordinates,
        authored_payload_hash=governed.authored_payload_hash,
        lowered_execution_payload_hash=governed.lowered_execution_payload_hash,
        lowerer_identities=governed.lowerer_identities,
    )
    expected = (
        governed.row_id,
        governed.planned_run_id,
        resolved.get("run_set_id"),
        resolved.get("row_id"),
        resolved.get("planned_run_id"),
        resolved.get("seed"),
        resolved.get("axis_coordinates"),
        resolved.get("overrides"),
    )
    observed = (
        row.row_id,
        planned,
        run_set_id,
        row.row_id,
        planned,
        governed.seed,
        governed.axis_coordinates,
        governed.overrides,
    )
    if expected != observed:
        raise MatrixAuthorityError(f"matrix row {row.row_id!r} planned identity is mismatched")
    payload_hash = training_spec_sha256(payload)
    inputs = canonicalize_immutable_input_identities(execution.immutable_inputs)
    root_hash = str(snapshot["root_hash"])
    execution_hash = training_run_execution_hash(root_hash, inputs)
    if (
        payload_hash != governed.lowered_execution_payload_hash
        or capsule.input_data_identities != inputs
        or capsule.intent_hash != execution.authored_intent.intent_hash
        or capsule.resolved_root_hash != root_hash
        or capsule.execution_hash != execution_hash
        or execution.resolved_snapshot.root_hash != root_hash
        or execution.execution_capsule.execution_hash != execution_hash
    ):
        raise MatrixAuthorityError(f"matrix row {row.row_id!r} execution identity is mismatched")
    config = payload.get("training_config")
    depth = config.get("n_batches") if isinstance(config, Mapping) else None
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise MatrixAuthorityError(f"matrix row {row.row_id!r} lacks a positive locked depth")
    if not is_native_training_command(row.launch.command):
        raise MatrixAuthorityError(f"matrix row {row.row_id!r} lacks the native event executor")
    return TrainingRunMatrixRowPreflightBinding(
        row_id=row.row_id,
        planned_run_id=planned,
        authored_payload_hash=governed.authored_payload_hash,
        lowered_execution_payload_hash=payload_hash,
        runtime_payload_sha256=execution.payload.sha256,
        resolved_root_hash=root_hash,
        execution_hash=execution_hash,
        locked_training_depth=depth,
    )


def _input_binding(item: ResolvedAssemblyInput) -> TrainingRunMatrixInputCustodyBinding:
    return TrainingRunMatrixInputCustodyBinding(
        role=item.identity.role,
        kind=item.identity.kind,
        identifier=item.identity.identifier,
        identity_sha256=item.identity.digest.value,
        provider_binding=item.custody.provider_binding,
        artifact_sha256=item.custody.artifact.sha256,
        custody_sha256=training_spec_sha256(
            item.custody.model_dump(mode="json", exclude_none=True)
        ),
    )


def _code_authorities(
    bundle: RunBundle,
    local_repos: Mapping[str, str | Path],
    protected_refs: Mapping[str, str],
) -> list[TrainingRunMatrixCodeAuthority]:
    repos = {str(name): Path(path) for name, path in local_repos.items()}
    refs = {str(name): str(ref) for name, ref in protected_refs.items()}
    if not repos or set(repos) != set(refs):
        raise MatrixAuthorityError("matrix preflight requires complete protected-ref policy")
    declarations = list(bundle.environment.repo_revisions)
    result = []
    for name, root in sorted(repos.items()):
        matches = [
            item
            for item in declarations
            if item.path == name or _same_path(Path.cwd() / item.path, root)
        ]
        if len(matches) != 1 or matches[0].dirty_allowed:
            raise MatrixAuthorityError(f"matrix repo {name!r} lacks one clean declaration")
        declaration = matches[0]
        observed = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
        protected = _git(root, "rev-parse", "--verify", f"{refs[name]}^{{commit}}")
        if _git(root, "status", "--porcelain", "--untracked-files=normal"):
            raise MatrixAuthorityError(f"matrix repo {name!r} is dirty")
        if len({declaration.revision, protected, observed}) != 1:
            raise MatrixAuthorityError(f"matrix repo {name!r} code authority is mismatched")
        result.append(
            TrainingRunMatrixCodeAuthority(
                repo=name,
                declared_revision=declaration.revision,
                protected_ref=refs[name],
                protected_revision=protected,
                observed_revision=observed,
                clean=True,
            )
        )
    if len(declarations) != len(result):
        raise MatrixAuthorityError("matrix revision declarations lack complete authority")
    return result


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _git(root: Path, *args: str) -> str:
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, env=environment, capture_output=True, text=True
        )
    except OSError as exc:
        raise MatrixAuthorityError("matrix code authority Git inspection failed") from exc
    if result.returncode:
        raise MatrixAuthorityError("matrix code authority Git inspection failed")
    return result.stdout.strip()
