"""Read the local attempt that owns a Studio staged training receipt."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from feedbax.contracts.manifest import TrainingRunManifest
from feedbax.orchestration.bundle import (
    RunBundle,
    canonical_run_bundle_sha256,
    default_orchestration_root,
)
from feedbax.orchestration.staged_training import require_staged_training_owner
from feedbax.orchestration.stages import (
    RETRY_LIMITS,
    STAGE_ASSEMBLE,
    STAGE_COLLECT,
    STAGE_LAUNCH,
    STAGE_TEARDOWN,
    OrchestrationStageError,
    load_collected_native_training_result,
)
from feedbax.orchestration.state import RunSetStateStore


@dataclass(frozen=True)
class NativeTrainingProjection:
    status: str
    result: TrainingRunManifest | None = None
    result_path: Path | None = None
    error: str | None = None
    teardown_pending: bool = False


def project_owned_native_training(
    manifest_id: str,
    metadata: Mapping[str, Any],
    *,
    receipt_root: Path,
) -> NativeTrainingProjection | None:
    owner = metadata.get("staged_attempt")
    if not isinstance(owner, dict):
        return None
    run_set_id = owner.get("run_set_id")
    if not isinstance(run_set_id, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}-[0-9a-f]{8}", run_set_id
    ):
        return None
    run_set_dir = default_orchestration_root(run_set_id).resolve()
    if owner.get("run_set_dir") != str(run_set_dir):
        return None

    try:
        bundle = RunBundle.model_validate_json((run_set_dir / "bundle.json").read_bytes())
        binding = bundle.staged_training
        if (
            bundle.run_set_id != run_set_id
            or canonical_run_bundle_sha256(bundle) != owner.get("bundle_sha256")
            or binding is None
            or binding.manifest_id != manifest_id
            or Path(binding.manifest_root).resolve() != receipt_root.resolve()
            or len(bundle.rows) != 1
        ):
            raise ValueError("Attempt does not match the staged training receipt.")
        row = bundle.rows[0]
        require_staged_training_owner(bundle, row)
        state = RunSetStateStore(run_set_dir / "state.json").load()
        if (
            state.run_set_id != run_set_id
            or set(state.rows) != {row.row_id}
            or state.stage(STAGE_ASSEMBLE).outputs.get("bundle_sha256") != owner["bundle_sha256"]
        ):
            raise ValueError("Attempt state does not match the staged training row.")
        row_state = state.rows[row.row_id]
        collect = state.stage(STAGE_COLLECT)
        teardown = state.stage(STAGE_TEARDOWN)
        custody = collect.outputs.get("checkpoint_custody", {})
        if not isinstance(custody, dict):
            raise ValueError("Invalid checkpoint evidence in the attempt state.")
        evidence = custody.get(row.row_id, {})
        if not isinstance(evidence, dict):
            raise ValueError("Invalid checkpoint evidence for the staged training row.")
        result_digest = evidence.get("training_manifest_sha256")
        if collect.status in ("completed", "failed") and isinstance(result_digest, str):
            path = run_set_dir / "collected" / row.row_id / "manifest.json"
            if row_state.collected_outputs.get("manifest.json") != str(path):
                raise ValueError("Collected result path does not match the staged training row.")
            result = load_collected_native_training_result(row, path, sha256=result_digest)
            certification = result.terminal_certification
            assert certification is not None
            reason = certification.termination_reason
            if reason == "cancelled" and (
                result.completed_batches is None
                or result.completed_batches != evidence.get("completed_training_batches")
            ):
                raise ValueError("Stopped result does not match the authenticated checkpoint progress.")
            status = {"completed": "completed", "cancelled": "stopped"}.get(reason, "failed")
            return NativeTrainingProjection(
                status,
                result,
                path,
                teardown.error,
                teardown_pending=teardown.status in ("pending", "running"),
            )

        if collect.status == "completed":
            raise ValueError("Attempt collection has no authenticated training result.")
        failed_stage = next(
            (
                stage
                for stage_id, stage in state.stages.items()
                if stage.status == "failed"
                and (
                    teardown.status in ("completed", "failed")
                    or stage.attempts >= RETRY_LIMITS.get(stage_id, 1)
                )
            ),
            None,
        )
        if failed_stage is not None:
            return NativeTrainingProjection("failed", error=failed_stage.error or row_state.error)
        if state.stage(STAGE_LAUNCH).status == "completed":
            return NativeTrainingProjection("running")
        return NativeTrainingProjection("claimed")
    except (OSError, ValueError, OrchestrationStageError) as exc:
        return NativeTrainingProjection("failed", error=str(exc))
