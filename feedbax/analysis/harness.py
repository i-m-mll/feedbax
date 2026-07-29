"""Generic row materialization harness for evaluation and analysis pipelines."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import multiprocessing
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from feedbax.analysis.rendering import render_markdown_note
from feedbax.contracts.manifest import (
    ArtifactRef,
    ParentRef,
    RegenerationCommand,
    RegenerationSpec,
    canonical_json_bytes,
    sha256_bytes,
    store_bytes_artifact,
    store_json_artifact,
    write_manifest,
)

CustodyMode = Literal["manifest", "content-addressed"]
SUPPORTED_EXECUTION_POLICY_OVERRIDE_FLAG = "--allow-large-per-row"


def _initialize_evaluation_batch_worker(plugins: tuple[str, ...]) -> None:
    """Load evaluation plugins once per persistent worker process."""
    from feedbax.plugins import load_training_method_plugins

    load_training_method_plugins(modules=plugins)


def _validate_evaluation_fragment_checkpoint(
    declaration: Any,
    batch: Any,
    fragment: ArtifactRef,
    *,
    matrix_intent_hash: str,
) -> None:
    """Validate one cached fragment against its current terminal declaration."""
    from feedbax.analysis.evaluation_compaction import _validate_fragment_ref

    try:
        _validate_fragment_ref(
            declaration,
            fragment,
            matrix_intent_hash=matrix_intent_hash,
            batch_id=batch.batch_id,
        )
    except ValueError as exc:
        raise ValueError("evaluation batch fragment checkpoint contract drifted") from exc


def _execute_evaluation_batch_partition(task: Mapping[str, Any]) -> dict[str, Any]:
    """Execute and compact one authenticated batch in a persistent worker."""
    timing_origin_ns = int(task["timing_origin_ns"])
    started_offset_ns = time.monotonic_ns() - timing_origin_ns
    from feedbax.analysis.evaluation_compaction import (
        EvaluationBatchConsumerInput,
        compact_evaluation_batch,
    )
    from feedbax.analysis.evaluation import (
        EvaluationBatchExecution,
        execute_evaluation_run_matrix,
        load_evaluation_states_cache,
    )
    from feedbax.contracts.evaluation_lifecycle import (
        EvaluationBatchConsumerDeclaration,
        EvaluationLifecycleRowOutcome,
        EvaluationMatrixBatchUnit,
    )
    from feedbax.contracts.manifest import ArtifactRef
    from feedbax.contracts.manifest import (
        AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
        AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
        ParentRef,
        sha256_bytes,
    )
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.analysis.execution_context import (
        resolve_staged_execution_context,
        with_staged_repo_root,
    )
    from feedbax.orchestration.bundle import RunBundle
    from feedbax.orchestration.input_materialization import (
        staged_execution_bindings_for_bundle,
    )

    bundle = RunBundle.model_validate_json(Path(task["bundle_path"]).read_text(encoding="utf-8"))
    projected = staged_execution_bindings_for_bundle(
        bundle,
        inputs_root=task["inputs_root"],
    )
    execution_context = resolve_staged_execution_context(
        projected.descriptor,
        artifact_provider_bindings=projected.artifact_provider_bindings,
        manifest_root_bindings=projected.manifest_root_bindings,
        checkpoint_custody_bindings=projected.checkpoint_custody_bindings,
    )
    execution_context = with_staged_repo_root(execution_context, task["repo_root"])
    payload = _read_json_object(task["payload_path"], description="evaluation matrix spec")
    batch = EvaluationMatrixBatchUnit.model_validate(task["batch"])
    checkpoint_identity = {
        "matrix_intent_hash": task["matrix_intent_hash"],
        "batch": batch.model_dump(mode="json", exclude_none=True),
        "consumers": task["consumers"],
    }
    checkpoint = Path(task["compaction_root"]) / "fragment-checkpoints" / f"{batch.batch_id}.json"
    if checkpoint.is_file():
        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
        if canonical_json_bytes(cached.get("checkpoint_identity")) != canonical_json_bytes(
            checkpoint_identity
        ):
            raise ValueError("evaluation batch fragment checkpoint identity drifted")
        provider = ImmutableArtifactBlobProvider(Path(task["compaction_root"]))
        declarations = [
            EvaluationBatchConsumerDeclaration.model_validate(value) for value in task["consumers"]
        ]
        for declaration, value in zip(
            declarations,
            cached.get("fragments", []),
            strict=True,
        ):
            fragment = ArtifactRef.model_validate(value)
            _validate_evaluation_fragment_checkpoint(
                declaration,
                batch,
                fragment,
                matrix_intent_hash=task["matrix_intent_hash"],
            )
            provider.get_bytes(fragment)
        for value, authority_value in zip(
            cached.get("outcomes", []),
            cached.get("parent_authorities", []),
            strict=True,
        ):
            outcome = EvaluationLifecycleRowOutcome.model_validate(value)
            if not Path(outcome.manifest_path).is_file():
                raise ValueError("evaluation batch fragment checkpoint manifest is unavailable")
            authority = ParentRef.model_validate(authority_value)
            manifest_bytes = Path(outcome.manifest_path).read_bytes()
            if (
                authority.id != outcome.manifest_id
                or authority.metadata.get("manifest_sha256") != sha256_bytes(manifest_bytes)
                or authority.metadata.get("size_bytes") != len(manifest_bytes)
            ):
                raise ValueError("evaluation batch fragment checkpoint parent authority drifted")
        completed_offset_ns = time.monotonic_ns() - timing_origin_ns
        return {
            **cached,
            "pid": os.getpid(),
            "reused_verified_fragments": True,
            "started_offset_ns": started_offset_ns,
            "completed_offset_ns": completed_offset_ns,
            "duration_ns": completed_offset_ns - started_offset_ns,
        }
    result = execute_evaluation_run_matrix(
        payload,
        root=Path(task["manifest_root"]),
        repo_root=task["repo_root"],
        execution_context=execution_context,
        batch=EvaluationBatchExecution(batch_units=((batch.batch_id, batch.ordered_row_ids),)),
    )
    outcomes_by_id = {row.row_id: row for row in result.rows}
    outcomes = []
    manifests = []
    states = []
    parent_authorities = []
    for row_id in batch.ordered_row_ids:
        row = outcomes_by_id[row_id]
        states_schema = getattr(row.result, "metadata", {}).get("states_schema")
        diagnostic_schema_ids = ()
        if isinstance(states_schema, str):
            diagnostic_schema_ids = (states_schema,)
        elif isinstance(states_schema, Mapping) and isinstance(states_schema.get("schema_id"), str):
            diagnostic_schema_ids = (states_schema["schema_id"],)
        outcome = EvaluationLifecycleRowOutcome(
            row_id=row.row_id,
            manifest_id=row.result.id,
            manifest_path=str(row.manifest_path),
            diagnostic_schema_ids=diagnostic_schema_ids,
        )
        outcomes.append(outcome)
        manifests.append(row.result.model_dump(mode="json"))
        manifest_bytes = Path(row.manifest_path).read_bytes()
        parent_authorities.append(
            ParentRef(
                kind="EvaluationRunManifest",
                id=row.result.id,
                role="evaluation_run",
                metadata={
                    "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
                    "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
                    "manifest_sha256": sha256_bytes(manifest_bytes),
                    "size_bytes": len(manifest_bytes),
                },
            )
        )
        if task["consumers"]:
            cache_path = Path(row.result.metadata["cache"]["states_path"])
            states.append(load_evaluation_states_cache(cache_path, manifest_id=row.result.id))
    fragments = []
    for value in task["consumers"]:
        declaration = EvaluationBatchConsumerDeclaration.model_validate(value)
        fragments.append(
            compact_evaluation_batch(
                declaration,
                EvaluationBatchConsumerInput(
                    matrix_intent_hash=task["matrix_intent_hash"],
                    batch=batch,
                    outcomes=tuple(outcomes),
                    manifests=tuple(manifests),
                    states=tuple(states),
                    parent_authorities=tuple(parent_authorities),
                    parameters=json.loads(json.dumps(declaration.parameters)),
                    execution_context=execution_context,
                ),
                custody_root=Path(task["compaction_root"]),
            )
        )
    completed = {
        "pid": os.getpid(),
        "batch_id": batch.batch_id,
        "batch_index": task["batch_index"],
        "outcomes": [item.model_dump(mode="json") for item in outcomes],
        "fragments": [item.model_dump(mode="json") for item in fragments],
        "parent_authorities": [item.model_dump(mode="json") for item in parent_authorities],
        "checkpoint_identity": checkpoint_identity,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed_offset_ns = time.monotonic_ns() - timing_origin_ns
    return {
        **completed,
        "reused_verified_fragments": False,
        "started_offset_ns": started_offset_ns,
        "completed_offset_ns": completed_offset_ns,
        "duration_ns": completed_offset_ns - started_offset_ns,
    }


@dataclass(frozen=True)
class SemanticChange:
    """One path-addressed structural difference."""

    path: str
    before: Any
    after: Any
    kind: Literal["added", "removed", "changed"]


@dataclass(frozen=True)
class MaterializedRow:
    """Artifacts and execution result for one resolved condition row."""

    row_id: str
    resolved: dict[str, Any]
    result: Any
    manifest_path: Path | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    regeneration: RegenerationSpec | None = None


@dataclass(frozen=True)
class HarnessResult:
    """Complete output of one harness invocation."""

    rows: tuple[MaterializedRow, ...]
    note: str
    escape_hatch_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ExecutionPolicy:
    content_sha256: str
    per_row_max_rows: int
    threshold_source: str
    override_flag: str


def _read_json_object(path: str, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load {description} {path!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} {path!r} must contain a JSON object")
    return payload


def _load_execution_policy(path: str) -> _ExecutionPolicy:
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"could not load execution policy {path!r}: {exc}") from exc
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load execution policy {path!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"execution policy {path!r} must contain a JSON object")
    per_row_max_rows = payload.get("per_row_max_rows")
    threshold_source = payload.get("threshold_source")
    override_flag = payload.get("override_flag")
    if type(per_row_max_rows) is not int or per_row_max_rows <= 0:
        raise ValueError("execution policy per_row_max_rows must be a positive integer")
    if not isinstance(threshold_source, str) or not threshold_source.strip():
        raise ValueError("execution policy threshold_source must be a non-blank string")
    if not isinstance(override_flag, str) or not override_flag.strip():
        raise ValueError("execution policy override_flag must be a non-blank string")
    if override_flag != SUPPORTED_EXECUTION_POLICY_OVERRIDE_FLAG:
        raise ValueError(
            f"execution policy override_flag {override_flag!r} does not match the supported "
            f"override surface {SUPPORTED_EXECUTION_POLICY_OVERRIDE_FLAG!r}"
        )
    return _ExecutionPolicy(
        content_sha256=sha256_bytes(content),
        per_row_max_rows=per_row_max_rows,
        threshold_source=threshold_source,
        override_flag=override_flag,
    )


def semantic_diff(before: Any, after: Any, *, path: str = "$") -> list[SemanticChange]:
    """Return a deterministic, path-tracked structural diff."""
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        changes: list[SemanticChange] = []
        before_keys = set(before)
        after_keys = set(after)
        for key in sorted(before_keys | after_keys, key=str):
            child = f"{path}.{key}"
            if key not in before:
                changes.append(SemanticChange(child, None, after[key], "added"))
            elif key not in after:
                changes.append(SemanticChange(child, before[key], None, "removed"))
            else:
                changes.extend(semantic_diff(before[key], after[key], path=child))
        return changes
    if (
        isinstance(before, Sequence)
        and not isinstance(before, (str, bytes))
        and isinstance(after, Sequence)
        and not isinstance(after, (str, bytes))
    ):
        changes = []
        for index in range(max(len(before), len(after))):
            child = f"{path}[{index}]"
            if index >= len(before):
                changes.append(SemanticChange(child, None, after[index], "added"))
            elif index >= len(after):
                changes.append(SemanticChange(child, before[index], None, "removed"))
            else:
                changes.extend(semantic_diff(before[index], after[index], path=child))
        return changes
    if before != after:
        return [SemanticChange(path, before, after, "changed")]
    return []


def diff_resolved_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[SemanticChange]:
    """Compare two fully resolved rows semantically."""
    return semantic_diff(left, right)


def diff_regenerated_archived(
    regenerated: Mapping[str, Any], archived: Mapping[str, Any]
) -> list[SemanticChange]:
    """Compare regenerated output with its archived representation."""
    return semantic_diff(archived, regenerated)


class MatrixMaterializerHarness:
    """Own row expansion, execution, custody, manifests, replay data, and notes."""

    def __init__(
        self,
        *,
        root: Path | str,
        custody: CustodyMode = "content-addressed",
        shared_store: bool = False,
    ):
        if custody not in ("manifest", "content-addressed"):
            raise ValueError(
                "unknown output custody route; registered routes are: content-addressed, manifest"
            )
        self.root = Path(root)
        self.custody = custody
        # Shared-store mode writes every row into one physical store at ``root``
        # (one manifest tree, one cache, one content-addressed artifact store)
        # instead of a self-contained per-row ``root/row_id`` store, and defers
        # index construction to the caller so the batch owner can build a single
        # store-wide index. Rows stay logically distinct through their globally
        # unique manifest identifiers.
        self.shared_store = shared_store

    def materialize(
        self,
        rows: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        execute: Callable[[str, Mapping[str, Any], Path], tuple[Any, Path | None]],
        command: Sequence[str],
        title: str,
        source: str,
        escape_hatch_reason: str | None = None,
        matrix_metadata: Mapping[str, Any] | None = None,
        regeneration_parameters: Mapping[str, Any] | None = None,
    ) -> HarnessResult:
        """Execute resolved rows and emit standard custody and regeneration records."""
        if not rows:
            raise ValueError("materializer requires at least one row or condition")
        if escape_hatch_reason is not None and not escape_hatch_reason.strip():
            raise ValueError("flat-spec escape hatch requires a stated non-empty reason")
        materialized: list[MaterializedRow] = []
        shared_metadata = dict(matrix_metadata or {})
        replay_parameters = dict(regeneration_parameters or {})
        for row_id, resolved in rows:
            if not row_id:
                raise ValueError("materializer row_id must be non-empty")
            row_root = self.root if self.shared_store else self.root / row_id
            row_root.mkdir(parents=True, exist_ok=True)
            result, manifest_path = execute(row_id, resolved, row_root)
            regeneration = RegenerationSpec(
                command=RegenerationCommand(argv=list(command)),
                parameters={
                    **replay_parameters,
                    "row_id": row_id,
                    "resolved": dict(resolved),
                },
                metadata={
                    "source": source,
                    "custody": self.custody,
                    "escape_hatch_reason": escape_hatch_reason,
                    **shared_metadata,
                },
            )
            artifacts: list[ArtifactRef] = []
            if self.custody == "content-addressed":
                artifacts.append(
                    store_json_artifact(
                        dict(resolved),
                        root=self.root,
                        role="resolved_row_spec",
                        logical_name=f"{row_id}.resolved.json",
                        metadata={"row_id": row_id, "source": source},
                    )
                )
                artifacts.append(
                    store_json_artifact(
                        regeneration.model_dump(mode="json", exclude_none=True),
                        root=self.root,
                        role="regeneration_spec",
                        logical_name=f"{row_id}.regeneration.json",
                        metadata={"row_id": row_id, "source": source},
                    )
                )
            if (
                manifest_path is not None
                and hasattr(result, "metadata")
                and hasattr(result, "artifacts")
            ):
                result.metadata.setdefault("matrix_harness", {}).update(
                    {
                        "row_id": row_id,
                        "source": source,
                        "custody": self.custody,
                        "escape_hatch_reason": escape_hatch_reason,
                        **shared_metadata,
                        "regeneration_spec": regeneration.model_dump(
                            mode="json", exclude_none=True
                        ),
                    }
                )
                known_artifact_ids = {artifact.artifact_id for artifact in result.artifacts}
                result.artifacts.extend(
                    artifact
                    for artifact in artifacts
                    if artifact.artifact_id not in known_artifact_ids
                )
                manifest_path = write_manifest(result, root=row_root, index=not self.shared_store)
            materialized.append(
                MaterializedRow(
                    row_id=row_id,
                    resolved=dict(resolved),
                    result=result,
                    manifest_path=manifest_path,
                    artifacts=tuple(artifacts),
                    regeneration=regeneration,
                )
            )
        note = render_markdown_note(
            title=title,
            narrative=(
                f"Flat-spec escape hatch: {escape_hatch_reason}"
                if escape_hatch_reason is not None
                else None
            ),
            rows=[("Source", source), ("Conditions", len(materialized))]
            + [("Row", row.row_id) for row in materialized],
        )
        if self.custody == "content-addressed":
            store_bytes_artifact(
                note.encode("utf-8"),
                root=self.root,
                role="materialization_note",
                logical_name="materialization.md",
                media_type="text/markdown",
                suffix=".md",
                metadata={"source": source, "escape_hatch_reason": escape_hatch_reason},
            )
        return HarnessResult(
            tuple(materialized),
            note,
            escape_hatch_reason,
            metadata={
                **shared_metadata,
                "regeneration_parameters": replay_parameters,
            },
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize a serialized evaluation matrix through the standard harness."""
    parser = argparse.ArgumentParser(prog="feedbax matrix-harness")
    parser.add_argument(
        "spec", help="EvaluationRunMatrixSpec or EvaluationRunMatrixDeltaSpec JSON path"
    )
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--repo-root")
    parser.add_argument("--plugin", action="append")
    parser.add_argument("--escape-hatch-reason")
    parser.add_argument("--parent-manifest-root")
    parser.add_argument("--execution-descriptor")
    parser.add_argument("--artifact-provider", action="append", default=[])
    parser.add_argument("--checkpoint-custody", action="append", default=[])
    parser.add_argument("--orchestration-bundle")
    parser.add_argument("--orchestration-inputs-root")
    parser.add_argument("--orchestration-row-id")
    parser.add_argument("--lifecycle-result")
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Execute the whole matrix through its registered batch recipe; this bypasses "
            "the execution policy's per-row row-count refusal."
        ),
    )
    parser.add_argument(
        "--locked-spec",
        help=(
            "Require the runtime matrix's exact ordered row_id sequence to equal the locked "
            "evaluation matrix sequence; the locked document may be either authoring kind."
        ),
    )
    parser.add_argument(
        "--execution-policy",
        help=(
            "JSON policy requiring positive-integer per_row_max_rows, non-blank "
            "threshold_source, and override_flag set to --allow-large-per-row."
        ),
    )
    parser.add_argument(
        "--allow-large-per-row",
        action="store_true",
        help=(
            "Explicitly bypass an execution policy's oversized per-row refusal; the decision "
            "is recorded in matrix-harness evidence."
        ),
    )
    args = parser.parse_args(argv)
    from feedbax.plugins import load_training_method_plugins

    load_training_method_plugins(modules=args.plugin)
    from feedbax.analysis.evaluation import (
        EvaluationBatchExecution,
        execute_evaluation_run_matrix,
        materialize_evaluation_run_matrix,
    )
    from feedbax.analysis.execution_context import (
        StagedArtifactProviderRootBinding,
        StagedCheckpointCustodyRootBinding,
        resolve_staged_execution_context,
    )
    from feedbax.contracts.evaluation_lifecycle import (
        EvaluationBatchCompactionEvidence,
        EvaluationBatchTimingEvidence,
        EvaluationLifecycleEvidence,
        EvaluationLifecycleRowOutcome,
        EvaluationMatrixBatchPlan,
        EvaluationWorkerProcessEvidence,
        EvaluationWorkerTopologyEvidence,
    )
    from feedbax.contracts.migrations import migrate_structured_spec_payload
    from feedbax.orchestration.bundle import RunBundle
    from feedbax.orchestration.input_materialization import (
        staged_execution_bindings_for_bundle,
    )

    def binding_parts(value: str, *, option: str) -> tuple[str, str]:
        name, separator, root = value.partition("=")
        if not separator or not name or not root:
            raise ValueError(f"{option} entries must use NAME=ROOT")
        return name, root

    try:
        payload = _read_json_object(args.spec, description="evaluation matrix spec")
        policy = (
            _load_execution_policy(args.execution_policy)
            if args.execution_policy is not None
            else None
        )
        runtime_rows = None
        if args.locked_spec is not None:
            locked_document = _read_json_object(
                args.locked_spec,
                description="locked evaluation matrix spec",
            )
            locked_payload = locked_document.get("evaluation_matrix", locked_document)
            locked_rows = materialize_evaluation_run_matrix(
                locked_payload,
                repo_root=args.repo_root,
            )
            runtime_rows = materialize_evaluation_run_matrix(
                payload,
                repo_root=args.repo_root,
            )
            locked_row_ids = tuple(row.row_id for row in locked_rows)
            runtime_row_ids = tuple(row.row_id for row in runtime_rows)
            if runtime_row_ids != locked_row_ids:
                raise ValueError(
                    "runtime evaluation matrix row_id sequence does not match locked spec: "
                    f"runtime={runtime_row_ids!r}, locked={locked_row_ids!r}"
                )
        if policy is not None:
            if runtime_rows is None:
                runtime_rows = materialize_evaluation_run_matrix(
                    payload,
                    repo_root=args.repo_root,
                )
            row_count = len(runtime_rows)
            if (
                not args.batch
                and not args.allow_large_per_row
                and row_count > policy.per_row_max_rows
            ):
                raise ValueError(
                    f"per-row execution refused for {row_count} rows; authored threshold "
                    f"is {policy.per_row_max_rows} rows from {policy.threshold_source!r}; "
                    f"pass authored override {policy.override_flag!r} to proceed"
                )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    policy_metadata = (
        {
            "execution_policy": {
                "content_sha256": policy.content_sha256,
                "per_row_max_rows": policy.per_row_max_rows,
                "threshold_source": policy.threshold_source,
                "override_flag": policy.override_flag,
                "execution_mode": "batch" if args.batch else "per-row",
                "explicit_override_used": args.allow_large_per_row and not args.batch,
            }
        }
        if policy is not None
        else None
    )
    execution_descriptor = (
        json.loads(Path(args.execution_descriptor).read_text(encoding="utf-8"))
        if args.execution_descriptor is not None
        else None
    )
    orchestration_values = (
        args.orchestration_bundle,
        args.orchestration_inputs_root,
        args.orchestration_row_id,
        args.lifecycle_result,
    )
    if any(value is not None for value in orchestration_values) and any(
        value is None for value in orchestration_values
    ):
        parser.error(
            "orchestration lifecycle execution requires bundle, inputs root, row id, "
            "and lifecycle result together"
        )
    resolved_context = None
    orchestration_batch_plan = None
    orchestration_worker_count = None
    if args.orchestration_bundle is not None:
        if (
            args.execution_descriptor is not None
            or args.artifact_provider
            or args.checkpoint_custody
        ):
            parser.error(
                "orchestration lifecycle bindings cannot be combined with caller-supplied "
                "staged execution options"
            )
        bundle = RunBundle.model_validate_json(
            Path(args.orchestration_bundle).read_text(encoding="utf-8")
        )
        if bundle.execution_family != "evaluation-matrix":
            parser.error("orchestration bundle is not an evaluation-matrix family bundle")
        matching_rows = [row for row in bundle.rows if row.row_id == args.orchestration_row_id]
        if len(matching_rows) != 1:
            parser.error("orchestration row id is absent or ambiguous in the bundle")
        try:
            batch_plan_payload = migrate_structured_spec_payload(
                "EvaluationMatrixBatchPlan",
                matching_rows[0].launch.metadata.get("batch_plan"),
                path="launch.metadata.batch_plan",
            ).payload
            orchestration_batch_plan = EvaluationMatrixBatchPlan.model_validate(batch_plan_payload)
        except ValueError as exc:
            parser.error(f"orchestration evaluation row lacks an authenticated batch plan: {exc}")
        orchestration_worker_count = min(
            bundle.launch_policy.max_parallel_rows,
            len(orchestration_batch_plan.batches),
        )
        projected = staged_execution_bindings_for_bundle(
            bundle,
            inputs_root=args.orchestration_inputs_root,
        )
        resolved_context = resolve_staged_execution_context(
            projected.descriptor,
            artifact_provider_bindings=projected.artifact_provider_bindings,
            manifest_root_bindings=projected.manifest_root_bindings,
            checkpoint_custody_bindings=projected.checkpoint_custody_bindings,
        )
    from feedbax.orchestration.events import RunEventEmitter

    emitter = RunEventEmitter.from_env(heartbeat_seconds=60.0)
    with emitter if emitter is not None else nullcontext():
        if emitter is not None:
            emitter.emit("ready", {"executor_family": "evaluation-matrix"})
        try:
            evidence = None
            if (
                args.batch
                and orchestration_batch_plan is not None
                and orchestration_worker_count is not None
            ):
                indexed_batches = [
                    {
                        "batch_id": batch.batch_id,
                        "batch_index": index,
                        "ordered_row_ids": list(batch.ordered_row_ids),
                    }
                    for index, batch in enumerate(orchestration_batch_plan.batches)
                ]
                compaction_root = Path(args.lifecycle_result).with_name(
                    "evaluation-batch-compaction"
                )
                common_task = {
                    "bundle_path": args.orchestration_bundle,
                    "inputs_root": args.orchestration_inputs_root,
                    "payload_path": args.spec,
                    "manifest_root": args.manifest_root,
                    "repo_root": args.repo_root,
                    "matrix_intent_hash": orchestration_batch_plan.matrix_intent_hash,
                    "compaction_root": str(compaction_root),
                }
                from feedbax.analysis.evaluation_compaction import (
                    merge_evaluation_batch_fragment,
                    publish_evaluation_compaction_products,
                    reclaim_evaluation_batch_caches,
                )
                from feedbax.contracts.manifest import ArtifactRef

                prior_states: dict[str, ArtifactRef] = {}
                reclamations = []
                completed_batches = []
                timing_origin_ns = time.monotonic_ns()
                tasks = [
                    {
                        **common_task,
                        "timing_origin_ns": timing_origin_ns,
                        "batch": {
                            "batch_id": item["batch_id"],
                            "ordered_row_ids": item["ordered_row_ids"],
                            "required_leaf_ids": (
                                orchestration_batch_plan.batches[
                                    item["batch_index"]
                                ].required_leaf_ids
                                or ()
                            ),
                        },
                        "batch_index": item["batch_index"],
                        "consumers": [
                            declaration.model_dump(mode="json")
                            for declaration in orchestration_batch_plan.consumers
                            if declaration.leaf_id
                            in (
                                orchestration_batch_plan.batches[
                                    item["batch_index"]
                                ].required_leaf_ids
                                or ()
                            )
                        ],
                    }
                    for item in indexed_batches
                ]
                with ProcessPoolExecutor(
                    max_workers=orchestration_worker_count,
                    initializer=_initialize_evaluation_batch_worker,
                    initargs=(tuple(args.plugin or ()),),
                    mp_context=multiprocessing.get_context("spawn"),
                ) as pool:
                    futures = [
                        pool.submit(_execute_evaluation_batch_partition, task)
                        for task in tasks[:orchestration_worker_count]
                    ]
                    next_task_index = orchestration_worker_count
                    for batch in orchestration_batch_plan.batches:
                        future = futures.pop(0)
                        completed_batch = future.result()
                        completed_batches.append(completed_batch)
                        outcomes_for_batch = tuple(
                            EvaluationLifecycleRowOutcome.model_validate(item)
                            for item in completed_batch["outcomes"]
                        )
                        acknowledgements = []
                        applicable_declarations = tuple(
                            declaration
                            for declaration in orchestration_batch_plan.consumers
                            if declaration.leaf_id in (batch.required_leaf_ids or ())
                        )
                        for declaration, fragment_value in zip(
                            applicable_declarations,
                            completed_batch["fragments"],
                            strict=True,
                        ):
                            acknowledgement = merge_evaluation_batch_fragment(
                                declaration,
                                matrix_intent_hash=orchestration_batch_plan.matrix_intent_hash,
                                batch=batch,
                                parent_authorities=tuple(
                                    ParentRef.model_validate(value)
                                    for value in completed_batch["parent_authorities"]
                                ),
                                fragment=ArtifactRef.model_validate(fragment_value),
                                prior_merge_state=prior_states.get(declaration.leaf_id),
                                custody_root=compaction_root,
                                execution_context=resolved_context,
                            )
                            prior_states[declaration.leaf_id] = acknowledgement.merge_state
                            acknowledgements.append(acknowledgement)
                        if orchestration_batch_plan.consumers:
                            reclamations.append(
                                reclaim_evaluation_batch_caches(
                                    batch,
                                    matrix_intent_hash=orchestration_batch_plan.matrix_intent_hash,
                                    batch_index=completed_batch["batch_index"],
                                    outcomes=outcomes_for_batch,
                                    acknowledgements=acknowledgements,
                                    required_declarations=applicable_declarations,
                                    custody_root=compaction_root,
                                    execution_context=resolved_context,
                                )
                            )
                        if next_task_index < len(tasks):
                            futures.append(
                                pool.submit(
                                    _execute_evaluation_batch_partition,
                                    tasks[next_task_index],
                                )
                            )
                            next_task_index += 1
                outcomes = tuple(
                    EvaluationLifecycleRowOutcome.model_validate(outcome)
                    for batch in completed_batches
                    for outcome in batch["outcomes"]
                )
                evidence = EvaluationLifecycleEvidence(
                    orchestration_row_id=args.orchestration_row_id,
                    ordered_row_ids=tuple(item.row_id for item in outcomes),
                    outcomes=outcomes,
                )
                topology = EvaluationWorkerTopologyEvidence(
                    requested_worker_count=orchestration_worker_count,
                    batch_count=len(indexed_batches),
                    processes=tuple(
                        EvaluationWorkerProcessEvidence(
                            pid=pid,
                            ordered_batch_ids=tuple(
                                item["batch_id"] for item in completed_batches if item["pid"] == pid
                            ),
                            batch_timings=tuple(
                                EvaluationBatchTimingEvidence(
                                    batch_id=item["batch_id"],
                                    started_offset_ns=item["started_offset_ns"],
                                    completed_offset_ns=item["completed_offset_ns"],
                                    duration_ns=item["duration_ns"],
                                    reused_verified_fragments=item[
                                        "reused_verified_fragments"
                                    ],
                                )
                                for item in completed_batches
                                if item["pid"] == pid
                            ),
                        )
                        for pid in dict.fromkeys(item["pid"] for item in completed_batches)
                    ),
                )
                compaction = EvaluationBatchCompactionEvidence(
                    matrix_intent_hash=orchestration_batch_plan.matrix_intent_hash,
                    ordered_batch_ids=tuple(
                        batch.batch_id for batch in orchestration_batch_plan.batches
                    ),
                    declared_leaf_ids=tuple(
                        item.leaf_id for item in orchestration_batch_plan.consumers
                    ),
                    required_leaf_ids_by_batch={
                        batch.batch_id: batch.required_leaf_ids or ()
                        for batch in orchestration_batch_plan.batches
                    },
                    reclamations=tuple(reclamations),
                    terminal_products=(
                        publish_evaluation_compaction_products(
                            orchestration_batch_plan.consumers,
                            prior_states,
                            outcomes,
                            custody_root=compaction_root,
                            execution_context=resolved_context,
                        )
                        if orchestration_batch_plan.consumers
                        else ()
                    ),
                )
                Path(args.lifecycle_result).write_text(
                    json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                Path(args.lifecycle_result).with_name("evaluation-worker-topology.json").write_text(
                    json.dumps(topology.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                Path(args.lifecycle_result).with_name(
                    "evaluation-batch-compaction.json"
                ).write_text(
                    json.dumps(compaction.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                result = None
            else:
                result = execute_evaluation_run_matrix(
                    payload,
                    root=args.manifest_root,
                    repo_root=args.repo_root,
                    escape_hatch_reason=args.escape_hatch_reason,
                    parent_manifest_root=args.parent_manifest_root,
                    execution_descriptor=execution_descriptor,
                    artifact_provider_bindings=[
                        StagedArtifactProviderRootBinding(
                            *binding_parts(value, option="--artifact-provider")
                        )
                        for value in args.artifact_provider
                    ],
                    checkpoint_custody_bindings=[
                        StagedCheckpointCustodyRootBinding(
                            *binding_parts(value, option="--checkpoint-custody")
                        )
                        for value in args.checkpoint_custody
                    ],
                    execution_context=resolved_context,
                    batch=EvaluationBatchExecution() if args.batch else None,
                    matrix_metadata=policy_metadata,
                )
            if args.lifecycle_result is not None and result is not None:
                outcomes = []
                for row in result.rows:
                    states_schema = getattr(row.result, "metadata", {}).get("states_schema")
                    diagnostic_schema_ids = ()
                    if isinstance(states_schema, str):
                        diagnostic_schema_ids = (states_schema,)
                    elif isinstance(states_schema, Mapping) and isinstance(
                        states_schema.get("schema_id"), str
                    ):
                        diagnostic_schema_ids = (states_schema["schema_id"],)
                    outcomes.append(
                        EvaluationLifecycleRowOutcome(
                            row_id=row.row_id,
                            manifest_id=row.result.id,
                            manifest_path=str(row.manifest_path),
                            diagnostic_schema_ids=diagnostic_schema_ids,
                        )
                    )
                evidence = EvaluationLifecycleEvidence(
                    orchestration_row_id=args.orchestration_row_id,
                    ordered_row_ids=tuple(row.row_id for row in result.rows),
                    outcomes=tuple(outcomes),
                )
                destination = Path(args.lifecycle_result)
                destination.write_text(
                    json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        except BaseException:
            if emitter is not None:
                emitter.emit_terminal(
                    "failed",
                    {"status": "failed", "executor_family": "evaluation-matrix"},
                )
            raise
        if emitter is not None:
            emitter.emit_terminal(
                "complete",
                {
                    "status": "completed",
                    "executor_family": "evaluation-matrix",
                    "ordered_row_ids": (
                        list(evidence.ordered_row_ids)
                        if evidence is not None
                        else [row.row_id for row in result.rows]
                    ),
                },
            )
    return 0


__all__ = [
    "HarnessResult",
    "MaterializedRow",
    "MatrixMaterializerHarness",
    "SemanticChange",
    "diff_regenerated_archived",
    "diff_resolved_rows",
    "semantic_diff",
]
