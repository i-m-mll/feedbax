"""Adopt pre-custody Equinox leaf streams into checkpoint custody."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
from pydantic import Field, ValidationError, model_validator

from feedbax.contracts.checkpoints import (
    CheckpointLatestPointer,
    LeafManifest,
    LeafManifestEntry,
    LeafManifestProvenance,
)
from feedbax.contracts.manifest import StrictModel, sha256_bytes
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.training import TrainingRunSpec
from feedbax.contracts.worker import PhaseProgramSpec, ProgressCoordinate
from feedbax.training.checkpoint_custody import (
    CheckpointCompatibilityError,
    CheckpointWriteResult,
    ResumeSlotTransform,
    load_latest_checkpoint,
    write_checkpoint_transaction,
)

LEGACY_MANIFEST_KIND = "LegacyCheckpointLeafManifest"
PATH_MAPPING_REGISTRY_SCHEMA_ID = "feedbax.training.legacy_checkpoint_path_mapping"
PATH_MAPPING_REGISTRY_SCHEMA_VERSION = "feedbax.training.legacy_checkpoint_path_mapping.v1"


class LegacyCheckpointAdoptionError(ValueError):
    """Base class for legacy checkpoint adoption failures."""


class LegacyManifestSchemaError(LegacyCheckpointAdoptionError):
    """Raised when a legacy leaf manifest cannot be accepted or migrated."""


class LegacyStreamMismatchError(LegacyCheckpointAdoptionError):
    """Raised when raw ``np.save`` records do not match the leaf manifest."""


class LegacyPathMappingError(LegacyCheckpointAdoptionError):
    """Raised when old leaf paths cannot be assigned to current templates."""


class LegacyWorktreeDumpError(LegacyCheckpointAdoptionError):
    """Raised when producing-commit manifest dumping fails."""


@dataclass(frozen=True)
class PathMappingRule:
    """One exact old-path to current-path rename rule."""

    old_path: str
    new_path: str


class PathMappingRegistryRule(StrictModel):
    """One versioned path-mapping registry rule."""

    tree: Literal["model", "optimizer"]
    old_path: str
    new_path: str


class PathMappingRegistry(StrictModel):
    """Versioned path-mapping registry for legacy checkpoint adoption."""

    schema_id: str = PATH_MAPPING_REGISTRY_SCHEMA_ID
    schema_version: str = PATH_MAPPING_REGISTRY_SCHEMA_VERSION
    rules: list[PathMappingRegistryRule] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "PathMappingRegistry":
        if self.schema_id != PATH_MAPPING_REGISTRY_SCHEMA_ID:
            raise ValueError(f"unsupported path mapping schema_id {self.schema_id!r}")
        if self.schema_version != PATH_MAPPING_REGISTRY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported path mapping schema_version {self.schema_version!r}"
            )
        return self

    def rules_for(self, tree: Literal["model", "optimizer"]) -> tuple[PathMappingRule, ...]:
        """Return exact rules for one tree."""
        return tuple(
            PathMappingRule(old_path=rule.old_path, new_path=rule.new_path)
            for rule in self.rules
            if rule.tree == tree
        )


@dataclass(frozen=True)
class StreamRecord:
    """One raw ``np.save`` record read from a legacy checkpoint stream."""

    index: int
    value: np.ndarray
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class StaticPathReport:
    """Static old/current comparison for one manifest path."""

    old_path: str
    current_path: str
    status: str
    old_static_repr_sha256: str | None
    current_static_repr_sha256: str | None


@dataclass(frozen=True)
class TreeAdoptionReport:
    """Assignment details for one adopted slot tree."""

    assigned_paths: tuple[str, ...]
    static_paths: tuple[StaticPathReport, ...]


@dataclass(frozen=True)
class LegacyCheckpointAdoptionResult:
    """Result returned after a custody checkpoint passes the resume round trip."""

    write: CheckpointWriteResult
    loaded_slots: dict[str, Any]
    model_report: TreeAdoptionReport
    optimizer_report: TreeAdoptionReport | None


@dataclass(frozen=True)
class ManifestDumpRequest:
    """One producing-commit ABI dump request."""

    commit: str
    spec_path: Path
    output_path: Path


@dataclass(frozen=True)
class ManifestDumpResult:
    """Path produced by a manifest dump request."""

    commit: str
    spec_path: Path
    output_path: Path


@dataclass
class _PathMap:
    by_old: dict[str, str] = field(default_factory=dict)

    def resolve(self, old_path: str) -> str:
        return self.by_old.get(old_path, old_path)


def accept_leaf_manifest(payload: Mapping[str, Any] | LeafManifest) -> LeafManifest:
    """Accept or migrate a legacy leaf manifest payload, then validate it."""
    if isinstance(payload, LeafManifest):
        return payload
    try:
        migrated = default_spec_registry.migrate(LEGACY_MANIFEST_KIND, payload)
        return LeafManifest.model_validate(migrated.payload)
    except (ValidationError, ValueError) as exc:
        raise LegacyManifestSchemaError(str(exc)) from exc


def load_leaf_manifest(path: Path | str) -> LeafManifest:
    """Load a schema-governed legacy leaf manifest JSON file."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyManifestSchemaError(f"could not read leaf manifest {path!s}") from exc
    return accept_leaf_manifest(payload)


def load_path_mapping_registry(path: Path | str | None) -> PathMappingRegistry:
    """Load a versioned path-mapping registry, or return an empty identity registry."""
    if path is None:
        return PathMappingRegistry()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyPathMappingError(f"could not read path mapping registry {path!s}") from exc
    try:
        return PathMappingRegistry.model_validate(payload)
    except ValidationError as exc:
        raise LegacyPathMappingError(str(exc)) from exc


def manifest_from_trees(
    *,
    model: Any,
    optimizer: Any | None,
    producing_commit: str,
    spec_ref: str | None = None,
    spec_hash: str | None = None,
    dumped_at: str | None = None,
    dumper_version: str = "feedbax-legacy-leaf-dumper.v1",
    metadata: Mapping[str, Any] | None = None,
) -> LeafManifest:
    """Build a current LeafManifest from old model and optimizer templates."""
    if not producing_commit:
        raise LegacyManifestSchemaError("producing_commit is required")
    return LeafManifest(
        model=_entries_from_tree(model),
        optimizer=[] if optimizer is None else _entries_from_tree(optimizer),
        provenance=LeafManifestProvenance(
            producing_commit=producing_commit,
            spec_ref=spec_ref,
            spec_hash=spec_hash,
            dumped_at=dumped_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            dumper_version=dumper_version,
        ),
        metadata=dict(metadata or {}),
    )


def read_legacy_stream(
    path: Path | str,
    manifest_entries: Sequence[LeafManifestEntry],
    *,
    tree: Literal["model", "optimizer"] = "model",
) -> dict[str, Any]:
    """Compatibility helper returning verified stream arrays keyed by manifest path."""
    del tree
    records = read_raw_np_save_stream(path, manifest_entries)
    entries = [entry for entry in manifest_entries if entry.kind == "array"]
    return {
        entry.tree_path: jnp.asarray(record.value)
        for entry, record in zip(entries, records, strict=True)
    }


def read_raw_np_save_stream(
    path: Path | str,
    manifest_entries: Sequence[LeafManifestEntry],
) -> tuple[StreamRecord, ...]:
    """Read consecutive ``np.save`` records and verify them against array entries."""
    stream_path = Path(path)
    expected = [entry for entry in manifest_entries if entry.kind == "array"]
    records: list[StreamRecord] = []
    with stream_path.open("rb") as stream:
        while True:
            position = stream.tell()
            if not stream.read(1):
                break
            stream.seek(position)
            try:
                value = np.load(stream, allow_pickle=False)
            except Exception as exc:
                raise LegacyStreamMismatchError(
                    f"{stream_path}: record {len(records)} is not a valid np.save array"
                ) from exc
            array = np.asarray(value)
            records.append(
                StreamRecord(
                    index=len(records),
                    value=array,
                    shape=tuple(int(dim) for dim in array.shape),
                    dtype=str(array.dtype),
                )
            )
    _verify_stream_records(stream_path, expected, records)
    return tuple(records)


def adopt_tree_from_legacy_stream(
    stream_path: Path | str,
    manifest_entries: Sequence[LeafManifestEntry],
    current_template: Any,
    *,
    mapping_rules: Sequence[PathMappingRule | Mapping[str, str]] = (),
    allow_current_shape_dtype_mismatch: bool = False,
) -> tuple[Any, TreeAdoptionReport]:
    """Populate a current slot template from a legacy stream by manifest path."""
    records = read_raw_np_save_stream(stream_path, manifest_entries)
    current_arrays = _array_paths(current_template)
    current_statics = _static_paths(current_template)
    path_map = _compile_path_map(mapping_rules)
    array_entries = [entry for entry in manifest_entries if entry.kind == "array"]
    replacements: dict[str, np.ndarray] = {}
    unmatched_old: list[str] = []
    duplicate_new: list[str] = []
    incompatible_current: list[str] = []

    for entry, record in zip(array_entries, records, strict=True):
        new_path = path_map.resolve(entry.tree_path)
        if new_path in replacements:
            duplicate_new.append(new_path)
            continue
        if new_path not in current_arrays:
            unmatched_old.append(f"{entry.tree_path} -> {new_path}")
            continue
        current_leaf = _get_by_path(current_template, current_arrays[new_path])
        current_array = np.asarray(current_leaf)
        shape_mismatch = tuple(int(dim) for dim in current_array.shape) != record.shape
        dtype_mismatch = str(current_array.dtype) != record.dtype
        if shape_mismatch and not allow_current_shape_dtype_mismatch:
            incompatible_current.append(
                f"{entry.tree_path} -> {new_path}: stream shape {record.shape} "
                f"!= current shape {tuple(int(dim) for dim in current_array.shape)}"
            )
            continue
        if dtype_mismatch and not allow_current_shape_dtype_mismatch:
            incompatible_current.append(
                f"{entry.tree_path} -> {new_path}: stream dtype {record.dtype} "
                f"!= current dtype {current_array.dtype}"
            )
            continue
        replacements[new_path] = record.value

    unfilled_current = sorted(set(current_arrays) - set(replacements))
    errors: list[str] = []
    if unmatched_old:
        errors.append(f"unmatched old array leaves: {unmatched_old!r}")
    if duplicate_new:
        errors.append(f"ambiguous current array assignments: {sorted(duplicate_new)!r}")
    if incompatible_current:
        errors.append(f"incompatible current array leaves: {incompatible_current!r}")
    if unfilled_current:
        errors.append(f"unfilled current array leaves: {unfilled_current!r}")
    if errors:
        raise LegacyPathMappingError("; ".join(errors))

    adopted = current_template
    for new_path, value in replacements.items():
        path_entries = current_arrays[new_path]
        old_leaf = _get_by_path(adopted, path_entries)
        replacement = _coerce_array(value, old_leaf)
        adopted = eqx.tree_at(
            lambda tree, p=path_entries: _get_by_path(tree, p),
            adopted,
            replacement,
        )

    static_report = _static_report(manifest_entries, current_statics, path_map)
    return adopted, TreeAdoptionReport(
        assigned_paths=tuple(sorted(replacements)),
        static_paths=tuple(static_report),
    )


def adopt_legacy_checkpoint(
    *,
    checkpoint_root: Path | str,
    run_spec: TrainingRunSpec,
    phase_program: PhaseProgramSpec,
    barrier_name: str,
    coordinate: ProgressCoordinate,
    current_slots: Mapping[str, Any],
    leaf_manifest: Mapping[str, Any] | LeafManifest,
    model_stream: Path | str,
    model_slot: str = "model",
    optimizer_stream: Path | str | None = None,
    optimizer_slot: str = "optimizer",
    fresh_optimizer: bool = False,
    model_mapping_rules: Sequence[PathMappingRule | Mapping[str, str]] = (),
    optimizer_mapping_rules: Sequence[PathMappingRule | Mapping[str, str]] = (),
    resume_slot_transform: ResumeSlotTransform | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LegacyCheckpointAdoptionResult:
    """Adopt model and optimizer legacy streams into custody with a strict round trip."""
    manifest = accept_leaf_manifest(leaf_manifest)
    if model_slot not in current_slots:
        raise LegacyPathMappingError(f"current slots do not contain model slot {model_slot!r}")
    slots = dict(current_slots)
    adopted_model, model_report = adopt_tree_from_legacy_stream(
        model_stream,
        manifest.model,
        current_slots[model_slot],
        mapping_rules=model_mapping_rules,
    )
    slots[model_slot] = adopted_model

    optimizer_report: TreeAdoptionReport | None = None
    if optimizer_stream is not None:
        if optimizer_slot not in current_slots:
            raise LegacyPathMappingError(
                f"current slots do not contain optimizer slot {optimizer_slot!r}"
            )
        adopted_optimizer, optimizer_report = adopt_tree_from_legacy_stream(
            optimizer_stream,
            manifest.optimizer,
            current_slots[optimizer_slot],
            mapping_rules=optimizer_mapping_rules,
            allow_current_shape_dtype_mismatch=resume_slot_transform is not None,
        )
        slots[optimizer_slot] = adopted_optimizer
    elif manifest.optimizer and not fresh_optimizer:
        raise LegacyPathMappingError(
            "optimizer manifest entries are present; pass optimizer_stream or fresh_optimizer=True"
        )

    adoption_metadata = {
        "legacy_checkpoint_adoption": {
            "manifest": manifest.model_dump(mode="json", exclude_none=True),
            "fresh_optimizer": fresh_optimizer and optimizer_stream is None,
            "model_slot": model_slot,
            "optimizer_slot": optimizer_slot if optimizer_stream is not None else None,
            "model_assigned_paths": list(model_report.assigned_paths),
            "optimizer_assigned_paths": (
                list(optimizer_report.assigned_paths) if optimizer_report is not None else []
            ),
        }
    }
    adoption_metadata.update(dict(metadata or {}))
    write = write_checkpoint_transaction(
        checkpoint_root,
        run_spec=run_spec,
        phase_program=phase_program,
        barrier_name=barrier_name,
        coordinate=coordinate,
        slots=slots,
        status="partial",
        metadata=adoption_metadata,
        publish_latest=False,
    )
    expected_slots = _expected_round_trip_slots(
        slots,
        resume_slot_transform=resume_slot_transform,
    )
    loaded_slots = _round_trip_before_publish(
        write,
        run_spec=run_spec,
        phase_program=phase_program,
        expected_slots=expected_slots,
        resume_slot_transform=resume_slot_transform,
    )
    return LegacyCheckpointAdoptionResult(
        write=write,
        loaded_slots=loaded_slots,
        model_report=model_report,
        optimizer_report=optimizer_report,
    )


def dump_leaf_manifests_via_worktrees(
    requests: Sequence[ManifestDumpRequest],
    *,
    repo: Path | str,
    builder: str | None = None,
    run_uv_sync: bool = True,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[ManifestDumpResult, ...]:
    """Dump one or more leaf manifests, grouping requests by producing commit."""
    repo_path = Path(repo)
    _verify_commits(repo_path, {request.commit for request in requests}, runner=runner)
    grouped: dict[str, list[ManifestDumpRequest]] = defaultdict(list)
    for request in requests:
        grouped[request.commit].append(request)

    results: list[ManifestDumpResult] = []
    temp_root = Path(tempfile.mkdtemp(prefix="feedbax-legacy-manifest-worktrees-"))
    try:
        for commit, group in sorted(grouped.items()):
            worktree = temp_root / f"commit-{commit[:12]}"
            _run_checked(
                runner,
                ["git", "-C", str(repo_path), "worktree", "add", "--detach", str(worktree), commit],
                f"could not create producing-commit worktree for {commit}",
            )
            worktree.mkdir(parents=True, exist_ok=True)
            try:
                if run_uv_sync:
                    _run_checked(
                        runner,
                        ["uv", "sync"],
                        f"could not provision producing-commit worktree for {commit}",
                        cwd=worktree,
                    )
                script_path = worktree / "dump_legacy_leaf_manifest.py"
                script_path.write_text(_DUMP_SCRIPT, encoding="utf-8")
                for request in group:
                    request.output_path.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        "uv",
                        "run",
                        "--no-sync",
                        "python",
                        str(script_path),
                        "--spec",
                        str(request.spec_path),
                        "--output",
                        str(request.output_path),
                        "--commit",
                        commit,
                    ]
                    if builder is not None:
                        command.extend(["--builder", builder])
                    _run_checked(
                        runner,
                        command,
                        f"could not dump legacy leaf manifest for {request.spec_path}",
                        cwd=worktree,
                    )
                    results.append(
                        ManifestDumpResult(
                            commit=commit,
                            spec_path=request.spec_path,
                            output_path=request.output_path,
                        )
                    )
            finally:
                _run_checked(
                    runner,
                    ["git", "-C", str(repo_path), "worktree", "remove", "--force", str(worktree)],
                    f"could not remove producing-commit worktree for {commit}",
                    allow_failure=True,
                )
                shutil.rmtree(worktree, ignore_errors=True)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return tuple(results)


def parse_mapping_rules(values: Iterable[str]) -> tuple[PathMappingRule, ...]:
    """Parse ``old=new`` path mapping rules."""
    rules: list[PathMappingRule] = []
    for value in values:
        if "=" not in value:
            raise LegacyPathMappingError(f"mapping rule must be old_path=new_path: {value!r}")
        old, new = value.split("=", 1)
        if not old or not new:
            raise LegacyPathMappingError(f"mapping rule must contain non-empty paths: {value!r}")
        rules.append(PathMappingRule(old_path=old, new_path=new))
    return tuple(rules)


def _verify_stream_records(
    path: Path,
    expected: Sequence[LeafManifestEntry],
    records: Sequence[StreamRecord],
) -> None:
    diffs: list[str] = []
    if len(records) != len(expected):
        diffs.append(
            f"record count mismatch: file={len(records)} manifest_arrays={len(expected)}"
        )
    for index, (entry, record) in enumerate(zip(expected, records)):
        if tuple(entry.shape or ()) != record.shape:
            diffs.append(
                f"record {index} {entry.tree_path}: shape file={record.shape} "
                f"manifest={tuple(entry.shape or ())}"
            )
        if str(np.dtype(entry.dtype)) != record.dtype:
            diffs.append(
                f"record {index} {entry.tree_path}: dtype file={record.dtype} "
                f"manifest={entry.dtype}"
            )
    if len(records) > len(expected):
        extra = [
            {"index": record.index, "shape": record.shape, "dtype": record.dtype}
            for record in records[len(expected) :]
        ]
        diffs.append(f"extra records: {extra!r}")
    if len(expected) > len(records):
        missing = [entry.tree_path for entry in expected[len(records) :]]
        diffs.append(f"missing records for manifest paths: {missing!r}")
    if diffs:
        raise LegacyStreamMismatchError(f"{path}: legacy stream mismatch; " + "; ".join(diffs))


def _compile_path_map(rules: Sequence[PathMappingRule | Mapping[str, str]]) -> _PathMap:
    parsed: list[PathMappingRule] = []
    for rule in rules:
        if isinstance(rule, PathMappingRule):
            parsed.append(rule)
        else:
            parsed.append(
                PathMappingRule(
                    old_path=str(rule["old_path"]),
                    new_path=str(rule["new_path"]),
                )
            )
    counts = Counter(rule.old_path for rule in parsed)
    ambiguous = sorted(path for path, count in counts.items() if count > 1)
    if ambiguous:
        raise LegacyPathMappingError(f"ambiguous mapping rules for old paths: {ambiguous!r}")
    return _PathMap({rule.old_path: rule.new_path for rule in parsed})


def _array_paths(tree: Any) -> dict[str, Any]:
    return {
        _key_path_to_text(path): path
        for path, leaf in jt.leaves_with_path(tree)
        if eqx.is_array(leaf)
    }


def _static_paths(tree: Any) -> dict[str, str]:
    return {
        _key_path_to_text(path): sha256_bytes(repr(leaf).encode("utf-8", errors="replace"))
        for path, leaf in jt.leaves_with_path(tree)
        if not eqx.is_array(leaf)
    }


def _entries_from_tree(tree: Any) -> list[LeafManifestEntry]:
    entries: list[LeafManifestEntry] = []
    for path, leaf in jt.leaves_with_path(tree):
        tree_path = _key_path_to_text(path)
        if eqx.is_array(leaf):
            array = np.asarray(leaf)
            entries.append(
                LeafManifestEntry(
                    tree_path=tree_path,
                    kind="array",
                    shape=tuple(int(dim) for dim in array.shape),
                    dtype=str(array.dtype),
                )
            )
        else:
            entries.append(
                LeafManifestEntry(
                    tree_path=tree_path,
                    kind="static",
                    static_repr_sha256=sha256_bytes(
                        repr(leaf).encode("utf-8", errors="replace")
                    ),
                )
            )
    return entries


def _static_report(
    entries: Sequence[LeafManifestEntry],
    current_statics: Mapping[str, str],
    path_map: _PathMap,
) -> list[StaticPathReport]:
    report: list[StaticPathReport] = []
    for entry in entries:
        if entry.kind != "static":
            continue
        new_path = path_map.resolve(entry.tree_path)
        current_hash = current_statics.get(new_path)
        if current_hash is None:
            status = "missing_current_static"
        elif current_hash == entry.static_repr_sha256:
            status = "same"
        else:
            status = "different"
        report.append(
            StaticPathReport(
                old_path=entry.tree_path,
                current_path=new_path,
                status=status,
                old_static_repr_sha256=entry.static_repr_sha256,
                current_static_repr_sha256=current_hash,
            )
        )
    return report


def _coerce_array(value: np.ndarray, template: Any) -> Any:
    array = jnp.asarray(value)
    if eqx.is_array(template):
        return array.astype(jnp.asarray(template).dtype)
    return array


def _get_by_path(tree: Any, path: Any) -> Any:
    value = tree
    for key in path:
        name = getattr(key, "name", None)
        idx = getattr(key, "idx", None)
        key_value = getattr(key, "key", None)
        if name is not None:
            value = getattr(value, name)
        elif idx is not None:
            value = value[idx]
        elif key_value is not None:
            value = value[key_value]
        else:
            raise LegacyPathMappingError(f"unsupported JAX path key {key!r}")
    return value


def _key_path_to_text(path: Any) -> str:
    if not path:
        return "/"
    parts: list[str] = []
    for key in path:
        name = getattr(key, "name", None)
        idx = getattr(key, "idx", None)
        key_value = getattr(key, "key", None)
        if name is not None:
            parts.append(str(name))
        elif idx is not None:
            parts.append(str(idx))
        elif key_value is not None:
            parts.append(str(key_value))
        else:
            parts.append(str(key))
    return "/" + "/".join(parts)


def _round_trip_before_publish(
    write: CheckpointWriteResult,
    *,
    run_spec: TrainingRunSpec,
    phase_program: PhaseProgramSpec,
    expected_slots: Mapping[str, Any],
    resume_slot_transform: ResumeSlotTransform | None,
) -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="feedbax-adoption-roundtrip-"))
    try:
        temp_transactions = temp_root / "transactions"
        temp_transactions.mkdir()
        shutil.copytree(write.transaction_dir, temp_transactions / write.manifest.transaction_id)
        _write_latest_pointer(temp_root / "latest.json", write.latest_pointer)
        loaded = load_latest_checkpoint(
            temp_root,
            expected_run_spec=run_spec,
            expected_phase_program=phase_program,
            expected_slots=expected_slots,
            resume_slot_transform=resume_slot_transform,
        )
        _assert_tree_equal(expected_slots, loaded.slots)
        _write_latest_pointer(write.latest_pointer_path, write.latest_pointer)
        return dict(loaded.slots)
    except Exception as exc:
        try:
            write.latest_pointer_path.unlink()
        except FileNotFoundError:
            pass
        if isinstance(exc, (CheckpointCompatibilityError, LegacyCheckpointAdoptionError)):
            raise
        raise LegacyCheckpointAdoptionError("adopted checkpoint failed strict round trip") from exc
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _expected_round_trip_slots(
    slots: Mapping[str, Any],
    *,
    resume_slot_transform: ResumeSlotTransform | None,
) -> Mapping[str, Any]:
    if resume_slot_transform is None:
        return slots
    try:
        return dict(resume_slot_transform(slots))
    except Exception as exc:
        raise LegacyCheckpointAdoptionError(
            "resume_slot_transform failed while preparing round-trip expectation"
        ) from exc


def _assert_tree_equal(expected: Mapping[str, Any], loaded: Mapping[str, Any]) -> None:
    mismatches: list[str] = []
    for slot, loaded_value in loaded.items():
        if slot not in expected:
            mismatches.append(f"{slot}: unexpected loaded slot")
            continue
        expected_value = expected[slot]
        for path, expected_leaf in jt.leaves_with_path(expected_value):
            loaded_leaf = _get_by_path(loaded_value, path)
            path_text = f"{slot}{_key_path_to_text(path)}"
            if eqx.is_array(expected_leaf):
                left = jnp.asarray(expected_leaf)
                right = jnp.asarray(loaded_leaf)
                if left.shape != right.shape or left.dtype != right.dtype:
                    mismatches.append(f"{path_text}: shape/dtype mismatch")
                elif not bool(jnp.all(left == right)):
                    mismatches.append(f"{path_text}: value mismatch")
            elif expected_leaf != loaded_leaf:
                mismatches.append(f"{path_text}: static value mismatch")
    if mismatches:
        raise LegacyCheckpointAdoptionError(
            "adopted checkpoint round-trip values differ: " + "; ".join(mismatches)
        )


def _write_latest_pointer(path: Path, latest: CheckpointLatestPointer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        latest.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        indent=2,
    )
    path.write_text(data + "\n", encoding="utf-8")


def _verify_commits(
    repo: Path,
    commits: Iterable[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    for commit in sorted(commits):
        _run_checked(
            runner,
            ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
            f"unknown or unreachable producing commit {commit}",
        )


def _run_checked(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    args: Sequence[str],
    message: str,
    *,
    cwd: Path | None = None,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("GIT_OPTIONAL_LOCKS", "0")
    result = runner(
        list(args),
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode and not allow_failure:
        stderr = (result.stderr or "").strip()
        raise LegacyWorktreeDumpError(f"{message}: {stderr or result.returncode}")
    return result


_DUMP_SCRIPT = r'''#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np


SCHEMA_ID = "feedbax.manifest.legacy_checkpoint_leaf_manifest"
SCHEMA_VERSION = "feedbax.manifest.legacy_checkpoint_leaf_manifest.v1"
DUMPER_VERSION = "feedbax-legacy-leaf-dumper.v1"


def _path_text(path: Any) -> str:
    if not path:
        return "/"
    parts = []
    for key in path:
        name = getattr(key, "name", None)
        idx = getattr(key, "idx", None)
        key_value = getattr(key, "key", None)
        if name is not None:
            parts.append(str(name))
        elif idx is not None:
            parts.append(str(idx))
        elif key_value is not None:
            parts.append(str(key_value))
        else:
            parts.append(str(key))
    return "/" + "/".join(parts)


def _is_array(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _static_hash(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8", errors="replace")).hexdigest()


def _entries(tree: Any) -> list[dict[str, Any]]:
    entries = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if _is_array(leaf):
            array = np.asarray(leaf)
            entries.append(
                {
                    "tree_path": _path_text(path),
                    "kind": "array",
                    "shape": [int(dim) for dim in array.shape],
                    "dtype": str(array.dtype),
                }
            )
        else:
            entries.append(
                {
                    "tree_path": _path_text(path),
                    "kind": "static",
                    "static_repr_sha256": _static_hash(leaf),
                }
            )
    return entries


def _load_builder(ref: str):
    module_name, _, func_name = ref.partition(":")
    if not module_name or not func_name:
        raise ValueError("manifest_builder must be 'module:function'")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--builder")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    builder_ref = args.builder or spec_payload.get("manifest_builder")
    if not isinstance(builder_ref, str):
        raise ValueError("spec JSON must contain manifest_builder='module:function'")
    built = _load_builder(builder_ref)(spec_payload)
    if isinstance(built, tuple):
        model, optimizer = built
    else:
        model = built["model"]
        optimizer = built.get("optimizer")
    payload = {
        "kind": "LegacyCheckpointLeafManifest",
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "model": _entries(model),
        "optimizer": [] if optimizer is None else _entries(optimizer),
        "provenance": {
            "producing_commit": args.commit,
            "spec_ref": str(spec_path),
            "spec_hash": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "dumped_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "dumper_version": DUMPER_VERSION,
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
