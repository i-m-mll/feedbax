"""Spec-conformance certificates for completed training run sets.

Project plugins may publish additional checks through the existing
``feedbax.plugins`` entry-point group. A plugin can expose either
``register_feedbax_conformance_checks(registry)`` or
``feedbax_conformance_checks()``, where each yielded item is a
``(check_id, callable)`` pair. The callable receives a
``ConformanceRowArtifacts`` instance and returns a ``CheckEntry``.
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.contracts.manifest import StrictModel, TrainingRunManifest, load_manifest
from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot
from feedbax.contracts.spec_storage import (
    canonicalize_immutable_input_identities,
    training_run_execution_hash,
)
from feedbax.contracts.training import OptimizerSpec
from feedbax.orchestration.bundle import ExecutionIdentityEnvelope, SchemaArtifactRef
from feedbax.orchestration.schedule_eval import (
    _MISSING as _SCHEDULE_MISSING,
    extract_resume_context,
    learning_rate_from_build_optimizer,
    require_schedule_context,
)

RUN_CONFORMANCE_SCHEMA_ID = "feedbax.run_conformance"
RUN_CONFORMANCE_SCHEMA_VERSION = "feedbax.run_conformance.v1"

CHECK_STATUS_PASS = "pass"
CHECK_STATUS_FAIL = "fail"
CHECK_STATUS_SKIPPED = "skipped"
CHECK_STATUSES = (CHECK_STATUS_PASS, CHECK_STATUS_FAIL, CHECK_STATUS_SKIPPED)

CheckStatus = Literal["pass", "fail", "skipped"]
OverallStatus = Literal["pass", "fail"]
CheckCallable = Callable[["ConformanceRowArtifacts"], "CheckEntry"]


class CheckEntry(StrictModel):
    """One conformance check outcome for one row."""

    check_id: str = Field(min_length=1)
    status: CheckStatus
    expected: Any = None
    observed: Any = None
    detail: str | None = None

    @model_validator(mode="after")
    def _validate_skipped_detail(self) -> "CheckEntry":
        if self.status == CHECK_STATUS_SKIPPED and not self.detail:
            raise ValueError("skipped conformance checks require a detail")
        return self


class CertificateRow(StrictModel):
    """Conformance checks attached to one row id."""

    checks: list[CheckEntry]


class RunConformanceCertificate(StrictModel):
    """Durable red/green conformance artifact for a run set."""

    schema_id: Literal["feedbax.run_conformance"] = RUN_CONFORMANCE_SCHEMA_ID
    schema_version: Literal["feedbax.run_conformance.v1"] = RUN_CONFORMANCE_SCHEMA_VERSION
    run_set_id: str = Field(min_length=1)
    generated_at: datetime
    overall: OverallStatus
    rows: dict[str, CertificateRow]

    @model_validator(mode="after")
    def _validate_overall(self) -> "RunConformanceCertificate":
        expected = aggregate_overall(self.rows)
        if self.overall != expected:
            raise ValueError(
                f"overall must be {expected!r} for the contained check statuses; "
                f"got {self.overall!r}"
            )
        return self


@dataclass(frozen=True)
class ConformanceRowArtifacts:
    """Collected row artifacts consumed by conformance checks.

    Missing required fields are check failures, not skips. ``event_log`` is the
    one legacy-safe exception: absence means the terminal-event protocol is not
    available on this branch/row and ``events_terminal`` records a skipped check
    with detail.
    """

    row_id: str
    execution: ExecutionIdentityEnvelope | None = None
    execution_identity_adapter: Any = None
    schema_registry: Any = None
    manifest_path: Path | str | None = None
    training_diagnostics: Mapping[str, Any] | None = None
    checkpoint_custody_root: Path | str | None = None
    event_log: Path | str | Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None
    bundle_row_spec: Mapping[str, Any] | None = None
    recorded_environment_fingerprint: Any = None
    preflight_normalized_payload: Mapping[str, Any] | None = None
    manifest_payload: Mapping[str, Any] | None = None


class CheckRegistry:
    """Deterministic registry of conformance checks."""

    def __init__(self, checks: Mapping[str, CheckCallable] | None = None) -> None:
        self._checks: dict[str, CheckCallable] = {}
        for check_id, check in (checks or {}).items():
            self.register(check_id, check)

    def register(self, check_id: str, check: CheckCallable) -> None:
        """Register one check callable."""
        if not check_id:
            raise ValueError("check_id must be non-empty")
        if check_id in self._checks:
            raise ValueError(f"conformance check already registered: {check_id!r}")
        self._checks[check_id] = check

    def extend(self, checks: Iterable[tuple[str, CheckCallable]]) -> None:
        """Register several ``(check_id, callable)`` pairs."""
        for check_id, check in checks:
            self.register(check_id, check)

    def items(self) -> tuple[tuple[str, CheckCallable], ...]:
        """Return checks sorted by id for deterministic certificate output."""
        return tuple((check_id, self._checks[check_id]) for check_id in sorted(self._checks))

    def __len__(self) -> int:
        return len(self._checks)


def pass_check(check_id: str, *, expected: Any = None, observed: Any = None) -> CheckEntry:
    """Build a passing check result."""
    return CheckEntry(
        check_id=check_id,
        status=CHECK_STATUS_PASS,
        expected=expected,
        observed=observed,
    )


def fail_check(
    check_id: str,
    *,
    expected: Any = None,
    observed: Any = None,
    detail: str | None = None,
) -> CheckEntry:
    """Build a failing check result."""
    return CheckEntry(
        check_id=check_id,
        status=CHECK_STATUS_FAIL,
        expected=expected,
        observed=observed,
        detail=detail,
    )


def skipped_check(
    check_id: str,
    *,
    expected: Any = None,
    observed: Any = None,
    detail: str,
) -> CheckEntry:
    """Build a skipped check result. ``detail`` is required by the model."""
    return CheckEntry(
        check_id=check_id,
        status=CHECK_STATUS_SKIPPED,
        expected=expected,
        observed=observed,
        detail=detail,
    )


def missing_input_check(check_id: str, *missing: str) -> CheckEntry:
    """Return a fail result for missing required inputs."""
    return fail_check(
        check_id,
        expected={"required": list(missing)},
        observed=None,
        detail="missing required input: " + ", ".join(missing),
    )


def aggregate_overall(rows: Mapping[str, CertificateRow]) -> OverallStatus:
    """Return pass iff every non-skipped check passes."""
    for row in rows.values():
        for check in row.checks:
            if check.status == CHECK_STATUS_FAIL:
                return CHECK_STATUS_FAIL
    return CHECK_STATUS_PASS


def assemble_certificate(
    *,
    run_set_id: str,
    row_checks: Mapping[str, Sequence[CheckEntry]],
    generated_at: datetime,
) -> RunConformanceCertificate:
    """Assemble and validate a deterministic certificate model."""
    rows = {
        row_id: CertificateRow(checks=sorted(checks, key=lambda check: check.check_id))
        for row_id, checks in sorted(row_checks.items())
    }
    return RunConformanceCertificate(
        run_set_id=run_set_id,
        generated_at=generated_at,
        overall=aggregate_overall(rows),
        rows=rows,
    )


def run_conformance_checks(
    *,
    run_set_id: str,
    rows: Sequence[ConformanceRowArtifacts],
    registry: CheckRegistry,
    generated_at: datetime | None = None,
) -> RunConformanceCertificate:
    """Run registered checks over collected row artifacts."""
    if len(registry) == 0:
        raise ValueError("CERTIFY requires at least one registered conformance check")
    row_results: dict[str, list[CheckEntry]] = {}
    for row in sorted(rows, key=lambda item: item.row_id):
        checks: list[CheckEntry] = []
        for check_id, check in registry.items():
            try:
                result = check(row)
            except Exception as exc:  # plugin/core failures belong in the certificate.
                result = fail_check(
                    check_id,
                    expected="check completed without exception",
                    observed=type(exc).__name__,
                    detail=str(exc),
                )
            if result.check_id != check_id:
                result = result.model_copy(update={"check_id": check_id})
            if result.status == CHECK_STATUS_SKIPPED:
                result = fail_check(
                    check_id,
                    expected=result.expected,
                    observed=result.observed,
                    detail=f"check did not produce a verdict: {result.detail}",
                )
            checks.append(result)
        row_results[row.row_id] = checks
    return assemble_certificate(
        run_set_id=run_set_id,
        row_checks=row_results,
        generated_at=generated_at or datetime.now(timezone.utc).replace(microsecond=0),
    )


def write_conformance_certificate(
    *,
    run_set_dir: Path | str,
    run_set_id: str,
    rows: Sequence[ConformanceRowArtifacts],
    registry: CheckRegistry,
    generated_at: datetime | None = None,
) -> RunConformanceCertificate:
    """Run checks and write ``<run_set_dir>/conformance.json`` deterministically."""
    certificate = run_conformance_checks(
        run_set_id=run_set_id,
        rows=rows,
        registry=registry,
        generated_at=generated_at,
    )
    output = Path(run_set_dir) / "conformance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = certificate.model_dump(mode="json")
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return certificate


def build_core_check_registry() -> CheckRegistry:
    """Return the built-in conformance checks."""
    return CheckRegistry(
        {
            "checkpoint_cadence": check_checkpoint_cadence,
            "completed_batches": check_completed_batches,
            "environment_fingerprint": check_environment_fingerprint,
            "execution_identity": check_execution_identity,
            "events_terminal": check_events_terminal,
            "lr_trace": check_lr_trace,
            "manifest_valid": check_manifest_valid,
            "seeds": check_seeds,
        }
    )


def check_execution_identity(row: ConformanceRowArtifacts) -> CheckEntry:
    """Verify ASSEMBLE identity evidence against independently emitted identity."""
    check_id = "execution_identity"
    if row.execution is None:
        return missing_input_check(check_id, "execution")

    try:
        envelope = ExecutionIdentityEnvelope.model_validate(row.execution)
        raw_manifest = _raw_manifest_payload(row)
    except Exception as exc:
        return fail_check(
            check_id,
            expected="typed execution envelope and final TrainingRunManifest",
            observed=type(exc).__name__,
            detail=str(exc),
        )

    identity_keys = (
        "intent_hash",
        "resolved_semantics_root_hash",
        "execution_hash",
        "input_data_identities",
    )
    missing = [key for key in identity_keys if key not in raw_manifest]
    if missing:
        return missing_input_check(
            check_id,
            *(f"manifest.{key}" for key in missing),
        )

    expected = {
        "intent_hash": envelope.authored_intent.intent_hash,
        "resolved_semantics_root_hash": envelope.resolved_snapshot.root_hash,
        "execution_hash": envelope.execution_capsule.execution_hash,
        "input_data_identities": canonicalize_immutable_input_identities(envelope.immutable_inputs),
    }
    try:
        _validate_envelope_artifacts(
            envelope,
            identity_adapter=row.execution_identity_adapter,
            schema_registry=row.schema_registry,
        )
        computed_execution_hash = training_run_execution_hash(
            envelope.resolved_snapshot.root_hash,
            expected["input_data_identities"],
        )
        if computed_execution_hash != envelope.execution_capsule.execution_hash:
            raise ValueError(
                "envelope execution_hash mismatch: "
                f"expected {computed_execution_hash}, "
                f"observed {envelope.execution_capsule.execution_hash}"
            )
        manifest = TrainingRunManifest.model_validate(raw_manifest)
        observed = {
            "intent_hash": manifest.intent_hash,
            "resolved_semantics_root_hash": manifest.resolved_semantics_root_hash,
            "execution_hash": manifest.execution_hash,
            "input_data_identities": canonicalize_immutable_input_identities(
                manifest.input_data_identities
            ),
        }
    except Exception as exc:
        return fail_check(
            check_id,
            expected=expected,
            observed={key: raw_manifest.get(key) for key in identity_keys},
            detail=str(exc),
        )

    mismatches = {
        field: {"expected": expected[field], "observed": observed[field]}
        for field in expected
        if expected[field] != observed[field]
    }
    if mismatches:
        return fail_check(
            check_id,
            expected=expected,
            observed=observed,
            detail=json.dumps(mismatches, sort_keys=True),
        )
    return pass_check(check_id, expected=expected, observed=observed)


def _validate_envelope_artifacts(
    envelope: ExecutionIdentityEnvelope,
    *,
    identity_adapter: Any = None,
    schema_registry: Any = None,
) -> None:
    payload = _load_schema_artifact(envelope.payload, registry=schema_registry)
    authored = _load_schema_artifact(envelope.authored_intent, registry=schema_registry)
    snapshot = _load_schema_artifact(envelope.resolved_snapshot, registry=schema_registry)
    capsule = _load_schema_artifact(envelope.execution_capsule, registry=schema_registry)

    adapter = identity_adapter or _builtin_identity_adapter(envelope.authored_intent.schema_id)
    intent_hash = adapter.intent_hash(authored)
    if intent_hash != envelope.authored_intent.intent_hash:
        raise ValueError(
            "authored_intent.intent_hash mismatch: "
            f"expected {intent_hash}, observed {envelope.authored_intent.intent_hash}"
        )

    decode_resolved_snapshot(snapshot)
    if snapshot.get("root_hash") != envelope.resolved_snapshot.root_hash:
        raise ValueError("resolved_snapshot.root_hash does not bind the decoded snapshot")

    canonical_inputs = canonicalize_immutable_input_identities(envelope.immutable_inputs)
    bindings = {
        "intent_hash": envelope.authored_intent.intent_hash,
        "resolved_root_hash": envelope.resolved_snapshot.root_hash,
        "input_data_identities": canonical_inputs,
        "execution_hash": envelope.execution_capsule.execution_hash,
    }
    observed_identities = adapter.capsule_identities(capsule)
    observed_bindings = {
        "intent_hash": observed_identities.intent_hash,
        "resolved_root_hash": observed_identities.resolved_root_hash,
        "input_data_identities": observed_identities.immutable_inputs,
        "execution_hash": observed_identities.execution_hash,
    }
    mismatches = {
        key: {"expected": expected, "observed": observed_bindings.get(key)}
        for key, expected in bindings.items()
        if observed_bindings.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "execution capsule binding mismatch: " + json.dumps(mismatches, sort_keys=True)
        )
    # Payload validation above is itself required evidence; keep a live reference so
    # a future refactor cannot accidentally omit its dereference.
    if not isinstance(payload, Mapping):  # pragma: no cover - registry validation guards this
        raise TypeError("registered executable payload must be an object")


def _builtin_identity_adapter(schema_id: str) -> Any:
    """Resolve adapters for standalone checks outside a request-based engine."""
    if schema_id == "feedbax.spec.training_run_matrix":
        from feedbax.training.spec_storage import TrainingRunIdentityAdapter

        return TrainingRunIdentityAdapter()
    if schema_id == "feedbax.spec.studio.training_assembly":
        from feedbax.contracts.studio_training import StudioTrainingIdentityAdapter

        return StudioTrainingIdentityAdapter()
    raise ValueError(f"no execution-identity adapter for authored schema {schema_id!r}")


def _load_schema_artifact(
    ref: SchemaArtifactRef,
    *,
    registry: Any = None,
) -> dict[str, Any]:
    if registry is None:
        from feedbax.contracts.migrations import default_spec_registry

        registry = default_spec_registry

    if not ref.uri:
        raise ValueError(f"artifact {ref.artifact_id!r} has no materialization URI")
    path = Path(ref.uri).expanduser()
    if not path.is_file():
        raise ValueError(f"artifact {ref.artifact_id!r} is unavailable at {path}")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != ref.sha256:
        raise ValueError(
            f"artifact byte sha256 mismatch for {ref.artifact_id!r}: "
            f"expected {ref.sha256}, observed {actual}"
        )
    payload = json.loads(data)
    if not isinstance(payload, Mapping):
        raise TypeError(f"artifact {ref.artifact_id!r} payload must be an object")
    family = next(
        (family for family in registry.families() if family.identity == ref.schema_id),
        None,
    )
    if family is None:
        raise ValueError(f"unknown registered artifact schema {ref.schema_id!r}")
    if payload.get("schema_id") != ref.schema_id:
        raise ValueError(f"artifact {ref.artifact_id!r} schema_id mismatch")
    if payload.get("schema_version") != ref.schema_version:
        raise ValueError(f"artifact {ref.artifact_id!r} schema_version mismatch")
    migrated = registry.migrate(
        family.kind,
        payload,
        source_version=ref.schema_version,
    )
    validated = migrated.payload
    if ref.schema_id == "feedbax.spec.training_run_matrix":
        from feedbax.contracts.run_matrix import TrainingRunMatrixSpec

        validated = TrainingRunMatrixSpec.model_validate(validated).model_dump(
            mode="json", exclude_none=True
        )
    elif ref.schema_id == "feedbax.spec.studio.training_assembly":
        from feedbax.contracts.studio_training import StudioTrainingAssemblySpec

        validated = StudioTrainingAssemblySpec.model_validate(validated).model_dump(
            mode="json", exclude_none=True
        )
    elif ref.schema_id == "feedbax.manifest.training_run_execution_capsule":
        from feedbax.contracts.spec_storage import TrainingRunExecutionCapsule

        validated = TrainingRunExecutionCapsule.model_validate(validated).model_dump(
            mode="json", exclude_none=True
        )
    return validated


def _raw_manifest_payload(row: ConformanceRowArtifacts) -> dict[str, Any]:
    if row.manifest_payload is not None:
        return dict(row.manifest_payload)
    if row.manifest_path is None:
        raise ValueError("missing final manifest")
    payload = json.loads(Path(row.manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("final manifest payload must be an object")
    return dict(payload)


def build_default_check_registry(*, include_plugins: bool = True) -> CheckRegistry:
    """Return core checks plus optional checks discovered from Feedbax plugins."""
    registry = build_core_check_registry()
    if include_plugins:
        from feedbax.plugins.discovery import load_conformance_check_plugins

        load_conformance_check_plugins(registry=registry)
    return registry


def assert_certificate_allows_completed_registration(
    certificate: RunConformanceCertificate | Mapping[str, Any],
) -> RunConformanceCertificate:
    """Return the certificate or raise if REGISTER must not mark completion."""
    cert = RunConformanceCertificate.model_validate(certificate)
    if cert.overall != CHECK_STATUS_PASS:
        raise ValueError("REGISTER cannot emit phase=completed for a failing certificate")
    return cert


def check_completed_batches(row: ConformanceRowArtifacts) -> CheckEntry:
    """Verify realized batch count equals the declared count."""
    check_id = "completed_batches"
    expected = _first_present(
        _path(row.bundle_row_spec, "expected_batches"),
        _path(row.bundle_row_spec, "completed_batches"),
        _path(row.bundle_row_spec, "training", "n_batches"),
        _path(row.bundle_row_spec, "training_config", "n_batches"),
        _path(row.bundle_row_spec, "n_batches"),
    )
    observed = _first_present(
        _path(row.training_diagnostics, "completed_batches"),
        _path(row.training_diagnostics, "summary", "completed_batches"),
        _path(row.training_diagnostics, "summary_metrics", "completed_batches"),
        _path(row.training_diagnostics, "summary_metrics", "n_batches"),
        _path(_manifest_payload(row), "summary_metrics", "completed_batches"),
        _path(_manifest_payload(row), "summary_metrics", "n_batches"),
    )
    if expected is _MISSING:
        return missing_input_check(check_id, "bundle_row_spec.n_batches")
    if observed is _MISSING:
        return missing_input_check(check_id, "training_diagnostics.completed_batches")
    if int(observed) == int(expected):
        return pass_check(check_id, expected=int(expected), observed=int(observed))
    return fail_check(check_id, expected=int(expected), observed=int(observed))


def check_seeds(row: ConformanceRowArtifacts) -> CheckEntry:
    """Verify realized seeds equal declared seeds."""
    check_id = "seeds"
    bundle_seed = _first_present(
        _path(row.bundle_row_spec, "seeds"),
        _path(row.bundle_row_spec, "seed"),
        _path(row.bundle_row_spec, "training", "seeds"),
        _path(row.bundle_row_spec, "metadata", "seeds"),
        _path(row.bundle_row_spec, "metadata", "seed"),
    )
    provenance_seed = _path(row.execution, "row_provenance", "seed")
    if provenance_seed is None:
        provenance_seed = _MISSING
    if (
        bundle_seed is not _MISSING
        and provenance_seed is not _MISSING
        and _canonical(_seed_sequence(bundle_seed)) != _canonical(_seed_sequence(provenance_seed))
    ):
        return fail_check(
            check_id,
            expected={"bundle_row_spec": bundle_seed},
            observed={"execution.row_provenance.seed": provenance_seed},
            detail="declared seeds disagree between bundle row and execution provenance",
        )
    expected = _first_present(bundle_seed, provenance_seed)
    observed = _first_present(
        _path(row.training_diagnostics, "seeds"),
        _path(row.training_diagnostics, "seed"),
        _path(_manifest_payload(row), "metadata", "seeds"),
        _path(_manifest_payload(row), "metadata", "seed"),
    )
    if expected is _MISSING:
        return missing_input_check(check_id, "bundle_row_spec.seeds")
    if observed is _MISSING:
        return missing_input_check(check_id, "manifest/training_diagnostics.seeds")
    if _canonical(_seed_sequence(expected)) == _canonical(_seed_sequence(observed)):
        return pass_check(check_id, expected=expected, observed=observed)
    return fail_check(check_id, expected=expected, observed=observed)


def check_environment_fingerprint(row: ConformanceRowArtifacts) -> CheckEntry:
    """Verify realized environment fingerprint equals the manifest stamp."""
    check_id = "environment_fingerprint"
    expected = row.recorded_environment_fingerprint
    observed = _first_present(
        _path(_manifest_payload(row), "metadata", "environment_fingerprint"),
        _path(_manifest_payload(row), "provenance", "metadata", "environment_fingerprint"),
    )
    if expected is None:
        return missing_input_check(check_id, "recorded_environment_fingerprint")
    if observed is _MISSING:
        return missing_input_check(check_id, "manifest.metadata.environment_fingerprint")
    if _canonical(expected) == _canonical(observed):
        return pass_check(check_id, expected=expected, observed=observed)
    return fail_check(check_id, expected=expected, observed=observed)


def check_manifest_valid(row: ConformanceRowArtifacts) -> CheckEntry:
    """Verify the final manifest loads and matches preflight-normalized payloads."""
    check_id = "manifest_valid"
    if row.manifest_path is None and row.manifest_payload is None:
        return missing_input_check(check_id, "manifest_path")
    if row.preflight_normalized_payload is None:
        return missing_input_check(check_id, "preflight_normalized_payload")
    try:
        payload = _manifest_payload(row, force_load=True)
    except Exception as exc:
        return fail_check(
            check_id,
            expected="TrainingRunManifest loads",
            observed=type(exc).__name__,
            detail=str(exc),
        )
    observed_spec = _training_spec_payload(payload)
    expected_spec = dict(row.preflight_normalized_payload)
    if observed_spec is _MISSING:
        return missing_input_check(check_id, "manifest.training_spec.inline")
    if _canonical(observed_spec) == _canonical(expected_spec):
        return pass_check(check_id, expected=expected_spec, observed=observed_spec)
    return fail_check(check_id, expected=expected_spec, observed=observed_spec)


def check_checkpoint_cadence(row: ConformanceRowArtifacts) -> CheckEntry:
    """Verify checkpoint transaction coordinates follow the declared cadence."""
    check_id = "checkpoint_cadence"
    interval = _first_present(
        _path(row.bundle_row_spec, "checkpoint_interval"),
        _path(row.bundle_row_spec, "checkpoint_cadence", "interval"),
        _path(row.bundle_row_spec, "training", "checkpoint_interval"),
        _path(row.bundle_row_spec, "checkpoint_progress", "checkpoint_interval"),
    )
    segment_completed = _path(row.training_diagnostics, "segment_completed_batches")
    if segment_completed is not _MISSING:
        completed = segment_completed
        completed_source = "training_diagnostics.segment_completed_batches"
    else:
        completed_candidates = (
            (
                _path(row.training_diagnostics, "completed_batches"),
                "training_diagnostics.completed_batches",
            ),
            (
                _path(row.training_diagnostics, "summary", "completed_batches"),
                "training_diagnostics.summary.completed_batches",
            ),
            (_path(row.bundle_row_spec, "expected_batches"), "bundle_row_spec.expected_batches"),
            (_path(row.bundle_row_spec, "n_batches"), "bundle_row_spec.n_batches"),
        )
        completed = _MISSING
        completed_source = "completed batch count"
        for candidate, source in completed_candidates:
            if candidate is not _MISSING:
                completed = candidate
                completed_source = source
                break
    coordinates = _checkpoint_coordinates(row)
    if interval is _MISSING:
        return missing_input_check(check_id, "bundle_row_spec.checkpoint_interval")
    if completed is _MISSING:
        return missing_input_check(check_id, "completed batch count")
    if coordinates is None:
        return missing_input_check(check_id, "checkpoint transaction coordinates")

    interval_int = int(interval)
    completed_int = int(completed)
    expected = _expected_checkpoint_coordinates(interval_int, completed_int)
    observed = sorted(int(coordinate) for coordinate in coordinates)
    if observed == expected:
        return pass_check(
            check_id,
            expected={"coordinate_interval": interval_int, "coordinates": expected},
            observed={"coordinates": observed, "realized_batches": completed_int},
        )
    return fail_check(
        check_id,
        expected={"coordinate_interval": interval_int, "coordinates": expected},
        observed={"coordinates": observed, "realized_batches": completed_int},
        detail=f"cadence length read from {completed_source}",
    )


def check_events_terminal(row: ConformanceRowArtifacts) -> CheckEntry:
    """Verify an optional event log ends in exactly one terminal event."""
    check_id = "events_terminal"
    if row.event_log is None:
        return skipped_check(
            check_id,
            expected="exactly one terminal event",
            observed=None,
            detail="no event log/API is available for this row",
        )
    try:
        events = _load_events(row.event_log)
    except Exception as exc:
        return fail_check(
            check_id,
            expected="event log loads",
            observed=type(exc).__name__,
            detail=str(exc),
        )
    terminal_events = [event for event in events if _is_terminal_event(event)]
    observed = {"terminal_count": len(terminal_events), "terminal_events": terminal_events}
    expected_status = _first_present(
        _path(row.bundle_row_spec, "expected_terminal_status"),
        _path(row.bundle_row_spec, "sentinel_status"),
        _path(row.training_diagnostics, "terminal_status"),
    )
    expected_status_value = None if expected_status is _MISSING else expected_status
    if len(terminal_events) != 1:
        return fail_check(
            check_id,
            expected={"terminal_count": 1, "terminal_status": expected_status_value},
            observed=observed,
        )
    if expected_status is not _MISSING:
        terminal_status = _event_status(terminal_events[0])
        if _canonical_terminal_status(terminal_status) != _canonical_terminal_status(
            expected_status
        ):
            return fail_check(
                check_id,
                expected={"terminal_count": 1, "terminal_status": expected_status},
                observed={"terminal_count": 1, "terminal_status": terminal_status},
            )
    return pass_check(
        check_id,
        expected={"terminal_count": 1, "terminal_status": expected_status_value},
        observed=observed,
    )


def check_lr_trace(row: ConformanceRowArtifacts) -> CheckEntry:
    """Verify realized learning-rate samples against the declared schedule."""
    check_id = "lr_trace"
    optimizer_spec_payload = _optimizer_spec_payload(row)
    trace = _lr_trace(row)
    context = extract_resume_context(row.bundle_row_spec, row.training_diagnostics)
    if optimizer_spec_payload is _MISSING:
        return missing_input_check(check_id, "bundle_row_spec optimizer spec")
    if trace is None:
        return missing_input_check(check_id, "training_diagnostics.lr_trace")
    missing_context = [
        key
        for key in ("schedule_origin_step", "current_step", "optimizer_count_at_current_step")
        if context.get(key) is _SCHEDULE_MISSING
    ]
    if missing_context:
        return missing_input_check(check_id, *missing_context)

    try:
        eval_context = require_schedule_context(context, label="resume_context")
        optimizer_spec = OptimizerSpec.model_validate(optimizer_spec_payload)
        samples = _selected_lr_samples(trace, optimizer_spec)
        expected = {
            step: learning_rate_from_build_optimizer(
                optimizer_spec,
                sample_step=step,
                schedule_origin_step=eval_context.schedule_origin_step,
                current_step=eval_context.current_step,
                optimizer_count_at_current_step=eval_context.optimizer_count_at_current_step,
            )
            for step in samples
        }
    except Exception as exc:
        return fail_check(
            check_id,
            expected="learning rates derived via build_optimizer",
            observed=type(exc).__name__,
            detail=str(exc),
        )

    observed = {step: float(trace[step]) for step in samples}
    mismatches = {
        step: {"expected": expected[step], "observed": observed[step]}
        for step in samples
        if not _close(observed[step], expected[step], rel_tol=1e-6)
    }
    if not mismatches:
        return pass_check(check_id, expected=expected, observed=observed)
    return fail_check(check_id, expected=expected, observed=observed, detail=str(mismatches))


def _selected_lr_samples(
    trace: Mapping[int, float],
    optimizer_spec: OptimizerSpec,
) -> tuple[int, ...]:
    steps = sorted(trace)
    if len(steps) < 3:
        raise ValueError("lr_trace requires at least three realized samples")
    schedule = optimizer_spec.lr_schedule
    candidates = {steps[0], steps[-1], steps[len(steps) // 2]}
    if schedule is not None:
        if schedule.constant_lr_iterations:
            candidates.add(int(schedule.constant_lr_iterations))
        if schedule.total_steps is not None:
            candidates.add(int(schedule.total_steps))
    selected = tuple(step for step in steps if step in candidates)
    if len(selected) >= 3:
        return selected[:3] if len(selected) > 3 else selected
    return (steps[0], steps[len(steps) // 2], steps[-1])


def _optimizer_spec_payload(row: ConformanceRowArtifacts) -> Any:
    return _first_present(
        _path(row.bundle_row_spec, "optimizer"),
        _path(row.bundle_row_spec, "optimizer_spec"),
        _path(row.bundle_row_spec, "training", "optimizer"),
        _path(row.bundle_row_spec, "training_spec", "method_payload", "payload", "optimizer"),
        _path(
            _training_spec_payload(_manifest_payload(row)), "method_payload", "payload", "optimizer"
        ),
        _path(row.bundle_row_spec, "method_payload", "payload", "optimizer"),
        _path(row.bundle_row_spec, "method_payload", "payload", "controller_optimizer"),
        _path(
            row.bundle_row_spec,
            "training_spec",
            "method_payload",
            "payload",
            "controller_optimizer",
        ),
        _path(
            _training_spec_payload(_manifest_payload(row)),
            "method_payload",
            "payload",
            "controller_optimizer",
        ),
    )


def _lr_trace(row: ConformanceRowArtifacts) -> dict[int, float] | None:
    raw = _first_present(
        _path(row.training_diagnostics, "lr_trace"),
        _path(row.training_diagnostics, "learning_rate_trace"),
    )
    if raw is _MISSING:
        return None
    if isinstance(raw, Mapping):
        return {int(step): float(value) for step, value in raw.items()}
    trace: dict[int, float] = {}
    for item in raw:
        step = _first_present(_path(item, "step"), _path(item, "batch"), _path(item, "coordinate"))
        value = _first_present(_path(item, "learning_rate"), _path(item, "lr"))
        if step is _MISSING or value is _MISSING:
            raise ValueError(f"invalid lr_trace item {item!r}")
        trace[int(step)] = float(value)
    return trace


def _checkpoint_coordinates(row: ConformanceRowArtifacts) -> list[int] | None:
    raw = _first_present(
        _path(row.training_diagnostics, "checkpoint_coordinates"),
        _path(row.training_diagnostics, "checkpoint_transactions"),
    )
    if raw is not _MISSING:
        if all(isinstance(item, int) for item in raw):
            return [int(item) for item in raw]
        return [
            int(
                _first_present(
                    _path(item, "coordinate", "program_step"),
                    _path(item, "coordinate", "step"),
                    _path(item, "program_step"),
                    _path(item, "step"),
                )
            )
            for item in raw
        ]
    if row.checkpoint_custody_root is None:
        return None
    root = Path(row.checkpoint_custody_root)
    if not root.exists():
        return None
    coordinates: list[int] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        coordinate = _first_present(
            _path(payload, "coordinate", "program_step"),
            _path(payload, "coordinate", "step"),
            _path(payload, "program_step"),
            _path(payload, "step"),
        )
        if coordinate is not _MISSING:
            coordinates.append(int(coordinate))
    return coordinates


def _expected_checkpoint_coordinates(interval: int, completed_batches: int) -> list[int]:
    if interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    coordinates = list(range(interval, completed_batches + 1, interval))
    if completed_batches not in coordinates:
        coordinates.append(completed_batches)
    return coordinates


def _load_events(
    event_log: Path | str | Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if isinstance(event_log, Mapping):
        events = event_log.get("events")
        if isinstance(events, Sequence):
            return list(events)
        return [event_log]
    if isinstance(event_log, Sequence) and not isinstance(event_log, (str, bytes)):
        return list(event_log)
    path = Path(event_log)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, Mapping) and isinstance(payload.get("events"), Sequence):
        return list(payload["events"])
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return [payload]
    raise ValueError(f"unsupported event log payload in {path}")


def _is_terminal_event(event: Mapping[str, Any]) -> bool:
    status = _event_status(event)
    return _canonical_terminal_status(status) in {
        "complete",
        "failed",
        "cancelled",
        "error",
    }


def _event_status(event: Mapping[str, Any]) -> str | None:
    value = _first_present(
        _path(event, "phase"),
        _path(event, "status"),
        _path(event, "event_type"),
        _path(event, "type"),
    )
    return None if value is _MISSING else str(value)


def _canonical_terminal_status(status: Any) -> str | None:
    if status is None or status is _MISSING:
        return None
    value = str(status).lower()
    if value in {"complete", "completed", "done", "success", "succeeded"}:
        return "complete"
    if value in {"failed", "fail", "failure"}:
        return "failed"
    if value in {"cancelled", "canceled"}:
        return "cancelled"
    if value == "error":
        return "error"
    return value


def _manifest_payload(row: ConformanceRowArtifacts, *, force_load: bool = False) -> Any:
    if row.manifest_payload is not None and not force_load:
        return row.manifest_payload
    if row.manifest_path is None:
        return row.manifest_payload or {}
    manifest = load_manifest(row.manifest_path)
    return manifest.model_dump(mode="json", exclude_none=True)


def _training_spec_payload(manifest_payload: Any) -> Any:
    training_spec = _path(manifest_payload, "training_spec")
    inline = _path(training_spec, "inline")
    if inline is not _MISSING:
        return inline
    return training_spec


class _Missing:
    pass


_MISSING = _Missing()


def _path(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if current is None:
            return _MISSING
        if isinstance(current, Mapping):
            if key not in current:
                return _MISSING
            current = current[key]
            continue
        current = getattr(current, key, _MISSING)
        if current is _MISSING:
            return _MISSING
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not _MISSING:
            return value
    return _MISSING


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _seed_sequence(value: Any) -> Any:
    """Normalize one declared or observed scalar seed to singleton-list form."""
    if (
        isinstance(value, Mapping)
        or isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
    ):
        return value
    return [value]


def _close(observed: float, expected: float, *, rel_tol: float) -> bool:
    return abs(observed - expected) <= rel_tol * max(abs(expected), 1.0)
