"""Spec-conformance certificates for completed training run sets.

Project plugins may publish additional checks through the existing
``feedbax.plugins`` entry-point group. A plugin can expose either
``register_feedbax_conformance_checks(registry)`` or
``feedbax_conformance_checks()``, where each yielded item is a
``(check_id, callable)`` pair. The callable receives a
``ConformanceRowArtifacts`` instance and returns a ``CheckEntry``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
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
from feedbax.orchestration.events import RUN_EVENT_TERMINAL_TYPES
from feedbax.orchestration.schedule_eval import (
    _MISSING as _SCHEDULE_MISSING,
    extract_resume_context,
    learning_rate_from_build_optimizer,
    require_schedule_context,
)
from feedbax.orchestration.state import RowState

RUN_CONFORMANCE_SCHEMA_ID = "feedbax.run_conformance"
RUN_CONFORMANCE_SCHEMA_VERSION_V1 = "feedbax.run_conformance.v1"
RUN_CONFORMANCE_SCHEMA_VERSION = "feedbax.run_conformance.v2"
REALIZED_DEPLOYMENT_RECORD_SCHEMA_ID = "feedbax.manifest.realized_deployment"
REALIZED_DEPLOYMENT_RECORD_SCHEMA_VERSION = "feedbax.manifest.realized_deployment.v1"
_IMMUTABLE_OCI_IMAGE_PATTERN = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")

CHECK_STATUS_PASS = "pass"
CHECK_STATUS_FAIL = "fail"
CHECK_STATUS_SKIPPED = "skipped"
CHECK_STATUSES = (CHECK_STATUS_PASS, CHECK_STATUS_FAIL, CHECK_STATUS_SKIPPED)

CheckStatus = Literal["pass", "fail", "skipped"]
OverallStatus = Literal["pass", "fail"]
CheckCallable = Callable[["ConformanceRowArtifacts"], "CheckEntry"]


class AuthorizedBatchStop(StrictModel):
    """Operational authorization to stop one row at a batch checkpoint.

    The limit is runtime control, not authored scientific identity.  Supplying
    it does not make a short run conformant by itself: the collected manifest,
    diagnostics, checkpoint evidence, and orchestration row state must all
    independently agree that the row stopped at this boundary.
    """

    stop_after_batches: int = Field(gt=0)
    reason: Literal["stop_after_batches"] = "stop_after_batches"


class RowConformanceRuntimeInputs(StrictModel):
    """Typed operational inputs available only to row conformance checks."""

    authorized_batch_stop: AuthorizedBatchStop | None = None


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


class RealizedDeploymentRecord(StrictModel):
    """Typed realized launch, environment, timing, and cost evidence."""

    schema_id: Literal["feedbax.manifest.realized_deployment"] = (
        REALIZED_DEPLOYMENT_RECORD_SCHEMA_ID
    )
    schema_version: Literal["feedbax.manifest.realized_deployment.v1"] = (
        REALIZED_DEPLOYMENT_RECORD_SCHEMA_VERSION
    )
    driver: str = Field(min_length=1)
    venue: Literal["local", "remote"]
    provider: str | None = None
    gpu_model: str | None = None
    gpu_count: int | None = Field(default=None, ge=1)
    region: str | None = None
    immutable_image_id: str | None = None
    environment_fingerprint: str | None = None
    provisioned_at: datetime | None = None
    billing_started_at: datetime | None = None
    row_started_at: datetime | None = None
    row_completed_at: datetime | None = None
    observed_at: datetime
    wall_time_seconds: float | None = Field(
        default=None, ge=0.0, strict=True, allow_inf_nan=False
    )
    hourly_rate: float | None = Field(default=None, ge=0.0, strict=True, allow_inf_nan=False)
    accrued_cost: float | None = Field(default=None, ge=0.0, strict=True, allow_inf_nan=False)
    currency: str | None = None
    cost_basis: str = Field(min_length=1)
    observation_basis: dict[str, str] = Field(min_length=1)
    provider_observations: dict[str, Any] = Field(default_factory=dict)
    unavailable: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_explicit_unavailable_reasons(self) -> "RealizedDeploymentRecord":
        nullable = (
            "provider",
            "gpu_model",
            "gpu_count",
            "region",
            "immutable_image_id",
            "environment_fingerprint",
            "provisioned_at",
            "billing_started_at",
            "row_started_at",
            "row_completed_at",
            "wall_time_seconds",
            "hourly_rate",
            "accrued_cost",
            "currency",
        )
        for field_name in nullable:
            reason = self.unavailable.get(field_name)
            if getattr(self, field_name) is None and (not reason or not reason.strip()):
                raise ValueError(f"missing {field_name!r} requires an unavailable reason")
            if getattr(self, field_name) is not None and reason is not None:
                raise ValueError(f"present {field_name!r} cannot be marked unavailable")
        unknown = self.unavailable.keys() - set(nullable)
        if unknown:
            raise ValueError(f"unknown unavailable evidence fields: {sorted(unknown)!r}")
        timestamps = {
            name: value
            for name, value in (
                ("provisioned_at", self.provisioned_at),
                ("billing_started_at", self.billing_started_at),
                ("row_started_at", self.row_started_at),
                ("row_completed_at", self.row_completed_at),
                ("observed_at", self.observed_at),
            )
            if value is not None
        }
        naive = [name for name, value in timestamps.items() if value.tzinfo is None]
        if naive:
            raise ValueError(f"realized deployment timestamps must be timezone-aware: {naive!r}")
        if all(
            value is not None
            for value in (
                self.provisioned_at,
                self.row_started_at,
                self.row_completed_at,
            )
        ) and not (
            self.provisioned_at <= self.row_started_at <= self.row_completed_at <= self.observed_at
        ):
            raise ValueError(
                "realized deployment chronology must satisfy "
                "provisioned_at <= row_started_at <= row_completed_at <= observed_at"
            )
        if self.billing_started_at is not None and self.billing_started_at > self.observed_at:
            raise ValueError("billing_started_at cannot be later than observed_at")
        if (
            self.venue == "remote"
            and self.immutable_image_id is not None
            and _IMMUTABLE_OCI_IMAGE_PATTERN.fullmatch(self.immutable_image_id) is None
        ):
            raise ValueError(
                "remote immutable_image_id must be a complete lowercase OCI digest reference"
            )
        if (
            self.row_started_at is not None
            and self.row_completed_at is not None
            and self.wall_time_seconds is not None
        ):
            expected_wall_time = (self.row_completed_at - self.row_started_at).total_seconds()
            if not math.isclose(
                self.wall_time_seconds, expected_wall_time, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError("wall_time_seconds does not match the observed row timestamps")
        if (
            self.billing_started_at is not None
            and self.hourly_rate is not None
            and self.accrued_cost is not None
        ):
            elapsed_hours = (self.observed_at - self.billing_started_at).total_seconds() / 3600.0
            expected_cost = self.hourly_rate * elapsed_hours
            if not math.isclose(
                self.accrued_cost, expected_cost, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(
                    "accrued_cost does not match hourly_rate and observed billing duration"
                )
        return self


class CertificateRow(StrictModel):
    """Conformance checks attached to one row id."""

    checks: list[CheckEntry]
    realized_deployment: RealizedDeploymentRecord | None = None
    realized_deployment_evidence: dict[str, Any] | None = None


class RunConformanceCertificate(StrictModel):
    """Durable red/green conformance artifact for a run set."""

    schema_id: Literal["feedbax.run_conformance"] = RUN_CONFORMANCE_SCHEMA_ID
    schema_version: Literal["feedbax.run_conformance.v2"] = RUN_CONFORMANCE_SCHEMA_VERSION
    run_set_id: str = Field(min_length=1)
    generated_at: datetime
    overall: OverallStatus
    rows: dict[str, CertificateRow] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_overall(self) -> "RunConformanceCertificate":
        if self.generated_at.tzinfo is None:
            raise ValueError("certificate generated_at must be timezone-aware")
        for row_id, row in self.rows.items():
            realized_checks = [
                check for check in row.checks if check.check_id == "realized_deployment"
            ]
            if len(realized_checks) != 1:
                raise ValueError(
                    f"certificate row {row_id!r} requires exactly one realized_deployment check"
                )
            if row.realized_deployment_evidence is None:
                raise ValueError(
                    f"certificate row {row_id!r} requires raw realized deployment evidence"
                )
            realized_check = realized_checks[0]
            if realized_check.status == CHECK_STATUS_PASS:
                if row.realized_deployment is None:
                    raise ValueError(
                        f"passing certificate row {row_id!r} requires typed realized deployment"
                    )
                if row.realized_deployment.observed_at != self.generated_at:
                    raise ValueError(
                        f"certificate row {row_id!r} observed_at must equal generated_at"
                    )
                evidence_record = RealizedDeploymentRecord.model_validate(
                    row.realized_deployment_evidence
                )
                if evidence_record != row.realized_deployment:
                    raise ValueError(
                        f"certificate row {row_id!r} typed realized deployment does not "
                        "match its raw evidence"
                    )
                if realized_check.observed != row.realized_deployment.model_dump(mode="json"):
                    raise ValueError(
                        f"certificate row {row_id!r} realized_deployment check does not "
                        "bind the typed record"
                    )
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
    row_status: str | None = None
    training_diagnostics: Mapping[str, Any] | None = None
    checkpoint_custody_root: Path | str | None = None
    event_log: Path | str | Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None
    bundle_row_spec: Mapping[str, Any] | None = None
    recorded_environment_fingerprint: Any = None
    preflight_normalized_payload: Mapping[str, Any] | None = None
    manifest_payload: Mapping[str, Any] | None = None
    row_state: RowState | None = None
    runtime_inputs: RowConformanceRuntimeInputs | None = None
    deployment_policy: Mapping[str, Any] | None = None
    realized_deployment_evidence: Mapping[str, Any] | None = None


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
    row_artifacts: Mapping[str, ConformanceRowArtifacts] | None = None,
) -> RunConformanceCertificate:
    """Assemble and validate a deterministic certificate model."""
    artifacts = dict(row_artifacts or {})
    rows = {}
    for row_id, checks in sorted(row_checks.items()):
        raw = artifacts.get(row_id)
        evidence = dict(raw.realized_deployment_evidence) if (
            raw is not None and raw.realized_deployment_evidence is not None
        ) else {"unavailable_evidence": "realized_deployment_evidence was not supplied"}
        evidence_check = next(
            (check for check in checks if check.check_id == "realized_deployment"), None
        )
        realized = None
        if evidence is not None and evidence_check is not None and evidence_check.status == "pass":
            realized = RealizedDeploymentRecord.model_validate(evidence)
        rows[row_id] = CertificateRow(
            checks=sorted(checks, key=lambda check: check.check_id),
            realized_deployment=realized,
            realized_deployment_evidence=evidence,
        )
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
    declared_inapplicable: Mapping[str, str] | None = None,
    generated_at: datetime | None = None,
) -> RunConformanceCertificate:
    """Run registered checks over collected row artifacts."""
    if len(registry) == 0:
        raise ValueError("CERTIFY requires at least one registered conformance check")
    declarations = dict(declared_inapplicable or {})
    if "realized_deployment" in declarations:
        raise ValueError("realized_deployment cannot be declared inapplicable")
    registered = dict(registry.items())
    registered.setdefault("realized_deployment", check_realized_deployment)
    unknown = declarations.keys() - registered.keys()
    if unknown:
        raise ValueError(f"unknown declared-inapplicable checks: {sorted(unknown)!r}")
    if any(not isinstance(reason, str) or not reason.strip() for reason in declarations.values()):
        raise ValueError("declared-inapplicable checks require non-empty reasons")
    row_results: dict[str, list[CheckEntry]] = {}
    row_artifacts: dict[str, ConformanceRowArtifacts] = {}
    for row in sorted(rows, key=lambda item: item.row_id):
        row_artifacts[row.row_id] = row
        checks: list[CheckEntry] = []
        for check_id, check in sorted(registered.items()):
            if check_id == "realized_deployment":
                check = check_realized_deployment
            if check_id in declarations:
                checks.append(
                    skipped_check(
                        check_id,
                        detail=f"inapplicable-by-declaration: {declarations[check_id]}",
                    )
                )
                continue
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
        row_artifacts=row_artifacts,
    )


def write_conformance_certificate(
    *,
    run_set_dir: Path | str,
    run_set_id: str,
    rows: Sequence[ConformanceRowArtifacts],
    registry: CheckRegistry,
    declared_inapplicable: Mapping[str, str] | None = None,
    generated_at: datetime | None = None,
) -> RunConformanceCertificate:
    """Run checks and write ``<run_set_dir>/conformance.json`` deterministically."""
    certificate = run_conformance_checks(
        run_set_id=run_set_id,
        rows=rows,
        registry=registry,
        declared_inapplicable=declared_inapplicable,
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
            "realized_deployment": check_realized_deployment,
            "seeds": check_seeds,
        }
    )


def check_realized_deployment(row: ConformanceRowArtifacts) -> CheckEntry:
    """Fail closed unless realized operational facts are independently evidenced."""
    check_id = "realized_deployment"
    raw = row.realized_deployment_evidence
    if raw is None:
        return missing_input_check(check_id, "realized_deployment_evidence")
    try:
        record = RealizedDeploymentRecord.model_validate(raw)
    except Exception as exc:
        return fail_check(
            check_id,
            expected="well-formed realized deployment evidence",
            observed=dict(raw),
            detail=str(exc),
        )

    policy = dict(row.deployment_policy or {})
    expected = {
        "driver": policy.get("driver"),
        "venue": policy.get("venue"),
        "requested_resources": policy.get("resources", {}),
    }
    problems: list[str] = []
    if record.driver != expected["driver"] or record.venue != expected["venue"]:
        problems.append("realized driver/venue does not match the deployment route")
    required_timing = (
        record.provisioned_at,
        record.row_started_at,
        record.row_completed_at,
        record.wall_time_seconds,
    )
    if any(item is None for item in required_timing):
        problems.append("provision and row timing are incomplete")
    if record.environment_fingerprint is None:
        problems.append("validated environment fingerprint is unavailable")

    if record.venue == "local":
        if record.provider != "local":
            problems.append("local evidence must identify provider=local")
        if any(
            value is not None
            for value in (record.gpu_model, record.gpu_count, record.region, record.immutable_image_id)
        ):
            problems.append("local GPU, image, and region must use explicit not-applicable facts")
        if (record.hourly_rate, record.accrued_cost, record.currency, record.cost_basis) != (
            0.0,
            0.0,
            "USD",
            "local-not-billable",
        ):
            problems.append("local cost evidence must be zero USD with local-not-billable basis")
    else:
        required_remote = {
            "provider": record.provider,
            "gpu_model": record.gpu_model,
            "gpu_count": record.gpu_count,
            "region": record.region,
            "immutable_image_id": record.immutable_image_id,
            "billing_started_at": record.billing_started_at,
            "hourly_rate": record.hourly_rate,
            "accrued_cost": record.accrued_cost,
            "currency": record.currency,
        }
        missing = [name for name, value in required_remote.items() if value is None]
        if missing:
            problems.append("remote evidence unavailable: " + ", ".join(missing))
        if record.cost_basis != "billing-start-to-certify-observation":
            problems.append("remote accrued cost lacks the CERTIFY observation-time basis")
        if (
            record.immutable_image_id
            and _IMMUTABLE_OCI_IMAGE_PATTERN.fullmatch(record.immutable_image_id) is None
        ):
            problems.append("remote image identity is not immutable")
        try:
            fingerprint = json.loads(record.environment_fingerprint or "")
            runtime = fingerprint["runtime"]
            if fingerprint.get("image_id") != record.immutable_image_id:
                problems.append("environment fingerprint does not prove the recorded image")
            if runtime.get("device_kind") != record.gpu_model:
                problems.append("environment fingerprint does not prove the recorded GPU model")
            if runtime.get("device_count") != record.gpu_count:
                problems.append("environment fingerprint does not prove the recorded GPU count")
        except (json.JSONDecodeError, KeyError, TypeError):
            problems.append("remote environment fingerprint lacks realized runtime proof")
        resources = expected["requested_resources"]
        if isinstance(resources, Mapping):
            requested_gpu = resources.get("gpu_id")
            requested_regions = resources.get("regions") or []
            if requested_gpu and record.gpu_model != requested_gpu:
                problems.append("observed GPU does not satisfy requested GPU policy")
            if requested_regions and record.region not in requested_regions:
                problems.append("observed region does not satisfy requested region policy")

    observed = record.model_dump(mode="json")
    if problems:
        return fail_check(check_id, expected=expected, observed=observed, detail="; ".join(problems))
    return pass_check(check_id, expected=expected, observed=observed)


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
    for row_id, row in cert.rows.items():
        realized_check = next(
            check for check in row.checks if check.check_id == "realized_deployment"
        )
        if realized_check.status != CHECK_STATUS_PASS or row.realized_deployment is None:
            raise ValueError(
                f"REGISTER cannot emit phase=completed without realized deployment proof "
                f"for row {row_id!r}"
            )
    return cert


def check_completed_batches(row: ConformanceRowArtifacts) -> CheckEntry:
    """Verify full completion or an authorized, internally consistent stop."""
    check_id = "completed_batches"
    expected = _first_present(
        _path(row.bundle_row_spec, "expected_batches"),
        _path(row.bundle_row_spec, "completed_batches"),
        _path(row.bundle_row_spec, "training", "n_batches"),
        _path(row.bundle_row_spec, "training_config", "n_batches"),
        _path(row.bundle_row_spec, "n_batches"),
    )
    observed = _first_present(
        _path(row.training_diagnostics, "segment_completed_batches"),
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
    authored_batches = int(expected)
    observed_batches = int(observed)
    if observed_batches == authored_batches:
        return pass_check(check_id, expected=authored_batches, observed=observed_batches)

    authorized_stop = (
        row.runtime_inputs.authorized_batch_stop if row.runtime_inputs is not None else None
    )
    if authorized_stop is None:
        return fail_check(check_id, expected=authored_batches, observed=observed_batches)

    stop_batches = int(authorized_stop.stop_after_batches)
    manifest = _manifest_payload(row)
    diagnostics = row.training_diagnostics
    row_state = row.row_state
    checkpoint_coordinates = _checkpoint_coordinates(row)
    checkpoint_completed_batches = _checkpoint_completed_batches(row)
    manifest_id = _path(manifest, "id")
    diagnostics_manifest_id = _path(diagnostics, "manifest_id")
    diagnostics_status = _path(diagnostics, "terminal_status")
    manifest_status = _path(manifest, "status")
    manifest_completed_batches = _first_present(
        _path(manifest, "completed_batches"),
        _path(manifest, "summary_metrics", "completed_batches"),
        _path(manifest, "summary_metrics", "n_batches"),
    )
    evidence = {
        "authorized_stop_after_batches": stop_batches,
        "diagnostics_completed_batches": observed_batches,
        "diagnostics_terminal_status": (
            None if diagnostics_status is _MISSING else diagnostics_status
        ),
        "manifest_status": None if manifest_status is _MISSING else manifest_status,
        "manifest_completed_batches": (
            None if manifest_completed_batches is _MISSING else manifest_completed_batches
        ),
        "manifest_id": None if manifest_id is _MISSING else manifest_id,
        "diagnostics_manifest_id": (
            None if diagnostics_manifest_id is _MISSING else diagnostics_manifest_id
        ),
        "row_status": None if row_state is None else row_state.status,
        "row_error": None if row_state is None else row_state.error,
        "checkpoint_coordinates": checkpoint_coordinates,
        "checkpoint_completed_batches": checkpoint_completed_batches,
    }
    expected_evidence = {
        "authored_batches": authored_batches,
        "authorized_stop_after_batches": stop_batches,
        "terminal_status": "cancelled",
        "row_status": "stopped",
        "row_error": "operator-stop-after-checkpoint",
        "final_checkpoint_batches": stop_batches,
    }
    failures: list[str] = []
    if stop_batches >= authored_batches:
        failures.append("authorized stop must be earlier than the authored batch budget")
    if observed_batches != stop_batches:
        failures.append("diagnostics completed batches do not match the authorized stop")
    if _canonical_terminal_status(diagnostics_status) != "cancelled":
        failures.append("training diagnostics do not report cancelled terminal status")
    if _canonical_terminal_status(manifest_status) != "cancelled":
        failures.append("training manifest does not report cancelled status")
    if manifest_completed_batches is _MISSING:
        failures.append("training manifest completed batch count is missing")
    elif int(manifest_completed_batches) != stop_batches:
        failures.append("training manifest completed batches do not match the authorized stop")
    if manifest_id is _MISSING or diagnostics_manifest_id is _MISSING:
        failures.append("manifest and training diagnostics require linked manifest ids")
    elif str(manifest_id) != str(diagnostics_manifest_id):
        failures.append("training diagnostics manifest id does not match the manifest")
    if row_state is None:
        failures.append("orchestration row state is missing")
    else:
        if row_state.status != "stopped":
            failures.append("orchestration row state is not stopped")
        if row_state.error != "operator-stop-after-checkpoint":
            failures.append("orchestration row stop reason is not checkpoint-stop")
    if not checkpoint_coordinates or max(checkpoint_coordinates) != stop_batches:
        failures.append("final checkpoint coordinate does not match the authorized stop")
    if not checkpoint_completed_batches or max(checkpoint_completed_batches) != stop_batches:
        failures.append("checkpoint transaction batch count does not match the authorized stop")

    if failures:
        return fail_check(
            check_id,
            expected=expected_evidence,
            observed=evidence,
            detail="; ".join(failures),
        )
    return pass_check(check_id, expected=expected_evidence, observed=evidence)


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
    if _canonical(json.loads(json.dumps(observed_spec), object_pairs_hook=lambda pairs: {key: value for key, value in pairs if value is not None})) == _canonical(json.loads(json.dumps(expected_spec), object_pairs_hook=lambda pairs: {key: value for key, value in pairs if value is not None})):
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
    expected_statuses = _expected_terminal_statuses(row)
    canonical_expected_statuses = {
        source: _canonical_terminal_status(status) for source, status in expected_statuses.items()
    }
    expected_status_value = next(iter(expected_statuses.values()), None)
    if len(terminal_events) != 1:
        return fail_check(
            check_id,
            expected={"terminal_count": 1, "terminal_status": expected_status_value},
            observed=observed,
        )
    terminal_event = terminal_events[0]
    carrier_type = _event_type(terminal_event)
    terminal_status = _event_status(terminal_event)
    canonical_terminal_status = _canonical_terminal_status(terminal_status)
    governed_terminal_status = None if terminal_status is None else terminal_status.lower()
    if governed_terminal_status not in {"completed", "failed", "cancelled"}:
        return fail_check(
            check_id,
            expected={"terminal_count": 1, "terminal_status": expected_status_value},
            observed={
                "terminal_count": 1,
                "carrier_type": carrier_type,
                "terminal_status": terminal_status,
            },
            detail="terminal event payload.status must name a governed terminal status",
        )
    allowed_statuses = {
        "complete": {"completed", "cancelled"},
        "failed": {"failed"},
    }
    if governed_terminal_status not in allowed_statuses[str(carrier_type)]:
        return fail_check(
            check_id,
            expected={
                "terminal_count": 1,
                "carrier_type": carrier_type,
                "terminal_status": sorted(allowed_statuses[str(carrier_type)]),
            },
            observed={
                "terminal_count": 1,
                "carrier_type": carrier_type,
                "terminal_status": terminal_status,
            },
            detail="terminal carrier type disagrees with payload.status",
        )
    expected_canonical_values = set(canonical_expected_statuses.values())
    if len(expected_canonical_values) > 1:
        return fail_check(
            check_id,
            expected={"terminal_count": 1, "terminal_statuses_agree": True},
            observed={
                "terminal_count": 1,
                "terminal_statuses": expected_statuses,
                "event_payload_status": terminal_status,
            },
            detail="row and diagnostics terminal statuses disagree",
        )
    if expected_canonical_values and canonical_terminal_status not in expected_canonical_values:
        return fail_check(
            check_id,
            expected={
                "terminal_count": 1,
                "terminal_status": expected_status_value,
                "terminal_statuses": expected_statuses,
            },
            observed={
                "terminal_count": 1,
                "carrier_type": carrier_type,
                "terminal_status": terminal_status,
            },
        )
    return pass_check(
        check_id,
        expected={
            "terminal_count": 1,
            "terminal_status": expected_status_value,
            "terminal_statuses": expected_statuses,
        },
        observed={
            **observed,
            "carrier_type": carrier_type,
            "terminal_status": terminal_status,
        },
    )


def check_lr_trace(row: ConformanceRowArtifacts) -> CheckEntry:
    """Verify realized learning-rate samples against the declared schedule."""
    check_id = "lr_trace"
    trace = _lr_trace(row)
    context = extract_resume_context(row.bundle_row_spec, row.training_diagnostics)
    try:
        optimizer_spec_payload = _optimizer_spec_payload(row)
        optimizer_spec = (
            None
            if optimizer_spec_payload is _MISSING
            else OptimizerSpec.model_validate(optimizer_spec_payload)
        )
    except Exception as exc:
        return fail_check(
            check_id,
            expected="one unambiguous governed optimizer spec",
            observed=type(exc).__name__,
            detail=str(exc),
        )
    if optimizer_spec_payload is _MISSING:
        return missing_input_check(check_id, "bundle_row_spec optimizer spec")
    if trace is None:
        return missing_input_check(check_id, "training_diagnostics.lr_trace")
    assert optimizer_spec is not None
    missing_context = [
        key
        for key in ("schedule_origin_step", "current_step", "optimizer_count_at_current_step")
        if context.get(key) is _SCHEDULE_MISSING
    ]
    context_independent = (
        optimizer_spec.lr_schedule is None or optimizer_spec.lr_schedule.kind == "constant"
    )
    if missing_context and not context_independent:
        return missing_input_check(check_id, *missing_context)

    try:
        eval_context = require_schedule_context(
            (
                {
                    "schedule_origin_step": 0,
                    "current_step": 0,
                    "optimizer_count_at_current_step": 0,
                }
                if missing_context
                else context
            ),
            label="resume_context",
        )
        declared_coordinates = _declared_mapping_coordinates(row)
        if declared_coordinates is not None and set(trace) != declared_coordinates:
            raise ValueError(
                "lr_trace mapped coordinate coverage mismatch; "
                f"declared={sorted(declared_coordinates)!r} observed={sorted(trace)!r}"
            )
        expected = {}
        observed = {}
        for coordinates, coordinate_trace in trace.items():
            samples = _selected_lr_samples(coordinate_trace, optimizer_spec)
            expected[coordinates] = {
                step: learning_rate_from_build_optimizer(
                    optimizer_spec,
                    sample_step=step,
                    schedule_origin_step=eval_context.schedule_origin_step,
                    current_step=eval_context.current_step,
                    optimizer_count_at_current_step=eval_context.optimizer_count_at_current_step,
                )
                for step in samples
            }
            observed[coordinates] = {
                step: float(coordinate_trace[step]) for step in samples
            }
    except Exception as exc:
        return fail_check(
            check_id,
            expected="learning rates derived via build_optimizer",
            observed=type(exc).__name__,
            detail=str(exc),
        )

    mismatches = {
        f"{coordinates!r}@{step}": {
            "expected": coordinate_expected[step],
            "observed": observed[coordinates][step],
        }
        for coordinates, coordinate_expected in expected.items()
        for step in coordinate_expected
        if not _close(observed[coordinates][step], coordinate_expected[step], rel_tol=1e-6)
    }
    if not mismatches:
        if set(trace) == {()}:
            return pass_check(check_id, expected=expected[()], observed=observed[()])
        return pass_check(
            check_id,
            expected={repr(key): value for key, value in expected.items()},
            observed={repr(key): value for key, value in observed.items()},
        )
    if set(trace) == {()}:
        return fail_check(
            check_id,
            expected=expected[()],
            observed=observed[()],
            detail=str(mismatches),
        )
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
    training_manifest = _training_spec_payload(_manifest_payload(row))
    candidates = (
        ("bundle_row_spec.optimizer", _path(row.bundle_row_spec, "optimizer")),
        ("bundle_row_spec.optimizer_spec", _path(row.bundle_row_spec, "optimizer_spec")),
        ("bundle_row_spec.training.optimizer", _path(row.bundle_row_spec, "training", "optimizer")),
        (
            "bundle_row_spec.training_config.optimizer",
            _path(row.bundle_row_spec, "training_config", "optimizer"),
        ),
        (
            "bundle_row_spec.training_spec.method_payload.payload.optimizer",
            _path(
                row.bundle_row_spec,
                "training_spec",
                "method_payload",
                "payload",
                "optimizer",
            ),
        ),
        (
            "bundle_row_spec.method_payload.payload.optimizer",
            _path(row.bundle_row_spec, "method_payload", "payload", "optimizer"),
        ),
        (
            "bundle_row_spec.method_payload.payload.training.optimizer",
            _path(row.bundle_row_spec, "method_payload", "payload", "training", "optimizer"),
        ),
        (
            "bundle_row_spec.method_payload.payload.controller_optimizer",
            _path(row.bundle_row_spec, "method_payload", "payload", "controller_optimizer"),
        ),
        (
            "bundle_row_spec.training_spec.method_payload.payload.controller_optimizer",
            _path(
                row.bundle_row_spec,
                "training_spec",
                "method_payload",
                "payload",
                "controller_optimizer",
            ),
        ),
        (
            "manifest.training_spec.method_payload.payload.optimizer",
            _path(training_manifest, "method_payload", "payload", "optimizer"),
        ),
        (
            "manifest.training_spec.method_payload.payload.training.optimizer",
            _path(training_manifest, "method_payload", "payload", "training", "optimizer"),
        ),
        (
            "manifest.training_spec.method_payload.payload.controller_optimizer",
            _path(training_manifest, "method_payload", "payload", "controller_optimizer"),
        ),
    )
    present = [(location, payload) for location, payload in candidates if payload is not _MISSING]
    if not present:
        return _MISSING

    normalized: list[tuple[str, dict[str, Any]]] = []
    for location, payload in present:
        try:
            spec = OptimizerSpec.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"invalid governed optimizer spec at {location}: {exc}") from exc
        normalized.append((location, spec.model_dump(mode="json")))
    authority = normalized[0][1]
    conflicts = [location for location, payload in normalized[1:] if payload != authority]
    if conflicts:
        locations = [normalized[0][0], *conflicts]
        raise ValueError(f"ambiguous governed optimizer specs at {locations!r}")
    return authority


def _lr_trace(
    row: ConformanceRowArtifacts,
) -> dict[tuple[tuple[str, int], ...], dict[int, float]] | None:
    raw = _first_present(
        _path(row.training_diagnostics, "lr_trace"),
        _path(row.training_diagnostics, "learning_rate_trace"),
    )
    if raw is _MISSING:
        return None
    if isinstance(raw, Mapping):
        return {(): {int(step): float(value) for step, value in raw.items()}}
    trace: dict[tuple[tuple[str, int], ...], dict[int, float]] = {}
    for item in raw:
        step = _first_present(_path(item, "step"), _path(item, "batch"), _path(item, "coordinate"))
        value = _first_present(_path(item, "learning_rate"), _path(item, "lr"))
        if step is _MISSING or value is _MISSING:
            raise ValueError(f"invalid lr_trace item {item!r}")
        raw_coordinates = _path(item, "axis_coordinates")
        if raw_coordinates is _MISSING or raw_coordinates is None:
            raw_coordinates = ()
        coordinates = tuple(
            (str(coordinate["axis"]), int(coordinate["index"]))
            for coordinate in raw_coordinates
        )
        coordinate_trace = trace.setdefault(coordinates, {})
        if int(step) in coordinate_trace:
            raise ValueError(
                f"duplicate lr_trace coordinate/step identity {(coordinates, int(step))!r}"
            )
        coordinate_trace[int(step)] = float(value)
    return trace


def _declared_mapping_coordinates(
    row: ConformanceRowArtifacts,
) -> set[tuple[tuple[str, int], ...]] | None:
    training = _training_spec_payload(_manifest_payload(row))
    worker = _first_present(
        _path(row.bundle_row_spec, "worker_execution"),
        _path(training, "worker_execution"),
    )
    levels = _path(worker, "mapping_levels")
    axes = _path(worker, "method_contract", "axes")
    if not isinstance(levels, list) or not levels:
        return None
    if len(levels) != 1 or not isinstance(axes, list):
        raise ValueError("lr_trace conformance supports exactly one declared mapping level")
    axis_name = str(levels[0]["axis"])
    declaration = next((axis for axis in axes if axis.get("name") == axis_name), None)
    if not isinstance(declaration, Mapping) or not isinstance(declaration.get("size"), int):
        raise ValueError(f"mapped axis {axis_name!r} lacks a declared size")
    return {((axis_name, index),) for index in range(int(declaration["size"]))}


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


def _checkpoint_completed_batches(row: ConformanceRowArtifacts) -> list[int] | None:
    transactions = _path(row.training_diagnostics, "checkpoint_transactions")
    if transactions is _MISSING:
        return None
    completed: list[int] = []
    for transaction in transactions:
        value = _first_present(
            _path(transaction, "cumulative_completed_batches"),
            _path(transaction, "completed_batches"),
        )
        if value is _MISSING:
            return None
        completed.append(int(value))
    return completed


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
    return _event_type(event) in RUN_EVENT_TERMINAL_TYPES


def _event_type(event: Mapping[str, Any]) -> str | None:
    value = _first_present(_path(event, "type"), _path(event, "event_type"))
    return None if value is _MISSING else str(value)


def _event_status(event: Mapping[str, Any]) -> str | None:
    value = _path(event, "payload", "status")
    return None if value is _MISSING else str(value)


def _expected_terminal_statuses(row: ConformanceRowArtifacts) -> dict[str, Any]:
    candidates = (
        (
            "bundle_row_spec.expected_terminal_status",
            _path(row.bundle_row_spec, "expected_terminal_status"),
        ),
        ("bundle_row_spec.sentinel_status", _path(row.bundle_row_spec, "sentinel_status")),
        ("row_status", _row_terminal_status(row.row_status)),
        (
            "training_diagnostics.terminal_status",
            _path(row.training_diagnostics, "terminal_status"),
        ),
    )
    return {source: status for source, status in candidates if status is not _MISSING}


def _row_terminal_status(status: str | None) -> Any:
    if status is None:
        return _MISSING
    canonical = str(status).lower()
    if canonical == "stopped":
        return "cancelled"
    return canonical


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
