"""Typed evidence for the bounded pre-launch remote smoke stage."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.execution_context import NativeExecutionProducerContext
from feedbax.contracts.base import StrictModel


REMOTE_SMOKE_EVIDENCE_SCHEMA_ID = "feedbax.orchestration.remote_smoke_evidence"
REMOTE_SMOKE_EVIDENCE_SCHEMA_VERSION = f"{REMOTE_SMOKE_EVIDENCE_SCHEMA_ID}.v1"


def _validate_sha256(value: str, *, field_name: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")
    return value


class RemoteSmokeRowEvidence(StrictModel):
    """Durable outcome of the bounded native smoke for one run-bundle row."""

    row_id: str = Field(min_length=1)
    status: Literal["passed", "failed", "opted-out"]
    planned_run_id: str | None = Field(default=None, min_length=1)
    derived_run_id: str | None = Field(default=None, min_length=1)
    derived_producer_context: NativeExecutionProducerContext | None = None
    scratch_namespace: str | None = Field(default=None, min_length=1)
    start_completed_batches: int | None = Field(default=None, ge=0)
    end_completed_batches: int | None = Field(default=None, ge=0)
    update_budget: int = Field(ge=1)
    payload_binding_status: Literal["verified", "not-run"]
    executor_result_sha256: str | None = None
    protected_paths_before: dict[str, str] = Field(default_factory=dict)
    protected_paths_after: dict[str, str] = Field(default_factory=dict)
    cleanup_status: Literal["removed", "not-created", "failed"]
    deadline_seconds: int = Field(ge=60)
    opt_out_reason: str | None = Field(default=None, min_length=1)

    @field_validator("update_budget", mode="before")
    @classmethod
    def _validate_update_budget(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("update_budget must be a positive integer, not a boolean")
        return value

    @field_validator("executor_result_sha256")
    @classmethod
    def _validate_executor_result_sha256(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_sha256(value, field_name="executor_result_sha256")
        return value

    @field_validator("protected_paths_before", "protected_paths_after")
    @classmethod
    def _validate_protected_path_digests(cls, value: dict[str, str]) -> dict[str, str]:
        for path_name, digest in value.items():
            if not path_name:
                raise ValueError("protected path digest names must be non-empty")
            _validate_sha256(digest, field_name=f"protected path {path_name!r} digest")
        return value

    @model_validator(mode="after")
    def _validate_status_contract(self) -> "RemoteSmokeRowEvidence":
        if self.status == "opted-out":
            if self.opt_out_reason is None:
                raise ValueError("opted-out remote smoke evidence requires opt_out_reason")
            if self.payload_binding_status != "not-run":
                raise ValueError("opted-out remote smoke evidence must record payload not-run")
            runtime_values = (
                self.planned_run_id,
                self.derived_run_id,
                self.derived_producer_context,
                self.scratch_namespace,
                self.start_completed_batches,
                self.end_completed_batches,
                self.executor_result_sha256,
            )
            if any(value is not None for value in runtime_values):
                raise ValueError("opted-out remote smoke evidence cannot claim runtime results")
            if self.protected_paths_before or self.protected_paths_after:
                raise ValueError("opted-out remote smoke evidence cannot claim protected digests")
            return self

        if self.opt_out_reason is not None:
            raise ValueError("executed remote smoke evidence cannot include opt_out_reason")
        required = {
            "planned_run_id": self.planned_run_id,
            "derived_run_id": self.derived_run_id,
            "derived_producer_context": self.derived_producer_context,
            "scratch_namespace": self.scratch_namespace,
            "start_completed_batches": self.start_completed_batches,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(f"executed remote smoke evidence lacks required fields: {missing!r}")
        assert self.planned_run_id is not None
        assert self.derived_run_id is not None
        assert self.derived_producer_context is not None
        if self.derived_run_id == self.planned_run_id:
            raise ValueError("derived_run_id must differ from planned_run_id")
        provenance = self.derived_producer_context.execution.row_provenance
        if provenance is None or provenance.planned_run_id != self.derived_run_id:
            raise ValueError("derived producer context planned_run_id must match derived_run_id")
        if self.end_completed_batches is not None:
            assert self.start_completed_batches is not None
            completed_delta = self.end_completed_batches - self.start_completed_batches
            if completed_delta < 0:
                raise ValueError("end_completed_batches cannot precede start_completed_batches")
            if completed_delta > self.update_budget:
                raise ValueError("completed-batch delta cannot exceed update_budget authority")
        if self.status == "passed":
            if self.end_completed_batches is None:
                raise ValueError("passed remote smoke evidence requires end_completed_batches")
            if self.payload_binding_status != "verified":
                raise ValueError("passed remote smoke evidence requires verified payload binding")
            if self.executor_result_sha256 is None:
                raise ValueError("passed remote smoke evidence requires executor result digest")
            if self.cleanup_status != "removed":
                raise ValueError("passed remote smoke evidence requires removed scratch custody")
            if self.protected_paths_before != self.protected_paths_after:
                raise ValueError("protected path contents changed during remote smoke")
        return self


class RemoteSmokeEvidence(StrictModel):
    """Versioned per-row evidence for the pre-launch RunPod native smoke stage."""

    schema_id: Literal["feedbax.orchestration.remote_smoke_evidence"] = (
        REMOTE_SMOKE_EVIDENCE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.orchestration.remote_smoke_evidence.v1"] = (
        REMOTE_SMOKE_EVIDENCE_SCHEMA_VERSION
    )
    run_set_id: str = Field(min_length=1)
    bundle_sha256: str
    rows: tuple[RemoteSmokeRowEvidence, ...] = Field(min_length=1)

    @field_validator("bundle_sha256")
    @classmethod
    def _validate_bundle_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="bundle_sha256")

    @model_validator(mode="after")
    def _validate_unique_rows(self) -> "RemoteSmokeEvidence":
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("remote smoke evidence contains duplicate row_id values")
        return self
