"""Durable run-bundle contract for Feedbax orchestration."""

from __future__ import annotations

import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.manifest import StrictModel


RUN_BUNDLE_SCHEMA_ID = "feedbax.orchestration.run_bundle"
RUN_BUNDLE_SCHEMA_VERSION_V1 = "feedbax.orchestration.run_bundle.v1"
RUN_BUNDLE_SCHEMA_VERSION = "feedbax.orchestration.run_bundle.v2"
ROW_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def mint_run_set_id(now: datetime | None = None) -> str:
    """Mint ``<utc-date>-<8-hex>`` run-set identity."""
    timestamp = now or datetime.now(timezone.utc)
    return f"{timestamp.astimezone(timezone.utc).date().isoformat()}-{secrets.token_hex(4)}"


def default_orchestration_root(run_set_id: str) -> Path:
    """Return the default orchestration root for a run set."""
    configured = os.environ.get("FEEDBAX_ORCHESTRATION_ROOT")
    if configured:
        return Path(configured).expanduser() / run_set_id
    return Path.home() / ".cache" / "feedbax" / "orchestration" / run_set_id


class RunRowSpec(StrictModel):
    """One row in a run bundle."""

    row_id: str = Field(min_length=1)
    run_spec: dict[str, Any] | str | None = None
    command: list[str] = Field(default_factory=list)
    entry: str | None = None
    payload_sha256: str | None = None
    collect: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("row_id")
    @classmethod
    def _validate_row_id(cls, value: str) -> str:
        if not ROW_ID_RE.fullmatch(value):
            raise ValueError("row_id must match ^[A-Za-z0-9_.-]+$")
        return value

    @model_validator(mode="after")
    def _validate_launch_entry(self) -> "RunRowSpec":
        if not self.command and not self.entry:
            raise ValueError("row requires command or entry")
        return self


class RepoRevision(StrictModel):
    """Repository revision declaration used in environment fingerprints."""

    path: str = "."
    revision: str
    dirty_allowed: bool = False


class EnvironmentDeclaration(StrictModel):
    """Declared execution environment for a run bundle."""

    python_version: str | None = None
    repo_revisions: list[RepoRevision] = Field(default_factory=list)
    lockfile_hashes: dict[str, str] = Field(default_factory=dict)
    overlay_steps: list[str] = Field(default_factory=list)
    image_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LaunchPolicy(StrictModel):
    """Row launch concurrency policy."""

    max_parallel_rows: int = Field(default=1, ge=1)
    warm_first: bool = False
    stagger_seconds: float = Field(default=0.0, ge=0.0)


class BudgetPolicy(StrictModel):
    """Run-set budget guards enforced by MONITOR."""

    max_wall_clock_seconds: float = Field(gt=0.0)
    max_spend_usd: float | None = Field(default=None, ge=0.0)


class InputCustodyPin(StrictModel):
    """Immutable input custody pin."""

    role: str = Field(min_length=1)
    checkpoint_transaction_id: str = Field(min_length=1)

    @field_validator("checkpoint_transaction_id")
    @classmethod
    def _reject_mutable_latest(cls, value: str) -> str:
        if "latest.json" in value:
            raise ValueError(
                "input custody pins must name checkpoint transactions, not latest.json"
            )
        return value


class RunBundle(StrictModel):
    """Schema-versioned orchestration request for a run set."""

    schema_id: Literal["feedbax.orchestration.run_bundle"] = RUN_BUNDLE_SCHEMA_ID
    schema_version: Literal["feedbax.orchestration.run_bundle.v2"] = RUN_BUNDLE_SCHEMA_VERSION
    run_set_id: str = Field(default_factory=mint_run_set_id)
    driver: str = "local"
    rows: list[RunRowSpec] = Field(min_length=1)
    environment: EnvironmentDeclaration
    launch_policy: LaunchPolicy = Field(default_factory=LaunchPolicy)
    budget: BudgetPolicy
    input_custody_pins: list[InputCustodyPin] = Field(default_factory=list)
    orchestration_root: str | None = None
    keep_alive: bool = False
    deadman_enabled: bool = False
    deadman_silence_seconds: int = Field(default=1800, ge=60)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_bundle(self) -> "RunBundle":
        seen: set[str] = set()
        for row in self.rows:
            if row.row_id in seen:
                raise ValueError(f"duplicate row_id: {row.row_id!r}")
            seen.add(row.row_id)
        return self

    @property
    def run_set_dir(self) -> Path:
        """Return the local directory where orchestration state is stored."""
        if self.orchestration_root:
            root = Path(self.orchestration_root).expanduser()
            return root if root.name == self.run_set_id else root / self.run_set_id
        return default_orchestration_root(self.run_set_id)

    def row(self, row_id: str) -> RunRowSpec:
        """Return one row spec by id."""
        for row in self.rows:
            if row.row_id == row_id:
                return row
        raise KeyError(row_id)
