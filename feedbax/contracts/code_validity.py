"""Portable receipt evidence for code and environment validity."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import AwareDatetime, Field, PositiveInt, field_validator, model_validator

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.computation_validity import ComputationTrace, DriverIdentity
from feedbax.contracts.request_refs import Name, ReferenceValue, Sha256


class SourcePath(ReferenceValue):
    scope: Name
    path: Name

    @field_validator("scope")
    @classmethod
    def _portable_scope(cls, value: str) -> str:
        distribution, separator, portion = value.removeprefix("plugin:").partition(":")
        if value not in {"project", "feedbax"} and not (
            value.startswith("plugin:") and distribution.isascii()
            and distribution.replace("-", "").isalnum()
            and (not separator or (portion.isascii() and portion.isdecimal()))
        ):
            raise ValueError("Expected a project, Feedbax, or installed plugin source scope")
        return value

    @field_validator("path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.as_posix() != value:
            raise ValueError("Expected a canonical path relative to the source scope")
        return value


class SourceSnapshot(ReferenceValue):
    path: SourcePath
    content: ContentObjectRef
    entered: tuple[Name, ...]
    host_product: tuple[Name, ...] = Field(default=(), exclude_if=lambda value: not value)


class SourceDifference(ReferenceValue):
    kind: Literal["source"] = "source"
    path: SourcePath
    before: Sha256 | None
    after: Sha256 | None
    status: Literal["changed", "removed", "unreadable", "invalid", "indeterminate"]
    statements: tuple[str, ...] = ()
    functions: tuple[str, ...] = ()


class RegistryBinding(ReferenceValue):
    family: Name
    requested_key: Name
    key: Name
    distribution: Name
    version: Name
    module: Name
    qualified_name: Name
    source_path: SourcePath | None = Field(default=None, exclude_if=lambda value: value is None)


class FileSnapshot(ReferenceValue):
    path: SourcePath
    sha256: Sha256


class ComputationUnavailable(ReferenceValue):
    reason: Literal["arguments"] = "arguments"
    driver: DriverIdentity
    completed_steps: PositiveInt
    detail: Name | None = None
    slots: tuple[Name, ...] = ()

    @model_validator(mode="after")
    def _exact_slot_count(self) -> ComputationUnavailable:
        if self.slots and (len(self.slots) != self.completed_steps or len(set(self.slots)) != len(self.slots)):
            raise ValueError("The unavailable computation must name each completed slot once")
        return self


class CodeTrace(ReferenceValue):
    sources: tuple[SourceSnapshot, ...]
    registry: tuple[RegistryBinding, ...]
    files: tuple[FileSnapshot, ...]
    environment: dict[Name, Name]
    indeterminate: tuple[SourcePath, ...]
    indeterminate_files: tuple[SourcePath, ...] = Field(default=(), exclude_if=lambda value: not value)
    prefixes: tuple[ContentObjectRef, ...] = Field(default=(), exclude_if=lambda value: not value)
    computation: ComputationTrace | None = Field(default=None, exclude_if=lambda value: value is None)
    computation_unavailable: ComputationUnavailable | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _computation_proof_shape(self) -> CodeTrace:
        if self.computation is not None and self.computation_unavailable is not None:
            raise ValueError("A computation has either a captured proof or a reason it is unavailable")
        return self


class AttemptProvenance(ReferenceValue):
    feedbax_commit: Name
    feedbax_dirty: bool
    lock_sha256: Sha256 | None
    platform: Name
    started_at: AwareDatetime
    ended_at: AwareDatetime

    @model_validator(mode="after")
    def _ordered_time(self) -> AttemptProvenance:
        if self.ended_at < self.started_at:
            raise ValueError("Attempt end precedes its start")
        return self


class ReceiptEvidence(ReferenceValue):
    attempt: Name
    trace: CodeTrace
    provenance: AttemptProvenance
