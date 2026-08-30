"""Provider-neutral invocation records for admitted scientific operations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from feedbax.contracts.manifest import StrictModel, canonical_json_bytes
from feedbax.workflow.plan import LogicalKey, WorkflowPlan


INVOCATION_SCHEMA_ID = "feedbax.spec.invocation"
INVOCATION_SCHEMA_VERSION = "feedbax.spec.invocation.v1"

_REALIZATION_KEYS = frozenset(
    {
        "backend",
        "backend_id",
        "custody_root",
        "execution_uri",
        "filesystem_root",
        "instance_id",
        "pod_id",
        "provider",
        "provider_id",
        "resource_handle",
        "storage_uri",
    }
)
_REALIZATION_PREFIXES = ("attempt_", "backend_", "custody_", "provider_", "retry_")


class FrozenRecord(StrictModel):
    """Immutable strict record used at the invocation boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InvocationInput(FrozenRecord):
    """One exact semantic or artifact input bound to an operation role."""

    role_path: tuple[str, ...] = Field(min_length=1)
    type_id: str = Field(min_length=1)
    reference: dict[str, Any]

    @field_validator("reference")
    @classmethod
    def _require_exact_neutral_reference(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("invocation input reference must identify an exact immutable input")
        _reject_realization_state(value, field_ref="invocation.input.reference")
        if not any(
            key in value
            for key in (
                "artifact_id",
                "content_hash",
                "execution_hash",
                "manifest_id",
                "receipt_id",
                "sha256",
            )
        ):
            raise ValueError(
                "invocation input reference requires an immutable identity or content digest"
            )
        return value


class InvocationOutput(FrozenRecord):
    """One typed output role requested from the operation."""

    role: str = Field(min_length=1)
    type_id: str = Field(min_length=1)


class InvocationExecutionPolicy(FrozenRecord):
    """Provider-neutral execution bounds that every realization must preserve."""

    timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(default=1, ge=1)
    cancellation: Literal["cooperative", "terminal-only"] = "cooperative"


class Invocation(FrozenRecord):
    """Exact provider-neutral request to execute one admitted workflow operation."""

    schema_id: Literal["feedbax.spec.invocation"] = INVOCATION_SCHEMA_ID
    schema_version: Literal["feedbax.spec.invocation.v1"] = INVOCATION_SCHEMA_VERSION
    invocation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    workflow_plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation_key: str = Field(min_length=1)
    operation: dict[str, Any]
    inputs: tuple[InvocationInput, ...] = ()
    requested_outputs: tuple[InvocationOutput, ...] = ()
    scientific_seeds: dict[str, int] = Field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    execution_policy: InvocationExecutionPolicy
    publication_policy_ref: str | None = None

    @model_validator(mode="after")
    def _validate_identity_and_canonical_order(self) -> "Invocation":
        _reject_realization_state(self.operation, field_ref="invocation.operation")
        if self.inputs != tuple(sorted(self.inputs, key=lambda item: item.role_path)):
            raise ValueError("invocation inputs must be in canonical role-path order")
        if len({item.role_path for item in self.inputs}) != len(self.inputs):
            raise ValueError("invocation input role paths must be unique")
        if self.requested_outputs != tuple(
            sorted(self.requested_outputs, key=lambda item: item.role)
        ):
            raise ValueError("invocation outputs must be in canonical role order")
        if len({item.role for item in self.requested_outputs}) != len(self.requested_outputs):
            raise ValueError("invocation output roles must be unique")
        if self.capabilities != tuple(sorted(set(self.capabilities))):
            raise ValueError("invocation capabilities must be unique and canonically ordered")
        if self.invocation_id != _invocation_identity(self):
            raise ValueError("invocation_id does not match canonical provider-neutral content")
        return self


class UnsupportedInvocationVersionError(ValueError):
    """An invocation document declares a schema family this build does not admit."""


def invocation_for_operation(
    plan: WorkflowPlan,
    operation_key: LogicalKey,
    *,
    bound_inputs: Mapping[tuple[str, ...], Mapping[str, Any]],
    execution_policy: InvocationExecutionPolicy,
    scientific_seeds: Mapping[str, int] | None = None,
    publication_policy_ref: str | None = None,
) -> Invocation:
    """Bind one admitted workflow operation to exact inputs without choosing a backend."""

    node = plan.node(operation_key)
    edges = plan.input_edges(operation_key)
    required = tuple(edge for edge in edges if edge.status in {"required", "guarded"})
    required_paths = {edge.role_path for edge in required}
    supplied_paths = set(bound_inputs)
    if required_paths != supplied_paths:
        missing = sorted(required_paths - supplied_paths)
        extra = sorted(supplied_paths - required_paths)
        raise ValueError(
            f"invocation bindings do not match operation inputs: missing={missing!r}, "
            f"extra={extra!r}"
        )
    inputs = tuple(
        InvocationInput(
            role_path=edge.role_path,
            type_id=edge.input_type,
            reference=dict(bound_inputs[edge.role_path]),
        )
        for edge in sorted(required, key=lambda item: item.role_path)
    )
    outputs = tuple(
        InvocationOutput(role=role, type_id=type_id)
        for role, type_id in sorted(node.operation.output_types.items())
    )
    content = {
        "workflow_plan_id": plan.identity,
        "operation_key": operation_key.text,
        "operation": node.operation.record(),
        "inputs": [item.model_dump(mode="json") for item in inputs],
        "requested_outputs": [item.model_dump(mode="json") for item in outputs],
        "scientific_seeds": dict(sorted((scientific_seeds or {}).items())),
        "capabilities": sorted(set(node.operation.capabilities)),
        "execution_policy": execution_policy.model_dump(mode="json"),
        "publication_policy_ref": publication_policy_ref,
    }
    return Invocation(invocation_id=_sha256(content), **content)


def invocation_from_document(document: Any) -> Invocation:
    """Load one explicit v1 invocation document or fail closed."""

    if not isinstance(document, Mapping):
        raise UnsupportedInvocationVersionError("invocation document must be a mapping")
    if document.get("schema_id") != INVOCATION_SCHEMA_ID:
        raise UnsupportedInvocationVersionError(
            f"unsupported invocation schema_id {document.get('schema_id')!r}; "
            f"expected {INVOCATION_SCHEMA_ID!r}"
        )
    if document.get("schema_version") != INVOCATION_SCHEMA_VERSION:
        raise UnsupportedInvocationVersionError(
            f"unsupported invocation schema_version {document.get('schema_version')!r}; "
            f"expected {INVOCATION_SCHEMA_VERSION!r}; no migration is defined"
        )
    return Invocation.model_validate(document)


def _invocation_identity(invocation: Invocation) -> str:
    return _sha256(
        invocation.model_dump(
            mode="json",
            exclude={"schema_id", "schema_version", "invocation_id"},
        )
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_realization_state(value: Any, *, field_ref: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower().replace("-", "_")
            if key in _REALIZATION_KEYS or key.startswith(_REALIZATION_PREFIXES):
                raise ValueError(
                    f"{field_ref} contains backend realization field {raw_key!r}; "
                    "provider and attempt state belongs in BackendPlan or Attempt"
                )
            _reject_realization_state(item, field_ref=f"{field_ref}.{raw_key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_realization_state(item, field_ref=f"{field_ref}[{index}]")


__all__ = [
    "INVOCATION_SCHEMA_ID",
    "INVOCATION_SCHEMA_VERSION",
    "Invocation",
    "InvocationExecutionPolicy",
    "InvocationInput",
    "InvocationOutput",
    "UnsupportedInvocationVersionError",
    "invocation_for_operation",
    "invocation_from_document",
]
