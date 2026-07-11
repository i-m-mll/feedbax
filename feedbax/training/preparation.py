"""Runtime-only preparation hooks for plugin-owned training methods."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from feedbax.contracts.training import TrainingRunSpec
from feedbax.objectives.service import LossService
from feedbax.training.checkpoint_custody import ResumeSlotTransform

_RESERVED_KERNEL_CONTEXT_KEYS = frozenset({"run_spec", "method_payload"})


class ExecutionPreparationError(ValueError):
    """Raised when a training method cannot prepare its runtime inputs."""


@dataclass(frozen=True)
class ExecutionPreparationRequest:
    """Validated runtime request passed to an execution-preparation provider."""

    run_spec: TrainingRunSpec
    run_id: str | None = None
    resume: bool = False


@dataclass(frozen=True)
class ExecutionPreparationResult:
    """Narrow set of runtime-only values a provider may supply to the executor."""

    initial_slots: Mapping[str, Any]
    kernel_context: Mapping[str, Any] = field(default_factory=dict)
    loss_service: LossService | None = None
    resume_slot_transform: ResumeSlotTransform | None = None


@runtime_checkable
class ExecutionPreparationProvider(Protocol):
    """Callable that prepares runtime-only inputs for one training method."""

    def __call__(self, request: ExecutionPreparationRequest) -> ExecutionPreparationResult:
        """Prepare executor inputs without mutating ``request.run_spec``."""
        ...


@dataclass(frozen=True)
class ExecutionPreparationRegistration:
    """One method-ref keyed execution-preparation provider registration."""

    method_ref: str
    provider: ExecutionPreparationProvider
    owner: str = "feedbax"


class ExecutionPreparationProviderRegistry:
    """Registry containing at most one runtime provider per training method ref."""

    def __init__(self) -> None:
        self._registrations: dict[str, ExecutionPreparationRegistration] = {}

    def register(self, registration: ExecutionPreparationRegistration) -> None:
        """Register one provider, rejecting ambiguous duplicate ownership."""
        if not registration.method_ref:
            raise ValueError("execution preparation method_ref must not be empty")
        if not callable(registration.provider):
            raise TypeError(
                f"execution preparation provider for {registration.method_ref!r} must be callable"
            )
        if registration.method_ref in self._registrations:
            existing = self._registrations[registration.method_ref]
            raise ValueError(
                "execution preparation provider already registered for "
                f"{registration.method_ref!r} by {existing.owner!r}"
            )
        self._registrations[registration.method_ref] = registration

    def available_keys(self) -> tuple[str, ...]:
        """Return method refs with registered preparation providers."""
        return tuple(sorted(self._registrations))

    def get(self, method_ref: str) -> ExecutionPreparationRegistration | None:
        """Return a provider registration when one exists for ``method_ref``."""
        return self._registrations.get(method_ref)

    def prepare(self, request: ExecutionPreparationRequest) -> ExecutionPreparationResult:
        """Invoke the matching provider while enforcing immutability and result shape."""
        method_ref = request.run_spec.method_ref.key
        registration = self.get(method_ref)
        if registration is None:
            raise ExecutionPreparationError(
                f"/method_ref {method_ref!r} has no execution-preparation provider; "
                f"available provider keys={list(self.available_keys())!r}"
            )

        provider_spec = deepcopy(request.run_spec)
        before = provider_spec.model_dump_json()
        provider_request = ExecutionPreparationRequest(
            run_spec=provider_spec,
            run_id=request.run_id,
            resume=request.resume,
        )
        try:
            result = registration.provider(provider_request)
        except Exception as exc:
            raise ExecutionPreparationError(
                f"execution preparation failed for method_ref {method_ref!r} "
                f"(provider owner={registration.owner!r}): {exc}"
            ) from exc

        if provider_spec.model_dump_json() != before:
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} mutated TrainingRunSpec"
            )
        if not isinstance(result, ExecutionPreparationResult):
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned "
                f"{type(result).__name__}; expected ExecutionPreparationResult"
            )
        if not isinstance(result.initial_slots, Mapping):
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned non-mapping "
                "initial_slots"
            )
        if not isinstance(result.kernel_context, Mapping):
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned non-mapping "
                "kernel_context"
            )
        reserved_keys = _RESERVED_KERNEL_CONTEXT_KEYS.intersection(result.kernel_context)
        if reserved_keys:
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned reserved "
                f"kernel_context keys {sorted(reserved_keys)!r}"
            )
        if result.loss_service is not None and not isinstance(result.loss_service, LossService):
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned invalid loss_service"
            )
        if result.resume_slot_transform is not None and not callable(result.resume_slot_transform):
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned non-callable "
                "resume_slot_transform"
            )
        return result


DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY = ExecutionPreparationProviderRegistry()


def require_execution_preparation_provider(
    *,
    method_ref: str,
    preparation_registry: ExecutionPreparationProviderRegistry,
) -> None:
    """Fail clearly when an opted-in method lacks its required runtime provider."""
    if preparation_registry.get(method_ref) is None:
        raise ExecutionPreparationError(
            f"/method_ref {method_ref!r} requires an execution-preparation provider, but none "
            f"is registered; available provider keys={list(preparation_registry.available_keys())!r}"
        )
