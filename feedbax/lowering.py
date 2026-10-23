"""Generic ordered lowering contributions.

This module provides extension machinery for consumers that need several
independent, pure contributors to lower one context into fragments.  It does
not prescribe a fragment schema or merge policy; consumers own those semantic
decisions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar


ContextT = TypeVar("ContextT")
FragmentT = TypeVar("FragmentT")


class LowererExecutionError(RuntimeError):
    """Raised when a registered lowerer fails while producing a fragment."""

    def __init__(self, *, lowerer_id: str, owner: str, detail: str) -> None:
        self.lowerer_id = lowerer_id
        self.owner = owner
        self.detail = detail
        super().__init__(f"lowerer {lowerer_id!r} owned by {owner!r} failed: {detail}")


@dataclass(frozen=True)
class LowererRegistration(Generic[ContextT, FragmentT]):
    """Registration for one context-to-fragment lowerer.

    Lowerers should be pure functions of their context.  Returning ``None``
    declares that the contribution is inactive for that context.
    """

    lowerer_id: str
    order: int
    owner: str
    lowerer: Callable[[ContextT], FragmentT | None]


@dataclass(frozen=True)
class LoweredContribution(Generic[FragmentT]):
    """One active fragment together with its registration provenance."""

    lowerer_id: str
    order: int
    owner: str
    fragment: FragmentT


class OrderedLowererRegistry(Generic[ContextT, FragmentT]):
    """Registry of independent lowerers executed by ``(order, lowerer_id)``.

    The registry deliberately returns an ordered tuple of contributions rather
    than merging fragments.  This keeps conflict handling and all domain
    semantics with the consumer that defines ``FragmentT``.
    """

    def __init__(
        self,
        registrations: Iterable[LowererRegistration[ContextT, FragmentT]] = (),
    ) -> None:
        self._registrations: dict[str, LowererRegistration[ContextT, FragmentT]] = {}
        for registration in registrations:
            self.register(registration)

    def register(
        self,
        registration: LowererRegistration[ContextT, FragmentT],
    ) -> None:
        """Register one lowerer, rejecting invalid or duplicate identities."""

        if not isinstance(registration, LowererRegistration):
            raise TypeError("ordered lowerer registration must be a LowererRegistration")
        if not isinstance(registration.lowerer_id, str) or not registration.lowerer_id:
            raise ValueError("lowerer_id must be a non-empty string")
        if registration.lowerer_id != registration.lowerer_id.strip():
            raise ValueError("lowerer_id must not contain leading or trailing whitespace")
        if not isinstance(registration.owner, str) or not registration.owner:
            raise ValueError(
                f"lowerer {registration.lowerer_id!r} owner must be a non-empty string"
            )
        if registration.owner != registration.owner.strip():
            raise ValueError(
                f"lowerer {registration.lowerer_id!r} owner must not contain "
                "leading or trailing whitespace"
            )
        if isinstance(registration.order, bool) or not isinstance(registration.order, int):
            raise TypeError(f"lowerer {registration.lowerer_id!r} order must be an integer")
        if not callable(registration.lowerer):
            raise TypeError(
                f"lowerer {registration.lowerer_id!r} owned by "
                f"{registration.owner!r} must be callable"
            )

        existing = self._registrations.get(registration.lowerer_id)
        if existing is not None:
            raise ValueError(
                f"lowerer {registration.lowerer_id!r} is already registered by "
                f"{existing.owner!r}; attempted owner={registration.owner!r}"
            )
        self._registrations[registration.lowerer_id] = registration

    def available_ids(self) -> tuple[str, ...]:
        """Return registered lowerer IDs in deterministic execution order."""

        return tuple(registration.lowerer_id for registration in self.registrations())

    def registrations(
        self,
    ) -> tuple[LowererRegistration[ContextT, FragmentT], ...]:
        """Return registrations in deterministic execution order."""

        return tuple(
            sorted(
                self._registrations.values(),
                key=lambda registration: (registration.order, registration.lowerer_id),
            )
        )

    def lower(self, context: ContextT) -> tuple[LoweredContribution[FragmentT], ...]:
        """Lower ``context`` through every active registered contribution."""

        contributions: list[LoweredContribution[FragmentT]] = []
        for registration in self.registrations():
            try:
                fragment = registration.lowerer(context)
            except Exception as exc:
                raise LowererExecutionError(
                    lowerer_id=registration.lowerer_id,
                    owner=registration.owner,
                    detail=str(exc) or type(exc).__name__,
                ) from exc
            if fragment is None:
                continue
            contributions.append(
                LoweredContribution(
                    lowerer_id=registration.lowerer_id,
                    order=registration.order,
                    owner=registration.owner,
                    fragment=fragment,
                )
            )
        return tuple(contributions)


__all__ = [
    "LoweredContribution",
    "LowererExecutionError",
    "LowererRegistration",
    "OrderedLowererRegistry",
]
