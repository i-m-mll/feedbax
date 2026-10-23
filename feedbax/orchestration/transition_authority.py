"""Typed ownership for durable orchestration transition domains."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionAuthority:
    """One durable identity domain and its closed transition vocabulary."""

    domain: str
    identity_field: str
    transitions: frozenset[str]

    def __post_init__(self) -> None:
        if not self.domain or not self.identity_field or not self.transitions:
            raise ValueError("transition authority must name a domain, identity, and transitions")


def assert_disjoint_transition_authorities(*authorities: TransitionAuthority) -> None:
    """Reject two state machines that claim the same durable identity domain."""
    domains = [authority.domain for authority in authorities]
    if len(domains) != len(set(domains)):
        raise ValueError(f"durable transition authority domains overlap: {domains!r}")


__all__ = ["TransitionAuthority", "assert_disjoint_transition_authorities"]
