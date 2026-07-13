"""Focused tests for the public ordered lowerer registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from feedbax import (
    LoweredContribution,
    LowererExecutionError,
    LowererRegistration,
    OrderedLowererRegistry,
)


@dataclass(frozen=True)
class Context:
    enabled: frozenset[str]


def _registration(
    lowerer_id: str,
    *,
    order: int,
    owner: str = "tests",
) -> LowererRegistration[Context, str]:
    return LowererRegistration(
        lowerer_id=lowerer_id,
        order=order,
        owner=owner,
        lowerer=lambda context: lowerer_id if lowerer_id in context.enabled else None,
    )


def test_lowering_is_deterministic_and_omits_inactive_contributions() -> None:
    registry = OrderedLowererRegistry[Context, str](
        [
            _registration("zeta", order=20, owner="package.zeta"),
            _registration("beta", order=10, owner="package.beta"),
            _registration("alpha", order=10, owner="package.alpha"),
        ]
    )

    assert registry.available_ids() == ("alpha", "beta", "zeta")
    assert registry.lower(Context(enabled=frozenset({"alpha", "zeta"}))) == (
        LoweredContribution(
            lowerer_id="alpha",
            order=10,
            owner="package.alpha",
            fragment="alpha",
        ),
        LoweredContribution(
            lowerer_id="zeta",
            order=20,
            owner="package.zeta",
            fragment="zeta",
        ),
    )


def test_fragment_merge_semantics_remain_consumer_owned() -> None:
    registry = OrderedLowererRegistry[Context, dict[str, int]](
        [
            LowererRegistration(
                lowerer_id="first",
                order=0,
                owner="tests",
                lowerer=lambda _context: {"first": 1},
            ),
            LowererRegistration(
                lowerer_id="second",
                order=1,
                owner="tests",
                lowerer=lambda _context: {"second": 2},
            ),
        ]
    )

    merged = {
        key: value
        for contribution in registry.lower(Context(enabled=frozenset()))
        for key, value in contribution.fragment.items()
    }

    assert merged == {"first": 1, "second": 2}


def test_duplicate_registration_names_existing_and_attempted_owners() -> None:
    registry = OrderedLowererRegistry[Context, str]()
    registry.register(_registration("shared", order=0, owner="first.owner"))

    with pytest.raises(
        ValueError,
        match="shared.*first\\.owner.*second\\.owner",
    ):
        registry.register(_registration("shared", order=1, owner="second.owner"))


@pytest.mark.parametrize(
    ("registration", "error_type", "message"),
    [
        (
            LowererRegistration("", 0, "tests", lambda _context: "fragment"),
            ValueError,
            "lowerer_id must be a non-empty string",
        ),
        (
            LowererRegistration(" padded ", 0, "tests", lambda _context: "fragment"),
            ValueError,
            "leading or trailing whitespace",
        ),
        (
            LowererRegistration("valid", 0, "", lambda _context: "fragment"),
            ValueError,
            "owner must be a non-empty string",
        ),
        (
            LowererRegistration("valid", cast(Any, True), "tests", lambda _context: "fragment"),
            TypeError,
            "order must be an integer",
        ),
        (
            LowererRegistration("valid", 0, "tests", cast(Any, "not-callable")),
            TypeError,
            "tests.*must be callable",
        ),
    ],
)
def test_invalid_registrations_are_rejected(
    registration: LowererRegistration[Context, str],
    error_type: type[Exception],
    message: str,
) -> None:
    registry = OrderedLowererRegistry[Context, str]()

    with pytest.raises(error_type, match=message):
        registry.register(registration)


def test_registration_requires_the_public_registration_type() -> None:
    registry = OrderedLowererRegistry[Context, str]()

    with pytest.raises(TypeError, match="must be a LowererRegistration"):
        registry.register(cast(Any, object()))


def test_execution_failure_attributes_lowerer_and_owner() -> None:
    def fail(_context: Context) -> str:
        raise LookupError("missing capability")

    registry = OrderedLowererRegistry[Context, str](
        [
            LowererRegistration(
                lowerer_id="broken",
                order=0,
                owner="package.broken",
                lowerer=fail,
            )
        ]
    )

    with pytest.raises(
        LowererExecutionError,
        match="broken.*package\\.broken.*missing capability",
    ) as exc_info:
        registry.lower(Context(enabled=frozenset()))

    assert exc_info.value.lowerer_id == "broken"
    assert exc_info.value.owner == "package.broken"
    assert isinstance(exc_info.value.__cause__, LookupError)
