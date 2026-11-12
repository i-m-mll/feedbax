"""Check the retained field and value spellings against the active identity schema."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from feedbax.contracts.canonical_json import canonical_json_v2_bytes


@dataclass(frozen=True)
class IdentityAbsent:
    """Explicit stored meaning when an authoring default changes."""

    value: Any


@dataclass(frozen=True)
class IdentityWideInteger:
    """Encode integers outside the JCS safe range as decimal text."""


def _retired(contract: dict) -> tuple[dict, dict]:
    fields = {
        (row["model"], row["field"]): row["migration"] for row in contract["retired"]["fields"]
    }
    values = {
        (row["model"], row["field"], canonical_json_v2_bytes(row["value"])): row["migration"]
        for row in contract["retired"]["values"]
    }
    if len(fields) != len(contract["retired"]["fields"]) or len(values) != len(
        contract["retired"]["values"]
    ):
        raise ValueError("Duplicate retired identity spelling")
    return fields, values


def validate_identity_contract(contract: dict) -> None:
    """Refuse an active field or value that already has a retired spelling."""
    fields, values = _retired(contract)
    for model, members in contract["fields"].items():
        for name, row in members.items():
            if (model, name) in fields:
                raise ValueError(f"Retired identity field reused: {model}.{name}")
            for value in row.get("values", []):
                if (model, name, canonical_json_v2_bytes(value)) in values:
                    raise ValueError(f"Retired identity value reused: {model}.{name} = {value!r}")


def _absent(row: dict[str, Any]) -> bytes:
    return canonical_json_v2_bytes({"absent": row["absent"]} if "absent" in row else {})


def validate_identity_transition(old: dict, new: dict, *, migrations: Collection[str] = ()) -> None:
    """Check permanent absences and explicit retirements against a previous revision."""
    validate_identity_contract(new)
    old_fields, old_values = _retired(old)
    new_fields, new_values = _retired(new)
    if not old_fields.items() <= new_fields.items() or not old_values.items() <= new_values.items():
        raise ValueError("A retired identity spelling was removed or changed")
    for key in new_fields.keys() - old_fields.keys():
        if new_fields[key] not in migrations:
            raise ValueError(f"Identity retirement has no registered migration: {key}")
    for key in new_values.keys() - old_values.keys():
        if new_values[key] not in migrations:
            raise ValueError(f"Identity retirement has no registered migration: {key}")
    for model, fields in old["fields"].items():
        current = new["fields"].get(model, {})
        for name, prior in fields.items():
            if name not in current:
                if (model, name) not in new_fields:
                    raise ValueError(f"Identity field removed without retirement: {model}.{name}")
                continue
            active = current[name]
            old_families = set(prior.get("families", ()))
            active_families = set(active.get("families", ()))
            if bool(old_families) != bool(active_families) or not old_families <= active_families:
                raise ValueError(f"Identity family discriminator changed: {model}.{name}")
            if _absent(prior) != _absent(active):
                raise ValueError(f"Identity absent value changed: {model}.{name}")
            accepted = {canonical_json_v2_bytes(value) for value in active.get("values", [])}
            for value in prior.get("values", []):
                encoded = canonical_json_v2_bytes(value)
                if encoded not in accepted and (model, name, encoded) not in new_values:
                    raise ValueError(
                        f"Identity value removed without retirement: {model}.{name} = {value!r}"
                    )
