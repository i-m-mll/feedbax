"""Interpretability APIs for flattened authored intent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import Field

from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.run_composition import FlattenedIntent

if TYPE_CHECKING:
    from feedbax.contracts.migrations import SpecSchemaRegistry


@dataclass(frozen=True)
class SemanticDifference:
    path: str
    left: Any
    right: Any
    left_layer: str | None = None
    right_layer: str | None = None


class ContractIdentityReference(StrictModel):
    schema_id: str
    schema_version: str
    layer: str | None = None


class StaleContractIdentity(StrictModel):
    schema_id: str
    found_version: str
    current_version: str
    layer: str | None = None


class StalenessReport(StrictModel):
    stale: list[StaleContractIdentity] = Field(default_factory=list)
    unresolved: list[ContractIdentityReference] = Field(default_factory=list)


def collect_contract_identities(flattened: FlattenedIntent) -> list[ContractIdentityReference]:
    """Collect schema identities and the authored layer that introduced each one."""
    identities: list[tuple[str, ContractIdentityReference]] = []
    _collect_contract_identities(flattened.payload, "", flattened.attribution, identities)
    return [
        identity
        for _, identity in sorted(identities, key=lambda item: _identity_sort_key(*item))
    ]


def staleness_report(
    flattened: FlattenedIntent,
    registry: SpecSchemaRegistry | None = None,
) -> StalenessReport:
    """Report stale and unknown schema identities without changing the authored intent."""
    if registry is None:
        from feedbax.contracts.migrations import default_spec_registry

        registry = default_spec_registry
    families_by_identity = {family.identity: family for family in registry.families()}
    stale: list[StaleContractIdentity] = []
    unresolved: list[ContractIdentityReference] = []
    for identity in collect_contract_identities(flattened):
        family = families_by_identity.get(identity.schema_id)
        if family is None:
            unresolved.append(identity)
        elif identity.schema_version != family.current_version:
            stale.append(
                StaleContractIdentity(
                    schema_id=identity.schema_id,
                    found_version=identity.schema_version,
                    current_version=family.current_version,
                    layer=identity.layer,
                )
            )
    return StalenessReport(stale=stale, unresolved=unresolved)


def layered_semantic_diff(
    left: FlattenedIntent | Mapping[str, Any],
    right: FlattenedIntent | Mapping[str, Any],
) -> list[SemanticDifference]:
    left_payload = left.payload if isinstance(left, FlattenedIntent) else left
    right_payload = right.payload if isinstance(right, FlattenedIntent) else right
    left_attr = left.attribution if isinstance(left, FlattenedIntent) else {}
    right_attr = right.attribution if isinstance(right, FlattenedIntent) else {}
    result: list[SemanticDifference] = []
    _diff(left_payload, right_payload, "", left_attr, right_attr, result)
    return result


def detect_near_duplicate_lanes(
    lanes: Mapping[str, FlattenedIntent | Mapping[str, Any]], *, max_differences: int = 1
) -> list[tuple[str, str, list[SemanticDifference]]]:
    names = sorted(lanes)
    result = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            differences = layered_semantic_diff(lanes[left], lanes[right])
            if len(differences) <= max_differences:
                result.append((left, right, differences))
    return result


def _diff(
    left: Any,
    right: Any,
    path: str,
    left_attr: Mapping[str, str],
    right_attr: Mapping[str, str],
    output: list[SemanticDifference],
) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            _diff(left.get(key), right.get(key), _join(path, str(key)), left_attr, right_attr, output)
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and isinstance(
        right, Sequence
    ) and not isinstance(right, (str, bytes)):
        for index in range(max(len(left), len(right))):
            lhs = left[index] if index < len(left) else None
            rhs = right[index] if index < len(right) else None
            _diff(lhs, rhs, _join(path, str(index)), left_attr, right_attr, output)
        return
    if left != right:
        output.append(
            SemanticDifference(path, left, right, left_attr.get(path), right_attr.get(path))
        )


def _join(parent: str, child: str) -> str:
    return child if not parent else f"{parent}.{child}"


def _collect_contract_identities(
    value: Any,
    path: str,
    attribution: Mapping[str, str],
    output: list[tuple[str, ContractIdentityReference]],
) -> None:
    if isinstance(value, Mapping):
        schema_id = value.get("schema_id")
        schema_version = value.get("schema_version")
        if isinstance(schema_id, str) and isinstance(schema_version, str):
            output.append(
                (
                    path,
                    ContractIdentityReference(
                        schema_id=schema_id,
                        schema_version=schema_version,
                        layer=_introducing_layer(path, attribution),
                    ),
                )
            )
        for key in sorted(value, key=str):
            _collect_contract_identities(value[key], _join(path, str(key)), attribution, output)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _collect_contract_identities(child, _join(path, str(index)), attribution, output)


def _introducing_layer(path: str, attribution: Mapping[str, str]) -> str | None:
    identity_paths = (
        _join(path, "schema_version"),
        _join(path, "schema_id"),
        path,
    )
    for candidate in identity_paths:
        if candidate in attribution:
            return attribution[candidate]

    prefixes = sorted(
        (candidate for candidate in attribution if path.startswith(f"{candidate}.")),
        key=len,
        reverse=True,
    )
    return attribution[prefixes[0]] if prefixes else None


def _identity_sort_key(
    path: str, identity: ContractIdentityReference
) -> tuple[str, str, str, str]:
    return (identity.schema_id, identity.schema_version, identity.layer or "", path)
