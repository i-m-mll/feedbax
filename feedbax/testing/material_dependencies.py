"""Clean-wheel conformance checks for material-dependency declarations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from feedbax.contracts.material_dependencies import (
    MaterialDependencyObservation,
    MaterialDependencySet,
    validate_material_dependency_admission,
)


@dataclass(frozen=True)
class MaterialDependencyContractReport:
    """Compact result for a downstream dependency-admission conformance test."""

    identity_sha256: str
    dependency_count: int
    missing_canary: bool
    unauthentic_canary: bool


def check_material_dependency_contract(
    spec: MaterialDependencySet,
    observations: Sequence[MaterialDependencyObservation],
) -> MaterialDependencyContractReport:
    """Check one positive declaration and missing/unauthentic negative canaries."""
    admitted = validate_material_dependency_admission(spec, observations)
    probe_name = spec.dependencies[0].name
    missing = [observation for observation in observations if observation.name != probe_name]
    try:
        validate_material_dependency_admission(spec, missing)
    except ValueError as exc:
        missing_canary = probe_name in str(exc) and "missing" in str(exc)
    else:
        missing_canary = False
    if not missing_canary:
        raise AssertionError(
            f"missing material-dependency canary did not name {probe_name!r}"
        )

    unauthentic = [
        observation.model_copy(
            update={
                "authentic": False,
                "diagnostic": "feedbax conformance unauthentic canary",
            }
        )
        if observation.name == probe_name
        else observation
        for observation in observations
    ]
    try:
        validate_material_dependency_admission(spec, unauthentic)
    except ValueError as exc:
        unauthentic_canary = probe_name in str(exc) and "unauthentic" in str(exc)
    else:
        unauthentic_canary = False
    if not unauthentic_canary:
        raise AssertionError(
            f"unauthentic material-dependency canary did not name {probe_name!r}"
        )

    return MaterialDependencyContractReport(
        identity_sha256=admitted.identity_sha256,
        dependency_count=len(spec.dependencies),
        missing_canary=True,
        unauthentic_canary=True,
    )
