"""Choose a placement from measured capacity and the resolved Evaluate workload."""

from hashlib import sha256

from pydantic import Field

from feedbax.checkpoint_coordinates import finite_coordinates
from feedbax.contracts.base import StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.request_refs import Sha256
from feedbax.contracts.validation_placement import ValidationPlacement


class ValidationBudget(StrictModel):
    durable_point_p95_ms: float = Field(gt=0, allow_inf_nan=False)
    steady_update_ms: float = Field(gt=0, allow_inf_nan=False)
    max_point_ms: float = Field(gt=0, allow_inf_nan=False)
    max_update_fraction: float = Field(gt=0, allow_inf_nan=False)


class ValidationQualification(StrictModel):
    """Trusted local measurements tied to the exact runtime and saved workload."""

    train_request: str
    evaluate_request: str
    runtime: Sha256
    evidence: str = Field(min_length=1)
    budget: ValidationBudget


def validation_runtime(launch: dict) -> str:
    """Identify the worker configuration without the request or any delivery sidecar."""
    profile = {
        "argv": list(launch["argv"]), "env": launch["env"], "cwd": launch["cwd"],
        "warm_entrypoint": launch["warm_entrypoint"], "preload": list(launch["preload"]),
    }
    return sha256(canonical_json_v2_bytes(profile)).hexdigest()


def place_validation(
    updates: tuple[int, ...],
    *,
    producer_started: bool,
    live_products_supported: bool,
    budget: ValidationBudget | None,
) -> ValidationPlacement:
    """Resolve placement for a finite series without changing its requested updates."""
    coordinates = sorted(finite_coordinates(updates))
    if producer_started:
        return ValidationPlacement(location="ordinary", reason="late")
    if not live_products_supported:
        return ValidationPlacement(location="ordinary", reason="unqualified")
    if budget is None:
        return ValidationPlacement(location="ordinary", reason="unmeasured")
    interval = min(right - left for left, right in zip((0, *coordinates), coordinates))
    fraction = budget.durable_point_p95_ms / (budget.steady_update_ms * interval)
    if budget.durable_point_p95_ms > budget.max_point_ms or fraction > budget.max_update_fraction:
        return ValidationPlacement(location="ordinary", reason="over_budget", predicted_update_fraction=fraction)
    return ValidationPlacement(location="live", reason="within_budget", predicted_update_fraction=fraction)
