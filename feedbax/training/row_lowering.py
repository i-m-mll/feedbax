"""Public content-pinned registry for downstream-authored training rows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from pydantic import ValidationError

from feedbax.contracts.run_matrix import (
    AuthoredTrainingRow,
    RowLowererIdentity,
    TRAINING_ROW_LOWERER_REF_FIELD,
    TrainingRowLowererRef,
    TrainingRowLoweringResult,
)


class TrainingRowLowererRegistryError(ValueError):
    """Raised when authored-row lowerer authority cannot be resolved exactly."""


@dataclass(frozen=True)
class TrainingRowLowererRegistration:
    """One exact authored-schema to execution-payload lowering registration."""

    authored_schema_id: str
    authored_schema_version: str
    lowerer_id: str
    lowerer_version: str
    implementation_sha256: str
    lower: Callable[[AuthoredTrainingRow], TrainingRowLoweringResult | Mapping[str, Any]]
    owner: str = "feedbax"

    def validate_structure(self) -> None:
        """Validate stable identities and the callable boundary."""
        identities = (
            self.authored_schema_id,
            self.authored_schema_version,
            self.lowerer_id,
            self.lowerer_version,
            self.owner,
        )
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise ValueError("training row lowerer identities and owner must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.implementation_sha256):
            raise ValueError("training row lowerer implementation_sha256 must be a sha256")
        if not callable(self.lower):
            raise TypeError("training row lowerer must be callable")


class TrainingRowLowererRegistry:
    """Registry resolving lowerers from exact authored and authority identities."""

    def __init__(self) -> None:
        self._registrations: dict[
            tuple[str, str, str, str], TrainingRowLowererRegistration
        ] = {}

    def register(self, registration: TrainingRowLowererRegistration) -> None:
        """Register one exact lowering authority, rejecting ambiguity."""
        if not isinstance(registration, TrainingRowLowererRegistration):
            raise TypeError("registration must be a TrainingRowLowererRegistration")
        registration.validate_structure()
        key = self._key(registration)
        existing = self._registrations.get(key)
        if existing is not None:
            raise TrainingRowLowererRegistryError(
                "ambiguous training row lowerer registration for "
                f"{key!r}: owners {existing.owner!r} and {registration.owner!r}"
            )
        self._registrations[key] = registration

    def available_keys(self) -> tuple[tuple[str, str, str, str], ...]:
        """Return exact registrations in deterministic order."""
        return tuple(sorted(self._registrations))

    def lower(
        self, authored_row: AuthoredTrainingRow
    ) -> TrainingRowLoweringResult | None:
        """Lower a declared authored row; pass explicit TrainingRunSpec rows through."""
        raw_ref = authored_row.payload.get(TRAINING_ROW_LOWERER_REF_FIELD)
        if raw_ref is None:
            return None
        if not isinstance(raw_ref, Mapping):
            raise TrainingRowLowererRegistryError(
                f"{TRAINING_ROW_LOWERER_REF_FIELD!r} must be an object"
            )
        try:
            authority = TrainingRowLowererRef.model_validate(raw_ref)
        except ValidationError as exc:
            raise TrainingRowLowererRegistryError(
                f"invalid {TRAINING_ROW_LOWERER_REF_FIELD!r} authority: {exc}"
            ) from exc
        authored_schema_id = authored_row.payload.get("schema_id")
        authored_schema_version = authored_row.payload.get("schema_version")
        if not isinstance(authored_schema_id, str) or not isinstance(
            authored_schema_version, str
        ):
            raise TrainingRowLowererRegistryError(
                "authored rows declaring a lowerer require schema_id and schema_version"
            )
        key = (
            authored_schema_id,
            authored_schema_version,
            authority.lowerer_id,
            authority.lowerer_version,
        )
        registration = self._registrations.get(key)
        if registration is None:
            known = ", ".join(repr(item) for item in self.available_keys()) or "<none>"
            raise TrainingRowLowererRegistryError(
                f"no exact training row lowerer registered for {key!r}; known: {known}"
            )
        observed_sha256 = training_row_lowerer_implementation_sha256(registration.lower)
        if authority.implementation_sha256 != registration.implementation_sha256 or (
            observed_sha256 != registration.implementation_sha256
        ):
            raise TrainingRowLowererRegistryError(
                f"training row lowerer {authority.lowerer_id!r} implementation drifted"
            )
        try:
            raw_result = registration.lower(authored_row.model_copy(deep=True))
            result = (
                raw_result
                if isinstance(raw_result, TrainingRowLoweringResult)
                else TrainingRowLoweringResult.model_validate(raw_result)
            )
        except Exception as exc:
            raise TrainingRowLowererRegistryError(
                f"training row lowerer {authority.lowerer_id!r} owned by "
                f"{registration.owner!r} failed: {exc}"
            ) from exc
        expected_identity = RowLowererIdentity(
            lowerer_id=registration.lowerer_id,
            lowerer_version=registration.lowerer_version,
        )
        if result.lowerer_identities != [expected_identity]:
            raise TrainingRowLowererRegistryError(
                f"training row lowerer {authority.lowerer_id!r} returned drifted identity; "
                f"expected {[expected_identity]!r}, observed {result.lowerer_identities!r}"
            )
        return result

    @staticmethod
    def _key(registration: TrainingRowLowererRegistration) -> tuple[str, str, str, str]:
        return (
            registration.authored_schema_id,
            registration.authored_schema_version,
            registration.lowerer_id,
            registration.lowerer_version,
        )


DEFAULT_TRAINING_ROW_LOWERER_REGISTRY = TrainingRowLowererRegistry()


def training_row_lowerer_implementation_sha256(lower: Callable[..., Any]) -> str:
    """Hash the source-module bytes that define an executable lowerer."""
    source_path = inspect.getsourcefile(lower)
    if source_path is None:
        raise TrainingRowLowererRegistryError("training row lowerer has no source module")
    return hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
