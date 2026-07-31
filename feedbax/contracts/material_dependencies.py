"""Material-dependency admission and identity contracts.

Provenance may remain maximal, but admission and identity are computed only
from the exact dependency values declared here.  Numeric dependency values use
the public :mod:`feedbax.contracts.value_identity` record rather than defining
another authored/semantic/realization tier.

Schema/migration table:

* ``feedbax.spec.material_dependencies.v1`` is the first durable declaration.
  Versionless and other-version declarations are rejected.
* ``feedbax.spec.admission_waiver.v1`` is the first durable waiver.  It can
  waive one exact incidental check for one exact manifest and one exact
  artifact digest.  Versionless and other-version waivers are rejected.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

from pydantic import Field, model_validator

from feedbax.contracts.manifest import ArtifactRef, ParentRef, StrictModel
from feedbax.contracts.spec_storage import training_spec_canonical_bytes
from feedbax.contracts.value_identity import ValueIdentityRecord


MATERIAL_DEPENDENCIES_SCHEMA_ID = "feedbax.spec.material_dependencies"
MATERIAL_DEPENDENCIES_SCHEMA_VERSION = f"{MATERIAL_DEPENDENCIES_SCHEMA_ID}.v1"
ADMISSION_WAIVER_SCHEMA_ID = "feedbax.spec.admission_waiver"
ADMISSION_WAIVER_SCHEMA_VERSION = f"{ADMISSION_WAIVER_SCHEMA_ID}.v1"
MATERIAL_DEPENDENCY_IDENTITY_SCHEMA_ID = "feedbax.identity.material_dependencies"
MATERIAL_DEPENDENCY_IDENTITY_SCHEMA_VERSION = (
    f"{MATERIAL_DEPENDENCY_IDENTITY_SCHEMA_ID}.v1"
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"

MaterialDependencyValue: TypeAlias = ParentRef | ArtifactRef | ValueIdentityRecord


class MaterialDependency(StrictModel):
    """One named dependency value and its direct dependency edges."""

    name: str = Field(min_length=1)
    value: MaterialDependencyValue
    depends_on: list[str] = Field(default_factory=list)


class AdmissionWaiver(StrictModel):
    """Narrow authored authorization for one incidental admission check."""

    schema_id: Literal["feedbax.spec.admission_waiver"]
    schema_version: Literal["feedbax.spec.admission_waiver.v1"]
    incidental_check: str = Field(min_length=1)
    manifest: ParentRef
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    reason: str = Field(min_length=1)


class MaterialDependencySet(StrictModel):
    """Versioned dependency closure, identity factoring, and provenance."""

    schema_id: Literal["feedbax.spec.material_dependencies"]
    schema_version: Literal["feedbax.spec.material_dependencies.v1"]
    dependencies: list[MaterialDependency] = Field(min_length=1)
    identity_inputs: list[str] = Field(min_length=1)
    provenance_metadata: dict[str, Any] = Field(default_factory=dict)
    waiver: AdmissionWaiver | None = None

    @model_validator(mode="after")
    def _validate_graph_and_factoring(self) -> "MaterialDependencySet":
        names = [dependency.name for dependency in self.dependencies]
        if len(set(names)) != len(names):
            raise ValueError("material dependency names must be unique")
        name_set = set(names)
        missing_identity = sorted(set(self.identity_inputs) - name_set)
        if missing_identity:
            raise ValueError(
                "identity inputs must be contained in declared material dependencies; "
                f"missing={missing_identity!r}"
            )
        if len(set(self.identity_inputs)) != len(self.identity_inputs):
            raise ValueError("identity_inputs must not contain duplicates")
        graph = {
            dependency.name: tuple(dependency.depends_on)
            for dependency in self.dependencies
        }
        for name, direct in graph.items():
            unknown = sorted(set(direct) - name_set)
            if unknown:
                raise ValueError(
                    f"material dependency {name!r} names unknown dependencies {unknown!r}"
                )
            if len(set(direct)) != len(direct):
                raise ValueError(
                    f"material dependency {name!r} contains duplicate dependency edges"
                )
        _topological_dependency_order(graph)
        if self.waiver is not None:
            if not any(
                isinstance(dependency.value, ParentRef)
                and dependency.value == self.waiver.manifest
                for dependency in self.dependencies
            ):
                raise ValueError(
                    "admission waiver manifest mismatch: the exact manifest is not "
                    "a declared material dependency"
                )
            artifact_digests = {
                digest
                for dependency in self.dependencies
                if dependency.value != self.waiver.manifest
                if (digest := dependency_value_sha256(dependency.value)) is not None
            }
            if self.waiver.artifact_sha256 not in artifact_digests:
                raise ValueError(
                    "admission waiver artifact hash mismatch: the exact artifact is not "
                    "a declared material dependency"
                )
        return self


class MaterialDependencyObservation(StrictModel):
    """Runtime availability and authenticity evidence for one dependency."""

    name: str = Field(min_length=1)
    value: MaterialDependencyValue
    available: bool
    authentic: bool
    diagnostic: str | None = None


class IncidentalAdmissionFailure(StrictModel):
    """One failed check that is outside the declared dependency closure."""

    check: str = Field(min_length=1)
    manifest: ParentRef
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    diagnostic: str = Field(min_length=1)
    material_dependency: str | None = None


class MaterialDependencyAdmission(StrictModel):
    """Successful admission result and dependency-scoped identity."""

    dependency_order: list[str]
    identity_inputs: list[str]
    identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    waived_checks: list[str] = Field(default_factory=list)


def material_dependency_identity_sha256(spec: MaterialDependencySet) -> str:
    """Hash only declared identity inputs, excluding provenance metadata."""
    by_name = {dependency.name: dependency for dependency in spec.dependencies}
    values = [
        {
            "name": name,
            "value": by_name[name].value.model_dump(mode="json", exclude_none=True),
        }
        for name in spec.identity_inputs
    ]
    envelope = {
        "schema_id": MATERIAL_DEPENDENCY_IDENTITY_SCHEMA_ID,
        "schema_version": MATERIAL_DEPENDENCY_IDENTITY_SCHEMA_VERSION,
        "inputs": values,
    }
    return hashlib.sha256(training_spec_canonical_bytes(envelope)).hexdigest()


def validate_material_dependency_admission(
    spec: MaterialDependencySet,
    observations: Sequence[MaterialDependencyObservation],
    *,
    incidental_failures: Sequence[IncidentalAdmissionFailure] = (),
) -> MaterialDependencyAdmission:
    """Validate dependency closure before accepting any incidental waiver."""
    declaration_by_name = {
        dependency.name: dependency for dependency in spec.dependencies
    }
    observation_by_name: dict[str, MaterialDependencyObservation] = {}
    for observation in observations:
        if observation.name in observation_by_name:
            raise ValueError(
                f"material dependency {observation.name!r} has duplicate observations"
            )
        observation_by_name[observation.name] = observation

    order = _topological_dependency_order(
        {
            dependency.name: tuple(dependency.depends_on)
            for dependency in spec.dependencies
        }
    )
    for name in order:
        declaration = declaration_by_name[name]
        observation = observation_by_name.get(name)
        if observation is None or not observation.available:
            detail = observation.diagnostic if observation is not None else "no observation"
            raise ValueError(f"material dependency {name!r} is missing: {detail}")
        if observation.value != declaration.value:
            raise ValueError(
                f"material dependency {name!r} resolved to a different declared value"
            )
        if not observation.authentic:
            detail = observation.diagnostic or "authentication failed"
            raise ValueError(f"material dependency {name!r} is unauthentic: {detail}")

    waived: list[str] = []
    for failure in incidental_failures:
        if failure.material_dependency is not None:
            raise ValueError(
                f"material dependency {failure.material_dependency!r} failed "
                f"{failure.check!r}: {failure.diagnostic}; a waiver cannot admit "
                "a material dependency failure"
            )
        waiver = spec.waiver
        if waiver is None:
            raise ValueError(
                f"incidental admission check {failure.check!r} failed without an authored waiver: "
                f"{failure.diagnostic}"
            )
        if waiver.incidental_check != failure.check:
            raise ValueError(
                "admission waiver incidental_check mismatch: "
                f"expected={failure.check!r}, observed={waiver.incidental_check!r}"
            )
        if waiver.manifest != failure.manifest:
            raise ValueError(
                f"admission waiver manifest mismatch for check {failure.check!r}"
            )
        if waiver.artifact_sha256 != failure.artifact_sha256:
            raise ValueError(
                f"admission waiver artifact hash mismatch for check {failure.check!r}"
            )
        waived.append(failure.check)

    return MaterialDependencyAdmission(
        dependency_order=order,
        identity_inputs=list(spec.identity_inputs),
        identity_sha256=material_dependency_identity_sha256(spec),
        waived_checks=waived,
    )


def dependency_value_sha256(value: MaterialDependencyValue) -> str | None:
    """Return an exact content digest when the dependency value declares one."""
    if isinstance(value, ArtifactRef):
        return value.sha256
    if isinstance(value, ValueIdentityRecord):
        return value.semantic_sha256
    digest = value.metadata.get("manifest_sha256")
    if isinstance(digest, str) and len(digest) == 64:
        return digest
    prefix = "artifact://sha256/"
    if value.uri and value.uri.startswith(prefix):
        return value.uri.removeprefix(prefix)
    return None


def _topological_dependency_order(
    graph: Mapping[str, Sequence[str]],
) -> list[str]:
    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"material dependency graph contains a cycle at {name!r}")
        visiting.add(name)
        for dependency in graph[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in graph:
        visit(name)
    return order


__all__ = [
    "ADMISSION_WAIVER_SCHEMA_ID",
    "ADMISSION_WAIVER_SCHEMA_VERSION",
    "MATERIAL_DEPENDENCIES_SCHEMA_ID",
    "MATERIAL_DEPENDENCIES_SCHEMA_VERSION",
    "AdmissionWaiver",
    "IncidentalAdmissionFailure",
    "MaterialDependency",
    "MaterialDependencyAdmission",
    "MaterialDependencyObservation",
    "MaterialDependencySet",
    "MaterialDependencyValue",
    "dependency_value_sha256",
    "material_dependency_identity_sha256",
    "validate_material_dependency_admission",
]
