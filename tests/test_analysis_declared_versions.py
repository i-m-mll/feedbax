"""One declared-schema-version resolution path for authored analysis documents.

The three analysis authoring resolvers — analysis runs, evaluation run matrices,
and analysis bundles — previously called the structured spec migration registry
through three different entrypoints with three different argument sets. These
tests pin the shared path's outcomes for every declaration case a durable
document can present: current, supported-old (migrate), explicitly rejected,
unknown, malformed (including an explicit null), and versionless.

A JSON null is a declaration, not an absence. It is pinned separately from the
other malformed values because it is the one malformed value a value-based
presence test cannot see. A missing declaration also fails closed uniformly;
loaders never stamp a versionless payload as current.
"""

from __future__ import annotations

from typing import Any

import pytest

from feedbax.analysis.bundles import (
    ANALYSIS_BUNDLE_SCHEMA_ID,
    ANALYSIS_BUNDLE_SCHEMA_VERSION,
    authored_analysis_bundle_from_payload,
    resolve_analysis_bundle_authoring,
)
from feedbax.analysis.declared_versions import migrate_authored_document
from feedbax.analysis.evaluation import resolve_evaluation_matrix_authoring
from feedbax.analysis.specs import resolve_analysis_run_authoring
from feedbax.contracts.figures import FIGURE_SPEC_SCHEMA_ID, FIGURE_SPEC_SCHEMA_VERSION
from feedbax.contracts.manifest import (
    ANALYSIS_RUN_SPEC_SCHEMA_ID,
    ANALYSIS_RUN_SPEC_SCHEMA_VERSION,
    ANALYSIS_RUN_SPEC_SCHEMA_VERSION_V1,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1,
    EVALUATION_RUN_SPEC_SCHEMA_ID,
    EVALUATION_RUN_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry

ANALYSIS_BUNDLE_SUPPORTED_OLD_VERSION = "feedbax.spec.analysis_bundle.v5"
ANALYSIS_BUNDLE_REJECTED_VERSION = "feedbax.spec.analysis_bundle.v1"
ANALYSIS_RUN_REJECTED_VERSION = f"{ANALYSIS_RUN_SPEC_SCHEMA_ID}.v0"
EVALUATION_MATRIX_REJECTED_VERSION = f"{EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID}.v0"
UNKNOWN_SUFFIX = ".v9999"


def analysis_run_document() -> dict[str, Any]:
    return {
        "schema_id": ANALYSIS_RUN_SPEC_SCHEMA_ID,
        "schema_version": ANALYSIS_RUN_SPEC_SCHEMA_VERSION,
        "analysis_type": "example.analysis",
        "inputs": [],
        "input_requirements": [],
        "params": {},
    }


def evaluation_matrix_document() -> dict[str, Any]:
    return {
        "schema_id": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "base": {
            "schema_id": EVALUATION_RUN_SPEC_SCHEMA_ID,
            "schema_version": EVALUATION_RUN_SPEC_SCHEMA_VERSION,
            "evaluation_type": "example.evaluate",
            "training_run_ids": ["train-a"],
            "params": {"gain": 1.0},
        },
        "rows": [{"row_id": "control", "deltas": [{"path": "params.gain", "value": 2.0}]}],
        "staged_parents": {},
    }


def analysis_bundle_document() -> dict[str, Any]:
    return {
        "schema_id": ANALYSIS_BUNDLE_SCHEMA_ID,
        "schema_version": ANALYSIS_BUNDLE_SCHEMA_VERSION,
        "name": "declared_versions_probe",
        "stages": [
            {
                "name": "figure",
                "kind": "figure",
                "figure": {
                    "schema_id": FIGURE_SPEC_SCHEMA_ID,
                    "schema_version": FIGURE_SPEC_SCHEMA_VERSION,
                    "name": "declared_versions_probe",
                    "assembler": "feedbax.grid_figure",
                },
            }
        ],
    }


def resolve_analysis_run(document: dict[str, Any]):
    return resolve_analysis_run_authoring(document)[0]


def resolve_evaluation_matrix(document: dict[str, Any]):
    return resolve_evaluation_matrix_authoring(document)[0]


def resolve_analysis_bundle(document: dict[str, Any]):
    return resolve_analysis_bundle_authoring(document)[0]


def with_version(document: dict[str, Any], version: Any) -> dict[str, Any]:
    return {**document, "schema_version": version}


def without_version(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "schema_version"}


# (label, document factory, resolver, current, supported_old, rejected)
AUTHORING_FAMILIES = [
    pytest.param(
        "AnalysisRunSpec",
        analysis_run_document,
        resolve_analysis_run,
        ANALYSIS_RUN_SPEC_SCHEMA_VERSION,
        ANALYSIS_RUN_SPEC_SCHEMA_VERSION_V1,
        ANALYSIS_RUN_REJECTED_VERSION,
        id="analysis_run",
    ),
    pytest.param(
        "EvaluationRunMatrixSpec",
        evaluation_matrix_document,
        resolve_evaluation_matrix,
        EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1,
        EVALUATION_MATRIX_REJECTED_VERSION,
        id="evaluation_matrix",
    ),
    pytest.param(
        "AnalysisBundleSpec",
        analysis_bundle_document,
        resolve_analysis_bundle,
        ANALYSIS_BUNDLE_SCHEMA_VERSION,
        ANALYSIS_BUNDLE_SUPPORTED_OLD_VERSION,
        ANALYSIS_BUNDLE_REJECTED_VERSION,
        id="analysis_bundle",
    ),
]


@pytest.mark.parametrize(
    "kind, factory, resolve, current, supported_old, rejected", AUTHORING_FAMILIES
)
def test_current_version_document_is_accepted_unmigrated(
    kind, factory, resolve, current, supported_old, rejected
) -> None:
    """A document at the current version resolves without any migration."""
    document = factory()

    result = migrate_authored_document(kind, document)

    assert result.source_version == current
    assert result.target_version == current
    assert not result.migration_records
    assert result.payload == document
    assert resolve(document).schema_version == current


@pytest.mark.parametrize(
    "kind, factory, resolve, current, supported_old, rejected", AUTHORING_FAMILIES
)
def test_supported_old_version_document_migrates_to_current(
    kind, factory, resolve, current, supported_old, rejected
) -> None:
    """A supported older version migrates through its registered rules."""
    document = with_version(factory(), supported_old)

    result = migrate_authored_document(kind, document)

    assert result.source_version == supported_old
    assert result.target_version == current
    assert result.migration_records
    assert result.migration_records[-1].target_schema_version == current
    assert resolve(document).schema_version == current


@pytest.mark.parametrize(
    "kind, factory, resolve, current, supported_old, rejected", AUTHORING_FAMILIES
)
def test_explicitly_rejected_version_fails_closed_with_actionable_error(
    kind, factory, resolve, current, supported_old, rejected
) -> None:
    """An explicitly unsupported version names the family, both versions, and a reason."""
    document = with_version(factory(), rejected)

    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        resolve(document)

    message = str(excinfo.value)
    assert kind in message
    assert rejected in message
    assert current in message
    assert "migration_intentionally_absent" in message
    assert "reason:" in message


@pytest.mark.parametrize(
    "kind, factory, resolve, current, supported_old, rejected", AUTHORING_FAMILIES
)
def test_unknown_version_fails_closed_without_inferring_a_version(
    kind, factory, resolve, current, supported_old, rejected
) -> None:
    """An unregistered version is never inferred forward or retried as current."""
    unknown = f"{default_spec_registry.resolve(kind).identity}{UNKNOWN_SUFFIX}"
    document = with_version(factory(), unknown)

    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        resolve(document)

    message = str(excinfo.value)
    assert kind in message
    assert unknown in message
    assert current in message


@pytest.mark.parametrize(
    "kind, factory, resolve, current, supported_old, rejected", AUTHORING_FAMILIES
)
def test_malformed_version_declaration_fails_closed(
    kind, factory, resolve, current, supported_old, rejected
) -> None:
    """A declaration that is present but is not a version string is malformed.

    Before unification this fell through to a version-free code path at the
    evaluation matrix site and surfaced as a field type error instead of an
    actionable schema diagnostic.
    """
    for malformed in (3, ["v1"], ""):
        document = with_version(factory(), malformed)

        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            resolve(document)

        message = str(excinfo.value)
        assert "malformed schema_version" in message
        assert kind in message
        assert current in message


@pytest.mark.parametrize(
    "kind, factory, resolve, current, supported_old, rejected", AUTHORING_FAMILIES
)
def test_explicit_null_version_declaration_is_malformed_not_versionless(
    kind, factory, resolve, current, supported_old, rejected
) -> None:
    """A document carrying ``"schema_version": null`` has declared a bad version.

    Reading the declaration with ``document.get("schema_version") is not None``
    collapses an explicit null onto absence. Presence is decided by the key so
    diagnostics distinguish a malformed declaration from a missing declaration.
    """
    document = with_version(factory(), None)

    for raises in (
        lambda: migrate_authored_document(kind, document),
        lambda: resolve(document),
    ):
        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            raises()

        message = str(excinfo.value)
        assert "malformed schema_version" in message
        assert "schema_version=None" in message
        assert "NoneType" in message
        assert kind in message
        assert current in message


@pytest.mark.parametrize(
    "kind, factory, resolve, current, supported_old, rejected", AUTHORING_FAMILIES
)
def test_versionless_document_fails_closed(
    kind, factory, resolve, current, supported_old, rejected
) -> None:
    """Every authored-document loader rejects a missing version declaration."""
    document = without_version(factory())
    assert "schema_version" not in document

    for raises in (
        lambda: migrate_authored_document(kind, document),
        lambda: resolve(document),
    ):
        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            raises()
        message = str(excinfo.value)
        assert "missing schema_version" in message
        assert kind in message


@pytest.mark.parametrize(
    "kind, factory, resolve, current, supported_old, rejected", AUTHORING_FAMILIES
)
def test_declared_version_string_migrates_by_its_own_declaration(
    kind, factory, resolve, current, supported_old, rejected
) -> None:
    """A real version string is read from the declaration."""
    for version, expect_records in ((current, False), (supported_old, True)):
        document = with_version(factory(), version)
        result = migrate_authored_document(kind, document)

        assert result.source_version == version
        assert result.target_version == current
        assert bool(result.migration_records) is expect_records


def test_bundle_payload_loader_shares_the_declared_version_path() -> None:
    """The packaged-resource bundle loader resolves versions the same way."""
    assert (
        authored_analysis_bundle_from_payload(analysis_bundle_document()).schema_version
        == ANALYSIS_BUNDLE_SCHEMA_VERSION
    )
    assert (
        authored_analysis_bundle_from_payload(
            with_version(analysis_bundle_document(), ANALYSIS_BUNDLE_SUPPORTED_OLD_VERSION)
        ).schema_version
        == ANALYSIS_BUNDLE_SCHEMA_VERSION
    )
    with pytest.raises(UnsupportedSpecVersion, match="missing schema_version"):
        authored_analysis_bundle_from_payload(without_version(analysis_bundle_document()))
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent"):
        authored_analysis_bundle_from_payload(
            with_version(analysis_bundle_document(), ANALYSIS_BUNDLE_REJECTED_VERSION)
        )
