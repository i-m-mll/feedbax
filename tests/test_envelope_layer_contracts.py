"""Focused tests for the envelope-layer prerequisite contracts.

Covers the four prerequisite surfaces the downstream envelope layers depend on:

1. versioned schema identity plus explicit migrate/reject for ``EvaluationRunSpec``,
   ``ReportSpec``, ``ReportManifest``, and the evaluation matrix family;
2. the additive ``feedbax-analysis evaluate`` execution entrypoint;
3. the generic ``RowIndexCustodyBindings`` writer;
4. checkpoint initialize/continue lowering under the closed matching rule.
"""

from __future__ import annotations

import json

import pytest

from feedbax.contracts.manifest import (
    EVALUATION_RUN_SPEC_SCHEMA_ID,
    EVALUATION_RUN_SPEC_SCHEMA_VERSION,
    REPORT_MANIFEST_SCHEMA_VERSION,
    REPORT_SPEC_SCHEMA_ID,
    REPORT_SPEC_SCHEMA_VERSION,
    EvaluationRunSpec,
    ParentRef,
    ReportSpec,
    evaluation_run_manifest_id,
    load_manifest_bytes,
    report_manifest_id,
    spec_payload,
)
from feedbax.contracts.migrations import (
    UnsupportedSpecVersion,
    default_spec_registry,
    migrate_evaluation_run_spec_payload,
    migrate_report_spec_payload,
)

# Golden identities computed on the pre-change tree (feedbax develop a754fcbf),
# before EvaluationRunSpec/ReportSpec carried in-document schema identity.
GOLDEN_EVALUATION_RUN_ID_MINIMAL = "feedbax-evaluation-run:32a6928cc690d9114a67dba68770c29d"
GOLDEN_EVALUATION_RUN_ID_FULL = "feedbax-evaluation-run:05f290a9920a092907464021838c6e1f"
GOLDEN_REPORT_ID_MINIMAL = "feedbax-report:6bffabfeb3e85dee37bc5adbeac90de8"
GOLDEN_REPORT_ID_FULL = "feedbax-report:1c5edccaf1129de4025b2be9c0d33844"


def _training_parent() -> ParentRef:
    return ParentRef(
        kind="TrainingRunManifest",
        id="feedbax-training-run:abc",
        role="training_run",
    )


def _analysis_parent() -> ParentRef:
    return ParentRef(
        kind="AnalysisRunManifest",
        id="feedbax-analysis-run:abc",
        role="analysis_run",
    )


class TestEvaluationRunSpecSchemaIdentity:
    def test_default_identity_is_the_registered_current_version(self) -> None:
        spec = EvaluationRunSpec(evaluation_type="tests.baseline")
        family = default_spec_registry.resolve("EvaluationRunSpec")
        assert spec.schema_id == EVALUATION_RUN_SPEC_SCHEMA_ID == family.identity
        assert spec.schema_version == EVALUATION_RUN_SPEC_SCHEMA_VERSION
        assert spec.schema_version == family.current_version

    def test_unversioned_document_is_accepted_as_the_named_v1_baseline(self) -> None:
        result = migrate_evaluation_run_spec_payload({"evaluation_type": "tests.baseline"})
        assert result.source_version == EVALUATION_RUN_SPEC_SCHEMA_VERSION
        assert result.target_version == EVALUATION_RUN_SPEC_SCHEMA_VERSION
        assert result.migration_records == []
        assert result.payload["schema_id"] == EVALUATION_RUN_SPEC_SCHEMA_ID
        assert result.payload["schema_version"] == EVALUATION_RUN_SPEC_SCHEMA_VERSION

    def test_current_version_document_is_accepted_unchanged(self) -> None:
        payload = {
            "schema_id": EVALUATION_RUN_SPEC_SCHEMA_ID,
            "schema_version": EVALUATION_RUN_SPEC_SCHEMA_VERSION,
            "evaluation_type": "tests.baseline",
        }
        result = migrate_evaluation_run_spec_payload(payload)
        assert result.payload == payload
        assert not result.migrated

    def test_removed_v0_rejects_with_an_actionable_message(self) -> None:
        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            migrate_evaluation_run_spec_payload(
                {
                    "schema_version": f"{EVALUATION_RUN_SPEC_SCHEMA_ID}.v0",
                    "evaluation_type": "tests.baseline",
                }
            )
        message = str(excinfo.value)
        assert "EvaluationRunSpec" in message
        assert f"{EVALUATION_RUN_SPEC_SCHEMA_ID}.v0" in message
        assert EVALUATION_RUN_SPEC_SCHEMA_VERSION in message
        assert "migration_intentionally_absent" in message

    def test_unknown_version_rejects_rather_than_being_inferred(self) -> None:
        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            migrate_evaluation_run_spec_payload(
                {
                    "schema_version": f"{EVALUATION_RUN_SPEC_SCHEMA_ID}.v99",
                    "evaluation_type": "tests.baseline",
                }
            )
        assert f"{EVALUATION_RUN_SPEC_SCHEMA_ID}.v99" in str(excinfo.value)

    def test_foreign_schema_id_rejects(self) -> None:
        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            migrate_evaluation_run_spec_payload(
                {"schema_id": "other.spec.evaluation_run", "evaluation_type": "tests.baseline"}
            )
        assert "other.spec.evaluation_run" in str(excinfo.value)

    def test_coercion_routes_serialized_documents_through_admission(self) -> None:
        from feedbax.analysis.evaluation import coerce_evaluation_run_spec

        spec = coerce_evaluation_run_spec({"evaluation_type": "tests.baseline"})
        assert spec.schema_version == EVALUATION_RUN_SPEC_SCHEMA_VERSION
        with pytest.raises(UnsupportedSpecVersion):
            coerce_evaluation_run_spec(
                {
                    "schema_version": f"{EVALUATION_RUN_SPEC_SCHEMA_ID}.v0",
                    "evaluation_type": "tests.baseline",
                }
            )

    def test_identity_is_byte_stable_against_pre_identity_golden_ids(self) -> None:
        minimal = EvaluationRunSpec(evaluation_type="tests.baseline")
        full = EvaluationRunSpec(
            evaluation_type="tests.baseline",
            training_run_ids=["r1"],
            inputs=[_training_parent()],
            params={"a": 1},
        )
        assert evaluation_run_manifest_id(minimal) == GOLDEN_EVALUATION_RUN_ID_MINIMAL
        assert evaluation_run_manifest_id(full) == GOLDEN_EVALUATION_RUN_ID_FULL


class TestReportSpecSchemaIdentity:
    def test_default_identity_is_the_registered_current_version(self) -> None:
        spec = ReportSpec(report_type="feedbax.ordered_figure_report")
        family = default_spec_registry.resolve("ReportSpec")
        assert spec.schema_id == REPORT_SPEC_SCHEMA_ID == family.identity
        assert spec.schema_version == REPORT_SPEC_SCHEMA_VERSION == family.current_version

    def test_unversioned_document_is_accepted_as_the_named_v1_baseline(self) -> None:
        result = migrate_report_spec_payload({"report_type": "feedbax.ordered_figure_report"})
        assert result.target_version == REPORT_SPEC_SCHEMA_VERSION
        assert result.payload["schema_id"] == REPORT_SPEC_SCHEMA_ID

    def test_removed_v0_rejects_with_an_actionable_message(self) -> None:
        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            migrate_report_spec_payload(
                {
                    "schema_version": f"{REPORT_SPEC_SCHEMA_ID}.v0",
                    "report_type": "feedbax.ordered_figure_report",
                }
            )
        message = str(excinfo.value)
        assert "ReportSpec" in message
        assert f"{REPORT_SPEC_SCHEMA_ID}.v0" in message
        assert "migration_intentionally_absent" in message

    def test_unknown_version_rejects(self) -> None:
        with pytest.raises(UnsupportedSpecVersion):
            migrate_report_spec_payload(
                {
                    "schema_version": f"{REPORT_SPEC_SCHEMA_ID}.v9",
                    "report_type": "feedbax.ordered_figure_report",
                }
            )

    def test_coercion_routes_serialized_documents_through_admission(self, tmp_path) -> None:
        from feedbax.analysis.reports import coerce_report_spec

        path = tmp_path / "report.json"
        path.write_text(
            json.dumps({"report_type": "feedbax.ordered_figure_report"}),
            encoding="utf-8",
        )
        assert coerce_report_spec(path).schema_version == REPORT_SPEC_SCHEMA_VERSION
        rejected = tmp_path / "rejected.json"
        rejected.write_text(
            json.dumps(
                {
                    "schema_version": f"{REPORT_SPEC_SCHEMA_ID}.v0",
                    "report_type": "feedbax.ordered_figure_report",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(UnsupportedSpecVersion):
            coerce_report_spec(rejected)

    def test_identity_is_byte_stable_against_pre_identity_golden_ids(self) -> None:
        minimal = ReportSpec(report_type="feedbax.ordered_figure_report")
        full = ReportSpec(
            report_type="feedbax.ordered_figure_report",
            inputs=[_analysis_parent()],
            params={"sections": []},
            narrative="hi",
        )
        assert report_manifest_id(minimal) == GOLDEN_REPORT_ID_MINIMAL
        assert report_manifest_id(full) == GOLDEN_REPORT_ID_FULL


class TestReportManifestSchemaAdmission:
    def _manifest_payload(self, *, schema_version: str) -> dict:
        payload = spec_payload(
            "ReportSpec",
            ReportSpec(report_type="feedbax.ordered_figure_report").model_dump(mode="json"),
        )
        return {
            "kind": "ReportManifest",
            "schema_version": schema_version,
            "id": "feedbax-report:test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "feedbax_version": "0.2.0",
            "status": "completed",
            "report_spec": payload.model_dump(mode="json", exclude_none=True),
        }

    def test_current_version_manifest_loads(self) -> None:
        raw = json.dumps(self._manifest_payload(schema_version=REPORT_MANIFEST_SCHEMA_VERSION))
        manifest = load_manifest_bytes(raw.encode())
        assert manifest.kind == "ReportManifest"
        assert manifest.schema_version == REPORT_MANIFEST_SCHEMA_VERSION

    def test_unknown_manifest_version_rejects_instead_of_loading(self) -> None:
        raw = json.dumps(self._manifest_payload(schema_version="feedbax.manifest.report.v9"))
        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            load_manifest_bytes(raw.encode())
        assert "feedbax.manifest.report.v9" in str(excinfo.value)

    def test_removed_manifest_v0_rejects(self) -> None:
        raw = json.dumps(self._manifest_payload(schema_version="feedbax.manifest.report.v0"))
        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            load_manifest_bytes(raw.encode())
        assert "migration_intentionally_absent" in str(excinfo.value)


class TestEvaluationMatrixCoexistingVersions:
    """The matrix family already coexists across v1, v2, and v3."""

    def test_v1_migrates_to_current_through_the_registered_chain(self) -> None:
        payload = {
            "schema_id": "feedbax.spec.evaluation_run_matrix",
            "schema_version": "feedbax.spec.evaluation_run_matrix.v1",
            "base": {"evaluation_type": "tests.baseline"},
            "rows": [{"row_id": "a"}],
        }
        result = default_spec_registry.migrate("EvaluationRunMatrixSpec", payload)
        assert result.target_version == "feedbax.spec.evaluation_run_matrix.v3"
        assert [record.migration_id for record in result.migration_records] == [
            "evaluation-run-matrix-v1-to-v2-staged-parents",
            "evaluation-run-matrix-v2-to-v3-combined-authoring",
        ]
        assert result.payload["staged_parents"] == {}

    def test_v2_migrates_to_current(self) -> None:
        payload = {
            "schema_id": "feedbax.spec.evaluation_run_matrix",
            "schema_version": "feedbax.spec.evaluation_run_matrix.v2",
            "base": {"evaluation_type": "tests.baseline"},
            "rows": [{"row_id": "a"}],
            "staged_parents": {},
        }
        result = default_spec_registry.migrate("EvaluationRunMatrixSpec", payload)
        assert result.target_version == "feedbax.spec.evaluation_run_matrix.v3"
        assert len(result.migration_records) == 1

    def test_removed_v0_rejects(self) -> None:
        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            default_spec_registry.migrate(
                "EvaluationRunMatrixSpec",
                {
                    "schema_version": "feedbax.spec.evaluation_run_matrix.v0",
                    "base": {"evaluation_type": "tests.baseline"},
                },
            )
        assert "migration_intentionally_absent" in str(excinfo.value)
