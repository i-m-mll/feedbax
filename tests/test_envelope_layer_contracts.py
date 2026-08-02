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


EVALUATE_CLI_EVALUATION_TYPE = "feedbax.test.envelope_layer_eval"


@pytest.fixture
def evaluate_cli_state(application_registry_bundle, monkeypatch):
    """Register a toy evaluation recipe and inject it into the analysis CLI."""
    import numpy as np

    from feedbax.analysis.evaluation import EvaluationRecipeResult
    from feedbax.bin import analysis as analysis_cli
    from feedbax.plugins.bootstrap import BootstrapState

    def recipe(run_spec: EvaluationRunSpec, _root, _states_path, _context):
        n_trials = int(run_spec.params["n_trials"])
        return EvaluationRecipeResult(
            states={"value": np.asarray(n_trials, dtype=np.int32)},
            summary_metrics={"n_trials": n_trials},
        )

    def batch_recipe(items, _context):
        return [recipe(item.spec, None, None, None) for item in items]

    application_registry_bundle.evaluation_recipes.register(
        EVALUATE_CLI_EVALUATION_TYPE,
        recipe,
        batch_recipe=batch_recipe,
    )

    async def compose_application(**_kwargs):
        return BootstrapState(application_registry_bundle, ())

    monkeypatch.setattr(analysis_cli, "compose_application", compose_application)
    return application_registry_bundle


def _matrix_payload(*, n_trials: tuple[int, ...]) -> dict:
    return {
        "schema_id": "feedbax.spec.evaluation_run_matrix",
        "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
        "base": {
            "evaluation_type": EVALUATE_CLI_EVALUATION_TYPE,
            "params": {"n_trials": n_trials[0]},
        },
        "rows": [
            {
                "row_id": f"row-{index}",
                "deltas": [{"path": "params.n_trials", "value": value}],
            }
            for index, value in enumerate(n_trials)
        ],
    }


class TestEvaluationExecutionEntrypoint:
    """`feedbax-analysis evaluate` is the additive evaluation execution entrypoint."""

    def test_matrix_document_executes_and_writes_manifests(
        self, tmp_path, capsys, evaluate_cli_state
    ) -> None:
        from feedbax.bin import analysis as analysis_cli
        from feedbax.contracts.manifest import load_manifest

        spec_path = tmp_path / "matrix.json"
        spec_path.write_text(json.dumps(_matrix_payload(n_trials=(2, 3))), encoding="utf-8")

        analysis_cli.main(["evaluate", str(spec_path), "--root", str(tmp_path / "runs")])

        payload = json.loads(capsys.readouterr().out)
        assert [row["row_id"] for row in payload["rows"]] == ["row-0", "row-1"]
        for row in payload["rows"]:
            manifest = load_manifest(row["manifest_path"])
            assert manifest.kind == "EvaluationRunManifest"
            assert manifest.id == row["manifest_id"]
            assert manifest.status == "completed"

    def test_row_subset_executes_only_the_named_rows(
        self, tmp_path, capsys, evaluate_cli_state
    ) -> None:
        from feedbax.bin import analysis as analysis_cli

        spec_path = tmp_path / "matrix.json"
        spec_path.write_text(json.dumps(_matrix_payload(n_trials=(2, 3, 4))), encoding="utf-8")

        analysis_cli.main(
            ["evaluate", str(spec_path), "--root", str(tmp_path / "runs"), "--rows", "row-2,row-0"]
        )

        payload = json.loads(capsys.readouterr().out)
        assert [row["row_id"] for row in payload["rows"]] == ["row-2", "row-0"]

    def test_flat_spec_requires_the_stated_escape_hatch_reason(
        self, tmp_path, evaluate_cli_state
    ) -> None:
        from feedbax.bin import analysis as analysis_cli

        spec_path = tmp_path / "flat.json"
        spec_path.write_text(
            json.dumps(
                {
                    "evaluation_type": EVALUATE_CLI_EVALUATION_TYPE,
                    "params": {"n_trials": 2},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="escape hatch requires a stated non-empty reason"):
            analysis_cli.main(["evaluate", str(spec_path), "--root", str(tmp_path / "runs")])

    def test_flat_spec_executes_with_a_stated_reason(
        self, tmp_path, capsys, evaluate_cli_state
    ) -> None:
        from feedbax.bin import analysis as analysis_cli

        spec_path = tmp_path / "flat.json"
        spec_path.write_text(
            json.dumps(
                {
                    "evaluation_type": EVALUATE_CLI_EVALUATION_TYPE,
                    "params": {"n_trials": 5},
                }
            ),
            encoding="utf-8",
        )
        analysis_cli.main(
            [
                "evaluate",
                str(spec_path),
                "--root",
                str(tmp_path / "runs"),
                "--escape-hatch-reason",
                "single legacy condition",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["escape_hatch_reason"] == "single legacy condition"
        assert [row["row_id"] for row in payload["rows"]] == ["flat"]

    def test_rejected_spec_version_fails_closed_before_execution(
        self, tmp_path, evaluate_cli_state
    ) -> None:
        from feedbax.bin import analysis as analysis_cli

        payload = _matrix_payload(n_trials=(2,))
        payload["schema_version"] = "feedbax.spec.evaluation_run_matrix.v0"
        spec_path = tmp_path / "matrix.json"
        spec_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(UnsupportedSpecVersion):
            analysis_cli.main(["evaluate", str(spec_path), "--root", str(tmp_path / "runs")])

    def test_bindings_require_an_execution_descriptor(self, tmp_path, evaluate_cli_state) -> None:
        from feedbax.bin import analysis as analysis_cli

        spec_path = tmp_path / "matrix.json"
        spec_path.write_text(json.dumps(_matrix_payload(n_trials=(2,))), encoding="utf-8")
        with pytest.raises(ValueError, match="require --execution-descriptor"):
            analysis_cli.main(
                [
                    "evaluate",
                    str(spec_path),
                    "--root",
                    str(tmp_path / "runs"),
                    "--artifact-provider",
                    "name=/tmp/does-not-matter",
                ]
            )

    def test_existing_subcommands_are_unchanged(self) -> None:
        from feedbax.bin import analysis as analysis_cli

        assert analysis_cli.RUN_SUBCOMMAND == "run"
        assert analysis_cli.REPORT_SUBCOMMAND == "report"
        assert analysis_cli.BUNDLE_SUBCOMMAND == "bundle"
        assert analysis_cli.EVALUATE_SUBCOMMAND == "evaluate"


#: Byte-exact shape of the hand-authored downstream sidecar this writer replaces
#: (rlrmp2 ``results/4e08c8d/target2x_target4x_floor.row_custody.v1.json``).
REFERENCE_ROW_CUSTODY_SIDECAR = """{
  "bindings": [
    {
      "binding_key": "k2",
      "parent": {
        "id": "feedbax-analysis-run:3d916232abe4ae8eccf16b676c890e7d",
        "kind": "AnalysisRunManifest",
        "metadata": {
          "manifest_sha256": \
"710b5922eabb11180d83f3abf35b0dc1022d7d15585ab825bfd188cc37e90f0f",
          "ref_schema_id": "feedbax.ref.authenticated_manifest",
          "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
          "size_bytes": 10658924
        },
        "role": "analysis_run"
      },
      "row_id": "target2x-floor"
    },
    {
      "binding_key": "k2",
      "parent": {
        "id": "feedbax-analysis-run:11b2be35e7a3118ef25d6822c992e79a",
        "kind": "AnalysisRunManifest",
        "metadata": {
          "manifest_sha256": \
"7170a4d1db43586d826b2fc94c90a2eff838f2faa20738616bf7e8c8e444cd94",
          "ref_schema_id": "feedbax.ref.authenticated_manifest",
          "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
          "size_bytes": 10658924
        },
        "role": "analysis_run"
      },
      "row_id": "target4x-floor"
    }
  ],
  "index_id": "target2x_target4x_floor",
  "index_sha256": \
"72b578432680b0f151c0eb20f0eb2e2166841ed78af4723bd20f76a7ae4b1aa4",
  "schema_id": "feedbax.spec.row_index_custody_bindings",
  "schema_version": "feedbax.spec.row_index_custody_bindings.v2"
}
"""


def _authenticated_parent(manifest_id: str, digest: str, size: int = 10658924) -> ParentRef:
    return ParentRef(
        kind="AnalysisRunManifest",
        id=manifest_id,
        role="analysis_run",
        metadata={
            "manifest_sha256": digest,
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "size_bytes": size,
        },
    )


def _reference_index():
    from feedbax.contracts.row_index import AuthenticatedRowIndex, RowIndexEntry, derive_row_label

    return AuthenticatedRowIndex(
        index_id="target2x_target4x_floor",
        rows=[
            RowIndexEntry(row_id=row_id, label=derive_row_label(row_id))
            for row_id in ("target2x-floor", "target4x-floor")
        ],
    )


def _reference_declaration() -> dict:
    return {
        "target2x-floor": {
            "k2": _authenticated_parent(
                "feedbax-analysis-run:3d916232abe4ae8eccf16b676c890e7d",
                "710b5922eabb11180d83f3abf35b0dc1022d7d15585ab825bfd188cc37e90f0f",
            )
        },
        "target4x-floor": {
            "k2": _authenticated_parent(
                "feedbax-analysis-run:11b2be35e7a3118ef25d6822c992e79a",
                "7170a4d1db43586d826b2fc94c90a2eff838f2faa20738616bf7e8c8e444cd94",
            )
        },
    }


class TestRowIndexCustodyWriter:
    """Feedbax owns construction, matching, authentication, and atomic writing."""

    def test_reproduces_the_reference_sidecar_bytes(self) -> None:
        from feedbax.contracts.row_index import (
            build_row_index_custody_bindings,
            serialize_row_index_custody_bindings,
        )

        document = build_row_index_custody_bindings(
            _reference_index(),
            _reference_declaration(),
            required_binding_keys=("k2",),
        )
        assert serialize_row_index_custody_bindings(document) == REFERENCE_ROW_CUSTODY_SIDECAR

    def test_write_is_atomic_and_round_trips(self, tmp_path) -> None:
        from feedbax.contracts.row_index import (
            build_row_index_custody_bindings,
            load_row_index_custody_bindings,
            write_row_index_custody_bindings,
        )

        index = _reference_index()
        document = build_row_index_custody_bindings(index, _reference_declaration())
        target = tmp_path / "nested" / "target2x_target4x_floor.row_custody.v1.json"
        written = write_row_index_custody_bindings(document, target, index=index)
        assert written == target
        assert target.read_text(encoding="utf-8") == REFERENCE_ROW_CUSTODY_SIDECAR
        assert list(target.parent.iterdir()) == [target]
        assert load_row_index_custody_bindings(target) == document

    def test_ordering_follows_index_order_then_binding_key(self) -> None:
        from feedbax.contracts.row_index import (
            AuthenticatedRowIndex,
            RowIndexEntry,
            build_row_index_custody_bindings,
            derive_row_label,
        )

        index = AuthenticatedRowIndex(
            index_id="ordered",
            rows=[
                RowIndexEntry(row_id=row_id, label=derive_row_label(row_id))
                for row_id in ("zeta", "alpha")
            ],
        )
        parent = _authenticated_parent("feedbax-analysis-run:one", "0" * 64)
        document = build_row_index_custody_bindings(
            index,
            {
                "alpha": {"b_key": parent, "a_key": parent},
                "zeta": {"a_key": parent},
            },
        )
        assert [
            (binding.row_id, binding.binding_key) for binding in document.bindings
        ] == [("zeta", "a_key"), ("alpha", "a_key"), ("alpha", "b_key")]

    def test_row_absent_from_the_index_fails_closed(self) -> None:
        from feedbax.contracts.row_index import (
            RowSelectionError,
            RowSelectionErrorCode,
            build_row_index_custody_bindings,
        )

        declaration = _reference_declaration()
        declaration["not-a-row"] = declaration["target2x-floor"]
        with pytest.raises(RowSelectionError) as excinfo:
            build_row_index_custody_bindings(_reference_index(), declaration)
        assert excinfo.value.code is RowSelectionErrorCode.UNRESOLVED_ROW_KEY
        assert "not-a-row" in str(excinfo.value)

    def test_missing_required_binding_key_fails_closed(self) -> None:
        from feedbax.contracts.row_index import (
            RowSelectionError,
            RowSelectionErrorCode,
            build_row_index_custody_bindings,
        )

        declaration = _reference_declaration()
        del declaration["target4x-floor"]
        with pytest.raises(RowSelectionError) as excinfo:
            build_row_index_custody_bindings(
                _reference_index(), declaration, required_binding_keys=("k2",)
            )
        assert excinfo.value.code is RowSelectionErrorCode.UNRESOLVED_ROW_KEY
        assert excinfo.value.binding_key == "k2"

    def test_unauthenticated_parent_is_refused(self) -> None:
        from feedbax.contracts.row_index import build_row_index_custody_bindings

        with pytest.raises(ValueError, match="authenticated manifest"):
            build_row_index_custody_bindings(
                _reference_index(),
                {
                    "target2x-floor": {
                        "k2": ParentRef(
                            kind="AnalysisRunManifest",
                            id="feedbax-analysis-run:unauthenticated",
                            role="analysis_run",
                        )
                    }
                },
            )

    def test_partial_authentication_profile_is_refused(self) -> None:
        from feedbax.contracts.row_index import build_row_index_custody_bindings

        partial = _authenticated_parent("feedbax-analysis-run:partial", "1" * 64)
        del partial.metadata["size_bytes"]
        with pytest.raises(ValueError, match="incomplete"):
            build_row_index_custody_bindings(
                _reference_index(), {"target2x-floor": {"k2": partial}}
            )

    def test_partial_custody_is_admissible_without_required_keys(self) -> None:
        from feedbax.contracts.row_index import build_row_index_custody_bindings

        declaration = _reference_declaration()
        del declaration["target4x-floor"]
        document = build_row_index_custody_bindings(_reference_index(), declaration)
        assert [binding.row_id for binding in document.bindings] == ["target2x-floor"]

    def test_loader_rejects_a_foreign_schema_version(self, tmp_path) -> None:
        from feedbax.contracts.row_index import load_row_index_custody_bindings

        path = tmp_path / "custody.json"
        payload = json.loads(REFERENCE_ROW_CUSTODY_SIDECAR)
        payload["schema_version"] = "feedbax.spec.row_index_custody_bindings.v1"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported RowIndexCustodyBindings schema_version"):
            load_row_index_custody_bindings(path)


def _leaf(path: str, *, shape=(2, 3), dtype: str = "float32"):
    from feedbax.contracts.checkpoints import SlotLeafFingerprint

    return SlotLeafFingerprint(path=path, leaf_type="jax.Array", shape=shape, dtype=dtype)


def _slot(name: str, role: str, paths, *, required: bool = True, treedef: str | None = None):
    from feedbax.contracts.checkpoint_initialization import CheckpointSlotStructure

    return CheckpointSlotStructure(
        slot=name,
        role=role,
        required=required,
        treedef=treedef or f"treedef:{name}",
        leaves=[_leaf(path) for path in paths],
    )


def _structure(*slots):
    from feedbax.contracts.checkpoint_initialization import CheckpointStructure

    return CheckpointStructure(slots=list(slots))


def _full_structure():
    return _structure(
        _slot("model", "model", ("net.weight", "net.bias")),
        _slot("optimizer", "optimizer", ("mu", "nu")),
        _slot("prng", "prng", ("key",)),
        _slot("batch_counter", "auxiliary", ("count",)),
        _slot("adaptive_state", "auxiliary", ("scale",), required=False),
    )


def _request(mode: str):
    from feedbax.contracts.checkpoint_initialization import CheckpointInitializationRequest

    return CheckpointInitializationRequest(
        mode=mode,
        source=ParentRef(
            kind="TrainingCheckpointTransactionManifest",
            id="feedbax-checkpoint:source",
            role="checkpoint",
            metadata={
                "manifest_sha256": "a" * 64,
                "ref_schema_id": "feedbax.ref.authenticated_manifest",
                "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
                "size_bytes": 1024,
            },
        ),
    )


class TestCheckpointInitializationLowering:
    """The closed matching rule: exact paths, exact PyTree def, shape, and dtype."""

    def test_exact_match_restores_every_target_path_on_continue(self) -> None:
        from feedbax.contracts.checkpoint_initialization import lower_checkpoint_initialization

        plan = lower_checkpoint_initialization(
            _request("continue_from"),
            source=_full_structure(),
            target=_full_structure(),
        )
        assert plan.fresh_paths == ()
        assert set(plan.restored_paths) == {
            "model/net.weight",
            "model/net.bias",
            "optimizer/mu",
            "optimizer/nu",
            "prng/key",
            "batch_counter/count",
            "adaptive_state/scale",
        }
        assert plan.ignored_source_slots == []

    def test_initialize_from_restores_model_only_and_freshens_the_rest(self) -> None:
        from feedbax.contracts.checkpoint_initialization import lower_checkpoint_initialization

        plan = lower_checkpoint_initialization(
            _request("initialize_from"),
            source=_full_structure(),
            target=_full_structure(),
        )
        assert set(plan.restored_paths) == {"model/net.weight", "model/net.bias"}
        assert set(plan.fresh_paths) == {
            "optimizer/mu",
            "optimizer/nu",
            "prng/key",
            "batch_counter/count",
            "adaptive_state/scale",
        }
        by_slot = {entry.slot: entry for entry in plan.slots}
        assert by_slot["optimizer"].fresh_reason == "mode_scope"
        assert by_slot["prng"].fresh_reason == "mode_scope"

    def test_missing_target_path_fresh_initializes(self) -> None:
        from feedbax.contracts.checkpoint_initialization import lower_checkpoint_initialization

        source = _structure(_slot("model", "model", ("net.weight",)))
        target = _structure(
            _slot("model", "model", ("net.weight",)),
            _slot("adaptive_state", "auxiliary", ("scale",), required=False),
        )
        plan = lower_checkpoint_initialization(
            _request("initialize_from"), source=source, target=target
        )
        assert plan.restored_paths == ("model/net.weight",)
        assert plan.fresh_paths == ("adaptive_state/scale",)
        by_slot = {entry.slot: entry for entry in plan.slots}
        assert by_slot["adaptive_state"].fresh_reason == "absent_from_source"

    def test_source_only_paths_and_slots_are_ignored(self) -> None:
        from feedbax.contracts.checkpoint_initialization import lower_checkpoint_initialization

        source = _structure(
            _slot("model", "model", ("net.weight", "net.bias")),
            _slot("legacy_extra", "auxiliary", ("dropped",)),
        )
        target = _structure(_slot("model", "model", ("net.weight", "net.bias")))
        plan = lower_checkpoint_initialization(
            _request("initialize_from"), source=source, target=target
        )
        assert plan.ignored_source_slots == ["legacy_extra"]
        assert plan.fresh_paths == ()

    def test_source_only_leaf_inside_a_matching_slot_is_ignored(self) -> None:
        from feedbax.contracts.checkpoint_initialization import lower_checkpoint_initialization

        treedef = "treedef:model"
        source = _structure(
            _slot("model", "model", ("net.weight", "net.dropped"), treedef=treedef)
        )
        target = _structure(_slot("model", "model", ("net.weight",), treedef=treedef))
        plan = lower_checkpoint_initialization(
            _request("initialize_from"), source=source, target=target
        )
        assert plan.slots[0].ignored_source_paths == ["net.dropped"]
        assert plan.restored_paths == ("model/net.weight",)

    def test_present_path_shape_mismatch_refuses_and_names_the_paths(self) -> None:
        from feedbax.contracts.checkpoint_initialization import (
            CheckpointInitializationErrorCode,
            CheckpointInitializationRefused,
            CheckpointSlotStructure,
            CheckpointStructure,
            lower_checkpoint_initialization,
        )

        source = _structure(_slot("model", "model", ("net.weight",)))
        target = CheckpointStructure(
            slots=[
                CheckpointSlotStructure(
                    slot="model",
                    role="model",
                    treedef="treedef:model",
                    leaves=[_leaf("net.weight", shape=(4, 3))],
                )
            ]
        )
        with pytest.raises(CheckpointInitializationRefused) as excinfo:
            lower_checkpoint_initialization(
                _request("initialize_from"), source=source, target=target
            )
        assert excinfo.value.code is CheckpointInitializationErrorCode.LEAF_MISMATCH
        assert excinfo.value.paths == ("model/net.weight",)
        assert "shape" in str(excinfo.value)

    def test_present_path_dtype_mismatch_refuses(self) -> None:
        from feedbax.contracts.checkpoint_initialization import (
            CheckpointInitializationErrorCode,
            CheckpointInitializationRefused,
            CheckpointSlotStructure,
            CheckpointStructure,
            lower_checkpoint_initialization,
        )

        source = _structure(_slot("model", "model", ("net.weight",)))
        target = CheckpointStructure(
            slots=[
                CheckpointSlotStructure(
                    slot="model",
                    role="model",
                    treedef="treedef:model",
                    leaves=[_leaf("net.weight", dtype="float64")],
                )
            ]
        )
        with pytest.raises(CheckpointInitializationRefused) as excinfo:
            lower_checkpoint_initialization(
                _request("initialize_from"), source=source, target=target
            )
        assert excinfo.value.code is CheckpointInitializationErrorCode.LEAF_MISMATCH
        assert "dtype" in str(excinfo.value)

    def test_treedef_mismatch_refuses_without_structural_search(self) -> None:
        from feedbax.contracts.checkpoint_initialization import (
            CheckpointInitializationErrorCode,
            CheckpointInitializationRefused,
            lower_checkpoint_initialization,
        )

        source = _structure(_slot("model", "model", ("net.weight",), treedef="treedef:a"))
        target = _structure(_slot("model", "model", ("net.weight",), treedef="treedef:b"))
        with pytest.raises(CheckpointInitializationRefused) as excinfo:
            lower_checkpoint_initialization(
                _request("initialize_from"), source=source, target=target
            )
        assert excinfo.value.code is CheckpointInitializationErrorCode.SLOT_TREEDEF_MISMATCH
        assert excinfo.value.slots == ("model",)
        assert "does not search for structurally similar subtrees" in str(excinfo.value)

    def test_continue_from_refuses_when_required_state_is_absent(self) -> None:
        from feedbax.contracts.checkpoint_initialization import (
            CheckpointInitializationErrorCode,
            CheckpointInitializationRefused,
            lower_checkpoint_initialization,
        )

        source = _structure(_slot("model", "model", ("net.weight",)))
        target = _structure(
            _slot("model", "model", ("net.weight",)),
            _slot("optimizer", "optimizer", ("mu",)),
        )
        with pytest.raises(CheckpointInitializationRefused) as excinfo:
            lower_checkpoint_initialization(
                _request("continue_from"), source=source, target=target
            )
        assert (
            excinfo.value.code
            is CheckpointInitializationErrorCode.MISSING_REQUIRED_CONTINUATION_STATE
        )
        assert excinfo.value.paths == ("optimizer/mu",)
        assert "initialize_from" in str(excinfo.value)

    def test_continue_from_admits_absent_optional_state(self) -> None:
        from feedbax.contracts.checkpoint_initialization import lower_checkpoint_initialization

        source = _structure(_slot("model", "model", ("net.weight",)))
        target = _structure(
            _slot("model", "model", ("net.weight",)),
            _slot("adaptive_state", "auxiliary", ("scale",), required=False),
        )
        plan = lower_checkpoint_initialization(
            _request("continue_from"), source=source, target=target
        )
        assert plan.fresh_paths == ("adaptive_state/scale",)

    def test_initialize_from_does_not_require_optimizer_state(self) -> None:
        from feedbax.contracts.checkpoint_initialization import lower_checkpoint_initialization

        source = _structure(_slot("model", "model", ("net.weight",)))
        target = _structure(
            _slot("model", "model", ("net.weight",)),
            _slot("optimizer", "optimizer", ("mu",)),
        )
        plan = lower_checkpoint_initialization(
            _request("initialize_from"), source=source, target=target
        )
        assert plan.fresh_paths == ("optimizer/mu",)

    def test_unauthenticated_source_is_refused_at_request_construction(self) -> None:
        from feedbax.contracts.checkpoint_initialization import CheckpointInitializationRequest

        with pytest.raises(ValueError, match="authenticated manifest ref"):
            CheckpointInitializationRequest(
                mode="continue_from",
                source=ParentRef(
                    kind="TrainingCheckpointTransactionManifest",
                    id="feedbax-checkpoint:unauthenticated",
                    role="checkpoint",
                ),
            )

    def test_schema_identities_are_registered_and_reject_v0(self) -> None:
        from feedbax.contracts.checkpoint_initialization import (
            CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_ID,
            CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_VERSION,
            CHECKPOINT_INITIALIZATION_SCHEMA_ID,
            CHECKPOINT_INITIALIZATION_SCHEMA_VERSION,
            CHECKPOINT_STRUCTURE_SCHEMA_ID,
            CHECKPOINT_STRUCTURE_SCHEMA_VERSION,
        )

        expected = {
            "CheckpointStructure": (
                CHECKPOINT_STRUCTURE_SCHEMA_ID,
                CHECKPOINT_STRUCTURE_SCHEMA_VERSION,
            ),
            "CheckpointInitializationRequest": (
                CHECKPOINT_INITIALIZATION_SCHEMA_ID,
                CHECKPOINT_INITIALIZATION_SCHEMA_VERSION,
            ),
            "CheckpointInitializationPlan": (
                CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_ID,
                CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_VERSION,
            ),
        }
        for kind, (schema_id, version) in expected.items():
            family = default_spec_registry.resolve(kind)
            assert (family.identity, family.current_version) == (schema_id, version)
            with pytest.raises(UnsupportedSpecVersion):
                default_spec_registry.migrate(kind, {"schema_version": f"{schema_id}.v0"})

    def test_structure_projects_from_a_durable_checkpoint_transaction(self) -> None:
        from feedbax.contracts.checkpoint_initialization import (
            checkpoint_structure_from_manifest,
        )
        from feedbax.contracts.checkpoints import (
            CheckpointSlotBlobRef,
            CheckpointSegmentLineage,
            CheckpointTransactionManifest,
            SlotContentDigest,
            StructuralAbiFingerprint,
        )

        fingerprint = StructuralAbiFingerprint(
            treedef="treedef:model",
            leaf_count=1,
            leaves=[_leaf("net.weight")],
            fingerprint_sha256="b" * 64,
        )
        manifest = CheckpointTransactionManifest(
            transaction_id="txn-1",
            run_id="feedbax-training-run:one",
            barrier="batch",
            completed_coordinate={
                "run_id": "feedbax-training-run:one",
                "phase": "train",
                "completed_barrier": "batch",
                "program_step": 1,
            },
            segment_lineage=CheckpointSegmentLineage(start_batch=0, segment_batch_count=1),
            consistency_predicate={"rules": [], "phase_program_digest": "c" * 64},
            run_contract_binding={
                "training_run_spec_schema_id": "feedbax.spec.training_run",
                "training_run_spec_schema_version": "feedbax.spec.training_run.v4",
                "training_run_spec_sha256": "c" * 64,
                "method_payload_schema_id": "feedbax.spec.worker.toy_payload",
                "method_payload_schema_version": "feedbax.spec.worker.toy_payload.v1",
                "method_payload_sha256": "d" * 64,
                "phase_program_sha256": "c" * 64,
            },
            slots=[
                CheckpointSlotBlobRef(
                    slot="model",
                    role="model",
                    relative_path="slots/model.pkl",
                    sha256="e" * 64,
                    size_bytes=10,
                    coordinate={
                "run_id": "feedbax-training-run:one",
                "phase": "train",
                "completed_barrier": "batch",
                "program_step": 1,
            },
                    structural_abi_fingerprint=fingerprint,
                    content_digest=SlotContentDigest(
                        slot="model",
                        blob_sha256="e" * 64,
                        blob_size_bytes=10,
                        slot_root_sha256="e" * 64,
                    ),
                )
            ],
            content_integrity_digest={"slots": [], "transaction_root_sha256": "f" * 64},
        )
        structure = checkpoint_structure_from_manifest(manifest)
        assert [slot.slot for slot in structure.slots] == ["model"]
        assert structure.slots[0].treedef == "treedef:model"
        assert [leaf.path for leaf in structure.slots[0].leaves] == ["net.weight"]


DRAFT_POLICY_ROW_IDS = ("report-surface", "evaluation-surface", "analysis-authoring")


def _policy_document_rows() -> dict[str, list[str]]:
    """Return backticked columns for each guarantees-table row, keyed by row id."""
    import re
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    text = (root / "docs" / "design" / "downstream_interface_stability.md").read_text(
        encoding="utf-8"
    )
    start = text.index("<!-- policy-guarantees:start -->")
    end = text.index("<!-- policy-guarantees:end -->")
    rows: dict[str, list[str]] = {}
    for line in text[start:end].splitlines():
        match = re.match(r"\| `([^`]+)` \|", line)
        if match is None:
            continue
        rows[match.group(1)] = [column.strip() for column in line.strip().strip("|").split("|")]
    return rows


class TestRatificationDraftRows:
    """The report, evaluation, and analysis rows are drafted, not silently ratified."""

    def test_draft_rows_are_present_in_the_policy_document(self) -> None:
        rows = _policy_document_rows()
        assert set(DRAFT_POLICY_ROW_IDS) <= set(rows)
        for row_id in DRAFT_POLICY_ROW_IDS:
            assert rows[row_id][-1] == "No external case"

    def test_draft_rows_are_marked_pending_owner_ratification(self) -> None:
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[1]
        text = (root / "docs" / "design" / "downstream_interface_stability.md").read_text(
            encoding="utf-8"
        )
        assert "## Pending owner ratification: envelope-layer prerequisite rows" in text
        assert "become ratified only when" in text

    def test_fixture_manifest_declares_the_draft_rows_without_external_cases(self) -> None:
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (
                root
                / "external"
                / "feedbax_conformance_fixture"
                / "src"
                / "feedbax_external_conformance"
                / "policy_manifest.v1.json"
            ).read_text(encoding="utf-8")
        )
        rows = {row["row_id"]: row for row in manifest["guaranteed_rows"]}
        for row_id in DRAFT_POLICY_ROW_IDS:
            assert rows[row_id]["coverage_status"] == "not-external-covered"
            assert rows[row_id]["case_ids"] == []
            assert set(rows[row_id]["schemas"]) == {"current", "migrated", "rejected"}

    def test_report_recipes_stays_outside_the_guaranteed_plugin_inventory(self) -> None:
        from feedbax.plugins import _NON_GUARANTEED_PLUGIN_EXPORTS

        assert "REPORT_RECIPES" in _NON_GUARANTEED_PLUGIN_EXPORTS

    def test_every_drafted_public_name_resolves(self) -> None:
        import importlib
        import re

        rows = _policy_document_rows()
        for row_id in DRAFT_POLICY_ROW_IDS:
            columns = rows[row_id]
            namespaces = re.findall(r"`([^`]+)`", columns[1])
            modules = [importlib.import_module(namespace) for namespace in namespaces]
            names = [
                name
                for name in re.findall(r"`([^`]+)`", columns[2])
                if not name.startswith("feedbax-")
            ]
            assert names
            unresolved = [
                name for name in names if not any(hasattr(module, name) for module in modules)
            ]
            assert unresolved == [], f"{row_id}: {unresolved}"

    def test_policy_consistency_check_accepts_the_draft(self) -> None:
        import importlib.util
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "feedbax_check_downstream_interface_policy",
            root / "scripts" / "check_downstream_interface_policy.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        module.check_policy()


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
