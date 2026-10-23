"""Uniform per-kind admission of cached manifests as fulfillment receipts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import plotly.graph_objects as go
import pytest

from feedbax.analysis.figures import RenderedFigure, plan_figure_execution
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.analysis.fulfillment import (
    FULFILLMENT_ADMISSION_SCHEMA_ID,
    FULFILLMENT_ADMISSION_SCHEMA_VERSION,
    AdmissionOutcome,
    admit_analysis_receipt,
    admit_evaluation_receipt,
    admit_figure_receipt,
    admit_report_receipt,
    artifact_bytes_path,
    evaluation_expected_parents,
    migrate_admission_outcome,
    receipt_path,
    resolve_artifact_bytes,
)
from feedbax.contracts.base import (
    ArtifactRef,
    ParentRef,
    Provenance,
)
from feedbax.contracts.artifact_store import store_bytes_artifact
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ReportManifest,
    ReportSpec,
    StagedEvaluationPrerequisite,
    analysis_run_manifest_id,
    evaluation_run_manifest_id,
    report_manifest_id,
    spec_payload,
    write_manifest,
)


AUTHORED_FIGURE = {
    "schema_id": "feedbax.spec.figure",
    "schema_version": "feedbax.spec.figure.v2",
    "name": "admission-figure",
    "assembler": "feedbax.grid_figure",
}


def _evaluation_spec(evaluation_type: str = "testpkg.admission") -> EvaluationRunSpec:
    return EvaluationRunSpec(evaluation_type=evaluation_type, params={"seed": 1})


def _write_evaluation(
    root: Path,
    spec: EvaluationRunSpec,
    *,
    status: str = "completed",
    manifest_id: str | None = None,
    artifacts: list[ArtifactRef] | None = None,
    parents: list[ParentRef] | None = None,
) -> EvaluationRunManifest:
    manifest = EvaluationRunManifest(
        id=manifest_id or evaluation_run_manifest_id(spec),
        status=status,
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            spec.model_dump(mode="json", exclude_none=True),
        ),
        provenance=Provenance(
            parents=parents if parents is not None else evaluation_expected_parents(spec)
        ),
        artifacts=artifacts or [],
    )
    write_manifest(manifest, root=root)
    return manifest


def test_completed_evaluation_receipt_admits(tmp_path: Path) -> None:
    spec = _evaluation_spec()
    _write_evaluation(tmp_path, spec)

    outcome = admit_evaluation_receipt(spec, root=tmp_path)

    assert outcome.admitted
    assert outcome.failures == []
    assert outcome.manifest_present
    assert outcome.manifest_path == str(
        receipt_path("evaluation", evaluation_run_manifest_id(spec), root=tmp_path)
    )
    assert "admitted" in outcome.describe()


def test_absent_receipt_is_not_present_and_not_admitted(tmp_path: Path) -> None:
    outcome = admit_evaluation_receipt(_evaluation_spec(), root=tmp_path)

    assert not outcome.admitted
    assert not outcome.manifest_present
    assert outcome.codes == ("manifest_absent",)


@pytest.mark.parametrize("status", ["failed", "running", "cancelled"])
def test_non_completed_status_refuses(tmp_path: Path, status: str) -> None:
    spec = _evaluation_spec()
    _write_evaluation(tmp_path, spec, status=status)

    outcome = admit_evaluation_receipt(spec, root=tmp_path)

    assert not outcome.admitted
    assert "status_not_completed" in outcome.codes
    assert outcome.manifest_present


def test_identity_mismatch_refuses(tmp_path: Path) -> None:
    spec = _evaluation_spec()
    manifest = _write_evaluation(tmp_path, spec, manifest_id="feedbax-evaluation-run:forged")

    outcome = admit_evaluation_receipt(spec, root=tmp_path, manifest=manifest)

    assert not outcome.admitted
    assert "identity_mismatch" in outcome.codes
    failure = next(f for f in outcome.failures if f.code == "identity_mismatch")
    assert failure.expected == evaluation_run_manifest_id(spec)
    assert failure.observed == "feedbax-evaluation-run:forged"


def test_wrong_kind_refuses(tmp_path: Path) -> None:
    spec = _evaluation_spec()
    manifest_id = evaluation_run_manifest_id(spec)
    decoy = AnalysisRunManifest(
        id=manifest_id,
        status="completed",
        analysis_spec=spec_payload("AnalysisRunSpec", {"analysis_type": "testpkg.decoy"}),
    )
    path = receipt_path("evaluation", manifest_id, root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(decoy.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")

    outcome = admit_evaluation_receipt(spec, root=tmp_path)

    assert not outcome.admitted
    assert "kind_mismatch" in outcome.codes


def test_embedded_spec_mismatch_refuses(tmp_path: Path) -> None:
    spec = _evaluation_spec()
    manifest = EvaluationRunManifest(
        id=evaluation_run_manifest_id(spec),
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            _evaluation_spec("testpkg.other").model_dump(mode="json", exclude_none=True),
        ),
        provenance=Provenance(parents=[]),
    )
    write_manifest(manifest, root=tmp_path)

    outcome = admit_evaluation_receipt(spec, root=tmp_path)

    assert not outcome.admitted
    assert "spec_mismatch" in outcome.codes


def test_embedded_spec_kind_mismatch_refuses(tmp_path: Path) -> None:
    spec = _evaluation_spec()
    manifest = EvaluationRunManifest(
        id=evaluation_run_manifest_id(spec),
        status="completed",
        evaluation_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "testpkg.wrong"},
        ),
        provenance=Provenance(parents=[]),
    )
    write_manifest(manifest, root=tmp_path)

    outcome = admit_evaluation_receipt(spec, root=tmp_path)

    assert not outcome.admitted
    assert "spec_kind_mismatch" in outcome.codes


def test_parent_mismatch_refuses(tmp_path: Path) -> None:
    spec = _evaluation_spec()
    _write_evaluation(
        tmp_path,
        spec,
        parents=[ParentRef(kind="TrainingRunManifest", id="feedbax-training-run:unexpected")],
    )

    outcome = admit_evaluation_receipt(spec, root=tmp_path)

    assert not outcome.admitted
    assert "parents_mismatch" in outcome.codes


def test_staged_prerequisite_parents_are_part_of_the_exact_expectation(tmp_path: Path) -> None:
    parent = ParentRef(kind="EvaluationRunManifest", id="feedbax-evaluation-run:prereq")
    spec = EvaluationRunSpec(
        evaluation_type="testpkg.admission",
        params={
            "staged_prerequisites": {
                "warmup": StagedEvaluationPrerequisite(parent=parent).model_dump(
                    mode="json", exclude_none=True
                )
            }
        },
    )
    assert evaluation_expected_parents(spec) == [parent]
    _write_evaluation(tmp_path, spec)
    assert admit_evaluation_receipt(spec, root=tmp_path).admitted

    _write_evaluation(tmp_path, spec, parents=[])
    assert "parents_mismatch" in admit_evaluation_receipt(spec, root=tmp_path).codes


def test_missing_required_output_role_refuses(tmp_path: Path) -> None:
    spec = _evaluation_spec()
    _write_evaluation(tmp_path, spec)

    outcome = admit_evaluation_receipt(
        spec, root=tmp_path, required_output_roles=("evaluation_states",)
    )

    assert not outcome.admitted
    assert "output_role_absent" in outcome.codes


def test_artifact_bytes_verify_and_corruption_refuses(tmp_path: Path) -> None:
    spec = _evaluation_spec()
    artifact = store_bytes_artifact(
        b"admission payload\n",
        root=tmp_path,
        role="evaluation_states",
        logical_name="states.bin",
    )
    _write_evaluation(tmp_path, spec, artifacts=[artifact])

    admitted = admit_evaluation_receipt(
        spec, root=tmp_path, required_output_roles=("evaluation_states",)
    )
    assert admitted.admitted

    stored = tmp_path / artifact.metadata["relative_path"]
    stored.chmod(0o600)
    stored.write_bytes(b"tampered payload!!\n")

    outcome = admit_evaluation_receipt(spec, root=tmp_path)
    assert not outcome.admitted
    assert "artifact_sha256_mismatch" in outcome.codes


def test_artifact_bytes_absent_refuses(tmp_path: Path) -> None:
    spec = _evaluation_spec()
    artifact = store_bytes_artifact(
        b"deleted payload\n",
        root=tmp_path,
        role="evaluation_states",
        logical_name="states.bin",
    )
    _write_evaluation(tmp_path, spec, artifacts=[artifact])
    (tmp_path / artifact.metadata["relative_path"]).unlink()

    outcome = admit_evaluation_receipt(spec, root=tmp_path)
    assert not outcome.admitted
    assert "artifact_bytes_absent" in outcome.codes


def test_artifact_without_digest_refuses(tmp_path: Path) -> None:
    spec = _evaluation_spec()
    _write_evaluation(
        tmp_path,
        spec,
        artifacts=[ArtifactRef(role="evaluation_states", logical_name="unhashed.bin")],
    )

    outcome = admit_evaluation_receipt(spec, root=tmp_path)
    assert not outcome.admitted
    assert "artifact_digest_absent" in outcome.codes


# --- Artifact byte location containment --------------------------------------


def _located_artifact(relative_path: str, *, data: bytes = b"located payload\n") -> ArtifactRef:
    """An artifact whose bytes are claimed at exactly one recorded location."""
    return ArtifactRef(
        role="evaluation_states",
        logical_name="states.bin",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        metadata={"relative_path": relative_path},
    )


@pytest.mark.parametrize(
    "relative_path",
    ["/etc/passwd", "../outside/states.bin", "artifacts/../../outside/states.bin"],
)
def test_recorded_relative_paths_that_leave_the_root_do_not_resolve(
    tmp_path: Path, relative_path: str
) -> None:
    """Absolute and upward-walking recorded paths resolve to nothing.

    This is the lexical half of containment and it holds by inspection of the
    recorded string, before any filesystem access.
    """
    artifact = _located_artifact(relative_path)

    resolution = resolve_artifact_bytes(artifact, root=tmp_path)

    assert artifact_bytes_path(artifact, root=tmp_path) is None
    assert resolution.reason == "relative_path_escapes_root"


def test_a_recorded_path_through_a_symlink_out_of_the_root_does_not_resolve(
    tmp_path: Path,
) -> None:
    """A lexically contained path whose bytes are physically outside is refused.

    Every component of ``artifacts/escape/states.bin`` is an ordinary name, so
    lexical containment accepts it; the bytes it reaches live outside custody.
    """
    root = tmp_path / "receipts"
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    payload = b"out-of-custody payload\n"
    (outside / "states.bin").write_bytes(payload)
    (root / "artifacts").mkdir(parents=True)
    (root / "artifacts" / "escape").symlink_to(outside, target_is_directory=True)
    artifact = _located_artifact("artifacts/escape/states.bin", data=payload)

    resolution = resolve_artifact_bytes(artifact, root=root)

    assert resolution.path is None, "out-of-custody bytes never stand in for a receipt's own"
    assert resolution.reason == "relative_path_traverses_symlink"

    spec = _evaluation_spec()
    _write_evaluation(root, spec, artifacts=[artifact])
    outcome = admit_evaluation_receipt(spec, root=root)
    assert not outcome.admitted
    assert "artifact_bytes_unresolvable" in outcome.codes
    unresolvable = next(f for f in outcome.failures if f.code == "artifact_bytes_unresolvable")
    assert unresolvable.details["reason"] == "relative_path_traverses_symlink"


def test_an_ambiguous_digest_prefix_refuses_instead_of_picking_by_sort_order(
    tmp_path: Path,
) -> None:
    """Two stored files sharing one digest prefix are two candidates, not a winner."""
    stored = store_bytes_artifact(
        b"ambiguous payload\n",
        root=tmp_path,
        role="evaluation_states",
        logical_name="states.bin",
    )
    digest = stored.sha256
    assert digest is not None
    shard = tmp_path / "artifacts" / "sha256" / digest[:2]
    (shard / f"{digest}.aliased").write_bytes(b"a different byte stream entirely\n")
    # The digest-prefix fallback only runs when no location was recorded.
    without_location = stored.model_copy(update={"metadata": {}})

    resolution = resolve_artifact_bytes(without_location, root=tmp_path)

    assert resolution.path is None
    assert resolution.reason == "digest_ambiguous"
    assert resolution.candidates == (digest, f"{digest}.aliased")
    assert artifact_bytes_path(without_location, root=tmp_path) is None

    spec = _evaluation_spec()
    _write_evaluation(tmp_path, spec, artifacts=[without_location])
    outcome = admit_evaluation_receipt(spec, root=tmp_path)
    assert not outcome.admitted
    unresolvable = next(f for f in outcome.failures if f.code == "artifact_bytes_unresolvable")
    assert unresolvable.details["reason"] == "digest_ambiguous"
    assert unresolvable.details["candidates"] == [digest, f"{digest}.aliased"]


def test_an_unambiguous_digest_prefix_still_resolves(tmp_path: Path) -> None:
    """The fallback is unchanged when exactly one stored file carries the digest."""
    stored = store_bytes_artifact(
        b"unambiguous payload\n",
        root=tmp_path,
        role="evaluation_states",
        logical_name="states.bin",
    )
    without_location = stored.model_copy(update={"metadata": {}})

    resolution = resolve_artifact_bytes(without_location, root=tmp_path)

    assert resolution.reason is None
    assert resolution.path == (
        tmp_path / "artifacts" / "sha256" / stored.sha256[:2] / stored.sha256
    )

    spec = _evaluation_spec()
    _write_evaluation(tmp_path, spec, artifacts=[without_location])
    assert admit_evaluation_receipt(spec, root=tmp_path).admitted


def test_analysis_receipt_admits_and_refuses_on_spec_drift(tmp_path: Path) -> None:
    spec = AnalysisRunSpec(analysis_type="testpkg.admission", params={"k": 1})
    manifest = AnalysisRunManifest(
        id=analysis_run_manifest_id(spec),
        status="completed",
        analysis_spec=spec_payload("AnalysisRunSpec", spec.model_dump(mode="json")),
        inputs=list(spec.inputs),
        provenance=Provenance(parents=list(spec.inputs)),
    )
    write_manifest(manifest, root=tmp_path)

    assert admit_analysis_receipt(spec, root=tmp_path).admitted

    drifted = AnalysisRunSpec(analysis_type="testpkg.admission", params={"k": 2})
    outcome = admit_analysis_receipt(
        drifted,
        root=tmp_path,
        manifest=manifest,
        path=receipt_path("analysis", manifest.id, root=tmp_path),
    )
    assert not outcome.admitted
    assert "identity_mismatch" in outcome.codes
    assert "spec_mismatch" in outcome.codes


def test_report_receipt_admits_and_refuses_on_parent_drift(tmp_path: Path) -> None:
    parent = ParentRef(kind="AnalysisRunManifest", id="feedbax-analysis-run:input")
    spec = ReportSpec(report_type="testpkg.admission", inputs=[parent])
    manifest = ReportManifest(
        id=report_manifest_id(spec),
        status="completed",
        report_spec=spec_payload("ReportSpec", spec.model_dump(mode="json", exclude_none=True)),
        inputs=list(spec.inputs),
        provenance=Provenance(parents=[parent]),
    )
    write_manifest(manifest, root=tmp_path)

    assert admit_report_receipt(spec, root=tmp_path).admitted

    outcome = admit_report_receipt(
        spec,
        root=tmp_path,
        expected_parents=[
            ParentRef(kind="AnalysisRunManifest", id="feedbax-analysis-run:rebound")
        ],
    )
    assert not outcome.admitted
    assert "parents_mismatch" in outcome.codes


def _authenticated_analysis_parent(root: Path, name: str) -> ParentRef:
    manifest = AnalysisRunManifest(
        id=f"feedbax-analysis-run:{name}",
        status="completed",
        analysis_spec=spec_payload("AnalysisRunSpec", {"analysis_type": "testpkg.figure_input"}),
    )
    path = write_manifest(manifest, root=root)
    return authenticated_manifest_ref(manifest, path, "analysis_run")


def _figure_plan(application_registry_bundle, *, runtime_inputs):
    return plan_figure_execution(
        AUTHORED_FIGURE,
        runtime_inputs=runtime_inputs,
        runtime_metadata={"node": "admission"},
        registry=application_registry_bundle.figures,
    )


def test_figure_admission_is_not_id_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    application_registry_bundle,
) -> None:
    """A matching figure id must not admit a differently-bound render."""
    from feedbax.analysis import figures as figures_module

    monkeypatch.setattr(
        figures_module,
        "_build_figures",
        lambda *_args: [RenderedFigure(name="only", figure=go.Figure())],
    )
    bound = _authenticated_analysis_parent(tmp_path, "bound")
    other = _authenticated_analysis_parent(tmp_path, "other")

    executed_plan = _figure_plan(application_registry_bundle, runtime_inputs=[bound])
    manifest, path = figures_module.execute_figure_spec(
        executed_plan.resolution,
        runtime_inputs=[bound],
        runtime_metadata={"node": "admission"},
        root=tmp_path,
        registry=application_registry_bundle.figures,
    )

    same = admit_figure_receipt(executed_plan, root=tmp_path)
    assert same.admitted

    rebound_plan = _figure_plan(application_registry_bundle, runtime_inputs=[other])
    # Figure identity is minted from the resolved authored spec, so the runtime
    # rebinding does not change the manifest id at all.
    assert rebound_plan.manifest_id == manifest.id
    assert rebound_plan.manifest_id == executed_plan.manifest_id

    outcome = admit_figure_receipt(rebound_plan, root=tmp_path, path=path)
    assert not outcome.admitted
    assert "runtime_inputs_mismatch" in outcome.codes
    assert "runtime_binding_mismatch" in outcome.codes
    assert "parents_mismatch" in outcome.codes


def test_figure_runtime_binding_absence_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    application_registry_bundle,
) -> None:
    from feedbax.analysis import figures as figures_module

    monkeypatch.setattr(
        figures_module,
        "_build_figures",
        lambda *_args: [RenderedFigure(name="only", figure=go.Figure())],
    )
    plain_plan = plan_figure_execution(
        AUTHORED_FIGURE, registry=application_registry_bundle.figures
    )
    assert plain_plan.runtime_binding_payload is None
    _manifest, path = figures_module.execute_figure_spec(
        plain_plan.resolution,
        root=tmp_path,
        registry=application_registry_bundle.figures,
    )
    assert admit_figure_receipt(plain_plan, root=tmp_path).admitted

    overlaid = _figure_plan(application_registry_bundle, runtime_inputs=[])
    outcome = admit_figure_receipt(overlaid, root=tmp_path, path=path)
    assert not outcome.admitted
    assert "runtime_binding_absent" in outcome.codes


def test_admission_outcome_schema_identity_migrates_or_rejects() -> None:
    outcome = AdmissionOutcome(
        admitted=True,
        node_kind="report",
        manifest_id="feedbax-report:x",
        manifest_kind="ReportManifest",
        manifest_present=True,
    )
    payload = outcome.model_dump(mode="json")
    assert payload["schema_id"] == FULFILLMENT_ADMISSION_SCHEMA_ID
    assert payload["schema_version"] == FULFILLMENT_ADMISSION_SCHEMA_VERSION
    assert migrate_admission_outcome(payload) == outcome
    assert migrate_admission_outcome(outcome) == outcome

    with pytest.raises(ValueError, match="unsupported AdmissionOutcome schema_version"):
        migrate_admission_outcome({**payload, "schema_version": "feedbax.fulfillment.x.v0"})
    with pytest.raises(ValueError, match="unsupported AdmissionOutcome schema_id"):
        migrate_admission_outcome({**payload, "schema_id": "other.family"})


def test_admission_outcome_refuses_inconsistent_decisions() -> None:
    with pytest.raises(ValueError, match="must name at least one admission failure"):
        AdmissionOutcome(
            admitted=False,
            node_kind="report",
            manifest_id="feedbax-report:x",
            manifest_kind="ReportManifest",
            manifest_present=True,
        )
