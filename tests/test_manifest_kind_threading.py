"""Callers that know the manifest kind they need state it to the lookup.

``find_manifest_by_id`` addresses a record by identifier. An identifier alone
does not decide what a record *is*, so a same-identifier manifest of another
kind is a corrupted or forged answer rather than a lower-priority one. The
optional ``expected_kind`` is how a caller that already knows the kind holds the
lookup to it, and these tests state that the three remaining callers do.

Each case is the accidental-corruption shape the gate exists for: a record of
the wrong kind sitting under a non-canonical location — a converter's output
directory, an import staging area — where the canonical-directory check carries
no claim and the identifier is the only thing addressing it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.analysis.bundles import _execution_parent_ref_for_manifest
from feedbax.analysis.reports import resolve_report_inputs
from feedbax.analysis.specs import ManifestKindMismatch
from feedbax.contracts.base import (
    ParentRef,
    Provenance,
)
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ReportSpec,
    TrainingRunManifest,
    spec_payload,
)

pytestmark = [pytest.mark.feedbax_contract]


SHARED_ID = "feedbax-shared:kind-threading"


def _write_misfiled(root: Path, manifest, *, name: str = "imported.json") -> Path:
    """Write ``manifest`` where no canonical directory claims a kind for it."""
    path = root / "manifests" / "imported" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        manifest.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
    )
    return path


def _analysis_manifest(manifest_id: str) -> AnalysisRunManifest:
    spec = AnalysisRunSpec(analysis_type="feedbax.test.kind_threading", inputs=[], params={})
    return AnalysisRunManifest(
        id=manifest_id,
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec", spec.model_dump(mode="json", exclude_none=True)
        ),
        provenance=Provenance(parents=[]),
    )


def _evaluation_manifest(manifest_id: str) -> EvaluationRunManifest:
    spec = EvaluationRunSpec(evaluation_type="feedbax.test.kind_threading", inputs=[])
    return EvaluationRunManifest(
        id=manifest_id,
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec", spec.model_dump(mode="json", exclude_none=True)
        ),
        provenance=Provenance(parents=[]),
    )


# -- bundles: ambient exact evaluation authority ----------------------------


def test_bundle_execution_parent_refuses_a_same_id_record_of_another_kind(
    tmp_path: Path,
) -> None:
    """A bundle asking for its evaluation root must not be handed a training run."""
    _write_misfiled(tmp_path, TrainingRunManifest(id=SHARED_ID, status="completed"))
    requested = _evaluation_manifest(SHARED_ID)

    with pytest.raises(ManifestKindMismatch) as excinfo:
        _execution_parent_ref_for_manifest(requested, root=tmp_path)

    assert SHARED_ID in str(excinfo.value)
    assert "EvaluationRunManifest" in str(excinfo.value)


def test_bundle_execution_parent_still_binds_the_right_kind(tmp_path: Path) -> None:
    """Control: the record of the required kind resolves and is bound."""
    manifest = _evaluation_manifest(SHARED_ID)
    _write_misfiled(tmp_path, manifest)

    ref = _execution_parent_ref_for_manifest(manifest, root=tmp_path)

    assert ref.kind == "EvaluationRunManifest"
    assert ref.id == SHARED_ID


# -- reports: identifier-only report inputs ---------------------------------


def _report_spec(ref: ParentRef) -> ReportSpec:
    return ReportSpec(report_type="feedbax.test.kind_threading", inputs=[ref])


def test_report_input_resolution_refuses_a_same_id_record_of_another_kind(
    tmp_path: Path,
) -> None:
    """A generic recipe asking for an analysis parent must not read a training run."""
    _write_misfiled(tmp_path, TrainingRunManifest(id=SHARED_ID, status="completed"))
    ref = ParentRef(kind="AnalysisRunManifest", id=SHARED_ID, role="analysis_run")

    with pytest.raises(ManifestKindMismatch) as excinfo:
        resolve_report_inputs(_report_spec(ref), root=tmp_path)

    assert SHARED_ID in str(excinfo.value)
    assert "AnalysisRunManifest" in str(excinfo.value)


def test_report_input_resolution_still_resolves_the_right_kind(tmp_path: Path) -> None:
    """Control: the analysis record of the declared kind resolves normally."""
    _write_misfiled(tmp_path, _analysis_manifest(SHARED_ID))
    ref = ParentRef(kind="AnalysisRunManifest", id=SHARED_ID, role="analysis_run")

    resolved = resolve_report_inputs(_report_spec(ref), root=tmp_path)

    assert [item.ref.id for item in resolved] == [SHARED_ID]
