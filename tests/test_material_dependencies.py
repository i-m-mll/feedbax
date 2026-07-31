from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from feedbax.analysis.exact_parents import (
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION_V1,
    migrate_staged_exact_parents,
)
from feedbax.contracts import (
    AdmissionWaiver,
    ADMISSION_WAIVER_SCHEMA_ID,
    ADMISSION_WAIVER_SCHEMA_VERSION,
    IncidentalAdmissionFailure,
    MaterialDependency,
    MaterialDependencyObservation,
    MaterialDependencySet,
    MATERIAL_DEPENDENCIES_SCHEMA_ID,
    MATERIAL_DEPENDENCIES_SCHEMA_VERSION,
    TRAINING_RUN_CERTIFICATION_SCHEMA_ID,
    TRAINING_RUN_CERTIFICATION_SCHEMA_VERSION,
    TrainingRunCertification,
    material_dependency_identity_sha256,
    training_run_certification,
    validate_material_dependency_admission,
)
from feedbax.contracts.manifest import (
    EvaluationRunSpec,
    ParentRef,
    TrainingRunManifest,
    evaluation_run_manifest_id,
)
from feedbax.testing import check_material_dependency_contract


pytestmark = [pytest.mark.feedbax_contract]


def _parent(digest: str = "a" * 64, *, provenance: str = "one") -> ParentRef:
    return ParentRef(
        kind="TrainingRunManifest",
        id="feedbax-training-run:material",
        role="training_run",
        uri=f"artifact://sha256/{digest}",
        metadata={
            "manifest_sha256": digest,
            "size_bytes": 100,
            "provenance_note": provenance,
        },
    )


def _checkpoint(digest: str = "b" * 64) -> ParentRef:
    return ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id="checkpoint:certified",
        role="training_checkpoint_custody",
        uri="transactions/certified.json",
        metadata={"manifest_sha256": digest},
    )


def _spec(
    *,
    parent: ParentRef | None = None,
    checkpoint: ParentRef | None = None,
    waiver: AdmissionWaiver | None = None,
) -> MaterialDependencySet:
    parent = parent or _parent()
    checkpoint = checkpoint or _checkpoint()
    return MaterialDependencySet(
        schema_id=MATERIAL_DEPENDENCIES_SCHEMA_ID,
        schema_version=MATERIAL_DEPENDENCIES_SCHEMA_VERSION,
        dependencies=[
            MaterialDependency(name="manifest_authority", value=parent),
            MaterialDependency(
                name="certified_checkpoint",
                value=checkpoint,
                depends_on=["manifest_authority"],
            ),
        ],
        identity_inputs=["certified_checkpoint"],
        provenance_metadata={"sampling_contract": "incidental"},
        waiver=waiver,
    )


def _observations(spec: MaterialDependencySet) -> list[MaterialDependencyObservation]:
    return [
        MaterialDependencyObservation(
            name=dependency.name,
            value=dependency.value,
            available=True,
            authentic=True,
        )
        for dependency in spec.dependencies
    ]


def test_identity_uses_only_declared_material_dependencies() -> None:
    first = _spec(parent=_parent(provenance="one"))
    second = _spec(parent=_parent(provenance="two")).model_copy(
        update={"provenance_metadata": {"sampling_contract": "changed"}}
    )

    assert material_dependency_identity_sha256(first) == (
        material_dependency_identity_sha256(second)
    )


def test_factoring_rejects_identity_input_outside_dependencies() -> None:
    with pytest.raises(ValidationError, match="identity inputs must be contained"):
        MaterialDependencySet(
            schema_id=MATERIAL_DEPENDENCIES_SCHEMA_ID,
            schema_version=MATERIAL_DEPENDENCIES_SCHEMA_VERSION,
            dependencies=[MaterialDependency(name="checkpoint", value=_checkpoint())],
            identity_inputs=["sampling_contract"],
        )


def test_evaluation_identity_revalidates_retained_dependency_declaration() -> None:
    spec = _spec()
    parent = _parent().model_copy(
        update={
            "metadata": {
                **_parent().metadata,
                "material_dependency_identity_sha256": material_dependency_identity_sha256(
                    spec
                ),
            }
        }
    )
    run_spec = EvaluationRunSpec(evaluation_type="test.material", inputs=[parent])
    with pytest.raises(ValueError, match="retain.*material_dependencies"):
        evaluation_run_manifest_id(run_spec)

    parent = parent.model_copy(
        update={
            "metadata": {
                **parent.metadata,
                "material_dependencies": spec.model_dump(mode="json"),
                "material_dependency_identity_sha256": "c" * 64,
            }
        }
    )
    with pytest.raises(ValueError, match="disagrees with its validated declaration"):
        evaluation_run_manifest_id(
            EvaluationRunSpec(evaluation_type="test.material", inputs=[parent])
        )


@pytest.mark.parametrize(("available", "authentic", "word"), [(False, True, "missing"), (True, False, "unauthentic")])
def test_material_dependency_failure_names_exact_dependency(
    available: bool,
    authentic: bool,
    word: str,
) -> None:
    spec = _spec()
    observations = _observations(spec)
    observations[1] = observations[1].model_copy(
        update={"available": available, "authentic": authentic}
    )

    with pytest.raises(ValueError, match=rf"certified_checkpoint.*{word}"):
        validate_material_dependency_admission(spec, observations)


def test_waiver_is_exact_and_cannot_waive_material_failure() -> None:
    parent = _parent()
    waiver = AdmissionWaiver(
        schema_id=ADMISSION_WAIVER_SCHEMA_ID,
        schema_version=ADMISSION_WAIVER_SCHEMA_VERSION,
        incidental_check="manifest_status_completed",
        manifest=parent,
        artifact_sha256="b" * 64,
        reason="execution diverged after the checkpoint was certified",
    )
    spec = _spec(parent=parent, waiver=waiver)
    failure = IncidentalAdmissionFailure(
        check="manifest_status_completed",
        manifest=parent,
        artifact_sha256="b" * 64,
        diagnostic="status='failed'",
    )
    admitted = validate_material_dependency_admission(
        spec,
        _observations(spec),
        incidental_failures=[failure],
    )
    assert admitted.waived_checks == ["manifest_status_completed"]

    for changed in (
        failure.model_copy(update={"manifest": _parent("c" * 64)}),
        failure.model_copy(update={"artifact_sha256": "c" * 64}),
    ):
        with pytest.raises(ValueError, match="waiver .* mismatch"):
            validate_material_dependency_admission(
                spec,
                _observations(spec),
                incidental_failures=[changed],
            )

    with pytest.raises(ValueError, match="cannot admit a material dependency failure"):
        validate_material_dependency_admission(
            spec,
            _observations(spec),
            incidental_failures=[
                failure.model_copy(update={"material_dependency": "certified_checkpoint"})
            ],
        )


def test_terminal_certification_separates_divergence_and_certified_prefix() -> None:
    checkpoint = _checkpoint()
    manifest = TrainingRunManifest(
        id="feedbax-training-run:diverged",
        status="failed",
        stopped=True,
        completed_at=datetime.now(timezone.utc),
        failure_kind="nan_guard",
        checkpoint_custody=[checkpoint],
        terminal_certification=TrainingRunCertification(
            schema_id=TRAINING_RUN_CERTIFICATION_SCHEMA_ID,
            schema_version=TRAINING_RUN_CERTIFICATION_SCHEMA_VERSION,
            termination_reason="diverged",
            certified_artifacts=[checkpoint],
        ),
    )
    assert training_run_certification(manifest).termination_reason == "diverged"
    assert training_run_certification(manifest).certified_artifacts == [checkpoint]

    empty = manifest.model_copy(
        update={
            "checkpoint_custody": [],
            "terminal_certification": TrainingRunCertification(
                schema_id=TRAINING_RUN_CERTIFICATION_SCHEMA_ID,
                schema_version=TRAINING_RUN_CERTIFICATION_SCHEMA_VERSION,
                termination_reason="diverged",
                certified_artifacts=[],
            ),
        }
    )
    assert training_run_certification(empty).certified_artifacts == []


def test_legacy_terminal_migration_completed_or_actionable_reject() -> None:
    checkpoint = _checkpoint()
    completed = TrainingRunManifest(
        id="feedbax-training-run:completed",
        status="completed",
        checkpoint_custody=[checkpoint],
    )
    projected = training_run_certification(completed)
    assert projected.termination_reason == "completed"
    assert projected.certified_artifacts == [checkpoint]

    failed = TrainingRunManifest(
        id="feedbax-training-run:old-failed",
        status="failed",
    )
    with pytest.raises(ValueError, match="cannot be migrated deterministically"):
        training_run_certification(failed)


def test_exact_parent_v1_migrates_and_unknown_version_rejects() -> None:
    payload = {
        "schema_id": STAGED_EXACT_PARENTS_SCHEMA_ID,
        "schema_version": STAGED_EXACT_PARENTS_SCHEMA_VERSION_V1,
        "parents": [
            {
                "parent": _parent().model_dump(mode="json"),
                "execution_uri": "manifest.json",
            }
        ],
    }
    migrated = migrate_staged_exact_parents(payload)
    assert migrated.schema_version == STAGED_EXACT_PARENTS_SCHEMA_VERSION
    assert migrated.parents[0].material_dependencies is None

    payload["schema_version"] = "feedbax.spec.staged_exact_parents.v0"
    with pytest.raises(ValueError, match="migration table"):
        migrate_staged_exact_parents(payload)


def test_public_conformance_slice_runs_positive_and_negative_canaries() -> None:
    spec = _spec()
    report = check_material_dependency_contract(spec, _observations(spec))
    assert report.dependency_count == 2
    assert report.missing_canary is True
    assert report.unauthentic_canary is True


def test_waiver_old_version_is_rejected() -> None:
    payload = AdmissionWaiver(
        schema_id=ADMISSION_WAIVER_SCHEMA_ID,
        schema_version=ADMISSION_WAIVER_SCHEMA_VERSION,
        incidental_check="manifest_status_completed",
        manifest=_parent(),
        artifact_sha256="b" * 64,
        reason="exact incident authorization",
    ).model_dump(mode="json")
    payload["schema_version"] = "feedbax.spec.admission_waiver.v0"
    with pytest.raises(ValidationError, match="admission_waiver.v1"):
        AdmissionWaiver.model_validate(payload)

    certification = TrainingRunCertification(
        schema_id=TRAINING_RUN_CERTIFICATION_SCHEMA_ID,
        schema_version=TRAINING_RUN_CERTIFICATION_SCHEMA_VERSION,
        termination_reason="completed",
        certified_artifacts=[],
    ).model_dump(mode="json")
    certification["schema_version"] = "feedbax.manifest.training_run_certification.v0"
    with pytest.raises(ValidationError, match="training_run_certification.v1"):
        TrainingRunCertification.model_validate(certification)

    for model, current in (
        (AdmissionWaiver, payload),
        (TrainingRunCertification, certification),
    ):
        for field_name in ("schema_id", "schema_version"):
            versionless = dict(current)
            versionless.pop(field_name)
            with pytest.raises(ValidationError, match=field_name):
                model.model_validate(versionless)

    material = _spec().model_dump(mode="json")
    for field_name in ("schema_id", "schema_version"):
        versionless = dict(material)
        versionless.pop(field_name)
        with pytest.raises(ValidationError, match=field_name):
            MaterialDependencySet.model_validate(versionless)
