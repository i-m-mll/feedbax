from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.manifest import (
    SCHEMA_VERSION,
    CheckpointCandidateRef,
    CheckpointScorerIdentity,
    CheckpointScoreSummary,
    CheckpointSelectionBank,
    CheckpointSelectionGroup,
    CheckpointSelectionManifest,
    CheckpointSelectionSpec,
    ParentRef,
    checkpoint_selection_manifest_id,
    load_manifest,
    normalize_checkpoint_selection_lineage,
    spec_payload,
    write_manifest,
)
from feedbax.migrations import UnsupportedSpecVersion, default_spec_registry


def _parent(kind: str, id_: str, role: str, uri: str | None = None) -> ParentRef:
    return ParentRef(kind=kind, id=id_, role=role, uri=uri)


def _scorer() -> CheckpointScorerIdentity:
    return CheckpointScorerIdentity(
        scorer_id="feedbax.test.fixed_bank.best_loss",
        name="Fixed-bank validation loss",
        version="2026-06-14",
        plugin="tests.fixed_bank_scorer",
        parameters={"primary_metric": "validation_loss", "objective": "minimize"},
    )


def _available_bank() -> CheckpointSelectionBank:
    return CheckpointSelectionBank(
        role="fixed",
        bank_id="eval-bank:fixed:demo",
        status="available",
        ref=_parent(
            "EvaluationRunManifest",
            "feedbax-evaluation-run:fixed-bank-demo",
            "evaluation_bank",
            "repo://manifests/evaluation_runs/fixed-bank-demo.json",
        ),
        metadata={"split": "validation", "n_trials": 128},
    )


def _missing_bank() -> CheckpointSelectionBank:
    return CheckpointSelectionBank(
        role="fixed",
        bank_id="eval-bank:missing:demo",
        status="missing",
        ref=_parent(
            "EvaluationRunManifest",
            "feedbax-evaluation-run:missing-bank-demo",
            "evaluation_bank",
        ),
        metadata={"unavailable_reason": "evaluation bank manifest was not materialized"},
    )


def _candidate(run: str, step: int, loss: float, rank: int) -> CheckpointCandidateRef:
    digest = f"{run}-{step}"
    return CheckpointCandidateRef(
        id=digest,
        checkpoint=_parent(
            "ArtifactRef",
            f"artifact://checkpoint/{digest}",
            "checkpoint",
            f"repo://artifacts/checkpoints/{digest}.eqx",
        ),
        model_artifact=_parent(
            "ModelArtifactManifest",
            f"feedbax-model-artifact:{digest}",
            "model_artifact",
        ),
        training_run=_parent(
            "TrainingRunManifest",
            f"feedbax-training-run:{run}",
            "training_run",
        ),
        run_id=run,
        step=step,
        metadata={"run": run, "validation_loss": loss, "rank": rank},
    )


def _score(candidate: CheckpointCandidateRef) -> CheckpointScoreSummary:
    loss = float(candidate.metadata["validation_loss"])
    rank = int(candidate.metadata["rank"])
    return CheckpointScoreSummary(
        candidate_id=candidate.id,
        primary_metric="validation_loss",
        objective="minimize",
        primary_value=loss,
        rank=rank,
        metrics={"validation_loss": loss, "success_rate": 1.0 - loss},
    )


def _spec(
    *,
    bank: CheckpointSelectionBank | None = None,
    candidates: list[CheckpointCandidateRef] | None = None,
) -> CheckpointSelectionSpec:
    return CheckpointSelectionSpec(
        selection_type="fixed_bank",
        scorer=_scorer(),
        bank=bank or _available_bank(),
        candidate_checkpoints=candidates or [
            _candidate("run-a", 100, 0.43, 2),
            _candidate("run-a", 200, 0.21, 1),
        ],
        group_by="run",
        fallback_allowed=False,
        metadata={"experiment": "checkpoint-selection-demo"},
    )


def test_fixed_bank_selection_records_selected_checkpoint_and_metadata(
    tmp_path: Path,
) -> None:
    candidates = [_candidate("run-a", 100, 0.43, 2), _candidate("run-a", 200, 0.21, 1)]
    spec = _spec(candidates=candidates)
    group = CheckpointSelectionGroup(
        run_id="run-a",
        candidate_checkpoints=candidates,
        selected_checkpoint=candidates[1],
        score_summaries=[_score(candidate) for candidate in candidates],
    )
    manifest = CheckpointSelectionManifest(
        id=checkpoint_selection_manifest_id(spec),
        status="completed",
        selection_status="selected",
        selection_spec=spec_payload(
            "CheckpointSelectionSpec",
            spec.model_dump(mode="json", exclude_none=True),
        ),
        scorer=spec.scorer,
        bank=spec.bank,
        selections=[group],
    )

    path = write_manifest(manifest, root=tmp_path, index=False)
    loaded = load_manifest(path)

    assert loaded.kind == "CheckpointSelectionManifest"
    assert loaded.id == checkpoint_selection_manifest_id(spec)
    assert loaded.selection_spec.schema_id == "feedbax.spec.checkpoint_selection"
    assert loaded.selection_spec.schema_version == "feedbax.spec.checkpoint_selection.v1"
    assert loaded.scorer.scorer_id == "feedbax.test.fixed_bank.best_loss"
    assert loaded.bank.bank_id == "eval-bank:fixed:demo"
    assert loaded.selection_status == "selected"
    assert (
        loaded.selections[0].selected_checkpoint.checkpoint.id
        == "artifact://checkpoint/run-a-200"
    )
    selected_score = loaded.selections[0].score_summaries[1]
    assert selected_score.candidate_id == loaded.selections[0].selected_checkpoint.id
    assert selected_score.primary_value == pytest.approx(0.21)


def test_per_replicate_selection_records_independent_groups_and_selected_refs(
    tmp_path: Path,
) -> None:
    replicate_0 = [_candidate("replicate-0", 100, 0.25, 2), _candidate("replicate-0", 200, 0.12, 1)]
    replicate_1 = [_candidate("replicate-1", 100, 0.31, 2), _candidate("replicate-1", 200, 0.16, 1)]
    spec = _spec(candidates=[*replicate_0, *replicate_1])
    groups = [
        CheckpointSelectionGroup(
            scope="replicate",
            run_id="replicate-0",
            replicate_id="0",
            candidate_checkpoints=replicate_0,
            selected_checkpoint=replicate_0[1],
            score_summaries=[_score(candidate) for candidate in replicate_0],
        ),
        CheckpointSelectionGroup(
            scope="replicate",
            run_id="replicate-1",
            replicate_id="1",
            candidate_checkpoints=replicate_1,
            selected_checkpoint=replicate_1[1],
            score_summaries=[_score(candidate) for candidate in replicate_1],
        ),
    ]

    manifest = CheckpointSelectionManifest(
        id=checkpoint_selection_manifest_id(spec),
        status="completed",
        selection_status="selected",
        selection_spec=spec_payload(
            "CheckpointSelectionSpec",
            spec.model_dump(mode="json", exclude_none=True),
        ),
        scorer=spec.scorer,
        bank=spec.bank,
        selections=groups,
    )

    loaded = load_manifest(write_manifest(manifest, root=tmp_path, index=False))

    assert [group.replicate_id for group in loaded.selections] == ["0", "1"]
    assert [group.selected_checkpoint.training_run.id for group in loaded.selections] == [
        "feedbax-training-run:replicate-0",
        "feedbax-training-run:replicate-1",
    ]
    assert [group.selected_checkpoint.step for group in loaded.selections] == [200, 200]


def test_missing_bank_cannot_record_selected_status_without_explicit_fallback() -> None:
    candidates = [_candidate("run-a", 100, 0.43, 1)]
    spec = _spec(bank=_missing_bank(), candidates=candidates)
    with pytest.raises(ValidationError) as excinfo:
        CheckpointSelectionManifest(
            id=checkpoint_selection_manifest_id(spec),
            status="failed",
            selection_status="selected",
            selection_spec=spec_payload(
                "CheckpointSelectionSpec",
                spec.model_dump(mode="json", exclude_none=True),
            ),
            scorer=spec.scorer,
            bank=spec.bank,
            selections=[
                CheckpointSelectionGroup(
                    run_id="run-a",
                    candidate_checkpoints=candidates,
                    selected_checkpoint=candidates[0],
                    score_summaries=[_score(candidates[0])],
                )
            ],
        )

    message = str(excinfo.value)
    assert "selection_status='selected'" in message
    assert "bank" in message
    assert "missing" in message


def test_fallback_selected_requires_policy_and_reason_or_fallback_bank() -> None:
    candidates = [_candidate("run-a", 100, 0.43, 1)]
    spec = _spec(bank=_missing_bank(), candidates=candidates)

    with pytest.raises(ValidationError) as no_policy:
        CheckpointSelectionManifest(
            id=checkpoint_selection_manifest_id(spec),
            status="completed",
            selection_status="fallback_selected",
            selection_spec=spec_payload(
                "CheckpointSelectionSpec",
                spec.model_dump(mode="json", exclude_none=True),
            ),
            scorer=spec.scorer,
            bank=spec.bank,
            selections=[
                CheckpointSelectionGroup(
                    run_id="run-a",
                    candidate_checkpoints=candidates,
                    selected_checkpoint=candidates[0],
                    score_summaries=[_score(candidates[0])],
                )
            ],
        )

    assert "fallback_allowed" in str(no_policy.value)

    with pytest.raises(ValidationError) as no_reason:
        CheckpointSelectionManifest(
            id=checkpoint_selection_manifest_id(spec),
            status="completed",
            selection_status="fallback_selected",
            selection_spec=spec_payload(
                "CheckpointSelectionSpec",
                spec.model_dump(mode="json", exclude_none=True),
            ),
            scorer=spec.scorer,
            bank=spec.bank,
            selections=[
                CheckpointSelectionGroup(
                    run_id="run-a",
                    candidate_checkpoints=candidates,
                    selected_checkpoint=candidates[0],
                    score_summaries=[_score(candidates[0])],
                )
            ],
            fallback_allowed=True,
        )

    assert "failure_reason" in str(no_reason.value)
    assert "fallback_reason" in str(no_reason.value)
    assert "fallback_ref" in str(no_reason.value)

    fallback = CheckpointSelectionManifest(
        id=checkpoint_selection_manifest_id(spec),
        status="completed",
        selection_status="fallback_selected",
        selection_spec=spec_payload(
            "CheckpointSelectionSpec",
            spec.model_dump(mode="json", exclude_none=True),
        ),
        scorer=spec.scorer,
        bank=spec.bank,
        selections=[
            CheckpointSelectionGroup(
                run_id="run-a",
                candidate_checkpoints=candidates,
                selected_checkpoint=candidates[0],
                score_summaries=[_score(candidates[0])],
            )
        ],
        fallback_allowed=True,
        failure_reason="using last completed checkpoint because the fixed bank is unavailable",
    )

    assert fallback.selection_status == "fallback_selected"
    assert fallback.failure_reason is not None


def test_manifest_lineage_indexes_selected_checkpoint_model_training_and_bank_refs(
    tmp_path: Path,
) -> None:
    candidates = [_candidate("run-a", 100, 0.43, 2), _candidate("run-a", 200, 0.21, 1)]
    spec = _spec(candidates=candidates)
    manifest = CheckpointSelectionManifest(
        id=checkpoint_selection_manifest_id(spec),
        status="completed",
        selection_status="selected",
        selection_spec=spec_payload(
            "CheckpointSelectionSpec",
            spec.model_dump(mode="json", exclude_none=True),
        ),
        scorer=spec.scorer,
        bank=spec.bank,
        selections=[
            CheckpointSelectionGroup(
                run_id="run-a",
                candidate_checkpoints=candidates,
                selected_checkpoint=candidates[1],
                score_summaries=[_score(candidate) for candidate in candidates],
            )
        ],
    )

    normalized = normalize_checkpoint_selection_lineage(manifest)
    lineage = {(ref.kind, ref.id, ref.role) for ref in normalized.provenance.parents}

    assert (
        "ArtifactRef",
        "artifact://checkpoint/run-a-200",
        "checkpoint",
    ) in lineage
    assert (
        "ModelArtifactManifest",
        "feedbax-model-artifact:run-a-200",
        "model_artifact",
    ) in lineage
    assert (
        "TrainingRunManifest",
        "feedbax-training-run:run-a",
        "training_run",
    ) in lineage
    assert (
        "EvaluationRunManifest",
        "feedbax-evaluation-run:fixed-bank-demo",
        "evaluation_bank",
    ) in lineage

    write_manifest(manifest, root=tmp_path, index=True)
    db_path = tmp_path / "index" / "feedbax.sqlite"
    with sqlite3.connect(db_path) as conn:
        indexed_lineage = set(
            conn.execute(
                """
                SELECT parent_kind, parent_id, role
                FROM lineage_edges
                WHERE child_id = ?
                """,
                (manifest.id,),
            ).fetchall()
        )

    assert lineage <= indexed_lineage


def test_checkpoint_selection_schema_registry_identities_and_v0_rejection() -> None:
    spec_family = default_spec_registry.resolve("CheckpointSelectionSpec")
    manifest_family = default_spec_registry.resolve("CheckpointSelectionManifest")

    assert spec_family.identity == "feedbax.spec.checkpoint_selection"
    assert spec_family.current_version == "feedbax.spec.checkpoint_selection.v1"
    assert manifest_family.identity == "feedbax.manifest.checkpoint_selection"
    assert manifest_family.current_version == SCHEMA_VERSION

    for family, old_version in (
        ("CheckpointSelectionSpec", "feedbax.spec.checkpoint_selection.v0"),
        ("CheckpointSelectionManifest", "feedbax.manifest.checkpoint_selection.v0"),
    ):
        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            default_spec_registry.migrate(family, {"schema_version": old_version})

        message = str(excinfo.value)
        assert f"family='{family}'" in message
        assert f"source_version='{old_version}'" in message
        assert "migration_intentionally_absent=yes" in message
