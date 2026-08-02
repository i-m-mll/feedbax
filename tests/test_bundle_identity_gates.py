"""Bundle identity gates that must not pass for want of anything to compare.

Three accidental-corruption scenarios that previously passed silently: an
exact-parent claim the pinned manifest never corroborated, a cached stage
receipt reused without proving its recorded artifacts still exist, and a root
set whose duplicate collapsed into the very key the comparison used.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from feedbax.analysis import bundles as bundle_module
from feedbax.analysis.bundles import (
    AnalysisBundleSpec,
    BundleStageOutputSpec,
    BundleStageSpec,
    DuplicateBundleRootError,
    ManifestPredicate,
    VerifiedBundleRoot,
    dry_run_staged_analysis_bundle,
    execute_staged_analysis_bundle,
)
from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.analysis.exact_parents import (
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    StagedExactParentEntry,
    StagedExactParents,
)
from feedbax.analysis.fulfillment import artifact_bytes_path
from feedbax.analysis.manifest_inputs import resolve_manifest_input
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    TrainingRunManifest,
    load_manifest,
    sha256_bytes,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.selection import SelectionSpec


EXACT_EVALUATION_TYPE = "feedbax.test.identity_gate_eval"


def _exact_bundle() -> AnalysisBundleSpec:
    return AnalysisBundleSpec(
        name="identity_gate_bundle",
        predicate=ManifestPredicate(
            manifest_kind="TrainingRunManifest",
            metadata_equals={"method": "minimax"},
        ),
        stages=[
            BundleStageSpec(
                name="evaluate",
                kind="evaluation",
                mode="per-run",
                evaluation_type=EXACT_EVALUATION_TYPE,
            )
        ],
    )


def _register_eval_recipe(registries, calls: list[int]) -> None:
    def recipe(run_spec: EvaluationRunSpec, _root, _states_path, _execution_context):
        n_trials = int(run_spec.params["n_trials"])
        calls.append(n_trials)
        return EvaluationRecipeResult(
            states={"value": np.asarray(n_trials, dtype=np.int32)},
            summary_metrics={"n_trials": n_trials},
        )

    registries.evaluation_recipes.register(EXACT_EVALUATION_TYPE, recipe)


# --------------------------------------------------------------------------
# A78: a governed fact the manifest declares must be stated, not absent
# --------------------------------------------------------------------------


def _write_exact_parent(root: Path, *, run_set_id: str | None) -> StagedExactParentEntry:
    run_id = "feedbax-training-run:exact-identity"
    manifest = TrainingRunManifest(
        id=run_id,
        status="completed",
        run_set_id=run_set_id,
        metadata={"method": "minimax", "row_id": "row-a", "planned_run_id": run_id},
    )
    raw = manifest.model_dump_json(indent=2).encode("utf-8")
    relative = Path("exact-inputs") / "identity.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    digest = sha256_bytes(raw)
    return StagedExactParentEntry(
        parent=ParentRef(
            kind="TrainingRunManifest",
            id=run_id,
            role="training_run",
            uri=f"artifact://sha256/{digest}",
            metadata={
                "manifest_sha256": digest,
                "size_bytes": len(raw),
                "run_set_id": "run-set-a",
                "row_id": "row-a",
                "manifest_status": "completed",
                "registration_status": "completed",
                "conformance_overall": "pass",
                "certificate_sha256": "c" * 64,
                "planned_run_id": run_id,
            },
        ),
        execution_uri=relative.as_posix(),
    )


def _exact_document(entry: StagedExactParentEntry) -> StagedExactParents:
    return StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=[entry],
        metadata={"authority": "test"},
    )


def test_exact_parent_states_the_run_set_identity_its_ref_claims(tmp_path: Path) -> None:
    """Control: a manifest that states the governed fact is admitted."""
    entry = _write_exact_parent(tmp_path, run_set_id="run-set-a")

    result = dry_run_staged_analysis_bundle(
        _exact_bundle(), root=tmp_path, exact_parents=_exact_document(entry)
    )

    assert result.matched_run_ids == [entry.parent.id]


def test_exact_parent_manifest_omitting_run_set_identity_refuses(tmp_path: Path) -> None:
    """A declared field left unset is a stated absence, never silent agreement."""
    entry = _write_exact_parent(tmp_path, run_set_id=None)

    with pytest.raises(ValueError, match="declares no 'run_set_id'"):
        dry_run_staged_analysis_bundle(
            _exact_bundle(), root=tmp_path, exact_parents=_exact_document(entry)
        )


def test_exact_parent_manifest_disagreeing_on_run_set_identity_refuses(
    tmp_path: Path,
) -> None:
    entry = _write_exact_parent(tmp_path, run_set_id="run-set-b")

    with pytest.raises(ValueError, match="disagrees with ParentRef metadata.run_set_id"):
        dry_run_staged_analysis_bundle(
            _exact_bundle(), root=tmp_path, exact_parents=_exact_document(entry)
        )


def test_every_governed_exact_parent_fact_names_an_authority() -> None:
    """The authority partition is total, so no fact defaults to an absent check."""
    assert not (
        set(bundle_module._GOVERNED_EXACT_PARENT_FACTS)
        - set(bundle_module._MANIFEST_DECLARED_EXACT_PARENT_FACTS)
        - bundle_module._REGISTRY_AUTHORITY_EXACT_PARENT_FACTS
    )


# --------------------------------------------------------------------------
# A77: a reused stage receipt is admitted, artifact custody included
# --------------------------------------------------------------------------


def _cache_case(root: Path, registries) -> tuple[AnalysisBundleSpec, Path, list[int]]:
    calls: list[int] = []
    _register_eval_recipe(registries, calls)
    write_manifest(
        TrainingRunManifest(
            id="feedbax-training-run:cache",
            status="completed",
            metadata={"method": "minimax"},
        ),
        root=root,
    )
    bundle = AnalysisBundleSpec(
        name="identity_gate_cache",
        predicate=ManifestPredicate(
            manifest_kind="TrainingRunManifest",
            metadata_equals={"method": "minimax"},
        ),
        stages=[
            BundleStageSpec(
                name="evaluate",
                kind="evaluation",
                evaluation_type=EXACT_EVALUATION_TYPE,
                local_params={"n_trials": 3},
                states_custody="durable",
                outputs=[
                    BundleStageOutputSpec(role="manifest"),
                    BundleStageOutputSpec(role="evaluation_states"),
                ],
            )
        ],
    )
    first = execute_staged_analysis_bundle(bundle, root=root, registries=registries)
    evaluation_path = resolve_manifest_input(first.stages[0].manifest_refs[0], root).path
    calls.clear()
    return bundle, evaluation_path, calls


def test_reused_evaluation_receipt_is_admitted_when_its_artifacts_verify(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    """Control: an intact cached receipt is reused without re-executing."""
    bundle, _evaluation_path, calls = _cache_case(tmp_path, application_registry_bundle)

    execute_staged_analysis_bundle(bundle, root=tmp_path, registries=application_registry_bundle)

    assert calls == []


def test_reused_evaluation_receipt_refuses_when_artifact_bytes_are_missing(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    """A receipt whose products are gone describes work that no longer exists."""
    bundle, evaluation_path, calls = _cache_case(tmp_path, application_registry_bundle)
    cached = load_manifest(evaluation_path)
    assert isinstance(cached, EvaluationRunManifest)
    assert cached.artifacts
    for artifact in cached.artifacts:
        bytes_path = artifact_bytes_path(artifact, root=tmp_path)
        assert bytes_path is not None and bytes_path.is_file()
        bytes_path.unlink()

    with pytest.raises(ValueError, match="not admissible as this stage's receipt"):
        execute_staged_analysis_bundle(
            bundle, root=tmp_path, registries=application_registry_bundle
        )

    assert calls == []


def test_reused_evaluation_receipt_refuses_when_artifact_bytes_changed(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    bundle, evaluation_path, calls = _cache_case(tmp_path, application_registry_bundle)
    cached = load_manifest(evaluation_path)
    assert isinstance(cached, EvaluationRunManifest)
    for artifact in cached.artifacts:
        bytes_path = artifact_bytes_path(artifact, root=tmp_path)
        assert bytes_path is not None
        original = bytes_path.read_bytes()
        bytes_path.write_bytes(original + b"\x00tampered")
        assert hashlib.sha256(bytes_path.read_bytes()).hexdigest() != artifact.sha256

    with pytest.raises(ValueError, match="not admissible as this stage's receipt"):
        execute_staged_analysis_bundle(
            bundle, root=tmp_path, registries=application_registry_bundle
        )

    assert calls == []


# --------------------------------------------------------------------------
# B44: duplicate root addresses never collapse into the comparison key
# --------------------------------------------------------------------------


def _write_selectable_training(root: Path, *, relative: str) -> TrainingRunManifest:
    manifest = TrainingRunManifest(
        id="feedbax-training-run:selectable",
        status="completed",
        run_set_id="run-set-a",
        metadata={"method": "minimax"},
    )
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def test_dry_run_selection_spec_refuses_a_duplicated_root_address(tmp_path: Path) -> None:
    """The selection_spec dry-run path scans, so it refuses a corrupt root too."""
    manifest = _write_selectable_training(
        tmp_path, relative="manifests/training_runs/original.json"
    )
    _write_selectable_training(tmp_path, relative="manifests/imported/copy.json")

    with pytest.raises(DuplicateBundleRootError, match="same root receipt more than once"):
        dry_run_staged_analysis_bundle(
            _exact_bundle(),
            root=tmp_path,
            selection_spec=SelectionSpec(
                mode="explicit",
                manifest_kind="TrainingRunManifest",
                ids=[manifest.id],
            ),
        )


def _verified_root(root: Path, *, kind: str, manifest) -> VerifiedBundleRoot:
    raw = manifest.model_dump_json(indent=2).encode("utf-8")
    path = root / "manifests" / "verified" / f"{kind}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return VerifiedBundleRoot(
        kind=kind,
        id=manifest.id,
        path=path,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


def test_prerequisite_input_set_refuses_two_roots_sharing_one_id(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    """Bindings address an input by id, so an id collapse is refused by id."""
    calls: list[int] = []
    _register_eval_recipe(application_registry_bundle, calls)
    shared_id = "feedbax-run:shared"
    training = TrainingRunManifest(
        id=shared_id, status="completed", run_set_id="run-set-a", metadata={"method": "minimax"}
    )
    evaluation = EvaluationRunManifest(
        id=shared_id,
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec", {"evaluation_type": EXACT_EVALUATION_TYPE}
        ),
    )
    roots = [
        _verified_root(tmp_path, kind="TrainingRunManifest", manifest=training),
        _verified_root(tmp_path, kind="EvaluationRunManifest", manifest=evaluation),
    ]

    with pytest.raises(DuplicateBundleRootError, match="same root receipt more than once"):
        execute_staged_analysis_bundle(
            _exact_bundle(),
            root=tmp_path,
            verified_roots=roots,
            registries=application_registry_bundle,
        )

    assert calls == []
