"""Fail-closed admission of stored analysis records in `feedbax.analysis.specs`.

Every test here states one accidental-corruption scenario that used to be
admitted silently: a completed manifest filed under the right identifier whose
contents are not this run's, a same-identifier record of the wrong kind, a
locator that cannot be resolved, an ambiguous durable state authority, and a
substitution between writing a manifest and reading it back.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feedbax.analysis.evaluation import execute_evaluation_run_spec, EvaluationRecipeResult
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedExecutionContext,
)
from feedbax.analysis.fulfillment import FulfillmentAdmissionError, artifact_bytes_path
from feedbax.analysis.specs import (
    AnalysisEvaluationStatesResolutionError,
    AnalysisRecipeResult,
    ManifestKindMismatch,
    ManifestLocationError,
    _admit_cached_analysis_manifest,
    _durable_state_source,
    execute_analysis_run_spec,
    find_manifest_by_id,
    resolve_analysis_inputs,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.contracts.evaluation_states import EVALUATION_STATES_ARTIFACT_ROLE
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArtifactRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    analysis_run_manifest_id,
    canonical_manifest_path,
    safe_manifest_key,
    spec_payload,
    write_manifest,
)
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.analysis_recipe_contract]


ADMISSION_ANALYSIS_TYPE = "feedbax.test.admission_analysis"
ADMISSION_EVALUATION_TYPE = "feedbax.test.admission_eval"


def _register_recipes(application_registry_bundle, calls: list[int] | None = None) -> None:
    def evaluation_recipe(run_spec, _root, _states_path, _execution_context):
        return EvaluationRecipeResult(
            states={"value": np.asarray(run_spec.params["n_trials"], dtype=np.int32)},
            summary_metrics={"n_trials": run_spec.params["n_trials"]},
        )

    def analysis_recipe(_spec, _root, inputs, _execution_context):
        value = sum(int(resolved.states["value"]) for resolved in inputs)
        if calls is not None:
            calls.append(value)
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=value),
        )

    application_registry_bundle.evaluation_recipes.register(
        ADMISSION_EVALUATION_TYPE, evaluation_recipe
    )
    application_registry_bundle.analysis_recipes.register(
        ADMISSION_ANALYSIS_TYPE, analysis_recipe
    )


def _execute_eval(root: Path, *, application_registry_bundle) -> tuple[EvaluationRunManifest, Path]:
    spec = EvaluationRunSpec(
        evaluation_type=ADMISSION_EVALUATION_TYPE,
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id="feedbax-training-run:admission",
                role="training_run",
            )
        ],
        params={"n_trials": 3},
    )
    return execute_evaluation_run_spec(
        spec,
        root=root,
        force=True,
        registry=application_registry_bundle.evaluation_recipes,
    )


def _analysis_spec(evaluation_id: str, eval_path: Path) -> AnalysisRunSpec:
    return AnalysisRunSpec(
        analysis_type=ADMISSION_ANALYSIS_TYPE,
        inputs=[
            ParentRef(
                kind="EvaluationRunManifest",
                id=evaluation_id,
                role="evaluation_run",
                uri=str(eval_path),
            )
        ],
        params={"requested_outputs": ["toy"]},
    )


def _run_analysis(spec: AnalysisRunSpec, root: Path, *, application_registry_bundle):
    return execute_analysis_run_spec(
        spec,
        root=root,
        fig_dump_formats=("json",),
        registry=application_registry_bundle.analysis_recipes,
        evaluation_registry=application_registry_bundle.evaluation_recipes,
        experiment_registry=application_registry_bundle.experiment_packages,
    )


def _completed_analysis_run(tmp_path: Path, application_registry_bundle):
    calls: list[int] = []
    _register_recipes(application_registry_bundle, calls)
    eval_manifest, eval_path = _execute_eval(
        tmp_path, application_registry_bundle=application_registry_bundle
    )
    spec = _analysis_spec(eval_manifest.id, eval_path)
    manifest, path = _run_analysis(
        spec, tmp_path, application_registry_bundle=application_registry_bundle
    )
    assert manifest.status == "completed"
    assert calls == [3]
    return spec, manifest, path, calls


def test_cached_analysis_manifest_with_a_substituted_spec_is_refused(
    tmp_path: Path, application_registry_bundle
) -> None:
    """A completed record under the right id whose embedded spec is another run's.

    The forged payload is internally coherent — its ``SpecPayload`` digest is
    recomputed over the substituted inline spec — so it loads cleanly. Only an
    admission that compares the embedded spec with the requested one can tell
    that this record is not this run's answer.
    """
    spec, manifest, path, calls = _completed_analysis_run(tmp_path, application_registry_bundle)

    substituted_inline = {
        **manifest.analysis_spec.inline,
        "params": {"requested_outputs": ["not-this-run"]},
    }
    forged = manifest.model_copy(
        update={"analysis_spec": spec_payload("AnalysisRunSpec", substituted_inline)}
    )
    tampered = forged.model_dump_json(indent=2, exclude_none=True) + "\n"
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(FulfillmentAdmissionError) as excinfo:
        _run_analysis(spec, tmp_path, application_registry_bundle=application_registry_bundle)

    assert "spec_mismatch" in excinfo.value.outcome.codes
    assert calls == [3], "a refused cache must not silently re-execute"
    assert path.read_text(encoding="utf-8") == tampered, (
        "the refused bytes are the evidence and must survive the refusal"
    )


def test_cached_analysis_manifest_with_substituted_parents_is_refused(
    tmp_path: Path, application_registry_bundle
) -> None:
    spec, _manifest, path, _calls = _completed_analysis_run(tmp_path, application_registry_bundle)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"]["parents"] = [
        {
            "kind": "EvaluationRunManifest",
            "id": "feedbax-evaluation-run:not-a-parent",
            "role": "evaluation_run",
        }
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(FulfillmentAdmissionError) as excinfo:
        _run_analysis(spec, tmp_path, application_registry_bundle=application_registry_bundle)

    assert "parents_mismatch" in excinfo.value.outcome.codes


def test_cached_analysis_manifest_with_corrupt_artifact_bytes_is_refused(
    tmp_path: Path, application_registry_bundle
) -> None:
    spec, manifest, _path, _calls = _completed_analysis_run(tmp_path, application_registry_bundle)

    assert manifest.artifacts, "this run must produce at least one artifact to corrupt"
    artifact_path = artifact_bytes_path(manifest.artifacts[0], root=tmp_path)
    assert artifact_path is not None
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")

    with pytest.raises(FulfillmentAdmissionError) as excinfo:
        _run_analysis(spec, tmp_path, application_registry_bundle=application_registry_bundle)

    assert "artifact_sha256_mismatch" in excinfo.value.outcome.codes


def test_cached_analysis_manifest_that_is_still_this_run_is_reused(
    tmp_path: Path, application_registry_bundle
) -> None:
    """Strict admission must still admit the record the run actually wrote."""
    spec, manifest, path, calls = _completed_analysis_run(tmp_path, application_registry_bundle)

    reused, reused_path = _run_analysis(
        spec, tmp_path, application_registry_bundle=application_registry_bundle
    )

    assert calls == [3], "an admitted cache hit must not re-execute the recipe"
    assert reused.id == manifest.id
    assert reused_path == path


def _provider_backed_receipt(tmp_path: Path):
    """Return a valid receipt whose one artifact lives outside the manifest root."""
    provider = ImmutableArtifactBlobProvider(tmp_path / "provider")
    artifact = provider.store_bytes(
        b'{"values": [1, 2, 3]}',
        role="certificate",
        logical_name="certificate.json",
        media_type="application/json",
    )
    spec = AnalysisRunSpec(
        analysis_type=ADMISSION_ANALYSIS_TYPE,
        inputs=[],
        params={"requested_outputs": ["toy"]},
    )
    manifest = AnalysisRunManifest(
        id=analysis_run_manifest_id(spec),
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec", spec.model_dump(mode="json", exclude_none=True)
        ),
        provenance=Provenance(parents=list(spec.inputs)),
        artifacts=[artifact],
    )
    root = tmp_path / "outputs"
    path = write_manifest(manifest, root=root)
    context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"certificates": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(),
    )
    return provider, spec, manifest, root, path, context


def test_provider_held_artifacts_are_admitted_against_the_provider(tmp_path: Path) -> None:
    _provider, spec, manifest, root, path, context = _provider_backed_receipt(tmp_path)

    outcome = _admit_cached_analysis_manifest(
        spec, manifest, root=root, path=path, execution_context=context
    )

    assert outcome.admitted, outcome.describe()


def test_provider_held_artifacts_are_refused_without_the_binding(tmp_path: Path) -> None:
    """The root cannot resolve these bytes, and 'unresolvable' is not 'fine'."""
    _provider, spec, manifest, root, path, _context = _provider_backed_receipt(tmp_path)

    outcome = _admit_cached_analysis_manifest(
        spec,
        manifest,
        root=root,
        path=path,
        execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
    )

    assert not outcome.admitted
    assert "artifact_bytes_unresolvable" in outcome.codes


def test_provider_held_artifacts_with_corrupt_bytes_are_refused(tmp_path: Path) -> None:
    provider, spec, manifest, root, path, context = _provider_backed_receipt(tmp_path)
    digest = manifest.artifacts[0].sha256
    assert digest is not None
    blob = provider.root / "artifacts" / "sha256" / digest[:2] / digest
    blob.write_bytes(b'{"values": [9, 9, 9]}')

    outcome = _admit_cached_analysis_manifest(
        spec, manifest, root=root, path=path, execution_context=context
    )

    assert not outcome.admitted
    assert "artifact_bytes_unresolvable" in outcome.codes


def _evaluation_manifest(manifest_id: str) -> EvaluationRunManifest:
    return EvaluationRunManifest(
        id=manifest_id,
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            {"evaluation_type": "test.admission", "inputs": [], "params": {}},
        ),
    )


def test_find_manifest_by_id_refuses_a_same_id_record_of_another_kind(tmp_path: Path) -> None:
    manifest = _evaluation_manifest("feedbax-evaluation-run:kinded")
    write_manifest(manifest, root=tmp_path)

    found, _path = find_manifest_by_id(
        manifest.id, root=tmp_path, expected_kind="EvaluationRunManifest"
    )
    assert found.id == manifest.id

    with pytest.raises(ManifestKindMismatch, match="AnalysisRunManifest"):
        find_manifest_by_id(manifest.id, root=tmp_path, expected_kind="AnalysisRunManifest")


def test_find_manifest_by_id_refuses_bytes_filed_under_another_kinds_directory(
    tmp_path: Path,
) -> None:
    manifest = _evaluation_manifest("feedbax-evaluation-run:misfiled")
    misfiled = (
        tmp_path / "manifests" / "analysis_runs" / f"{safe_manifest_key(manifest.id)}.json"
    )
    misfiled.parent.mkdir(parents=True, exist_ok=True)
    misfiled.write_text(
        manifest.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestLocationError, match="canonical directory"):
        find_manifest_by_id(manifest.id, root=tmp_path)


def test_unresolvable_manifest_candidate_refuses_instead_of_being_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _evaluation_manifest("feedbax-evaluation-run:unresolvable")
    path = write_manifest(manifest, root=tmp_path)
    real_resolve = Path.resolve

    def failing_resolve(self: Path, *args, **kwargs):
        if self.name == path.name:
            raise OSError(62, "Too many levels of symbolic links")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", failing_resolve)

    with pytest.raises(ManifestLocationError, match="could not be"):
        find_manifest_by_id(manifest.id, root=tmp_path)


def test_analysis_input_ref_refuses_a_same_id_manifest_of_another_kind(
    tmp_path: Path, application_registry_bundle
) -> None:
    """An authored ref names a kind as well as an id; both must hold."""
    stored = _evaluation_manifest("feedbax-shared-id:collision")
    write_manifest(stored, root=tmp_path)
    spec = AnalysisRunSpec(
        analysis_type=ADMISSION_ANALYSIS_TYPE,
        inputs=[ParentRef(kind="TrainingRunManifest", id=stored.id, role="training_run")],
        params={},
    )

    with pytest.raises(ManifestKindMismatch, match="TrainingRunManifest"):
        resolve_analysis_inputs(
            spec,
            registry=application_registry_bundle.analysis_recipes,
            evaluation_registry=application_registry_bundle.evaluation_recipes,
            root=tmp_path,
        )


def _states_artifact(name: str) -> ArtifactRef:
    return ArtifactRef(
        role=EVALUATION_STATES_ARTIFACT_ROLE,
        logical_name=name,
        artifact_id=f"feedbax-artifact:{name}",
        sha256="0" * 64,
        size_bytes=1,
        metadata={
            "schema_id": "feedbax.evaluation_states",
            "schema_version": "feedbax.evaluation_states.v1",
            "storage_backend": "feedbax-local",
        },
    )


def _evaluation_manifest_with_state_artifacts(count: int) -> EvaluationRunManifest:
    return EvaluationRunManifest(
        id="feedbax-evaluation-run:states",
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            {"evaluation_type": "test.admission", "inputs": [], "params": {}},
        ),
        artifacts=[_states_artifact(f"states-{index}") for index in range(count)],
    )


@pytest.mark.parametrize(
    ("count", "code"),
    [(0, "missing_durable_states"), (2, "provenance_mismatch")],
    ids=["absent", "ambiguous"],
)
def test_durable_state_provenance_requires_exactly_one_state_artifact(
    tmp_path: Path, count: int, code: str
) -> None:
    """First-wins indexing would record one of several artifacts as *the* authority."""
    manifest = _evaluation_manifest_with_state_artifacts(count)
    path = write_manifest(manifest, root=tmp_path)
    ref = ParentRef(kind="EvaluationRunManifest", id=manifest.id, role="evaluation_run")

    with pytest.raises(AnalysisEvaluationStatesResolutionError) as excinfo:
        _durable_state_source(ref, manifest, manifest_path=path)

    assert excinfo.value.diagnostic.code == code


def test_durable_state_provenance_accepts_a_unique_state_artifact(tmp_path: Path) -> None:
    manifest = _evaluation_manifest_with_state_artifacts(1)
    path = write_manifest(manifest, root=tmp_path)
    ref = ParentRef(kind="EvaluationRunManifest", id=manifest.id, role="evaluation_run")

    source = _durable_state_source(ref, manifest, manifest_path=path)

    assert source.source_kind == "durable"
    assert source.artifact_id == "feedbax-artifact:states-0"


def test_written_analysis_manifest_identity_is_settled_from_the_returned_bytes(
    tmp_path: Path, application_registry_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest handed back is parsed from the bytes the read returned.

    Execution writes a record and then opens the path again to return it. If the
    bytes at that path are no longer this run's, the re-read must refuse rather
    than hand a foreign record to the caller as its own result.
    """
    _register_recipes(application_registry_bundle)
    eval_manifest, eval_path = _execute_eval(
        tmp_path, application_registry_bundle=application_registry_bundle
    )
    spec = _analysis_spec(eval_manifest.id, eval_path)
    substitute = AnalysisRunManifest(
        id="feedbax-analysis-run:substituted",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "test.substituted", "inputs": [], "params": {}},
        ),
    )

    def substituting_run(_analyses, _data, context, **_kwargs):
        context.finalize()
        assert context.manifest_path is not None
        context.manifest_path.write_text(
            substitute.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8",
        )
        return None, None, None

    monkeypatch.setattr(
        "feedbax.analysis.execution.run_analyses_with_context", substituting_run
    )

    with pytest.raises(ValueError, match="feedbax-analysis-run:substituted"):
        _run_analysis(spec, tmp_path, application_registry_bundle=application_registry_bundle)

    written = canonical_manifest_path(
        "AnalysisRunManifest", analysis_run_manifest_id(spec), root=tmp_path
    )
    assert json.loads(written.read_text(encoding="utf-8"))["id"] == substitute.id
