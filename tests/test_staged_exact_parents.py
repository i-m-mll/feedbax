from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from feedbax.analysis.bundles import (
    AnalysisBundleSpec,
    BundleStageSpec,
    dry_run_staged_analysis_bundle,
    execute_staged_analysis_bundle,
)
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
)
from feedbax.analysis.evaluation_inputs import (
    EvaluationInputAmbiguityError,
    resolve_evaluation_inputs,
)
from feedbax.analysis.exact_parents import (
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION_V1,
    StagedExactParentEntry,
    StagedExactParents,
)
from feedbax.analysis.execution_context import (
    StagedArtifactProviderRootBinding,
    StagedCheckpointCustodyRootBinding,
)
from feedbax.analysis.manifest_inputs import resolve_manifest_input
from feedbax.analysis.reports import BUNDLE_SUMMARY_REPORT_TYPE
from feedbax.contracts.manifest import (
    ArtifactRef,
    EvaluationRunSpec,
    ParentRef,
    TRAINING_RUN_CERTIFICATION_SCHEMA_ID,
    TRAINING_RUN_CERTIFICATION_SCHEMA_VERSION,
    TrainingRunCertification,
    TrainingRunManifest,
    evaluation_run_manifest_id,
    sha256_bytes,
    write_manifest,
)
from feedbax.contracts.material_dependencies import (
    ADMISSION_WAIVER_SCHEMA_ID,
    ADMISSION_WAIVER_SCHEMA_VERSION,
    MATERIAL_DEPENDENCIES_SCHEMA_ID,
    MATERIAL_DEPENDENCIES_SCHEMA_VERSION,
    AdmissionWaiver,
    MaterialDependency,
    MaterialDependencySet,
    dependency_value_sha256,
)
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.selection import ManifestPredicate, TopKByMetricPerGroup
from feedbax.contracts.staged_execution import (
    STAGED_CHECKPOINT_CUSTODY_BACKEND,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedCheckpointCustodySpec,
    StagedExecutionDescriptor,
)
from tests.test_checkpoint_custody import (
    _resolver_parent_ref,
    _write_resolver_checkpoint,
)
from feedbax.persistence.artifact_custody import open_immutable_artifact_blob_provider
from feedbax.plugins.bootstrap import BootstrapState


pytestmark = [pytest.mark.feedbax_contract, pytest.mark.analysis_recipe_contract]

EXACT_EVALUATION_TYPE = "feedbax.test.staged_exact_parent_eval"


class EvaluationCalls(list[EvaluationRunSpec]):
    def __init__(self, registries):
        super().__init__()
        self.registries = registries


@pytest.fixture
def exact_evaluation_calls(application_registry_bundle):
    calls = EvaluationCalls(application_registry_bundle)

    def recipe(
        run_spec: EvaluationRunSpec,
        root: Path,
        _states_path: Path,
        execution_context,
    ):
        resolved = resolve_evaluation_inputs(
            run_spec,
            manifest_root=root,
            execution_context=execution_context,
        )
        assert resolved[0].ref == run_spec.inputs[0]
        calls.append(run_spec)
        return EvaluationRecipeResult(
            states={"value": np.asarray(len(calls), dtype=np.int32)},
            summary_metrics={"parent_count": len(run_spec.inputs)},
        )

    application_registry_bundle.evaluation_recipes.register(EXACT_EVALUATION_TYPE, recipe)
    return calls


def _bundle(
    *,
    predicate: ManifestPredicate | None = None,
    include_report: bool = False,
) -> AnalysisBundleSpec:
    stages = [
        BundleStageSpec(
            name="evaluate",
            kind="evaluation",
            mode="per-run",
            evaluation_type=EXACT_EVALUATION_TYPE,
        )
    ]
    if include_report:
        stages.append(
            BundleStageSpec(
                name="group",
                kind="report",
                mode="grouped",
                depends_on=["evaluate"],
                report_type=BUNDLE_SUMMARY_REPORT_TYPE,
            )
        )
    return AnalysisBundleSpec(
        name="exact_parent_bundle",
        predicate=predicate
        or ManifestPredicate(
            manifest_kind="TrainingRunManifest",
            metadata_equals={"method": "minimax"},
        ),
        stages=stages,
    )


def _write_exact_parent(
    root: Path,
    *,
    run_id: str,
    row_id: str,
    suffix: str = "",
    status: str = "completed",
) -> StagedExactParentEntry:
    manifest = TrainingRunManifest(
        id=run_id,
        status=status,
        run_set_id="run-set-a",
        metadata={
            "method": "minimax",
            "row_id": row_id,
            "planned_run_id": run_id,
            "content_variant": suffix,
        },
    )
    raw_bytes = manifest.model_dump_json(indent=2).encode("utf-8")
    relative_path = Path("exact-inputs") / f"{row_id}{suffix}.json"
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)
    digest = sha256_bytes(raw_bytes)
    return StagedExactParentEntry(
        parent=ParentRef(
            kind="TrainingRunManifest",
            id=run_id,
            role="training_run",
            uri=f"artifact://sha256/{digest}",
            metadata={
                "manifest_sha256": digest,
                "size_bytes": len(raw_bytes),
                "run_set_id": "run-set-a",
                "row_id": row_id,
                "manifest_status": "completed",
                "registration_status": "completed",
                "conformance_overall": "pass",
                "certificate_sha256": "c" * 64,
                "planned_run_id": run_id,
            },
        ),
        execution_uri=relative_path.as_posix(),
    )


def _exact_document(*entries: StagedExactParentEntry) -> StagedExactParents:
    return StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=list(entries),
        metadata={"authority": "test"},
    )


def _write_diverged_exact_parent(
    root: Path,
    *,
    suffix: str = "",
    certified: bool = True,
    checkpoint: ParentRef | ArtifactRef | None = None,
    waiver_manifest_digest: str | None = None,
    waiver_artifact_digest: str | None = None,
) -> StagedExactParentEntry:
    if checkpoint is None:
        checkpoint_result = _write_resolver_checkpoint(root / "checkpoint-custody")
        checkpoint = _resolver_parent_ref(checkpoint_result)
        checkpoint = checkpoint.model_copy(
            update={
                "metadata": {
                    **checkpoint.metadata,
                    "checkpoint_custody_binding": "certified-checkpoints",
                }
            }
        )
    checkpoint_digest = dependency_value_sha256(checkpoint)
    assert checkpoint_digest is not None
    run_id = "feedbax-training-run:diverged"
    manifest = TrainingRunManifest(
        id=run_id,
        status="failed",
        stopped=True,
        completed_at="2026-07-31T00:00:00Z",
        failure_kind="nan_guard",
        run_set_id="run-set-a",
        checkpoint_custody=[checkpoint],
        terminal_certification=TrainingRunCertification(
            schema_id=TRAINING_RUN_CERTIFICATION_SCHEMA_ID,
            schema_version=TRAINING_RUN_CERTIFICATION_SCHEMA_VERSION,
            termination_reason="diverged",
            certified_artifacts=[checkpoint] if certified else [],
        ),
        metadata={
            "method": "minimax",
            "row_id": "row-diverged",
            "planned_run_id": run_id,
            "provenance_note": suffix,
        },
    )
    raw_bytes = manifest.model_dump_json(indent=2).encode("utf-8")
    relative_path = Path("exact-inputs") / f"diverged{suffix}.json"
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)
    digest = sha256_bytes(raw_bytes)
    parent = ParentRef(
        kind="TrainingRunManifest",
        id=run_id,
        role="training_run",
        uri=f"artifact://sha256/{digest}",
        metadata={
            "manifest_sha256": digest,
            "size_bytes": len(raw_bytes),
            "run_set_id": "run-set-a",
            "row_id": "row-diverged",
            "manifest_status": "failed",
            "registration_status": "completed",
            "conformance_overall": "pass",
            "certificate_sha256": "c" * 64,
            "planned_run_id": run_id,
        },
    )
    waiver_parent = (
        parent
        if waiver_manifest_digest is None
        else parent.model_copy(
            update={
                "uri": f"artifact://sha256/{waiver_manifest_digest}",
                "metadata": {
                    **parent.metadata,
                    "manifest_sha256": waiver_manifest_digest,
                },
            }
        )
    )
    material_dependencies = MaterialDependencySet(
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
        provenance_metadata={"sampling_contract": suffix},
        waiver=AdmissionWaiver(
            schema_id=ADMISSION_WAIVER_SCHEMA_ID,
            schema_version=ADMISSION_WAIVER_SCHEMA_VERSION,
            incidental_check="manifest_status_completed",
            manifest=parent,
            artifact_sha256=checkpoint_digest,
            reason="execution diverged after the checkpoint was certified",
        ),
    )
    if waiver_manifest_digest is not None or waiver_artifact_digest is not None:
        material_dependencies = material_dependencies.model_copy(
            update={
                "waiver": AdmissionWaiver(
                    schema_id=ADMISSION_WAIVER_SCHEMA_ID,
                    schema_version=ADMISSION_WAIVER_SCHEMA_VERSION,
                    incidental_check="manifest_status_completed",
                    manifest=waiver_parent,
                    artifact_sha256=waiver_artifact_digest or checkpoint_digest,
                    reason="execution diverged after the checkpoint was certified",
                )
            }
        )
    return StagedExactParentEntry(
        parent=parent,
        execution_uri=relative_path.as_posix(),
        material_dependencies=material_dependencies,
    )


def _material_dependency_execution_kwargs(
    root: Path,
    *,
    checkpoint_root: Path | None = None,
) -> dict[str, object]:
    return {
        "execution_descriptor": StagedExecutionDescriptor(
            schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
            schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
            artifact_providers={},
            checkpoint_custody={
                "certified-checkpoints": StagedCheckpointCustodySpec(
                    backend=STAGED_CHECKPOINT_CUSTODY_BACKEND
                )
            },
        ),
        "checkpoint_custody_bindings": [
            StagedCheckpointCustodyRootBinding(
                "certified-checkpoints",
                checkpoint_root or root / "checkpoint-custody",
            )
        ],
    }


def _artifact_dependency_execution_kwargs(root: Path) -> dict[str, object]:
    return {
        "execution_descriptor": StagedExecutionDescriptor(
            schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
            schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
            artifact_providers={"certified-artifacts": ImmutableArtifactBlobProviderSpec()},
            checkpoint_custody={},
        ),
        "artifact_provider_bindings": [
            StagedArtifactProviderRootBinding(
                "certified-artifacts",
                root / "artifact-provider",
            )
        ],
    }


def _rewrite_manifest_metadata(
    root: Path,
    entry: StagedExactParentEntry,
    *,
    field_name: str,
    value: object,
) -> StagedExactParentEntry:
    path = root / entry.execution_uri
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"][field_name] = value
    raw_bytes = json.dumps(payload, indent=2).encode("utf-8")
    path.write_bytes(raw_bytes)
    digest = sha256_bytes(raw_bytes)
    parent_metadata = {
        **entry.parent.metadata,
        "manifest_sha256": digest,
        "size_bytes": len(raw_bytes),
    }
    return entry.model_copy(
        update={
            "parent": entry.parent.model_copy(
                update={
                    "uri": f"artifact://sha256/{digest}",
                    "metadata": parent_metadata,
                }
            )
        }
    )


def test_four_exact_parents_remain_per_run_until_grouped_downstream(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
) -> None:
    entries = [
        _write_exact_parent(
            tmp_path,
            run_id=f"feedbax-training-run:row-{index}",
            row_id=f"row-{index}",
        )
        for index in range(4)
    ]
    exact = _exact_document(*entries)

    execution = execute_staged_analysis_bundle(
        _bundle(include_report=True),
        root=tmp_path,
        exact_parents=exact,
        registries=exact_evaluation_calls.registries,
    )

    assert execution.matched_run_ids == [entry.parent.id for entry in entries]
    assert exact_evaluation_calls == [
        EvaluationRunSpec(
            evaluation_type=EXACT_EVALUATION_TYPE,
            inputs=[entry.parent],
        )
        for entry in entries
    ]
    evaluation_stage = execution.stages[0]
    assert evaluation_stage.inputs == [entry.parent for entry in entries]
    assert len(evaluation_stage.manifest_refs) == 4
    assert len({ref.id for ref in evaluation_stage.manifest_refs}) == 4
    assert all(ref.id for ref in evaluation_stage.manifest_refs)
    for entry, output_ref in zip(entries, evaluation_stage.manifest_refs, strict=True):
        manifest = resolve_manifest_input(output_ref, tmp_path).manifest
        assert manifest.input_training_runs == [entry.parent]
        assert manifest.provenance.parents == [entry.parent]
        assert manifest.evaluation_spec.inline["inputs"] == [entry.parent.model_dump(mode="json")]

    grouped_stage = execution.stages[1]
    assert grouped_stage.inputs == evaluation_stage.manifest_refs
    assert all(ref.kind == "EvaluationRunManifest" for ref in grouped_stage.inputs)
    assert len(grouped_stage.manifest_refs) == 1


def test_diverged_run_admits_certified_checkpoint_and_scopes_evaluation_identity(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _write_diverged_exact_parent(first_root, suffix="-provenance-one")
    checkpoint = first.material_dependencies.dependencies[1].value
    assert isinstance(checkpoint, ParentRef)
    second = _write_diverged_exact_parent(
        second_root,
        suffix="-provenance-two",
        checkpoint=checkpoint,
    )

    first_execution = execute_staged_analysis_bundle(
        _bundle(),
        root=first_root,
        exact_parents=_exact_document(first),
        **_material_dependency_execution_kwargs(first_root),
        registries=exact_evaluation_calls.registries,
    )
    second_execution = execute_staged_analysis_bundle(
        _bundle(),
        root=second_root,
        exact_parents=_exact_document(second),
        **_material_dependency_execution_kwargs(
            second_root,
            checkpoint_root=first_root / "checkpoint-custody",
        ),
        registries=exact_evaluation_calls.registries,
    )

    first_ref = first_execution.stages[0].manifest_refs[0]
    second_ref = second_execution.stages[0].manifest_refs[0]
    assert first_ref.id == second_ref.id
    assert first.parent.metadata["manifest_sha256"] != second.parent.metadata["manifest_sha256"]
    assert (
        exact_evaluation_calls[0].inputs[0].metadata["material_dependency_identity_sha256"]
        == exact_evaluation_calls[1].inputs[0].metadata["material_dependency_identity_sha256"]
    )


def test_material_dependency_dry_run_authenticates_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    entry = _write_diverged_exact_parent(tmp_path)
    checkpoint = entry.material_dependencies.dependencies[1].value
    assert isinstance(checkpoint, ParentRef)
    checkpoint_path = tmp_path / "checkpoint-custody" / str(checkpoint.uri)
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="certified_checkpoint.*unauthentic"):
        dry_run_staged_analysis_bundle(
            _bundle(),
            root=tmp_path,
            exact_parents=_exact_document(entry),
            **_material_dependency_execution_kwargs(tmp_path),
        )

    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()


def test_material_dependency_dry_run_authenticates_provider_artifact_bytes(
    tmp_path: Path,
) -> None:
    provider_root = tmp_path / "artifact-provider"
    provider_root.mkdir(parents=True)
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(),
        explicit_root=provider_root,
    )
    artifact = provider.store_bytes(
        b"certified analysis input",
        role="certified_analysis_input",
        logical_name="certified.bin",
    )
    entry = _write_diverged_exact_parent(tmp_path, checkpoint=artifact)

    result = dry_run_staged_analysis_bundle(
        _bundle(),
        root=tmp_path,
        exact_parents=_exact_document(entry),
        **_artifact_dependency_execution_kwargs(tmp_path),
    )

    assert result.matched_run_ids == [entry.parent.id]
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("uncertified_checkpoint", "certified_checkpoint.*missing"),
        ("missing_checkpoint_bytes", "certified_checkpoint.*missing"),
        ("tampered_checkpoint_bytes", "certified_checkpoint.*unauthentic"),
        ("waiver_manifest", "waiver manifest mismatch"),
        ("waiver_artifact", "waiver artifact hash mismatch"),
    ],
)
def test_material_dependency_admission_rejects_before_outputs(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
    case: str,
    message: str,
) -> None:
    if case.startswith("waiver_"):
        with pytest.raises(ValidationError, match=message):
            _write_diverged_exact_parent(
                tmp_path,
                waiver_manifest_digest="d" * 64 if case == "waiver_manifest" else None,
                waiver_artifact_digest="d" * 64 if case == "waiver_artifact" else None,
            )
    else:
        entry = _write_diverged_exact_parent(
            tmp_path,
            certified=case != "uncertified_checkpoint",
        )
        checkpoint = entry.material_dependencies.dependencies[1].value
        assert isinstance(checkpoint, ParentRef)
        checkpoint_path = tmp_path / "checkpoint-custody" / str(checkpoint.uri)
        if case == "missing_checkpoint_bytes":
            checkpoint_path.unlink()
        elif case == "tampered_checkpoint_bytes":
            checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")
        with pytest.raises(ValueError, match=message):
            execute_staged_analysis_bundle(
                _bundle(),
                root=tmp_path,
                exact_parents=_exact_document(entry),
                **_material_dependency_execution_kwargs(tmp_path),
                registries=exact_evaluation_calls.registries,
            )

    assert exact_evaluation_calls == []
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()


def test_manifest_hash_changes_evaluation_identity_for_same_parent_id(tmp_path: Path) -> None:
    first = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:same",
        row_id="row-same",
        suffix="-first",
    )
    second = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:same",
        row_id="row-same",
        suffix="-second",
    )

    first_id = evaluation_run_manifest_id(
        EvaluationRunSpec(evaluation_type=EXACT_EVALUATION_TYPE, inputs=[first.parent])
    )
    second_id = evaluation_run_manifest_id(
        EvaluationRunSpec(evaluation_type=EXACT_EVALUATION_TYPE, inputs=[second.parent])
    )

    assert first.parent.id == second.parent.id
    assert first.parent.metadata["manifest_sha256"] != second.parent.metadata["manifest_sha256"]
    assert first_id != second_id


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("hash", "SHA-256"),
        ("size", "byte size"),
        ("bool_size", "nonnegative integer"),
        ("uri_hash", "artifact URI digest"),
        ("escape", "escapes"),
        ("encoded_escape", "escapes"),
        ("missing", "does not exist"),
        ("nonregular", "not a regular file"),
        ("kind", "kind"),
        ("role", "role"),
        ("manifest_status_metadata", "manifest_status"),
        ("registration_status", "registration_status"),
        ("conformance", "conformance_overall"),
        ("certificate", "certificate_sha256"),
        ("planned", "planned_run_id"),
        ("loaded_status", "status must be 'completed'"),
        ("loaded_row", "metadata.row_id disagrees"),
    ],
)
def test_exact_parent_preflight_rejects_before_recipe_or_outputs(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
    case: str,
    message: str,
) -> None:
    entry = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:bad",
        row_id="row-bad",
        status="pending" if case == "loaded_status" else "completed",
    )
    payload = entry.model_dump(mode="json")
    parent = payload["parent"]
    metadata = parent["metadata"]
    if case == "hash":
        metadata["manifest_sha256"] = "d" * 64
        parent["uri"] = f"artifact://sha256/{'d' * 64}"
    elif case == "size":
        metadata["size_bytes"] += 1
    elif case == "bool_size":
        metadata["size_bytes"] = True
    elif case == "uri_hash":
        parent["uri"] = f"artifact://sha256/{'d' * 64}"
    elif case == "escape":
        payload["execution_uri"] = "../outside.json"
    elif case == "encoded_escape":
        payload["execution_uri"] = "%2e%2e/outside.json"
    elif case == "missing":
        payload["execution_uri"] = "exact-inputs/missing.json"
    elif case == "nonregular":
        nonregular = tmp_path / "exact-inputs" / "directory.json"
        nonregular.mkdir()
        payload["execution_uri"] = "exact-inputs/directory.json"
    elif case == "kind":
        parent["kind"] = "EvaluationRunManifest"
    elif case == "role":
        parent["role"] = "evaluation_run"
    elif case == "manifest_status_metadata":
        metadata["manifest_status"] = "pending"
    elif case == "registration_status":
        metadata["registration_status"] = "pending"
    elif case == "conformance":
        metadata["conformance_overall"] = "fail"
    elif case == "certificate":
        metadata["certificate_sha256"] = "INVALID"
    elif case == "planned":
        metadata["planned_run_id"] = "different"
    elif case == "loaded_row":
        metadata["row_id"] = "different"

    exact = _exact_document(StagedExactParentEntry.model_validate(payload))
    with pytest.raises(ValueError, match=message):
        execute_staged_analysis_bundle(
            _bundle(),
            root=tmp_path,
            exact_parents=exact,
            registries=exact_evaluation_calls.registries,
        )

    assert exact_evaluation_calls == []
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()


@pytest.mark.parametrize(
    ("field_name", "conflicting_value"),
    [
        ("manifest_status", "failed"),
        ("registration_status", "pending"),
        ("conformance_overall", "fail"),
        ("certificate_sha256", "d" * 64),
    ],
)
def test_exact_parent_rejects_available_governed_manifest_fact_conflicts(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
    field_name: str,
    conflicting_value: str,
) -> None:
    entry = _write_exact_parent(
        tmp_path,
        run_id=f"feedbax-training-run:{field_name}",
        row_id=f"row-{field_name}",
    )
    entry = _rewrite_manifest_metadata(
        tmp_path,
        entry,
        field_name=field_name,
        value=conflicting_value,
    )

    with pytest.raises(ValueError, match=field_name):
        execute_staged_analysis_bundle(
            _bundle(),
            root=tmp_path,
            exact_parents=_exact_document(entry),
            registries=exact_evaluation_calls.registries,
        )

    assert exact_evaluation_calls == []
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("ref", "duplicate ParentRef"),
        ("id", "duplicate ParentRef id"),
        ("location", "duplicate execution location"),
    ],
)
def test_exact_parent_preflight_rejects_duplicate_membership(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
    case: str,
    message: str,
) -> None:
    first = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:first",
        row_id="row-first",
    )
    if case == "ref":
        second = first
    elif case == "id":
        second_payload = _write_exact_parent(
            tmp_path,
            run_id=first.parent.id,
            row_id="row-second",
        ).model_dump(mode="json")
        second_payload["parent"]["metadata"]["certificate_sha256"] = "e" * 64
        second = StagedExactParentEntry.model_validate(second_payload)
    else:
        second_payload = _write_exact_parent(
            tmp_path,
            run_id="feedbax-training-run:second",
            row_id="row-second",
        ).model_dump(mode="json")
        second_payload["execution_uri"] = first.execution_uri
        second = StagedExactParentEntry.model_validate(second_payload)

    with pytest.raises(ValueError, match=message):
        execute_staged_analysis_bundle(
            _bundle(),
            root=tmp_path,
            exact_parents=_exact_document(first, second),
            registries=exact_evaluation_calls.registries,
        )

    assert exact_evaluation_calls == []
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()


def test_exact_parent_preflight_rejects_symlink(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
) -> None:
    entry = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:symlink",
        row_id="row-symlink",
    )
    link_path = tmp_path / "exact-inputs" / "linked.json"
    link_path.symlink_to(Path(entry.execution_uri).name)
    entry = entry.model_copy(update={"execution_uri": "exact-inputs/linked.json"})

    with pytest.raises(ValueError, match="symlink"):
        execute_staged_analysis_bundle(
            _bundle(),
            root=tmp_path,
            exact_parents=_exact_document(entry),
            registries=exact_evaluation_calls.registries,
        )

    assert exact_evaluation_calls == []
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()


@pytest.mark.parametrize("relation", ["missing", "extra"])
def test_exact_parent_predicate_cannot_change_membership(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
    relation: str,
) -> None:
    entries = [
        _write_exact_parent(
            tmp_path,
            run_id=f"feedbax-training-run:{row_id}",
            row_id=row_id,
        )
        for row_id in ("a", "b")
    ]
    predicate_ids = [entries[0].parent.id]
    if relation == "extra":
        predicate_ids = [*predicate_ids, entries[1].parent.id, "unlisted"]
    predicate = ManifestPredicate(
        manifest_kind="TrainingRunManifest",
        run_ids=predicate_ids,
    )

    with pytest.raises(ValueError, match="cannot add, remove, or narrow"):
        execute_staged_analysis_bundle(
            _bundle(predicate=predicate),
            root=tmp_path,
            exact_parents=_exact_document(*entries),
            registries=exact_evaluation_calls.registries,
        )

    assert exact_evaluation_calls == []
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()


def test_exact_parent_predicate_rejects_top_k_and_nonmatching_terms(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
) -> None:
    entry = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:predicate",
        row_id="row-predicate",
    )
    exact = _exact_document(entry)
    top_k = ManifestPredicate(
        manifest_kind="TrainingRunManifest",
        top_k_by_metric_per_group=TopKByMetricPerGroup(
            metric_path="summary_metrics.loss",
            group_by_path="metadata.method",
        ),
    )
    with pytest.raises(ValueError, match="top_k"):
        execute_staged_analysis_bundle(
            _bundle(predicate=top_k),
            root=tmp_path,
            exact_parents=exact,
            registries=exact_evaluation_calls.registries,
        )

    mismatch = ManifestPredicate(
        manifest_kind="TrainingRunManifest",
        metadata_equals={"method": "wrong"},
    )
    with pytest.raises(ValueError, match="does not satisfy"):
        execute_staged_analysis_bundle(
            _bundle(predicate=mismatch),
            root=tmp_path,
            exact_parents=exact,
            registries=exact_evaluation_calls.registries,
        )
    assert exact_evaluation_calls == []


def test_exact_parent_root_evaluation_cannot_group_parents(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
) -> None:
    entry = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:grouped-root",
        row_id="row-grouped-root",
    )
    bundle = _bundle()
    bundle = bundle.model_copy(
        update={"stages": [bundle.stages[0].model_copy(update={"mode": "grouped"})]}
    )

    with pytest.raises(ValueError, match="grouping is only valid downstream"):
        execute_staged_analysis_bundle(
            bundle,
            root=tmp_path,
            exact_parents=_exact_document(entry),
            registries=exact_evaluation_calls.registries,
        )

    assert exact_evaluation_calls == []
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()


def test_exact_parents_require_explicit_root_and_exclude_run_ids(
    tmp_path: Path, application_registry_bundle
) -> None:
    entry = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:exclusive",
        row_id="row-exclusive",
    )
    exact = _exact_document(entry)

    with pytest.raises(ValueError, match="explicit manifest root"):
        execute_staged_analysis_bundle(
            _bundle(),
            exact_parents=exact,
            registries=application_registry_bundle,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        execute_staged_analysis_bundle(
            _bundle(),
            root=tmp_path,
            run_ids=[entry.parent.id],
            exact_parents=exact,
            registries=application_registry_bundle,
        )


def test_staged_exact_parent_schema_is_versioned_and_nonempty() -> None:
    assert STAGED_EXACT_PARENTS_SCHEMA_ID == "feedbax.spec.staged_exact_parents"
    assert STAGED_EXACT_PARENTS_SCHEMA_VERSION == "feedbax.spec.staged_exact_parents.v2"
    with pytest.raises(ValidationError, match="at least 1 item"):
        StagedExactParents(
            schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
            schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
            parents=[],
        )
    for missing_field in ("schema_id", "schema_version"):
        payload = {
            "schema_id": STAGED_EXACT_PARENTS_SCHEMA_ID,
            "schema_version": STAGED_EXACT_PARENTS_SCHEMA_VERSION,
            "parents": [{}],
        }
        payload.pop(missing_field)
        with pytest.raises(ValidationError, match=missing_field):
            StagedExactParents.model_validate(payload)
    with pytest.raises(ValidationError, match="staged_exact_parents.v2"):
        StagedExactParents.model_validate(
            {
                "schema_id": STAGED_EXACT_PARENTS_SCHEMA_ID,
                "schema_version": STAGED_EXACT_PARENTS_SCHEMA_VERSION_V1,
                "parents": [{}],
            }
        )


def test_executor_revalidates_mutated_exact_parent_schema(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
) -> None:
    entry = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:mutated-schema",
        row_id="row-mutated-schema",
    )
    exact = _exact_document(entry)
    exact.schema_version = "feedbax.spec.staged_exact_parents.future"  # type: ignore[assignment]

    with pytest.raises(ValidationError, match="staged_exact_parents.v2"):
        execute_staged_analysis_bundle(
            _bundle(),
            root=tmp_path,
            exact_parents=exact,
            registries=exact_evaluation_calls.registries,
        )

    assert exact_evaluation_calls == []
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()


def test_exact_mode_uses_explicit_location_without_ambient_same_id_discovery(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
) -> None:
    entry = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:ambient-duplicate",
        row_id="row-authoritative",
    )
    write_manifest(
        TrainingRunManifest(
            id=entry.parent.id,
            status="completed",
            run_set_id="different-run-set",
            metadata={"method": "wrong", "row_id": "ambient"},
        ),
        root=tmp_path,
    )
    execution_ref = entry.parent.model_copy(update={"uri": entry.execution_uri})
    spec = EvaluationRunSpec(
        evaluation_type=EXACT_EVALUATION_TYPE,
        inputs=[execution_ref],
    )

    with pytest.raises(EvaluationInputAmbiguityError):
        resolve_evaluation_inputs(spec, manifest_root=tmp_path)
    resolved = resolve_evaluation_inputs(
        spec,
        manifest_root=tmp_path,
        require_unique_manifest_id=False,
    )[0]
    assert resolved.ref == execution_ref
    assert resolved.sha256 == entry.parent.metadata["manifest_sha256"]
    assert resolved.size_bytes == entry.parent.metadata["size_bytes"]

    execution = execute_staged_analysis_bundle(
        _bundle(),
        root=tmp_path,
        exact_parents=_exact_document(entry),
        registries=exact_evaluation_calls.registries,
    )
    assert execution.matched_run_ids == [entry.parent.id]
    assert exact_evaluation_calls[0].inputs == [entry.parent]


def test_staged_cli_exact_parent_round_trip(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
    monkeypatch,
    capsys,
) -> None:
    from feedbax.bin import analysis as analysis_cli

    entry = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:cli",
        row_id="row-cli",
    )
    exact = _exact_document(entry)
    exact_path = tmp_path / "exact-parents.json"
    exact_path.write_text(exact.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.setattr(
        analysis_cli,
        "load_analysis_bundle",
        lambda *_args, **_kwargs: _bundle(),
    )

    async def compose_application(**_kwargs):
        return BootstrapState(exact_evaluation_calls.registries, ())

    monkeypatch.setattr(analysis_cli, "compose_application", compose_application)

    analysis_cli.main(
        [
            "--bundle",
            "test/exact",
            "--manifest-root",
            str(tmp_path),
            "--exact-parents",
            str(exact_path),
            "--fig-dump-dir",
            str(tmp_path / "figures"),
            "--fig-dump-formats",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_id"] == "feedbax.manifest.analysis_bundle_execution"
    assert payload["matched_run_ids"] == [entry.parent.id]
    assert payload["stages"][0]["inputs"] == [
        entry.parent.model_dump(mode="json", exclude_none=True)
    ]
    assert exact_evaluation_calls[0].inputs == [entry.parent]


def test_staged_cli_rejects_runs_conflict_and_unversioned_exact_document(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
    monkeypatch,
) -> None:
    from feedbax.bin import analysis as analysis_cli

    with pytest.raises(SystemExit):
        analysis_cli.build_arg_parser().parse_args(
            [
                "--bundle",
                "test/exact",
                "--runs",
                "run-a",
                "--exact-parents",
                "parents.json",
            ]
        )

    entry = _write_exact_parent(
        tmp_path,
        run_id="feedbax-training-run:unversioned",
        row_id="row-unversioned",
    )
    payload = _exact_document(entry).model_dump(mode="json")
    payload.pop("schema_version")
    exact_path = tmp_path / "unversioned.json"
    exact_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        analysis_cli,
        "load_analysis_bundle",
        lambda *_args, **_kwargs: _bundle(),
    )

    with pytest.raises(ValueError, match="requires explicit schema_id and schema_version"):
        analysis_cli.main(
            [
                "--bundle",
                "test/exact",
                "--manifest-root",
                str(tmp_path),
                "--exact-parents",
                str(exact_path),
            ]
        )
    assert exact_evaluation_calls == []
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()

    payload = _exact_document(entry).model_dump(mode="json")
    payload["schema_version"] = "feedbax.spec.staged_exact_parents.unknown"
    exact_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported StagedExactParents schema_version"):
        analysis_cli.main(
            [
                "--bundle",
                "test/exact",
                "--manifest-root",
                str(tmp_path),
                "--exact-parents",
                str(exact_path),
            ]
        )
    assert exact_evaluation_calls == []
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()


def test_legacy_root_and_run_ids_remain_dynamic(
    tmp_path: Path,
    exact_evaluation_calls: list[EvaluationRunSpec],
) -> None:
    training = TrainingRunManifest(
        id="feedbax-training-run:legacy",
        status="completed",
        metadata={"method": "minimax"},
    )
    write_manifest(training, root=tmp_path)

    execution = execute_staged_analysis_bundle(
        _bundle(),
        root=tmp_path,
        run_ids=[training.id],
        registries=exact_evaluation_calls.registries,
    )

    assert execution.matched_run_ids == [training.id]
    assert exact_evaluation_calls[0].inputs == [
        ParentRef(
            kind="TrainingRunManifest",
            id=training.id,
            role="training_run",
        )
    ]
