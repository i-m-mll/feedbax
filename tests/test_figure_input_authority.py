from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.bundles import (
    AnalysisBundleSpec,
    BundleStageSpec,
    execute_staged_analysis_bundle,
)
from feedbax.analysis.execution_context import (
    StagedArtifactProviderRootBinding,
    StagedExecutionContext,
    StagedParentExecutionLocation,
)
from feedbax.analysis.figures import (
    FigureInputAuthorityError,
    _figure_expression_context,
    execute_figure_spec,
    resolve_figure_inputs,
)
from feedbax.analysis.manifest_inputs import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    resolve_manifest_input,
)
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    register_analysis_recipe,
    unregister_analysis_recipe,
)
from feedbax.contracts.figures import (
    FigureArtifactPayload,
    FigureInputAuthority,
    FigureSpec,
)
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    ParentRef,
    canonical_json_bytes,
    sha256_bytes,
    spec_payload,
)
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.selection import ManifestPredicate
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedExecutionDescriptor,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from tests.analysis_fixtures import build_toy_analysis_data


_STAGED_PROVIDER_ANALYSIS_TYPE = "feedbax.test.staged_figure_provider"


class _ProviderArtifactAnalysis(AbstractAnalysis):
    """Expose one provider-backed ref for manifest recording."""

    def compute(self, data, artifact, **kwargs):
        return {"artifact": artifact}


def _authority_case(tmp_path: Path):
    provider = ImmutableArtifactBlobProvider(tmp_path / "provider")
    certificate = {
        "schema_id": "rlrmp.bridge.certificate",
        "schema_version": "rlrmp.bridge.certificate.v1",
        "values": [1, 2, 3],
    }
    artifact = provider.store_bytes(
        json.dumps(certificate).encode(),
        role="rlrmp-bridge-standard-certificate",
        logical_name="certificate.json",
        media_type="application/json",
    )
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:figure-authority",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "test.figure.authority", "inputs": [], "params": {}},
        ),
        artifacts=[artifact],
        metadata={"uri": "/must/not/reach/constructor", "root": str(tmp_path)},
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_artifact = provider.store_bytes(
        manifest_bytes,
        role="analysis_manifest",
        logical_name="manifest.json",
        media_type="application/json",
    )
    parent = ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="grouped_analysis",
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": sha256_bytes(manifest_bytes),
            "size_bytes": len(manifest_bytes),
        },
    )
    context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"certificates": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=parent,
                root=provider.root,
                execution_uri=str(provider.canonical_relative_path(manifest_artifact)),
                artifact_provider="certificates",
            ),
        ),
    )
    selector = FigureArtifactPayload(
        name="certificate",
        manifest_role="grouped_analysis",
        artifact_role="rlrmp-bridge-standard-certificate",
        artifact_provider="certificates",
        payload_schema_id="rlrmp.bridge.certificate",
        payload_schema_version="rlrmp.bridge.certificate.v1",
    )
    spec = FigureSpec(
        name="authority",
        assembler="feedbax.grid_figure",
        inputs=[parent],
        input_authorities=[
            FigureInputAuthority(parent=parent, artifact_payloads=[selector])
        ],
    )
    return provider, artifact, certificate, context, spec


def test_provider_canonical_payload_is_decoded_and_constructor_context_is_sanitized(
    tmp_path: Path,
) -> None:
    _provider, artifact, certificate, context, spec = _authority_case(tmp_path)
    resolved = resolve_figure_inputs(spec, execution_context=context)

    assert resolved[0].artifact_payloads == {"certificate": certificate}
    assert resolved[0].artifact_refs == (artifact,)
    expression_context = _figure_expression_context(spec, resolved)
    payload = expression_context.items["grouped_analysis"].payload
    assert payload["artifact_payloads"]["certificate"] == certificate
    serialized = json.dumps(payload)
    assert str(tmp_path) not in serialized
    assert '"uri"' not in serialized
    assert '"root"' not in serialized


def test_direct_execution_records_exact_consumed_artifact(tmp_path: Path) -> None:
    _provider, artifact, _certificate, context, spec = _authority_case(tmp_path)
    manifest, path = execute_figure_spec(
        spec,
        root=tmp_path / "outputs",
        execution_context=context,
    )
    assert path.is_file()
    assert manifest.regeneration_specs == [artifact]


def test_staged_figure_resolves_prior_stage_output_through_executor_context(
    tmp_path: Path,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "provider")
    certificate = {
        "schema_id": "rlrmp.bridge.certificate",
        "schema_version": "rlrmp.bridge.certificate.v1",
        "values": [1, 2, 3],
    }
    artifact = provider.store_bytes(
        json.dumps(certificate).encode(),
        role="rlrmp-bridge-standard-certificate",
        logical_name="certificate.json",
        media_type="application/json",
    )

    def recipe(_spec, _root, _inputs, _execution_context):
        return AnalysisRecipeResult(
            analyses={
                "provider": _ProviderArtifactAnalysis(variant="provider", cache_result=True)
            },
            data=build_toy_analysis_data(value=0),
            common_inputs={"artifact": artifact},
        )

    stage_one = BundleStageSpec(
        name="produce",
        kind="analysis",
        mode="grouped",
        analysis_type=_STAGED_PROVIDER_ANALYSIS_TYPE,
        requested_outputs=["provider"],
    )
    descriptor = StagedExecutionDescriptor(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={"certificates": ImmutableArtifactBlobProviderSpec()},
        checkpoint_custody={},
    )
    bindings = [StagedArtifactProviderRootBinding("certificates", provider.root)]
    root = tmp_path / "outputs"
    register_analysis_recipe(_STAGED_PROVIDER_ANALYSIS_TYPE, recipe, replace=True)
    try:
        first_execution = execute_staged_analysis_bundle(
            AnalysisBundleSpec(
                name="authority-producer",
                predicate=ManifestPredicate(manifest_kind="TrainingRunManifest"),
                stages=[stage_one],
            ),
            root=root,
            execution_descriptor=descriptor,
            artifact_provider_bindings=bindings,
        )
        produced_parent = first_execution.stages[0].manifest_refs[0]
        selector = FigureArtifactPayload(
            name="certificate",
            manifest_role="analysis_run",
            artifact_role="rlrmp-bridge-standard-certificate",
            artifact_provider="certificates",
            payload_schema_id="rlrmp.bridge.certificate",
            payload_schema_version="rlrmp.bridge.certificate.v1",
        )

        def bundle_for(parent: ParentRef) -> AnalysisBundleSpec:
            figure = FigureSpec(
                name="authority",
                assembler="feedbax.grid_figure",
                inputs=[parent],
                input_authorities=[
                    FigureInputAuthority(parent=parent, artifact_payloads=[selector])
                ],
            )
            return AnalysisBundleSpec(
                name="authority-bundle",
                predicate=ManifestPredicate(manifest_kind="TrainingRunManifest"),
                stages=[
                    stage_one,
                    BundleStageSpec(
                        name="figure",
                        kind="figure",
                        depends_on=["produce"],
                        figure=figure,
                    ),
                ],
            )

        execution = execute_staged_analysis_bundle(
            bundle_for(produced_parent),
            root=root,
            execution_descriptor=descriptor,
            artifact_provider_bindings=bindings,
        )
        assert execution.stages[0].manifest_refs == [produced_parent]
        assert execution.stages[1].inputs == [produced_parent]
        staged_manifest = resolve_manifest_input(
            execution.stages[1].manifest_refs[0], root
        ).manifest
        assert artifact in staged_manifest.regeneration_specs

        missing_parent = produced_parent.model_copy(
            update={"id": f"{produced_parent.id}:missing"}
        )
        with pytest.raises(
            FigureInputAuthorityError,
            match="authority rejected exact parent",
        ):
            execute_staged_analysis_bundle(
                bundle_for(missing_parent),
                root=root,
                execution_descriptor=descriptor,
                artifact_provider_bindings=bindings,
            )
    finally:
        unregister_analysis_recipe(_STAGED_PROVIDER_ANALYSIS_TYPE)


def test_contained_manifest_hardlink_precedes_direct_figure_effects(tmp_path: Path) -> None:
    provider, _artifact, _certificate, context, spec = _authority_case(tmp_path)
    parent = spec.inputs[0]
    source = tmp_path / "contained" / "manifest.json"
    source.parent.mkdir()
    source.write_bytes(context.resolve_manifest_input(parent).raw_bytes)
    os.link(source, tmp_path / "manifest-hardlink.json")
    contained_context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"certificates": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=parent,
                root=source.parent,
                execution_uri=source.name,
            ),
        ),
    )
    output_root = tmp_path / "outputs"

    with pytest.raises(FigureInputAuthorityError, match="authority rejected exact parent"):
        execute_figure_spec(spec, root=output_root, execution_context=contained_context)
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "is missing"),
        ("duplicate", "is duplicated"),
        ("wrong_role", "is missing"),
        ("wrong_media", "media type mismatch"),
        ("wrong_schema", "schema_id mismatch"),
        ("wrong_provider", "provider rejected"),
    ],
)
def test_semantic_selector_failures_precede_figure_effects(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    provider, artifact, _certificate, context, spec = _authority_case(tmp_path)
    selector = spec.input_authorities[0].artifact_payloads[0]
    if mutation == "missing":
        selector = selector.model_copy(update={"artifact_role": "absent"})
    elif mutation == "wrong_role":
        selector = selector.model_copy(update={"manifest_role": "wrong"})
        message = "manifest role mismatch"
    elif mutation == "wrong_media":
        selector = selector.model_copy(update={"media_type": "application/cbor"})
    elif mutation == "wrong_schema":
        selector = selector.model_copy(update={"payload_schema_id": "wrong.schema"})
    elif mutation == "wrong_provider":
        selector = selector.model_copy(update={"artifact_provider": "wrong"})
    elif mutation == "duplicate":
        duplicate = artifact.model_copy(update={"logical_name": "duplicate.json"})
        resolved_manifest = context.resolve_manifest_input(spec.inputs[0]).manifest
        replaced = resolved_manifest.model_copy(
            update={"artifacts": [artifact, duplicate]}, deep=True
        )
        raw = canonical_json_bytes(replaced)
        stored = provider.store_bytes(
            raw,
            role="analysis_manifest",
            logical_name="duplicate-manifest.json",
            media_type="application/json",
        )
        parent = spec.inputs[0].model_copy(
            update={
                "metadata": {
                    **spec.inputs[0].metadata,
                    "manifest_sha256": sha256_bytes(raw),
                    "size_bytes": len(raw),
                }
            }
        )
        context = StagedExecutionContext(
            descriptor=None,
            opened_artifact_providers={"certificates": provider},
            checkpoint_custody_roots={},
            parent_execution_locations=(
                StagedParentExecutionLocation(
                    parent=parent,
                    root=provider.root,
                    execution_uri=str(provider.canonical_relative_path(stored)),
                    artifact_provider="certificates",
                ),
            ),
        )
        spec = spec.model_copy(
            update={
                "inputs": [parent],
                "input_authorities": [
                    FigureInputAuthority(parent=parent, artifact_payloads=[selector])
                ],
            },
            deep=True,
        )
    if mutation != "duplicate":
        spec = spec.model_copy(
            update={
                "input_authorities": [
                    FigureInputAuthority(parent=spec.inputs[0], artifact_payloads=[selector])
                ]
            },
            deep=True,
        )
    output_root = tmp_path / "outputs"
    with pytest.raises(FigureInputAuthorityError, match=message):
        execute_figure_spec(spec, root=output_root, execution_context=context)
    assert not output_root.exists()
