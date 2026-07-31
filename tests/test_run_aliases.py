"""Compile-time symbolic aliases for authenticated run references."""

from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.analysis.specs import (
    AnalysisRecipeExecutionError,
    AnalysisRecipeResult,
    resolve_analysis_run_authoring,
)
from feedbax.bin import analysis as analysis_cli
from feedbax.contracts.analysis_composition import (
    AnalysisRunDeltaSpec,
    analysis_composition_provenance,
)
from feedbax.contracts.manifest import (
    ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID,
    ANALYSIS_RUN_DELTA_SPEC_SCHEMA_VERSION,
    AnalysisRunSpec,
    ParentRef,
    canonical_json_bytes,
    load_manifest,
    sha256_bytes,
)
from feedbax.contracts.run_aliases import (
    RUN_ALIAS_CATALOG_SCHEMA_ID,
    RUN_ALIAS_CATALOG_SCHEMA_VERSION,
    RUN_ALIAS_REF_SCHEMA_ID,
    RUN_ALIAS_REF_SCHEMA_VERSION,
    RunAliasCatalog,
    resolve_run_aliases,
)
from feedbax.plugins import (
    ANALYSIS_RECIPES,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
    bootstrap_application,
    new_registration_context,
)
from tests.analysis_fixtures import (
    ToyAnalysis,
    build_toy_analysis_data,
    execute_toy_evaluation,
)

pytestmark = [pytest.mark.feedbax_contract]


def _analysis_state(analysis_type: str, recipe):
    def register(context) -> None:
        context.registry(ANALYSIS_RECIPES).register(analysis_type, recipe)

    return asyncio.run(
        bootstrap_application(
            new_registration_context(local_component_source=None),
            registrations=(
                PluginRegistration(
                    PluginDeclaration(
                        "tests.run_aliases",
                        "1",
                        families=(FamilyRequirement("analysis_recipes"),),
                    ),
                    register,
                ),
            ),
        )
    )


def _authenticated_parent(*, digest: str = "a" * 64) -> ParentRef:
    return ParentRef(
        kind="EvaluationRunManifest",
        id="feedbax-evaluation-run:example",
        role="evaluation_run",
        metadata={
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": digest,
            "size_bytes": 123,
        },
    )


def _alias_ref(alias: str) -> dict[str, str]:
    return {
        "schema_id": RUN_ALIAS_REF_SCHEMA_ID,
        "schema_version": RUN_ALIAS_REF_SCHEMA_VERSION,
        "alias": alias,
    }


def _catalog(*aliases: dict[str, object]) -> dict[str, object]:
    return {
        "schema_id": RUN_ALIAS_CATALOG_SCHEMA_ID,
        "schema_version": RUN_ALIAS_CATALOG_SCHEMA_VERSION,
        "aliases": list(aliases),
    }


def test_alias_catalog_expands_chains_once_without_changing_direct_refs() -> None:
    parent = _authenticated_parent()
    direct = ParentRef(kind="Note", id="direct", role="annotation")
    payload = {
        "inputs": [_alias_ref("selected"), direct.model_dump(mode="json")],
        "nested": {"input": _alias_ref("source")},
    }
    catalog = _catalog(
        {"alias": "source", "target": parent.model_dump(mode="json")},
        {"alias": "selected", "target": _alias_ref("source")},
    )

    resolved = resolve_run_aliases(payload, [catalog])

    expected = parent.model_dump(mode="json", exclude_none=True)
    assert resolved["inputs"] == [expected, direct.model_dump(mode="json")]
    assert resolved["nested"]["input"] == expected
    assert RUN_ALIAS_REF_SCHEMA_ID not in json.dumps(resolved, sort_keys=True)


@pytest.mark.parametrize(
    ("catalogs", "message"),
    [
        (
            [_catalog({"alias": "other", "target": _authenticated_parent().model_dump()})],
            "not declared",
        ),
        (
            [
                _catalog({"alias": "same", "target": _authenticated_parent().model_dump()}),
                _catalog({"alias": "same", "target": _authenticated_parent().model_dump()}),
            ],
            "ambiguous across supplied catalogs",
        ),
        (
            [
                _catalog(
                    {"alias": "a", "target": _alias_ref("b")},
                    {"alias": "b", "target": _alias_ref("a")},
                )
            ],
            "cycle detected",
        ),
        (
            [
                _catalog(
                    {
                        "alias": "missing-pin",
                        "target": ParentRef(
                            kind="EvaluationRunManifest",
                            id="run",
                            role="evaluation_run",
                        ).model_dump(),
                    }
                )
            ],
            "must be an authenticated manifest ParentRef",
        ),
    ],
)
def test_alias_resolution_fails_closed(
    catalogs: list[dict[str, object]],
    message: str,
) -> None:
    alias = (
        "missing"
        if "not declared" in message
        else ("a" if "cycle" in message else "same" if "ambiguous" in message else "missing-pin")
    )
    with pytest.raises(ValueError, match=message):
        resolve_run_aliases({"input": _alias_ref(alias)}, catalogs)


@pytest.mark.parametrize(
    "catalog",
    [
        _catalog(
            {"alias": "a", "target": _alias_ref("b")},
            {"alias": "b", "target": _alias_ref("a")},
        ),
        _catalog(
            {
                "alias": "unused",
                "target": ParentRef(
                    kind="EvaluationRunManifest",
                    id="run",
                    role="evaluation_run",
                ).model_dump(),
            }
        ),
        _catalog({"alias": "unused", "target": _alias_ref("missing")}),
    ],
)
def test_unused_invalid_alias_declarations_fail_closed(
    catalog: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        resolve_run_aliases({"inputs": []}, [catalog])


def test_catalog_schema_rejects_unsupported_version_and_duplicate_names() -> None:
    parent = _authenticated_parent().model_dump()
    unsupported = _catalog({"alias": "run", "target": parent})
    unsupported["schema_version"] = "feedbax.spec.run_alias_catalog.v0"
    with pytest.raises(ValidationError, match="schema_version"):
        RunAliasCatalog.model_validate(unsupported)

    with pytest.raises(ValidationError, match="ambiguous aliases"):
        RunAliasCatalog.model_validate(
            _catalog(
                {"alias": "run", "target": parent},
                {"alias": "run", "target": parent},
            )
        )


def test_delta_compilation_resolves_alias_before_final_identity_and_provenance(
    tmp_path: Path,
) -> None:
    parent = _authenticated_parent()
    base = {
        "schema_id": "feedbax.spec.analysis_run",
        "schema_version": "feedbax.spec.analysis_run.v2",
        "analysis_type": "feedbax.example.analysis",
        "inputs": [_alias_ref("source-run")],
        "params": {"value": 1},
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    delta = AnalysisRunDeltaSpec.model_validate(
        {
            "schema_id": ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID,
            "schema_version": ANALYSIS_RUN_DELTA_SPEC_SCHEMA_VERSION,
            "parent": {
                "ref": base_path.name,
                "sha256": sha256_bytes(canonical_json_bytes(base)),
            },
            "deltas": [
                {
                    "layer_id": "variant",
                    "patches": [{"op": "replace", "path": "params.value", "value": 2}],
                }
            ],
        }
    )

    run_spec, flattened = resolve_analysis_run_authoring(
        delta,
        repo_root=tmp_path,
        run_alias_catalogs=[
            _catalog(
                {
                    "alias": "source-run",
                    "target": parent.model_dump(mode="json", exclude_none=True),
                }
            )
        ],
    )

    assert flattened is not None
    expected = parent.model_dump(mode="json", exclude_none=True)
    assert run_spec.inputs[0].model_dump(mode="json", exclude_none=True) == expected
    assert flattened.payload["inputs"] == [expected]
    assert RUN_ALIAS_REF_SCHEMA_ID not in json.dumps(flattened.payload, sort_keys=True)
    assert analysis_composition_provenance(flattened)["flattened_spec_sha256"] == sha256_bytes(
        canonical_json_bytes(flattened.payload)
    )


def test_analysis_cli_expands_alias_to_durable_pin(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis_type = "feedbax.test.run_alias_cli"
    evaluation, evaluation_path = execute_toy_evaluation(tmp_path)
    parent = authenticated_manifest_ref(evaluation, evaluation_path, "evaluation_run")
    spec_path = tmp_path / "analysis.json"
    catalog_path = tmp_path / "run-aliases.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_id": "feedbax.spec.analysis_run",
                "schema_version": "feedbax.spec.analysis_run.v2",
                "analysis_type": analysis_type,
                "inputs": [_alias_ref("source-run")],
                "params": {"requested_outputs": ["toy"], "value": 4},
            }
        ),
        encoding="utf-8",
    )
    catalog_path.write_text(
        json.dumps(
            _catalog(
                {
                    "alias": "source-run",
                    "target": parent.model_dump(mode="json", exclude_none=True),
                }
            )
        ),
        encoding="utf-8",
    )

    def recipe(spec: AnalysisRunSpec, _root: Path, _inputs, _execution_context):
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=int(spec.params["value"])),
        )

    analysis_cli.main(
        [
            "run",
            str(spec_path),
            "--run-aliases",
            str(catalog_path),
            "--root",
            str(tmp_path),
        ],
        bootstrap_state=_analysis_state(analysis_type, recipe),
    )

    result = json.loads(capsys.readouterr().out)
    manifest = load_manifest(result["manifest_path"])
    expected = parent.model_dump(mode="json", exclude_none=True)
    assert manifest.analysis_spec.inline["inputs"] == [expected]
    assert [item.model_dump(mode="json", exclude_none=True) for item in manifest.inputs] == [
        expected
    ]
    assert RUN_ALIAS_REF_SCHEMA_ID not in manifest.model_dump_json()


def test_analysis_cli_rejects_alias_target_whose_manifest_bytes_drifted(
    tmp_path: Path,
) -> None:
    evaluation, evaluation_path = execute_toy_evaluation(tmp_path)
    parent = authenticated_manifest_ref(evaluation, evaluation_path, "evaluation_run")
    drifted = parent.model_copy(
        update={
            "metadata": {
                **parent.metadata,
                "manifest_sha256": "0" * 64,
            }
        }
    )
    spec_path = tmp_path / "analysis.json"
    catalog_path = tmp_path / "run-aliases.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_id": "feedbax.spec.analysis_run",
                "schema_version": "feedbax.spec.analysis_run.v2",
                "analysis_type": "feedbax.test.alias_drift",
                "inputs": [_alias_ref("source-run")],
            }
        ),
        encoding="utf-8",
    )
    catalog_path.write_text(
        json.dumps(
            _catalog(
                {
                    "alias": "source-run",
                    "target": drifted.model_dump(mode="json", exclude_none=True),
                }
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(AnalysisRecipeExecutionError) as excinfo:
        analysis_cli.main(
            [
                "run",
                str(spec_path),
                "--run-aliases",
                str(catalog_path),
                "--root",
                str(tmp_path),
            ],
            bootstrap_state=_analysis_state(
                "feedbax.test.alias_drift",
                lambda *_args: AnalysisRecipeResult(),
            ),
        )
    assert "SHA-256 mismatch" in str(excinfo.value.__cause__)
