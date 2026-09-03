"""A root figure's inputs, from what an envelope states to what execution reads.

A root figure used to compile with its inputs *recorded and unbound*: the lock
carried one typed reference per input role, the compiled ``FigureSpec`` was
emitted with empty ``inputs`` and empty ``input_authorities``, and nothing
anywhere said which artifact of the bound manifest the figure was entitled to
read. The figure then rendered, successfully, holding none of the data it named.

The claims under test are:

* an envelope at ``feedbax.experiment_envelope.v5`` states each root figure
  input's closed artifact contract — artifact role, provider, media type, decoded
  payload identity, and an explicit payload name — and the compile carries that
  contract into the compile lock, at
  current compile-lock grammar, without moving anything into the
  figure's own scientific identity;
* fulfillment builds the runtime
  :class:`~feedbax.contracts.figures.FigureInputRoleAuthority` from that contract
  and from nothing else, addressing the exact parent the same lowering bound and
  located;
* figure execution then reaches each declared payload exactly, and every way the
  declaration can disagree with the produced manifest — the manifest role, the
  provider, the artifact role, the media type, the decoded payload schema, a
  parent that is missing, a parent that is ambiguous — is a refusal before any
  render effect;
* a figure input bound under a grammar that could not state a contract is refused
  by name, with the re-authoring it needs, rather than rendered unbound.

The figure, the roles, and the artifacts are ``quillon``'s invented vocabulary,
because none of this is about any particular science.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.analysis.execution_context import (
    StagedExecutionContext,
    StagedParentArtifactProviderBinding,
    StagedParentExecutionLocation,
)
from feedbax.analysis.figures import (
    FigureInputAuthorityError,
    plan_figure_execution,
    resolve_figure_inputs,
)
from feedbax.analysis.fulfillment_adapters import FulfillmentEnvironment
from feedbax.workflow.derivation import (
    derive_workflow_plan,
    read_compiled_outputs,
)
from feedbax.workflow.execution import workflow_requests, prepare_workflow
from feedbax.workflow.operation_execution import NodeLoweringError
from feedbax.workflow.plan import LogicalKey
from feedbax.contracts.experiment_envelope import ExperimentEnvelopeRejection
from feedbax.contracts.experiment_envelope_dialect import (
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V4,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V5,
    EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_ID,
    EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_VERSION,
)
from feedbax.contracts.figures import FigureInputRoleAuthority
from feedbax.contracts.base import (
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    spec_payload,
)
from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.envelope import kernel_for
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

from tests.fake_project_experiment import (
    PROJECT_DECLARATION,
    envelope_path,
    write_repo,
)

#: The two artifacts one produced analysis run retains, and what each decodes to.
#: They differ in role and in payload schema, which is what makes "the selector
#: reached the right one" a claim rather than a coincidence.
RETAINED_PAYLOADS: dict[str, dict[str, Any]] = {
    "result": {"schema_id": "quillon.span_result", "schema_version": "quillon.span_result.v1"},
    "report": {"schema_id": "quillon.span_report", "schema_version": "quillon.span_report.v1"},
}

MANIFEST_ID = "feedbax-analysis-run:quillon-span"
AUTHORITY_REF = "authorities/rooted-span.json"
FIGURE_NAME = "rooted-span"

#: Which figure input reads which retained artifact. The payload name is stated
#: rather than read off the role, so the two deliberately differ: a figure whose
#: traces name ``summary`` keeps naming it when the role is renamed.
ROOT_INPUTS: dict[str, dict[str, str]] = {
    "primary_span": {"artifact_role": "result", "payload_name": "summary"},
    "narrative_span": {"artifact_role": "report", "payload_name": "narrative"},
}


def _root_authority() -> dict[str, Any]:
    """Return the content-pinned figure authority the root envelope selects."""
    return {
        "schema_id": EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_ID,
        "schema_version": EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_VERSION,
        "kind": "figure",
        "assembler": "quillon.span_assembler",
        "assembler_params": {"height": 300},
        "panels": [{"name": "span", "title": "span", "row": 1, "col": 1}],
        "trace_families": [
            {
                "name": f"{payload}-span",
                "index": {"values": ["near", "far"]},
                "trace": {
                    "name": f"{payload}-span-{{index}}",
                    "constructor": "quillon.span_band",
                    "panel": "span",
                    "data": {"x": {"item": payload, "path": "artifact_payloads.x"}},
                },
            }
            for payload in ("summary", "narrative")
        ],
    }


def _contract(input_role: str, **overrides: Any) -> dict[str, Any]:
    """Return the authored artifact contract one root figure input states."""
    declared = ROOT_INPUTS[input_role]
    payload = RETAINED_PAYLOADS[declared["artifact_role"]]
    contract = {
        "kind": "AnalysisRunManifest",
        "artifact_role": declared["artifact_role"],
        "artifact_provider": "quillon.custody",
        "media_type": "application/json",
        "payload_name": declared["payload_name"],
        "payload_schema_id": payload["schema_id"],
        "payload_schema_version": payload["schema_version"],
    }
    contract.update(overrides)
    return contract


def _root_envelope(
    *,
    digest: str,
    size_bytes: int,
    schema: str = EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V5,
    contracts: bool = True,
    **contract_overrides: Any,
) -> dict[str, Any]:
    """Return one authored root figure envelope over the produced manifest."""
    inputs = []
    for input_role in ROOT_INPUTS:
        item: dict[str, Any] = {
            "input_role": input_role,
            "ref": {
                "kind": "receipt",
                "manifest_kind": "AnalysisRunManifest",
                "manifest_id": MANIFEST_ID,
                "manifest_sha256": digest,
                "size_bytes": size_bytes,
            },
        }
        if contracts:
            item["contract"] = _contract(input_role, **contract_overrides)
        inputs.append(item)
    authority = _root_authority()
    return {
        "schema": schema,
        "name": FIGURE_NAME,
        "figure": {
            "mode": "root",
            "root": {"ref": AUTHORITY_REF, "sha256": canonical_sha256(authority)},
            "inputs": inputs,
        },
    }


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _produced_manifest(provider: ImmutableArtifactBlobProvider) -> tuple[bytes, dict[str, str]]:
    """Produce one completed analysis run retaining both declared artifacts."""
    artifacts = [
        provider.store_bytes(
            json.dumps(payload).encode("utf-8"),
            role=role,
            logical_name=f"{role}.json",
            media_type="application/json",
        )
        for role, payload in RETAINED_PAYLOADS.items()
    ]
    manifest = AnalysisRunManifest(
        id=MANIFEST_ID,
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "quillon.span_summary", "inputs": [], "params": {}},
        ),
        artifacts=artifacts,
    )
    raw = canonical_json_bytes(manifest)
    stored = provider.store_bytes(
        raw,
        role="analysis_manifest",
        logical_name="span.manifest.json",
        media_type="application/json",
    )
    return raw, {MANIFEST_ID: str(provider.canonical_relative_path(stored))}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    write_repo(tmp_path)
    return tmp_path


@pytest.fixture
def provider(tmp_path: Path) -> ImmutableArtifactBlobProvider:
    return ImmutableArtifactBlobProvider(tmp_path / "provider")


@pytest.fixture
def produced(provider: ImmutableArtifactBlobProvider) -> tuple[bytes, dict[str, str]]:
    return _produced_manifest(provider)


@pytest.fixture
def environment(tmp_path: Path, repo: Path, application_registry_bundle):
    return FulfillmentEnvironment(
        root=tmp_path / "receipts", registries=application_registry_bundle, repo_root=repo
    )


def _author(repo: Path, produced: tuple[bytes, dict[str, str]], **kwargs: Any) -> None:
    """Write the authority and the root figure envelope this repo compiles."""
    raw, _locations = produced
    _write_json(repo / AUTHORITY_REF, _root_authority())
    _write_json(
        envelope_path(repo, FIGURE_NAME),
        _root_envelope(digest=sha256_bytes(raw), size_bytes=len(raw), **kwargs),
    )


def _compile(repo: Path) -> dict[str, Any]:
    """Compile the root figure envelope and write the two files it emits."""
    outcome = kernel_for(PROJECT_DECLARATION).compile_envelope_file(
        envelope_path(repo, FIGURE_NAME), repo_root=repo
    )
    directory = repo / "compiled"
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / f"{outcome.name}.compile-lock.json", outcome.compile_lock)
    family = outcome.compile_lock["compiled_document"]["family"]
    _write_json(directory / f"{outcome.name}.{family}.json", outcome.document)
    return outcome.compile_lock


def _receipt_root(environment: FulfillmentEnvironment, raw: bytes) -> None:
    """Store the produced manifest where an external reference resolves it."""
    from feedbax.contracts.manifest import canonical_manifest_path

    path = canonical_manifest_path("AnalysisRunManifest", MANIFEST_ID, root=environment.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _figure_request(repo: Path, environment: FulfillmentEnvironment):
    index = read_compiled_outputs(repo / "compiled")
    plan = derive_workflow_plan(index, target=FIGURE_NAME)
    closure = prepare_workflow(plan, index)
    key = LogicalKey("figure", FIGURE_NAME)
    return workflow_requests(closure, environment=environment, stop_at=key)[-1]


def _execution_context(
    provider: ImmutableArtifactBlobProvider,
    parents,
    locations: dict[str, str],
) -> StagedExecutionContext:
    return StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"custody": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=tuple(
            StagedParentExecutionLocation(
                parent=parent,
                root=provider.root,
                execution_uri=locations[parent.id],
                artifact_provider="custody",
            )
            for parent in parents
        ),
        parent_artifact_provider_bindings=tuple(
            StagedParentArtifactProviderBinding(parent, "quillon.custody", "custody")
            for parent in parents
        ),
    )


# -- the compile carries the contract, and moves nothing into identity -----


def test_the_compile_records_each_root_input_contract_in_the_lock(
    repo: Path, produced: tuple[bytes, dict[str, str]]
) -> None:
    _author(repo, produced)

    lock = _compile(repo)

    assert lock["envelope"]["schema"] == EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V5
    assert lock["compiler_contract"]["contract_version"] == (
        "feedbax.experiment_envelope.compiler.v4"
    )
    contracts = {
        reference["consumer"]["input_role"]: reference["consumer"]["contract"]
        for reference in lock["references"]
        if reference.get("consumer", {}).get("consumer") == "figure_runtime_input"
    }
    assert set(contracts) == set(ROOT_INPUTS)
    assert contracts["primary_span"] == {
        "input_role": "primary_span",
        "kind": "AnalysisRunManifest",
        "authority": "artifact_provider",
        "artifact_role": "result",
        "artifact_provider": "quillon.custody",
        "manifest_status": "completed",
        "media_type": "application/json",
        "payload_name": "summary",
        "payload_schema_id": "quillon.span_result",
        "payload_schema_version": "quillon.span_result.v1",
    }


def test_the_compiled_figure_keeps_its_inputs_out_of_its_own_identity(
    repo: Path, produced: tuple[bytes, dict[str, str]]
) -> None:
    """Contracts are a plan fact, and ``feedbax.spec.figure.v2`` does not move."""
    _author(repo, produced)
    _compile(repo)

    document = json.loads(
        next((repo / "compiled").glob(f"{FIGURE_NAME}.*figure*.json")).read_text(encoding="utf-8")
    )

    assert document["schema_version"] == "feedbax.spec.figure.v2"
    assert document["inputs"] == []
    assert document["input_authorities"] == []


# -- fulfillment builds the authority from the lock and nothing else -------


def test_lowering_authorizes_the_exact_parent_it_bound(
    repo: Path,
    produced: tuple[bytes, dict[str, str]],
    environment: FulfillmentEnvironment,
) -> None:
    raw, _locations = produced
    _author(repo, produced)
    _compile(repo)
    _receipt_root(environment, raw)

    request = _figure_request(repo, environment)

    assert request.node_kind == "figure"
    assert [parent.role for parent in request.runtime_inputs] == list(ROOT_INPUTS)
    assert {parent.id for parent in request.runtime_inputs} == {MANIFEST_ID}
    authorities = request.runtime_input_authorities
    assert all(isinstance(item, FigureInputRoleAuthority) for item in authorities)
    assert [item.input_role for item in authorities] == list(ROOT_INPUTS)
    # The authority addresses one exact declared parent, by the role the lock
    # bound it under, and resolves back to it rather than to a second copy.
    assert [item.resolve_parent(request.runtime_inputs) for item in authorities] == list(
        request.runtime_inputs
    )


def test_the_authority_carries_the_whole_declared_selector(
    repo: Path,
    produced: tuple[bytes, dict[str, str]],
    environment: FulfillmentEnvironment,
) -> None:
    raw, _locations = produced
    _author(repo, produced)
    _compile(repo)
    _receipt_root(environment, raw)

    request = _figure_request(repo, environment)

    payload = request.runtime_input_authorities[0].artifact_payloads[0]
    assert payload.name == "summary"
    assert payload.manifest_role == "primary_span"
    assert payload.artifact_role == "result"
    assert payload.artifact_provider == "quillon.custody"
    assert payload.media_type == "application/json"
    assert payload.manifest_status == "completed"
    assert payload.payload_schema_id == "quillon.span_result"
    assert payload.payload_schema_version == "quillon.span_result.v1"


def test_a_grammar_that_could_not_state_a_contract_is_refused_by_name(
    repo: Path,
    produced: tuple[bytes, dict[str, str]],
    environment: FulfillmentEnvironment,
) -> None:
    """The legacy state is a refusal before effects, not an unbound render."""
    raw, _locations = produced
    _author(repo, produced, schema=EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V4, contracts=False)
    _compile(repo)
    _receipt_root(environment, raw)

    with pytest.raises(NodeLoweringError) as caught:
        _figure_request(repo, environment)

    message = str(caught.value)
    assert EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V4 in message
    assert EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V5 in message
    assert "explicit payload name" in message


# -- execution reaches each declared payload, exactly ----------------------


def _resolved_inputs(
    repo: Path,
    environment: FulfillmentEnvironment,
    provider: ImmutableArtifactBlobProvider,
    locations: dict[str, str],
    *,
    authorities=None,
    inputs=None,
):
    request = _figure_request(repo, environment)
    runtime_inputs = list(request.runtime_inputs if inputs is None else inputs)
    runtime_authorities = list(
        request.runtime_input_authorities if authorities is None else authorities
    )
    context = _execution_context(provider, runtime_inputs, locations)
    plan = plan_figure_execution(
        request.spec,
        runtime_inputs=runtime_inputs,
        runtime_input_authorities=runtime_authorities,
        repo_root=repo,
        execution_context=context,
        registry=environment.registries.figures,
    )
    return resolve_figure_inputs(plan.execution_spec, execution_context=context)


def test_each_declared_selector_reaches_its_own_retained_payload(
    repo: Path,
    produced: tuple[bytes, dict[str, str]],
    provider: ImmutableArtifactBlobProvider,
    environment: FulfillmentEnvironment,
) -> None:
    raw, locations = produced
    _author(repo, produced)
    _compile(repo)
    _receipt_root(environment, raw)

    resolved = _resolved_inputs(repo, environment, provider, locations)

    assert [item.ref.role for item in resolved] == list(ROOT_INPUTS)
    assert [item.manifest.id for item in resolved] == [MANIFEST_ID, MANIFEST_ID]
    assert [item.artifact_payloads for item in resolved] == [
        {"summary": RETAINED_PAYLOADS["result"]},
        {"narrative": RETAINED_PAYLOADS["report"]},
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("manifest_role", "elsewhere", "manifest role mismatch"),
        ("artifact_role", "absent", "payload is missing"),
        ("media_type", "application/cbor", "media type mismatch"),
        ("payload_schema_id", "quillon.wrong", "schema_id mismatch"),
        ("payload_schema_version", "quillon.span_result.v9", "schema_version mismatch"),
        ("artifact_provider", "quillon.elsewhere", "provider rejected payload"),
    ],
)
def test_every_selector_disagreement_refuses_before_render_effects(
    repo: Path,
    produced: tuple[bytes, dict[str, str]],
    provider: ImmutableArtifactBlobProvider,
    environment: FulfillmentEnvironment,
    field: str,
    value: str,
    message: str,
) -> None:
    raw, locations = produced
    _author(repo, produced)
    _compile(repo)
    _receipt_root(environment, raw)
    request = _figure_request(repo, environment)
    first = request.runtime_input_authorities[0]
    mutated = first.model_copy(
        update={
            "artifact_payloads": [first.artifact_payloads[0].model_copy(update={field: value})]
        },
        deep=True,
    )

    with pytest.raises(FigureInputAuthorityError, match=message):
        _resolved_inputs(
            repo,
            environment,
            provider,
            locations,
            authorities=[mutated, *request.runtime_input_authorities[1:]],
        )


def test_an_authority_whose_role_no_input_fills_refuses(
    repo: Path,
    produced: tuple[bytes, dict[str, str]],
    provider: ImmutableArtifactBlobProvider,
    environment: FulfillmentEnvironment,
) -> None:
    raw, locations = produced
    _author(repo, produced)
    _compile(repo)
    _receipt_root(environment, raw)
    request = _figure_request(repo, environment)
    orphaned = request.runtime_input_authorities[0].model_copy(
        update={"input_role": "never_bound"}, deep=True
    )

    with pytest.raises(FigureInputAuthorityError) as caught:
        _resolved_inputs(repo, environment, provider, locations, authorities=[orphaned])

    assert "runtime input/authority binding is invalid" in str(caught.value)
    assert "matches no declared" in str(caught.value.__cause__)


def test_an_authority_whose_role_two_inputs_fill_refuses(
    repo: Path,
    produced: tuple[bytes, dict[str, str]],
    provider: ImmutableArtifactBlobProvider,
    environment: FulfillmentEnvironment,
) -> None:
    """Two distinct parents under one role is ambiguity, not a wider selection."""
    raw, locations = produced
    _author(repo, produced)
    _compile(repo)
    _receipt_root(environment, raw)
    request = _figure_request(repo, environment)
    first = request.runtime_inputs[0]
    other_id = f"{MANIFEST_ID}-other"
    duplicate = first.model_copy(update={"id": other_id}, deep=True)

    with pytest.raises(FigureInputAuthorityError) as caught:
        _resolved_inputs(
            repo,
            environment,
            provider,
            {**locations, other_id: locations[MANIFEST_ID]},
            inputs=[first, duplicate],
            authorities=[request.runtime_input_authorities[0]],
        )

    assert "runtime input/authority binding is invalid" in str(caught.value)
    assert "is ambiguous" in str(caught.value.__cause__)


# -- the authored grammar decides what may be stated -----------------------


def test_a_root_input_without_a_contract_is_refused_at_the_current_grammar(
    repo: Path, produced: tuple[bytes, dict[str, str]]
) -> None:
    _author(repo, produced, contracts=False)

    with pytest.raises(ExperimentEnvelopeRejection, match="states no artifact contract"):
        _compile(repo)


def test_a_root_input_contract_without_an_explicit_payload_name_is_refused(
    repo: Path, produced: tuple[bytes, dict[str, str]]
) -> None:
    raw, _locations = produced
    _write_json(repo / AUTHORITY_REF, _root_authority())
    envelope = _root_envelope(digest=sha256_bytes(raw), size_bytes=len(raw))
    for item in envelope["figure"]["inputs"]:
        item["contract"].pop("payload_name")
    _write_json(envelope_path(repo, FIGURE_NAME), envelope)

    with pytest.raises(ExperimentEnvelopeRejection, match="no explicit 'payload_name'"):
        _compile(repo)


def test_a_prior_grammar_stating_a_root_input_contract_is_refused_by_version(
    repo: Path, produced: tuple[bytes, dict[str, str]]
) -> None:
    _author(repo, produced, schema=EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V4)

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _compile(repo)

    assert "figure.inputs[0].contract" in str(caught.value)
    assert EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V5 in str(caught.value)


def test_an_artifact_free_root_figure_states_no_inputs_and_compiles(repo: Path) -> None:
    """A root figure that reads nothing is a whole figure, not a partial one."""
    authority = _root_authority()
    authority["trace_families"] = []
    _write_json(repo / AUTHORITY_REF, authority)
    _write_json(
        envelope_path(repo, FIGURE_NAME),
        {
            "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V5,
            "name": FIGURE_NAME,
            "figure": {
                "mode": "root",
                "root": {"ref": AUTHORITY_REF, "sha256": canonical_sha256(authority)},
            },
        },
    )

    lock = _compile(repo)

    assert not [
        reference
        for reference in lock["references"]
        if reference.get("consumer", {}).get("consumer") == "figure_runtime_input"
    ]
