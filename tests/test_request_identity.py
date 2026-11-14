from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from feedbax.contracts.canonical_json import CanonicalJsonError, CanonicalJsonErrorCode
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.request_identity import (
    REQUEST_KINDS,
    RequestDocumentMemo,
    decode_request,
    encode_request,
    generate_identity_fields,
    identity_contract,
)
from feedbax.contracts.request_identity_guard import (
    validate_identity_contract,
    validate_identity_transition,
)
from feedbax.contracts.request_refs import (
    BarrierMember,
    ContentRef,
    InputMember,
    ProductRef,
    SlotMember,
)
from feedbax.contracts.request_specs import (
    EvaluationRequestSpec,
    NumericalSettings,
    RenderRequestSpec,
    TrainingRequestSpec,
)
from feedbax.contracts.training import TrainingConfig

_CORPUS = Path(__file__).parent / "fixtures" / "request_identity_corpus.json"


def _read_corpus():
    return json.loads(_CORPUS.read_bytes())["requests"]


def test_downstream_identity_corpus_round_trips():
    rows = _read_corpus()
    assert set(REQUEST_KINDS) == {row["kind"] for row in rows}
    for row in rows:
        spec_type = REQUEST_KINDS[row["kind"]].spec_type
        spec = spec_type.model_validate_json(json.dumps(row["spec"]))
        request = encode_request(spec, row["inputs"])
        stored = decode_request(request.data, request.documents.__getitem__)
        assert stored.id == request.id == row["id"], row["source"]
        assert encode_request(stored.spec, stored.inputs) == request


def test_factored_document_reuse_keeps_current_values_and_input_bindings():
    source = next(row for row in _read_corpus() if row["kind"] == "train")
    raw = source["spec"]
    raw["method_payload"]["payload"]["typed_value"] = True
    spec = TrainingRequestSpec.model_validate(raw)
    memo = RequestDocumentMemo()
    first = encode_request(spec, {}, document_memo=memo)
    assert first == encode_request(spec, {})

    new_seed = spec.seed + 1 if spec.seed is not None else 0
    spec = spec.model_copy(update={"seed": new_seed})
    seeded = encode_request(spec, {}, document_memo=memo)
    assert seeded == encode_request(spec, {})
    assert seeded.id != first.id and seeded.documents == first.documents

    inputs = {"checkpoint": (InputMember(
        key="first", reference=ProductRef(request_id=first.id, role="checkpoint")
    ),)}
    bound = encode_request(spec, inputs, document_memo=memo)
    assert bound == encode_request(spec, inputs)
    assert bound.id != seeded.id and bound.documents == seeded.documents

    spec.method_payload.payload["typed_value"] = 1
    changed = encode_request(spec, inputs, document_memo=memo)
    assert changed == encode_request(spec, inputs)
    assert changed.id != bound.id and len(changed.documents.keys() - bound.documents.keys()) == 1


def test_absent_values_and_retired_spellings_contract():
    checked = identity_contract()
    assert checked["fields"] == generate_identity_fields()
    validate_identity_contract(checked)

    model = "feedbax.contracts.training.TrainingConfig"
    changed = deepcopy(checked)
    changed["fields"][model]["grad_clip"]["absent"] = None
    with pytest.raises(ValueError, match="absent value changed"):
        validate_identity_transition(checked, changed)

    changed = deepcopy(checked)
    changed["retired"]["fields"].append(
        {"model": model, "field": "grad_clip", "migration": "training-v2"}
    )
    with pytest.raises(ValueError, match="field reused"):
        validate_identity_contract(changed)
    del changed["fields"][model]["grad_clip"]
    with pytest.raises(ValueError, match="no registered migration"):
        validate_identity_transition(checked, changed)
    validate_identity_transition(checked, changed, migrations={"training-v2"})


def test_author_default_move_keeps_nested_stored_absence():
    source = _read_corpus()[0]
    assert source["kind"] == "train"
    original = TrainingRequestSpec.model_validate_json(json.dumps(source["spec"]))
    config = TrainingConfig()
    original = original.model_copy(update={"training_config": config})
    old = encode_request(original, {})

    field = TrainingConfig.model_fields["grad_clip"]
    default = field.default
    precision = NumericalSettings.model_fields["matmul_precision"]
    old_precision = precision.default
    try:
        field.default = None
        precision.default = "high"
        TrainingConfig.model_rebuild(force=True)
        NumericalSettings.model_rebuild(force=True)
        TrainingRequestSpec.model_rebuild(force=True)
        author = original.model_copy(
            update={
                "training_config": TrainingConfig(),
                "numerical_settings": NumericalSettings(),
            }
        )
        new = encode_request(author, {})
        assert new.id != old.id
        assert json.loads(new.data)["spec"]["training_config"]["grad_clip"] is None

        stored = decode_request(old.data, old.documents.__getitem__)
        assert isinstance(stored.spec, TrainingRequestSpec)
        assert stored.spec.training_config.grad_clip == default
        assert stored.spec.numerical_settings.matmul_precision == old_precision
        assert json.loads(new.data)["spec"]["numerical_settings"]["matmul_precision"] == "high"
        assert stored.id == old.id
    finally:
        field.default = default
        precision.default = old_precision
        TrainingConfig.model_rebuild(force=True)
        NumericalSettings.model_rebuild(force=True)
        TrainingRequestSpec.model_rebuild(force=True)


def test_cached_absence_preserves_canonical_number_and_boolean_distinctions():
    source = next(row for row in _read_corpus() if row["kind"] == "train")
    spec = TrainingRequestSpec.model_validate(source["spec"]).model_copy(
        update={"training_config": TrainingConfig()}
    )
    memo = RequestDocumentMemo()
    baseline = encode_request(spec, {}, document_memo=memo)

    def with_clip(value):
        # A validated model can be changed by model_copy without another validation.
        config = spec.training_config.model_copy(update={"grad_clip": value})
        return spec.model_copy(update={"training_config": config})

    assert encode_request(with_clip(1), {}, document_memo=memo) == baseline
    changed = encode_request(with_clip(True), {}, document_memo=memo)
    assert changed.id != baseline.id
    assert json.loads(changed.data)["spec"]["training_config"]["grad_clip"] is True
    with pytest.raises(ValueError, match="unsafe_integer"):
        encode_request(with_clip(2**53), {}, document_memo=memo)


def test_request_references_and_runtime_settings_are_typed_identity():
    spec = EvaluationRequestSpec(evaluation_type="rollout", params={"gain": 1})
    first = InputMember(
        key="a",
        reference=ProductRef(
            request_id="train:" + "a" * 64, role="checkpoint", select=SlotMember(slot="model")
        ),
    )
    second = InputMember(
        key="b", reference=ContentRef(sha256="b" * 64, size=100, media_type="application/json")
    )
    request = encode_request(spec, {"initial": (first, second)})
    equivalent = spec.model_copy(update={"params": {"gain": 1.0}})
    assert encode_request(equivalent, {"initial": (first, second)}).id == request.id
    assert encode_request(spec, {"initial": (second, first)}).id != request.id
    at_barrier = InputMember(
        key="a",
        reference=ProductRef(
            request_id="train:" + "a" * 64,
            role="checkpoint",
            select=BarrierMember(name="warmup"),
        ),
    )
    assert encode_request(spec, {"initial": (at_barrier, second)}).id != request.id
    changed = spec.model_copy(
        update={"numerical_settings": NumericalSettings(matmul_precision="high")}
    )
    assert encode_request(changed, {"initial": (first, second)}).id != request.id
    assert decode_request(request.data, request.documents.__getitem__).inputs == {
        "initial": (first, second)
    }
    with pytest.raises(ValueError, match="Duplicate input key"):
        encode_request(spec, {"initial": (first, first)})


def test_figure_raw_mapping_cannot_collide_with_a_typed_expression():
    figure = FigureSpec.model_validate(
        {
            "name": "example",
            "assembler": "plot",
            "traces": [
                {
                    "name": "one",
                    "constructor": "sample",
                    "data": {"map": {"$family": "opaque-data", "value": 42}},
                }
            ],
        }
    )
    request = encode_request(RenderRequestSpec(figure=figure), {})
    decoded = decode_request(request.data, request.documents.__getitem__)
    assert isinstance(decoded.spec, RenderRequestSpec)
    assert decoded.spec.figure == figure


def test_training_documents_keep_semantic_types_and_large_integers():
    source = _read_corpus()[0]
    spec = TrainingRequestSpec.model_validate_json(json.dumps(source["spec"]))
    empty = encode_request(spec.model_copy(update={"graph": GraphSpec()}), {})
    assert empty.documents[json.loads(empty.data)["spec"]["graph"]] == b"{}"
    assert decode_request(empty.data, empty.documents.__getitem__).id == empty.id
    graph = GraphSpec.model_validate(
        {
            "subgraphs": {
                "plant": {
                    "schema_id": "feedbax.spec.acausal_graph",
                    "physical_domain": "translational",
                    "solver": {"solver_type": "euler", "dt": 0.01},
                },
            },
            "nodes": {
                "node": {
                    "type": "matrix",
                    "params": {
                        "table": {
                            "schema_id": "feedbax.spec.component_param.array_value",
                            "schema_version": "feedbax.spec.component_param.array_value.v1",
                            "encoding": "constant",
                            "dtype": "uint64",
                            "shape": [1],
                            "nonfinite": "forbid",
                            "value": (1 << 64) - 1,
                        }
                    },
                },
            },
        }
    )
    spec = spec.model_copy(update={"graph": graph, "seed": (1 << 64) - 1})
    request = encode_request(spec, {})
    data = json.loads(request.data)
    graph_bytes = request.documents[data["spec"]["graph"]]
    assert b'"value":"18446744073709551615"' in graph_bytes
    assert data["spec"]["seed"] == "18446744073709551615"
    decoded = decode_request(request.data, request.documents.__getitem__)
    assert decoded.spec == spec
    with pytest.raises(ValueError, match="Invalid request document bytes"):
        decode_request(request.data, lambda digest: b"{}")
    edited = spec.model_copy(update={"training_config": TrainingConfig(learning_rate=0.01)})
    another = encode_request(edited, {})
    assert another.documents == request.documents
    assert another.id != request.id


def test_shared_spec_keeps_canonical_error_order_and_paths():
    spec = RenderRequestSpec(figure=FigureSpec(name="example", assembler="plot", traces=[]))
    malformed = spec.model_copy(update={
        "figure": spec.figure.model_copy(update={"name": chr(0xD800)}),
    })
    member = InputMember.model_construct(
        key=chr(0xD800),
        reference=ProductRef(request_id="train:" + "a" * 64, role="checkpoint"),
    )
    with pytest.raises(CanonicalJsonError) as error:
        encode_request(malformed, {"initial": (member,)})
    assert error.value.code == CanonicalJsonErrorCode.LONE_SURROGATE
    assert error.value.path == "/spec/figure/name"

    with pytest.raises(CanonicalJsonError) as error:
        encode_request(spec, {"initial": (member,)})
    assert error.value.code == CanonicalJsonErrorCode.LONE_SURROGATE
    assert error.value.path == "/inputs/initial/0/key"


def test_shared_spec_keeps_storage_only_input_diagnostics():
    spec = EvaluationRequestSpec(evaluation_type="rollout")
    member = InputMember(key="candidate", reference=ContentRef(
        sha256="a" * 64, size=2**53,
    ))
    with pytest.raises(CanonicalJsonError) as error:
        encode_request(spec, {"initial": (member,)})
    assert error.value.code == CanonicalJsonErrorCode.UNSAFE_INTEGER
    assert error.value.path == "/inputs/initial/0/reference/size"
