from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from feedbax.contracts import (
    ARRAY_VALUE_SCHEMA_ID,
    ARRAY_VALUE_SCHEMA_VERSION,
    ArrayValueSpec,
    ConstantArrayValueSpec,
    SparseCooArrayValueSpec,
    SparseCooEntrySpec,
    authored_value_sha256,
    materialize_array_value,
    semantic_value_sha256,
)
from feedbax.contracts.graph import ComponentSpec


def _sparse(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": ARRAY_VALUE_SCHEMA_ID,
        "schema_version": ARRAY_VALUE_SCHEMA_VERSION,
        "encoding": "sparse_coo",
        "shape": [2, 3],
        "dtype": "float32",
        "nonfinite": "forbid",
        "fill": 0.0,
        "entries": [
            {"coordinate": [1, 2], "value": 3.0},
            {"coordinate": [0, 1], "value": -2.0},
        ],
    }
    payload.update(updates)
    return payload


def test_sparse_coo_canonicalizes_row_major_and_materializes_fill() -> None:
    spec = TypeAdapter(ArrayValueSpec).validate_python(_sparse(fill=0.5))

    assert isinstance(spec, SparseCooArrayValueSpec)
    assert [entry.coordinate for entry in spec.entries] == [(0, 1), (1, 2)]
    assert materialize_array_value(spec).tolist() == [
        [0.5, -2.0, 0.5],
        [0.5, 0.5, 3.0],
    ]


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"shape": []}, "at least 1 item"),
        ({"shape": [2, 0]}, "greater than 0"),
        ({"shape": [True, 2]}, "valid integer"),
        ({"dtype": "<f4"}, "Input should be"),
        (
            {
                "entries": [
                    {"coordinate": [0, 1], "value": 1.0},
                    {"coordinate": [0, 1], "value": 2.0},
                ]
            },
            "duplicated",
        ),
        ({"entries": [{"coordinate": [2, 0], "value": 1.0}]}, "outside shape"),
        ({"entries": [{"coordinate": [1], "value": 1.0}]}, "coordinate rank"),
        ({"dtype": "int8", "fill": 128, "entries": []}, "outside dtype"),
        ({"dtype": "int32", "fill": 0.5, "entries": []}, "must be an integer"),
        ({"dtype": "float16", "fill": 1e100, "entries": []}, "finite range"),
        ({"fill": "nan", "entries": []}, "nonfinite='forbid'"),
        (
            {"dtype": "int32", "nonfinite": "allow", "fill": "+inf", "entries": []},
            "floating dtype",
        ),
        ({"fill": math.nan, "entries": []}, "explicit non-finite token"),
    ],
)
def test_sparse_coo_rejects_noncanonical_values(
    updates: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        TypeAdapter(ArrayValueSpec).validate_python(_sparse(**updates))


def test_explicit_nonfinite_policy_uses_json_safe_tokens() -> None:
    spec = ConstantArrayValueSpec(
        schema_id=ARRAY_VALUE_SCHEMA_ID,
        schema_version=ARRAY_VALUE_SCHEMA_VERSION,
        encoding="constant",
        shape=(2,),
        dtype="float64",
        nonfinite="allow",
        value="-inf",
    )

    materialized = materialize_array_value(spec)
    assert np.isneginf(materialized).all()
    assert json.loads(spec.model_dump_json())["value"] == "-inf"


def test_dense_sparse_and_dense_constant_share_semantic_identity() -> None:
    dense_sparse = np.asarray([[0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    sparse = SparseCooArrayValueSpec(
        schema_id=ARRAY_VALUE_SCHEMA_ID,
        schema_version=ARRAY_VALUE_SCHEMA_VERSION,
        encoding="sparse_coo",
        shape=(2, 2),
        dtype="float32",
        nonfinite="forbid",
        fill=0.0,
        entries=(SparseCooEntrySpec(coordinate=(0, 1), value=2.0),),
    )
    dense_constant = np.full((2, 3), 1.25, dtype=np.float32)
    constant = ConstantArrayValueSpec(
        schema_id=ARRAY_VALUE_SCHEMA_ID,
        schema_version=ARRAY_VALUE_SCHEMA_VERSION,
        encoding="constant",
        shape=(2, 3),
        dtype="float32",
        nonfinite="forbid",
        value=1.25,
    )

    assert semantic_value_sha256(dense_sparse, dtype="float32") == semantic_value_sha256(
        materialize_array_value(sparse), dtype="float32"
    )
    assert semantic_value_sha256(dense_constant, dtype="float32") == semantic_value_sha256(
        materialize_array_value(constant), dtype="float32"
    )
    dense_authored = authored_value_sha256(
        encoding_kind="dense",
        encoding_schema_id=ARRAY_VALUE_SCHEMA_ID,
        encoding_schema_version=ARRAY_VALUE_SCHEMA_VERSION,
        arguments={"value": dense_sparse.tolist(), "dtype": "float32"},
    )
    sparse_authored = authored_value_sha256(
        encoding_kind=sparse.encoding,
        encoding_schema_id=sparse.schema_id,
        encoding_schema_version=sparse.schema_version,
        arguments=sparse.model_dump(
            mode="json",
            exclude={"schema_id", "schema_version", "encoding"},
        ),
    )
    assert dense_authored != sparse_authored


def test_component_params_canonicalize_nested_array_declarations_and_round_trip() -> None:
    component = ComponentSpec.model_validate(
        {
            "type": "fixture.Component",
            "params": {"nested": [{"matrix": _sparse()}]},
        }
    )

    encoded = component.params["nested"][0]["matrix"]
    assert [entry["coordinate"] for entry in encoded["entries"]] == [[0, 1], [1, 2]]
    assert ComponentSpec.model_validate_json(component.model_dump_json()) == component


@pytest.mark.parametrize(
    "value",
    [
        {"schema_id": ARRAY_VALUE_SCHEMA_ID, "encoding": "constant"},
        {"schema_version": ARRAY_VALUE_SCHEMA_VERSION, "encoding": "constant"},
        {
            "schema_id": ARRAY_VALUE_SCHEMA_ID,
            "schema_version": f"{ARRAY_VALUE_SCHEMA_ID}.v0",
            "encoding": "constant",
        },
        {
            "schema_id": "third.party.array",
            "schema_version": ARRAY_VALUE_SCHEMA_VERSION,
            "encoding": "constant",
        },
    ],
)
def test_component_params_reject_partial_mismatched_and_unknown_reserved_tags(
    value: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="both exact reserved tags"):
        ComponentSpec.model_validate({"type": "fixture.Component", "params": {"value": value}})


def test_ordinary_untagged_dictionaries_remain_ordinary_component_params() -> None:
    ordinary = {"shape": [2, 2], "entries": [{"row": 0, "column": 1, "value": 3.0}]}

    component = ComponentSpec.model_validate(
        {"type": "fixture.Component", "params": {"configuration": ordinary}}
    )

    assert component.params["configuration"] == ordinary


def test_four_36_by_36_sparse_matrices_fit_authored_size_ceiling() -> None:
    matrices = [
        SparseCooArrayValueSpec(
            schema_id=ARRAY_VALUE_SCHEMA_ID,
            schema_version=ARRAY_VALUE_SCHEMA_VERSION,
            encoding="sparse_coo",
            shape=(36, 36),
            dtype="float32",
            nonfinite="forbid",
            fill=0.0,
            entries=(SparseCooEntrySpec(coordinate=(index, 35 - index), value=0.25),),
        ).model_dump(mode="json")
        for index in range(4)
    ]
    authored = json.dumps({"delta_A": matrices}, separators=(",", ":"))

    assert len(authored.encode("utf-8")) < 4_096
    assert len(authored.splitlines()) < 100
