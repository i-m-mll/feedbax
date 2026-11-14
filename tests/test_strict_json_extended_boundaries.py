"""Compatibility vectors for strict JSON outside the contracts package."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from feedbax.analysis.evaluation import coerce_evaluation_run_spec
from feedbax.contracts.strict_json import (
    DuplicateJsonKeyError,
    strict_model_validate_json,
)

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.strict_json_boundary_contract]


class _TextPayload(BaseModel):
    value: str


class _FloatPayload(BaseModel):
    value: float


@pytest.mark.parametrize("document", ('{"value":"\\ud800"}', '{"value":"ok"} trailing'))
def test_strict_pydantic_boundary_preserves_original_rejection(document: str) -> None:
    with pytest.raises(ValidationError) as original:
        _TextPayload.model_validate_json(document)
    with pytest.raises(type(original.value)) as strict:
        strict_model_validate_json(_TextPayload, document)
    assert strict.value.errors() == original.value.errors()


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_strict_pydantic_boundary_preserves_non_finite_numbers(constant: str) -> None:
    document = f'{{"value":{constant}}}'
    original = _FloatPayload.model_validate_json(document)
    strict = strict_model_validate_json(_FloatPayload, document)
    assert (
        math.isnan(strict.value) if math.isnan(original.value) else strict.value == original.value
    )


def test_strict_pydantic_boundary_adds_only_duplicate_refusal() -> None:
    with pytest.raises(DuplicateJsonKeyError):
        strict_model_validate_json(_TextPayload, '{"value":"first","value":"second"}')


def test_public_evaluation_loader_refuses_duplicate_authored_members(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.json"
    path.write_text('{"evaluation_type":"first","evaluation_type":"second"}', encoding="utf-8")
    with pytest.raises(DuplicateJsonKeyError):
        coerce_evaluation_run_spec(path)
