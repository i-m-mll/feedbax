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
from feedbax.orchestration.events import RunEventProtocolError, RunEventReader
from feedbax.training.run_matrix import RunMatrixError, _load_pinned_canonical_document

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


def test_pinned_matrix_loader_keeps_its_domain_error_for_duplicate_json(tmp_path: Path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text('{"schema_version":"first","schema_version":"second"}', encoding="utf-8")
    with pytest.raises(RunMatrixError, match="cannot load pinned JSON document"):
        _load_pinned_canonical_document(
            tmp_path,
            ref=path.name,
            expected_sha256="0" * 64,
            field="/matrix",
        )


def test_event_reader_keeps_its_protocol_error_for_duplicate_json(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"seq":1,"seq":2}\n', encoding="utf-8")
    with pytest.raises(RunEventProtocolError, match="Invalid RunEvent JSONL"):
        RunEventReader(path).read_all()
