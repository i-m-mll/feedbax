from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from feedbax.web.api import runs

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.no_silent_substitution_contract]


def test_create_eval_run_fails_when_persistence_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_db_session():
        raise RuntimeError("database offline")

    monkeypatch.setattr(runs, "db_session", fail_db_session)

    payload = runs.CreateEvalRunRequest(
        training_run_id="training-a",
        name="eval-a",
        eval_params={"perturbation": "none"},
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(runs.create_eval_run(payload))

    assert excinfo.value.status_code == 500
    assert "Could not persist evaluation run" in str(excinfo.value.detail)
