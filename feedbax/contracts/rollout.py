"""Authored rollout parameters and saved Evaluate dataset schema identities."""

import re
from typing import Literal

from pydantic import Field, field_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.manifest import OverridePatch

STUDIO_DEFAULT_EVALUATION_TYPE = "feedbax.studio.default_eval"
ROLLOUT_TYPE = "feedbax.rollout"
STATES_SCHEMA = "feedbax.candidate.evaluate.states.v1"
RAW_STATES_SCHEMA = "feedbax.candidate.evaluate.raw_states.v1"


class RolloutParameters(StrictModel):
    seed: int = Field(strict=True, ge=0, lt=2**32)
    trials: tuple[str, ...] = Field(min_length=1)
    trial_policy: Literal["paired"]
    retention: Literal["raw_states", "observations"] = "raw_states"
    variation: tuple[OverridePatch, ...]

    @field_validator("variation", mode="before")
    @classmethod
    def _ordered_variation(cls, value):
        if not isinstance(value, dict):
            return value
        if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in value):
            raise ValueError("expected path-safe variation edit IDs")
        edits = tuple(OverridePatch.model_validate(value[name]) for name in sorted(value))
        paths = {edit.path for edit in edits}
        if len(paths) != len(edits):
            raise ValueError("named edits have duplicate targets")
        for path in paths:
            while "." in path:
                path = path.rpartition(".")[0]
                if path in paths:
                    raise ValueError(f"named edits overlap at {path!r}")
        return edits

    @field_validator("trials")
    @classmethod
    def _stable_trial_ids(cls, values):
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("expected distinct nonempty authored trial IDs")
        return values

    @property
    def states_schema(self) -> str:
        return RAW_STATES_SCHEMA if self.retention == "raw_states" else STATES_SCHEMA
