"""Authored rollout parameters and saved Evaluate dataset schema identities."""

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

    @field_validator("trials")
    @classmethod
    def _stable_trial_ids(cls, values):
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("expected distinct nonempty authored trial IDs")
        return values

    @property
    def states_schema(self) -> str:
        return RAW_STATES_SCHEMA if self.retention == "raw_states" else STATES_SCHEMA
