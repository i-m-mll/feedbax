"""Authored rollout parameters and saved Evaluate dataset schema identities."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project_sources import pointer

STUDIO_DEFAULT_EVALUATION_TYPE = "feedbax.studio.default_eval"
ROLLOUT_TYPE = "feedbax.rollout"
STATES_SCHEMA = "feedbax.candidate.evaluate.states.v1"
RAW_STATES_SCHEMA = "feedbax.candidate.evaluate.raw_states.v1"
_MAX_TRIALS = 10_000

type VariationEdits = tuple[OverridePatch, ...] | dict[str, OverridePatch]


class EveryCondition(StrictModel):
    kind: Literal["every_condition"]


class SelectedConditions(StrictModel):
    kind: Literal["conditions"]
    ids: tuple[str, ...] = Field(min_length=1)


class SampleTrials(StrictModel):
    kind: Literal["sample"]
    count: int = Field(strict=True, ge=1, le=_MAX_TRIALS)


_TRIAL_RULE = TypeAdapter(Annotated[EveryCondition | SelectedConditions | SampleTrials, Field(discriminator="kind")])


def trial_indices(ids: tuple[str, ...], conditions: tuple[str, ...]) -> tuple[int, ...]:
    if len(set(conditions)) != len(conditions):
        raise ValueError("Task conditions have duplicate coordinates.")
    if len(set(ids)) != len(ids):
        raise ValueError("Trial IDs must be distinct.")
    indices = {name: index for index, name in enumerate(conditions)}
    result = []
    for name in ids:
        if conditions:
            if name not in indices:
                raise ValueError(f"Unknown Task condition {name!r}.")
            result.append(indices[name])
        else:
            if re.fullmatch(r"0|[1-9][0-9]{0,3}", name) is None or int(name) >= _MAX_TRIALS:
                raise ValueError(f"Invalid Task sample ID {name!r}.")
            result.append(int(name))
    return tuple(result)


def resolve_trial_ids(raw: object, conditions: tuple[str, ...]) -> tuple[str, ...]:
    """Lower one authored rule to the existing ordered rollout request field."""
    rule = _TRIAL_RULE.validate_python(raw)
    if isinstance(rule, SampleTrials):
        if conditions:
            raise ValueError("This Task declares a fixed cohort. Select its conditions.")
        return tuple(map(str, range(rule.count)))
    if not conditions:
        raise ValueError("This Task declares a sampler. Choose a sample count.")
    ids = conditions if isinstance(rule, EveryCondition) else rule.ids
    trial_indices(ids, conditions)
    return ids


def authored_rollout_settings(raw: object) -> RolloutSettings:
    if not isinstance(raw, dict):
        return RolloutSettings.model_validate(raw)
    return RolloutSettings.model_validate({name: value for name, value in raw.items() if name not in {"trial_set", "trials"}})


def authored_variation(edits: VariationEdits) -> Iterator[tuple[str, OverridePatch]]:
    entries = sorted(edits.items()) if isinstance(edits, dict) else enumerate(edits)
    for name, edit in entries:
        yield pointer(("params", "variation", str(name))), edit


class RolloutSettings(StrictModel):
    seed: int = Field(strict=True, ge=0, lt=2**32)
    trial_policy: Literal["paired"]
    retention: Literal["raw_states", "observations"] = "raw_states"
    variation: VariationEdits

    @field_validator("variation")
    @classmethod
    def _ordered_variation(cls, value):
        if not isinstance(value, dict):
            return value
        if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in value):
            raise ValueError("expected path-safe variation edit IDs")
        paths = {edit.path for edit in value.values()}
        if len(paths) != len(value):
            raise ValueError("named edits have duplicate targets")
        for path in paths:
            while "." in path:
                path = path.rpartition(".")[0]
                if path in paths:
                    raise ValueError(f"named edits overlap at {path!r}")
        return value

    @property
    def states_schema(self) -> str:
        return RAW_STATES_SCHEMA if self.retention == "raw_states" else STATES_SCHEMA


class RolloutParameters(RolloutSettings):
    trials: tuple[str, ...] = Field(min_length=1)

    @field_validator("trials")
    @classmethod
    def _stable_trial_ids(cls, values):
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("expected distinct nonempty trial IDs")
        return values
