"""Authored rollout parameters and saved Evaluate dataset schema identities."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator

from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.request_refs import Sha256

if TYPE_CHECKING:
    from feedbax.compiler.graph import GraphKeySchedule
    from feedbax.contracts.graph import GraphSpec
    from feedbax.contracts.objective_document import ObjectiveDocument
    from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument

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

    def condition_id(self) -> str:
        return canonical_sha256(self.model_dump(mode="json")["variation"])


def rollout_objective_sha256(objective: ObjectiveDocument) -> str:
    return sha256_bytes(canonical_json_v2_bytes(objective.model_dump(mode="json")))


class ResolvedRolloutMember(StrictModel):
    coordinate: int
    model_tree: Sha256
    model: Sha256
    task: Sha256
    trials: tuple[str, ...]


def rollout_semantics(
    graph: GraphSpec, schedule: GraphKeySchedule, task_source: TaskDocument | PythonTaskSelection,
    input_bindings: Mapping[str, str] | tuple[tuple[str, str], ...], source_overrides: Mapping[str, float],
) -> tuple[str, str]:
    """Identify the resolved model and task without using their numerical state."""
    model = sha256_bytes(canonical_json_v2_bytes({
        "graph": graph.model_dump(mode="json"),
        "key_schedule": schedule.model_dump(mode="json"),
        "source_overrides": source_overrides,
    }))
    task = sha256_bytes(canonical_json_v2_bytes({
        "source": task_source.model_dump(mode="json"),
        "input_bindings": dict(input_bindings),
    }))
    return model, task


class ResolvedRolloutBinding(StrictModel):
    """Scientific inputs bound by the worker before starting the paired rollout."""

    schema_id: Literal["feedbax.evaluate.binding.v1"] = "feedbax.evaluate.binding.v1"
    members: tuple[ResolvedRolloutMember, ...] = Field(min_length=1)
    objective: Sha256
    seed: int
    trial_policy: Literal["paired"]

    def digest(self) -> str:
        return sha256_bytes(canonical_json_v2_bytes(self.model_dump(mode="json")))
