"""Locate typed Objective operands in a posed Scene and in a trial record."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_check import references
from feedbax.contracts.expression_registry import Evaluation
from feedbax.contracts.expression_values import NameValue, NodePath
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.trial_record import (
    TrialAnchorPlace, TrialComputedPlace, TrialPlace, TrialRelation, TrialSignalPlace, TrialVariablePlace,
)
from feedbax.objectives.document import ObjectiveExecutor
from feedbax.tasks.document import TrialSource


class BoundComputedPlace(StrictModel):
    kind: Literal["computed"] = "computed"
    index: int = Field(ge=0)


type BoundPlace = Annotated[
    TrialAnchorPlace | TrialVariablePlace | TrialSignalPlace | BoundComputedPlace,
    Field(discriminator="kind"),
]


class BoundTermPlaces(StrictModel):
    relations: tuple[tuple[BoundPlace, BoundPlace], ...]
    places: tuple[BoundPlace, ...]


@dataclass(frozen=True)
class TrialPlacePlan:
    terms: dict[str, BoundTermPlaces] = field(default_factory=dict)
    observations: dict[str, tuple[NodePath, ...]] = field(default_factory=dict)


def bind_trial_places(
    objective: ObjectiveExecutor | None, task: TrialSource, model_places: Mapping[str, ModelAddress],
    *, unobserved: frozenset[str] = frozenset(),
) -> TrialPlacePlan:
    if objective is None:
        return TrialPlacePlan()
    declared: dict[str, BoundPlace] = {
        **{name.id: TrialVariablePlace(kind="variable", id=name.id) for name in task.interface.variables},
        **{name.id: TrialSignalPlace(kind="signal", id=name.id) for name in task.interface.signals},
        **{id: TrialAnchorPlace(kind="anchor", address=address) for id, address in model_places.items()},
    }
    terms, observations = {}, {}
    for term, layout in objective.point_operands().items():
        observations_by_path: dict[NodePath, int] = {}
        places_by_path = {}
        for path, operand in layout.expressions.items():
            if isinstance(operand, NameValue) and operand.reference in declared:
                places_by_path[path] = declared[operand.reference]
            elif not (references(operand) & unobserved):
                places_by_path[path] = BoundComputedPlace(index=len(observations_by_path))
                observations_by_path[path] = len(observations_by_path)
        relations = tuple((places_by_path[a], places_by_path[b]) for a, b in layout.relations
                          if a in places_by_path and b in places_by_path)
        places = tuple(places_by_path[path] for path in layout.places if path in places_by_path)
        if relations or places:
            terms[term] = BoundTermPlaces(relations=relations, places=places)
            observations[term] = tuple(observations_by_path)
    return TrialPlacePlan(terms, observations)


def finite_points(values, defined):
    values = np.asarray(values)
    defined = np.asarray(defined)
    if defined.ndim == values.ndim:
        defined = defined.all(axis=-1)
    valid = defined & np.isfinite(values).all(axis=-1)
    return np.where(valid[..., None], values, 0.0), valid


def point_samples(values, defined) -> TrialComputedPlace:
    values, valid = finite_points(values, defined)
    samples = [tuple(point) if use else None for point, use in zip(values.tolist(), valid, strict=True)]
    return TrialComputedPlace(kind="computed", samples=samples)


def project_term_places(
    bound: BoundTermPlaces | None, compute: Callable[[int], Evaluation | None],
) -> tuple[list[TrialRelation], list[TrialPlace]]:
    if bound is None:
        return [], []

    def resolve(place: BoundPlace) -> TrialPlace | None:
        if not isinstance(place, BoundComputedPlace):
            return place
        value = compute(place.index)
        return point_samples(value.value, value.defined) if value is not None else None

    relations = []
    for first, second in bound.relations:
        if (a := resolve(first)) is not None and (b := resolve(second)) is not None:
            relations.append(TrialRelation(first=a, second=b))
    return relations, [place for item in bound.places if (place := resolve(item)) is not None]
