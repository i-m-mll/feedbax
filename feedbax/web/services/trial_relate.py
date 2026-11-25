"""Prepare the Objective term written by a Studio Relate gesture."""

from feedbax.compiler.model_anchors import model_anchors
from feedbax.compiler.model_signals import ResolvedSignalLeaf, model_measure_scope
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.expression_values import CallValue, Expression, LENGTH, LiteralValue, NameValue, ValueBase, ValueType
from feedbax.contracts.model_address import ComponentElement, ComponentPortAddress, FieldElement, OutputBoundaryEnd, PortElement
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.objective_document import ObjectiveDocument, ObjectiveTerm
from feedbax.contracts.studio_trials import StudioRelatePoint, StudioRelatePort, StudioTrialRelate
from feedbax.contracts.trial_record import TrialAnchorPlace, TrialSignalPlace
from feedbax.execution.trial_resolution import TrialPair
from feedbax.project_members import generated_name


def related_term(
    change: StudioTrialRelate, model: ModelDocument, pair: TrialPair,
    objective: ObjectiveDocument, components: ComponentRegistry,
) -> ObjectiveTerm:
    outputs = model_measure_scope(model, pair.model.signals).leaves

    def output_preference(leaf: ResolvedSignalLeaf) -> tuple[int, int, str]:
        element = leaf.port.element
        assert isinstance(element, PortElement)
        if not leaf.port.composites and isinstance(element.end, OutputBoundaryEnd):
            return 0, len(leaf.path), leaf.address.to_text()
        origin = leaf.origin
        if origin is not None and isinstance(origin.element, FieldElement):
            parent = origin.element.parent
            if (
                origin.composites == leaf.port.composites and isinstance(parent, ComponentElement)
                and isinstance(element.end, ComponentPortAddress) and element.end.component == parent.id
            ):
                return 1, len(leaf.path), leaf.address.to_text()
        return 2, len(leaf.path), leaf.address.to_text()

    def read(place) -> tuple[Expression, ValueType]:
        if isinstance(place, StudioRelatePoint):
            return CallValue(
                function="point", args=tuple(LiteralValue(value=coordinate, unit="m") for coordinate in place.position),
            ), ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH, space="point")
        if isinstance(place, TrialAnchorPlace | StudioRelatePort):
            if isinstance(place, TrialAnchorPlace):
                origin = next(
                    (anchor.signal for anchor in model_anchors(model, components) if anchor.address == place.address),
                    None,
                )
                leaf = None
            else:
                leaf = outputs.get(place.address.to_text())
                located = pair.model.signals.at(place.address)
                origin = located.origin if located is not None else None
            if leaf is None and origin is not None:
                leaf = min(
                    (value for value in outputs.values() if value.origin == origin),
                    key=output_preference, default=None,
                )
            if leaf is None or leaf.type is None:
                raise ValueError("This Model place has no readable signal.")
            return NameValue(reference=leaf.address.to_text()), leaf.type
        values = pair.task.interface.signals if isinstance(place, TrialSignalPlace) else pair.task.interface.variables
        value = next((value for value in values if value.id == place.id), None)
        if value is None:
            raise ValueError("This Task place no longer exists.")
        if value.type.space != "point" or value.type.shape != (2,):
            raise ValueError("A Task place must be a planar point.")
        return NameValue(reference=value.id), value.type

    first, first_type = read(change.first)
    if first_type.base == ValueBase.BOOLEAN:
        raise ValueError("A term must read a numeric value.")
    if change.second is not None:
        second, second_type = read(change.second)
        if any(kind.space != "point" or kind.shape != (2,) or kind.dimensions != LENGTH
               for kind in (first_type, second_type)):
            raise ValueError("A relation needs two planar points.")
        first = CallValue(function="sub", args=(first, second))
    taken = {term.id for term in objective.terms}
    return ObjectiveTerm(
        id=generated_name("term", taken.__contains__),
        measure=CallValue(function="sqnorm", args=(first,)),
        measure_type=ValueType(base=ValueBase.REAL, dimensions=first_type.dimensions.raised(2)),
    )
