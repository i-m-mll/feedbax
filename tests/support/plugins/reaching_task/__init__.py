"""A Python reaching task with an authored point and a fixed trial horizon, and a point mass a task must start, which
tests install as a plugin."""


from dataclasses import replace
from dataclasses import replace as dataclass_replace

from pydantic import Field, FiniteFloat

from feedbax.contracts.expression_values import (
    LENGTH, CallValue, Level, LiteralValue, NameValue, Site, ValueBase, ValueRecord, ValueType,
)
from feedbax.contracts.task_document import Cell, Epoch, Rest, Signal, TaskDocument, Variable
from feedbax.contracts.model_values import StateReset
from feedbax.builtins.mechanics.direct_force_components import PointMass
from feedbax.tasks.registry import PythonTask, PythonTaskTemplate


_POINT = ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH)


class ReachingParameters(ValueRecord):
    n_steps: int = Field(default=80, gt=0, le=10_000)
    x: FiniteFloat = Field(default=0.15, title="Target x (m)")
    y: FiniteFloat = Field(default=0.0, title="Target y (m)")


def reaching_document(params: ReachingParameters) -> TaskDocument:
    return TaskDocument(
        id="reaching",
        fixed_steps=params.n_steps,
        variables=(
            Variable(id="target", name="target", type=_POINT, value=CallValue(
                kind="call",
                function="vector",
                args=(LiteralValue(value=params.x, unit="m"), LiteralValue(value=params.y, unit="m")),
            )),
        ),
        epochs=(Epoch(id="move", name="Move", end=Rest()),),
        signals=(Signal(
            id="reach_target", name="reach_target", type=_POINT,
            cells=(Cell(id="move", first_epoch="move", last_epoch="move", value=NameValue(
                kind="name", reference="target",
            )),),
        ),),
    )


class Reaching(PythonTask[ReachingParameters]):
    label = "Reaching"
    settings_model = ReachingParameters
    parameter_sites = {
        "n_steps": Site(expected=ValueType(base=ValueBase.INTEGER), finest=Level.RUN, structural=True),
        "x": Site(expected=ValueType(base=ValueBase.REAL, dimensions=LENGTH), finest=Level.RUN),
        "y": Site(expected=ValueType(base=ValueBase.REAL, dimensions=LENGTH), finest=Level.RUN),
    }

    @staticmethod
    def template(settings: ReachingParameters) -> PythonTaskTemplate:
        document = reaching_document(settings)

        def prepare(model, functions):
            from feedbax.tasks.document import compile_task

            return compile_task(document, functions, model=replace(model, inputs=()))

        return PythonTaskTemplate(
            signals={signal.name: signal.type for signal in document.signals}, prepare=prepare,
        )


REACHING_TASK = "reaching_task.Reaching"


class AnchoredPointMass(PointMass):
    """A point mass with no start of its own: a task must initialize its effector's position."""

    model_facts = dataclass_replace(PointMass.model_facts, states=tuple(
        dataclass_replace(state, reset=StateReset(required=True))
        if state.symbol == "effector/position" else state
        for state in PointMass.model_facts.states
    ))


ANCHORED_POINT_MASS = "reaching_task.AnchoredPointMass"
