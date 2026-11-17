"""A registered reaching task with an authored point and a fixed trial horizon."""


from dataclasses import replace

from pydantic import Field, FiniteFloat

from feedbax.contracts.expression_values import (
    LENGTH, CallValue, Level, LiteralValue, NameValue, ScopedValue, Site, ValueBase, ValueRecord, ValueType,
)
from feedbax.contracts.task_document import Cell, Epoch, PythonTaskSelection, Signal, TaskDocument, Variable
from feedbax.tasks.registry import PythonTaskDefinition, PythonTaskRegistry, PythonTaskTemplate


REACHING_TASK = "feedbax.task.reaching.v1"
_POINT = ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH)

_TARGET_SIGNALS = {
    "simple-reaches": "target_pos",
    "delayed-reaches": "target_pos",
    "stabilization": "target_pos",
    "reaching": "reach_target",
}


def is_world_target(task: TaskDocument | PythonTaskSelection, signal: ScopedValue) -> bool:
    """Identify the qualified reaching target in the existing Task families."""
    target = "reach_target" if isinstance(task, PythonTaskSelection) and task.type == REACHING_TASK else (
        _TARGET_SIGNALS.get(task.id) if isinstance(task, TaskDocument) else None
    )
    return signal.id == target and signal.type == _POINT


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
        epochs=(Epoch(id="move", name="Move"),),
        signals=(Signal(
            id="reach_target", name="reach_target", type=_POINT,
            cells=(Cell(first_epoch="move", last_epoch="move", value=NameValue(
                kind="name", reference="target",
            )),),
        ),),
    )


def reaching_template(params: ReachingParameters) -> PythonTaskTemplate:
    document = reaching_document(params)
    def prepare(model, functions):
        from feedbax.tasks.document import compile_task

        return compile_task(document, functions, model=replace(model, inputs=()))

    return PythonTaskTemplate(
        signals={signal.name: signal.type for signal in document.signals}, prepare=prepare,
    )


def builtin_python_tasks() -> PythonTaskRegistry:
    registry = PythonTaskRegistry()
    registry.register(PythonTaskDefinition(
        REACHING_TASK, "Reaching", ReachingParameters, reaching_template,
        {
            "n_steps": Site(expected=ValueType(base=ValueBase.INTEGER), finest=Level.RUN, structural=True),
            "x": Site(expected=ValueType(base=ValueBase.REAL, dimensions=LENGTH), finest=Level.RUN),
            "y": Site(expected=ValueType(base=ValueBase.REAL, dimensions=LENGTH), finest=Level.RUN),
        },
    ))
    return registry
