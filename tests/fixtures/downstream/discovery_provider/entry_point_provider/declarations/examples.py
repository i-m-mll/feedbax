from typing import Annotated

from pydantic import BaseModel, Field

from feedbax.analysis.operations import DatasetOutput, operation
from feedbax.contracts.component import PortType
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.figure_document import FigureDocument, FigurePanel, LineEncodings, LineLayer
from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.portable import ComponentOptions, Ports
from feedbax.dataset import Dataset, Variable
from feedbax.plot.templates import figure_template
from feedbax.runtime.graph import Component
from feedbax.tasks.registry import PythonTask, PythonTaskTemplate
from feedbax.training.methods import TrainingState, training_method


class ExternalGain(Component):
    ports = Ports({"input": PortType(dtype="any")}, {"output": PortType(dtype="any")})
    component_options = ComponentOptions(instantaneous=True)
    gain: float = 2.0

    def __call__(self, inputs, state, *, key):
        return {"output": self.gain * inputs["input"]}, state


class Settings(BaseModel):
    increments: list[int] = Field(default_factory=lambda: [5])
    unit: str = Field("test", json_schema_extra=lambda schema: schema.update(feedbax={"static": True}))


class ExternalTrial(PythonTask[Settings]):
    settings_model = Settings

    @staticmethod
    def template(settings: Settings) -> PythonTaskTemplate:
        from dataclasses import replace

        from feedbax.contracts.task_document import Epoch, Rest, TaskDocument
        from feedbax.tasks.document import compile_task

        document = TaskDocument(
            id="external", variables=(), signals=(), fixed_steps=sum(settings.increments),
            epochs=(Epoch(id="move", name="Move", end=Rest()),),
        )
        return PythonTaskTemplate(
            signals={}, prepare=lambda model, functions: compile_task(document, functions, model=replace(model, inputs=())),
        )


def initialize(settings: Settings, model, key):
    return TrainingState(model)


@training_method(settings=Settings, init=initialize)
def ExternalMethod(state, ctx):
    return state, {}


_PORT = DatasetPort(schema_ids=("fixture.series.v1",), variables={"value": FieldDescriptor(dims=("step",))})


@operation(settings=Settings, renamed_from=("entry_point_provider.offset_series",))
def shift_series(
    source: Annotated[Dataset, _PORT], params: Settings,
) -> Annotated[Dataset, DatasetOutput("shifted", _PORT)]:
    return Dataset.from_arrays(
        {"value": Variable(source.data("value") + sum(params.increments), ("step",))},
        schema_id="fixture.series.v1",
    )


@figure_template(settings=Settings)
def external_plot(settings: Settings) -> FigureDocument:
    return FigureDocument(
        data=("source",),
        layout=(FigurePanel(name="series", row=1, col=1),),
        layers=(LineLayer(
            mark="line", name="values", role="source", panel="series",
            encode=LineEncodings(x="step", y="value"),
        ),),
        caption=settings.unit,
    )
