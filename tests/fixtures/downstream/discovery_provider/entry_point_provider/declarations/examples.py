from typing import Annotated

from pydantic import BaseModel, Field

from feedbax import Gain
from feedbax.analysis.operations import DatasetOutput, operation
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.figure_document import FigureDocument, FigurePanel, LineEncodings, LineLayer
from feedbax.contracts.operation import DatasetPort
from feedbax.dataset import Dataset, Variable
from feedbax.plot.templates import figure_template
from feedbax.tasks.registry import PythonTask, PythonTaskTemplate
from feedbax.training.methods import TrainingState, training_method


class ExternalGain(Gain):
    pass


class Settings(BaseModel):
    increments: list[int] = Field(default_factory=lambda: [5])
    unit: str = Field("test", json_schema_extra=lambda schema: schema.update(feedbax={"static": True}))


class ExternalTrial(PythonTask[Settings]):
    settings_model = Settings

    @staticmethod
    def template(settings: Settings) -> PythonTaskTemplate:
        from feedbax.builtins.reaching import Reaching, ReachingParameters

        return Reaching.template(ReachingParameters(n_steps=sum(settings.increments)))


def initialize(settings: Settings, model, key):
    return TrainingState(model)


@training_method(settings=Settings, init=initialize)
def ExternalMethod(state, ctx):
    return state, {}


_PORT = DatasetPort(schema_ids=("fixture.series.v1",), variables={"value": FieldDescriptor(dims=("step",))})


@operation(settings=Settings, revision=2, renamed_from=("entry_point_provider.offset_series",))
def shift_series(
    source: Annotated[Dataset, _PORT], params: Settings,
) -> Annotated[Dataset, DatasetOutput("shifted", _PORT)]:
    return Dataset.from_arrays(
        {"value": Variable(source.data("value") + sum(params.increments), ("step",))},
        schema_id="fixture.series.v1",
    )


@figure_template(settings=Settings, revision=4)
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
