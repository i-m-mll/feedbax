from typing import Annotated

from feedbax.analysis.operations import DatasetOutput, operation
from feedbax.builtins.runtime.components import Linear
from feedbax.builtins.operations import _SPEED, _SPEED_STATES, TrialSpeedParameters, trial_speed
from feedbax.dataset import Dataset


@operation(settings=TrialSpeedParameters)
def speed(
    states: Annotated[Dataset, _SPEED_STATES], params: TrialSpeedParameters,
) -> Annotated[Dataset, DatasetOutput("speed", _SPEED)]:
    return trial_speed(states, params)


class Readout(Linear):
    """The built-in linear map, declared by this package."""
