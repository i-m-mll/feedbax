"""Dataset fields of the Model state shared by rollout planning and recording."""

from dataclasses import dataclass

from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import ANGLE, LENGTH, TIME, SignalSpace
from feedbax.contracts.model_values import StateValue
from feedbax.contracts.semantic_values import semantic_key


_QUANTITIES = {
    LENGTH.combine(TIME, -1): "velocity",
    ANGLE: "angle",
    ANGLE.combine(TIME, -1): "angular_velocity",
    UNITS["N"].dimensions: "force",
    UNITS["N*m"].dimensions: "torque",
}


@dataclass(frozen=True)
class RetainedStateField:
    key: str
    axes: tuple[str, ...]
    units: str | None
    space: SignalSpace | None = None
    position: bool = False
    configuration: bool = False

    def descriptor(self, sample_axes: tuple[str, ...]) -> FieldDescriptor:
        unit = UNITS.get(self.units)
        quantity = None
        if unit is not None:
            if self.space == "point":
                quantity = "position"
            elif self.space == "free" and unit.dimensions == LENGTH:
                quantity = "displacement"
            else:
                quantity = _QUANTITIES.get(unit.dimensions)
        return FieldDescriptor(
            dims=(*sample_axes, "state_time", *self.axes), units=self.units, quantity=quantity,
            frame="world" if self.space is not None else None,
        )


def owner_state_field(
    owner: tuple[str, ...], declaration: StateValue, shape: tuple[int, ...], *, suffix: str = "",
) -> RetainedStateField:
    """Name one storage leaf from its owner declaration and resolved array shape."""
    symbol = declaration.symbol + suffix
    occurrence = "/".join(owner)
    if declaration.subtree:
        axes = tuple(f"{occurrence}:{symbol}:axis{axis}" for axis in range(len(shape)))
    elif declaration.axis == "xy":
        if shape != (2,):
            raise ValueError(f"{symbol} requires two Cartesian coordinates")
        axes = ("xy",)
    elif declaration.axis == "muscle" and shape == ():
        axes = ()
    else:
        if len(shape) != 1:
            raise ValueError(f"{symbol} requires a one-dimensional owner axis")
        axes = (f"{occurrence}:{declaration.axis}",)
    return RetainedStateField(
        semantic_key("model", "state", owner, symbol), axes, declaration.units,
        space=declaration.space, position=declaration.space == "point", configuration=declaration.configuration,
    )
