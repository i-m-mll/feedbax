"""Unit spellings and canonical magnitudes for authored expressions."""

from dataclasses import dataclass
from math import pi
from types import MappingProxyType

from feedbax.contracts.expression_values import ANGLE, DIMENSIONLESS, LENGTH, TIME, Dimensions


@dataclass(frozen=True)
class Unit:
    dimensions: Dimensions
    scale: float = 1.0


UNITS = MappingProxyType(
    {
        "s": Unit(TIME),
        "ms": Unit(TIME, 0.001),
        "st": Unit(TIME),
        "m": Unit(LENGTH),
        "m/s": Unit(LENGTH.combine(TIME, -1)),
        "cm": Unit(LENGTH, 0.01),
        "mm": Unit(LENGTH, 0.001),
        "rad": Unit(ANGLE),
        "°": Unit(ANGLE, pi / 180),
        "kg": Unit(Dimensions(mass=1)),
        "N": Unit(Dimensions(length=1, mass=1, time=-2)),
        "Hz": Unit(Dimensions(time=-1)),
        "1": Unit(DIMENSIONLESS),
        "": Unit(DIMENSIONLESS),
    }
)
