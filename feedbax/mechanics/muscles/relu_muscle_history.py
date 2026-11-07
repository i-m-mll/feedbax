"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .relu_muscle import ReluMuscle

HISTORIES = {
    ReluMuscle: History(legacy=(LegacyEndpoint("ReluMuscle", "1", 1),)),
}
