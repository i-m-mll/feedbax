"""Positive controls for the three Python channels.

Nothing here runs. The scanner reads it with the ast module.
"""

from __future__ import annotations

import feedbax.contracts.graph as graph_mod
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts import SparseCooArrayValueSpec, materialize_array_value
from feedbax.lowering import *  # noqa: F403 - star-import control
from feedbax.plugins import (
    TRAINING_METHODS,
    FamilyRequirement,
    PluginRegistration,
    RegistrationContext,
)


def _control() -> object:
    """Attribute-channel control: a module alias resolved to a guaranteed name."""
    return (
        graph_mod.GraphSpec,
        GraphSpec,
        SparseCooArrayValueSpec,
        materialize_array_value,
        TRAINING_METHODS,
        FamilyRequirement,
        PluginRegistration,
        RegistrationContext,
    )
