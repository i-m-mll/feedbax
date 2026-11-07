"""Exact historical template endpoints."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .cde_templates import CDEStandard, CDEDecay, CDEAntiNF, CDEHybridV9b

HISTORIES = {
    CDEStandard: History(
        legacy=(LegacyEndpoint("CDE Standard", "1", 1, matches_unversioned=lambda params: True),)
    ),
    CDEDecay: History(
        legacy=(LegacyEndpoint("CDE + Decay", "1", 1, matches_unversioned=lambda params: True),)
    ),
    CDEAntiNF: History(
        legacy=(LegacyEndpoint("CDE + Anti-NF", "1", 1, matches_unversioned=lambda params: True),)
    ),
    CDEHybridV9b: History(
        legacy=(LegacyEndpoint("CDE Hybrid v9b", "1", 1, matches_unversioned=lambda params: True),)
    ),
}
