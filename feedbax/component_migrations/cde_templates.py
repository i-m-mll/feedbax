"""Frozen component migration records for cde templates."""

from feedbax.compiler.cde_templates import CDEAntiNF, CDEDecay, CDEHybridV9b, CDEStandard
from feedbax.component_registry.history import History, LegacyEndpoint

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
