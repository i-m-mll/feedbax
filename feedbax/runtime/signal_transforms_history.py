"""Frozen historical wire endpoints for native component owners."""

from feedbax.component_registry.history import Change, History, LegacyEndpoint

from .signal_transforms import AddNoise, ConstantInput, NetworkClamp, NetworkConstantInput


def _legacy_clamp(params):
    return "scale" in params


def _drop_clamp_scale(params):
    return {name: value for name, value in params.items() if name != "scale"}


def _legacy_addition(params):
    return set(params) <= {"scale", "active"}


HISTORIES = {
    AddNoise: History(
        legacy=(LegacyEndpoint('AddNoise', '1', 1), LegacyEndpoint('AddNoise', None, 1)),
    ),
    NetworkClamp: History(
        Change(2, migrate=_drop_clamp_scale),
        legacy=(
            LegacyEndpoint("NetworkClamp", "1", 1, matches_unversioned=_legacy_clamp),
            LegacyEndpoint("NetworkClamp", "2", 2),
        ),
    ),
    NetworkConstantInput: History(
        Change(2, identity=True),
        legacy=(
            LegacyEndpoint("NetworkConstantInput", "1", 1, matches_unversioned=_legacy_addition),
            LegacyEndpoint("NetworkConstantInput", "2", 2),
        ),
    ),
    ConstantInput: History(
        Change(2, identity=True),
        legacy=(
            LegacyEndpoint("ConstantInput", "1", 1, matches_unversioned=_legacy_addition),
            LegacyEndpoint("ConstantInput", "2", 2),
        ),
    ),
}
