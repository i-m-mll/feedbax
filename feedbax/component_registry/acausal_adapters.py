"""Compose the built-in physical owners and their external histories."""

from .discovery import ComponentPack


def acausal_component_pack() -> ComponentPack:
    from feedbax.acausal import boundaries, mechanics, multibody, rotational, translational
    from feedbax.acausal.boundaries_history import HISTORIES as boundaries_history
    from feedbax.acausal.mechanics_history import HISTORIES as mechanics_history
    from feedbax.acausal.multibody_history import HISTORIES as multibody_history
    from feedbax.acausal.rotational_history import HISTORIES as rotational_history
    from feedbax.acausal.translational_history import HISTORIES as translational_history

    return ComponentPack(
        namespace="feedbax",
        modules=(boundaries, mechanics, multibody, rotational, translational),
        histories={**boundaries_history, **mechanics_history, **multibody_history, **rotational_history, **translational_history},
    )
