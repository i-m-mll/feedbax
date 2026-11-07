"""Compose the built-in physical owners and their external histories."""

from .discovery import ComponentPack


def acausal_component_pack() -> ComponentPack:
    from feedbax.acausal import boundaries, mechanics, multibody, rotational, translational
    from feedbax.component_migrations.acausal import HISTORIES

    return ComponentPack(
        namespace="feedbax",
        modules=(boundaries, mechanics, multibody, rotational, translational),
        histories=HISTORIES,
    )
