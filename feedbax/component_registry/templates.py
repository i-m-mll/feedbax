"""Composition of built-in graph template owners and histories."""

from .discovery import ComponentPack


def builtin_graph_template_pack() -> ComponentPack:
    from feedbax.compiler import mechanics_templates, templates
    from feedbax.compiler.mechanics_templates_history import HISTORIES as mechanics_history
    from feedbax.compiler.templates_history import HISTORIES as graph_history

    return ComponentPack(namespace="feedbax", modules=(templates, mechanics_templates), histories={**mechanics_history, **graph_history})


def builtin_muscle_template_pack() -> ComponentPack:
    from feedbax.compiler import muscle_templates
    from feedbax.compiler.muscle_templates_history import HISTORIES

    return ComponentPack(namespace="feedbax", modules=(muscle_templates,), histories=HISTORIES)
