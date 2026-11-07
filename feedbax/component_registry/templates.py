"""Composition of built-in graph template owners and histories."""

from .discovery import ComponentPack


def builtin_graph_template_pack() -> ComponentPack:
    from feedbax.compiler import mechanics_templates, templates
    from feedbax.component_migrations.graph_templates import HISTORIES

    return ComponentPack(
        namespace="feedbax",
        modules=(templates, mechanics_templates),
        histories=HISTORIES,
    )


def builtin_muscle_template_pack() -> ComponentPack:
    from feedbax.compiler import muscle_templates
    from feedbax.component_migrations.muscle_templates import HISTORIES

    return ComponentPack(namespace="feedbax", modules=(muscle_templates,), histories=HISTORIES)
