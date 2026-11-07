"""Composition of CDE graph template owners and histories."""

from .discovery import ComponentPack


def cde_template_pack() -> ComponentPack:
    from feedbax.compiler import cde_templates
    from feedbax.component_migrations.cde_templates import HISTORIES

    return ComponentPack(namespace="feedbax", modules=(cde_templates,), histories=HISTORIES)
