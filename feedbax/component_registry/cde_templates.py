"""Opt-in catalogue pack for the built-in CDE graph variants."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .discovery import ComponentPack


def cde_template_pack() -> ComponentPack:
    from feedbax.compiler.cde_templates import CDE_TEMPLATE_VARIANTS, _cde_template_definition

    from .discovery import ComponentPack

    return ComponentPack(
        components=tuple(
            partial(_cde_template_definition, *variant) for variant in CDE_TEMPLATE_VARIANTS
        ),
    )
