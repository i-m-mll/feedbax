"""Opt-in catalogue pack for the built-in CDE graph variants."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Protocol

from feedbax.compiler.cde_templates import CDE_TEMPLATE_VARIANTS, _cde_template_definition

if TYPE_CHECKING:
    from .declarations import DeclaredComponent
    from .discovery import ComponentPack


def cde_template_pack() -> ComponentPack:
    from .discovery import ComponentPack

    return ComponentPack(
        components=tuple(
            partial(_cde_template_definition, *variant) for variant in CDE_TEMPLATE_VARIANTS
        ),
    )


class _Registry(Protocol):
    def register(self, meta: DeclaredComponent) -> None: ...


def register_cde_templates(registry: _Registry) -> None:
    for variant in CDE_TEMPLATE_VARIANTS:
        registry.register(_cde_template_definition(*variant))
