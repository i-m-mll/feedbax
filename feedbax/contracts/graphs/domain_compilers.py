"""Domain compiler registry for non-causal graph interiors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from feedbax.runtime.graph import Component


DomainCompiler = Callable[[Any, str, Any], Component]


_COMPILERS: dict[str, DomainCompiler] = {}
_BUILTINS_REGISTERED = False


def register_domain_compiler(compiler_id: str, fn: DomainCompiler) -> None:
    """Register a compiler for a graph-domain ``compiler_id``."""

    if not compiler_id:
        raise ValueError("Domain compiler id must be non-empty")
    if not callable(fn):
        raise TypeError(f"Domain compiler {compiler_id!r} must be callable")
    _COMPILERS[compiler_id] = fn


def get_domain_compiler(compiler_id: str) -> DomainCompiler:
    """Return a registered compiler.

    Builtin compilers are registered lazily so importing this module does not
    import JAX/diffrax-heavy runtime code unless a non-causal interior is built.
    """

    register_builtin_domain_compilers()
    try:
        return _COMPILERS[compiler_id]
    except KeyError as exc:
        raise ValueError(f"No domain compiler registered for {compiler_id!r}") from exc


def register_builtin_domain_compilers() -> None:
    """Register Feedbax builtin domain compilers once."""

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from feedbax.contracts.graphs.acausal_compiler import compile_acausal_graph
    from feedbax.contracts.graphs.penzai_compiler import PENZAI_COMPILER_ID

    register_domain_compiler("feedbax.compiler.acausal", compile_acausal_graph)
    register_domain_compiler(
        PENZAI_COMPILER_ID,
        lambda interior_spec, node_name, component_registry: interior_spec,
    )
    _BUILTINS_REGISTERED = True
