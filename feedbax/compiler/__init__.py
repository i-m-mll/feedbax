"""Scientific graph compiler public contract."""

import importlib


_GRAPH_EXPORTS = (
    "COMPILATION_FAILURE_SCHEMA_ID",
    "COMPILATION_FAILURE_SCHEMA_VERSION",
    "COMPILATION_RECORD_SCHEMA_ID",
    "COMPILATION_RECORD_SCHEMA_VERSION",
    "GRAPH_COMPILER_ID",
    "GRAPH_COMPILER_VERSION",
    "GRAPH_DOCUMENT_SCHEMA_ID",
    "GRAPH_DOCUMENT_SCHEMA_VERSION",
    "RESOLVED_GRAPH_SCHEMA_ID",
    "RESOLVED_GRAPH_SCHEMA_VERSION",
    "CompilationFailureRecord",
    "CompilationRecord",
    "CompilerDiagnostic",
    "CompilerPhase",
    "DiagnosticSeverity",
    "DocumentRoot",
    "ExecutableGraph",
    "GraphCompilationError",
    "GraphDocument",
    "GraphKeySchedule",
    "GraphSourceMap",
    "GraphSourceMapEntry",
    "ResolvedGraph",
    "compile_graph",
)

_LAZY_EXPORTS = {name: ("feedbax.compiler.graph", name) for name in _GRAPH_EXPORTS}

__all__ = list(_GRAPH_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value
