"""
:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import importlib
import importlib.metadata
import logging
import os

_LAZY_EXPORTS = {
    "cde_controller": ("feedbax.models.cde", "cde_controller"),
    "cde_controller_spec": ("feedbax.builtins.templates.cde_templates", "cde_controller_spec"),
    "Component": ("feedbax.runtime.graph", "Component"),
    "GraphNode": ("feedbax.runtime.graph", "GraphNode"),
    "Graph": ("feedbax.runtime.graph", "Graph"),
    "LoweredContribution": ("feedbax.lowering", "LoweredContribution"),
    "LowererExecutionError": ("feedbax.lowering", "LowererExecutionError"),
    "LowererRegistration": ("feedbax.lowering", "LowererRegistration"),
    "OrderedLowererRegistry": ("feedbax.lowering", "OrderedLowererRegistry"),
    "Selection": ("feedbax.runtime.selectors", "Selection"),
    "Wire": ("feedbax.runtime.graph", "Wire"),
    "WhereDict": ("feedbax.runtime.mapping", "WhereDict"),
    "is_termtree": ("feedbax.objectives.base", "is_termtree"),
    "init_state_from_component": ("feedbax.runtime.graph", "init_state_from_component"),
    "network_template_graph": ("feedbax.builtins.templates.templates", "network_template_graph"),
    "recurrent_controller_template_graph": (
        "feedbax.builtins.templates.templates",
        "recurrent_controller_template_graph",
    ),
    "select": ("feedbax.runtime.selectors", "select"),
    "simple_feedback_template_graph": (
        "feedbax.builtins.templates.templates",
        "simple_feedback_template_graph",
    ),
}

# from feedbax.config.logging import enable_logging_handlers

__all__ = [
    "Component",
    "GraphNode",
    "Graph",
    "LOG_LEVEL",
    "LoweredContribution",
    "LowererExecutionError",
    "LowererRegistration",
    "OrderedLowererRegistry",
    "Selection",
    "WhereDict",
    "Wire",
    "__version__",
    "cde_controller",
    "cde_controller_spec",
    "init_state_from_component",
    "is_termtree",
    "network_template_graph",
    "recurrent_controller_template_graph",
    "select",
    "simple_feedback_template_graph",
]


__version__ = importlib.metadata.version("feedbax")


# logging.config.fileConfig('../logging.conf')

if os.environ.get("FEEDBAX_DEBUG", False) == "True":
    DEFAULT_LOG_LEVEL = "DEBUG"
else:
    DEFAULT_LOG_LEVEL = "INFO"

LOG_LEVEL = os.environ.get("FEEDBAX_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()


logger = logging.getLogger(__package__)
logger.addHandler(logging.NullHandler())


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value
