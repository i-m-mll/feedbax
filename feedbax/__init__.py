"""
:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import importlib
import importlib.metadata
import logging
import os

_LAZY_EXPORTS = {
    "BUILTIN_GRAPH_TEMPLATES": ("feedbax.contracts.graphs.templates", "BUILTIN_GRAPH_TEMPLATES"),
    "CDENetwork": ("feedbax.models.cde", "CDENetwork"),
    "CDENetworkState": ("feedbax.models.cde", "CDENetworkState"),
    "Component": ("feedbax.runtime.graph", "Component"),
    "Graph": ("feedbax.runtime.graph", "Graph"),
    "GraphTemplateMetadata": ("feedbax.contracts.graphs.templates", "GraphTemplateMetadata"),
    "LoweredContribution": ("feedbax.lowering", "LoweredContribution"),
    "LowererExecutionError": ("feedbax.lowering", "LowererExecutionError"),
    "LowererRegistration": ("feedbax.lowering", "LowererRegistration"),
    "OrderedLowererRegistry": ("feedbax.lowering", "OrderedLowererRegistry"),
    "Selection": ("feedbax.runtime.selectors", "Selection"),
    "Wire": ("feedbax.runtime.graph", "Wire"),
    "AbstractTask": ("feedbax.tasks", "AbstractTask"),
    "DELAYED_CENTER_OUT_PRESET": ("feedbax.tasks.presets", "DELAYED_CENTER_OUT_PRESET"),
    "DelayedReaches": ("feedbax.tasks", "DelayedReaches"),
    "DelayedReachTaskInputs": ("feedbax.tasks", "DelayedReachTaskInputs"),
    "SimpleReaches": ("feedbax.tasks", "SimpleReaches"),
    "TaskTrialSpec": ("feedbax.tasks", "TaskTrialSpec"),
    "TrialTimeline": ("feedbax.tasks", "TrialTimeline"),
    "WhereDict": ("feedbax.config.mapping", "WhereDict"),
    "centreout_endpoints": ("feedbax.tasks", "centreout_endpoints"),
    "delayed_center_out_reaches_params": (
        "feedbax.tasks.presets",
        "delayed_center_out_reaches_params",
    ),
    "eval_ensemble_on_trials": ("feedbax.tasks", "eval_ensemble_on_trials"),
    "forceless_task_inputs": ("feedbax.tasks", "_forceless_task_inputs"),
    "gen_epoch_lengths": ("feedbax.tasks", "gen_epoch_lengths"),
    "get_masks": ("feedbax.tasks", "get_masks"),
    "get_masked_seqs": ("feedbax.tasks", "get_masked_seqs"),
    "get_scalar_epoch_seq": ("feedbax.tasks", "get_scalar_epoch_seq"),
    "is_intervenor": ("feedbax.intervene", "is_intervenor"),
    "is_termtree": ("feedbax.objectives.loss", "is_termtree"),
    "init_state_from_component": ("feedbax.runtime.graph", "init_state_from_component"),
    "network_template_graph": ("feedbax.contracts.graphs.templates", "network_template_graph"),
    "pos_only_states": ("feedbax.tasks", "_pos_only_states"),
    "prepare_trial": ("feedbax.tasks", "prepare_trial"),
    "recurrent_controller_template_graph": (
        "feedbax.contracts.graphs.templates",
        "recurrent_controller_template_graph",
    ),
    "select": ("feedbax.runtime.selectors", "select"),
    "simple_feedback_template_graph": (
        "feedbax.contracts.graphs.templates",
        "simple_feedback_template_graph",
    ),
}

# from feedbax.config.logging import enable_logging_handlers

__all__ = [
    "AbstractTask",
    "BUILTIN_GRAPH_TEMPLATES",
    "CDENetwork",
    "CDENetworkState",
    "Component",
    "DelayedReaches",
    "DelayedReachTaskInputs",
    "DELAYED_CENTER_OUT_PRESET",
    "Graph",
    "GraphTemplateMetadata",
    "LOG_LEVEL",
    "LoweredContribution",
    "LowererExecutionError",
    "LowererRegistration",
    "OrderedLowererRegistry",
    "Selection",
    "SimpleReaches",
    "TaskTrialSpec",
    "TrialTimeline",
    "WhereDict",
    "Wire",
    "__version__",
    "centreout_endpoints",
    "delayed_center_out_reaches_params",
    "eval_ensemble_on_trials",
    "forceless_task_inputs",
    "gen_epoch_lengths",
    "init_state_from_component",
    "is_intervenor",
    "is_termtree",
    "get_masks",
    "get_masked_seqs",
    "get_scalar_epoch_seq",
    "network_template_graph",
    "pos_only_states",
    "prepare_trial",
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
