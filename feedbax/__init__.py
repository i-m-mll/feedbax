"""
:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import importlib.metadata
import logging
import os

from feedbax.config.mapping import WhereDict
from feedbax.runtime.selectors import Selection, select
from feedbax.runtime.graph import Component, Graph, Wire, init_state_from_component
from feedbax.contracts.graphs.templates import (
    BUILTIN_GRAPH_TEMPLATES,
    GraphTemplateMetadata,
    network_template_graph,
    recurrent_controller_template_graph,
    simple_feedback_template_graph,
)
from feedbax.models.cde import CDENetwork, CDENetworkState
from feedbax.intervene import is_intervenor
from feedbax.objectives.loss import is_termtree
from feedbax.tasks import (
    AbstractTask,
    DelayedReaches,
    DelayedReachTaskInputs,
    SimpleReaches,
    TaskTrialSpec,
    TrialTimeline,
    _forceless_task_inputs as forceless_task_inputs,
    _pos_only_states as pos_only_states,
    centreout_endpoints,
    gen_epoch_lengths,
    get_masks,
    get_masked_seqs,
    get_scalar_epoch_seq,
    prepare_trial,
)
from feedbax.tasks.presets import (
    DELAYED_CENTER_OUT_PRESET,
    delayed_center_out_reaches_params,
)

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
    "Selection",
    "SimpleReaches",
    "TaskTrialSpec",
    "TrialTimeline",
    "WhereDict",
    "Wire",
    "__version__",
    "centreout_endpoints",
    "delayed_center_out_reaches_params",
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
