"""Skeleton for analysis modules imported in __init__.py.

For example, `analysis.part1.plant_perts` is such a module.
"""

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import Optional

from equinox import Module

from rlrmp.analysis import AbstractAnalysis
from rlrmp.analysis.analysis import NoPorts
from rlrmp.analysis.state_utils import vmap_eval_ensemble
from rlrmp.types import TreeNamespace
from rlrmp.types import LDict


# Import transform machinery from execution module
from rlrmp.analysis.execution import AnalysisModuleTransformSpec


"""Specify any additional colorscales needed for this analysis. 
These will be included in the `colors` kwarg passed to `AbstractAnalysis` methods
"""
COLOR_FUNCS: dict[str, Callable[[TreeNamespace], Sequence]] = dict(
    some_variable=lambda hps: hps.some_variable,  #! e.g.
)


def setup_eval_tasks_and_models(task_base: Module, models_base: LDict[float, Module], hps: TreeNamespace):
    """Specify how to set up the PyTrees of evaluation tasks and models, given a base task and 
    a spread of models.
    
    Also, make any necessary modifications to `hps` as they will be available during analysis. 
    """
    # Trivial example
    tasks = task_base
    models = models_base 
    
    # Provides any additional data needed for the analysis
    extras = SimpleNamespace()  
    
    return tasks, models, hps, extras

    
"""Depending on the structure of `setup_eval_tasks_and_models`, e.g. the use of `vmap`, it may be 
necessary to define a more complex function here.

For example, check out `analysis.part2.unit_perts`.
"""
eval_func: Callable = vmap_eval_ensemble


# Define any subclasses of `AbstractAnalysis` that are specific to this task
class SomeAnalysis(AbstractAnalysis[NoPorts]):
    variant: Optional[str] = "full"
    
    ...
 
   
"""Specify transformations to apply at different stages of analysis execution.
If not present, no transformations are applied."""
TRANSFORMS = AnalysisModuleTransformSpec(
    # Examples:
    # pre_setup=dict(models=get_best_replicate),  # Granular - apply only to models
    # pre_setup=dict(task=some_task_transform, models=get_best_replicate),  # Both
    # pre_setup=lambda task, models: (task, tree_map(get_best_replicate, models)),  # Combined function
    
    # post_eval=dict(states=some_states_transform),  # Granular - states only (backward compatible)
    # post_eval=dict(models=some_model_transform, states=some_states_transform),  # Multiple components
    # post_eval=lambda models, tasks, states: (models, tasks, transformed_states),  # Combined function
)


"""Determines which analyses are performed by `run_analysis.py`, for this module."""
ANALYSES = {
    "analysis_label": SomeAnalysis(),
}

"""Analyses which may be referenced as inputs to other analyses, 
but whose results are not returned, nor figures rendered."""
DEPENDENCIES = {
    # "dep_label": SomeOtherAnalysis(),
}