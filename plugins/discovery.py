"""Automatic discovery of experiment packages via entry points."""

import importlib
import importlib.metadata
from typing import Optional
import logging

from .registry import ExperimentRegistry, get_default_registry

logger = logging.getLogger(__name__)


def discover_experiment_packages(
    registry: Optional[ExperimentRegistry] = None,
    entry_point_group: str = "feedbax_experiments.packages"
) -> ExperimentRegistry:
    """Discover and register experiment packages using entry points.
    
    Args:
        registry: Registry instance to populate (creates new one if None)
        entry_point_group: Entry point group to search for packages
        
    Returns:
        The populated registry
    """
    if registry is None:
        registry = get_default_registry()
    
    # Discover packages via entry points
    try:
        # Python 3.10+ syntax
        entry_points = importlib.metadata.entry_points(group=entry_point_group)
    except TypeError:
        # Fallback for older Python versions
        all_entry_points = importlib.metadata.entry_points()
        if hasattr(all_entry_points, 'get'):
            entry_points = all_entry_points.get(entry_point_group, [])
        else:
            # Even older versions - filter manually
            entry_points = [ep for ep in all_entry_points if ep.group == entry_point_group]
    
    for entry_point in entry_points:
        try:
            # Load the registration function
            register_func = entry_point.load()
            
            # Call it to register the package
            register_func(registry)
            
            logger.info(f"Discovered experiment package '{entry_point.name}' via entry point")
            
        except Exception as e:
            logger.warning(f"Failed to load experiment package '{entry_point.name}': {e}")
            continue
    
    if not registry.get_package_names():
        logger.warning(f"No experiment packages found in entry point group '{entry_point_group}'")
    
    return registry


def register_package_from_module_info(
    registry: ExperimentRegistry,
    package_name: str,
    package_module_name: str,
    parts: list[str],
    analysis_module_root: str = "analysis.modules", 
    training_module_root: str = "training.modules",
    config_resource_root: str = "config",
) -> None:
    """Helper function to register a package from module information.
    
    Args:
        registry: Registry to register with
        package_name: Name of the package (e.g., "rlrmp")  
        package_module_name: Python module name (e.g., "rlrmp")
        parts: List of experiment parts (e.g., ["part1", "part2"])
        analysis_module_root: Subpackage path for analysis modules
        training_module_root: Subpackage path for training modules  
        config_resource_root: Subpackage path for config resources
    """
    try:
        package_module = importlib.import_module(package_module_name)
        registry.register_package(
            name=package_name,
            package_module=package_module,
            parts=parts,
            analysis_module_root=analysis_module_root,
            training_module_root=training_module_root,
            config_resource_root=config_resource_root,
        )
    except ImportError as e:
        logger.error(f"Failed to import package module '{package_module_name}': {e}")
        raise