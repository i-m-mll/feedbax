"""Automatic discovery of experiment packages via entry points."""

import importlib
import importlib.metadata
import logging
from typing import Optional

from .registry import ExperimentRegistry, get_default_registry

logger = logging.getLogger(__name__)


def discover_experiment_packages(
    registry: Optional[ExperimentRegistry] = None,
    entry_point_group: str = "feedbax.plugins",
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
        if hasattr(all_entry_points, "get"):
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
            if _is_recipe_validation_error(e):
                raise RuntimeError(
                    f"Failed to register experiment package {entry_point.name!r}: {e}"
                ) from e
            logger.warning(f"Failed to load experiment package '{entry_point.name}': {e}")
            continue

    if not registry.get_package_names():
        logger.warning(f"No experiment packages found in entry point group '{entry_point_group}'")

    return registry


def _is_recipe_validation_error(exc: Exception) -> bool:
    exc_type = type(exc)
    return (
        exc_type.__module__ == "feedbax.analysis.validation"
        and exc_type.__name__ == "RecipeValidationError"
    )


def register_package_from_module_info(
    registry: ExperimentRegistry,
    package_name: str,
    package_module_name: str,
    parts: list[str],
    analysis_module_root: str = "modules.analysis",
    training_module_root: str = "modules.training",
    config_resource_root: str = "config",
    figure_routing: Optional[dict] = None,
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
        figure_routing: Optional routing config dict for figure saving.  When provided,
            ``feedbax.plot.save_figure`` will use it to resolve spec and render directories
            relative to the package's repository root.  Schema::

                {
                    "spec_dir_template": "results/{experiment}/figures/{topic}",
                    "render_dir_template": "_artifacts/{experiment}/figures/{topic}",
                    "spec_format": "json",
                    "render_format": "html",
                    "create_symlink_in_spec_dir": True,
                }

            The template variables ``{experiment}`` and ``{topic}`` are substituted at
            save time.  Omit (or pass ``None``) if figure routing is not needed for this
            package; calling ``feedbax.plot.save_figure`` with such a package will raise
            a descriptive ``ValueError``.
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
            figure_routing=figure_routing,
        )
    except ImportError as e:
        logger.error(f"Failed to import package module '{package_module_name}': {e}")
        raise
