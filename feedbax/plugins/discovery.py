"""Automatic discovery of feedbax plugins via entry points."""

import importlib
import importlib.metadata
import logging
import inspect
from collections.abc import Iterable, Sequence
from typing import Any, Optional, cast

from .registry import ExperimentRegistry, get_default_registry

logger = logging.getLogger(__name__)

DEFAULT_ENTRY_POINT_GROUP = "feedbax.plugins"
TRAINING_METHOD_REGISTRAR_NAMES = (
    "register_feedbax_training_methods",
    "register_training_methods",
)


def discover_experiment_packages(
    registry: Optional[ExperimentRegistry] = None,
    entry_point_group: str = DEFAULT_ENTRY_POINT_GROUP,
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

    entry_points = feedbax_plugin_entry_points(entry_point_group)

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


def feedbax_plugin_entry_points(entry_point_group: str = DEFAULT_ENTRY_POINT_GROUP) -> Iterable[Any]:
    """Return entry points published through the shared Feedbax plugin group."""
    try:
        return importlib.metadata.entry_points(group=entry_point_group)
    except TypeError:
        all_entry_points = importlib.metadata.entry_points()
        entry_points_getter = getattr(all_entry_points, "get", None)
        if callable(entry_points_getter):
            return cast(Iterable[Any], entry_points_getter(entry_point_group, []))
        return [
            entry_point
            for entry_point in all_entry_points
            if entry_point.group == entry_point_group
        ]


def load_training_method_plugins(
    *,
    registry: Any | None = None,
    entry_point_group: str = DEFAULT_ENTRY_POINT_GROUP,
    entry_points: Iterable[Any] | None = None,
    modules: Sequence[str] | None = None,
) -> None:
    """Load training-method registration hooks from Feedbax plugins.

    Entry points share the existing ``feedbax.plugins`` group. A plugin can expose
    ``register_feedbax_training_methods(registry)`` or ``register_training_methods(registry)``
    on the loaded object/module. Explicit module names are imported as an escape
    hatch for local/downstream code that is not installed as an entry point.
    """
    if registry is None:
        from feedbax.contracts.training import DEFAULT_TRAINING_METHOD_REGISTRY

        registry = DEFAULT_TRAINING_METHOD_REGISTRY

    for entry_point in entry_points if entry_points is not None else feedbax_plugin_entry_points(
        entry_point_group
    ):
        provenance = _entry_point_provenance(entry_point)
        try:
            plugin = entry_point.load()
        except Exception as exc:
            logger.warning("Failed to load Feedbax plugin entry point %s: %s", provenance, exc)
            continue
        _register_training_methods_from_plugin(plugin, registry, provenance=provenance)

    for module_name in modules or ():
        module = importlib.import_module(module_name)
        _register_training_methods_from_plugin(module, registry, provenance=f"module:{module_name}")


def _register_training_methods_from_plugin(
    plugin: Any,
    registry: Any,
    *,
    provenance: str,
) -> None:
    registrar = _training_method_registrar(plugin)
    if registrar is None:
        return
    try:
        registrar(registry)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to register Feedbax training methods from {provenance}: {exc}"
        ) from exc
    logger.info("Registered Feedbax training methods from %s", provenance)


def _training_method_registrar(plugin: Any) -> Any | None:
    for attr in TRAINING_METHOD_REGISTRAR_NAMES:
        registrar = getattr(plugin, attr, None)
        if callable(registrar):
            return registrar
    if not callable(plugin):
        return None
    try:
        signature = inspect.signature(plugin)
    except (TypeError, ValueError):
        return None
    parameter_names = set(signature.parameters)
    if parameter_names & {"training_method_registry", "method_registry", "training_methods"}:
        return plugin
    return None


def _entry_point_provenance(entry_point: Any) -> str:
    dist = getattr(entry_point, "dist", None)
    if dist is not None:
        metadata = getattr(dist, "metadata", {})
        package_name = metadata.get("Name") if hasattr(metadata, "get") else None
        if package_name:
            return f"package:{package_name}"
    return f"entry-point:{getattr(entry_point, 'name', '<unknown>')}"


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
