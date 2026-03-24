"""API router for analysis package discovery."""

from __future__ import annotations

import dataclasses
import inspect
import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class AnalysisClassInfo(BaseModel):
    """Describes a single analysis class available in a package."""

    name: str
    description: str
    category: str
    inputPorts: list[str]
    outputPorts: list[str]
    defaultParams: dict[str, Any]
    icon: str


class AnalysisPackageInfo(BaseModel):
    """A group of related analysis classes."""

    name: str
    description: str
    analyses: list[AnalysisClassInfo]


class AnalysisPackagesResponse(BaseModel):
    """Top-level response wrapper for the packages endpoint."""

    packages: list[AnalysisPackageInfo]


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def _get_input_ports(analysis_instance) -> list[str]:
    """Extract input port names from the analysis's Ports class.

    The Ports class is an equinox.Module subclass whose dataclass fields
    define the input ports.
    """
    ports_cls = analysis_instance.Ports
    try:
        fields = dataclasses.fields(ports_cls)
        return [f.name for f in fields]
    except TypeError:
        return []


def _get_output_ports(analysis_instance) -> list[str]:
    """Infer output port names from the analysis class.

    Convention: if the class overrides ``make_figs``, it produces a figure.
    Otherwise it produces generic data from ``compute``.
    """
    cls = type(analysis_instance)
    # Walk the MRO to check if make_figs is overridden at this class level
    has_make_figs = "make_figs" in cls.__dict__
    has_compute = "compute" in cls.__dict__

    if has_make_figs and has_compute:
        return ["data", "figure"]
    elif has_make_figs:
        return ["figure"]
    elif has_compute:
        return ["data"]
    else:
        return ["data"]


def _get_default_params(analysis_instance) -> dict[str, Any]:
    """Extract default parameter values from the analysis instance.

    Uses ``fig_params`` (the user-facing configuration surface) as the
    source of default parameters.
    """
    fig_params = getattr(analysis_instance, "fig_params", {})
    result = {}
    for key, value in fig_params.items():
        # Only include JSON-serializable values
        if isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        elif isinstance(value, dict):
            result[key] = value
        else:
            result[key] = str(value)
    return result


def _get_description(analysis_instance) -> str:
    """Extract a one-line description from the class docstring."""
    doc = type(analysis_instance).__doc__
    if doc:
        # Take the first non-empty line
        for line in doc.strip().splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return ""


def _introspect_analysis(
    analysis_instance,
    category: str,
) -> AnalysisClassInfo:
    """Build an AnalysisClassInfo from a live AbstractAnalysis instance."""
    return AnalysisClassInfo(
        name=type(analysis_instance).__name__,
        description=_get_description(analysis_instance),
        category=category,
        inputPorts=_get_input_ports(analysis_instance),
        outputPorts=_get_output_ports(analysis_instance),
        defaultParams=_get_default_params(analysis_instance),
        icon="",  # Frontend supplies icon defaults per class name
    )


def _discover_packages() -> list[AnalysisPackageInfo]:
    """Walk the experiment registry and introspect all analysis classes.

    Returns an empty list if no packages are installed or if introspection
    fails.  The frontend falls back to hardcoded stub data in that case.
    """
    try:
        from feedbax.plugins import EXPERIMENT_REGISTRY
    except Exception:
        logger.warning("Could not import EXPERIMENT_REGISTRY", exc_info=True)
        return []

    packages: list[AnalysisPackageInfo] = []

    for package_name in EXPERIMENT_REGISTRY.get_package_names():
        try:
            metadata = EXPERIMENT_REGISTRY.get_package_metadata(package_name)
        except ValueError:
            continue

        all_analyses: list[AnalysisClassInfo] = []
        seen_class_names: set[str] = set()

        for part in metadata.parts:
            module_key = part
            try:
                analysis_module = EXPERIMENT_REGISTRY.get_analysis_module(module_key)
            except (ValueError, ImportError):
                logger.debug(
                    "Could not load analysis module '%s' from package '%s'",
                    module_key, package_name,
                )
                continue

            analyses_dict = getattr(analysis_module, "ANALYSES", None)
            if not isinstance(analyses_dict, dict):
                continue

            category = part.replace("_", " ").title()

            for name, instance in analyses_dict.items():
                class_name = type(instance).__name__
                if class_name in seen_class_names:
                    continue
                seen_class_names.add(class_name)

                try:
                    info = _introspect_analysis(instance, category=category)
                    all_analyses.append(info)
                except Exception:
                    logger.warning(
                        "Failed to introspect analysis '%s' in %s/%s",
                        name, package_name, part,
                        exc_info=True,
                    )

        if all_analyses:
            packages.append(
                AnalysisPackageInfo(
                    name=package_name,
                    description=f"Analysis classes from the {package_name} experiment package",
                    analyses=all_analyses,
                )
            )

    return packages


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/packages")
async def list_analysis_packages() -> AnalysisPackagesResponse:
    """List all available analysis packages with their analysis classes.

    Introspects the experiment registry to discover installed packages,
    their parts, and the analysis classes defined in each part's analysis
    module.  Returns port definitions and default parameters for each
    analysis class so the frontend palette can be populated from live
    data instead of hardcoded stubs.
    """
    packages = _discover_packages()
    return AnalysisPackagesResponse(packages=packages)
