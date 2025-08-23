"""Experiment package registry for managing discovered packages."""

import importlib
from collections.abc import Sequence
from typing import Optional
from types import ModuleType
import logging

logger = logging.getLogger(__name__)


class PackageMetadata:
    """Metadata for a registered experiment package."""
    
    def __init__(
        self,
        name: str,
        package_module: ModuleType,
        parts: Sequence[str],
        analysis_module_root: str,
        training_module_root: str,
        config_resource_root: str,
    ):
        self.name = name
        self.package_module = package_module
        self.parts = list(parts)
        self.analysis_module_root = analysis_module_root
        self.training_module_root = training_module_root
        self.config_resource_root = config_resource_root


class ExperimentRegistry:
    """Registry for managing experiment packages and their components."""
    
    def __init__(self):
        self._packages: dict[str, PackageMetadata] = {}
        self._module_cache: dict[str, ModuleType] = {}
    
    def register_package(
        self,
        name: str,
        package_module: ModuleType,
        parts: Sequence[str],
        analysis_module_root: str,
        training_module_root: str,
        config_resource_root: str,
    ) -> None:
        """Register an experiment package with the registry.
        
        Args:
            name: Package name (e.g., "rlrmp")
            package_module: The imported package module
            parts: List of experiment parts (e.g., ["part1", "part2"])
            analysis_module_root: Subpackage path for analysis modules (e.g., "analysis.modules")
            training_module_root: Subpackage path for training modules (e.g., "training.modules")
            config_resource_root: Subpackage path for config resources (e.g., "config")
        """
        metadata = PackageMetadata(
            name=name,
            package_module=package_module,
            parts=parts,
            analysis_module_root=analysis_module_root,
            training_module_root=training_module_root,
            config_resource_root=config_resource_root,
        )
        self._packages[name] = metadata
        logger.info(f"Registered experiment package '{name}' with parts: {parts}")
    
    def get_package_names(self) -> list[str]:
        """Get list of registered package names."""
        return list(self._packages.keys())
    
    def get_package_metadata(self, package_name: str) -> PackageMetadata:
        """Get metadata for a registered package."""
        if package_name not in self._packages:
            raise ValueError(f"Package '{package_name}' not registered")
        return self._packages[package_name]
    
    def get_analysis_module(self, module_key: str) -> ModuleType:
        """Load an analysis module from the registered packages.
        
        Args:
            module_key: Module key like "part1.plant_perts"
            
        Returns:
            The loaded module
            
        Raises:
            ValueError: If the module cannot be found in any registered package
        """
        cache_key = f"analysis.{module_key}"
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]
        
        # Try to load from each registered package
        for package_name, metadata in self._packages.items():
            try:
                module_name = f"{metadata.package_module.__name__}.{metadata.analysis_module_root}.{module_key}"
                module = importlib.import_module(module_name)
                self._module_cache[cache_key] = module
                logger.debug(f"Loaded analysis module '{module_key}' from package '{package_name}'")
                return module
            except ModuleNotFoundError:
                continue
        
        raise ValueError(f"Analysis module '{module_key}' not found in any registered package")
    
    def get_training_module(self, module_key: str) -> ModuleType:
        """Load a training module from the registered packages.
        
        Args:
            module_key: Module key like "part1"
            
        Returns:
            The loaded module
            
        Raises:
            ValueError: If the module cannot be found in any registered package
        """
        cache_key = f"training.{module_key}"
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]
        
        # Try to load from each registered package
        for package_name, metadata in self._packages.items():
            try:
                module_name = f"{metadata.package_module.__name__}.{metadata.training_module_root}.{module_key}"
                module = importlib.import_module(module_name)
                self._module_cache[cache_key] = module
                logger.debug(f"Loaded training module '{module_key}' from package '{package_name}'")
                return module
            except ModuleNotFoundError:
                continue
        
        raise ValueError(f"Training module '{module_key}' not found in any registered package")
    
    def get_config_resource_root(self, module_key: str) -> tuple[str, str]:
        """Get the config resource root for a module.
        
        Args:
            module_key: Module key like "part1.plant_perts" 
            
        Returns:
            Tuple of (package_name, resource_root) for the package containing this module
            
        Raises:
            ValueError: If the module cannot be found in any registered package
        """
        # Determine which package contains this module by checking parts
        part = module_key.split(".")[0]
        
        for package_name, metadata in self._packages.items():
            if part in metadata.parts:
                resource_root = f"{metadata.package_module.__name__}.{metadata.config_resource_root}"
                return package_name, resource_root
        
        raise ValueError(f"Config resource for '{module_key}' not found in any registered package")


# Global registry instance
_default_registry: Optional[ExperimentRegistry] = None


def get_default_registry() -> ExperimentRegistry:
    """Get the default experiment registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ExperimentRegistry()
    return _default_registry