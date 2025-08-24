"""Experiment package registry for managing discovered packages."""

import importlib
import logging
from collections.abc import Sequence
from importlib import resources
from importlib.util import find_spec
from types import ModuleType
from typing import Literal, Optional

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

    def _parse_module_key(self, module_key: str) -> tuple[str | None, str]:
        """Parse a module key that may be package-prefixed.

        Accepted forms:
        - 'pkg/part1.some_analysis'   (preferred)
        - 'pkg.part1.some_analysis'   (legacy explicit)
        - 'part1.some_analysis'       (unqualified)
        Returns:
        (package_name or None, relative_dotted_key)
        """
        # Preferred "pkg/..." form
        if "/" in module_key:
            pkg, rest = module_key.split("/", 1)
            if not rest:
                raise ValueError(f"Empty module name after package prefix '{pkg}/'.")
            if pkg not in self._packages:
                raise ValueError(f"Package '{pkg}' not found in registry")
            return pkg, rest

        # Legacy explicit "pkg.part..." form (only if the first token is a real package)
        if "." in module_key:
            first, rest = module_key.split(".", 1)
            if first in self._packages:
                # (Optionally: warnings.warn("Use 'pkg/...'", DeprecationWarning))
                return first, rest

        # Unqualified module key
        return None, module_key

    def _find_packages_with_module(self, module_key: str, module_type: str) -> list[str]:
        """Find all packages that contain the given module.

        Args:
            module_key: Module key like "part1"
            module_type: Either "training" or "analysis"

        Returns:
            List of package names that contain this module
        """
        matching_packages = []

        for package_name, metadata in self._packages.items():
            try:
                if module_type == "training":
                    module_name = f"{metadata.package_module.__name__}.{metadata.training_module_root}.{module_key}"
                elif module_type == "analysis":
                    module_name = f"{metadata.package_module.__name__}.{metadata.analysis_module_root}.{module_key}"
                else:
                    raise ValueError(f"Unknown module type: {module_type}")

                find_spec(module_name)
                matching_packages.append(package_name)
            except (ModuleNotFoundError, ImportError):
                continue

        return matching_packages

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

    def single_package_name(self) -> str | None:
        pkgs = list(self._packages.keys())
        return pkgs[0] if len(pkgs) == 1 else None

    def _split_explicit_pkg(self, key: str) -> tuple[str, str]:
        """Return (pkg, rest) if key is 'pkg.rest'. Error if no explicit pkg or unknown pkg."""
        if "." not in key:
            raise ValueError(
                f"Key '{key}' is not package-qualified. Use 'pkg.{key}' or pass a domain_hint."
            )
        pkg, rest = key.split(".", 1)
        if pkg not in self._packages:
            raise ValueError(f"Package '{pkg}' not found in registry")
        return pkg, rest

    def _canonicalize_key(
        self,
        key: str,
        *,
        kind: Literal["module", "batch"],
        domain: Optional[Literal["training", "analysis"]] = None,
    ) -> tuple[str, str]:
        """
        Resolve a user-facing key into (package_name, relative_key).

        Grammar:
        - Explicit: 'pkg/...'  (package-qualified)
        - Unqualified: '...'   (must resolve uniquely across packages)

        For kind="module": relative_key is a dotted path like "part2.plant_perts".
        For kind="batch":  relative_key is a simple name like "my_batch".

        Raises ValueError on ambiguity or not found.
        """
        # 1) Explicit package via slash
        if "/" in key:
            pkg, rest = key.split("/", 1)
            if not rest:
                raise ValueError(f"Empty {kind} key after package prefix '{pkg}/'.")
            if pkg not in self._packages:
                raise ValueError(f"Package '{pkg}' not found in registry")
            return pkg, rest

        # 2) Unqualified; prefer the single-package convenience
        single = self.single_package_name()
        if single:
            return single, key

        if not self._packages:
            raise ValueError(f"No packages are registered; cannot resolve '{key}'.")

        # 3) Probe for a unique owning package
        matches: list[str] = []
        if kind == "module":
            if domain is None:
                raise ValueError("domain is required for module resolution")
            matches = self._find_packages_with_module(key, domain)
        else:  # kind == "batch"
            if domain is None:
                raise ValueError("domain is required for batch resolution")
            for pkg_name, md in self._packages.items():
                root = f"{md.package_module.__name__}.{md.config_resource_root}.batched.{domain}"
                try:
                    if resources.files(root).joinpath(f"{key}.yml").is_file():
                        matches.append(pkg_name)
                except Exception:
                    # package may not provide this batched namespace
                    pass

        if not matches:
            raise ValueError(
                f"{kind.title()} '{key}' not found in any registered package for domain '{domain}'."
            )
        if len(matches) > 1:
            opts = "', '".join(f"{p}/{key}" for p in matches)
            raise ValueError(
                f"{kind.title()} '{key}' is ambiguous across packages. Use one of '{opts}'."
            )

        return matches[0], key

    def resolve_package_for_module_key(
        self, key: str, *, domain: Literal["training", "analysis"]
    ) -> str:
        pkg, _ = self._canonicalize_key(key, kind="module", domain=domain)
        return pkg

    def resolve_package_for_batch_key(
        self, key: str, *, domain: Literal["training", "analysis"]
    ) -> str:
        pkg, _ = self._canonicalize_key(key, kind="batch", domain=domain)
        return pkg

    def get_module(self, module_key: str, domain: Literal["training", "analysis"]) -> ModuleType:
        """Load a module by key for the given domain."""
        pkg, rel_key = self._canonicalize_key(module_key, kind="module", domain=domain)

        cache_key = f"{domain}@{pkg}:{rel_key}"
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]

        md = self._packages[pkg]
        module_root = md.training_module_root if domain == "training" else md.analysis_module_root
        module_name = ".".join([md.package_module.__name__, module_root, rel_key])

        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise ValueError(
                f"{domain.title()} module '{rel_key}' not found in package '{pkg}'."
            ) from e

        self._module_cache[cache_key] = module
        logger.debug(f"Loaded {domain} module '{rel_key}' from package '{pkg}'")
        return module

    def get_analysis_module(self, module_key: str) -> ModuleType:
        return self.get_module(module_key, "analysis")

    def get_training_module(self, module_key: str) -> ModuleType:
        return self.get_module(module_key, "training")

    def get_config_resource_root(
        self,
        module_key: str,
        domain: Optional[Literal["training", "analysis"]] = None,
    ) -> tuple[str, str]:
        """
        Return (package_name, resource_root) where resource_root == '{pkg_module}.config'.

        Accepted keys:
        - Global-style: 'pkg/resource'        → use explicit package; ignore 'resource'
        - Module-style: 'pkg/partX.foo'       → use explicit package
        - Module-style: 'partX.foo'           → resolve uniquely across packages
            * If 'domain' is provided, only probe that domain
            * If 'domain' is None, probe both domains and require a unique match
        - Global-style without explicit package ('resource'):
            * If exactly one package is registered, use it; else raise ambiguity

        Raises:
        ValueError on ambiguity, unknown package, or inability to resolve.
        """
        # 1) Explicit package via slash form — covers both global ('pkg/resource')
        #    and module ('pkg/partX.foo') callers. Domain is irrelevant here.
        if "/" in module_key:
            pkg, _ = module_key.split("/", 1)
            if not pkg:
                raise ValueError("Empty package in '<pkg>/...' key.")
            if pkg not in self._packages:
                raise ValueError(f"Package '{pkg}' not found in registry")
            md = self._packages[pkg]
            return pkg, f"{md.package_module.__name__}.{md.config_resource_root}"

        # 2) No slash — could be:
        #    a) global resource without explicit package ('paths'), or
        #    b) module key ('partX.foo').
        #
        #    If caller passes domain, treat as (b) and resolve in that domain only.
        #    If domain is None:
        #       - If exactly one package is installed, use it (works for a or b).
        #       - Else, try to resolve (b) by probing both domains; if unique, use that pkg.
        #       - If still ambiguous (or not found), ask caller to qualify with '<pkg>/'.
        single = self.single_package_name()
        if single:
            md = self._packages[single]
            return single, f"{md.package_module.__name__}.{md.config_resource_root}"

        if domain is not None:
            # Domain-constrained module resolution (unqualified 'partX.foo')
            pkg, _rel = self._canonicalize_key(module_key, kind="module", domain=domain)
            md = self._packages[pkg]
            return pkg, f"{md.package_module.__name__}.{md.config_resource_root}"

        # domain is None and multiple packages registered → try probing both domains
        analysis_matches = set(self._find_packages_with_module(module_key, "analysis"))
        training_matches = set(self._find_packages_with_module(module_key, "training"))
        matches = sorted(analysis_matches | training_matches)

        if not matches:
            # At this point we cannot tell if 'module_key' was meant as a global 'resource'
            # or a module. Require explicit '<pkg>/resource' or '<pkg>/module'.
            pkgs = "', '".join(sorted(self._packages.keys()))
            raise ValueError(
                f"Cannot resolve '{module_key}' to a package with multiple packages installed "
                f"({pkgs}). Use '<package>/{module_key}'."
            )
        if len(matches) > 1:
            suggestions = "', '".join(f"{p}/{module_key}" for p in matches)
            raise ValueError(
                f"'{module_key}' is ambiguous across packages. Use one of '{suggestions}'."
            )

        pkg = matches[0]
        md = self._packages[pkg]
        return pkg, f"{md.package_module.__name__}.{md.config_resource_root}"


# Global registry instance
_default_registry: Optional[ExperimentRegistry] = None


def get_default_registry() -> ExperimentRegistry:
    """Get the default experiment registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ExperimentRegistry()
    return _default_registry
