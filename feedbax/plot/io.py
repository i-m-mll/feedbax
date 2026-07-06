"""I/O helpers for saving figures alongside reproducibility specs.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import hashlib
import importlib.metadata
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Editable-install guard
# ---------------------------------------------------------------------------

_EDITABLE_INSTALL_CACHE: dict[str, bool] = {}
_MAX_PARENT_WALK = 8  # levels up from __file__ to look for .git


def _check_editable_install(package_module: ModuleType) -> None:
    """Warn if *package_module* is not in an editable (development) install.

    Walks up to ``_MAX_PARENT_WALK`` parent directories from the module's
    ``__file__`` looking for a ``.git`` directory.  If none is found, emits a
    ``WARNING`` — routing-to-package-relative-paths will silently write files
    into the wrong location (e.g. site-packages).

    Args:
        package_module: The top-level module of the application package.
    """
    module_file = getattr(package_module, "__file__", None)
    if module_file is None:
        logger.warning(
            "Package '%s' has no __file__ attribute; cannot verify editable install. "
            "Figure routing paths may be incorrect.",
            package_module.__name__,
        )
        return

    pkg_name = package_module.__name__
    if pkg_name in _EDITABLE_INSTALL_CACHE:
        if not _EDITABLE_INSTALL_CACHE[pkg_name]:
            logger.warning(
                "Package '%s' does not appear to be editable-installed (no .git found "
                "within %d levels of %s). Figure routing will write relative to that "
                "directory, which may not be the repo root.",
                pkg_name,
                _MAX_PARENT_WALK,
                module_file,
            )
        return

    path = Path(module_file).resolve()
    found = False
    for _ in range(_MAX_PARENT_WALK):
        path = path.parent
        if (path / ".git").exists():
            found = True
            break

    _EDITABLE_INSTALL_CACHE[pkg_name] = found

    if not found:
        logger.warning(
            "Package '%s' does not appear to be editable-installed (no .git found "
            "within %d levels of %s). Figure routing will write relative to that "
            "directory, which may not be the repo root.",
            pkg_name,
            _MAX_PARENT_WALK,
            module_file,
        )


def _find_repo_root(package_module: ModuleType) -> Path:
    """Return the repo root for *package_module* by walking parents for ``.git``.

    Args:
        package_module: The top-level module of the application package.

    Returns:
        The first parent directory containing ``.git``.

    Raises:
        ValueError: If no ``.git`` directory is found within ``_MAX_PARENT_WALK``
            levels, meaning the package does not appear to be editable-installed.
    """
    module_file = getattr(package_module, "__file__", None)
    if module_file is None:
        raise ValueError(
            f"Package '{package_module.__name__}' has no __file__ attribute; "
            "cannot determine repo root for figure routing."
        )

    path = Path(module_file).resolve()
    for _ in range(_MAX_PARENT_WALK):
        path = path.parent
        if (path / ".git").exists():
            return path

    raise ValueError(
        f"No .git directory found within {_MAX_PARENT_WALK} levels of "
        f"'{module_file}' for package '{package_module.__name__}'. "
        "Is the package editable-installed in a git checkout?"
    )


def _sha256(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_version(name: str) -> str:
    """Return installed version string for *name*, or ``'unknown'`` if not found."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _build_versions(extra_packages: Optional[list[str]] = None) -> dict[str, str]:
    """Collect version strings for feedbax, jax, plotly, and any extras."""
    core = ["feedbax", "jax", "plotly", "numpy", "equinox"]
    packages = core + (extra_packages or [])
    return {pkg: _package_version(pkg) for pkg in packages}


def save_figure_with_spec(
    fig: Any,
    spec: dict[str, Any],
    dst_dir: str | Path,
    *,
    name: Optional[str] = None,
    save_render: bool = True,
    render_format: str = "json",
    extra_packages: Optional[list[str]] = None,
) -> tuple[Path, Optional[Path]]:
    """Save a figure and a JSON reproducibility spec to *dst_dir*.

    The spec records input artifact paths and their SHA-256 digests, the
    data-transform pipeline, plot kwargs, version pins, a random seed, and a
    timestamp — everything needed to reproduce the figure from its sources.

    Arguments:
        fig: A ``plotly.graph_objs.Figure`` or ``matplotlib.figure.Figure``.
        spec: A dict with any subset of the following keys:

            - ``inputs`` (*list[dict]*): each entry has ``"path"`` and
              optionally ``"sha256"`` (computed automatically when absent).
            - ``transform`` (*list[dict]*): each entry has ``"name"`` and
              ``"kwargs"``.
            - ``plot_kwargs`` (*dict*): the kwargs passed to the plot function.
            - ``seed`` (*int | None*): random seed, if applicable.

            ``versions`` and ``timestamp`` are always added/overwritten by this
            function.
        dst_dir: Directory in which to write the output files.  Created if it
            does not exist.
        name: Base filename (without extension) for the output files.  When
            ``None`` the current UTC timestamp is used.
        save_render: Whether to also write the figure itself to disk.
        render_format: ``"json"`` (Plotly JSON; default), ``"html"``,
            ``"png"``, ``"svg"``, or ``"pdf"``.
        extra_packages: Additional package names to include in
            ``spec["versions"]``.

    Returns:
        A ``(spec_path, render_path)`` tuple.  *render_path* is ``None`` when
        ``save_render`` is ``False``.

    Raises:
        TypeError: If *fig* is not a recognised figure type and *save_render*
            is ``True``.
        ValueError: If *render_format* is not one of the supported formats.
        FileNotFoundError: If any path listed in ``spec["inputs"]`` does not
            exist when computing its SHA-256 digest.
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ------------------------------------------------------------------
    # Enrich the spec: resolve SHA-256 digests, add versions + timestamp
    # ------------------------------------------------------------------
    enriched: dict[str, Any] = dict(spec)  # shallow copy — do not mutate caller's dict

    # Resolve sha256 for each input artifact
    inputs = enriched.get("inputs", [])
    resolved_inputs = []
    for entry in inputs:
        entry = dict(entry)  # copy so we don't mutate
        path = Path(entry["path"])
        if "sha256" not in entry:
            entry["sha256"] = _sha256(path)
        entry["path"] = str(path)  # normalise to str for JSON serialisation
        resolved_inputs.append(entry)
    enriched["inputs"] = resolved_inputs

    # Always overwrite versions and timestamp
    enriched["versions"] = _build_versions(extra_packages)
    enriched["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Write spec JSON
    # ------------------------------------------------------------------
    spec_path = dst_dir / f"{name}.json"
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, sort_keys=True)
    logger.info("Wrote figure spec to %s", spec_path)

    # ------------------------------------------------------------------
    # Write figure render (optional)
    # ------------------------------------------------------------------
    render_path: Optional[Path] = None
    if save_render:
        render_path = _write_figure(fig, dst_dir, name, render_format)

    return spec_path, render_path


# ---------------------------------------------------------------------------
# Project-config-driven routing
# ---------------------------------------------------------------------------

def save_figure(
    fig: Any,
    spec: dict[str, Any],
    *,
    package: str,
    experiment: str,
    topic: str,
    extra_packages: Optional[list[str]] = None,
) -> dict[str, Optional[Path]]:
    """Save a figure using project-defined routing from the package registry.

    Looks up *package* in the ``EXPERIMENT_REGISTRY``, reads its
    ``figure_routing`` config, resolves the spec and render directories from
    templates, writes the spec JSON, writes the rendered figure, and
    optionally creates a relative symlink from the spec directory to the
    render file.

    The package must be registered with a ``figure_routing`` config (passed to
    ``register_package_from_module_info``).  An editable-install guard warns if
    the package's ``__file__`` is not inside a git checkout — in that case the
    repo-root-relative paths may be wrong.

    Args:
        fig: A ``plotly.graph_objs.Figure`` or ``matplotlib.figure.Figure``.
        spec: Reproducibility spec dict (same schema as ``save_figure_with_spec``).
            ``versions`` and ``timestamp`` are always added/overwritten.
        package: Name of the registered application package (e.g. ``"rlrmp"``).
        experiment: Experiment identifier substituted into directory templates
            (e.g. ``"part2_5"``).
        topic: Figure topic substituted into directory templates
            (e.g. ``"adversarial_losses"``).
        extra_packages: Additional package names to include in
            ``spec["versions"]``.

    Returns:
        A dict with keys:
        - ``"spec_path"`` (:class:`~pathlib.Path`): written spec JSON.
        - ``"render_path"`` (:class:`~pathlib.Path` or ``None``): written
          figure render, or ``None`` if ``render_format`` is absent from the
          routing config.
        - ``"symlink_path"`` (:class:`~pathlib.Path` or ``None``): symlink in
          the spec directory pointing at the render file, or ``None`` if
          ``create_symlink_in_spec_dir`` is ``False`` or not set.

    Raises:
        ValueError: If *package* is not registered, has no ``figure_routing``
            config, or the routing config is missing required keys.
        ValueError: If the package is not editable-installed and no repo root
            can be found (see :func:`_find_repo_root`).
    """
    # Late import to avoid circular dependency (plugins imports registry;
    # plot should not import plugins at module level).
    from feedbax.plugins import EXPERIMENT_REGISTRY  # noqa: PLC0415

    # Validate package and routing config
    figure_routing = EXPERIMENT_REGISTRY.get_figure_routing(package)
    if figure_routing is None:
        raise ValueError(
            f"Package '{package}' is registered but has no figure_routing configured. "
            "Pass figure_routing={{...}} to register_package_from_module_info."
        )

    _required_keys = {"spec_dir_template", "render_dir_template"}
    missing = _required_keys - figure_routing.keys()
    if missing:
        raise ValueError(
            f"Package '{package}' figure_routing config is missing required key(s): "
            f"{sorted(missing)!r}"
        )

    # Resolve package module and repo root
    package_metadata = EXPERIMENT_REGISTRY.get_package_metadata(package)
    package_module = package_metadata.package_module

    _check_editable_install(package_module)
    repo_root = _find_repo_root(package_module)

    # Resolve directories from templates
    spec_dir = repo_root / figure_routing["spec_dir_template"].format(
        experiment=experiment, topic=topic
    )
    render_dir = repo_root / figure_routing["render_dir_template"].format(
        experiment=experiment, topic=topic
    )

    render_format: str = figure_routing.get("render_format", "html")
    create_symlink: bool = figure_routing.get("create_symlink_in_spec_dir", False)

    # Determine figure filename extension
    ext = _render_extension(render_format)
    figure_filename = f"figure.{ext}"

    # Write spec + render using the existing helper
    spec_path, render_path = save_figure_with_spec(
        fig,
        spec,
        render_dir,
        name="figure",
        save_render=True,
        render_format=render_format,
        extra_packages=extra_packages,
    )

    # The spec JSON lives in spec_dir, not render_dir; move it there.
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_dest = spec_dir / "spec.json"
    if render_path is not None:
        # spec_path currently lives in render_dir as "figure.json" (from name="figure")
        # but we actually want the spec separate.  save_figure_with_spec wrote both to
        # render_dir; move only the spec JSON out.
        import shutil
        shutil.move(str(spec_path), str(spec_dest))
        spec_path = spec_dest
    else:
        import shutil
        shutil.move(str(spec_path), str(spec_dest))
        spec_path = spec_dest

    # Create relative symlink in spec_dir pointing at the render file
    symlink_path: Optional[Path] = None
    if create_symlink and render_path is not None:
        symlink_path = spec_dir / figure_filename
        rel_target = os.path.relpath(render_path, spec_dir)
        if symlink_path.is_symlink() or symlink_path.exists():
            symlink_path.unlink()
        symlink_path.symlink_to(rel_target)
        logger.info("Created symlink %s -> %s", symlink_path, rel_target)

    return {
        "spec_path": spec_path,
        "render_path": render_path,
        "symlink_path": symlink_path,
    }


def _render_extension(render_format: str) -> str:
    """Return the file extension for a given render format."""
    _ext_map = {
        "json": "fig.json",
        "html": "html",
        "png": "png",
        "svg": "svg",
        "pdf": "pdf",
    }
    try:
        return _ext_map[render_format]
    except KeyError as exc:
        supported = ", ".join(sorted(_ext_map))
        raise ValueError(
            f"Unsupported render_format {render_format!r}; expected one of: {supported}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_figure(
    fig: Any,
    dst_dir: Path,
    name: str,
    render_format: str,
) -> Path:
    """Dispatch figure serialisation based on type and *render_format*."""
    # Lazy import to avoid hard dependency at module level
    try:
        import plotly.graph_objs as go  # type: ignore[import]
        _plotly_figure = go.Figure
    except ImportError:
        _plotly_figure = None

    try:
        import matplotlib.figure as mpl_fig  # type: ignore[import]
        _mpl_figure = mpl_fig.Figure
    except ImportError:
        _mpl_figure = None

    if _plotly_figure is not None and isinstance(fig, _plotly_figure):
        return _write_plotly(fig, dst_dir, name, render_format)
    elif _mpl_figure is not None and isinstance(fig, _mpl_figure):
        return _write_matplotlib(fig, dst_dir, name, render_format)
    else:
        raise TypeError(
            f"Unrecognised figure type {type(fig)!r}.  "
            "Pass a plotly or matplotlib Figure, or set save_render=False."
        )


def _write_plotly(fig: Any, dst_dir: Path, name: str, render_format: str) -> Path:
    """Write a Plotly figure to disk."""
    _render_extension(render_format)
    if render_format == "json":
        path = dst_dir / f"{name}.fig.json"
        fig.write_json(str(path))
    elif render_format == "html":
        path = dst_dir / f"{name}.html"
        fig.write_html(str(path))
    else:
        path = dst_dir / f"{name}.{render_format}"
        fig.write_image(str(path))
    logger.info("Wrote Plotly figure to %s", path)
    return path


def _write_matplotlib(fig: Any, dst_dir: Path, name: str, render_format: str) -> Path:
    """Write a Matplotlib figure to disk."""
    _render_extension(render_format)
    ext = "png" if render_format == "json" else render_format
    path = dst_dir / f"{name}.{ext}"
    fig.savefig(str(path))
    logger.info("Wrote Matplotlib figure to %s", path)
    return path
