"""I/O helpers for saving figures alongside reproducibility specs.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import hashlib
import importlib.metadata
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
            ``"png"``, ``"svg"``, or any format accepted by
            ``fig.write_image`` / ``fig.savefig``.
        extra_packages: Additional package names to include in
            ``spec["versions"]``.

    Returns:
        A ``(spec_path, render_path)`` tuple.  *render_path* is ``None`` when
        ``save_render`` is ``False``.

    Raises:
        TypeError: If *fig* is not a recognised figure type and *save_render*
            is ``True``.
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
    ext = "png" if render_format == "json" else render_format
    path = dst_dir / f"{name}.{ext}"
    fig.savefig(str(path))
    logger.info("Wrote Matplotlib figure to %s", path)
    return path
