"""Resolve authored solver settings into Diffrax solvers."""

from math import isfinite
from typing import Literal

import diffrax as dfx
import optimistix as optx

from feedbax.contracts.acausal import AcausalPhysicalDomain, RootFinderSpec


PlanarSolverType = Literal["euler", "tsit5"]
DEFAULT_PLANAR_SOLVER: PlanarSolverType = "euler"
MIN_PLANAR_DT = 0.0001

_EXPLICIT_SOLVER_TYPES = {
    "euler": dfx.Euler,
    "tsit5": dfx.Tsit5,
}
_IMPLICIT_SOLVER_TYPES = {
    "implicit_euler": dfx.ImplicitEuler,
    "kvaerno5": dfx.Kvaerno5,
}


def resolve_acausal_solver(
    solver_name: str,
    dt: float,
    *,
    physical_domain: AcausalPhysicalDomain,
    root_finder: RootFinderSpec | None = None,
) -> dfx.AbstractSolver:
    """Resolve the solver for an executable acausal interior."""
    if not isfinite(dt) or dt <= 0:
        raise ValueError("Solver dt must be finite and positive")
    if physical_domain == "planar_multibody" and dt < MIN_PLANAR_DT:
        raise ValueError(f"Planar multibody solver dt must be at least {MIN_PLANAR_DT}")
    explicit_solver = _EXPLICIT_SOLVER_TYPES.get(solver_name)
    if explicit_solver is not None:
        if root_finder is not None:
            raise ValueError(f"Explicit solver {solver_name!r} does not accept root_finder")
        return explicit_solver()
    implicit_solver = _IMPLICIT_SOLVER_TYPES.get(solver_name)
    if implicit_solver is None:
        raise ValueError(f"Unsupported solver type {solver_name!r}")
    if physical_domain in {"planar_multibody", "spatial_multibody"}:
        label = "Planar" if physical_domain == "planar_multibody" else "Spatial"
        raise ValueError(
            f"{label} multibody integration requires an explicit solver; got {solver_name!r}"
        )
    root_spec = root_finder or RootFinderSpec()
    if root_spec.method != "newton":
        raise ValueError(f"Unsupported root finder method {root_spec.method!r}")
    if not isfinite(root_spec.rtol) or root_spec.rtol < 0:
        raise ValueError("Root finder rtol must be finite and nonnegative")
    if not isfinite(root_spec.atol) or root_spec.atol <= 0:
        raise ValueError("Root finder atol must be finite and positive")
    if root_spec.max_steps <= 0:
        raise ValueError("Root finder max_steps must be positive")
    newton = optx.Newton(rtol=root_spec.rtol, atol=root_spec.atol)
    if root_finder is None:
        # Keep the Diffrax iteration budget for interiors without authored root settings.
        return implicit_solver(root_finder=newton)
    return implicit_solver(root_finder=newton, root_find_max_steps=root_spec.max_steps)


def planar_solver_name(solver: dfx.AbstractSolver) -> str:
    """Recover the authored solver name from an analytical plant backend."""
    for name, solver_type in _EXPLICIT_SOLVER_TYPES.items():
        if type(solver) is solver_type:
            return name
    raise ValueError(
        f"AnalyticalMusculoskeletalPlant cannot serialize solver {type(solver).__name__!r}"
    )
