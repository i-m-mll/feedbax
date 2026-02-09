"""Core types for the acausal modeling framework.

Acausal components are construction-time equation descriptors (plain Python
dataclasses) that get compiled into a single ``vector_field(t, y, args)``
at ``__init__`` time.  The resulting ``AcausalSystem`` is a standard
``DAEComponent`` that the feedbax Graph treats like any other component.

Domains
-------
Each physical domain (translational, rotational) defines:
- **Across variables**: quantities that are *shared* at a connection point
  (position/velocity for translational; angle/angular_vel for rotational).
- **Through variable**: the quantity that is *summed* at a connection node
  (force for translational; torque for rotational).

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0.  See LICENSE for details.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

class Domain(Enum):
    """Physical domain for acausal ports."""
    TRANSLATIONAL = "translational"
    ROTATIONAL = "rotational"


# ---------------------------------------------------------------------------
# Port / Variable / Equation descriptors
# ---------------------------------------------------------------------------

@dataclass
class AcausalPort:
    """A physical connection point on an element.

    Attributes:
        name: Port identifier (unique within its element).
        domain: Physical domain the port belongs to.
        across_vars: Names of across-variable slots, ordered.
            E.g. ``("pos", "vel")`` for translational.
        through_var: Name of the through-variable slot.
            E.g. ``"force"`` for translational.
    """
    name: str
    domain: Domain
    across_vars: tuple[str, ...]
    through_var: str


@dataclass
class AcausalVar:
    """A scalar variable in the assembled acausal system.

    Attributes:
        name: Fully qualified name ``"element.port.slot"``.
        is_differential: ``True`` when this variable has a d/dt equation.
        is_eliminated: ``True`` when this variable was removed by a
            connection constraint and replaced by its canonical alias.
        alias_of: If eliminated, the canonical variable name.
        is_grounded: ``True`` when the variable is fixed at zero.
        is_input: ``True`` when the variable is driven by a causal input.
        initial_value: Starting value for the variable.
    """
    name: str
    is_differential: bool = True
    is_eliminated: bool = False
    alias_of: Optional[str] = None
    is_grounded: bool = False
    is_input: bool = False
    initial_value: float = 0.0


@dataclass
class AcausalEquation:
    """An equation contributed by an acausal element.

    The equation can be either:
    - **Differential**: ``d(lhs_var)/dt = rhs_fn(var_values)``
    - **Algebraic**: ``0 = rhs_fn(var_values)``  (used for through-var
      definitions like ``force = k * (x_a - x_b)``).

    Attributes:
        lhs_var: Variable name this equation defines (or whose derivative
            it defines, depending on ``is_through_def``).
        rhs_fn: ``Callable[[dict[str, scalar]], scalar]``.  The dict maps
            fully-qualified variable names to their current values.
        depends_on: Variable names read by ``rhs_fn``.
        is_through_def: If ``True``, this equation *defines* a through
            variable (force/torque) rather than a time derivative.
    """
    lhs_var: str
    rhs_fn: Callable
    depends_on: tuple[str, ...]
    is_through_def: bool = False


# ---------------------------------------------------------------------------
# Element / Connection
# ---------------------------------------------------------------------------

@dataclass
class AcausalElement:
    """Base for acausal element descriptors (construction-time only).

    Subclasses populate ``ports``, ``equations``, and ``params`` during
    ``__init__`` to describe their physics.

    Attributes:
        name: Unique element identifier.
        ports: Mapping ``port_name -> AcausalPort``.
        equations: List of equations this element contributes.
        params: Named scalar parameters (e.g. mass, stiffness).
        element_type: Discriminator string used by the assembly algorithm
            to identify special elements (``"ground"``, ``"force_source"``,
            ``"prescribed_motion"``, ``"sensor"``, ``"gear_ratio"``).
            Regular elements leave this as ``"standard"``.
        sensor_output: If this element is a sensor, the
            ``(port_name, slot_name)`` it reads.
    """
    name: str
    ports: dict[str, AcausalPort] = dc_field(default_factory=dict)
    equations: list[AcausalEquation] = dc_field(default_factory=list)
    params: dict[str, float] = dc_field(default_factory=dict)
    element_type: str = "standard"
    sensor_output: Optional[tuple[str, str]] = None


@dataclass
class AcausalConnection:
    """Connection between two ports on (possibly different) elements.

    Attributes:
        element_a: Name of the first element.
        port_a: Port name on the first element.
        element_b: Name of the second element.
        port_b: Port name on the second element.
    """
    element_a: str
    port_a: str
    element_b: str
    port_b: str

    def __init__(
        self,
        port_a: tuple[str, str],
        port_b: tuple[str, str],
    ):
        """Create a connection from two ``(element_name, port_name)`` tuples."""
        self.element_a, self.port_a = port_a
        self.element_b, self.port_b = port_b


# ---------------------------------------------------------------------------
# State layout (output of the assembly step)
# ---------------------------------------------------------------------------

@dataclass
class StateLayout:
    """Maps variable names to indices in the flat state vector.

    Built by the assembly algorithm; consumed by the compiled vector field
    and by ``AcausalSystem`` to bridge between named variables and JAX
    arrays.

    Attributes:
        _vars: All variables in the system (including eliminated ones).
        _differential: Ordered list of differential variable names that
            appear in the state vector ``y``.
        _algebraic: Algebraic variable names (not yet used).
        _eliminated: ``var_name -> canonical_var_name``.
        _grounded: Set of variable names fixed at zero.
        _inputs: ``var_name -> index`` in the causal input array.
        _input_specs: ``var_name -> (element_name, slot_index)``.
        _outputs: ``output_label -> var_name`` for sensor readings.
        total_size: Length of the flat state vector.
    """
    _vars: dict[str, AcausalVar]
    _differential: list[str]
    _algebraic: list[str]
    _eliminated: dict[str, str]
    _grounded: set[str]
    _inputs: dict[str, int]
    _input_specs: dict[str, tuple[str, int]]
    _outputs: dict[str, str]
    total_size: int

    def var_index(self, name: str) -> int:
        """Return the index of *name* in the differential state vector."""
        return self._differential.index(name)

    def resolve(self, name: str) -> str:
        """Follow elimination aliases to the canonical variable name."""
        visited: set[str] = set()
        while name in self._eliminated:
            if name in visited:
                raise RuntimeError(f"Cyclic alias chain for '{name}'")
            visited.add(name)
            name = self._eliminated[name]
        return name
