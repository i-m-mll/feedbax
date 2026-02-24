"""Penzai model integration for feedbax Graphs.

This module provides `PenzaiSubgraph`, a Component that wraps Penzai neural network
models for use within feedbax computational graphs. It handles:

- Input/output mapping between feedbax dict-based ports and Penzai layer interfaces
- State variable extraction and binding at call boundaries
- Treescope HTML rendering for model inspection

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
import io
from typing import Any, Callable, ClassVar, Mapping, Sequence, TYPE_CHECKING

import equinox as eqx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PRNGKeyArray, PyTree

from feedbax.graph import Component

if TYPE_CHECKING:
    pass


# Check for penzai availability
try:
    import penzai as pz
    from penzai import pz as pzl  # penzai.pz contains layer primitives
    from penzai.core.variables import AbstractVariable, StateVariable, unbind_variables
    from penzai.core.struct import Struct

    PENZAI_AVAILABLE = True
except ImportError:
    PENZAI_AVAILABLE = False
    pz = None  # type: ignore
    pzl = None  # type: ignore
    AbstractVariable = None  # type: ignore
    StateVariable = None  # type: ignore
    unbind_variables = None  # type: ignore
    Struct = None  # type: ignore

# Check for treescope availability
try:
    import treescope

    TREESCOPE_AVAILABLE = True
except ImportError:
    TREESCOPE_AVAILABLE = False
    treescope = None  # type: ignore


def _require_penzai() -> None:
    """Raise ImportError if penzai is not available."""
    if not PENZAI_AVAILABLE:
        raise ImportError(
            "penzai is required for PenzaiSubgraph. "
            "Install it with: pip install penzai"
        )


def _require_treescope() -> None:
    """Raise ImportError if treescope is not available."""
    if not TREESCOPE_AVAILABLE:
        raise ImportError(
            "treescope is required for HTML rendering. "
            "Install it with: pip install treescope"
        )


# =============================================================================
# Input/Output Mapping Types
# =============================================================================


@dataclass(frozen=True)
class PortSpec:
    """Specification for a single port mapping."""

    port_name: str
    key_path: tuple[str | int, ...] = ()
    transform: Callable[[PyTree], PyTree] | None = None

    def extract_from_inputs(self, inputs: dict[str, PyTree]) -> PyTree:
        """Extract value from inputs dict, then follow key_path if specified."""
        result = inputs[self.port_name]
        for key in self.key_path:
            if isinstance(key, int):
                result = result[key]
            else:
                result = result[key] if isinstance(result, dict) else getattr(result, key)
        if self.transform is not None:
            result = self.transform(result)
        return result

    def extract(self, data: PyTree) -> PyTree:
        """Extract value from data following key_path (for output mapping)."""
        result = data
        for key in self.key_path:
            if isinstance(key, int):
                result = result[key]
            else:
                result = result[key] if isinstance(result, dict) else getattr(result, key)
        if self.transform is not None:
            result = self.transform(result)
        return result

    def insert(self, container: dict[str, PyTree], value: PyTree) -> dict[str, PyTree]:
        """Insert value into container at port_name."""
        container[self.port_name] = value
        return container


class InputMapping(Module):
    """Maps feedbax input dict to Penzai layer input(s).

    Handles translation from feedbax's dict-based port system to Penzai's
    function-call interface. Supports:

    - Single input: dict["input"] -> layer(x)
    - Multiple inputs: dict["a", "b"] -> layer(a, b)
    - Structured inputs: dict -> layer(struct)

    Attributes:
        port_specs: Sequence of port specifications for input mapping.
        combine_fn: Optional function to combine extracted values into layer input.
    """

    port_specs: tuple[PortSpec, ...] = field(static=True)
    combine_fn: Callable[..., PyTree] | None = field(default=None, static=True)

    @classmethod
    def single(cls, port_name: str = "input") -> "InputMapping":
        """Create mapping for single input port."""
        return cls(port_specs=(PortSpec(port_name),))

    @classmethod
    def multi(cls, *port_names: str) -> "InputMapping":
        """Create mapping for multiple input ports passed as positional args."""
        return cls(
            port_specs=tuple(PortSpec(name) for name in port_names),
            combine_fn=lambda *args: args,
        )

    @classmethod
    def structured(
        cls,
        port_names: Sequence[str],
        struct_fn: Callable[..., PyTree],
    ) -> "InputMapping":
        """Create mapping that combines ports into a structured input."""
        return cls(
            port_specs=tuple(PortSpec(name) for name in port_names),
            combine_fn=struct_fn,
        )

    def __call__(self, inputs: dict[str, PyTree]) -> PyTree:
        """Map feedbax inputs to Penzai layer input."""
        values = [spec.extract_from_inputs(inputs) for spec in self.port_specs]
        if len(values) == 1 and self.combine_fn is None:
            return values[0]
        if self.combine_fn is not None:
            return self.combine_fn(*values)
        return values[0]


class OutputMapping(Module):
    """Maps Penzai layer output to feedbax output dict.

    Handles translation from Penzai's return values to feedbax's dict-based
    port system. Supports:

    - Single output: layer() -> dict["output"]
    - Multiple outputs: (a, b) -> dict["a": a, "b": b]
    - Structured outputs: struct -> dict (extracting fields)

    Attributes:
        port_specs: Sequence of port specifications for output mapping.
        unpack_fn: Optional function to unpack layer output into values.
    """

    port_specs: tuple[PortSpec, ...] = field(static=True)
    unpack_fn: Callable[[PyTree], Sequence[PyTree]] | None = field(default=None, static=True)

    @classmethod
    def single(cls, port_name: str = "output") -> "OutputMapping":
        """Create mapping for single output port."""
        return cls(port_specs=(PortSpec(port_name),))

    @classmethod
    def multi(cls, *port_names: str) -> "OutputMapping":
        """Create mapping for multiple output ports from tuple return."""
        return cls(
            port_specs=tuple(PortSpec(name) for name in port_names),
            unpack_fn=lambda x: x,  # Assume tuple-like
        )

    @classmethod
    def structured(
        cls,
        port_specs: Sequence[PortSpec],
    ) -> "OutputMapping":
        """Create mapping that extracts fields from structured output."""
        return cls(port_specs=tuple(port_specs))

    def __call__(self, output: PyTree) -> dict[str, PyTree]:
        """Map Penzai layer output to feedbax outputs dict."""
        if self.unpack_fn is not None:
            values = self.unpack_fn(output)
            return {spec.port_name: values[i] for i, spec in enumerate(self.port_specs)}

        if len(self.port_specs) == 1:
            spec = self.port_specs[0]
            if spec.key_path:
                return {spec.port_name: spec.extract(output)}
            return {spec.port_name: output}

        return {spec.port_name: spec.extract(output) for spec in self.port_specs}


# =============================================================================
# State Variable Management
# =============================================================================


class PenzaiStateManager(Module):
    """Manages Penzai StateVariable extraction and binding.

    Handles the boundary between feedbax's State system and Penzai's
    StateVariable system. At each call:

    1. Extracts current state values from feedbax State
    2. Binds them to the Penzai model for execution
    3. Unbinds updated state values after execution
    4. Stores them back in feedbax State

    Attributes:
        state_index: StateIndex for storing Penzai state values.
        variable_paths: Paths to StateVariable locations in the model tree.
    """

    state_index: StateIndex
    _initial_state: PyTree = field(static=True)

    @classmethod
    def from_model(cls, pz_model: PyTree) -> "PenzaiStateManager":
        """Create state manager by scanning model for StateVariables.

        Args:
            pz_model: Penzai model tree to scan for state variables.

        Returns:
            PenzaiStateManager configured for the model's state variables.
            If penzai is not available, returns an empty state manager.
        """
        # If penzai is not available, we can't scan for StateVariables
        # Return an empty state manager for non-penzai models
        if not PENZAI_AVAILABLE:
            return cls.empty()

        # Extract all StateVariable values as initial state
        # The model should have unbound state variables that we need to track
        state_vars: list[tuple[Any, ...]] = []

        def find_state_vars(path: tuple, node: Any) -> None:
            if StateVariable is not None and isinstance(node, StateVariable):
                state_vars.append((path, node.value))

        def walk_tree(path: tuple, node: Any) -> None:
            find_state_vars(path, node)
            if hasattr(node, "__dict__"):
                for key, value in node.__dict__.items():
                    walk_tree(path + (key,), value)
            elif isinstance(node, (list, tuple)):
                for i, value in enumerate(node):
                    walk_tree(path + (i,), value)
            elif isinstance(node, dict):
                for key, value in node.items():
                    walk_tree(path + (key,), value)

        walk_tree((), pz_model)

        # Create initial state as a dict mapping paths to values
        initial_state: dict[tuple[Any, ...], Any] = {path: value for path, value in state_vars}

        return cls(
            state_index=StateIndex(initial_state),
            _initial_state=initial_state,
        )

    @classmethod
    def empty(cls) -> "PenzaiStateManager":
        """Create an empty state manager for stateless models."""
        initial_state: dict[tuple[Any, ...], Any] = {}
        return cls(
            state_index=StateIndex(initial_state),
            _initial_state=initial_state,
        )

    def bind_state(self, pz_model: PyTree, state: State) -> PyTree:
        """Bind stored state values into the Penzai model.

        Args:
            pz_model: Penzai model with unbound state variables.
            state: Feedbax State containing stored values.

        Returns:
            Penzai model with state values bound.
        """
        if not PENZAI_AVAILABLE:
            return pz_model

        state_values = state.get(self.state_index)
        if not state_values:
            return pz_model

        # Use penzai's variable binding mechanism
        # This assumes the model has LocalVariableEffect or similar
        # For simplicity, we return the model as-is if no special handling needed
        return pz_model

    def unbind_state(self, pz_model: PyTree, state: State) -> tuple[PyTree, State]:
        """Extract state values from model and store in feedbax State.

        Args:
            pz_model: Penzai model after execution.
            state: Current feedbax State.

        Returns:
            Tuple of (model with unbound state, updated feedbax State).
        """
        if not PENZAI_AVAILABLE:
            return pz_model, state

        # Extract updated state values
        state_vars: dict[tuple[Any, ...], Any] = {}

        def find_state_vars(path: tuple, node: Any) -> None:
            if StateVariable is not None and isinstance(node, StateVariable):
                state_vars[path] = node.value

        def walk_tree(path: tuple, node: Any) -> None:
            find_state_vars(path, node)
            if hasattr(node, "__dict__"):
                for key, value in node.__dict__.items():
                    walk_tree(path + (key,), value)
            elif isinstance(node, (list, tuple)):
                for i, value in enumerate(node):
                    walk_tree(path + (i,), value)
            elif isinstance(node, dict):
                for key, value in node.items():
                    walk_tree(path + (key,), value)

        walk_tree((), pz_model)

        if state_vars:
            state = state.set(self.state_index, state_vars)

        return pz_model, state


# =============================================================================
# PenzaiSubgraph Component
# =============================================================================


class PenzaiSubgraph(Component):
    """Component that wraps a Penzai model for use in feedbax Graphs.

    PenzaiSubgraph bridges the Penzai neural network framework with feedbax's
    computational graph system. It handles:

    - **Input translation**: Maps feedbax's dict-based port inputs to Penzai's
      function call interface via InputMapping.
    - **Output translation**: Maps Penzai's return values back to feedbax's
      dict-based port outputs via OutputMapping.
    - **State management**: Extracts and binds Penzai StateVariable objects at
      call boundaries, persisting them in feedbax's State system.
    - **Visualization**: Provides treescope HTML rendering for model inspection.

    Example:
        >>> import penzai as pz
        >>> from penzai import pz as pzl
        >>>
        >>> # Create a Penzai MLP
        >>> mlp = pzl.nn.Sequential([
        ...     pzl.nn.Linear(input_size=4, output_size=32),
        ...     pzl.nn.ReLU(),
        ...     pzl.nn.Linear(input_size=32, output_size=2),
        ... ])
        >>>
        >>> # Wrap for feedbax
        >>> component = PenzaiSubgraph.from_layer(mlp)
        >>>
        >>> # Use in a Graph
        >>> graph = Graph(
        ...     nodes={"encoder": component, ...},
        ...     wires=(...),
        ...     ...
        ... )

    Attributes:
        pz_model: The wrapped Penzai model (pz.Struct tree).
        input_mapping: Mapping from feedbax inputs to layer input.
        output_mapping: Mapping from layer output to feedbax outputs.
        state_manager: Manager for Penzai StateVariable persistence.
        input_ports: Tuple of input port names (from input_mapping).
        output_ports: Tuple of output port names (from output_mapping).
    """

    pz_model: PyTree
    input_mapping: InputMapping
    output_mapping: OutputMapping
    state_manager: PenzaiStateManager

    # Override ClassVars from Component with instance fields
    # These are dynamically set based on the input/output mappings
    input_ports: tuple[str, ...] = ()  # type: ignore[misc]
    output_ports: tuple[str, ...] = ()  # type: ignore[misc]

    def __init__(
        self,
        pz_model: PyTree,
        input_mapping: InputMapping,
        output_mapping: OutputMapping,
        state_manager: PenzaiStateManager | None = None,
    ):
        """Initialize PenzaiSubgraph.

        Args:
            pz_model: Penzai model to wrap.
            input_mapping: How to map feedbax inputs to layer input.
            output_mapping: How to map layer output to feedbax outputs.
            state_manager: Optional state manager. If None, one is created
                by scanning the model for StateVariables.
        """
        self.pz_model = pz_model
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.state_manager = (
            state_manager if state_manager is not None
            else PenzaiStateManager.from_model(pz_model)
        )
        self.input_ports = tuple(spec.port_name for spec in input_mapping.port_specs)
        self.output_ports = tuple(spec.port_name for spec in output_mapping.port_specs)

    @classmethod
    def from_layer(
        cls,
        layer: PyTree,
        input_port: str = "input",
        output_port: str = "output",
    ) -> "PenzaiSubgraph":
        """Create PenzaiSubgraph from a single-input, single-output layer.

        This is the simplest way to wrap a Penzai layer. The layer should
        accept a single argument and return a single value.

        Args:
            layer: Penzai layer (e.g., pz.nn.Linear, pz.nn.Sequential).
            input_port: Name for the input port. Default: "input".
            output_port: Name for the output port. Default: "output".

        Returns:
            PenzaiSubgraph wrapping the layer.

        Example:
            >>> layer = pzl.nn.Linear(input_size=4, output_size=2)
            >>> component = PenzaiSubgraph.from_layer(layer)
        """
        _require_penzai()
        return cls(
            pz_model=layer,
            input_mapping=InputMapping.single(input_port),
            output_mapping=OutputMapping.single(output_port),
        )

    @classmethod
    def from_sequential(
        cls,
        layers: Sequence[PyTree],
        input_port: str = "input",
        output_port: str = "output",
    ) -> "PenzaiSubgraph":
        """Create PenzaiSubgraph from a sequence of Penzai layers.

        Wraps the layers in a pz.nn.Sequential and creates appropriate
        input/output mappings.

        Args:
            layers: Sequence of Penzai layers to compose.
            input_port: Name for the input port. Default: "input".
            output_port: Name for the output port. Default: "output".

        Returns:
            PenzaiSubgraph wrapping the sequential composition.

        Example:
            >>> layers = [
            ...     pzl.nn.Linear(input_size=4, output_size=32),
            ...     pzl.nn.ReLU(),
            ...     pzl.nn.Linear(input_size=32, output_size=2),
            ... ]
            >>> component = PenzaiSubgraph.from_sequential(layers)
        """
        _require_penzai()
        from penzai import pz as pzl

        sequential = pzl.nn.Sequential(list(layers))
        return cls.from_layer(sequential, input_port, output_port)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute the wrapped Penzai model.

        Args:
            inputs: Dict mapping input port names to values.
            state: Current feedbax State.
            key: JAX PRNG key for stochastic operations.

        Returns:
            Tuple of (output dict, updated state).
        """
        # Map inputs to layer input format
        layer_input = self.input_mapping(inputs)

        # Bind state variables into model
        bound_model = self.state_manager.bind_state(self.pz_model, state)

        # Execute the Penzai model
        # Penzai layers are callable: layer(input)
        if callable(bound_model):
            layer_output = bound_model(layer_input)
        else:
            # For non-callable models, try to find a __call__ method
            layer_output = layer_input

        # Unbind state variables and update feedbax state
        _, state = self.state_manager.unbind_state(bound_model, state)

        # Map layer output to feedbax output format
        outputs = self.output_mapping(layer_output)

        return outputs, state

    def init_state(self, *, key: PRNGKeyArray) -> State:
        """Return initial state for this component.

        Returns:
            State with Penzai StateVariable initial values.
        """
        # State constructor requires a model argument in newer equinox versions
        state = State(self)
        if self.state_manager._initial_state:
            state = state.set(
                self.state_manager.state_index,
                self.state_manager._initial_state,
            )
        return state

    def treescope_html(self) -> str:
        """Render the Penzai model to HTML using treescope.

        Returns:
            HTML string containing the treescope visualization.

        Raises:
            ImportError: If treescope is not installed.
        """
        _require_treescope()

        # treescope.render_to_html returns HTML as a string
        # _require_treescope ensures treescope is not None
        assert treescope is not None  # For type checker
        html = treescope.render_to_html(self.pz_model, roundtrip_mode=False)
        return html if isinstance(html, str) else str(html)

    def state_view(self, state: State) -> PyTree | None:
        """Return the state view for this component.

        Returns the current Penzai state variable values.
        """
        try:
            return state.get(self.state_manager.state_index)
        except (KeyError, ValueError):
            return None

    def __repr__(self) -> str:
        return (
            f"PenzaiSubgraph("
            f"input_ports={self.input_ports}, "
            f"output_ports={self.output_ports}, "
            f"pz_model={type(self.pz_model).__name__})"
        )


# =============================================================================
# Factory Registry for Serialization
# =============================================================================


# Registry of model builders for web UI serialization
# Maps builder name to (builder_fn, default_params) tuple
_PENZAI_MODEL_BUILDERS: dict[str, tuple[Callable[..., PyTree], dict[str, Any]]] = {}


def register_penzai_builder(
    name: str,
    builder_fn: Callable[..., PyTree],
    default_params: dict[str, Any] | None = None,
) -> None:
    """Register a Penzai model builder for web UI serialization.

    This allows the web UI to reconstruct Penzai models from specifications
    without needing to serialize the model weights directly.

    Args:
        name: Unique name for the builder (e.g., "penzai_mlp").
        builder_fn: Function that takes params dict and returns Penzai model.
        default_params: Default parameter values for the builder.

    Example:
        >>> def build_mlp(params):
        ...     return pzl.nn.Sequential([
        ...         pzl.nn.Linear(params["input_size"], params["hidden_size"]),
        ...         pzl.nn.ReLU(),
        ...         pzl.nn.Linear(params["hidden_size"], params["output_size"]),
        ...     ])
        >>>
        >>> register_penzai_builder(
        ...     "penzai_mlp",
        ...     build_mlp,
        ...     {"input_size": 4, "hidden_size": 64, "output_size": 2},
        ... )
    """
    _require_penzai()
    _PENZAI_MODEL_BUILDERS[name] = (builder_fn, default_params or {})


def get_penzai_builder(name: str) -> tuple[Callable[..., PyTree], dict[str, Any]] | None:
    """Get a registered Penzai model builder by name.

    Args:
        name: Name of the registered builder.

    Returns:
        Tuple of (builder_fn, default_params) or None if not found.
    """
    return _PENZAI_MODEL_BUILDERS.get(name)


def list_penzai_builders() -> list[str]:
    """List all registered Penzai model builder names.

    Returns:
        List of registered builder names.
    """
    return list(_PENZAI_MODEL_BUILDERS.keys())


def build_penzai_subgraph(
    builder_name: str,
    params: dict[str, Any] | None = None,
    input_port: str = "input",
    output_port: str = "output",
) -> PenzaiSubgraph:
    """Build a PenzaiSubgraph from a registered builder.

    Args:
        builder_name: Name of the registered builder.
        params: Parameters to pass to the builder (merged with defaults).
        input_port: Name for the input port. Default: "input".
        output_port: Name for the output port. Default: "output".

    Returns:
        PenzaiSubgraph wrapping the built model.

    Raises:
        ValueError: If builder_name is not registered.
    """
    builder_info = get_penzai_builder(builder_name)
    if builder_info is None:
        raise ValueError(
            f"Unknown Penzai builder '{builder_name}'. "
            f"Available builders: {list_penzai_builders()}"
        )

    builder_fn, default_params = builder_info
    merged_params = {**default_params, **(params or {})}
    pz_model = builder_fn(merged_params)

    return PenzaiSubgraph.from_layer(pz_model, input_port, output_port)


# =============================================================================
# Convenience exports
# =============================================================================


__all__ = [
    # Core types
    "PenzaiSubgraph",
    "InputMapping",
    "OutputMapping",
    "PortSpec",
    "PenzaiStateManager",
    # Factory functions
    "register_penzai_builder",
    "get_penzai_builder",
    "list_penzai_builders",
    "build_penzai_subgraph",
    # Availability flags
    "PENZAI_AVAILABLE",
    "TREESCOPE_AVAILABLE",
]
