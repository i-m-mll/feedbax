"""Neural network architectures.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from typing import ClassVar, Literal, Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import Module, field
from equinox.nn import State, StateIndex
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from pydantic import BaseModel, ConfigDict
from pydantic import Field as ConfigField

from feedbax.contracts.component import PortType
from feedbax.contracts.graph import ParameterConstraintSpec
from feedbax.contracts.portable import ComponentOptions, ConfigBinding, Ports
from feedbax.models.support import (
    interleave_unequal,
)
from feedbax.runtime.components import (
    recurrent_output_prototype,
    resolve_activation,
    tanh,
)
from feedbax.runtime.graph import Component
from feedbax.runtime.parameter_constraints import (
    recurrent_parameter_role,
    register_parameter_role_resolver,
)

logger = logging.getLogger(__name__)

POPULATION_STRUCTURE_SCHEMA_ID = "feedbax.spec.population_structure"
POPULATION_STRUCTURE_SCHEMA_VERSION = "feedbax.spec.population_structure.v1"

RECURRENT_CELL_TYPE_ALIASES = {
    "GRU": "GRU",
    "GRUCell": "GRU",
    "LSTM": "LSTM",
    "LSTMCell": "LSTM",
    "VanillaRNN": "VanillaRNN",
    "VanillaRNNCell": "VanillaRNN",
    "RNN": "VanillaRNN",
    "RNNCell": "VanillaRNN",
    "LeakyRNNCell": "VanillaRNN",
}


def recurrent_cell_vocabulary() -> tuple[str, ...]:
    """Return accepted recurrent-cell vocabulary names."""

    return tuple(RECURRENT_CELL_TYPE_ALIASES)


def normalize_recurrent_cell_type(cell_type: object, *, path: str = "cell_type") -> str:
    """Normalize a recurrent-cell vocabulary name to its executable component type."""

    name = str(cell_type)
    try:
        return RECURRENT_CELL_TYPE_ALIASES[name]
    except KeyError as exc:
        vocabulary = ", ".join(recurrent_cell_vocabulary())
        raise ValueError(
            f"Unknown recurrent cell type at {path}: {name!r}. Supported values: {vocabulary}"
        ) from exc


def recurrent_gate_count(cell_type: object, *, path: str = "cell_type") -> int:
    """Return the input-kernel gate count for a recurrent-cell vocabulary name."""

    normalized_cell_type = normalize_recurrent_cell_type(cell_type, path=path)
    if normalized_cell_type == "LSTM":
        return 4
    if normalized_cell_type == "GRU":
        return 3
    return 1


def orthogonal_gru_cell(
    input_size: int,
    hidden_size: int,
    use_bias: bool = True,
    scale: float = 1.0,
    *,
    key: PRNGKeyArray,
):
    """Returns an `eqx.nn.GRUCell` with orthogonal weight matrix initialization."""
    net = eqx.nn.GRUCell(input_size, hidden_size, use_bias=use_bias, key=key)
    initializer = jax.nn.initializers.orthogonal(scale=scale, column_axis=-1)
    ortho_weight_hh = jnp.concatenate(
        [initializer(k, (hidden_size, hidden_size)) for k in jr.split(key, 3)],
        axis=0,
    )
    net = eqx.tree_at(
        lambda net: net.weight_hh,
        net,
        ortho_weight_hh,
    )
    return net


def contiguous_assignment(
    hidden_size: int,
    n_input_only: int,
    n_readout_only: int,
    n_recurrent_only: int,
    n_input_readout: int,
    key: PRNGKeyArray,
):
    """Contiguous assignment function for population structure.

    Assigns populations in contiguous blocks:
    - Units 0 to n_input_only-1: input-only
    - Units n_input_only to n_input_only+n_readout_only-1: readout-only
    - Units ... to ...: recurrent-only
    - Remaining units: input-readout

    Arguments:
        hidden_size: Total number of hidden units.
        n_input_only: Number of input-only units.
        n_readout_only: Number of readout-only units.
        n_recurrent_only: Number of recurrent-only units.
        n_input_readout: Number of input-readout units.
        key: Random key (unused, but required for interface compatibility).

    Returns:
        Tuple of (input_only_indices, readout_only_indices, recurrent_only_indices, input_readout_indices).
    """
    input_only_indices = jnp.arange(0, n_input_only)
    readout_only_indices = jnp.arange(n_input_only, n_input_only + n_readout_only)
    recurrent_only_indices = jnp.arange(
        n_input_only + n_readout_only,
        n_input_only + n_readout_only + n_recurrent_only,
    )
    input_readout_indices = jnp.arange(
        n_input_only + n_readout_only + n_recurrent_only,
        hidden_size,
    )
    return input_only_indices, readout_only_indices, recurrent_only_indices, input_readout_indices


class PopulationStructure(Module):
    """Defines partitioning of hidden units into subpopulations with different connectivity.

    Hidden units can be partitioned into four types:
    - Input-only: receive inputs but don't contribute to readout
    - Readout-only: don't receive inputs but contribute to readout
    - Recurrent-only: neither receive inputs nor contribute to readout (internal dynamics only)
    - Input-readout: both receive inputs and contribute to readout

    Attributes:
        n_input_only: Number of units that only receive inputs.
        n_readout_only: Number of units that only contribute to readout.
        n_recurrent_only: Number of units that are recurrent-only (no input/output).
        n_input_readout: Number of units that both receive inputs and contribute to readout.
        input_indices: Indices of all units receiving inputs (input-only + input-readout).
        readout_indices: Indices of all units contributing to readout (readout-only + input-readout).
        input_only_indices: Indices of input-only units.
        readout_only_indices: Indices of readout-only units.
        recurrent_only_indices: Indices of recurrent-only units.
        input_readout_indices: Indices of input-readout units.
    """

    n_input_only: int
    n_readout_only: int
    n_recurrent_only: int
    n_input_readout: int

    # Indices for each population
    input_indices: Array  # shape (n_input_only + n_input_readout,)
    readout_indices: Array  # shape (n_readout_only + n_input_readout,)
    input_only_indices: Array  # shape (n_input_only,)
    readout_only_indices: Array  # shape (n_readout_only,)
    recurrent_only_indices: Array  # shape (n_recurrent_only,)
    input_readout_indices: Array  # shape (n_input_readout,)

    @classmethod
    def from_indices(
        cls,
        *,
        input_only_indices: Sequence[int],
        readout_only_indices: Sequence[int],
        recurrent_only_indices: Sequence[int],
        input_readout_indices: Sequence[int],
    ) -> "PopulationStructure":
        """Create a population structure from explicit unit assignments."""

        input_only = jnp.asarray(input_only_indices, dtype=int)
        readout_only = jnp.asarray(readout_only_indices, dtype=int)
        recurrent_only = jnp.asarray(recurrent_only_indices, dtype=int)
        input_readout = jnp.asarray(input_readout_indices, dtype=int)
        input_indices = jnp.concatenate([input_only, input_readout])
        readout_indices = jnp.concatenate([readout_only, input_readout])
        return cls(
            n_input_only=int(input_only.shape[0]),
            n_readout_only=int(readout_only.shape[0]),
            n_recurrent_only=int(recurrent_only.shape[0]),
            n_input_readout=int(input_readout.shape[0]),
            input_indices=input_indices,
            readout_indices=readout_indices,
            input_only_indices=input_only,
            readout_only_indices=readout_only,
            recurrent_only_indices=recurrent_only,
            input_readout_indices=input_readout,
        )

    @classmethod
    def create(
        cls,
        hidden_size: int,
        n_input_only: int = 0,
        n_readout_only: int = 0,
        n_recurrent_only: int = 0,
        n_input_readout: int = 0,
        assignment_fn: Optional[Callable] = None,
        *,
        key: PRNGKeyArray,
    ) -> "PopulationStructure":
        """Create a population structure with the specified population sizes.

        If the subpopulation sizes (e.g. `n_input_only`) sum up to less than
        the `hidden_size`, then `n_input_readout` will be increased so the
        overall sizes match. That means that calling this method with only
        a `hidden_size` will create a network with no input-output segregation.
        (Alternatively, in the future we could ask only for the subpopulation
        structure and not for the total `hidden_size`, and just infer it.)

        Arguments:
            hidden_size: Total number of hidden units.
            n_input_only: Number of input-only units.
            n_readout_only: Number of readout-only units.
            n_recurrent_only: Number of recurrent-only units.
            n_input_readout: Number of input-readout units.
            assignment_fn: Optional callable that takes (hidden_size, n_input_only,
                n_readout_only, n_recurrent_only, n_input_readout, key) and returns
                a tuple of 4 arrays (input_only_indices, readout_only_indices,
                recurrent_only_indices, input_readout_indices). If None, uses random
                assignment.
            key: Random key for assignment.
        """
        total = n_input_only + n_readout_only + n_recurrent_only + n_input_readout
        if total > hidden_size:
            raise ValueError("Sum of subpopulation sizes is larger than network size")
        elif total < hidden_size:
            n_input_readout += hidden_size - total

        if assignment_fn is None:
            # Default: random assignment
            all_indices = jr.permutation(key, hidden_size)
            input_only_indices = all_indices[:n_input_only]
            readout_only_indices = all_indices[n_input_only : n_input_only + n_readout_only]
            recurrent_only_indices = all_indices[
                n_input_only + n_readout_only : n_input_only + n_readout_only + n_recurrent_only
            ]
            input_readout_indices = all_indices[n_input_only + n_readout_only + n_recurrent_only :]
        else:
            # Custom assignment
            (
                input_only_indices,
                readout_only_indices,
                recurrent_only_indices,
                input_readout_indices,
            ) = assignment_fn(
                hidden_size, n_input_only, n_readout_only, n_recurrent_only, n_input_readout, key
            )

        # Compute combined indices for input and readout
        input_indices = jnp.concatenate([input_only_indices, input_readout_indices])
        readout_indices = jnp.concatenate([readout_only_indices, input_readout_indices])

        return cls(
            n_input_only=n_input_only,
            n_readout_only=n_readout_only,
            n_recurrent_only=n_recurrent_only,
            n_input_readout=n_input_readout,
            input_indices=input_indices,
            readout_indices=readout_indices,
            input_only_indices=input_only_indices,
            readout_only_indices=readout_only_indices,
            recurrent_only_indices=recurrent_only_indices,
            input_readout_indices=input_readout_indices,
        )

    def to_spec(self) -> dict[str, object]:
        """Return a durable, explicit population assignment spec."""

        def indices(values: Array) -> list[int]:
            return [int(value) for value in jnp.asarray(values).tolist()]

        return {
            "schema_id": POPULATION_STRUCTURE_SCHEMA_ID,
            "schema_version": POPULATION_STRUCTURE_SCHEMA_VERSION,
            "assignment": "explicit",
            "n_input_only": int(self.n_input_only),
            "n_readout_only": int(self.n_readout_only),
            "n_recurrent_only": int(self.n_recurrent_only),
            "n_input_readout": int(self.n_input_readout),
            "input_only_indices": indices(self.input_only_indices),
            "readout_only_indices": indices(self.readout_only_indices),
            "recurrent_only_indices": indices(self.recurrent_only_indices),
            "input_readout_indices": indices(self.input_readout_indices),
        }


def validate_population_structure_spec(spec: Mapping[str, object]) -> dict[str, object]:
    """Validate a serialized population-structure spec against the schema registry."""
    from feedbax.contracts.migrations import (
        UnsupportedSpecVersion,
        default_spec_registry,
    )

    schema_id = spec.get("schema_id")
    if schema_id is not None and schema_id != POPULATION_STRUCTURE_SCHEMA_ID:
        raise ValueError(
            "Unsupported PopulationStructureSpec schema_id: "
            f"schema_id={schema_id!r}, expected={POPULATION_STRUCTURE_SCHEMA_ID!r}"
        )
    schema_version = spec.get("schema_version")
    if (
        isinstance(schema_version, str)
        and schema_version.startswith(f"{POPULATION_STRUCTURE_SCHEMA_ID}.v")
        and schema_version != POPULATION_STRUCTURE_SCHEMA_VERSION
    ):
        raise UnsupportedSpecVersion(
            "Unsupported PopulationStructureSpec schema_version: "
            f"schema_id={POPULATION_STRUCTURE_SCHEMA_ID!r}, "
            f"source_version={schema_version!r}, "
            f"current_version={POPULATION_STRUCTURE_SCHEMA_VERSION!r}; "
            "future population-structure versions must register an explicit migration"
        )
    result = default_spec_registry.migrate(
        "PopulationStructureSpec",
        spec,
        assume_current=True,
    )
    return result.payload


def population_structure_from_spec(
    hidden_size: int,
    spec: Mapping[str, object] | PopulationStructure,
    *,
    key: PRNGKeyArray | None = None,
) -> PopulationStructure:
    """Build a ``PopulationStructure`` from a serialized high-level spec."""

    if isinstance(spec, PopulationStructure):
        return spec
    spec = validate_population_structure_spec(spec)

    def int_param(name: str, default: int = 0) -> int:
        value = spec.get(name, default)
        if isinstance(value, (int, float, str)):
            return int(value)
        raise ValueError(f"population_structure {name!r} must be numeric")

    def indices_param(name: str) -> Sequence[int]:
        value = spec.get(name, ())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return cast(Sequence[int], value)
        raise ValueError(f"population_structure {name!r} must be a sequence of integers")

    assignment = str(spec.get("assignment", "random"))
    if assignment == "explicit":
        return PopulationStructure.from_indices(
            input_only_indices=indices_param("input_only_indices"),
            readout_only_indices=indices_param("readout_only_indices"),
            recurrent_only_indices=indices_param("recurrent_only_indices"),
            input_readout_indices=indices_param("input_readout_indices"),
        )
    n_input_only = int_param("n_input_only")
    n_readout_only = int_param("n_readout_only")
    n_recurrent_only = int_param("n_recurrent_only")
    n_input_readout = int_param("n_input_readout")
    if assignment == "contiguous":
        assignment_fn = contiguous_assignment
    elif assignment == "random":
        assignment_fn = None
    else:
        raise ValueError(f"Unsupported population assignment {assignment!r}")
    if key is None:
        key = jr.PRNGKey(int_param("seed"))
    return PopulationStructure.create(
        hidden_size,
        n_input_only=n_input_only,
        n_readout_only=n_readout_only,
        n_recurrent_only=n_recurrent_only,
        n_input_readout=n_input_readout,
        assignment_fn=assignment_fn,
        key=key,
    )


def _population_base_input_mask(
    population_structure: PopulationStructure,
    input_size: int,
) -> Array:
    hidden_size = (
        population_structure.n_input_only
        + population_structure.n_readout_only
        + population_structure.n_recurrent_only
        + population_structure.n_input_readout
    )
    mask = jnp.zeros((hidden_size, input_size))
    return mask.at[population_structure.input_indices, :].set(1.0)


def population_input_kernel_mask(
    population_structure: PopulationStructure,
    input_size: int,
    *,
    gate_count: int,
) -> Array:
    """Return an input-kernel mask with hidden rows repeated per recurrent gate."""

    if gate_count < 1:
        raise ValueError("gate_count must be positive")
    return jnp.tile(_population_base_input_mask(population_structure, input_size), (gate_count, 1))


def population_readout_kernel_mask(
    population_structure: PopulationStructure,
    out_size: int,
) -> Array:
    """Return a readout weight mask selecting population readout columns."""

    hidden_size = (
        population_structure.n_input_only
        + population_structure.n_readout_only
        + population_structure.n_recurrent_only
        + population_structure.n_input_readout
    )
    mask = jnp.zeros((out_size, hidden_size))
    return mask.at[:, population_structure.readout_indices].set(1.0)


def lower_population_constraints(
    population_structure: PopulationStructure | Mapping[str, object],
    *,
    hidden_size: int,
    input_size: int,
    out_size: int | None,
    cell_type: str,
    cell_node: str = "cell",
    readout_node: str = "readout",
) -> tuple[ParameterConstraintSpec, ...]:
    """Lower high-level population connectivity to graph parameter constraints."""

    structure = population_structure_from_spec(hidden_size, population_structure)
    gate_count = recurrent_gate_count(cell_type, path="population_constraints.cell_type")
    constraints = [
        ParameterConstraintSpec(
            node=cell_node,
            role="input_kernel",
            mask=population_input_kernel_mask(
                structure,
                input_size,
                gate_count=gate_count,
            ).tolist(),
            value=0.0,
        ),
    ]
    if out_size is not None:
        constraints.append(
            ParameterConstraintSpec(
                node=readout_node,
                role="weight",
                mask=population_readout_kernel_mask(structure, out_size).tolist(),
                value=0.0,
            )
        )
    return tuple(constraints)


class LeakyRNNCell(Module):
    """Custom `RNNCell` with persistent, leaky state.

    Timing values must be finite. The time constant `tau` must be positive.
    The timestep `dt` may be zero for deterministic updates and must be
    positive when noise is enabled.

    Based on `eqx.nn.GRUCell` and the leaky RNN from

        [1] G. R. Yang, M. R. Joglekar, H. F. Song, W. T. Newsome,
            and X.-J. Wang, “Task representations in neural networks trained
            to perform many cognitive tasks,” Nat Neurosci, vol. 22, no. 2,
            pp. 297–306, Feb. 2019, doi: 10.1038/s41593-018-0310-2.
    """

    weight_hh: Array
    weight_ih: Array
    bias: Optional[Array]
    input_size: int
    hidden_size: int
    use_bias: bool
    use_noise: bool
    noise_strength: float
    dt: float
    tau: float
    nonlinearity: Callable = field(static=True)

    @jax.named_scope("fbx.RNNCell")
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        use_noise: bool = False,
        noise_strength: float = 0.01,
        dt: float = 1.0,
        tau: float = 1.0,
        nonlinearity: Callable = tanh,
        dtype: object = jnp.float32,
        *,  # this forces the user to pass the following as keyword arguments
        key: PRNGKeyArray,
        **kwargs,
    ):
        dt, tau = float(dt), float(tau)
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("LeakyRNNCell requires a finite tau > 0")
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("LeakyRNNCell requires a finite dt >= 0")
        if use_noise and dt == 0:
            raise ValueError("LeakyRNNCell requires dt > 0 when use_noise=True")

        dtype = dtype or jnp.float32
        ihkey, hhkey, bkey = jr.split(key, 3)
        lim = math.sqrt(1 / hidden_size)

        if input_size > 0:
            self.weight_ih = jr.uniform(
                ihkey,
                (hidden_size, input_size),
                minval=-lim,
                maxval=lim,
                dtype=dtype,
            )
        else:
            self.weight_ih = jnp.array(0, dtype=dtype)

        self.weight_hh = jr.uniform(
            hhkey,
            (hidden_size, hidden_size),
            minval=-lim,
            maxval=lim,
            dtype=dtype,
        )

        if use_bias:
            self.bias = jr.uniform(
                bkey,
                (hidden_size,),
                minval=-lim,
                maxval=lim,
                dtype=dtype,
            )
        else:
            self.bias = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.use_noise = use_noise
        self.noise_strength = noise_strength
        self.dt = dt
        self.tau = tau
        self.nonlinearity = nonlinearity

    def __call__(self, input: Array, state: Array, key: PRNGKeyArray | None = None):
        """Vanilla RNN cell."""
        if self.use_bias:
            bias = self.bias
        else:
            bias = 0

        if self.use_noise:
            if key is None:
                raise ValueError("LeakyRNNCell requires an RNG key when use_noise=True")
            noise_std = jnp.asarray(self.noise_std, dtype=state.dtype)
            noise = noise_std * jr.normal(key, state.shape, dtype=state.dtype)
        else:
            noise = 0

        state = (1 - self.alpha) * state + self.alpha * self.nonlinearity(
            jnp.dot(self.weight_ih, input) + jnp.dot(self.weight_hh, state) + bias + noise
        )

        return state  #! 0D PyTree

    @cached_property
    def alpha(self):
        return self.dt / self.tau

    @cached_property
    def noise_std(self):
        if self.use_noise:
            return math.sqrt(2 / self.alpha) * self.noise_strength
        else:
            return None


class VanillaRNNState(Module):
    """State view for a leaf vanilla RNN cell."""

    hidden: Array
    output: Array


class _VanillaRNNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dtype: str = ConfigField(default="float32")
    input_size: int = ConfigField(default=4, ge=1)
    hidden_size: int = ConfigField(default=4, ge=1)
    use_bias: bool = ConfigField(default=True)
    use_noise: bool = ConfigField(default=False)
    noise_strength: float = ConfigField(default=0.01, ge=0)
    dt: float = ConfigField(default=1.0, ge=0)
    tau: float = ConfigField(default=1.0, ge=0)
    activation: Literal["tanh", "relu", "identity", "sigmoid", "softmax"] = ConfigField(
        default="tanh"
    )


class VanillaRNN(Component):
    """Standalone vanilla RNN cell."""

    config: ClassVar[type[BaseModel]] = _VanillaRNNConfig

    ports: ClassVar[Ports] = Ports(
        inputs={"input": PortType(dtype="vector"), "hidden": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector"), "hidden": PortType(dtype="vector")},
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=recurrent_output_prototype,
        build_context_fields=("key",),
        trainable_by_default=True,
    )
    config_bindings: ClassVar[tuple[ConfigBinding, ...]] = (
        ConfigBinding("activation_name", name="activation"),
        ConfigBinding("cell.use_bias", name="use_bias"),
        ConfigBinding("cell.use_noise", name="use_noise"),
        ConfigBinding("cell.noise_strength", name="noise_strength"),
        ConfigBinding("cell.dt", name="dt"),
        ConfigBinding("cell.tau", name="tau"),
    )

    cell: LeakyRNNCell
    input_size: int = field(static=True)
    hidden_size: int = field(static=True)
    activation_name: str = field(static=True)
    dtype: object = field(static=True)
    state_index: StateIndex

    def __init__(
        self,
        input_size: int = _VanillaRNNConfig.model_fields["input_size"].default,
        hidden_size: int = _VanillaRNNConfig.model_fields["hidden_size"].default,
        *,
        activation_name: str = "tanh",
        nonlinearity: Callable | None = None,
        use_bias: bool = _VanillaRNNConfig.model_fields["use_bias"].default,
        use_noise: bool = _VanillaRNNConfig.model_fields["use_noise"].default,
        noise_strength: float = _VanillaRNNConfig.model_fields["noise_strength"].default,
        dt: float = _VanillaRNNConfig.model_fields["dt"].default,
        tau: float = _VanillaRNNConfig.model_fields["tau"].default,
        dtype: object = _VanillaRNNConfig.model_fields["dtype"].default,
        key: PRNGKeyArray | None = None,
    ):
        dtype = dtype or jnp.float32
        if nonlinearity is None:
            _, nonlinearity = resolve_activation(activation_name)
        if key is None:
            key = jr.PRNGKey(0)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.activation_name = str(activation_name)
        self.dtype = jnp.dtype(dtype).name
        self.cell = LeakyRNNCell(
            self.input_size,
            self.hidden_size,
            use_bias=use_bias,
            use_noise=use_noise,
            noise_strength=noise_strength,
            dt=dt,
            tau=tau,
            nonlinearity=nonlinearity,
            dtype=dtype,
            key=key,
        )
        hidden = jnp.zeros(self.hidden_size, dtype=dtype)
        initial_state = VanillaRNNState(hidden=hidden, output=hidden)
        self.state_index = StateIndex(initial_state)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        current_state: VanillaRNNState = state.get(self.state_index)
        hidden = inputs.get("hidden", current_state.hidden)
        new_hidden = self.cell(inputs["input"], hidden, key=key)
        state = state.set(
            self.state_index,
            VanillaRNNState(hidden=new_hidden, output=new_hidden),
        )
        return {"output": new_hidden, "hidden": new_hidden}, state

    @classmethod
    def from_config(cls, config, **context):
        values = config.model_dump(mode="python")
        values["activation_name"] = values.pop("activation")
        return cls(**values, **context)


register_parameter_role_resolver(VanillaRNN, recurrent_parameter_role)


def n_layer_linear(
    hidden_sizes: Sequence[int],
    input_size: int,
    out_size: int,
    use_bias: bool = True,
    nonlinearity: Callable[[Float], Float] = jnp.tanh,
    *,
    key,
):
    """A simple n-layer linear network with nonlinearity."""
    keys = jr.split(key, len(hidden_sizes) + 1)
    sizes = (input_size,) + tuple(hidden_sizes) + (out_size,)
    layers = [
        eqx.nn.Linear(size0, size1, use_bias=use_bias, key=keys[i])
        for i, (size0, size1) in enumerate(zip(sizes[:-1], sizes[1:]))
    ]
    return eqx.nn.Sequential(list(interleave_unequal(layers, [nonlinearity] * len(hidden_sizes))))


def two_layer_linear(
    hidden_size, input_size, out_size, use_bias=True, nonlinearity=jnp.tanh, *, key
):
    """A two-layer linear network with nonlinearity.

    Just a convenience over `n_layer` since two-layer readouts may be more
    common than other n-layer readouts for RNNs.
    """
    return n_layer_linear(
        (hidden_size,),
        input_size,
        out_size,
        use_bias=use_bias,
        nonlinearity=nonlinearity,
        key=key,
    )


def gru_weight_idxs_func(
    label: Literal["candidate", "update", "reset"],
) -> Callable[[Array], slice]:
    """Returns a function that returns a slice of a subset of the GRU weights.

    TODO: Should probably just return a function that returns the subset of weights directly, rather than their indices.
    """

    def gru_weight_idxs(weights):
        len_by_3 = weights.shape[-2] // 3
        return {
            "reset": slice(0, len_by_3),
            "update": slice(len_by_3, 2 * len_by_3),
            "candidate": slice(2 * len_by_3, None),
        }[label]

    return gru_weight_idxs


from feedbax.models.controllers import recurrent_controller as recurrent_controller
