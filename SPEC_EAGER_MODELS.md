# Feedbax Eager Models Architecture Specification

## Overview

This document specifies the migration from feedbax's current "staged model" architecture to an "eager graph" architecture. The new system provides:

1. **Explicit graph structure** - Models are graphs of components with typed ports and wires
2. **Unified state management** - All state (including intervention params) uses Equinox's StateIndex pattern
3. **Single-step models** - Models compute one timestep; iteration is external
4. **Components, not intervenors** - Interventions are regular components inserted via graph surgery
5. **Eager execution** - Explicit, debuggable execution order (no lazy evaluation)

## Design Principles

1. **Everything is a Component** - Neural networks, mechanics, channels, interventions - all share the same interface
2. **Graphs are Components** - Hierarchical composition; a graph can be a node in another graph
3. **State is unified** - Model state, solver state, intervention params all live in one State object
4. **Iteration is external** - Models don't know about time steps; tasks/trainers handle iteration
5. **Graph surgery for flexibility** - Insert/remove components by modifying graph structure

---

## Cycles and Iteration

A key insight: **graphs can have cycles, and cycles imply iteration**.

When a Graph has a wire that forms a cycle (e.g., `mechanics.effector` → `feedback.input`), this means:
- The output from step N becomes the input to step N+1
- The Graph must iterate internally using `lax.scan`
- This is detected at construction time

This unifies:
- **Internal feedback loops** (e.g., effector → feedback channel within SimpleFeedback)
- **Agent-environment loops** (model ↔ task)
- **Nested iteration** (a Graph with cycles inside another Graph with cycles)

### Task as Component

The Task can be viewed as a Component in the overall system:
- **Outputs**: targets, intervention params, timing signals
- **Inputs**: model outputs (for online feedback or next-trial adaptation)

This means the agent-environment loop is just a cycle in a larger graph:

```
┌─────────────────────────────────────────────┐
│                                             │
│   ┌──────┐          ┌───────┐              │
│   │ Task │─────────▶│ Model │              │
│   └──────┘          └───────┘              │
│       ▲                  │                  │
│       └──────────────────┘                  │
│                                             │
└─────────────────────────────────────────────┘
              (cycle = iteration)
```

For current feedbax use cases (fixed-length trials), the Task provides static inputs.
For future RL/online scenarios, the Task responds dynamically to model outputs.

A `TaskComponent` adapter wraps `AbstractTask` for graph composition when needed.

---

## Core Types

### State Management

Uses Equinox's StateIndex pattern. Each component that needs persistent state creates indices in `__init__`.

```python
from equinox.nn import State, StateIndex

class Channel(Component):
    """A delay line that stores a queue of values."""

    queue_index: StateIndex  # Created in __init__

    def __init__(self, delay: int, init_value: PyTree):
        # StateIndex bundles the key with initial value
        initial_queue = (init_value,) * delay
        self.queue_index = StateIndex(initial_queue)
        # ... other init

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        queue = state.get(self.queue_index)

        # Pop oldest, push newest
        output = queue[0]
        new_queue = queue[1:] + (inputs["input"],)

        state = state.set(self.queue_index, new_queue)
        return {"output": output}, state
```

**Key properties**:
- Each component owns its state via its StateIndex instances
- State is passed through calls, returned updated (functional style)
- `eqx.nn.make_with_state(Model)(...)` collects all indices into initial State
- State can hold anything: arrays, solver state, intervention params, queues

### Component Base Class

```python
from abc import abstractmethod
from typing import ClassVar
import equinox as eqx
from equinox import Module
from equinox.nn import State
from jaxtyping import PRNGKeyArray, PyTree

class Component(Module):
    """Base class for all graph nodes.

    Components have named input ports and output ports.
    They receive inputs as a dict, return outputs as a dict.
    They can read/write persistent state via StateIndex.
    """

    # Subclasses define these as class variables or properties
    input_ports: ClassVar[tuple[str, ...]]
    output_ports: ClassVar[tuple[str, ...]]

    @abstractmethod
    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute the component.

        Args:
            inputs: Dict mapping input port names to values
            state: Current state (read via state.get(index))
            key: PRNG key for stochastic operations

        Returns:
            outputs: Dict mapping output port names to values
            state: Updated state (written via state.set(index, value))
        """
        ...

    def init_state(self, *, key: PRNGKeyArray) -> State:
        """Return initial state for this component.

        Default implementation collects all StateIndex instances.
        Override for custom initialization logic.
        """
        # eqx.nn.make_with_state handles this automatically
        # This method is for explicit control if needed
        return State()
```

### Ports and Wires

```python
class Wire(Module):
    """A connection between an output port and an input port."""

    source_node: str      # Name of source node
    source_port: str      # Name of output port on source
    target_node: str      # Name of target node
    target_port: str      # Name of input port on target

    def __repr__(self):
        return f"Wire({self.source_node}.{self.source_port} -> {self.target_node}.{self.target_port})"
```

### Graph

```python
from functools import cached_property
from collections import OrderedDict

class Graph(Component):
    """A computational graph of components.

    Graphs are themselves Components, enabling hierarchical composition.
    A Graph can be used as a node inside another Graph.
    """

    nodes: dict[str, Component]
    wires: tuple[Wire, ...]

    # Graph-level ports (how this graph connects to the outside)
    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]

    # Wiring from graph inputs to internal nodes
    # Maps: graph_input_port -> (node_name, node_input_port)
    input_bindings: dict[str, tuple[str, str]]

    # Wiring from internal nodes to graph outputs
    # Maps: graph_output_port -> (node_name, node_output_port)
    output_bindings: dict[str, tuple[str, str]]

    @cached_property
    def _cycle_analysis(self) -> tuple[tuple[str, ...], tuple[Wire, ...]]:
        """Analyze graph for cycles and determine execution order.

        Returns:
            execution_order: Nodes in topological order (excluding cycle-back edges)
            cycle_wires: Wires that form cycles (cross timestep boundaries)
        """
        return self._analyze_cycles()

    @property
    def _execution_order(self) -> tuple[str, ...]:
        """Topologically sorted node names (for acyclic portion)."""
        return self._cycle_analysis[0]

    @property
    def _cycle_wires(self) -> tuple[Wire, ...]:
        """Wires that form cycles (values flow from step N to step N+1)."""
        return self._cycle_analysis[1]

    @property
    def _needs_iteration(self) -> bool:
        """Whether this graph has cycles and needs to iterate."""
        return len(self._cycle_wires) > 0

    def _analyze_cycles(self) -> tuple[tuple[str, ...], tuple[Wire, ...]]:
        """Detect cycles and compute execution order.

        Uses DFS to find back-edges (cycle-forming wires).
        The remaining acyclic subgraph is topologically sorted.

        Note: Uses shared implementation from feedbax._graph.topological_sort
        """
        from feedbax._graph import detect_cycles_and_sort

        # Build adjacency from wires
        adjacency = {name: set() for name in self.nodes}
        wire_lookup = {}  # (source, target) -> Wire
        for wire in self.wires:
            adjacency[wire.source_node].add(wire.target_node)
            wire_lookup[(wire.source_node, wire.target_node)] = wire

        # Detect cycles and get execution order
        execution_order, back_edges = detect_cycles_and_sort(adjacency)

        # Convert back edges to wires
        cycle_wires = tuple(
            wire_lookup[(src, tgt)]
            for src, tgt in back_edges
            if (src, tgt) in wire_lookup
        )

        return tuple(execution_order), cycle_wires

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
        n_steps: int | None = None,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute the graph.

        If the graph has cycles, iterates using lax.scan.
        If acyclic, executes nodes in topological order once.

        Args:
            inputs: Dict mapping input port names to values.
                    For iterating graphs, values should have shape (n_steps, ...).
            state: Current state
            key: PRNG key
            n_steps: Number of iterations (required if graph has cycles and
                     can't be inferred from input shape)
        """
        if self._needs_iteration:
            return self._call_with_iteration(inputs, state, key=key, n_steps=n_steps)
        else:
            return self._call_single_step(inputs, state, key=key)

    def _call_single_step(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute acyclic graph once."""
        keys = jax.random.split(key, len(self.nodes))

        # Accumulator for all port values produced during execution
        port_values: dict[tuple[str, str], PyTree] = {}

        # Bind external inputs to internal ports
        for ext_port, (node_name, node_port) in self.input_bindings.items():
            port_values[(node_name, node_port)] = inputs[ext_port]

        # Execute nodes in topological order
        for node_name, node_key in zip(self._execution_order, keys):
            node = self.nodes[node_name]

            # Gather inputs for this node from port_values
            node_inputs = {}
            for port_name in node.input_ports:
                key_tuple = (node_name, port_name)
                if key_tuple in port_values:
                    node_inputs[port_name] = port_values[key_tuple]

            # Execute node
            node_outputs, state = node(node_inputs, state, key=node_key)

            # Store outputs in port_values
            for port_name, value in node_outputs.items():
                port_values[(node_name, port_name)] = value

        # Apply wires (non-cycle)
        for wire in self.wires:
            if wire not in self._cycle_wires:
                source_key = (wire.source_node, wire.source_port)
                target_key = (wire.target_node, wire.target_port)
                if source_key in port_values:
                    port_values[target_key] = port_values[source_key]

        # Collect graph outputs
        outputs = {}
        for ext_port, (node_name, node_port) in self.output_bindings.items():
            outputs[ext_port] = port_values[(node_name, node_port)]

        return outputs, state

    def _call_with_iteration(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
        n_steps: int | None = None,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute graph with cycles using lax.scan."""
        # Infer n_steps from input shape if not provided
        if n_steps is None:
            # Get first input array and check its leading dimension
            first_input = next(iter(inputs.values()))
            first_leaf = jax.tree_util.tree_leaves(first_input)[0]
            n_steps = first_leaf.shape[0]

        keys = jax.random.split(key, n_steps)

        def step(carry, args):
            state, prev_cycle_values = carry
            step_inputs, step_key = args

            # For cycle wires, use previous step's output as this step's input
            # prev_cycle_values: dict mapping (target_node, target_port) -> value
            port_values = dict(prev_cycle_values)

            # Add external inputs for this step
            for ext_port, (node_name, node_port) in self.input_bindings.items():
                port_values[(node_name, node_port)] = step_inputs[ext_port]

            # Execute single step
            outputs, state = self._execute_step(port_values, state, key=step_key)

            # Extract cycle values for next step
            new_cycle_values = {}
            for wire in self._cycle_wires:
                source_key = (wire.source_node, wire.source_port)
                target_key = (wire.target_node, wire.target_port)
                new_cycle_values[target_key] = outputs.get(source_key, port_values.get(source_key))

            return (state, new_cycle_values), outputs

        # Initialize cycle values from state (first step)
        init_cycle_values = self._get_initial_cycle_values(state)

        # Index inputs for each step
        step_inputs_seq = jax.vmap(lambda i: jt.map(lambda x: x[i], inputs))(jnp.arange(n_steps))

        # Run scan
        (final_state, _), outputs_seq = lax.scan(
            step,
            (state, init_cycle_values),
            (step_inputs_seq, keys),
        )

        return outputs_seq, final_state

    # ========== Graph Surgery API ==========

    def add_node(self, name: str, component: Component) -> "Graph":
        """Return a new graph with an additional node."""
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        return eqx.tree_at(
            lambda g: g.nodes,
            self,
            {**self.nodes, name: component},
        )

    def remove_node(self, name: str) -> "Graph":
        """Return a new graph with a node removed.

        Also removes all wires connected to that node.
        """
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' does not exist")

        new_nodes = {k: v for k, v in self.nodes.items() if k != name}
        new_wires = tuple(
            w for w in self.wires
            if w.source_node != name and w.target_node != name
        )

        return eqx.tree_at(
            lambda g: (g.nodes, g.wires),
            self,
            (new_nodes, new_wires),
        )

    def add_wire(self, wire: Wire) -> "Graph":
        """Return a new graph with an additional wire."""
        # Validate wire endpoints exist
        if wire.source_node not in self.nodes:
            raise ValueError(f"Source node '{wire.source_node}' does not exist")
        if wire.target_node not in self.nodes:
            raise ValueError(f"Target node '{wire.target_node}' does not exist")

        return eqx.tree_at(
            lambda g: g.wires,
            self,
            self.wires + (wire,),
        )

    def remove_wire(self, wire: Wire) -> "Graph":
        """Return a new graph with a wire removed."""
        new_wires = tuple(w for w in self.wires if w != wire)
        return eqx.tree_at(
            lambda g: g.wires,
            self,
            new_wires,
        )

    def insert_between(
        self,
        node_name: str,
        component: Component,
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str,
    ) -> "Graph":
        """Insert a component between two connected nodes.

        Removes the wire from source to target.
        Adds wires: source -> new_node -> target.

        Assumes the new component has ports "input" and "output".
        """
        # Remove old wire
        old_wire = Wire(source_node, source_port, target_node, target_port)
        graph = self.remove_wire(old_wire)

        # Add new node
        graph = graph.add_node(node_name, component)

        # Add new wires
        graph = graph.add_wire(Wire(source_node, source_port, node_name, "input"))
        graph = graph.add_wire(Wire(node_name, "output", target_node, target_port))

        return graph
```

---

## Iteration

With the cycle-based iteration model, **Graphs with cycles iterate internally**.
For acyclic Graphs or individual Components, a standalone iteration function is provided.

```python
import jax
import jax.lax as lax
import jax.random as jr
import jax.tree as jt
from equinox.nn import State

def iterate_component(
    component: Component,
    inputs: PyTree,  # Shape: (n_steps, ...)
    init_state: State,
    n_steps: int,
    key: PRNGKeyArray,
    state_filter: PyTree[bool] = True,  # Which parts of state to save
) -> tuple[PyTree, State, PyTree | None]:
    """Iterate an acyclic component over multiple timesteps.

    For Graphs with cycles, use graph(inputs, state, key=key) directly,
    which handles iteration internally.

    Args:
        component: An acyclic Component
        inputs: PyTree with leading time dimension of size n_steps
        init_state: Initial state
        n_steps: Number of steps to iterate
        key: PRNG key
        state_filter: Which parts of state to save in history.
                      If True, save all. If False, save none.

    Returns:
        outputs: PyTree of outputs with leading time dimension
        final_state: State after all steps
        state_history: PyTree of states (or None if state_filter=False)
    """
    keys = jr.split(key, n_steps)

    save_history = state_filter is not False

    def step(carry, args):
        state = carry
        step_input, step_key = args

        outputs, new_state = component(step_input, state, key=step_key)

        if save_history:
            state_to_save = eqx.filter(new_state, state_filter)
            return new_state, (outputs, state_to_save)
        else:
            return new_state, outputs

    # Index into inputs for each step
    step_inputs = jax.vmap(lambda i: jt.map(lambda x: x[i], inputs))(jnp.arange(n_steps))

    if save_history:
        final_state, (outputs, state_history) = lax.scan(step, init_state, (step_inputs, keys))
        return outputs, final_state, state_history
    else:
        final_state, outputs = lax.scan(step, init_state, (step_inputs, keys))
        return outputs, final_state, None
```

**Key distinction**:
- **Graphs with cycles**: Call `graph(inputs, state, key=key)` - iteration is automatic
- **Acyclic Graphs/Components**: Use `iterate_component()` for external iteration
- **Nested iteration**: Composes naturally (inner cycles iterate inside outer iteration)

---

## Task Integration

Tasks provide trial specifications and evaluate models.

```python
class AbstractTask(Module):
    """Base class for tasks."""

    loss_func: AbstractVar[AbstractLoss]
    n_steps: AbstractVar[int]

    @abstractmethod
    def get_train_trial(self, key: PRNGKeyArray) -> TaskTrialSpec:
        """Generate a training trial specification."""
        ...

    def eval_trials(
        self,
        model: Component,
        trial_specs: TaskTrialSpec,
        keys: PRNGKeyArray,
    ) -> tuple[PyTree, State]:
        """Evaluate model on a batch of trials."""

        def eval_single(trial_spec, key):
            key_init, key_run = jr.split(key)

            # Initialize state (collects all StateIndex instances)
            _, init_state = eqx.nn.make_with_state(lambda: model)(key=key_init)

            # Apply task-specified initial conditions
            for where, init_val in trial_spec.inits.items():
                # Need to find the StateIndex and set it
                init_state = self._apply_init(model, init_state, where, init_val)

            # Set intervention params in state
            for label, params in trial_spec.intervene.items():
                init_state = self._set_intervention_params(model, init_state, label, params)

            # Run consistency update
            init_state = model.state_consistency_update(init_state)

            # Iterate model
            outputs, final_state = iterate_model(
                model,
                trial_spec.inputs,
                init_state,
                self.n_steps,
                key_run,
            )

            return outputs, final_state

        return jax.vmap(eval_single)(trial_specs, keys)
```

---

## TaskComponent

A unified component for the "other side" of the agent loop. Can work in two modes:

```python
class TaskComponent(Component):
    """The task/environment side of an agent loop.

    Attributes:
        task: The AbstractTask (provides n_steps, generates trial specs)
        trial_spec: Current trial specification (may have pre-computed sequences)
        mode: "open_loop" (pre-computed sequences) or "closed_loop" (feedback-driven)
    """

    task: AbstractTask
    trial_spec: TaskTrialSpec
    mode: Literal["open_loop", "closed_loop"] = "open_loop"

    # n_steps comes from task.n_steps (field, not state)
    # When this component is in a cycle, iterator.n_steps == task.n_steps

    input_ports = ("agent_output",)  # For closed_loop mode
    output_ports = ("target", "observation", "intervention_params")

    def __call__(self, inputs, state, *, key):
        if self.mode == "open_loop":
            # Index into pre-computed sequences
            step = state.get(self._step_index)
            outputs = {
                "target": jt.map(lambda x: x[step], self.trial_spec.inputs),
                "intervention_params": jt.map(lambda x: x[step], self.trial_spec.intervene),
                "observation": None,  # Not used in open loop
            }
        else:  # closed_loop
            # Compute from current agent output (RL-style)
            agent_output = inputs["agent_output"]
            env_state = state.get(self._env_state_index)

            # Step environment based on agent action
            new_env_state, obs = self._step_env(env_state, agent_output, key)
            state = state.set(self._env_state_index, new_env_state)

            outputs = {
                "target": self._get_target(new_env_state),
                "observation": obs,
                "intervention_params": {},
            }

        return outputs, state
```

### n_steps Constraint

When a cycle contains a TaskComponent:
- Iterator's `n_steps` is constrained to match `task.n_steps`
- In a UI, these would be linked fields that move together
- Both modes (open/closed loop) have the same episode length

This unifies the "Task" and "Environment" concepts - both have fixed episode lengths,
the difference is whether sequences are pre-computed or computed online.

---

## Example: SimpleFeedback as a Graph

```python
class Network(Component):
    """A simple feedforward network."""

    input_ports = ("target", "feedback")
    output_ports = ("output",)

    layers: eqx.nn.MLP
    hidden_index: StateIndex  # For RNN hidden state, if applicable

    def __call__(self, inputs, state, *, key):
        target = inputs["target"]
        feedback = inputs["feedback"]

        # Concatenate inputs
        x = jnp.concatenate([target, feedback], axis=-1)

        # Forward pass
        output = self.layers(x)

        return {"output": output}, state


class FeedbackChannel(Component):
    """Delayed, noisy feedback of a signal."""

    input_ports = ("input",)
    output_ports = ("output",)

    delay: int
    noise_std: float
    queue_index: StateIndex

    def __init__(self, delay: int, noise_std: float, init_shape: tuple[int, ...]):
        self.delay = delay
        self.noise_std = noise_std
        initial_queue = tuple(jnp.zeros(init_shape) for _ in range(delay))
        self.queue_index = StateIndex(initial_queue)

    def __call__(self, inputs, state, *, key):
        queue = state.get(self.queue_index)

        # Pop oldest, push newest
        output = queue[0]
        new_input = inputs["input"] + self.noise_std * jr.normal(key, inputs["input"].shape)
        new_queue = queue[1:] + (new_input,)

        state = state.set(self.queue_index, new_queue)
        return {"output": output}, state


class Mechanics(Component):
    """Mechanical plant with Diffrax ODE integration."""

    input_ports = ("force",)
    output_ports = ("effector",)

    plant: AbstractPlant
    solver: dfx.AbstractSolver
    dt: float

    plant_state_index: StateIndex
    solver_state_index: StateIndex

    def __call__(self, inputs, state, *, key):
        force = inputs["force"]

        plant_state = state.get(self.plant_state_index)
        solver_state = state.get(self.solver_state_index)

        # ODE term
        term = dfx.ODETerm(self.plant.vector_field)

        # Single step
        new_plant_state, _, _, new_solver_state, _ = self.solver.step(
            term, 0, self.dt, plant_state, force, solver_state, made_jump=False
        )

        # Update state
        state = state.set(self.plant_state_index, new_plant_state)
        state = state.set(self.solver_state_index, new_solver_state)

        # Compute effector position
        effector = self.plant.skeleton.effector(new_plant_state.skeleton)

        return {"effector": effector}, state


class CurlField(Component):
    """Velocity-dependent force field intervention."""

    input_ports = ("effector", "force")
    output_ports = ("force",)

    params_index: StateIndex  # Intervention params set by task

    def __init__(self):
        # Default params: inactive
        default_params = CurlFieldParams(active=False, strength=0.0, angle=0.0)
        self.params_index = StateIndex(default_params)

    def __call__(self, inputs, state, *, key):
        effector = inputs["effector"]
        force = inputs["force"]

        params = state.get(self.params_index)

        # Apply curl field if active
        def apply_field(force, vel):
            rotation = jnp.array([
                [jnp.cos(params.angle), -jnp.sin(params.angle)],
                [jnp.sin(params.angle), jnp.cos(params.angle)],
            ])
            perturbation = params.strength * rotation @ vel
            return force + perturbation

        new_force = jax.lax.cond(
            params.active,
            lambda: apply_field(force, effector.vel),
            lambda: force,
        )

        return {"force": new_force}, state


# Constructing the SimpleFeedback model as a Graph
def make_simple_feedback(
    network: Network,
    mechanics: Mechanics,
    feedback_delay: int = 5,
    feedback_noise: float = 0.01,
) -> Graph:
    """Construct a SimpleFeedback model as an explicit graph.

    This graph has a cycle: mechanics.effector -> feedback.input
    This means calling the graph will iterate automatically.
    """

    feedback = FeedbackChannel(
        delay=feedback_delay,
        noise_std=feedback_noise,
        init_shape=(2,),  # 2D effector position
    )

    return Graph(
        nodes={
            "feedback": feedback,
            "network": network,
            "mechanics": mechanics,
        },
        wires=(
            Wire("feedback", "output", "network", "feedback"),
            Wire("network", "output", "mechanics", "force"),
            # CYCLE WIRE: mechanics output -> feedback input
            # This wire crosses timestep boundaries (step N output -> step N+1 input)
            Wire("mechanics", "effector", "feedback", "input"),
        ),
        input_ports=("target",),
        output_ports=("effector",),
        input_bindings={
            "target": ("network", "target"),
        },
        output_bindings={
            "effector": ("mechanics", "effector"),
        },
    )

# Usage:
# The graph detects the cycle and iterates automatically
# outputs, final_state = simple_feedback(
#     {"target": target_trajectory},  # Shape: (n_steps, ...)
#     init_state,
#     key=key,
# )


# Adding an intervention via graph surgery
def add_curl_field(model: Graph) -> Graph:
    """Insert a curl field between network and mechanics."""

    curl_field = CurlField()

    return model.insert_between(
        node_name="curl_field",
        component=curl_field,
        source_node="network",
        source_port="output",
        target_node="mechanics",
        target_port="force",
    )
```

---

## Migration Path

### Phase 0: Shared Infrastructure

1. Create `feedbax/_graph.py` with shared graph utilities:
   - `topological_sort(graph: dict[str, set[str]]) -> list[str]`
   - `detect_cycles_and_sort(adjacency) -> (order, back_edges)`
   - Refactor `feedbax/analysis/_dependencies.py` to use shared implementation

### Phase 1: Core Infrastructure

1. Implement `Component` base class
2. Implement `Wire` and `Graph` with cycle detection
3. Implement `iterate_component` function
4. Port State management to use Equinox's StateIndex pattern

### Phase 2: Convert Existing Components

Convert in order of dependency (leaf components first):

1. `Channel` -> component with queue StateIndex
2. `AbstractPlant`, `Mechanics` -> component with plant/solver StateIndex
3. `Network` (SimpleStagedNetwork) -> component with hidden StateIndex
4. Intervenors (`CurlField`, `AddNoise`, etc.) -> regular components

### Phase 3: Convert Models to Graphs

1. `SimpleFeedback` -> Graph of components with cycle wires
2. Ensure graph surgery works (insert/remove nodes)
3. Verify nested iteration composes correctly

### Phase 4: Update Task/Training Infrastructure

1. Create `TaskComponent` adapter for graph composition
2. Update `AbstractTask.eval_trials` for new graph interface
3. Update `TaskTrainer` if needed
4. Update intervention scheduling to use graph surgery + state initialization

### Phase 5: Cleanup

1. Remove old `AbstractStagedModel`, `ModelStage`, `Iterator`, `ForgetfulIterator`
2. Remove `ModelInput` (no longer needed)
3. Update documentation and examples

### Phase 6: Close Issues

After confirming satisfaction with the implementation:

1. Close resolved issues with detailed remarks:
   - #10 (DAG Architecture): Core of this implementation
   - #11 (Model Input Routing): Eliminated; intervention params in State
   - #12 (Intervenor Type Association): Intervenors are regular components
   - #13 (Abstract Final Pattern): No more special intervenors field
   - #15 (Model Stage Typing): Explicit graph structure replaces stages
   - #33 (Iterator Interface): Iterator eliminated; cycles imply iteration

2. Update any related issues that are partially addressed

---

## Issues Resolved by This Architecture

1. **#10 DAG Architecture** - Core of this spec
2. **#11 Model Input Routing** - Eliminated; intervention params are in State
3. **#12 Intervenor Type Association** - Intervenors are regular components
4. **#13 Abstract Final Pattern** - No more intervenors field confusion
5. **#15 Model Stage Typing** - No more stages; graphs have explicit structure
6. **#33 Iterator Interface** - Iterator eliminated; models are single-step

---

## Fields vs State Design Guidance

When designing a component, decide for each value:

| Question | If Yes → | If No → |
|----------|----------|---------|
| Does it affect JAX tracing? (shapes, loop counts) | Field | Could be either |
| Does it vary between trials? | State | Probably Field |
| Does it change during a single trial? | State | Could be either |
| Is it a "toggle" on the component? (active, enabled) | Field | - |
| Would changing it require "a different component"? | Field | State |

**Start with Field, promote to State if needed.** The architecture supports either.

### Online Learning Support

For online weight updates during a trial, put weights in State:

```python
class OnlineLearningNetwork(Component):
    """Network with weights in State for online learning."""

    weights_index: StateIndex  # Weights live in State, not as field
    learning_rate: float  # This is a field (doesn't change during trial)

    def __call__(self, inputs, state, *, key):
        weights = state.get(self.weights_index)
        output = forward_pass(inputs["input"], weights)

        # Online weight update (Hebbian, gradient descent, etc.)
        new_weights = self.learning_rule(weights, inputs, output)
        state = state.set(self.weights_index, new_weights)

        return {"output": output}, state
```

No architectural changes needed - StateIndex can hold anything, including weights.

### Noise Unification

Noise components can be unified with intervention-style parameters:

```python
class NoiseComponent(Component):
    """Adds noise with optional per-trial scale."""

    scale_index: StateIndex  # Scale can vary per trial (in State)
    noise_type: str = "normal"  # Fixed (field)

    def __call__(self, inputs, state, *, key):
        scale = state.get(self.scale_index)
        noise = self._sample_noise(key, inputs["input"].shape)
        output = inputs["input"] + scale * noise
        return {"output": output}, state
```

Channels can use this internally, with scale optionally set per-trial by the task.

---

## Open Questions / Future Work

**Deferred (no architectural impact):**

1. **Port typing** - Type annotations for validation. Can add later without breaking changes.

2. **Optional ports** - Handle missing inputs gracefully in `__call__`. Local change.

3. **State initialization from trial spec** - API for task to set initial state. Pattern:
   ```python
   # Model exposes addressable state indices
   index = model.get_state_index("mechanics.effector")
   state = state.set(index, trial_spec.inits[...])
   ```
   Details to work out during implementation.

4. **Visualization** - Graph structure is explicit and can be visualized. Future web app feature.

5. **Multiple ODE systems** - Keep Diffrax internal to components. Can aggregate later if needed.

6. **Memory-efficient iteration** - Use `state_filter` parameter in iteration (like ForgetfulIterator's `memory_spec`). Already in spec.

7. **Adaptive-length trials** - See GitHub issue. Would require dynamic iteration, currently hard with JAX.

---

## Summary

The eager graph architecture provides:

| Aspect | Old (Staged) | New (Graph) |
|--------|--------------|-------------|
| Model structure | OrderedDict of stages | Explicit nodes + wires |
| State management | Parallel Model + ModelState classes | StateIndex pattern (unified) |
| Iteration | Iterator wrapper class | Cycles in graph → automatic iteration |
| Interventions | Special intervenors field + routing | Regular components via graph surgery |
| Input routing | ModelInput(value, intervene) | Dict of port values; params in State |
| Composability | Stages call nested models | Graphs contain graphs (hierarchical) |
| Feedback loops | Implicit in state passing | Explicit cycle wires |
| Task integration | Separate from model | Can be a Component in agent-env loop |
| Debugging | Step through stages | Step through nodes in topo order |
| UI potential | Limited | Direct graph visualization; surgery = editing |

### Key Insights

1. **Cycles = iteration**: No separate Iterator class. A graph with a cycle iterates automatically.

2. **Everything is a Component**: Network, mechanics, channels, interventions, even tasks - same interface.

3. **Graph surgery is intervention**: Adding a curl field = inserting a node + rewiring.

4. **State is unified**: Model state, solver state, intervention params - all in one State object.

5. **Hierarchical composition**: A Graph is a Component. Graphs can contain graphs.
