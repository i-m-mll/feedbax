# Commit: Eager Graph Cutover (Spec → Implementation)

## Overview

This commit replaces Feedbax’s staged model pipeline with an explicit eager‑graph
architecture. Models are now graphs of components with named ports and explicit
wires; cycles imply iteration; and all persistent values (including intervention
parameters) live in a unified Equinox `State`. This delivers a simpler execution
model, clearer wiring semantics, and a safer foundation for graph surgery,
interventions, and nested composition.

This README is a commit‑specific, feature‑oriented summary intended to help
readers unfamiliar with the prior system orient quickly.

## Motivation (Why this change?)

The staged architecture accumulated friction around:

1. **Implicit routing**: `ModelInput` and intervenor routing required special
   logic and non‑local knowledge about where values “should” go.
2. **Special‑cased intervenors**: interventions weren’t regular components,
   which complicated composition, traversal, and graph editing.
3. **Iteration indirection**: `Iterator`/`ForgetfulIterator` were wrappers for
   single‑step models even though time‑steps are a property of cycles in the
   computation graph.
4. **Debuggability**: execution order and data flow were implicit; eager graphs
   make these relationships explicit and inspectable.

The new architecture resolves these by making structure explicit and state
uniform, without adding a large framework layer between user code and JAX.

## Major Changes & Technical Improvements

### 1) Eager Graph Core (Component, Wire, Graph)
Models are explicit graphs. Each node is a `Component` with input/output ports.
The `Graph`:
- validates wiring,
- detects cycles and uses `lax.scan` for iteration,
- exposes graph surgery (add/remove nodes & wires, insert between).

This removes the need for staged model “stage lists” and external iterators.

**Example: Minimal cyclic graph**
```python
class Inc(Component):
    input_ports = ("x",)
    output_ports = ("x",)
    def __call__(self, inputs, state, *, key):
        return {"x": inputs["x"] + 1}, state

graph = Graph(
    nodes={"inc": Inc()},
    wires=(Wire("inc", "x", "inc", "x"),),  # cycle
    input_ports=(),
    output_ports=("x",),
    input_bindings={},
    output_bindings={"x": ("inc", "x")},
)

outputs, _ = graph({}, init_state, key=key, n_steps=3,
                   cycle_init={("inc", "x"): jnp.array(0)})
```

### 2) Unified State via StateIndex
All persistent values (queues, solver state, intervention params, etc.) live in
an Equinox `State`, with each component owning its `StateIndex` instances.

`init_state_from_component` now walks a component tree and collects *all*
`StateIndex` values, including intervention params and nested graphs. This makes
initialization robust and consistent across components.

### 3) Interventions as Components
Interventions (curl fields, noise, clamps, constant inputs) are now ordinary
components with parameters stored in state. Scheduling attaches parameter
specs to task trial specs, and graph surgery inserts interventions into the
model wiring.

**Example: Inserting a curl field**
```python
model = model.insert_between(
    node_name="curl_field",
    component=CurlField(),
    source_node="net",
    source_port="output",
    target_node="mechanics",
    target_port="force",
)
```

### 4) SimpleFeedback Rebuilt as a Graph
The feedback controller is now a `Graph` composed of:
- feedback channels (delays/noise),
- neural controller (`SimpleStagedNetwork` as a component),
- optional force filter,
- mechanics component.

A cycle (mechanics → feedback) triggers internal iteration. `state_view_fn`
and `state_consistency_fn` provide convenient state projections and inverse
kinematics updates.

### 5) Task/Training Updates
Tasks now evaluate models via `run_component`/`iterate_component` rather than
iterators. Intervention params and init‑state overrides are applied by directly
setting `StateIndex` values in `State`. Training uses the same flow.

### 6) TaskComponent Adapter
`TaskComponent` provides an adapter to treat tasks/environments as components.
This unifies open‑loop and closed‑loop usage for agent‑environment cycles in a
single graph abstraction.

### 7) Documentation & API Refresh
Docs were updated to reflect the eager‑graph architecture:
- staged/iterator API pages were replaced with graph/iteration utilities,
- intervention docs now list component‑style APIs,
- structure and index docs removed staged/iterator language.

## Potential Issues / Downsides

- **Breaking API changes**: staged model classes, iterators, and `ModelInput`
  are gone. Downstream code using those APIs will need refactoring.
- **Cycle initialization**: cyclic graphs require explicit initial values for
  cycle targets. The graph now raises a clear error if they’re missing.
- **StateIndex internals**: state initialization relies on Equinox’s StateIndex
  shape; there is a defensive fallback, but upstream changes could require
  minor adjustments.
- **Example notebooks**: notebooks/examples may still mention staged concepts.
  The core docs are updated, but some tutorial content likely needs refresh.

## Deferred (Intentionally Not Included Here)

These are explicitly *out of scope* for this commit:

1. **Notebook/example rewrites** to fully align with eager graphs.
2. **Closed‑loop TaskComponent integrations** in higher‑level tasks/training
   (the adapter exists, but RL‑style loops aren’t wired through the training API).
3. **Optional UI tooling** for graph visualization/editing; the current graph API
   provides the necessary structure but no UI layer yet.

## Testing

New tests were added (graph iteration, component behavior, TaskComponent, and
state initialization). Tests were not executed in this sandbox because the
runtime lacked `equinox`/`pytest`. Suggested command:

```
python -m pytest tests/test_graph.py tests/test_components.py tests/test_task_component.py tests/test_state_init.py -q
```

---

If you want the README to include additional examples (e.g., a full SimpleFeedback
graph construction) or a migration checklist, I can expand it further.
