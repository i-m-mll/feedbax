# Commit: [eager-models] Add spec for eager graph architecture

## Overview

This commit introduces a comprehensive specification for migrating feedbax from
the current "staged model" architecture to an "eager graph" architecture. The
spec addresses several long-standing issues: awkward ModelInput routing,
special-cased intervenors, and the redundant Iterator wrapper around models.

## Changes

### SPEC_EAGER_MODELS.md (new)

A 1000+ line specification document covering:

**Core Types**
- `Component`: Base class with named input/output ports
- `Wire`: Typed connections between ports with optional transformations
- `Graph`: Container that is itself a Component, enabling hierarchical composition

**Cycles and Iteration**
- Cycles in the graph (e.g., mechanics.effector → feedback.input) are detected
  at construction time
- Detected cycles automatically imply iteration via `lax.scan`
- Eliminates the need for a separate Iterator class

**State Management**
- Adopts Equinox's StateIndex/State pattern
- Each component creates its own StateIndex instances
- State object threaded through calls, enabling intervention parameters without
  special routing

**TaskComponent**
- Unified Task/Environment concept with open_loop and closed_loop modes
- n_steps is a field on the task, constrains iteration when task is in a cycle

**Migration Path**
- 6-phase migration plan from current architecture to new system
- Identifies shared code opportunities (topological_sort in analysis module)

## Rationale

The current architecture has accumulated several pain points:

1. **ModelInput routing**: The `ModelInput(value, intervene)` pattern forces all
   intervention parameters through a single channel, requiring awkward routing
   logic in staged models.

2. **Intervenors as special cases**: Currently intervenors are inserted via
   PyTree surgery and require special handling. In the new architecture, they
   become regular components connected via wires.

3. **Iterator redundancy**: The Iterator class wraps models to handle timestep
   iteration, but this is better expressed as cycle detection in the graph
   structure itself.

4. **Future UI alignment**: An explicit graph structure with typed ports maps
   cleanly to visual flowchart representations (planned web UI).

The eager (vs lazy) evaluation choice aligns with Equinox's philosophy: explicit
control flow, debuggable execution, and no framework magic between user code and
JAX.

## Files Changed

- `SPEC_EAGER_MODELS.md`: New specification document (1048 lines)
