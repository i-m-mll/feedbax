# Penzai & JAX Ecosystem Analysis for Feedbax (Integrated)

## Executive Summary

- **Adopt Treescope** for visualization immediately; it is standalone and works with any PyTree.
- **Consider Penzai selectors** for expressive tree surgery on PyTrees without adopting Penzai's model core.
- **Skip Penzai model core** unless you want its Variable-based paradigm and NN-centric workflow.
- **Named axes**: Haliax is the more explicit/robust system; Penzai's `pz.nx` is lighter-weight. If you are not feeling axis pain yet, keep jaxtyping + disciplined naming.
- **Chex** is a testing and runtime assertion toolkit that complements (not replaces) jaxtyping; it is optional but valuable for JAX-specific testing and trace/value checks.
- **Model-as-data is mostly a convention**: you can make Feedbax's explicit representations more inspectable by standardizing NN internals and metadata, without adopting Penzai's model core.

---

## How Penzai Works (Architectural View)

Penzai is not just a toolkit; it is a **model-as-data** paradigm designed primarily for mechanistic interpretability workflows. The architecture splits into independent pieces:

1. **Selectors (`pz.select`)**: A general PyTree querying and editing system. Works on any PyTree and is not tied to Penzai's model core.
2. **Treescope integration**: Rich interactive visualization of PyTrees/arrays. Treescope is standalone and can be used with or without Penzai.
3. **Named axes (`pz.nx`)**: Lightweight named-axis support that lifts normal JAX functions using `nmap` and `untag`/`tag`.
4. **Model core (`pz.Struct` + v2 variables)**: A full modeling paradigm with explicit Variable objects (`Parameter`, `StateVariable`) embedded as leaves in the model tree.

**Key point**: You can take selectors and Treescope without adopting the Penzai model core.

---

## Fit with Feedbax Eager Models

Your eager core (Component/Graph with StateIndex) is already a strong match for Feedbax’s domain. It is explicit, port-based, and functional in its state updates. That is a different paradigm than Penzai’s variable objects embedded in the model tree.

**Important nuance**: Penzai does not have a monopoly on hierarchy or “model-as-data.” You already can (and sometimes do) represent NN internals as nested Equinox Modules. The difference is that Penzai treats this as the primary, uniform representation for the *entire* model stack, and its tooling assumes that style.

**What Penzai adds that your eager core does not**:
- A uniform layer interface geared around interpretability workflows.
- Built-in variable objects that allow easy parameter sharing or state mutation *inside* the tree.
- A strong bias toward model inspection and structural editing.

**What your eager core does better**:
- Explicit graph wiring and ports that reflect agent/mechanics/task structure.
- Purely functional state management via `StateIndex` and `State`.
- First-class graph surgery for interventions (already part of the design).

**If you want more “model-as-data” in Feedbax**:
- Standardize a canonical tree representation of NN internals (consistent layer types + metadata).
- Make selectors/visualization first-class on that representation.
- Treat the explicit representation as the source of truth, not just a debugging artifact.

### Penzai Model Core vs Feedbax Eager Core (Table)

| Aspect | Penzai model core | Feedbax eager core |
|---|---|---|
| Core abstraction | `pz.Struct` frozen PyTree dataclasses | `Component` and `Graph` (Equinox Modules) |
| Parameter/state handling | Explicit variable objects (`Parameter`, `StateVariable`) that are mutable and can be shared | `StateIndex` with explicit state passing (functional) |
| Execution model | Single-argument `__call__` with keyword side inputs | Dict-based port inputs/outputs, graph execution with explicit wiring |
| Composition | Hierarchical nesting of layers | Graphs are Components; wiring defines dataflow |
| Mutation pattern | Variables are mutable objects; special utilities functionalize them for JAX | No mutable parameters in the forward pass; state updated explicitly via State |
| Editing models | `pz.select` and selectors integrate with `Struct` | Graph surgery via nodes/wires; Equinox tools for PyTree edits |
| Intended use | Mechanistic interpretability and model-as-data workflows | Explicit graph modeling of agents, mechanics, channels, and NN components |

**Conclusion**: Penzai model core is a paradigm shift. It does not obviously outperform the eager core for your use cases, except if you specifically want its variable system and interpretability-oriented modeling style.

---

## Tree Operations: Penzai Selectors + Treescope vs Equinox utilities

| Goal | Penzai selectors | Equinox utilities |
|---|---|---|
| Targeted edits | `pz.select(...).at(...).set(...)` with method chaining and type/keypath-based selectors | `eqx.tree_at` for focused replacement of a subtree |
| Partitioning | `Selection.partition()` + `pz.combine(...)` | `eqx.filter` / `eqx.partition` / `eqx.combine` |
| Visual inspection | Treescope for interactive PyTree/array visualization | No built-in visualizer; relies on external tools |
| Model integration | Works with any PyTree; `.select()` convenience for `pz.Struct` | Works with any PyTree; native in Equinox |

**Practical difference**:
- `eqx.tree_at` is ideal for simple, known edits.
- Penzai selectors are more expressive for chained, type-driven, or pattern-based selections.
- Treescope is orthogonal and valuable regardless of which editing utilities you prefer.

**Note on integration**: `pz.select` works on any PyTree, so it is not “special” to Penzai model objects. The `.select()` method on `pz.Struct` is just a convenience. This means you can use selectors with Feedbax models without adopting Penzai’s model core.

**Recommendation**: Use Equinox for filtering/partitioning and optionally add Penzai selectors for more expressive tree surgery. Treescope is the obvious win.

---

## Named Axes: Penzai vs Haliax

**Penzai named axes (`pz.nx`)**:
- `NamedArray` with **locally positional** axis names.
- Functions are lifted with `nmap`, using `untag`/`tag` to convert between named and positional axes.
- Lightweight and easy to use as a wrapper over existing JAX code.

**Haliax**:
- Uses explicit `Axis` objects that carry **name and size**.
- Encourages axis-aware APIs and more explicit shape reasoning.
- Better suited to large-scale training and sharding-aware workflows.

**Using both**:
- Possible but usually not worthwhile because each defines its own `NamedArray` type and requires conversions.
- A sensible hybrid is Haliax for axis semantics and Penzai selectors/Treescope for tree operations.

**Recommendation**: If you need named axes, prefer Haliax; otherwise keep jaxtyping + consistent dimension conventions.

---

## Chex vs Jaxtyping (Static & Runtime Checking)

### Static checking

- **jaxtyping**: Static type checkers treat `Float[Array, "batch features"]` as just `Array`. Shape/dtype constraints are *not* validated statically.
- **chex**: No static typing support; it is runtime assertions and testing utilities.

### Runtime checking

- **jaxtyping**: `@jaxtyped` + a runtime type checker enforces dtype/shape constraints. In `jit`, checks run at trace time, not in the compiled code.
- **chex**: Explicit assertion helpers for shape/rank/dtype, plus value checks via `chexify` for jitted code.

### Side-by-side example

**Jaxtyping (annotation-driven)**

```python
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype
import jax.numpy as jnp

@jaxtyped(typechecker=beartype)
def proj(x: Float[Array, "batch d"]) -> Float[Array, "batch d"]:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
```

**Chex (assertion-driven)**

```python
import chex
import jax.numpy as jnp

def proj(x):
    chex.assert_rank(x, 2)
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
```

**Value checks inside `jit`**

```python
import chex
import jax
import jax.numpy as jnp

@chex.chexify
@jax.jit
def proj_safe(x):
    chex.assert_tree_all_finite(x)
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
```

### Where they overlap

- Both can enforce shape/dtype constraints at runtime.
- Both integrate with JAX tracing and can be used inside `jit` (jaxtyping checks at trace time; chex static assertions are trace-time, and value assertions work via `chexify`).

### What Chex adds beyond jaxtyping

- **JAX-specific testing utilities** (variants, device testing, numerical checks).
- **Trace behavior assertions** (`assert_max_traces`).
- **Value checks inside jitted code** (`chexify`).
- **Convenient assertion helpers** for rank, shape, equality, gradients.

### Where jaxtyping is superior

- **Expressive annotations** (named axes/dtypes) that double as documentation.
- **Better integration with IDEs and static type checkers** for overall typing.
- **Single style**: signature annotations plus optional runtime checking.

### Recommendation

- Keep **jaxtyping** for signature-level contracts and clarity.
- Use **chex** selectively in tests or critical runtime checks (finite values, trace control, variant testing).

---

## Recommendations (Condensed)

1. **Adopt Treescope** for inspection in notebooks.
2. **Use Penzai selectors** if you need richer tree surgery than `eqx.tree_at`.
3. **Skip Penzai model core** unless you want its variable-centric paradigm.
4. **Prefer Haliax** if named axes become essential; otherwise keep jaxtyping.
5. **Add Chex optionally** for JAX-specific testing utilities and value checks.

---

## Sources

- Penzai Documentation: https://penzai.readthedocs.io/en/stable/
- Penzai Selectors: https://penzai.readthedocs.io/en/v0.1.4/notebooks/selectors.html
- Penzai Struct: https://penzai.readthedocs.io/en/v0.2.4/_autosummary/leaf/penzai.core.struct.Struct.html
- Penzai Named Axes: https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html
- Penzai V2 API: https://penzai.readthedocs.io/en/v0.1.3/api/penzai.experimental.v2.html
- Penzai Variables: https://penzai.readthedocs.io/en/stable/_autosummary/penzai.core.variables.html
- Penzai StateVariable: https://penzai.readthedocs.io/en/v0.1.5/_autosummary/leaf/penzai.experimental.v2.core.variables.StateVariable.html
- Treescope GitHub: https://github.com/google-deepmind/treescope
- Treescope Docs: https://treescope.readthedocs.io/en/stable/
- Treescope basic_interactive_setup: https://treescope.readthedocs.io/en/latest/api/treescope.basic_interactive_setup.html
- Haliax Documentation: https://haliax.readthedocs.io/en/latest/
- Haliax GitHub: https://github.com/stanford-crfm/haliax
- Equinox Documentation: https://docs.kidger.site/equinox/
- Equinox manipulation: https://docs.kidger.site/equinox/api/manipulation/
- Chex Documentation: https://chex.readthedocs.io/en/latest/index.html
- Chex API: https://chex.readthedocs.io/en/latest/api.html
- Chex GitHub: https://github.com/google-deepmind/chex
- jaxtyping runtime checking: https://docs.kidger.site/jaxtyping/api/runtime-type-checking/
- jaxtyping FAQ: https://docs.kidger.site/jaxtyping/faq/
- safecheck PyPI: https://pypi.org/project/safecheck/
