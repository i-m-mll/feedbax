# Sketch: Canonical NN Internal Representation for Feedbax

## Goal

Define a consistent, selector-friendly, and visualization-friendly internal tree for neural network components so that "model-as-data" is real and first-class without adopting Penzai's model core.

## Design Principles

1. **Uniformity**: Every NN component exposes a predictable tree shape.
2. **Queryability**: Types, tags, and metadata make selectors reliable.
3. **Non-intrusive**: Keep Equinox Modules and the eager graph intact.
4. **Source of truth**: The explicit representation is not just a debug artifact.

## Proposed Representation

### 1) Canonical NN Node Types

Define a small set of internal node classes (Equinox Modules) that wrap the "semantic" layer structure:

- `NNBlock` (base type)
- `LinearBlock`, `ConvBlock`, `MLPBlock`, `AttentionBlock`, `NormBlock`, `EmbedBlock`, etc.
- `SequentialBlock` (ordered children)
- `ParallelBlock` (named children)

Each block has:

- `name: str`
- `role: str` (e.g., "q_proj", "mlp_up", "gate")
- `tags: frozenset[str]` (selectors and UI grouping)
- `children: tuple[NNBlock, ...]` or `dict[str, NNBlock]`
- `params: PyTree` (arrays or submodules) - optional for leaves

This is a lightweight layer tree distinct from the full graph wiring.

### 2) Metadata Schema

Define a stable metadata schema used by selectors and visualization:

- `role`: short semantic identifier
- `tags`: free-form categorical tags ("mlp", "attn", "residual", "stateful")
- `path_hint`: optional string for display or stable referencing
- `axes`: optional dimension hints or named-axis tags

This does not replace jaxtyping; it complements it.

### 3) Canonical Accessors

Add accessors on NN components:

- `nn_tree(self) -> NNBlock`
- `nn_metadata(self) -> dict` (optional)
- `nn_summary(self) -> dict` (optional lightweight stats)

This keeps the explicit representation as a real artifact and allows tools to work without deep reflection.

### 4) Selector-Friendly Conventions

Make sure the NN tree:

- Uses consistent class types per role
- Avoids hiding parameters in anonymous lambdas or closures
- Uses tags and roles so that selectors can do pattern-based edits

### 5) Treescope Configuration

Provide a helper that registers renderers for `NNBlock` types and highlights tags/roles.

---

## Example (Illustrative)

```python
class MLPBlock(eqx.Module):
    name: str
    role: str
    tags: frozenset[str]
    children: tuple[eqx.Module, ...]

# A concrete model exposes:
# model.nn_tree() -> NNBlock tree with roles and tags
```

---

## Benefits

- **Selector-friendly surgery** without changing runtime execution.
- **Tree visualization** that matches semantic structure.
- **Uniformity** across NN classes (no special-case logic).
- **Future-proof**: can opt-in to more interpretability tooling later.

---

## Risks / Costs

- Requires discipline to maintain the canonical structure.
- Adds a parallel representation that must stay in sync with runtime params.
- Might feel redundant unless tooling (selectors/visualization) is actively used.

---

## Minimal viable step

Start with one NN class and expose `nn_tree()` returning a simple `SequentialBlock` with tags/roles. Validate that selectors and Treescope are immediately more usable.
