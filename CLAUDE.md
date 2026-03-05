# feedbax

## Python/JAX Coding Conventions

### Coding Style & Naming
- Follow PEP 8: 4-space indentation, 100-char soft line limit, type hints required for public APIs.
- Always place imports at the top of files, except in the rare case that they should be in a conditional for performance or typing reasons.
- Naming: modules/packages `lower_snake_case`; functions/variables `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Docstrings: Google style; include shapes/dtypes for JAX arrays when relevant.

### Environment Management
- Use `uv` for all package management. Do not run `pip install` directly.

### Equinox Modules
- Subclass `equinox.Module` for dataclasses-that-are-PyTrees; do not add `@dataclass` again. `Module` subclasses are already dataclasses and PyTree nodes.
- Treat `Module` instances as immutable. Use `equinox.tree_at` (or `eqx.tree_at`) for out-of-place updates; avoid direct attribute assignment.
- Use `eqx.field` for defaults/converters. Only implement custom flattening when necessary; otherwise rely on `Module`'s default behavior.

### JAX Tree API
- Import once as `import jax.tree as jt` and use the `jt.*` namespace throughout (e.g., `jt.map`, `jt.leaves`, `jt.structure`, `jt.flatten`, `jt.unflatten`).
- Do not use deprecated `jax.tree_*` helpers (e.g., `jax.tree_map`, `jax.tree_leaves`). Prefer `jax.tree.*` consistently.

### jax_cookbook Helpers
- `import jax_cookbook.tree as jtree` for PyTree utilities not in core JAX (e.g., `jtree.unzip`, `jtree.get_ensemble`). Use `jtree.*` for these helpers.
- `from jax_cookbook import is_type, is_module, is_none` for convenient shorthands. For example, `jt.map(..., is_leaf=is_type(tuple))`, `jt.map(..., is_leaf=is_module)` (for `equinox.Module` instances).
