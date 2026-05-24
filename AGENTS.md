# feedbax Project Instructions

## Python/JAX Coding Conventions

### Coding Style & Naming

- Follow PEP 8: 4-space indentation and a 100-character soft line limit.
- Use type hints for public APIs.
- Keep imports at the top of files unless a local import is needed for
  performance, optional dependencies, or typing.
- Use `lower_snake_case` for modules, packages, functions, and variables;
  `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants.
- Use Google-style docstrings when docstrings are useful; include shapes and
  dtypes for JAX arrays when relevant.

### Environment Management

- Use `uv` for package management. Do not run `pip install` directly.

### Equinox Modules

- Subclass `equinox.Module` for dataclasses-that-are-PyTrees; do not also add
  `@dataclass`.
- Treat `Module` instances as immutable. Use `equinox.tree_at` or
  `eqx.tree_at` for out-of-place updates; avoid direct attribute assignment.
- Use `eqx.field` for defaults and converters. Rely on `Module`'s default
  PyTree behavior unless custom flattening is truly needed.

### JAX Tree API

- Import once as `import jax.tree as jt` and use `jt.*` consistently
  (`jt.map`, `jt.leaves`, `jt.structure`, `jt.flatten`, `jt.unflatten`).
- Do not use deprecated `jax.tree_*` helpers such as `jax.tree_map` or
  `jax.tree_leaves`.

### jax_cookbook Helpers

- Use `import jax_cookbook.tree as jtree` for PyTree utilities not in core JAX,
  such as `jtree.unzip` and `jtree.get_ensemble`.
- Use `from jax_cookbook import is_type, is_module, is_none` for common
  `is_leaf` predicates and shorthands.

## Project-Specific Rules

- Protected branch: `develop`.
- The repo root tracks `main` for releases only; start implementation work from
  `worktrees/develop/` and create feature worktrees from there.
- The graph is the model. Studio canvas nodes and subgraphs are the source of
  truth; do not synthesize background architecture or silently fall back to
  stale outer params.
- Backward compatibility is not a concern for saved graph formats. Raise clear
  errors rather than preserving fallback paths or compatibility shims.
- Studio needs both processes: frontend `cd web && npm run dev` and backend
  `uv run uvicorn feedbax.web.app:app --port 8000`.
