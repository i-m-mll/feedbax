# jax-cookbook provenance

This directory contains the `jax_cookbook` Python package from
<https://github.com/i-m-mll/jax-cookbook> at commit
`95060b2448cafb0783ecdc7220818eab9a61c630` (version `0.1.0`).

The upstream Apache License 2.0 is retained in `LICENSE`. The pinned upstream
tree contains no `NOTICE` file.

Feedbax changes four files solely to relocate their internal absolute imports
under `feedbax._vendor.jax_cookbook`:

- `_array.py`
- `_func.py`
- `_vmap.py`
- `progress.py`

All other Python source files retain the exact upstream bytes. Feedbax source
imports use the internal namespace directly; this bundle does not install or
alias the external `jax_cookbook` namespace.
