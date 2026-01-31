# Commit: [develop] Add code generation for Equinox component wrappers

## Overview
This commit adds infrastructure for automatically generating typed Component wrappers
from Equinox neural network classes. The generation script introspects Equinox class
signatures and produces full static-typing-compatible wrapper classes that can be used
as leaf nodes in feedbax computation graphs.

## Changes

### Generation Script (`scripts/generate_eqx_components.py`)
- Defines `ComponentSpec` dataclass specifying per-class metadata (ports, state handling)
- Introspects `__init__` signatures using `inspect` module
- Handles edge cases: `Literal` types, callable defaults, factory defaults, VAR_POSITIONAL
- Generates properly typed wrapper classes with full docstrings
- Organizes output by category (linear, conv, rnn, norm, pool, attention, other)

### Generated Components (`feedbax/eqx_components.py`)
32 Component wrappers covering:
- **Linear**: `Linear`, `Identity`, `MLP`
- **Conv**: `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d`
- **RNN**: `GRUCell`, `LSTMCell`
- **Normalization**: `LayerNorm`, `RMSNorm`, `GroupNorm`, `BatchNorm`
- **Pooling**: `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveMaxPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d`
- **Attention**: `MultiheadAttention`, `RotaryPositionalEmbedding`
- **Other**: `Embedding`, `Dropout`, `PReLU`

## Rationale
The discussion explored three approaches for wrapping Equinox classes:
1. Manual one-to-one wrapper classes (full static typing but tedious)
2. Runtime factory + registration (less boilerplate but loses static typing)
3. Code generation (best of both: committed output has full static typing)

Code generation was chosen because:
- Equinox is stable (~1 release/year), so regeneration is rare
- The generated file is committed and has full IDE support
- The script can be extended to wrap other JAX libraries in the future
- No runtime magic or loss of type information

## Files Changed
- `scripts/generate_eqx_components.py` - Generation script with class specs and introspection
- `feedbax/eqx_components.py` - Auto-generated Component wrappers (do not edit manually)
