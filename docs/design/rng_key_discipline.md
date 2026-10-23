# RNG key discipline

Feedbax treats a JAX PRNG key as a single-use capability. A function that receives
one key owns splitting it into the independent streams it needs, and it must pass
only child keys to downstream stochastic work.

## Boundary rules

- Callers pass one key to a component, task, reducer, or analysis entry point.
- The callee splits that key once for its semantic streams: trial generation,
  intervention parameters, self-dependencies, model rollout, cache identity, or
  Monte Carlo samples.
- Batched work receives batched keys. The parent splits once for the batch axis,
  then vmaps the callee over those keys.
- PyTree callable leaves receive per-leaf keys unless the API explicitly marks a
  callable as sharing a stream.
- Defaults must not capture `PRNGKey(0)` in function signatures. Constructors may
  use factory defaults only for stable object fields; stochastic computations must
  either require a key or split a caller-provided key.

## Current owned streams

- `AbstractTask.get_train_trial_with_intervenor_params` splits training keys into
  trial, intervention-parameter, and self-dependency streams.
- `AbstractTask.validation_trials` derives validation trials, validation
  intervention parameters, and validation self-dependencies from separate streams
  under `seed_validation`.
- `evaluate_intervenor_params` splits per callable parameter leaf.
- `AddNoise` splits per signal PyTree leaf.
- `SimpleStagedNetwork` splits separate streams for stochastic hidden cells and
  network-level hidden noise.
- Hutchinson reducers require an explicit key and split it by sample.
- Legacy evaluation state pickle filenames include a digest of the evaluation key
  when manifest-canonical state cache identity is not used.

## Review gate

Run `scripts/check_rng_discipline.py` to catch the reviewed RNG anti-patterns
that motivated this convention: validation stream reuse, HOLD/REACH key reuse,
one-key PyTree noise, fixed Hutchinson defaults, and PRNG-less legacy state cache
filenames. The gate is intentionally narrow; it is a regression tripwire, not a
complete static analysis for every valid JAX key misuse.
