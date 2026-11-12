# Chained phase qualification

The phase-correctness candidate starts from signed qualification `1dbff054` and uses downstream `50993e39`. The adopted shared contract is [artifact-version:a0c9074e1076]. The candidate is for review under [issue:2f7807e]. Public M2 checkpoint and Evaluate contracts, observer products, and automatic reuse remain gated.

## Historical nominal source

The signed preflight compiled the original flat and rewarm suffixes, which share the governed batch-12,000 nominal source and declare 64 trials per replica across five replicas. The corrected public archive reader refuses the authenticated `pickle.v1` model slot. [artifact-version:eb746487c4ba]

The isolated reader used signed Feedbax `c9b4b5db` with its offline lock and downstream `2f84e27`. The exporter authenticated the archive, manifest, transaction and batch coordinate. It exported 53 typed numerical leaves and verified exact dtype, shape and value hashes after reloading with pickle disabled. The original successful receipt names the exact source checkpoint. [artifact-blob:bcfe3e9afd57] [artifact-version:20aa52e57cc8] [artifact-version:17d9918f4547]

The historical model's initial-hidden weight and bias and their Adam first and second moments contain nonzero values. Current nominal preparation for the original task omits that encoder, and no approved conversion accounts for the historical controller and optimizer layouts. The export preserves numerical evidence. It supplies no current-format successor input. The separate historical one-update adaptive witness remains on its original held-out-probe path. [issue:d347220]

## Independent current-format runs

The bounded fixture trains a newly authored two-update hold-free nominal prefix with the encoder, 32 hidden units, two sampled trials per replica, and five replicas. It authors floating-point zero weight decay before training so the saved optimizer hyperparameter matches the suffix template. The historical and current fixture request identities are distinct.

The prefix receipt and authenticated completion checkpoint provide a single source for both bounded suffixes. Public fork normalization initially refused sealed optimizer mappings. The proof supplies the same ordinary target trees used by the native executor and checks their full declared compatibility projection. An independently produced integer-zero prefix also failed closed at the optimizer weight-decay dtype and weak type. No target transaction or latest pointer survived those refusals. [artifact-version:431b77d277a8]

Both adaptive suffixes bound the same completed prefix receipt and checkpoint. A substituted manifest digest was rejected. Model, optimizer, PRNG and counter semantic roots match the prefix at both bootstraps. The target-only transform initializes the adaptive state on each replica. Both suffixes completed updates 3 and 4 through the native executor. They used two trials per replica, carried the Adam count from 2 to 4 and held schedule origin at 2. The flat rate stayed at approximately `3e-5`. The rewarm rate advanced from approximately `3e-5` to `3.297e-5`. [artifact-version:1d347a93d7e3]

The first compact flat receipt reports one declared kernel preparation step and 16.74 seconds for tracing, lowering and compilation or cache loading together. The sampled indices, applied perturbation shape, losses and committed state establish a real trial batch above one. An independently recomputed per-trial loss reduction and an actual backend compilation count remain open. [artifact-version:318e1fadacfa] [artifact-version:1d347a93d7e3]

Run one suffix per bounded lease to retain its report independently. Use the governed parent bytes from [artifact-blob:56756c94a6d2], the locked Feedbax environment, and the pinned downstream source and dependency import directory used in the preflight.

```sh
uv run --no-sync python scripts/qualify_current_phase_successor.py \
  --downstream "$downstream" --parent "$parent" --hidden-dim 32 \
  --scratch "$first_run" --suffix flat_3e-5
uv run --no-sync python scripts/qualify_current_phase_successor.py \
  --downstream "$downstream" --parent "$parent" --hidden-dim 32 \
  --prefix-root "$first_run" --scratch "$second_run" --suffix rewarm_3e-3
```

## Successor restart

The signed baseline records the two successor refusals and the ordinary nominal restart that reproduced all four checkpoint slots from the update-1 pointer. [artifact-version:1d347a93d7e3] [artifact-version:2fb3ef383b3e]

Each adaptive successor now stops after committing update 3. Its checkpoint retains schedule origin 2, optimizer count 3, one committed update in the authored segment, and the applied continuation request with target 4. A fresh process resumes the exact stopped checkpoint, commits only update 4, and certifies the same model, optimizer, PRNG, counter, and adaptive-state semantic roots as uninterrupted execution. The complete update observations also match, including sampled indices, loss, applied rate, adaptive value, and PRNG. The stopped receipt retains its hash.

A separate history fixture retains the authored segment's complete batch history across a stop and restart. Its resumed checkpoint names the stopped snapshot as its immediate attempt parent and keeps the original prefix as the segment parent. The caller supplies the prefix root when history materialization spans three custody roots. The test also exercises terminal replay with zero updates and a substituted checkpoint-set parent.

## Jobs completion

The local host's opt-in authenticated completion edge follows explicit continuations of its admitted producer. The daemon persists the selected attempt, receipt, and checkpoint together before releasing the dependent. It rechecks the pinned files before delivery. The training worker verifies the joint receipt and checkpoint before publishing completion, and a training consumer verifies the exact binding before executing.

The real PointMass/GRU daemon test stops at update 1, restarts the daemon with the dependent pending, resumes the exact checkpoint through update 2, and pins the successful descendant. A second daemon restart and replacement of the mutable latest pointer leave one dependent delivery bound to the original receipt and checkpoint. Controlled daemon tests exercise unrelated reruns, definitive absence, corruption after binding, and resuming a dependent with its original binding.
