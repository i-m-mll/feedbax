# Chained phase qualification

The executable baseline is corrected Feedbax `6e62b55f` with the signed `beae372e` preflight and downstream `50993e39`. The adopted contract is [artifact-version:a0c9074e1076]. This branch changes qualification code and tests. Production checkpoint and Evaluate contracts, runtime names, observer products, and automatic reuse remain gated.

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

## Jobs readiness and restart

The real local daemon with controlled worker events kept a completion dependency queued after an interior publication. The bound attempt's success released it once. A checkpoint stop instead made the dependency terminally blocked, and a successful continuation of the predecessor did not rebind it. The daemon edge does not authenticate a receipt and final checkpoint as a joint product. The native scientific execution and the daemon fixture qualify different boundaries.

An ordinary nominal restart used the authenticated update-1 pointer in a fresh process. It committed only update 2 and reproduced the uninterrupted model, optimizer, PRNG and counter semantic roots exactly. The source receipt and latest pointer retained their hashes. [artifact-version:1d347a93d7e3] [artifact-version:2fb3ef383b3e]

Both adaptive stops committed update 3 and published cancelled receipts. The saved checkpoints use origin 2 and a segment length of 2, although only one successor update committed. They omit the continuation-applied metadata. Both fresh processes refused with `checkpoint continuation source offset mismatch; lineage_total=4 requested=2` before any update or new checkpoint. The Planner also identified later guards that compare resumed progress and optimizer count with the original origin. Preserve all four values independently: origin 2, current progress 3, one committed successor update, and target 4. The chained readiness fence and exact successor restart remain open. [artifact-version:1d347a93d7e3] [issue:1f46470] [issue:de9e2ac]

The affected planner conservatively selected all tests for the new qualification scripts. Nineteen scoped host, executor and worker checks passed, including mapped resume, declared wrapper construction, and fork custody. The proof and tests leave the production implementation unchanged. [artifact-version:1445368ac3f0]
