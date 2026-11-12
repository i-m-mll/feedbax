# Adaptive phase seam preflight

The preflight uses corrected Feedbax `6e62b55f`, downstream `50993e39`, and the original governed nominal checkpoint. The downstream checkout is read-only. The public envelope compiler and declared mapped method compiler produce distinct executable specs for the actual `flat_3e-5` and `rewarm_3e-3` rows. Both resolve the same authored nominal parent and declare 4,500 additional updates, five replicas and an actual training batch specification of 64 trials per replica. Reported learning rates are evaluations of the declared schedules.

The original checkpoint archive is [artifact-blob:d1206f225f39]. The governed parent JSON is [artifact-blob:56756c94a6d2]. Resolve both by digest using Mandible, verify the full hashes, and pass the local bytes as `--archive` and `--parent`. Use the locked Feedbax environment and expose the exact downstream source and its pinned `jax_cookbook` package through `PYTHONPATH`. The measured run used only that downstream dependency in a scratch import directory.

```sh
test "$(git -C "$downstream" rev-parse HEAD)" = 50993e39aaca4678216a14267286d32f7713576e
env PYTHONPATH="$PWD:$downstream/src:$dependency_overlay" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  uv run --offline --script /Users/mll/.codex/feedbax-4007bf2-resources/capacity.py run \
  --label "adaptive phase seam report qualification" --ram-gib 2 --disk-gib 0.1 --cpus 2 -- \
  gtimeout 45 uv run --no-sync python scripts/qualify_adaptive_phase_seam.py \
  --downstream "$downstream" --parent "$parent" --archive "$archive" > "$report"
```

Lease `c85fbec2b574c409` authenticated the 5,728,278-byte archive and reproduced the public reader refusal. The model slot uses `pickle.v1`; this reader requires `pickle.v3`. The source destination remained unpublished. Actual trial sampling, initialized state, optimizer carry, updates and restart await compatible authenticated source custody and the native continuation path.

The preserved historical test at downstream `2f84e27` authenticated and reloaded a prepared fork and ran one direct jitted/vmapped update. Its held-out-probe method and original reader differ from this two-suffix preflight. [issue:d347220]
