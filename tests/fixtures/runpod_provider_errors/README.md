# RunPod create-failure corpus

`create_failure_corpus.json` contains curated, sanitized RunPod `pod create` failure payloads. The tests preserve the classifications established after provider wording caused production halts in [issue:88307fb] and [issue:32d1d73].

## What is in the corpus, and why

Every entry records the `(returncode, stdout, stderr)` from a `runpodctl pod create` failure and the resulting classification.

- **`retryable-rejection`** applies to a definitive capacity rejection. The engine can try another authorized candidate within its budget.
- **`terminal-rejection`** applies to a definitive request or authorization error. The engine records the failed intent and stops.
- **`ambiguous`** applies when the response leaves pod ownership uncertain. The engine reconciles the persisted acquisition intent and stops if it cannot prove cleanup.

`tests/test_runpod_orchestration_driver.py` checks the classifications and their effect on acquisition.

## Curation workflow for a new entry

This corpus grows only from real incidents, never from synthetic guesses at
the provider's error taxonomy:

1. **Capture the raw failure durably.** When a launch halts or mis-classifies
   a create failure, keep the original `runpodctl` stdout/stderr (and
   returncode) in incident custody (e.g. attached to the debugging issue)
   before touching any code.
2. **Human-adjudicate.** Decide, from the raw payload and provider
   documentation/behavior, whether the failure is genuinely definitive
   (no pod was created) or must stay ambiguous. This is a judgment call, not
   something the corpus or classifier infers automatically.
3. **Sanitize and minimize.** Strip anything account- or credential-specific,
   trim to the smallest reproducing payload, and give the entry a `name` and
   `description` explaining what real event it represents.
4. **Add it to `create_failure_corpus.json`** with its adjudicated
   `expected_classification` and `expected_behavior`, and let the existing
   parametrized tests exercise it.

There is deliberately no harvesting script, no classifier DSL, and no attempt
to enumerate RunPod's full error surface here — only the small set of
payloads a human has actually seen and adjudicated.
