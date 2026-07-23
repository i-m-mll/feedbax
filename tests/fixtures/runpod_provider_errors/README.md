# RunPod create-failure corpus

`create_failure_corpus.json` is a small, hand-curated set of sanitized RunPod
`pod create` failure payloads, each with an adjudicated classification. It
exists so a future provider wording variant is caught by a failing test
instead of silently recurring as a production halt (as happened twice: once
fixed under [issue:88307fb], once under [issue:32d1d73]).

## What is in the corpus, and why

Every entry records the exact `(returncode, stdout, stderr)` a
`runpodctl pod create` failure produced, plus the classification a human
adjudicated for it:

- **`non-retryable`** ("definitive"): the provider's response proves no pod
  was created, so the acquisition loop should reject the datacenter and try
  the next candidate. Covers both production-observed no-capacity wordings,
  one permanent request/configuration error, and one authentication/quota
  failure.
- **`retryable`** ("ambiguous"): the response does not prove that no pod was
  created (malformed body, unrelated error, lost transport, or a structured
  message outside the adjudicated vocabulary). The unconditional
  ambiguous-to-halt backstop in `_classify_create_failure` /
  `_engine_owned_provision` must stop provisioning for every one of these,
  because guessing risks orphaning a billable pod.

`tests/test_runpod_orchestration_driver.py` iterates this file, asserting
both the classification (`_classify_create_failure`) and the acquisition
behavior it drives (region-rejected-and-continue vs. halt).

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
