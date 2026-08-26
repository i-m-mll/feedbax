# Guarantee dependency scan controls

Every channel in `scripts/maintenance/guarantee_dependency_scan.py` carries a positive
control here. The scanner runs this corpus on every invocation and refuses to
report a zero from a channel whose own control came back empty, so "nothing
depends on this row" is never confused with "this channel never worked".

- `positive/` — one known-live dependency per channel. Each must yield at least
  one `dependency`-class record on its channel.
- `negative/` — known false positives. Same spellings, different owners, or a
  guaranteed name used as prose. These must yield no `dependency`-class record.
- `restatement/` — a policy allowlist that restates a surface as data. It is
  evidence of a policy, not of a dependency, and is classified `restatement`.

These files are fixtures. They are not imported, executed, or loaded by anything
except the scanner.
