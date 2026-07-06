# Dashboard Retirement

Issue: `c6c8af2` - Retire feedbax/dashboard (superseded by Studio)

Date: 2026-07-06

`feedbax/dashboard/` and `feedbax.bin.dashboard` were removed after review found
the legacy Dash figure-review app was isolated to its own launcher, had no tests,
and was superseded by Studio. The `dashboard` optional dependency extra was
removed as part of the same change.

Follow-up analysis and figure browsing should be implemented in Studio routes or
tabs. The old dashboard's only unique risk was a module-level SQLAlchemy session
shared across Dash callbacks; deleting the app removes that unsafe execution
surface rather than carrying it forward.
