"""Negative control: a downstream package's own result-schema constants.

These are spelled exactly like guaranteed names in the `result-role-binding`
row, and they have nothing to do with this library. They are defined here, not
imported from anywhere, which is the whole distinction.
"""

from __future__ import annotations

RESULT_SCHEMA_ID = "acme.analysis.local_thing"
RESULT_SCHEMA_VERSION = "acme.analysis.local_thing.v1"
REQUIRED_CASE_IDS = ("acme_case",)


def payload() -> dict[str, str]:
    return {"schema_id": RESULT_SCHEMA_ID, "schema_version": RESULT_SCHEMA_VERSION}
