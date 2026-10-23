"""Internal identity rules for the Studio worker HTTP transport."""

from __future__ import annotations

import re


_WORKER_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_WORKER_JOB_ID_ERROR = (
    "worker job_id must be a path-safe transport identifier containing only "
    "letters, digits, '.', '_', or '-', and must not be '.' or '..'"
)


def require_worker_job_id(value: object) -> str:
    """Return an exact worker transport ID or raise ``ValueError``.

    The worker HTTP transport is intentionally narrower than the durable
    ``RunRowSpec`` contract: the canonical grammar admits ``.`` and ``..``,
    while the transport rejects them because it uses row IDs in local paths.
    Values are never stripped or otherwise normalized.
    """
    if (
        not isinstance(value, str)
        or not value
        or not _WORKER_JOB_ID_RE.fullmatch(value)
        or value in {".", ".."}
    ):
        raise ValueError(_WORKER_JOB_ID_ERROR)
    return value
