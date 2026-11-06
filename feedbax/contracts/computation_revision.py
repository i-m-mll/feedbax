"""Validation for declared analysis computation revisions."""

# Bump when Feedbax changes a built-in analysis or shared execution in a way
# that changes scientific results or materialized outputs for the same inputs.
ANALYSIS_ENGINE_REVISION = "2"


def validate_computation_revision(revision: str | None) -> str | None:
    """Return a declared revision unchanged, or ``None`` when undeclared."""
    if revision is not None and (type(revision) is not str or not revision.strip()):
        raise ValueError("computation_revision must be a nonempty string or None")
    return revision
