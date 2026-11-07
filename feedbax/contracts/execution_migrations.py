"""Document migrations for durable execution state and controller events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from feedbax.contracts.execution_schema import (
    CONTROLLER_EVENT_SCHEMA_ID,
    CONTROLLER_EVENT_SCHEMA_VERSION,
    RUN_SET_STATE_SCHEMA_VERSION,
)


_CONTROLLER_EVENT_TYPES_V1 = frozenset(
    {
        "intent_admitted",
        "intent_rejected",
        "intent_cancelled",
        "intent_superseded",
        "backend_plan_selected",
        "backend_plan_rejected",
        "effect_reservation_created",
        "effect_reservation_authenticated",
        "effect_reservation_expired",
        "effect_reservation_cancelled",
        "effect_reservation_invalidated",
        "external_effect_dispatched",
        "external_effect_observed",
        "external_effect_reconciled",
        "external_effect_abandoned",
        "attempt_started",
        "attempt_heartbeat_observed",
        "attempt_terminal_observed",
        "attempt_state_unknown",
        "output_staged",
        "publication_committed",
        "publication_failed",
        "invocation_satisfied",
        "retry_admitted",
        "invocation_permanently_failed",
        "operator_action_required",
    }
)


class ControllerProtocolError(ValueError):
    """A durable controller document or transition violates the protocol."""


def migrate_controller_event_v1_document(document: Mapping[str, Any]) -> dict[str, Any]:
    migrated = dict(document)
    declared_schema_id = migrated.get("schema_id")
    if declared_schema_id not in {None, CONTROLLER_EVENT_SCHEMA_ID}:
        raise ControllerProtocolError(
            f"unsupported ControllerEvent schema_id {declared_schema_id!r}; "
            f"expected {CONTROLLER_EVENT_SCHEMA_ID!r}"
        )
    event_type = migrated.get("event_type")
    if event_type is not None and event_type not in _CONTROLLER_EVENT_TYPES_V1:
        raise ControllerProtocolError(
            f"ControllerEvent v1 does not define event_type {event_type!r}"
        )
    migrated.setdefault("schema_id", CONTROLLER_EVENT_SCHEMA_ID)
    migrated["schema_version"] = CONTROLLER_EVENT_SCHEMA_VERSION
    return migrated


def migrate_run_set_state_v5_to_v6(payload: dict[str, Any]) -> dict[str, Any]:
    """Make every legacy PID explicitly non-authoritative.

    Version 5 persisted only a numeric PID. The number remains diagnostic data,
    while the new identity field is deliberately empty so recovery cannot adopt
    or signal a process until a current driver proves run-owned identity.
    """
    migrated = dict(payload)
    rows = {}
    for row_id, row_value in dict(migrated.get("rows", {})).items():
        row = dict(row_value)
        row["process_identity"] = None
        rows[row_id] = row
    migrated["rows"] = rows
    migrated["schema_version"] = RUN_SET_STATE_SCHEMA_VERSION
    return migrated
