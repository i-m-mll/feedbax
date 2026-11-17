"""Schedule receipt replays through the bounded Host worker queue."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.receipt_validity import (
    CheckOutcome, ReceiptValidity, ValidityAssessment, ValidityCheckDelivery, ValidityCheckEvidence,
    ValidityCheckScope,
)
from feedbax.host.protocol import AdmissionItem, HostError, Job, Launch, LIVE_STATES, Slot, VALIDITY_CHECK_ENTRYPOINT
from feedbax.persistence.effective_success import PublicationConflict, SuccessView
from feedbax.persistence.receipt_validity import record_validity
from feedbax.validity.project_admission import ObservedReceiptValidity, ProjectValidity


def schedule_check(
    host, *, project: str, project_root: Path, store: Path, observation: ObservedReceiptValidity,
    request_id: str, producer: dict[str, Any] | None, worker: Launch, slot: Slot,
) -> str:
    scope = ValidityCheckScope(
        request_id=request_id, receipt=observation.decision.receipt,
        checks=observation.decision.checks, conditions=observation.conditions,
    )
    authority = {"project": project, "root": str(store), "scope": scope.model_dump(mode="json")}
    identity = sha256(canonical_json_v2_bytes(authority)).hexdigest()
    original = producer.get("launch") if producer is not None else None
    bindings = tuple(
        edge["binding"] for edge in producer.get("edges", ()) if edge.get("binding") is not None
    ) if producer is not None else ()
    delivery = ValidityCheckDelivery(
        project_root=project_root, store_root=store, request_id=request_id, receipt=scope.receipt,
        checks=scope.checks, producer_launch=original, completion_bindings=bindings,
    )
    launch = replace(worker, warm_entrypoint=VALIDITY_CHECK_ENTRYPOINT, project_root=str(project_root),
                     payload=delivery.model_dump(mode="json"))
    job = Job(
        id=identity, project=project, request=f"validity-check:{identity}", launch=launch, slot=slot,
        data={"manifest_root": str(store), "validity_check": scope.model_dump(mode="json")},
    )
    host.admit(
        f"validity-check:{identity}", (AdmissionItem(run_key=request_id, job=job),), record={"project": project},
    )
    return identity


def wait_for_check(host, *, project: str, job_id: str) -> dict[str, Any]:
    snapshot = host.jobs_snapshot(project=project, job_id=job_id)
    (job,) = snapshot["jobs"]
    cursor = snapshot["cursor"]
    while job["state"] in LIVE_STATES:
        page = host.events(cursor, projects=(project,), wait=30)
        cursor = page["cursor"]
        if not page["events"] or any(
            event["job_id"] == job_id and event["kind"] == "terminal" for event in page["events"]
        ):
            (job,) = host.jobs(project=project, job_id=job_id)
    return job


def settle_check(
    view: SuccessView, observer: ProjectValidity, row: dict[str, Any],
) -> ReceiptValidity | None:
    if row["state"] in LIVE_STATES or row["data"].get("manifest_root") != str(view.store.root):
        return None
    scope = ValidityCheckScope.model_validate(row["data"]["validity_check"])
    if view.authored_ref(scope.request_id) != scope.receipt:
        return None
    published = view.read_success(scope.receipt)
    observer = observer.refresh()
    current = observer.inspect(view, published)
    owed = set(current.decision.checks).intersection(scope.checks)
    if current.conditions != scope.conditions or not owed:
        return current.decision

    outcome = row.get("outcome") or {}
    if row["state"] == "succeeded":
        evidence = ValidityCheckEvidence.model_validate(outcome["validity_evidence"])
        if any((getattr(evidence, domain) is not None) != (domain in scope.checks)
               for domain in ("code", "environment")):
            raise HostError("invalid_request", "The check returned a different set of comparisons")
    else:
        detail = (outcome.get("error") or {}).get("message") or "The validity worker did not complete."
        evidence = ValidityCheckEvidence(**{
            domain: CheckOutcome(outcome="unavailable", detail=detail) for domain in scope.checks
        })
    assessment = ValidityAssessment(
        receipt=scope.receipt, conditions=scope.conditions,
        evidence=ValidityCheckEvidence(**{domain: getattr(evidence, domain) for domain in owed}),
        assessed_at=datetime.now(UTC),
    )
    try:
        record_validity(view.store, scope.request_id, assessment)
    except PublicationConflict:
        return None
    return observer.inspect(view, published).decision
