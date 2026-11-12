"""The authored Train Plan ref carried by the Host admission transaction."""

import json
from hashlib import sha256
from typing import TypedDict

from feedbax.host.protocol import JsonObject


class PlanSuccess(TypedDict):
    request_id: str
    job_id: str


class PlanRowRef(TypedDict):
    request_id: str
    last: PlanSuccess | None
    prior: PlanSuccess | None


class AuthoredTrainPlanRef(TypedDict):
    plan: JsonObject
    rows: dict[str, PlanRowRef]


def plan_run_key(variant: str, step: str, row: str) -> str:
    return json.dumps((variant, step, row), separators=(",", ":"))


def plan_ref_key(project: str, store: str) -> str:
    scope = json.dumps((project, store), separators=(",", ":")).encode()
    return f"studio/train-plan/v1/{sha256(scope).hexdigest()}"


def settled_results(
    row: PlanRowRef, success: PlanSuccess | None
) -> tuple[PlanSuccess | None, PlanSuccess | None]:
    last, prior = row["last"], row["prior"]
    if success is None:
        return last, prior
    if last is not None and last["request_id"] != success["request_id"]:
        return success, last
    return success, prior


def advance_plan(
    plan: JsonObject,
    current: dict[str, str],
    previous: AuthoredTrainPlanRef | None,
    successes: dict[str, PlanSuccess],
) -> AuthoredTrainPlanRef:
    rows: dict[str, PlanRowRef] = {}
    before = previous["rows"] if previous else {}
    for key, request_id in current.items():
        row = before.get(key)
        last, prior = settled_results(row, successes.get(row["request_id"])) if row else (None, None)
        rows[key] = {"request_id": request_id, "last": last, "prior": prior}
    return {"plan": plan, "rows": rows}
