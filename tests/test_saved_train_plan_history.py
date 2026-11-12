from dataclasses import asdict
import sys

import pytest

from feedbax.host import AdmissionItem, HostError, Job, Launch
from feedbax.host.plan_history import advance_plan, plan_ref_key, plan_run_key, settled_results
from feedbax.host.store import Store


def test_partial_admission_and_late_success_preserve_the_pinned_result():
    first = {"request_id": "train:a", "job_id": "first"}
    late = {"request_id": "train:a", "job_id": "late"}
    changed = {"request_id": "train:b", "job_id": "changed"}
    baseline = advance_plan({}, {"row": "train:a", "sibling": "train:s"}, None, {})
    edited = advance_plan(
        {}, {"row": "train:b", "sibling": "train:s"}, baseline, {"train:a": first}
    )
    assert edited["rows"]["row"]["last"] == first
    assert settled_results(edited["rows"]["row"], None) == (first, None)
    readmitted = advance_plan(
        {}, {"row": "train:c", "sibling": "train:s"}, edited, {"train:a": late}
    )
    assert readmitted["rows"]["row"]["last"] == first
    assert settled_results(edited["rows"]["row"], changed) == (changed, first)


def test_same_request_success_keeps_the_preceding_distinct_result():
    first = {"request_id": "train:a", "job_id": "first"}
    changed = {"request_id": "train:b", "job_id": "changed"}
    rerun = {"request_id": "train:b", "job_id": "rerun"}
    baseline = advance_plan({}, {"row": "train:a"}, None, {})
    edited = advance_plan({}, {"row": "train:b"}, baseline, {"train:a": first})
    readmitted = advance_plan({}, {"row": "train:b"}, edited, {"train:b": changed})
    assert settled_results(readmitted["rows"]["row"], rerun) == (rerun, first)


def test_plan_ref_and_attempt_share_one_admission_commit(tmp_path):
    database = tmp_path / "host.sqlite3"
    root = str(tmp_path / "results")
    store = Store(database)
    selected = plan_run_key("default", "selected", "default")
    sibling = plan_run_key("default", "sibling", "default")
    first_id = "train:" + "1" * 64
    sibling_id = "train:" + "2" * 64
    edited_id = "train:" + "3" * 64

    def admit(token, job_id, job_request, requests):
        job = Job(
            id=job_id,
            project="project",
            request=job_request,
            launch=Launch(
                argv=(sys.executable,),
                payload={
                    "project_id": "project",
                    "request_id": job_request,
                    "manifest_root": root,
                    "canonical_request": {"test": "admission metadata"},
                },
            ),
            data={"manifest_root": root},
        )
        return store.admit(token, [asdict(AdmissionItem(run_key=selected, job=job))], {
            "project": "project",
            "command": {"kind": "canonical_train", "store": root},
            "authored_train_plan": {"object": {"token": token}, "requests": requests},
        })

    key = plan_ref_key("project", root)
    first = admit("first", "first", first_id, {selected: first_id, sibling: sibling_id})
    original = store.record_get(key)
    with pytest.raises(HostError, match="does not match the authored Plan"):
        admit("conflict", "conflict", edited_id, {selected: first_id, sibling: edited_id})
    assert store.get("conflict") is None and store.record_get(key) == original
    next_plan = admit("edit", "unused", first_id, {selected: first_id, sibling: edited_id})
    assert next_plan["items"][0]["disposition"] == "attached"
    assert store.record_get(key)["rows"][sibling]["request_id"] == edited_id
    store.db.close()
    store = Store(database)
    try:
        assert admit("first", "unused", first_id, {selected: first_id, sibling: sibling_id}) == first
        assert store.record_get(key)["rows"][sibling]["request_id"] == edited_id
        assert store.get("first")["request"] == first_id
    finally:
        store.db.close()
