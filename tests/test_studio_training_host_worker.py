"""Numerical continuation and live metrics through the real Studio host worker."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
import pytest

from feedbax.contracts.manifest import canonical_manifest_path
from feedbax.contracts.spec_storage import canonical_training_run_spec_sha256
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.training_requests import lower_studio_training_runs
from feedbax.training.checkpoint_custody import (
    authenticate_published_checkpoint_custody,
    load_checkpoint_custody_documents,
    require_checkpoint_semantic_state_equal,
)
from feedbax.training.studio_executor import StudioUpdateKernel, studio_training_key
from feedbax.web.worker import execution
from web.e2e.studio.fixtures.reaching_project import reaching_project


_STATE = ("model", "optimizer", "prng", "batch_counter")
_SCIENTIFIC_METRICS = ("train_loss", "loss_terms", "grad_norm")
_PROCESS_ENV = (
    "PATH", "TMPDIR", "LANG", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
_SOURCE = Path(__file__).absolute().parents[1]


def _array_digests(tree):
    return [
        hashlib.sha256(array.tobytes()).hexdigest()
        for array in map(np.asarray, filter(eqx.is_array, jax.tree.leaves(tree)))
    ]


def _readout(graph):
    return graph.nodes["readout"].layer.weight


def _observe_worker(audit_path: Path, stop_after: int, distinct_preparation: bool) -> int:
    from feedbax import training_host_worker
    from feedbax.training import executor

    compile_run = execution.compile_training_run
    update = StudioUpdateKernel.reported_update
    execute = executor.execute_training_run_spec

    def prepare(*args, **kwargs):
        compiled = compile_run(*args, **kwargs)
        if distinct_preparation:
            # Change only the resume worker's template before AOT compilation.
            compiled.graph = eqx.tree_at(_readout, compiled.graph, _readout(compiled.graph) + 0.25)
        return compiled

    def observe(kernel, slots):
        compiled = kernel._compiled
        model = slots["model"]
        result = update(kernel, slots)
        assert compiled.trainable_nodes == ("cell",)
        assert np.any(np.asarray(_readout(model)) != 0)
        assert _array_digests(model.nodes["cell"]) != _array_digests(result["model"].nodes["cell"])
        evidence = {
            "pid": os.getpid(),
            "update": int(result["batch_counter"]),
            "before": {name: _array_digests(slots[name]) for name in _STATE},
            "after": {name: _array_digests(result[name]) for name in _STATE},
            "prepared_readout": _array_digests(_readout(compiled.graph)),
            "input_readout": _array_digests(_readout(model)),
            "output_readout": _array_digests(_readout(result["model"])),
            "metrics": {
                name: result[name] for name in (*_SCIENTIFIC_METRICS, "step_time_ms")
            },
        }
        with audit_path.open("a") as stream:
            stream.write(json.dumps(evidence, allow_nan=False) + "\n")
        return result

    def stop_at_commit(*args, **kwargs):
        callback = kwargs["committed_update_callback"]

        def observe_commit(coordinate, completed, slots):
            callback(coordinate, completed, slots)
            if completed == stop_after:
                Path(os.environ["FEEDBAX_HOST_STOP_FILE"]).touch()

        return execute(*args, **{**kwargs, "committed_update_callback": observe_commit})

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(execution, "compile_training_run", prepare)
        patch.setattr(StudioUpdateKernel, "reported_update", observe)
        patch.setattr(executor, "execute_training_run_spec", stop_at_commit)
        # Force an immediate first emission and one final partial interval.
        patch.setattr(training_host_worker._Events, "elapsed", lambda _events: 0.1)
        return training_host_worker.main([])


def _authored_run():
    project = reaching_project(batches=8)
    project.graph.nodes["readout"].params["trainable"] = False
    row, = lower_studio_training_runs(
        project.workspace, project.graph, stage_id="stage:train",
        registry_bundle=new_application_registry_bundle(),
    )
    return row


def _launch(root, job, spec, *, stop_after=0, resume=None, source=_SOURCE, fail=False):
    control = root / job
    control.mkdir()
    log = control / "events.jsonl"
    audit = control / "updates.jsonl"
    launch_input = control / "input.json"
    launch_input.write_text(json.dumps({
        "spec": spec.model_dump(mode="json"),
        "manifest_root": str(root / "runs"),
        "project_id": "frozen-readout",
        "request_id": canonical_training_run_spec_sha256(spec),
        "resume": resume,
    }))
    env = {name: os.environ[name] for name in _PROCESS_ENV if name in os.environ}
    env.update(
        PYTHONPATH=str(source),
        FEEDBAX_HOST_JOB_ID=job,
        FEEDBAX_HOST_INPUT=str(launch_input),
        FEEDBAX_HOST_EVENT_LOG=str(log),
        FEEDBAX_HOST_STOP_FILE=str(control / "stop"),
        FEEDBAX_HOST_SCRATCH=str(control),
        JAX_PLATFORMS="cpu",
        JAX_COMPILATION_CACHE_DIR=str(root / "jax"),
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
        JAX_COMPILATION_CACHE_MAX_SIZE=str(32 * 1024 * 1024),
    )
    child = subprocess.run(
        [sys.executable, __file__, str(audit), str(stop_after), str(int(resume is not None))],
        cwd=source, env=env, capture_output=True, text=True, timeout=20, check=False,
    )
    events = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    assert child.returncode == int(fail), (job, child.stdout, child.stderr, events)
    assert events[-1]["kind"] == "result"
    result = events[-1]["data"]
    if fail:
        assert result["outcome"] == "failed"
        return result
    assert result["outcome"] == ("stopped" if stop_after else "succeeded")
    binding, = [event["data"] for event in events if event["kind"] == "binding"]
    assert binding["resume"] == resume
    checkpoint = result["checkpoint"]
    custody = authenticate_published_checkpoint_custody(checkpoint["checkpoint_root"])
    pointer = load_checkpoint_custody_documents(checkpoint["checkpoint_root"]).latest_pointer.document
    assert pointer.model_dump(mode="json") == checkpoint["pointer"]
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    receipt = canonical_manifest_path(
        "TrainingRunManifest", result["receipt"]["manifest_id"], root=root / "runs",
    )
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == result["receipt"]["manifest_sha256"]
    return checkpoint, custody.manifest, rows, events, receipt


def _require_reductions(reductions, samples):
    assert set(reductions) == set(samples[0])
    for name, reduction in reductions.items():
        values = [sample[name] for sample in samples]
        if isinstance(values[0], dict):
            _require_reductions(reduction, values)
        else:
            assert reduction == {
                "count": len(values),
                "mean": pytest.approx(math.fsum(values) / len(values), rel=1e-14),
                "min": min(values),
                "max": max(values),
                "last": values[-1],
            }


def _require_progress(events, rows, start):
    initial, first, final = [event["data"] for event in events if event["kind"] == "progress"]
    assert initial["completed_updates"] == start
    assert initial["metrics"] == {}
    assert "metric_interval" not in initial
    for event, updates in ((first, rows[:1]), (final, rows[1:])):
        interval = event["metric_interval"]
        assert interval["start_exclusive"] == updates[0]["update"] - 1
        assert interval["end_inclusive"] == event["completed_updates"] == updates[-1]["update"]
        assert event["metrics"] == updates[-1]["metrics"]
        _require_reductions(interval["metrics"], [row["metrics"] for row in updates])


def test_reaching_continues_checkpoint_frozen_readout_across_two_stops(tmp_path):
    from feedbax.contracts.studio_training import STUDIO_INDEPENDENT_KEY_SCHEDULE

    row = _authored_run()
    assert row.spec.method_payload.payload["key_schedule"] == STUDIO_INDEPENDENT_KEY_SCHEDULE
    legacy = row.spec.model_copy(deep=True)
    del legacy.method_payload.payload["key_schedule"]
    assert canonical_training_run_spec_sha256(legacy) != row.request_id
    assert len({
        np.asarray(studio_training_key(0, stream)).tobytes()
        for stream in ("model", "update", "observation")
    } | {np.asarray(jr.PRNGKey(0)).tobytes()}) == 4

    _, reference, whole, whole_events, _ = _launch(tmp_path, "whole", row.spec)
    first, first_manifest, prefix, prefix_events, first_receipt = _launch(
        tmp_path, "first", row.spec, stop_after=3,
    )
    first_bytes = first_receipt.read_bytes()
    second, second_manifest, middle, middle_events, second_receipt = _launch(
        tmp_path, "second", row.spec, resume=first, stop_after=6,
    )
    second_bytes = second_receipt.read_bytes()
    _, last_manifest, suffix, suffix_events, _ = _launch(tmp_path, "last", row.spec, resume=second)

    assert [entry["update"] for entry in whole] == list(range(1, 9))
    assert whole[0]["before"]["prng"] == _array_digests(studio_training_key(0, "update"))
    assert [entry["update"] for entry in prefix] == [1, 2, 3]
    assert [entry["update"] for entry in middle] == [4, 5, 6]
    assert [entry["update"] for entry in suffix] == [7, 8]
    assert len({part[0]["pid"] for part in (whole, prefix, middle, suffix)}) == 4
    for expected, actual in zip(whole, prefix + middle + suffix, strict=True):
        assert actual["input_readout"] == expected["prepared_readout"]
        assert actual["output_readout"] == expected["prepared_readout"]
        assert actual["before"] == expected["before"]
        assert actual["after"] == expected["after"]
        assert actual["metrics"]["grad_norm"] > 0
        for name in _SCIENTIFIC_METRICS:
            assert actual["metrics"][name] == expected["metrics"][name]
    for updates in (middle, suffix):
        assert all(entry["prepared_readout"] != entry["input_readout"] for entry in updates)

    assert second_manifest.parent_lineage[0].transaction_id == first_manifest.transaction_id
    assert last_manifest.parent_lineage[0].transaction_id == second_manifest.transaction_id
    for name in _STATE:
        require_checkpoint_semantic_state_equal(reference, last_manifest, name)
    assert first_receipt.read_bytes() == first_bytes
    assert second_receipt.read_bytes() == second_bytes
    for events, updates, start in (
        (whole_events, whole, 0), (prefix_events, prefix, 0),
        (middle_events, middle, 3), (suffix_events, suffix, 6),
    ):
        _require_progress(events, updates, start)

    rejected = _launch(tmp_path, "wrong-binding", legacy, resume=first, fail=True)
    assert rejected["error"]["code"] == "incompatible_checkpoint"
    rejected = _launch(tmp_path, "old-seed-zero-start", legacy, fail=True)
    assert rejected["error"]["code"] == "invalid_request"
    assert "Create a new Studio request" in rejected["error"]["message"]


if __name__ == "__main__":
    raise SystemExit(_observe_worker(Path(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3]))))
