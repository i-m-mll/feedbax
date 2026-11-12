"""Real Jobs fence and private typed reaching evaluation on one saved producer."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.dataset import DatasetProductRef
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.job_completion import verify_bound_training_completion
from feedbax.jobs import JobService
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_dataset
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_train_admission import _saved_step
from tests.test_local_job_host import host as host_fixture
from tests.test_local_job_host import job_row, restart, wait_until
from tests.test_studio_training_host_worker import _authored_run

host = host_fixture


@contextmanager
def _damaged_prng(checkpoint):
    directory = (
        Path(checkpoint["checkpoint_root"]) / checkpoint["pointer"]["manifest_relative_path"]
    ).parent
    manifest = json.loads((directory / "manifest.json").read_bytes())
    (prng,) = (slot for slot in manifest["slots"] if slot["slot"] == "prng")
    path = directory / prng["relative_path"]
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    try:
        yield
    finally:
        path.write_bytes(original)


def test_saved_train_finite_selection_and_private_evaluation(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    jobs = JobService(host, new_application_registry_bundle())
    folders = ProjectFolderService(tmp_path / "projects")
    snapshot = folders.new().snapshot
    spec = _authored_run(batches=12, sampled_hold=True).spec
    payload = TrainingRequestSpec(
        graph=spec.graph.inline,
        task=spec.task,
        task_binding_spec=spec.task_binding_spec,
        objective=spec.objective,
        method_ref=spec.method_ref,
        method_payload=spec.method_payload,
        training_config=spec.training_config.model_copy(update={"snapshot_interval": 2}),
    ).model_dump(mode="json", exclude_none=True)
    snapshot = _saved_step(
        folders, snapshot, "reaching", {"kind": "train", "base": {"inline": payload}}
    )
    reviewed = folders.read_plan(snapshot.project_id, revision=snapshot.revision)
    key = RunKey("default", "reaching", "default")
    host.policy(compute=1)

    (tmp_path / "finite_gate.py").write_text("""
import json
import os
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch
from feedbax.training import executor
from feedbax.training_host_worker import main as train

def main(argv):
    root = Path(os.environ["FINITE_PROOF_ROOT"])
    source = json.loads(Path(os.environ["FEEDBAX_HOST_INPUT"]).read_bytes())
    stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
    execute = executor.execute_training_run_spec

    def wait(path):
        deadline = monotonic() + 22
        while not path.exists():
            if monotonic() > deadline:
                raise TimeoutError(f"Jobs did not release {path.name}")
            sleep(.002)

    def wrapped(*args, **kwargs):
        commit = kwargs["committed_update_callback"]
        published = kwargs["checkpoint_published"]

        def on_commit(coordinate, completed, slots):
            commit(coordinate, completed, slots)
            if source.get("resume") is None and completed == 5:
                (root / "before-stop").touch()
                wait(stop)

        def on_publish(write):
            published(write)
            if source.get("resume") and write.latest_pointer.completed_training_batches == 6:
                (root / "resumed-interior").touch()
                wait(root / "finish-train")

        kwargs.update(committed_update_callback=on_commit, checkpoint_published=on_publish)
        return execute(*args, **kwargs)

    with patch.object(executor, "execute_training_run_spec", wrapped):
        return train(argv)
""")
    jobs._training_worker = replace(
        jobs._training_worker,
        env=jobs._training_worker.env
        | {
            "JAX_PLATFORMS": "cpu",
            "FINITE_PROOF_ROOT": str(tmp_path),
            "PYTHONPATH": f"{tmp_path}:{Path.cwd()}",
        },
        warm_entrypoint="finite_gate:main",
    )
    submitted = jobs.admit_plan(
        folders,
        reviewed,
        run_keys=(key,),
        token="finite-producer",
        checkpoint_catalog=(2, 4, 6, 9, 12),
    )
    producer = submitted.items[0].job_id
    patch_mass = OverridePatch(path="nodes.plant.params.mass", value=2.0)

    def admit(source, coordinates, *, token):
        return jobs.admit_private_checkpoint_evaluation(
            source,
            coordinates,
            evaluate_update=coordinates[0],
            variation=patch_mass,
            token=token,
        )

    def terminal(job_id, expected):
        def ready():
            row = job_row(host, job_id)
            return (
                row
                if row and row["state"] in {"failed", "succeeded", "stopped", "blocked"}
                else None
            )

        row = wait_until(ready, timeout=18)
        assert row["state"] == expected, row.get("outcome")
        return row

    consumer = admit(producer, (6, 2), token="finite-private")
    open_path = admit(producer, (2, 9), token="finite-unclosed-lineage")
    pinned_then_corrupt = admit(producer, (4, 6), token="finite-corrupt-at-delivery")
    wait_until(lambda: (tmp_path / "before-stop").exists(), timeout=16)
    queued = job_row(host, consumer, "queued")
    assert queued["delivery"] == 0 and not queued["edges"][0]["satisfied"]
    jobs.stop([producer], "checkpoint")
    stopped = terminal(producer, "stopped")
    assert stopped["outcome"]["checkpoint"]["completed_updates"] == 5
    restart(host)
    assert admit(producer, (6, 2), token="finite-private") == consumer
    assert not job_row(host, consumer, "queued")["edges"][0]["satisfied"]

    resumed = jobs.resume(producer, token="finite-resume")
    wait_until(lambda: (tmp_path / "resumed-interior").exists(), timeout=16)
    missing = admit(resumed.job_id, (6, 9), token="finite-missing")
    corrupt = admit(resumed.job_id, (6, 12), token="finite-corrupt")
    assert all(job_row(host, candidate, "queued") for candidate in (missing, corrupt))

    def pinned():
        row = job_row(host, pinned_then_corrupt, "queued")
        return row and row["edges"][0]["binding"]

    pinned_product = wait_until(pinned, timeout=16)["checkpoint_selection"]["members"][0]
    assert pinned_product["checkpoint"]["completed_updates"] == 4
    with _damaged_prng(pinned_product["checkpoint"]):
        host.policy(compute=2)
        evaluated = terminal(consumer, "succeeded")
        delivery_refused = terminal(pinned_then_corrupt, "blocked")
        assert delivery_refused["delivery"] == 0
        assert delivery_refused["outcome"]["error"]["code"] == "dependency_corrupt"
    assert job_row(host, resumed.job_id, "running") is not None
    bound = evaluated["edges"][0]["binding"]["checkpoint_selection"]
    assert bound["requested"] == [6, 2] and bound["lineage_tip"] == resumed.job_id
    assert bound["lineage"] == [producer, resumed.job_id]
    assert bound["members"][0]["publication_cursor"] > bound["members"][1]["publication_cursor"]
    assert [item["producer_job_id"] for item in bound["members"]] == [resumed.job_id, producer]

    provider = ImmutableArtifactBlobProvider(reviewed.store_path)
    receipt_ref = ContentObjectRef.model_validate(evaluated["outcome"]["private_evaluation"])
    receipt = json.loads(provider.get_object(receipt_ref))
    assert receipt["selected"] == [member["checkpoint"] for member in bound["members"]]
    assert receipt["evaluated"] == bound["members"][0]["checkpoint"]
    assert receipt["restored_frozen_and_learned"] and receipt["paired_target_draws"] >= 2
    assert receipt["source_mass"] == 1 and receipt["variant_mass"] == 2
    assert receipt["dimensions"] == {
        "checkpoint": 1,
        "condition": 2,
        "trial": 3,
        "state_time": 101,
        "transition": 100,
        "xy": 2,
    }
    product = DatasetProductRef.model_validate(receipt["dataset"])
    reads = []
    original = ArrayContainerReader.read_tensor

    def track(reader, name):
        reads.append(name)
        return original(reader, name)

    with (
        patch.object(ArrayContainerReader, "read_tensor", track),
        open_dataset(
            provider.object_path(product.object_ref),
            expected_descriptor_sha256=product.descriptor_sha256,
        ) as dataset,
    ):
        np.testing.assert_array_equal(
            dataset.data("position", (0, 1, 0, slice(0, 3))),
            receipt["selected_position"],
        )
    assert reads == ["position"]

    with _damaged_prng(bound["members"][0]["checkpoint"]):
        (tmp_path / "finish-train").touch()
        completed = terminal(resumed.job_id, "succeeded")
        verify_bound_training_completion(completed["outcome"]["completion"])
        absent = terminal(missing, "blocked")
        bad = terminal(corrupt, "blocked")
        assert absent["outcome"]["error"]["code"] == "input_unavailable"
        assert "9" in absent["outcome"]["error"]["message"]
        assert bad["outcome"]["error"]["code"] == "dependency_corrupt"
        assert "Completion file changed" in bad["outcome"]["error"]["message"]
        assert absent["delivery"] == bad["delivery"] == 0
        assert job_row(host, open_path, "queued")["edges"][0]["binding"] is None

    late_log = tmp_path / "host" / "deliveries" / resumed.job_id / "1" / "events.jsonl"
    with late_log.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "publication",
                    "data": {
                        "key": "checkpoint:9",
                        "checkpoint": bound["members"][0]["checkpoint"],
                    },
                }
            )
            + "\n"
        )
    restart(host)
    assert admit(producer, (6, 2), token="finite-private") == consumer
    assert (
        terminal(consumer, "succeeded")["edges"][0]["binding"] == evaluated["edges"][0]["binding"]
    )
    events = host.events(projects=(reviewed.project_id,))["events"]
    assert not any(
        event["job_id"] == resumed.job_id
        and event["kind"] == "publication"
        and event["data"]["key"] == "checkpoint:9"
        for event in events
    )
    binds = [
        event
        for event in events
        if event["job_id"] == consumer and event["kind"] == "dependency_bound"
    ]
    assert len(binds) == 1
    (tmp_path / "finite-evaluation-evidence.json").write_text(
        json.dumps(
            {
                "producer": producer,
                "resume": resumed.job_id,
                "selection": bound,
                "one_bind_cursor": binds[0]["cursor"],
                "receipt": receipt,
                "receipt_ref": receipt_ref.model_dump(mode="json"),
                "missing": absent["outcome"]["error"],
                "invalid": bad["outcome"]["error"],
                "invalid_before_delivery": delivery_refused["outcome"]["error"],
                "resumable_ancestor_still_pending": open_path,
                "selective_reads": reads,
            },
            indent=2,
        )
    )
