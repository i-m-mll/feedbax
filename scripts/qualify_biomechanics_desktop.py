"""Qualify retained biomechanics on source and on an isolated MJX CPU release."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import chdir, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

CASES = ("planar-rigid", "spatial-mixed")
STEPS = 4


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def host_job(fixture, job_id):
    from feedbax.host import HostClient

    snapshot = HostClient(fixture.paths.host).jobs_snapshot(
        job_id=job_id, start_host=False,
    )
    (job,) = snapshot["jobs"]
    require(job["id"] == job_id
            and job["request"].startswith("retained-biomechanics:")
            and job["data"]["runtime"] == str(fixture.release.resource("python"))
            and job["data"]["release"] == fixture.release.digest,
            "job does not belong to this installed runtime")
    return job, snapshot["cursor"]


def author():
    # These existing physics fixtures are source inputs. Installed commands consume only JSON.
    from scripts.biomechanics_fixture import planar_interior
    from tests.test_compliant_mjx import _document
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

    planar = planar_interior(10, law_profile="rigid")
    with TemporaryDirectory(prefix="fbx-authored-mjcf-") as temporary:
        provider = ImmutableArtifactBlobProvider(temporary)
        spatial = _document(provider, 0.001).graph.subgraphs["physics"]
        xml = provider.get_object(spatial.mujoco_asset.asset).decode("utf-8")
    cases = {}
    for name, interior, width, target, bias in (
        (CASES[0], planar, 6, [0.60, 0.05], -2.9),
        (CASES[1], spatial, 2, [0.16, 0.05, -0.02], -0.8),
    ):
        data = interior.model_dump(mode="json", exclude_none=True)
        for node in data["nodes"].values():
            if node["type"] == "feedbax.SensorOutput" and node["params"]["quantity"] == "effector":
                node["params"] = {"port_name": "position", "quantity": "position"}
        if name == CASES[0]:
            del data["nodes"]["state"]
        cases[name] = {
            "interior": data, "width": width, "target": target, "bias": bias,
            "control_port": "force" if name == CASES[0] else "excitation",
        }
    cases[CASES[1]]["mjcf"] = xml
    return cases


def request_for(name, case, *, batches, checkpoint_interval):
    from feedbax.compiler.normalization import stamp_component_versions
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec
    from feedbax.contracts.request_specs import TrainingRequestSpec
    from feedbax.contracts.studio_training import (
        STUDIO_GRAPH_METHOD,
        STUDIO_GRAPH_METHOD_PAYLOAD_ID,
        STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
        StudioGraphMethodPayload,
    )
    control = case["control_port"]
    graph = GraphSpec(
        nodes={
            "controller": {
                "type": "feedbax.Linear", "input_ports": ["input"], "output_ports": ["output"],
                "params": {
                    "input_size": 1, "output_size": case["width"], "activation": "sigmoid",
                    "initialization": "zeros", "bias_init": case["bias"],
                },
            },
            "physics": {
                "type": "feedbax.AcausalSystem", "input_ports": [control], "output_ports": ["position"],
            },
        },
        wires=[{
            "source_node": "controller", "source_port": "output",
            "target_node": "physics", "target_port": control,
        }],
        subgraphs={"physics": case["interior"]},
        output_ports=["position"],
        output_bindings={"position": ("physics", "position")},
    )
    binding = StudioTaskBindingSpec.model_validate({
        "exposed_data": [{
            "id": "drive", "label": "Controller drive", "kind": "signal", "role": "model_input",
            "path": "inputs.drive", "bindable": True, "expected_shape": ["time", 1],
            "value_spec": {"mode": "constant", "value": [1.0], "dtype": "float32", "shape": ["time", 1]},
        }],
        "bindings": [{
            "id": "task:drive->controller:input", "source_data_id": "drive",
            "target_node_id": "controller", "target_port": "input", "role": "model_input",
        }],
    })
    optimizer = {"type": "adam", "params": {"learning_rate": 0.005}}
    loss = {
        "type": "TargetStateLoss", "label": "Endpoint position", "selector": "graph_output:position",
        "target_value": case["target"], "weight": 1.0, "norm": "squared_l2",
    }
    task = {"type": "SimpleReaches", "params": {"n_steps": STEPS}}
    payload = StudioGraphMethodPayload(optimizer=optimizer, target_total=batches)
    return TrainingRequestSpec(
        graph=stamp_component_versions(graph, ComponentRegistry()),
        task=task, task_binding_spec=binding,
        training_config={
            "n_batches": batches, "batch_size": 1, "n_reach_steps": STEPS,
            "learning_rate": optimizer["params"]["learning_rate"],
            "snapshot_interval": min(batches, checkpoint_interval),
        },
        objective={"loss": loss}, method_ref=STUDIO_GRAPH_METHOD,
        method_payload={
            "schema_id": STUDIO_GRAPH_METHOD_PAYLOAD_ID,
            "schema_version": STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
            "payload": payload.model_dump(mode="json"),
        },
    )


class Fixture:
    def __init__(self, contents, fixture_root):
        import feedbax
        from feedbax.desktop.release import AppPaths, Release

        self.release = Release.load(contents)
        require(self.release.fixture_root == fixture_root.resolve(strict=True), "not the requested fixture release")
        require(Path(sys.executable).resolve(strict=True) == self.release.resource("python")
                and Path(feedbax.__file__).resolve().is_relative_to(self.release.contents),
                "qualification must use Feedbax from the exact bundled interpreter")
        self.paths = AppPaths.resolve(instance_root=self.release.fixture_root)
        self.paths.configure_studio()
        os.environ["FEEDBAX_DESKTOP_RELEASE_ROOT"] = str(self.release.contents)
        os.environ.pop("PYTHONPATH", None)


class Studio(Fixture):
    def __init__(self, contents, fixture_root):
        from feedbax.desktop.control import call

        super().__init__(contents, fixture_root)
        health = call(self.paths, "health")
        runtime = health["runtime"]
        physics = runtime["physics"]
        require(
            health["status"] == "ready" and health["release"] == self.release.digest
            and Path(health["installed_path"]) == self.release.contents
            and Path(runtime["python"]) == self.release.resource("python")
            and runtime["source_commit"] == self.release.document["source_commit"]
            and runtime["profile"] == "mjx-cpu" and runtime["backend"] == "cpu"
            and (physics["provider"], physics["implementation"], physics["device"]) == ("mujoco.mjx", "jax", "cpu"),
            "the exact installed MJX CPU service is not ready",
        )
        self.health = health

def prepare(request, name, case, batches, checkpoint_interval):
    if "mjcf" in case:
        asset = request("POST", "/api/graphs/assets/mujoco", case["mjcf"].encode())
        require(asset == case["interior"]["mujoco_asset"]["asset"], "imported MJCF identity changed")
    training = request_for(name, case, batches=batches, checkpoint_interval=checkpoint_interval)
    snapshot = request("POST", "/api/projects")["snapshot"]
    project_id = snapshot["project_id"]
    path = f"/api/projects/{project_id}"
    saved = request("POST", path + "/files", {
        "revision": snapshot["revision"],
        "file": {"kind": "step", "id": "train", "step": {
            "kind": "train", "base": {"inline": training.model_dump(mode="json", exclude_none=True)},
        }},
    })["snapshot"]
    verify_project(request, project_id, case["interior"])
    return {"project_id": project_id, "revision": saved["revision"]}


def verify_project(request, project_id, interior):
    from feedbax.compiler.normalization import stamp_component_versions
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.acausal import AcausalGraphSpec

    project = request("GET", f"/api/projects/{project_id}/open")
    graph = project["snapshot"]["project"]["steps"]["train"]["base"]["inline"]["graph"]
    actual = graph["subgraphs"]["physics"]
    expected = stamp_component_versions(AcausalGraphSpec.model_validate(interior), ComponentRegistry())
    require(AcausalGraphSpec.model_validate(actual) == expected,
            "the persisted authored mechanics changed")
    return project


def gui_pids(studio):
    app = studio.release.contents.parent
    script = """
ObjC.import("AppKit");
function run(args) {
  const list = $.NSWorkspace.sharedWorkspace.runningApplications;
  const found = [];
  for (let i = 0; i < list.count; i++) {
    const item = list.objectAtIndex(i);
    if (ObjC.unwrap(item.bundleIdentifier) === args[0] &&
        ObjC.unwrap(item.bundleURL.path) === args[1]) {
      found.push(item.processIdentifier);
    }
  }
  return JSON.stringify(found);
}
"""
    result = subprocess.run(
        ["/usr/bin/osascript", "-l", "JavaScript", "-e", script, studio.release.document["bundle_id"], str(app)],
        capture_output=True, text=True, check=True, timeout=5,
    )
    return json.loads(result.stdout)


def snapshot(studio, job_id, status, after, *, after_update=-1, gui=None):
    deadline = time.monotonic() + 42
    wanted = "succeeded" if status == "completed" else status
    while time.monotonic() < deadline:
        record, cursor = host_job(studio, job_id)
        require(record["state"] not in {"failed", "interrupted", "blocked"}, f"training failed: {record}")
        progress = record["progress"]
        if record["state"] == wanted and cursor > after and (
            wanted != "running" or progress and progress["completed_updates"] > max(0, after_update)
        ):
            break
        time.sleep(0.2)
    else:
        raise RuntimeError(f"{job_id} did not reach {status} after event {after}: {record}")
    job = {
        "job_id": job_id, "project": record["project"], "status": record["state"],
        "progress": progress, "event_sequence": cursor,
    }
    result = {"job": job, "service": studio.health}
    if status == "running":
        pid = record["worker_pid"]
        require(type(pid) is int and pid > 1 and os.getpgid(pid) == pid, "training owner disappeared")
        result["worker"] = {"pid": pid, "group": pid}
    if gui is not None:
        result["gui_pids"] = gui_pids(studio)
        require(bool(result["gui_pids"]) == (gui == "present"), f"expected GUI {gui}")
    return result


def same_job(before, quit_sample, reopened):
    states = (before, quit_sample, reopened)
    require(before["gui_pids"] and not quit_sample["gui_pids"] and reopened["gui_pids"],
            "expected open, quit, reopen GUI sequence")
    for data in states:
        require(data["job"]["status"] == "running" and data["job"]["job_id"] == before["job"]["job_id"],
                "the original job did not stay live")
        require(data["worker"] == before["worker"], "worker identity changed")
        require(all(data["service"][key] == before["service"][key] for key in ("pid", "generation", "release")),
                "service identity changed")
    events = [data["job"]["event_sequence"] for data in states]
    updates = [data["job"]["progress"]["completed_updates"] for data in states]
    require(events[0] < events[1] < events[2] and updates[0] < updates[1] < updates[2],
            f"job did not progress through Quit/reopen: {events}, {updates}")
    return {"job_id": before["job"]["job_id"], "events": events, "updates": updates, "worker": before["worker"],
            "generation": before["service"]["generation"]}


def inspect_model(model, case):
    import jax
    import numpy as np

    physics = model.nodes["physics"]
    weights = np.asarray(model.nodes["controller"].layer.weight)
    key = jax.random.key(17)
    state = physics.init_state(key=key)
    excitation = 1 / (1 + np.exp(-np.asarray(model.nodes["controller"].layer.bias)))
    outputs, state = physics({case["control_port"]: excitation}, state, key=key)
    position = np.asarray(outputs["position"])
    plant = physics.state_view(state).plant
    qpos = np.asarray(plant.skeleton.qpos)
    dimension = len(case["target"])
    require(position.shape == (dimension,) and np.isfinite(position).all(), "invalid Cartesian observation")
    require(qpos.shape == ((4,) if dimension == 3 else (2,)) and np.isfinite(qpos).all(),
            "unexpected native configuration")
    if dimension == 3:
        require(abs(float(np.linalg.norm(qpos)) - 1) < 2e-6, "spatial quaternion lost normalization")
    binding = case["interior"].get("mujoco_asset")
    expected_asset = binding["asset"]["object_id"] if binding else None
    require(physics.descriptor.asset_identity == expected_asset, "native model asset identity changed")
    muscle_widths = [int(np.size(muscle)) for muscle in plant.muscles]
    expected_widths = [2, 1] if dimension == 3 else [1] * case["width"]
    require(muscle_widths == expected_widths, f"unexpected muscle state widths: {muscle_widths}")
    require(np.isfinite(weights).all() and np.linalg.norm(weights) > 0, "controller never updated")
    return {"model": physics.descriptor.identity, "asset": physics.descriptor.asset_identity,
            "position": position.tolist(), "qpos": qpos.tolist(),
            "muscle_state_widths": muscle_widths,
            "controller_weights": weights.tolist()}


def attempt_root(workspace, job_id):
    require(re.fullmatch(r"[0-9a-f]{32}", job_id) is not None, "invalid qualification job id")
    root = workspace / "retained-biomechanics" / job_id
    require(root.resolve() == root, "qualification custody leaves its expected workspace")
    return root


def read_terminal_checkpoint(root, job_id, *, stopped):
    from feedbax.contracts.base import ParentRef
    from feedbax.contracts.checkpoints import CheckpointLatestPointer
    from feedbax.contracts.manifest import canonical_manifest_path, load_manifest, training_run_manifest_id
    from feedbax.training.checkpoint_custody import authenticate_checkpoint_custody_ref

    manifest_id = training_run_manifest_id(job_id)
    run = load_manifest(canonical_manifest_path("TrainingRunManifest", manifest_id, root=root / "runs"))
    expected = {"cancelled"} if stopped else {"cancelled", "completed"}
    require(run.id == manifest_id and run.job_id == job_id and run.status in expected,
            "retained training did not settle as requested")
    parent = run.terminal_checkpoint
    require(isinstance(parent, ParentRef), "retained training has no terminal checkpoint")
    transaction = authenticate_checkpoint_custody_ref(parent, allowed_root=root / "checkpoints")
    state = transaction.manifest
    require(state.run_id == job_id and state.completed_training_batches == run.completed_batches
            and state.barrier == "after_train_batch", "terminal checkpoint differs from the settled update")
    return CheckpointLatestPointer(
        run_id=job_id, transaction_id=state.transaction_id, manifest_relative_path=parent.uri,
        manifest_sha256=transaction.manifest_sha256,
        transaction_root_sha256=state.content_integrity_digest.transaction_root_sha256,
        completed_coordinate=state.completed_coordinate,
        completed_training_batches=state.completed_training_batches,
    )


def load_attempt_spec(root, name, case):
    from feedbax.contracts.training import TrainingRunSpec

    data = (root / "spec.json").read_bytes()
    spec = TrainingRunSpec.model_validate_json(data)
    expected, _ = retained_spec(
        name, case, batches=spec.training_config.n_batches,
        checkpoint_interval=spec.checkpoint_progress.checkpoint_interval,
    )
    require(spec == expected, "retained spec does not match the original fixture")
    return spec, data


def checkpoint(fixture, job_id, name, case, service, *, stopped=False):
    from feedbax.training.checkpoint_custody import load_checkpoint_set

    root = attempt_root(fixture.paths.workspace, job_id)
    if not stopped:
        job, _ = host_job(fixture, job_id)
        require(job["state"] in {"stopped", "succeeded"}, "job has not settled")
    spec, _ = load_attempt_spec(root, name, case)
    pointer = read_terminal_checkpoint(root, job_id, stopped=stopped)
    if not stopped:
        require(job["checkpoint"] == pointer.model_dump(mode="json"),
                "retained transaction differs from the Host publication")
    loaded = reopen_checkpoint(spec, root / "checkpoints", pointer)
    checkpoint_set = load_checkpoint_set(root / "checkpoints", pointer.transaction_id)
    return {
        "job_id": job_id, "release": fixture.release.digest, "service": service,
        "batches": pointer.completed_training_batches, "transaction": pointer.transaction_id,
        "root_sha256": pointer.transaction_root_sha256,
        "checkpoint_set": checkpoint_set.exact_ref.model_dump(mode="json"),
        "parent_checkpoint_set": checkpoint_set.parent.model_dump(mode="json") if checkpoint_set.parent else None,
        "lineage": loaded.manifest.segment_lineage.model_dump(mode="json"),
        "model": inspect_model(loaded.slots["model"], case),
    }


def continuation(before, after):
    require(before["release"] == after["release"], "release changed before resuming")
    require(before["service"]["generation"] != after["service"]["generation"], "service was not restarted")
    require(before["job_id"] != after["job_id"] and before["batches"] < after["batches"], "no successor progress")
    require(after["parent_checkpoint_set"] == before["checkpoint_set"], "successor checkpoint set is unlinked")
    lineage = after["lineage"]
    require(lineage["parent_transaction_id"] == before["transaction"] and lineage["start_batch"] == before["batches"],
            "successor started from a different checkpoint")
    require(all(before["model"][key] == after["model"][key] for key in ("model", "asset", "muscle_state_widths")),
            "authored model or muscle state layout changed")
    require(before["model"]["controller_weights"] != after["model"]["controller_weights"],
            "controller weights did not advance after resume")
    return {"source": before, "successor": after}


def admit_retained(fixture, spec_data, *, project=None, source=None):
    from feedbax._atomic import write_bytes_atomic
    from feedbax.contracts.base import sha256_bytes
    from feedbax.host import AdmissionItem, HostClient, Job, Launch

    script = Path(__file__).resolve(strict=True)
    require(not script.is_relative_to(fixture.release.contents), "copy the qualifier outside the app")
    job_id = uuid4().hex
    project = project or f"retained-biomechanics-{job_id}"
    root = attempt_root(fixture.paths.workspace, job_id)
    write_bytes_atomic(root / "spec.json", spec_data, replace=False)
    python = str(fixture.release.resource("python"))
    digest = sha256_bytes(spec_data)
    source_args = () if source is None else (
        "--source-job", source.run_id, "--source-pointer", source.model_dump_json(),
    )
    launch = Launch(
        (python, "-I", "-B", str(script), "worker", job_id, digest, *source_args),
        cwd=str(fixture.paths.workspace),
    )
    request = f"retained-biomechanics:{digest}"
    job = Job(
        job_id, project, request, launch, continues=source.run_id if source else None,
        data={"runtime": python, "release": fixture.release.digest},
    )
    host = HostClient(fixture.paths.host)
    with chdir(fixture.paths.workspace):
        admission = host.admit(job_id, [AdmissionItem(job_id, job)], record={})
    (item,) = admission["items"]
    require(item["disposition"] == "admitted" and item["job_id"] == job_id,
            f"retained fixture did not admit: {item}")
    return {"project_id": project, "job_id": job_id, "release": fixture.release.digest}


def run_host_worker(job_id, spec_sha256, source_job, source_pointer):
    from feedbax.contracts.base import sha256_bytes
    from feedbax.contracts.checkpoints import CheckpointLatestPointer
    from feedbax.contracts.training import TrainingRunSpec
    from feedbax.host.io import append
    from feedbax.plugins.retained import new_retained_registry_bundle
    from feedbax.training.interruption import RunInterruptionController
    from feedbax.training.invocation import prepare_training_invocation
    from feedbax.training.retained_executor import execute_training_run_spec

    require(os.environ["FEEDBAX_HOST_JOB_ID"] == job_id, "unexpected Host delivery")
    root = attempt_root(Path.cwd(), job_id)
    event_log = Path(os.environ["FEEDBAX_HOST_EVENT_LOG"])
    stop_file = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
    data = (root / "spec.json").read_bytes()
    require(sha256_bytes(data) == spec_sha256, "retained spec differs from the submitted bytes")
    spec = TrainingRunSpec.model_validate_json(data)
    pointer = None
    source_root = None
    if source_job is not None:
        pointer = CheckpointLatestPointer.model_validate_json(source_pointer)
        source = attempt_root(Path.cwd(), source_job)
        require(pointer.run_id == source_job and data == (source / "spec.json").read_bytes(),
                "successor differs from the authenticated source run")
        source_root = source / "checkpoints"

    registries = new_retained_registry_bundle()
    prepared = prepare_training_invocation(spec, registries, run_id=job_id, resume=pointer is not None)
    total = spec.training_config.n_batches
    first_update = (pointer.completed_training_batches if pointer else 0) + 1

    def progress(_coordinate, completed, slots):
        append(event_log, "progress", {
            "completed_updates": completed, "total_updates": total,
            "metrics": check_update(slots, require_gradient=completed == first_update),
        })

    def published(write):
        pointer = write.latest_pointer
        append(event_log, "publication", {
            "key": f"checkpoint:{pointer.completed_training_batches}",
            "checkpoint": pointer.model_dump(mode="json"),
        }, sync=True)

    with RunInterruptionController(interactive=False, stop_file=stop_file) as interruption:
        result = execute_training_run_spec(
            spec, run_id=job_id, preparation=prepared, registry=registries.training_programs,
            manifest_root=root / "runs", checkpoint_root=root / "checkpoints",
            resume=pointer is not None, checkpoint_source_root=source_root, checkpoint_source_pointer=pointer,
            retain_history=False, cancellation_probe=lambda _coordinate: interruption.poll(),
            committed_update_callback=progress, checkpoint_published=published,
        )
    if result.status == "completed":
        require(result.end_completed_batches == total, "retained run ended before its requested target")
        outcome = "succeeded"
    elif result.status == "cancelled" and result.checkpoint_writes:
        outcome = "stopped"
    else:
        raise RuntimeError(f"retained worker did not publish a resumable result: {result.status}")
    append(event_log, "result", {"outcome": outcome}, sync=True)


def cache_probe(paths):
    paths.configure_studio()
    import jax
    import jax.numpy as jnp

    cache = paths.cache / "jax"
    require(jax.config.jax_compilation_cache_dir == str(cache), "JAX cache is outside the fixture authority")
    before = set(cache.glob("*-cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    stamp = time.monotonic_ns() % 65521
    value = jax.jit(lambda x: x * x + stamp)(jnp.array([1, 2], dtype=jnp.int32))
    require(value.tolist() == [stamp + 1, stamp + 4], "cached CPU computation differs")
    files = sorted(set(cache.glob("*-cache")) - before)
    require(files and jax.default_backend() == "cpu", "no CPU executable was written to the fixture JIT cache")
    return {"backend": "cpu", "cache": str(cache), "entries": [path.name for path in files]}


def source_smoke(name, case):
    from feedbax.desktop.release import AppPaths

    with TemporaryDirectory(prefix="fbx-mjx-source-smoke-") as temporary:
        root = Path(temporary)
        AppPaths.resolve(instance_root=root).configure_studio(desktop=False)
        return retained_source_update(name, case, root)


def retained_spec(name, case, *, batches, checkpoint_interval):
    from feedbax.contracts.training import TrainingRunSpec
    from feedbax.plugins.retained import new_retained_registry_bundle

    request = request_for(name, case, batches=batches, checkpoint_interval=checkpoint_interval)
    registries = new_retained_registry_bundle()
    # A fresh run must declare the key schedule even when its authored seed is zero.
    method = request.method_payload.model_copy(update={
        "payload": {**request.method_payload.payload, "key_schedule": "independent_v1"},
    })
    resolved = registries.training_programs.resolve_execution(request.method_ref, method)
    spec = TrainingRunSpec(
        graph={"inline": request.graph.model_dump(mode="json", exclude_none=True)},
        task=request.task, task_binding_spec=request.task_binding_spec,
        objective=request.objective, training_config=request.training_config,
        method_ref=request.method_ref, method_payload=method,
        worker_execution={"method_contract": resolved.contract, "effective_phase": resolved.effective_phase},
        checkpoint_progress={"checkpoint_interval": request.training_config.snapshot_interval, "progress_interval": 1},
    )
    return spec, registries


def bind_fixture_asset(case):
    from feedbax.persistence.model_assets import import_mujoco_asset

    if "mjcf" in case:
        imported = import_mujoco_asset(case["mjcf"].encode())
        require(imported.model_dump(mode="json") == case["interior"]["mujoco_asset"]["asset"],
                "import changed the fixture asset identity")


def check_update(slots, *, require_gradient=True):
    loss, grad_norm = (float(slots[slot]) for slot in ("train_loss", "grad_norm"))
    require(math.isfinite(loss) and math.isfinite(grad_norm), "fixture update has nonfinite metrics")
    require(not require_gradient or grad_norm > 0, "fixture's first update has no gradient")
    return {"loss": loss, "grad_norm": grad_norm}


def reopen_checkpoint(spec, custody, pointer):
    from feedbax.plugins.retained import new_retained_registry_bundle
    from feedbax.training.checkpoint_custody import load_latest_checkpoint
    from feedbax.training.invocation import prepare_training_invocation

    registries = new_retained_registry_bundle()
    prepared = prepare_training_invocation(spec, registries, run_id=pointer.run_id, resume=True)
    resolved = registries.training_programs.resolve_execution(spec.method_ref, spec.method_payload)
    loaded = load_latest_checkpoint(
        custody, expected_run_spec=spec, expected_phase_program=resolved.effective_phase.phase_program,
        expected_slots=prepared.initial_slots, published_pointer=pointer,
    )
    require(int(loaded.slots["batch_counter"]) == pointer.completed_training_batches,
            "checkpoint reopened at a different update")
    return loaded


def retained_source_update(name, case, root):
    from feedbax.training.checkpoint_custody import authenticate_published_checkpoint_custody
    from feedbax.training.invocation import prepare_training_invocation
    from feedbax.training.retained_executor import execute_training_run_spec

    bind_fixture_asset(case)
    spec, registries = retained_spec(name, case, batches=2, checkpoint_interval=1)
    prepared = prepare_training_invocation(spec, registries, run_id=name, resume=False)
    checkpoints = root / "checkpoints"
    result = execute_training_run_spec(
        spec, run_id=name, preparation=prepared, registry=registries.training_programs,
        manifest_root=root / "runs", checkpoint_root=checkpoints, stop_after_barrier="after_train_batch",
    )
    require(result.status == "completed" and result.end_completed_batches == 1,
            f"source fixture did not checkpoint after its first update: {result.status}")
    metrics = check_update(result.final_slots)
    (write,) = result.checkpoint_writes
    pointer = write.latest_pointer
    custody = authenticate_published_checkpoint_custody(checkpoints, require_single_transaction=True)
    require(custody.manifest_sha256 == pointer.manifest_sha256
            and custody.manifest.transaction_id == pointer.transaction_id
            and custody.manifest.completed_training_batches == 1,
            "checkpoint custody differs from the published update")
    resumed = reopen_checkpoint(spec, checkpoints, pointer)
    model = inspect_model(resumed.slots["model"], case)
    require(model == inspect_model(result.final_slots["model"], case), "restored model differs from the committed update")
    return {
        "case": name, **metrics, "model": model,
        "checkpoint": {
            "transaction": pointer.transaction_id, "manifest_sha256": pointer.manifest_sha256,
            "batches": pointer.completed_training_batches,
            "checkpoint_set": write.checkpoint_set.exact_ref.model_dump(mode="json"),
        },
    }


def source_plan(name, case):
    from feedbax.desktop.release import AppPaths
    from fastapi.testclient import TestClient

    with TemporaryDirectory(prefix="fbx-mjx-source-plan-") as temporary:
        paths = AppPaths.resolve(instance_root=Path(temporary))
        paths.configure_studio()
        from feedbax.web.app import create_app

        app = create_app(desktop_job_root=paths.state / "jobs")
        with TestClient(app) as client:
            def request(method, path, body=None):
                kwargs = {"content": body} if isinstance(body, bytes) else {"json": body}
                response = client.request(method, path, **kwargs)
                require(response.is_success, f"{method} {path}: {response.status_code}: {response.text[:1800]}")
                return response.json()

            plan = prepare(request, name, case, 2, 1)
            settled = app.state.project_folders.read_plan(plan["project_id"], revision=plan["revision"])
            require(not settled.plan.diagnostics, f"fixture Plan did not compile: {settled.plan.diagnostics}")
            (run,) = settled.plan.runs
            require(run.kind == "train" and run.admission_problem is None,
                    f"fixture did not compile a runnable Train: {run.admission_problem}")
            return {**plan, "request_id": run.request_id}


def evaluate(args):
    if args.action == "author":
        return author()
    if args.action in {"prove-quit", "prove-resume"}:
        verify = same_job if args.action == "prove-quit" else continuation
        return verify(*(json.loads(path.read_bytes()) for path in args.records))
    if args.action == "cache-probe":
        return cache_probe(Fixture(args.contents, args.fixture_root).paths)
    if args.action == "worker":
        return run_host_worker(args.job, args.spec_sha256, args.source_job, args.source_pointer)
    case = json.loads(args.fixtures.read_bytes())[args.case] if hasattr(args, "fixtures") else None
    if args.action == "source-smoke":
        return source_smoke(args.case, case)
    if args.action == "source-plan":
        return source_plan(args.case, case)
    if args.action == "checkpoint" and args.source_snapshot is not None:
        fixture = Fixture(args.contents, args.fixture_root)
        sample = json.loads(args.source_snapshot.read_bytes())
        require(sample["job"]["job_id"] == args.job and sample["job"]["status"] == "running"
                and sample["service"]["release"] == fixture.release.digest,
                "pre-stop snapshot does not belong to this job and release")
        return checkpoint(fixture, args.job, args.case, case, sample["service"], stopped=True)
    studio = Studio(args.contents, args.fixture_root)
    if args.action == "start":
        require(args.batches > 0 and args.checkpoint_interval > 0, "use positive batch and checkpoint counts")
        bind_fixture_asset(case)
        spec, _ = retained_spec(args.case, case, batches=args.batches, checkpoint_interval=args.checkpoint_interval)
        return admit_retained(studio, spec.model_dump_json(exclude_none=True).encode()) | {"case": args.case}
    if args.action == "snapshot":
        return snapshot(
            studio, args.job, args.status, args.after_sequence, after_update=args.after_update, gui=args.gui,
        )
    if args.action == "checkpoint":
        return checkpoint(studio, args.job, args.case, case, studio.health)
    job, _ = host_job(studio, args.job)
    require(job["state"] == "stopped" and job["project"] == args.project,
            "resume requires a stopped retained fixture in the same project")
    root = attempt_root(studio.paths.workspace, args.job)
    spec, data = load_attempt_spec(root, args.case, case)
    pointer = read_terminal_checkpoint(root, args.job, stopped=True)
    from feedbax.contracts.base import sha256_bytes

    require(job["request"] == f"retained-biomechanics:{sha256_bytes(data)}"
            and job["checkpoint"] == pointer.model_dump(mode="json"),
            "retained source disagrees with the Host publication")
    require(pointer.completed_training_batches < spec.training_config.n_batches,
            "retained source already reached its target")
    return admit_retained(studio, data, project=args.project, source=pointer)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("author")
    sub.add_parser("cache-probe")
    for action in ("start", "checkpoint", "source-smoke", "source-plan"):
        cmd = sub.add_parser(action)
        cmd.add_argument("fixtures", type=Path)
        cmd.add_argument("case", choices=CASES)
        if action == "start":
            cmd.add_argument("--batches", type=int, required=True)
            cmd.add_argument("--checkpoint-interval", type=int, required=True)
        if action == "checkpoint":
            cmd.add_argument("job")
            cmd.add_argument("--source-snapshot", type=Path)
    cmd = sub.add_parser("snapshot")
    cmd.add_argument("job")
    cmd.add_argument("--status", choices=("running", "stopped", "completed"), required=True)
    cmd.add_argument("--after-sequence", type=int, default=-2)
    cmd.add_argument("--after-update", type=int, default=-1)
    cmd.add_argument("--gui", choices=("present", "absent"))
    cmd = sub.add_parser("resume")
    cmd.add_argument("job")
    cmd.add_argument("project")
    cmd.add_argument("fixtures", type=Path)
    cmd.add_argument("case", choices=CASES)
    worker = sub.add_parser("worker")
    worker.add_argument("job")
    worker.add_argument("spec_sha256")
    worker.add_argument("--source-job")
    worker.add_argument("--source-pointer")
    for action in ("start", "snapshot", "checkpoint", "resume", "cache-probe"):
        sub.choices[action].add_argument("--contents", type=Path, required=True)
        sub.choices[action].add_argument("--fixture-root", type=Path, required=True)
    for action in ("prove-quit", "prove-resume"):
        sub.add_parser(action).add_argument("records", type=Path, nargs=3 if action == "prove-quit" else 2)
    args = parser.parse_args()
    if args.action == "worker" and (args.source_job is None) != (args.source_pointer is None):
        worker.error("pass --source-job and --source-pointer together")
    with redirect_stdout(sys.stderr):
        result = evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
