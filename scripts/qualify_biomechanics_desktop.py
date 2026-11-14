"""Author and qualify the MJX CPU Studio fixtures through installed entry points."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import subprocess
import sys
import time
from contextlib import closing, redirect_stdout
from http.cookies import SimpleCookie
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

CASES = ("planar-rigid", "spatial-mixed")
STEPS = 4


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def host_job(fixture, job_id, *, include_launch=False):
    from feedbax.host import HostClient

    snapshot = HostClient(fixture.paths.host).jobs_snapshot(
        job_id=job_id, include_launch=include_launch, start_host=False,
    )
    (job,) = snapshot["jobs"]
    require(job["id"] == job_id and job["data"]["runtime"] == str(fixture.release.resource("python")),
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
        from feedbax.desktop.release import AppPaths, Release

        self.release = Release.load(contents)
        require(self.release.fixture_root == fixture_root.resolve(strict=True), "not the requested fixture release")
        self.paths = AppPaths.resolve(instance_root=self.release.fixture_root)


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
        self.origin = health["origin"]
        ticket = call(self.paths, "ticket", body={})["ticket"]
        with self.connection() as connection:
            connection.request("POST", "/_desktop/exchange", headers={
                "X-Feedbax-Ticket": ticket, "Origin": self.origin,
            })
            reply = connection.getresponse()
            require(reply.status == 204, "could not establish a fixture session")
            self.cookie = SimpleCookie(reply.getheader("Set-Cookie"))["feedbax_studio"].OutputString(attrs=[])

    def connection(self):
        return closing(http.client.HTTPConnection("127.0.0.1", int(self.origin.rsplit(":", 1)[-1]), timeout=8))

    def request(self, method, path, body=None):
        mime = "application/xml" if isinstance(body, bytes) else "application/json"
        data = body if body is None or isinstance(body, bytes) else json.dumps(body)
        with self.connection() as connection:
            connection.request(method, path, data, {
                "Cookie": self.cookie, "Origin": self.origin, "Content-Type": mime,
            })
            reply = connection.getresponse()
            raw = reply.read()
            require(reply.status < 400, f"{method} {path} returned {reply.status}: {raw[:1800]!r}")
            return json.loads(raw)

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


def snapshot(studio, job_id, status, after, *, gui):
    deadline = time.monotonic() + 42
    wanted = "succeeded" if status == "completed" else status
    while time.monotonic() < deadline:
        record, cursor = host_job(studio, job_id)
        projected = studio.request("GET", f'/api/jobs?project={record["project"]}')
        job = next((row for row in projected["jobs"] if row["job_id"] == job_id), None)
        if job is not None:
            require(job["status"] not in {"failed", "interrupted", "corrupt"}, f"training failed: {job}")
            if job["status"] == wanted and cursor > after:
                break
        time.sleep(0.2)
    else:
        raise RuntimeError(f"{job_id} did not reach {status} after event {after}: {job}")
    result = {"job": job | {"event_sequence": cursor}, "service": studio.health}
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
    require(events[0] < events[1] < events[2], f"job did not progress through Quit/reopen: {events}")
    return {"job_id": before["job"]["job_id"], "events": events, "worker": before["worker"],
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


def checkpoint(fixture, job_id, case, service, *, stopped=False):
    import asyncio

    from feedbax.compiler import GraphDocument, compile_graph
    from feedbax.contracts.base import ParentRef
    from feedbax.contracts.studio_training import PublishedTrainingCheckpoint, StudioGraphMethodPayload
    from feedbax.plugins.composition import compose_application
    from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
    from feedbax.training.checkpoint_custody import (
        authenticate_checkpoint_custody_ref, load_checkpoint_set, open_published_model_slot,
    )
    from feedbax.training.studio_executor import studio_training_key
    from feedbax.compiler.prototypes import prototypes_from_task_bindings

    job, _ = host_job(fixture, job_id, include_launch=True)
    expected = {"stopped"} if stopped else {"stopped", "succeeded"}
    require(job["state"] in expected, "job has not settled as requested")
    binding = BoundStudioTrainingJobInput.model_validate(job["launch"]["payload"])
    require(binding.canonical_request is not None, "fixture did not use saved Plan admission")
    checkpoint = PublishedTrainingCheckpoint.model_validate(job["checkpoint"])
    custody = checkpoint.checkpoint_root.resolve(strict=True)
    require(custody.is_relative_to(fixture.paths.web / "projects" / job["project"] / "results"),
            "checkpoint directory leaves the fixture project")
    pointer = checkpoint.pointer
    transaction = authenticate_checkpoint_custody_ref(ParentRef(
        kind="TrainingCheckpointTransactionManifest", id=pointer.transaction_id,
        role="training_checkpoint_custody", uri=pointer.manifest_relative_path,
        metadata={"manifest_sha256": pointer.manifest_sha256},
    ), allowed_root=custody)
    manifest = transaction.manifest
    checkpoint_set = load_checkpoint_set(custody, transaction.parent_ref.id)
    slot = open_published_model_slot(custody, pointer)
    require(slot is not None, "checkpoint has no portable model slot")
    method = StudioGraphMethodPayload.model_validate(binding.spec.method_payload.payload)
    registry = asyncio.run(compose_application(modules=binding.bootstrap_modules)).bundle
    executable = compile_graph(
        GraphDocument(graph=slot.graph), registry.components,
        input_prototypes=prototypes_from_task_bindings(binding.spec.task_binding_spec),
        initialization_key=studio_training_key(method.seed, "model"), artifact_provider=slot.provider,
    )
    model = inspect_model(slot.restore(executable.graph), case)
    return {"job_id": job_id, "release": fixture.release.digest, "service": service,
            "batches": manifest.completed_training_batches, "transaction": transaction.parent_ref.id,
            "root_sha256": manifest.content_integrity_digest.transaction_root_sha256,
            "checkpoint_set": checkpoint_set.exact_ref.model_dump(mode="json"),
            "parent_checkpoint_set": checkpoint_set.parent.model_dump(mode="json") if checkpoint_set.parent else None,
            "lineage": manifest.segment_lineage.model_dump(mode="json") if manifest.segment_lineage else None,
            "model": model}


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
    import jax
    import jax.numpy as jnp

    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.training import TrainingSpec
    from feedbax.plugins.application import new_application_registry_bundle
    from feedbax.studio.canonical_train import lower_canonical_train
    from feedbax.training.optimizers import build_optimizer
    from feedbax.training.studio_executor import StudioUpdateKernel
    from feedbax.persistence.model_assets import import_mujoco_asset
    from feedbax.training.retained_graph import compile_training_run, evaluate_graph_loss, rollout_graph

    if "mjcf" in case:
        require(import_mujoco_asset(case["mjcf"].encode()).model_dump(mode="json") == (
            case["interior"]["mujoco_asset"]["asset"]
        ), "source import changed the fixture identity")
    spec = lower_canonical_train(
        request_for(name, case, batches=2, checkpoint_interval=1),
        new_application_registry_bundle().training_programs,
    )
    optimizer_spec = spec.method_payload.payload["optimizer"]
    compiled = compile_training_run(
        graph_spec=spec.graph.inline, component_registry=ComponentRegistry(),
        training_spec=TrainingSpec(
            optimizer=optimizer_spec, loss=spec.objective.loss, n_batches=2, batch_size=1,
        ),
        task_spec=spec.task.model_dump(mode="json", exclude_none=True),
        task_binding_spec=spec.task_binding_spec, cfg=spec.training_config,
    )
    import equinox as eqx

    optimizer = build_optimizer(optimizer_spec, schedule_origin_step=0, current_step=0, gradient_clip=None)
    trainable, _ = eqx.partition(compiled.graph, compiled.trainable_filter)
    update = StudioUpdateKernel(
        compiled=compiled, optimizer=optimizer, evaluate_loss_fn=evaluate_graph_loss,
        rollout_fn=lambda graph, plan, key: rollout_graph(graph, plan, key=key),
    )
    result = update.reported_update({
        "model": compiled.graph, "optimizer": optimizer.init(trainable),
        "prng": jax.random.PRNGKey(0), "batch_counter": jnp.array(0, dtype=jnp.int32),
    })
    require(int(result["batch_counter"]) == 1 and math.isfinite(result["train_loss"])
            and result["grad_norm"] > 0, "source fixture failed its first numerical update")
    return {"case": name, "loss": result["train_loss"], "grad_norm": result["grad_norm"],
            "model": inspect_model(result["model"], case)}


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
            require(run.kind == "train" and run.admission_problem is None, "fixture did not compile a runnable Train")
            return {**plan, "request_id": run.request_id}


def evaluate(args):
    if args.action == "author":
        return author()
    if args.action in {"prove-quit", "prove-resume"}:
        verify = same_job if args.action == "prove-quit" else continuation
        return verify(*(json.loads(path.read_bytes()) for path in args.records))
    if args.action == "cache-probe":
        return cache_probe(Fixture(args.contents, args.fixture_root).paths)
    case = json.loads(args.fixtures.read_bytes())[args.case] if hasattr(args, "fixtures") else None
    if args.action == "source-smoke":
        return source_smoke(args.case, case)
    if args.action == "source-plan":
        return source_plan(args.case, case)
    if args.action == "checkpoint" and args.source_snapshot is not None:
        fixture = Fixture(args.contents, args.fixture_root)
        sample = json.loads(args.source_snapshot.read_bytes())
        require(sample["job"]["job_id"] == args.job and sample["service"]["release"] == fixture.release.digest,
                "pre-stop snapshot does not belong to this job and release")
        return checkpoint(fixture, args.job, case, sample["service"], stopped=True)
    studio = Studio(args.contents, args.fixture_root)
    if args.action == "start":
        plan = prepare(studio.request, args.case, case, args.batches, args.checkpoint_interval)
        job = studio.request("POST", f'/api/projects/{plan["project_id"]}/train-runs', {
            "revision": plan["revision"], "token": uuid4().hex,
            "run_keys": [{"variant": "default", "step": "train", "row": "default"}],
        })
        (item,) = job["items"]
        require(item["disposition"] == "admitted", f"fixture did not admit: {item}")
        return {**plan, "job_id": item["job_id"], "case": args.case, "release": studio.release.digest}
    if args.action == "snapshot":
        return snapshot(studio, args.job, args.status, args.after_sequence, gui=args.gui)
    if args.action == "checkpoint":
        return checkpoint(studio, args.job, case, studio.health)
    verify_project(studio.request, args.project, case["interior"])
    return studio.request("POST", f"/api/jobs/{args.job}/resume", {"token": uuid4().hex})


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
    cmd.add_argument("--gui", choices=("present", "absent"))
    cmd = sub.add_parser("resume")
    cmd.add_argument("job")
    cmd.add_argument("project")
    cmd.add_argument("fixtures", type=Path)
    cmd.add_argument("case", choices=CASES)
    for action in ("start", "snapshot", "checkpoint", "resume", "cache-probe"):
        sub.choices[action].add_argument("--contents", type=Path, required=True)
        sub.choices[action].add_argument("--fixture-root", type=Path, required=True)
    for action in ("prove-quit", "prove-resume"):
        sub.add_parser(action).add_argument("records", type=Path, nargs=3 if action == "prove-quit" else 2)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        result = evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
