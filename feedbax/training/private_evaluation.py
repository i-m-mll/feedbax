"""Private CPU qualification over one authored finite Train checkpoint selection."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from math import isfinite
import os
from pathlib import Path
from typing import Sequence

from pydantic import model_validator

from feedbax.checkpoint_coordinates import finite_coordinates
from feedbax.contracts.base import StrictModel
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.strict_json import strict_json_loads


class PrivateEvaluationRequest(StrictModel):
    manifest_root: Path
    producer: str
    producer_request: str
    coordinates: tuple[int, ...]
    evaluate_update: int
    variation: OverridePatch

    @model_validator(mode="after")
    def _validate_scope(self) -> PrivateEvaluationRequest:
        finite_coordinates(self.coordinates)
        if self.evaluate_update not in self.coordinates:
            raise ValueError("/evaluate_update: must name a selected checkpoint")
        patch = self.variation
        if (
            patch.path != "nodes.plant.params.mass"
            or patch.op != "replace"
            or type(patch.value) not in (int, float)
            or not isfinite(patch.value)
            or patch.value <= 0
        ):
            raise ValueError(
                "/variation: the private witness supports a positive PointMass mass replacement"
            )
        if not self.manifest_root.is_absolute():
            raise ValueError("/manifest_root: expected an absolute project store")
        return self


def _bound_selection(request: PrivateEvaluationRequest):
    from feedbax.job_completion import load_bound_checkpoint_selection, read_host_completion_binding

    binding = read_host_completion_binding()
    if binding["dependency_job_id"] != request.producer:
        raise ValueError("/inputs: Host selection differs from the authored producer")
    pinned = load_bound_checkpoint_selection(
        binding, root=request.manifest_root, producer_request=request.producer_request,
        coordinates=request.coordinates,
    )
    return binding["checkpoint_selection"]["lineage_tip"], pinned


def _evaluate(request: PrivateEvaluationRequest, checkpoint, launch):
    import equinox as eqx
    import jax
    import numpy as np

    from feedbax.compiler import GraphDocument, compile_graph
    from feedbax.contracts.graph import GraphSpec
    from feedbax.contracts.matrix_core import apply_override_patches
    from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD, StudioGraphMethodPayload
    from feedbax.contracts.training import TrainingSpec
    from feedbax.dataset import Dataset, Variable
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.persistence.dataset import open_dataset, publish_dataset
    from feedbax.plugins.composition import compose_application
    from feedbax.runtime._graph_key_schedule import derive_key
    from feedbax.runtime.graph import GraphTraceRequest, init_state_from_component
    from feedbax.training.checkpoint_custody import open_published_model_slot
    from feedbax.training.studio_executor import studio_training_key
    from feedbax.web.worker.execution import (
        _evaluate_loss,
        compile_training_run,
        prototypes_from_task_bindings,
        rollout_graph,
    )

    spec = launch.spec
    if jax.default_backend() != "cpu" or spec.method_ref != STUDIO_GRAPH_METHOD:
        raise ValueError("/method: private reaching evaluation requires the Studio CPU method")
    if spec.task_binding_spec is None or spec.objective.loss is None:
        raise ValueError(
            "/objective: private reaching evaluation requires the shared authored term plan"
        )
    slot = open_published_model_slot(checkpoint.checkpoint_root, checkpoint.pointer)
    if slot is None or {name: node.type for name, node in slot.graph.nodes.items()} != {
        "cell": "feedbax.GRU",
        "readout": "feedbax.Linear",
        "plant": "feedbax.PointMass",
    }:
        raise ValueError("/slots/model: the private witness requires the typed reaching graph")
    if slot.graph.nodes["readout"].params.get("trainable") is not False:
        raise ValueError("/graph/nodes/readout: the witness requires a frozen readout")

    method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
    registry = asyncio.run(compose_application(modules=launch.bootstrap_modules)).bundle

    def build(seed):
        return compile_graph(
            GraphDocument(graph=slot.graph),
            registry.components,
            input_prototypes=prototypes_from_task_bindings(spec.task_binding_spec),
            initialization_key=studio_training_key(seed, "model"),
            artifact_provider=slot.provider,
        )

    executable = build(method.seed)
    restored = slot.restore(executable.graph)
    frozen = restored.nodes["readout"].layer
    fresh = executable.graph
    other = build((method.seed + 1) % 2**32).graph
    if (
        not eqx.tree_equal(frozen, fresh.nodes["readout"].layer)
        or eqx.tree_equal(frozen, other.nodes["readout"].layer)
        or eqx.tree_equal(restored.nodes["cell"].cell, fresh.nodes["cell"].cell)
    ):
        raise ValueError(
            "/slots/model: retained learned and frozen controls do not distinguish restoration"
        )

    graph_document = GraphSpec.model_validate(
        apply_override_patches(slot.graph.model_dump(mode="json"), [request.variation]),
    )
    plant = restored.nodes["plant"]
    config = plant.configuration_values()
    mass = graph_document.nodes["plant"].params["mass"]
    if mass == config["mass"] or config["damping"] != 0:
        raise ValueError("/variation: expected a distinct mass on the undamped reaching plant")
    changed = type(plant).from_config(config | {"mass": mass})
    variant = eqx.tree_at(lambda graph: graph.nodes["plant"], restored, changed).prepare()
    if not all(
        eqx.tree_equal(restored.nodes[name], variant.nodes[name]) for name in ("cell", "readout")
    ):
        raise ValueError("/variation: mass replacement changed a restored controller owner")
    compiled = compile_training_run(
        graph_spec=slot.graph,
        training_spec=TrainingSpec(
            optimizer=method.optimizer,
            loss=spec.objective.loss,
            n_batches=spec.training_config.n_batches,
            batch_size=1,
        ),
        task_spec=spec.task.model_dump(mode="json", exclude_none=True),
        task_binding_spec=spec.task_binding_spec,
        cfg=spec.training_config,
        component_registry=registry.components,
        executable_graph=replace(executable, graph=restored),
    )
    trace = GraphTraceRequest(
        kind="edge",
        selector="proof_force",
        source_node="readout",
        source_port="output",
        target_node="plant",
        target_port="force",
    )
    compiled = replace(compiled, trace_requests=(*compiled.trace_requests, trace))
    trial_ids = ("paired-0", "paired-1", "paired-2")
    keys = [
        derive_key(jax.random.PRNGKey(731), "private-reaching-evaluation", trial)
        for trial in trial_ids
    ]

    def rollout(model, key):
        state = init_state_from_component(model)
        component = model.nodes["plant"]
        reset = component.initial_outputs(state.get(component.state_index))["effector"].pos
        trace = rollout_graph(model, compiled, key=key)
        position = np.asarray(trace["outputs"]["effector"].pos)
        target = np.asarray(trace["task_data"]["reach_target"])
        loss, terms = _evaluate_loss(compiled, trace)
        np.testing.assert_allclose(loss, np.sum((position[-1] - target[-1]) ** 2), rtol=2e-5)
        np.testing.assert_allclose(loss, sum(map(float, terms.values())), rtol=2e-5)
        return (
            np.concatenate((np.asarray(reset)[None, :], position)),
            np.asarray(trace["trace"]["proof_force"]),
            target,
            float(loss),
        )

    conditions = [[rollout(model, key) for key in keys] for model in (restored, variant)]
    positions, forces, targets, losses = (
        np.asarray(field)
        for field in zip(
            *(list(zip(*condition, strict=True)) for condition in conditions),
            strict=True,
        )
    )
    np.testing.assert_array_equal(targets[0], targets[1])
    np.testing.assert_array_equal(forces[0], forces[1])
    np.testing.assert_allclose(
        positions[1, :, 1:],
        positions[0, :, 1:] * (config["mass"] / mass),
        rtol=3e-5,
        atol=1e-8,
    )
    if len({trial.tobytes() for trial in targets[0]}) < 2:
        raise ValueError("/trials: sampled held-out targets did not distinguish paired draws")
    time = np.arange(compiled.n_steps + 1, dtype=np.float64) * float(plant.dt)
    axes = ("checkpoint", "condition", "trial")
    dataset = Dataset.from_arrays(
        {
            "position": Variable(positions[None], (*axes, "state_time", "xy"), units="m"),
            "force": Variable(forces[None], (*axes, "transition", "xy"), units="N"),
            "target": Variable(targets[None], (*axes, "transition", "xy"), units="m"),
            "objective": Variable(losses[None], axes),
        },
        coords={
            "checkpoint": Variable(np.asarray([checkpoint.completed_updates]), ("checkpoint",)),
            "condition": Variable(np.asarray(["source", "mass_variant"]), ("condition",)),
            "trial": Variable(np.asarray(trial_ids), ("trial",)),
            "xy": Variable(np.asarray(["x", "y"]), ("xy",)),
            "state_time": Variable(time, ("state_time",), units="s"),
            "transition_start": Variable(time[:-1], ("transition",), units="s"),
            "transition_end": Variable(time[1:], ("transition",), units="s"),
        },
        schema_id="feedbax.proof.private-reaching.v1",
    )
    provider = ImmutableArtifactBlobProvider(request.manifest_root)
    ref = publish_dataset(provider, dataset)
    with open_dataset(
        provider.object_path(ref.object_ref), expected_descriptor_sha256=ref.descriptor_sha256
    ) as opened:
        selected = opened.data("position", (0, 1, 0, slice(0, 3)))
        np.testing.assert_array_equal(selected, positions[1, 0, :3])
        dimensions = opened.descriptor.dimensions
    return {
        "dataset": ref.model_dump(mode="json"),
        "dimensions": dimensions,
        "selected_position": selected.tolist(),
        "trial_keys": [np.asarray(jax.random.key_data(key)).tolist() for key in keys],
        "paired_target_draws": len({trial.tobytes() for trial in targets[0]}),
        "source_mass": config["mass"],
        "variant_mass": mass,
        "objective": losses.tolist(),
        "restored_frozen_and_learned": True,
    }


def main(argv: Sequence[str]) -> int:
    from feedbax.contracts.authored_canonical import canonical_bytes
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.training_host_worker import _Events, _Result, _error

    events = _Events(Path(os.environ["FEEDBAX_HOST_EVENT_LOG"]))
    try:
        if argv:
            raise ValueError("Private evaluation takes the authored Host input")
        request = PrivateEvaluationRequest.model_validate(
            strict_json_loads(Path(os.environ["FEEDBAX_HOST_INPUT"]).read_bytes()),
        )
        tip, selected = _bound_selection(request)
        checkpoint, launch = next(
            pair for pair in selected if pair[0].completed_updates == request.evaluate_update
        )
        result = _evaluate(request, checkpoint, launch)
        provider = ImmutableArtifactBlobProvider(request.manifest_root)
        receipt = provider.store_rooted_object(
            canonical_bytes(
                {
                    "job": os.environ["FEEDBAX_HOST_JOB_ID"],
                    "authored": request.model_dump(mode="json"),
                    "lineage_tip": tip,
                    "selected": [item.model_dump(mode="json") for item, _ in selected],
                    "evaluated": checkpoint.model_dump(mode="json"),
                    **result,
                }
            ),
            namespace="private-reaching-evaluation",
        )
        events.append(
            "result",
            {"outcome": "succeeded", "private_evaluation": receipt.model_dump(mode="json")},
            sync=True,
        )
        return 0
    except Exception as error:
        return events.result(_Result("failed", error=_error(error, "restoring")))
