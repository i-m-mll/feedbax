"""Seeded execution is a function of canonical graph semantics at every depth."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from feedbax.compiler import (
    CompilationFailureRecord,
    CompilationRecord,
    GraphCompilationError,
    GraphDocument,
    ResolvedGraph,
    compile_graph,
)
from feedbax.compiler.templates import recurrent_node_output_initializer
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphMetadata, GraphSpec, WireSpec
from feedbax.runtime._graph_key_schedule import (
    derive_key,
    keyed_nodes,
    sample_steps,
    step_key,
    step_keys,
)
from feedbax.runtime.components import Noise
from feedbax.runtime.graph import Graph, init_state_from_component
from feedbax.runtime.iteration import iterate_component


def test_nested_noise_reset_keys_follow_producers() -> None:
    gains = ("first", "second")
    child = GraphSpec(
        nodes={
            "noise": ComponentSpec(
                type="Noise", params={"std": 1.0, "shape": [4]}, output_ports=["output"]
            ),
            **{
                name: ComponentSpec(
                    type="Gain",
                    params={"gain": 1.0},
                    input_ports=["input"],
                    output_ports=["output"],
                )
                for name in gains
            },
        },
        wires=[
            WireSpec(
                source_node=name,
                source_port="output",
                target_node=name,
                target_port="input",
                temporality="recurrent",
                recurrent_initializer=recurrent_node_output_initializer(
                    "noise", "output", state_slot="input"
                ),
            )
            for name in gains
        ],
        output_ports=[*gains, "live"],
        output_bindings={
            **{name: (name, "output") for name in gains},
            "live": ("noise", "output"),
        },
    )
    siblings = ("left", "right")
    outputs = {f"{node}_{port}": (node, port) for node in siblings for port in child.output_ports}
    spec = GraphSpec(
        nodes={
            name: ComponentSpec(type="Subgraph", output_ports=child.output_ports)
            for name in siblings
        },
        subgraphs=dict.fromkeys(siblings, child),
        output_ports=list(outputs),
        output_bindings=outputs,
    )
    compiled = compile_graph(
        GraphDocument(graph=spec), ComponentRegistry()
    )
    # Both sides of the reset fix originally emitted this resolved identity.
    assert compiled.resolved.resolved_sha256 != (
        "49d8b3239b0a67763291f46017c5b5f95cf3fc8502cb59c0899e7f8806e27c5a"
    )
    graph = compiled.graph
    key = jax.random.key(17)

    def rollout(graph, n_steps):
        return graph({}, init_state_from_component(graph), key=key, n_steps=n_steps)[0]

    scanned = rollout(graph, 2)
    assert not jnp.array_equal(scanned["left_first"][0], scanned["right_first"][0])
    for node in siblings:
        assert jnp.array_equal(scanned[f"{node}_first"], scanned[f"{node}_second"])
        assert not jnp.array_equal(scanned[f"{node}_first"][0], scanned[f"{node}_live"][0])
        assert not jnp.array_equal(scanned[f"{node}_live"][0], scanned[f"{node}_live"][1])

    edited_graph = graph.add_node("ahead", Noise(shape=(4,)))
    edited_child = graph.nodes["left"].add_node("ahead", Noise(shape=(4,)))
    edited_graph = eqx.tree_at(lambda g: g.nodes["left"], edited_graph, edited_child)
    replayed = rollout(edited_graph, 3)

    with pytest.raises(ValueError, match="Missing recurrent carry for subgraph"):
        graph.step({}, init_state_from_component(graph), {}, key=key)

    state, carry = init_state_from_component(graph), None
    for t in range(2):
        stepped, state, carry = graph.step({}, state, carry, key=key, t=t)
        for name in outputs:
            assert jnp.array_equal(scanned[name][t], replayed[name][t])
            assert jnp.array_equal(scanned[name][t], stepped[name])


@pytest.mark.parametrize("depth", [0, 2])
def test_reordered_random_nodes_preserve_compiler_evidence_and_seeded_execution(depth):
    compiled = []
    direct = []
    for names in (("z", "a"), ("a", "z")):
        bindings = {name: (name, "output") for name in ("a", "z")}
        spec = GraphSpec(
            nodes={
                name: ComponentSpec(
                    type="Noise", params={"std": 1.0, "shape": [4]}, output_ports=["output"]
                )
                for name in names
            },
            output_ports=["a", "z"],
            output_bindings=bindings,
        )
        graph = Graph(
            nodes={name: Noise(shape=(4,)) for name in names},
            wires=(),
            output_ports=("a", "z"),
            output_bindings=bindings,
        )
        for _ in range(depth):
            bindings = {name: ("nested", name) for name in ("a", "z")}
            spec = GraphSpec(
                nodes={"nested": ComponentSpec(type="Subgraph", output_ports=["a", "z"])},
                subgraphs={"nested": spec},
                output_ports=["a", "z"],
                output_bindings=bindings,
            )
            graph = Graph(
                nodes={"nested": graph},
                wires=(),
                output_ports=("a", "z"),
                output_bindings=bindings,
            )
        compiled.append(
            compile_graph(GraphDocument(graph=spec), ComponentRegistry())
        )
        direct.append(graph)

    assert compiled[0].resolved == compiled[1].resolved
    assert compiled[0].record == compiled[1].record
    assert compiled[0].record.resolved_sha256 == compiled[1].record.resolved_sha256
    outputs = [
        graph({}, init_state_from_component(graph), key=jax.random.PRNGKey(17))[0]
        for graph in (compiled[0].graph, compiled[1].graph, *direct)
    ]
    for actual in outputs[1:]:
        for name in ("a", "z"):
            assert jnp.array_equal(actual[name], outputs[0][name])
    assert not jnp.array_equal(outputs[0]["a"], outputs[0]["z"])


def test_construction_keys_are_scoped_and_support_explicit_overrides() -> None:
    linear = ComponentSpec(
        type="Linear",
        params={"input_size": 3, "output_size": 2},
        input_ports=["input"],
        output_ports=["output"],
    )
    inner = GraphSpec(nodes={"linear": linear})
    siblings = {"left": inner, "right": inner}
    spec = GraphSpec(
        nodes={name: ComponentSpec(type="Subgraph") for name in siblings},
        subgraphs=siblings,
    )
    registry = ComponentRegistry()

    def build(spec, **kwargs):
        graph = compile_graph(GraphDocument(graph=spec), registry, **kwargs).graph
        return tuple(graph.nodes[name].nodes["linear"].layer.weight for name in siblings)

    left, right = build(spec)
    assert not jnp.array_equal(left, right)

    relabeled = spec.model_copy(
        update={
            "metadata": GraphMetadata(
                name="display name",
                created_at="2026-01-01",
                updated_at="2026-01-01",
            )
        }
    )
    edited = relabeled.model_copy(
        update={
            "subgraphs": {**siblings, "left": GraphSpec(nodes={"ahead": linear, "linear": linear})},
        }
    )
    edited_left, edited_right = build(edited)
    assert jnp.array_equal(left, edited_left)
    assert jnp.array_equal(right, edited_right)

    explicit = jax.random.key(37)
    overridden_left, overridden_right = build(
        spec,
        initialization_keys={
            ("left", "linear"): explicit,
            ("right", "linear"): explicit,
        },
    )
    assert jnp.array_equal(overridden_left, overridden_right)
    assert not jnp.array_equal(left, overridden_left)


def test_leaf_iteration_has_stable_prefix_for_legacy_keys() -> None:
    noise = Noise(shape=(4,))
    key = jax.random.PRNGKey(11)
    state = init_state_from_component(noise)
    short, _, _ = iterate_component(noise, {}, state, 2, key)
    long, _, _ = iterate_component(noise, {}, state, 4, key)
    assert jnp.array_equal(short["output"], long["output"][:2])


def test_v4_key_derivation_frames_tokens_and_pins_words() -> None:
    key = jax.random.key(42)
    derived = derive_key(key, "occurrence", "a/b", "c", index=7)
    assert jax.random.key_data(derived).tolist() == [60332732, 983897167]
    assert not jnp.array_equal(derived, derive_key(key, "occurrence", "a", "b/c", index=7))


@pytest.mark.parametrize("t", [jnp.asarray(1.5), jnp.asarray(-1), 2**32])
def test_manual_timestep_rejects_invalid_rng_indices(t) -> None:
    graph = Graph(nodes={}, wires=())
    with pytest.raises(ValueError, match="Random stream index"):
        graph.step({}, init_state_from_component(graph), key=jax.random.key(0), t=t)


@pytest.mark.parametrize("implementation", ["rbg", "unsafe_rbg"])
def test_rbg_keys_and_draws_match_scalar_execution(implementation) -> None:
    key = jax.random.key(12, impl=implementation)
    names = ("alpha", "beta")
    for name, actual in keyed_nodes(names, key):
        assert jnp.array_equal(actual, derive_key(key, "occurrence", name))
    for t, actual in enumerate(step_keys(key, 2)):
        assert jnp.array_equal(actual, step_key(key, t))

    def draw(step_key):
        return jax.random.normal(step_key, (2,))

    assert jnp.array_equal(
        sample_steps(draw, key, 2), jnp.stack([draw(step_key(key, t)) for t in range(2)])
    )


def test_branching_seeded_execution_and_cycle_diagnostics_ignore_hash_seed():
    script = textwrap.dedent(
        """
        import json
        import jax
        from feedbax.runtime.graph import Component, Graph, Wire, init_state_from_component

        class RandomStep(Component):
            input_ports = ("input",)
            output_ports = ("output",)

            def __call__(self, inputs, state, *, key):
                return {"output": inputs.get("input", 0) + jax.random.normal(key, ())}, state

        graph = Graph(
            nodes={name: RandomStep() for name in {"root", "a", "aa", "b", "z"}},
            wires=tuple(Wire(src, "output", tgt, "input")
                        for src, tgt in {("root", "a"), ("root", "b"), ("a", "aa")}),
            output_ports=("aa", "b", "z"),
            output_bindings={name: (name, "output") for name in ("aa", "b", "z")},
        )
        outputs, _ = graph({}, init_state_from_component(graph), key=jax.random.PRNGKey(4))
        try:
            cyclic = Graph(
                nodes={name: RandomStep() for name in {"a", "b", "c"}},
                wires=tuple(Wire(src, "output", tgt, "input")
                            for src, tgt in {("a", "b"), ("a", "c"),
                                             ("b", "c"), ("c", "a")}),
            )
            cyclic._execution_order
        except ValueError as error:
            cycle = str(error)
        else:
            raise AssertionError("instant cycle accepted")
        assert graph._execution_order == ("root", "a", "aa", "b", "z")
        print(json.dumps({"outputs": {k: float(v) for k, v in outputs.items()}, "cycle": cycle}))
        """
    )
    results = [
        subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
        for seed in ("1", "37")
    ]
    assert results[0] == results[1]


def test_historical_compiler_evidence_requires_authored_document_recompilation():
    registry = ComponentRegistry()
    compiled = compile_graph(GraphDocument(graph=GraphSpec()), registry)
    for model, current in (
        (ResolvedGraph, compiled.resolved),
        (CompilationRecord, compiled.record),
    ):
        historical_versions = tuple(str(v) for v in range(1, int(current.schema_version)))
        for version in (*historical_versions, "unknown"):
            payload = current.model_dump(mode="json")
            payload["schema_version"] = version
            with pytest.raises(
                ValidationError,
                match="migration_intentionally_absent=yes; recompile the authored GraphDocument",
            ):
                model.model_validate(payload)

        payload = current.model_dump(mode="json")
        payload["key_schedule"]["schedule_id"] = (
            "feedbax.graph_key_schedule.canonical_topological_split.v3"
        )
        with pytest.raises(ValidationError, match="original producer revision and environment"):
            model.model_validate(payload)

    with pytest.raises(GraphCompilationError) as caught:
        compile_graph(
            GraphDocument(graph=GraphSpec(nodes={"bad": ComponentSpec(type="missing")})), registry
        )
    for model, record in (
        (CompilationRecord, compiled.record),
        (CompilationFailureRecord, caught.value.record),
    ):
        for version in ("1", "2", "3", "4", "unknown"):
            payload = record.model_dump(mode="json")
            payload["compiler_version"] = version
            with pytest.raises(ValidationError, match="recompile the authored GraphDocument"):
                model.model_validate(payload)

    unversioned_failure = caught.value.record.model_dump(mode="json", exclude={"compiler_version"})
    with pytest.raises(ValidationError, match="compiler_version"):
        CompilationFailureRecord.model_validate(unversioned_failure)
