"""Seeded execution is a function of canonical graph semantics at every depth."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

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
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.runtime.components import Noise
from feedbax.runtime.graph import Graph, init_state_from_component


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
            compile_graph(GraphDocument(graph=spec), ComponentRegistry(load_user_components=False))
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
    registry = ComponentRegistry(load_user_components=False)
    compiled = compile_graph(GraphDocument(graph=GraphSpec()), registry)
    for model, current in ((ResolvedGraph, compiled.resolved), (CompilationRecord, compiled.record)):
        for version in range(1, int(current.schema_version)):
            payload = current.model_dump(mode="json")
            payload["schema_version"] = str(version)
            with pytest.raises(ValidationError, match="recompile the authored GraphDocument"):
                model.model_validate(payload)

    with pytest.raises(GraphCompilationError) as caught:
        compile_graph(
            GraphDocument(graph=GraphSpec(nodes={"bad": ComponentSpec(type="missing")})), registry
        )
    for model, record in (
        (CompilationRecord, compiled.record),
        (CompilationFailureRecord, caught.value.record),
    ):
        for version in ("1", "2", "unknown"):
            payload = record.model_dump(mode="json")
            payload["compiler_version"] = version
            with pytest.raises(ValidationError, match="recompile the authored GraphDocument"):
                model.model_validate(payload)
