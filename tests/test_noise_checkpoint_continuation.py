"""Native checkpoint continuation for delayed registered Channel noise."""

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.contracts.training import GraphTopologySourceSpec
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.runtime.iteration import iterate_component
from feedbax.training.checkpoint_custody import (
    load_latest_checkpoint,
    write_checkpoint_transaction,
)
from tests.test_checkpoint_custody import _run_spec

_NOISE_MODELS = (
    "signal_dependent_gaussian",
    "signal_dependent_plus_additive",
    "additive_gaussian",
    "none",
)


def _setup():
    graph_spec = GraphSpec(
        nodes={
            noise: ComponentSpec(
                type="Channel",
                params={
                    "delay": 1,
                    "input_shape": [2],
                    "noise_model": noise,
                    "additive_noise_std": 0.25,
                    "signal_dependent_noise_std": 0.125,
                },
                input_ports=["input"],
                output_ports=["output"],
            )
            for noise in _NOISE_MODELS
        },
        input_ports=list(_NOISE_MODELS),
        output_ports=list(_NOISE_MODELS),
        input_bindings={noise: (noise, "input") for noise in _NOISE_MODELS},
        output_bindings={noise: (noise, "output") for noise in _NOISE_MODELS},
    )
    model = compile_graph(
        GraphDocument(graph=graph_spec), ComponentRegistry()
    ).graph.prepare()
    spec = _run_spec().model_copy(
        update={"graph": GraphTopologySourceSpec(inline=graph_spec.model_dump(mode="json"))}
    )
    return model, spec, spec.worker_execution.method_contract.phase_program


def _slots(model, carry, key):
    return {
        "model": (model, carry),
        "optimizer": (),
        "prng": key,
        "batch_counter": jnp.asarray(1, dtype=jnp.int32),
    }


def _continue(slots):
    model, carry = slots["model"]
    inputs = jnp.array([[0.0, 0.0], [1.5, -2.5], [0.5, -0.75]], dtype=jnp.float32)
    outputs, final_carry, _ = iterate_component(
        model, dict.fromkeys(_NOISE_MODELS, inputs), carry, len(inputs), slots["prng"]
    )
    return outputs, final_carry


def _resume(root):
    model, spec, program = _setup()
    fresh_key = jax.random.PRNGKey(88)
    loaded = load_latest_checkpoint(
        root / "checkpoint",
        expected_run_spec=spec,
        expected_phase_program=program,
        expected_slots=_slots(model, model.init_state(key=fresh_key), fresh_key),
    )
    carry = loaded.slots["model"][1]
    actual = jax.tree.leaves((carry, _continue(loaded.slots)))
    with np.load(root / "uninterrupted.npz") as expected:
        assert len(actual) == len(expected.files)
        for index, array in enumerate(actual):
            wanted = expected[f"arr_{index}"]
            actual_array = np.asarray(array)
            assert actual_array.shape == wanted.shape
            assert actual_array.dtype == wanted.dtype
            assert actual_array.tobytes() == wanted.tobytes(), f"continuation leaf {index}"


def test_registered_channel_checkpoint_continues_in_fresh_process(tmp_path):
    model, spec, program = _setup()
    prefix = jnp.array([[-0.5, 1.25], [-2.0, 3.0]], dtype=jnp.float32)
    prefix_key, continuation_key = jax.random.split(jax.random.PRNGKey(4))
    _, carry, _ = iterate_component(
        model,
        dict.fromkeys(_NOISE_MODELS, prefix),
        model.init_state(key=prefix_key),
        len(prefix),
        prefix_key,
    )
    slots = _slots(model, carry, continuation_key)
    outputs, final_carry = _continue(slots)

    np.testing.assert_array_equal(outputs["none"][0], prefix[-1])
    np.testing.assert_array_equal(outputs["signal_dependent_gaussian"][1], [0.0, 0.0])
    for noise in _NOISE_MODELS[:-1]:
        assert not np.array_equal(outputs[noise][0], outputs["none"][0])
    for noise in ("signal_dependent_plus_additive", "additive_gaussian"):
        assert np.any(outputs[noise][1] != 0)
    np.savez(tmp_path / "uninterrupted.npz", *jax.tree.leaves((carry, (outputs, final_carry))))

    write_checkpoint_transaction(
        tmp_path / "checkpoint",
        run_spec=spec,
        phase_program=program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="run-2",
            phase="train_batch",
            program_step=1,
            completed_barrier="after_train_batch",
        ),
        slots=slots,
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
import sys
from scripts.full_suite import configure_native_threads
configure_native_threads()
from tests.test_noise_checkpoint_continuation import _resume
_resume(Path(sys.argv[1]))
""",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert process.returncode == 0, process.stdout + process.stderr
