"""Behavioral witnesses for exact and environment-only validity evidence."""

from hashlib import sha256
import os
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.computation_validity import (
    ComputationDifferent,
    ComputationEqual,
    ComputationStart,
    ComputationTrace,
    ComputationUnconfirmable,
    DriverChanged,
    EnvironmentCall,
    PrefixChanged,
    PrefixConfirmed,
    PrefixUnavailable,
    ProgramsChanged,
    StartsChanged,
    UnidentifiedCall,
)
from feedbax.validity.computation import (
    argument_signature,
    bounded_program_archive,
    capture_arguments,
    capture_program,
    capture_train_prefix,
    compare_computation,
    compare_environment_prefix,
    pytree_operands,
)


class Objects:
    def __init__(self):
        self.data = {}

    def store(self, data):
        digest = sha256(data).hexdigest()
        self.data[digest] = data
        return ContentObjectRef(
            object_id=f"content-object://sha256/{digest}",
            sha256=digest,
            size_bytes=len(data),
        )

    def read(self, reference):
        return self.data[reference.sha256]


def _trace(traced, args, *, platform="cpu", retain=None):
    arguments = capture_arguments(args)
    reference = traced.lower(lowering_platforms=("cpu",))
    execution = reference if platform == "cpu" else traced.lower(lowering_platforms=(platform,))
    program = capture_program(
        execution,
        slot="slot",
        specialization=arguments.signature,
        execution_platform=platform,
        reference_lowered=reference,
        retain_program=retain,
    )
    return ComputationTrace(
        driver={"name": "test", "revision": 1},
        starts=(ComputationStart(slot="slot", arguments=arguments),),
        programs=(program,),
        completed_steps=5,
    )


def test_private_rename_keeps_exact_computation_but_numeric_edit_and_driver_bump_do_not():
    @jax.jit
    def alpha(x):
        return x * 3

    @jax.jit
    def beta(x):
        return x * 3

    @jax.jit
    def changed(x):
        return x * 4

    args = (jnp.full((2,), 2, dtype=jnp.float32),)
    original = _trace(jax.jit(lambda x: alpha(x) + 7).trace(*args), args)
    renamed = _trace(jax.jit(lambda x: beta(x) + 7).trace(*args), args)
    numerical = _trace(jax.jit(lambda x: changed(x) + 7).trace(*args), args)

    assert isinstance(compare_computation(original, renamed), ComputationEqual)
    assert compare_computation(original, numerical) == ComputationDifferent(
        changes=(ProgramsChanged(slots=("slot",)),)
    )
    bumped = renamed.model_copy(
        update={"driver": renamed.driver.model_copy(update={"revision": 2})}
    )
    assert compare_computation(original, bumped) == ComputationDifferent(
        changes=(DriverChanged(previous=original.driver, present=bumped.driver),)
    )


def test_cpu_reference_does_not_hide_an_execution_platform_edit():
    def lower(factor):
        return jax.jit(
            lambda x: jax.lax.platform_dependent(
                x,
                cpu=lambda y: y + 1,
                cuda=lambda y: y * factor,
            )
        )

    args = (jnp.array([1, 2], dtype=jnp.int32),)
    old = _trace(lower(2).trace(*args), args, platform="cuda")
    new = _trace(lower(3).trace(*args), args, platform="cuda")
    assert old.programs[0].reference == new.programs[0].reference
    assert compare_computation(old, new) == ComputationDifferent(
        changes=(ProgramsChanged(slots=("slot",)),)
    )


def test_unknown_result_callback_cannot_confirm_even_with_same_lowered_program():
    def method(factor):
        return jax.jit(lambda x: jax.pure_callback(lambda value: np.asarray(value) * factor, x, x))

    args = (jnp.array([2, 4], dtype=jnp.int32),)
    former, current = method(2), method(3)
    old = _trace(former.trace(*args), args)
    new = _trace(current.trace(*args), args)
    np.testing.assert_array_equal(former(*args), [4, 8])
    np.testing.assert_array_equal(current(*args), [6, 12])
    assert old.programs[0].execution.sha256 == new.programs[0].execution.sha256
    calls = old.programs[0].execution.calls
    assert len(calls) == 1 and isinstance(calls[0], UnidentifiedCall)
    assert compare_computation(old, new) == ComputationUnconfirmable(
        reason="foreign_call",
        targets=(calls[0].target,),
    )


def test_jaxlib_lapack_call_belongs_to_the_environment():
    args = (jnp.eye(2, dtype=jnp.float32),)
    flow = _trace(jax.jit(jnp.linalg.cholesky).trace(*args), args)
    calls = flow.programs[0].execution.calls
    assert calls and all(isinstance(call, EnvironmentCall) for call in calls)
    assert isinstance(compare_computation(flow, flow), ComputationEqual)


def test_argument_identity_includes_values_tree_weak_type_and_placement():
    key = jax.random.key(4)
    for left, right in (
        ((key,), (jax.random.key(5),)),
        ((jnp.array(-0.0),), (jnp.array(0.0),)),
        ((jnp.array(1.0),), (jnp.asarray(1.0, dtype=jnp.float32),)),
        ((jnp.array([1.0]),), (jax.device_put(jnp.array([1.0]), jax.devices("cpu")[0]),)),
    ):
        assert capture_arguments(left) != capture_arguments(right)
    assert argument_signature((), static=(0.0,)) != argument_signature((), static=(-0.0,))
    value = {"a": (key, 1)}
    arrays, static = pytree_operands(value)
    other_arrays, other_static = pytree_operands({"a": [key, 1]})
    assert capture_arguments(arrays, static=static) != capture_arguments(
        other_arrays, static=other_static
    )
    linear = eqx.nn.Linear(2, 1, key=key)
    same = eqx.nn.Linear(2, 1, key=key)
    selected = jax.tree_util.GetAttrKey("weight")
    arrays, static = pytree_operands((linear, selected))
    other_arrays, other_static = pytree_operands((same, selected))
    assert capture_arguments(arrays, static=static) == capture_arguments(
        other_arrays, static=other_static
    )
    _, other_static = pytree_operands((same, jax.tree_util.GetAttrKey("bias")))
    assert argument_signature(arrays, static=static) != argument_signature(
        arrays, static=other_static
    )


def test_equinox_argument_digest_survives_a_new_worker_hash_seed():
    snippet = """
import equinox as eqx
import jax
from feedbax.validity.computation import capture_arguments, pytree_operands

linear = eqx.nn.Linear(2, 1, key=jax.random.key(7))
arrays, static = pytree_operands(linear)
print(capture_arguments(arrays, static=static).sha256)
"""
    digests = [
        subprocess.check_output(
            [sys.executable, "-c", snippet],
            env={**os.environ, "PYTHONHASHSEED": str(seed), "JAX_PLATFORMS": "cpu"},
            timeout=20,
        ).strip()
        for seed in (19, 271)
    ]
    assert len(digests[0]) == 64 and digests[0] == digests[1]


def test_old_normalization_requires_retained_program_for_exact_comparison():
    objects = Objects()
    args = (jnp.array([4], dtype=jnp.int32),)
    traced = jax.jit(lambda x: x + 2).trace(*args)
    present = _trace(traced, args)
    previous = present.model_copy(update={"normalization_version": 2})
    assert compare_computation(previous, present) == ComputationUnconfirmable(
        reason="normalization",
        targets=("slot",),
    )
    archived = _trace(traced, args, retain=objects.store).model_copy(
        update={"normalization_version": 2}
    )
    assert isinstance(
        compare_computation(archived, present, read_blob=objects.read), ComputationEqual
    )
    assert len(archived.content_references()) == 1
    assert compare_computation(archived, present, read_blob=lambda _: b"wrong object") == (
        ComputationUnconfirmable(reason="normalization", targets=("slot",))
    )


def test_optional_program_archive_deduplicates_without_extending_the_attempt_budget():
    objects = Objects()
    archive = bounded_program_archive(objects.store, byte_budget=8)
    first = archive(b"abcde")
    assert first is not None and archive(b"abcde") == first
    assert archive(b"wxyz") is None
    second = archive(b"abc")
    assert second is not None
    assert set(objects.data) == {first.sha256, second.sha256}


def test_environment_prefix_applies_loss_and_parameter_tolerances_separately():
    objects = Objects()
    args = (jnp.array([1], dtype=jnp.int32),)
    traced = jax.jit(lambda x: x + 2).trace(*args)
    trace = _trace(traced, args)

    def with_prefix(loss=1.0, value=1.0, *, dtype=np.float32, count=5):
        return trace.model_copy(
            update={
                "prefix": capture_train_prefix(
                    np.full(count, loss, dtype=dtype),
                    {
                        "parameter": np.asarray([value], dtype=dtype),
                        "mixed": np.zeros(1, dtype=np.float32),
                    },
                    store_blob=objects.store,
                )
            }
        )

    old = with_prefix()
    stable = with_prefix(loss=1.00002, value=1.0001)
    assert isinstance(
        compare_environment_prefix(old, stable, read_blob=objects.read), PrefixConfirmed
    )
    for changed in (with_prefix(loss=1.00004), with_prefix(value=1.001)):
        result = compare_environment_prefix(old, changed, read_blob=objects.read)
        assert isinstance(result, PrefixChanged) and result.reason == "numerical_drift"
    assert compare_environment_prefix(old, with_prefix(value=np.inf), read_blob=objects.read) == (
        PrefixChanged(reason="nonfinite_outputs")
    )
    short = with_prefix(count=2)
    assert short.prefix is not None and short.prefix.losses == (1.0, 1.0)
    assert compare_environment_prefix(old, short, read_blob=objects.read) == PrefixUnavailable(
        reason="profile"
    )
    assert isinstance(
        compare_environment_prefix(
            with_prefix(loss=0), with_prefix(loss=0), read_blob=objects.read
        ),
        PrefixConfirmed,
    )
    assert (
        compare_environment_prefix(
            with_prefix(dtype=np.float64),
            with_prefix(value=1.0001, dtype=np.float64),
            read_blob=objects.read,
        ).status
        == "different"
    )

    changed_args = (jnp.array([2], dtype=jnp.int32),)
    changed_start = _trace(traced, changed_args).model_copy(update={"prefix": old.prefix})
    assert compare_environment_prefix(old, changed_start, read_blob=objects.read) == PrefixChanged(
        reason="arguments"
    )
    assert compare_computation(old, changed_start) == ComputationDifferent(
        changes=(StartsChanged(slots=("slot",)),)
    )
