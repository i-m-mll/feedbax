import jax
import jax.numpy as jnp

from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.contracts.expression_values import TIME, Level, ScopedValue, ValueBase, ValueType
from feedbax.contracts.window import Window
from feedbax.expression_compile import Inputs
from feedbax.runtime.epoch_windows import CompiledWindow, EpochRecord, compile_window


def _compile(spans, epochs, names=()):
    window = compile_window(
        Window.model_validate({"spans": spans}),
        epochs=epochs, period=0.01, names=names, functions=FunctionRegistry(),
        field_address=("fixture", "window"), path=("window",),
    )
    assert isinstance(window, CompiledWindow), window
    return window


def test_epoch_windows_preserve_leading_trial_axes():
    starts = jnp.array([[[0, 1], [0, 3]], [[0, 0], [0, 2]]])
    ends = jnp.array([[[1, 4], [3, 4]], [[0, 3], [2, 4]]])
    epochs = EpochRecord(starts=starts, ends=ends, outcomes=jnp.zeros_like(starts))
    window = _compile([{"kind": "epoch", "epoch": "move"}], ("go", "move"))

    def movement(record):
        return window.mask(record, 4, Inputs((), {}, None, None))

    expected = jnp.array([
        [[False, True, True, True], [False, False, False, True]],
        [[True, True, True, False], [False, False, True, True]],
    ])
    actual = jax.jit(movement)(epochs)
    assert jnp.array_equal(actual, expected)
    assert jnp.array_equal(actual, jax.vmap(jax.vmap(movement))(epochs))


def test_half_open_seconds_and_last_epoch_sample_follow_each_trial():
    starts = jnp.array([[3, 38], [7, 39], [-1, -1], [2, 2]])
    ends = jnp.array([[38, 45], [39, 41], [-1, -1], [2, 2]])
    record = EpochRecord(starts, ends, jnp.zeros_like(starts))
    go = {"kind": "epoch", "epoch": "go", "edge": "start"}
    hold = {"kind": "epoch", "epoch": "hold", "edge": "end"}
    interval = _compile([{
        "kind": "interval",
        "start": {"anchor": go, "offset": {"value": .1, "unit": "s"}},
        "end": {"anchor": go, "offset": {"value": .3, "unit": "s"}},
    }], ("go", "hold"))
    moment = _compile([{"kind": "moment", "at": {"anchor": hold}}], ("go", "hold"))

    inputs = Inputs((), {}, None, None)
    assert [jnp.flatnonzero(mask).tolist() for mask in interval.mask(record, 45, inputs)] == [
        list(range(13, 33)), list(range(17, 37)), [], list(range(12, 32)),
    ]
    assert [jnp.flatnonzero(mask).tolist() for mask in moment.mask(record, 45, inputs)] == [
        [44], [40], [], [],
    ]


def test_repeated_epoch_relative_windows_read_a_trial_variable_and_can_select_one_occurrence():
    starts = jnp.array([2, 12])
    ends = jnp.array([7, 18])
    record = EpochRecord(starts, ends, jnp.zeros_like(starts))
    length = ScopedValue(
        id="length", name="length", type=ValueType(base=ValueBase.REAL, dimensions=TIME), level=Level.TRIAL,
    )
    window = _compile([
        {
            "kind": "interval",
            "start": {
                "anchor": {"kind": "epoch", "epoch": "hold", "edge": "start"},
                "offset": {"value": -1, "unit": "st"},
            },
            "end": {"offset": {"kind": "name", "reference": "length"}},
        },
        {"kind": "epoch", "epoch": "hold", "occurrence": 1},
    ], ("hold", "hold"), (length,))

    chosen = window.mask(record, 20, Inputs((), {"length": jnp.asarray(.02)}, None, None))
    assert jnp.flatnonzero(chosen).tolist() == [1, 2, 11, 12, 13, 14, 15, 16, 17]
