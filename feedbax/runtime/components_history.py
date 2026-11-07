"""Frozen historical wire endpoints for native component owners."""

from feedbax.component_registry.history import Change, History, LegacyEndpoint

from .components import (
    GRU,
    LSTM,
    MLP,
    Activation,
    Constant,
    Damper,
    DeadZone,
    DelayLine,
    Demux,
    ElementwiseAffineModulator,
    Gain,
    Input,
    Linear,
    MatMul,
    Multiply,
    Mux,
    Noise,
    Pulse,
    Ramp,
    RateLimiter,
    Ravel,
    Reshape,
    Saturation,
    Scale,
    Sigmoid,
    Sine,
    Spring,
    Subtract,
    Sum,
    Switch,
)


def _linear_v1(params):
    return {**params, "initialization": "random"}


HISTORIES = {
    Gain: History(
        legacy=(LegacyEndpoint('Gain', '1', 1), LegacyEndpoint('Gain', None, 1)),
    ),
    Input: History(
        legacy=(LegacyEndpoint('Input', '1', 1), LegacyEndpoint('Input', None, 1)),
    ),
    Sum: History(
        legacy=(LegacyEndpoint('Sum', '1', 1), LegacyEndpoint('Sum', None, 1)),
    ),
    Subtract: History(
        legacy=(LegacyEndpoint('Subtract', '1', 1), LegacyEndpoint('Subtract', None, 1)),
    ),
    Multiply: History(
        legacy=(LegacyEndpoint('Multiply', '1', 1), LegacyEndpoint('Multiply', None, 1)),
    ),
    Reshape: History(
        legacy=(LegacyEndpoint('Reshape', '1', 1), LegacyEndpoint('Reshape', None, 1)),
    ),
    MatMul: History(
        legacy=(LegacyEndpoint('MatMul', '1', 1), LegacyEndpoint('MatMul', None, 1)),
    ),
    Scale: History(
        legacy=(LegacyEndpoint('Scale', '1', 1), LegacyEndpoint('Scale', None, 1)),
    ),
    Sigmoid: History(
        legacy=(LegacyEndpoint('Sigmoid', '1', 1), LegacyEndpoint('Sigmoid', None, 1)),
    ),
    Activation: History(
        legacy=(LegacyEndpoint('Activation', '1', 1), LegacyEndpoint('Activation', None, 1)),
    ),
    ElementwiseAffineModulator: History(
        legacy=(LegacyEndpoint('ElementwiseAffineModulator', '1', 1), LegacyEndpoint('ElementwiseAffineModulator', None, 1)),
    ),
    Saturation: History(
        legacy=(LegacyEndpoint('Saturation', '1', 1), LegacyEndpoint('Saturation', None, 1)),
    ),
    Constant: History(
        legacy=(LegacyEndpoint('Constant', '1', 1), LegacyEndpoint('Constant', None, 1)),
    ),
    Ravel: History(
        legacy=(LegacyEndpoint('Ravel', '1', 1), LegacyEndpoint('Ravel', None, 1)),
    ),
    Ramp: History(
        legacy=(LegacyEndpoint('Ramp', '1', 1), LegacyEndpoint('Ramp', None, 1)),
    ),
    Sine: History(
        legacy=(LegacyEndpoint('Sine', '1', 1), LegacyEndpoint('Sine', None, 1)),
    ),
    Pulse: History(
        legacy=(LegacyEndpoint('Pulse', '1', 1), LegacyEndpoint('Pulse', None, 1)),
    ),
    Noise: History(
        legacy=(LegacyEndpoint('Noise', '1', 1), LegacyEndpoint('Noise', None, 1)),
    ),
    Spring: History(
        legacy=(LegacyEndpoint('Spring', '1', 1), LegacyEndpoint('Spring', None, 1)),
    ),
    Damper: History(
        legacy=(LegacyEndpoint('Damper', '1', 1), LegacyEndpoint('Damper', None, 1)),
    ),
    DelayLine: History(
        Change(2, identity=True),
        legacy=(
            LegacyEndpoint("DelayLine", "1", 1),
            LegacyEndpoint("DelayLine", "2", 2),
        ),
    ),
    Linear: History(
        Change(2, migrate=_linear_v1),
        legacy=(
            LegacyEndpoint("Linear", "1", 1),
            LegacyEndpoint("Linear", "2", 2),
        ),
    ),
    MLP: History(
        legacy=(LegacyEndpoint('MLP', '1', 1), LegacyEndpoint('MLP', None, 1)),
    ),
    GRU: History(
        legacy=(LegacyEndpoint('GRU', '1', 1), LegacyEndpoint('GRU', None, 1)),
    ),
    LSTM: History(
        legacy=(LegacyEndpoint('LSTM', '1', 1), LegacyEndpoint('LSTM', None, 1)),
    ),
    Mux: History(
        legacy=(LegacyEndpoint('Mux', '1', 1), LegacyEndpoint('Mux', None, 1)),
    ),
    Demux: History(
        legacy=(LegacyEndpoint('Demux', '1', 1), LegacyEndpoint('Demux', None, 1)),
    ),
    Switch: History(
        legacy=(LegacyEndpoint('Switch', '1', 1), LegacyEndpoint('Switch', None, 1)),
    ),
    DeadZone: History(
        legacy=(LegacyEndpoint('DeadZone', '1', 1), LegacyEndpoint('DeadZone', None, 1)),
    ),
    RateLimiter: History(
        legacy=(LegacyEndpoint('RateLimiter', '1', 1), LegacyEndpoint('RateLimiter', None, 1)),
    ),
}
