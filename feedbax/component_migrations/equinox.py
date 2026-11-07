"""Generated frozen Equinox wire evidence; regenerate with generate_eqx_components.py."""

from functools import partial

from feedbax.component_registry.history import History, LegacyEndpoint
from feedbax.components.equinox import (
    Linear,
    Identity,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    GRUCell,
    LSTMCell,
    LayerNorm,
    RMSNorm,
    GroupNorm,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    MultiheadAttention,
    RotaryPositionalEmbedding,
    Dropout,
)


def _revision_one_fields(params, *, fields):
    return set(params) <= fields


HISTORIES = {
    Linear: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.Linear",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(("in_features", "out_features", "use_bias", "dtype")),
                ),
            ),
        )
    ),
    Identity: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.Identity",
                "1",
                1,
                partial(_revision_one_fields, fields=frozenset(())),
            ),
        )
    ),
    Conv1d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.Conv1d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(
                        (
                            "in_channels",
                            "out_channels",
                            "kernel_size",
                            "stride",
                            "padding",
                            "dilation",
                            "groups",
                            "use_bias",
                            "padding_mode",
                            "dtype",
                        )
                    ),
                ),
            ),
        )
    ),
    Conv2d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.Conv2d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(
                        (
                            "in_channels",
                            "out_channels",
                            "kernel_size",
                            "stride",
                            "padding",
                            "dilation",
                            "groups",
                            "use_bias",
                            "padding_mode",
                            "dtype",
                        )
                    ),
                ),
            ),
        )
    ),
    Conv3d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.Conv3d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(
                        (
                            "in_channels",
                            "out_channels",
                            "kernel_size",
                            "stride",
                            "padding",
                            "dilation",
                            "groups",
                            "use_bias",
                            "padding_mode",
                            "dtype",
                        )
                    ),
                ),
            ),
        )
    ),
    ConvTranspose1d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.ConvTranspose1d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(
                        (
                            "in_channels",
                            "out_channels",
                            "kernel_size",
                            "stride",
                            "output_padding",
                            "padding",
                            "dilation",
                            "groups",
                            "use_bias",
                            "padding_mode",
                            "dtype",
                        )
                    ),
                ),
            ),
        )
    ),
    ConvTranspose2d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.ConvTranspose2d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(
                        (
                            "in_channels",
                            "out_channels",
                            "kernel_size",
                            "stride",
                            "output_padding",
                            "padding",
                            "dilation",
                            "groups",
                            "use_bias",
                            "padding_mode",
                            "dtype",
                        )
                    ),
                ),
            ),
        )
    ),
    ConvTranspose3d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.ConvTranspose3d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(
                        (
                            "in_channels",
                            "out_channels",
                            "kernel_size",
                            "stride",
                            "output_padding",
                            "padding",
                            "dilation",
                            "groups",
                            "use_bias",
                            "padding_mode",
                            "dtype",
                        )
                    ),
                ),
            ),
        )
    ),
    GRUCell: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.GRUCell",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(("input_size", "hidden_size", "use_bias", "dtype")),
                ),
            ),
        )
    ),
    LSTMCell: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.LSTMCell",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(("input_size", "hidden_size", "use_bias", "dtype")),
                ),
            ),
        )
    ),
    LayerNorm: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.LayerNorm",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(
                        ("shape", "eps", "use_weight", "use_bias", "dtype", "elementwise_affine")
                    ),
                ),
            ),
        )
    ),
    RMSNorm: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.RMSNorm",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(("shape", "eps", "use_weight", "use_bias", "dtype")),
                ),
            ),
        )
    ),
    GroupNorm: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.GroupNorm",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(("groups", "channels", "eps", "channelwise_affine", "dtype")),
                ),
            ),
        )
    ),
    MaxPool1d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.MaxPool1d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(("kernel_size", "stride", "padding", "use_ceil")),
                ),
            ),
        )
    ),
    MaxPool2d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.MaxPool2d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(("kernel_size", "stride", "padding", "use_ceil")),
                ),
            ),
        )
    ),
    MaxPool3d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.MaxPool3d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(("kernel_size", "stride", "padding", "use_ceil")),
                ),
            ),
        )
    ),
    AvgPool1d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.AvgPool1d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(("kernel_size", "stride", "padding", "use_ceil")),
                ),
            ),
        )
    ),
    AvgPool2d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.AvgPool2d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(("kernel_size", "stride", "padding", "use_ceil")),
                ),
            ),
        )
    ),
    AvgPool3d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.AvgPool3d",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(("kernel_size", "stride", "padding", "use_ceil")),
                ),
            ),
        )
    ),
    AdaptiveMaxPool1d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.AdaptiveMaxPool1d",
                "1",
                1,
                partial(_revision_one_fields, fields=frozenset(("target_shape",))),
            ),
        )
    ),
    AdaptiveMaxPool2d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.AdaptiveMaxPool2d",
                "1",
                1,
                partial(_revision_one_fields, fields=frozenset(("target_shape",))),
            ),
        )
    ),
    AdaptiveMaxPool3d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.AdaptiveMaxPool3d",
                "1",
                1,
                partial(_revision_one_fields, fields=frozenset(("target_shape",))),
            ),
        )
    ),
    AdaptiveAvgPool1d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.AdaptiveAvgPool1d",
                "1",
                1,
                partial(_revision_one_fields, fields=frozenset(("target_shape",))),
            ),
        )
    ),
    AdaptiveAvgPool2d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.AdaptiveAvgPool2d",
                "1",
                1,
                partial(_revision_one_fields, fields=frozenset(("target_shape",))),
            ),
        )
    ),
    AdaptiveAvgPool3d: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.AdaptiveAvgPool3d",
                "1",
                1,
                partial(_revision_one_fields, fields=frozenset(("target_shape",))),
            ),
        )
    ),
    MultiheadAttention: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.MultiheadAttention",
                "1",
                1,
                partial(
                    _revision_one_fields,
                    fields=frozenset(
                        (
                            "num_heads",
                            "query_size",
                            "key_size",
                            "value_size",
                            "output_size",
                            "qk_size",
                            "vo_size",
                            "use_query_bias",
                            "use_key_bias",
                            "use_value_bias",
                            "use_output_bias",
                            "dropout_p",
                            "inference",
                            "dtype",
                        )
                    ),
                ),
            ),
        )
    ),
    RotaryPositionalEmbedding: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.RotaryPositionalEmbedding",
                "1",
                1,
                partial(
                    _revision_one_fields, fields=frozenset(("embedding_size", "theta", "dtype"))
                ),
            ),
        )
    ),
    Dropout: History(
        legacy=(
            LegacyEndpoint(
                "feedbax.equinox.Dropout",
                "1",
                1,
                partial(
                    _revision_one_fields, fields=frozenset(("p", "inference", "deterministic"))
                ),
            ),
        )
    ),
}
