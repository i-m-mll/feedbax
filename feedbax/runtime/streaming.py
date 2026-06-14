"""Small helpers shared by streaming scan paths."""

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt


def init_streaming_state_window(state_view, order: int):
    """Create a rolling state window initialized from one state view."""
    if order < 1:
        return state_view

    def _init_leaf(x):
        if eqx.is_array(x):
            return jnp.stack([x] * (order + 1), axis=0)
        return x

    return jt.map(_init_leaf, state_view)


def update_streaming_state_window(window, state_view):
    """Append ``state_view`` to a rolling state window."""

    def _update_leaf(w, x):
        if eqx.is_array(w) and eqx.is_array(x):
            return jnp.concatenate([w[1:], x[None]], axis=0)
        return x

    return jt.map(_update_leaf, window, state_view)


def current_state_from_window(window):
    """Return the newest state view from a rolling state window."""

    def _current_leaf(x):
        if eqx.is_array(x):
            return x[-1]
        return x

    return jt.map(_current_leaf, window)
