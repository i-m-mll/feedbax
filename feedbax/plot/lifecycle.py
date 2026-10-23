"""Figure lifecycle helpers."""

from typing import Any


def close_figure(fig: Any) -> None:
    """Close Matplotlib figures; leave non-Matplotlib figures untouched."""
    try:
        import matplotlib.figure as mplfig
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if isinstance(fig, mplfig.Figure):
        plt.close(fig)
