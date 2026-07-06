"""Figure lifecycle helpers."""

from typing import Any

import matplotlib.figure as mplfig
import matplotlib.pyplot as plt


def close_figure(fig: Any) -> None:
    """Close Matplotlib figures; leave non-Matplotlib figures untouched."""
    if isinstance(fig, mplfig.Figure):
        plt.close(fig)
