"""Figure lifecycle helpers."""

import sys
from typing import Any


def is_matplotlib_figure(fig: Any) -> bool:
    """Recognize Matplotlib figures without loading Matplotlib."""
    # An instance (including a subclass) requires matplotlib.figure to be loaded.
    figure_type = getattr(sys.modules.get("matplotlib.figure"), "Figure", None)
    return figure_type is not None and isinstance(fig, figure_type)


def close_figure(fig: Any) -> None:
    """Close Matplotlib figures; leave non-Matplotlib figures untouched."""
    if is_matplotlib_figure(fig):
        import matplotlib.pyplot as plt

        plt.close(fig)
