"""The Feedbax Plotly template default is applied on first use, not on import.

Assigning `plotly.io.templates.default` loads Plotly's bundled template data and costs
well over a hundred milliseconds. `feedbax.plot` used to do that at module scope, so
every console script that transitively imports the package paid for a plotting global it
may never touch. These tests pin both halves of the contract: importing the package (or
an entry point that reaches it) must not mutate Plotly's process-global default, and
anything that actually draws must still get the Feedbax default.

Each case runs in its own interpreter, because the assertion is about what a fresh
process does at import time and because the state under test is process-global.
"""

import json
from pathlib import Path
import subprocess
import sys


def _run_probe(source: str) -> dict:
    """Run *source* in a fresh interpreter and return the JSON object it prints."""
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_orchestration_entry_point_leaves_plotly_default_untouched() -> None:
    probe = _run_probe(
        "import json, plotly.io as pio\n"
        "import feedbax.bin.orchestrate  # noqa: F401\n"
        "import sys\n"
        "print(json.dumps({\n"
        "    'default': pio.templates.default,\n"
        "    'plot_imported': 'feedbax.plot' in sys.modules,\n"
        "}))\n"
    )
    # The package is reached transitively, and still leaves Plotly's own default in place.
    assert probe["plot_imported"] is True
    assert probe["default"] == "plotly"


def test_public_attribute_access_applies_the_feedbax_template() -> None:
    probe = _run_probe(
        "import json, plotly.io as pio\n"
        "import feedbax.plot as fbp\n"
        "before = pio.templates.default\n"
        "fbp.trajectories\n"
        "print(json.dumps({\n"
        "    'before': before,\n"
        "    'after': pio.templates.default,\n"
        "    'declared': fbp.DEFAULT_TEMPLATE,\n"
        "}))\n"
    )
    assert probe["before"] == "plotly"
    assert probe["after"] == "plotly_white"
    assert probe["declared"] == "plotly_white"


def test_importing_a_figure_drawing_module_applies_the_feedbax_template() -> None:
    probe = _run_probe(
        "import json, plotly.io as pio\n"
        "import feedbax.plot.trajectories  # noqa: F401\n"
        "print(json.dumps({'default': pio.templates.default}))\n"
    )
    assert probe["default"] == "plotly_white"


def test_a_deliberately_chosen_template_survives_a_later_plot_import() -> None:
    """Deferral must not let a lazy plot import overwrite a caller's own choice.

    `feedbax-analysis` selects a template inside `main()`, and figure-drawing modules are
    imported after that. An import-time assignment could never have raced it; a deferred
    one could, so the deferral yields to any non-Plotly default already in place.
    """
    probe = _run_probe(
        "import json, plotly.io as pio\n"
        "pio.templates.default = 'simple_white'\n"
        "import feedbax.plot.trajectories  # noqa: F401\n"
        "import feedbax.plot as fbp\n"
        "fbp.trajectories\n"
        "print(json.dumps({'default': pio.templates.default}))\n"
    )
    assert probe["default"] == "simple_white"


def test_default_colors_resolve_against_the_feedbax_template() -> None:
    probe = _run_probe(
        "import json, plotly.io as pio\n"
        "from feedbax.plot.colors import DEFAULT_COLORS\n"
        "print(json.dumps({\n"
        "    'default': pio.templates.default,\n"
        "    'colors': list(DEFAULT_COLORS),\n"
        "    'expected': list(pio.templates['plotly_white'].layout.colorway),\n"
        "}))\n"
    )
    assert probe["default"] == "plotly_white"
    assert probe["colors"] == probe["expected"]


def test_constructed_figures_still_carry_the_feedbax_template() -> None:
    """A figure built through the registry dispatch inherits `plotly_white`.

    Plotly copies the process-default template into a figure as it is constructed, so
    deferring the assignment past figure construction would silently change every
    rendered figure. This is the behavioural guard on that.
    """
    probe = _run_probe(
        "import json, plotly.io as pio\n"
        "from feedbax.plot.constructors import FigureRegistry, get_figure_constructor\n"
        "from feedbax.plot.constructors import register_default_figure_constructors\n"
        "registry = FigureRegistry()\n"
        "register_default_figure_constructors(registry)\n"
        "before = pio.templates.default\n"
        "key = sorted(r.key for r in registry.constructors())[0]\n"
        "get_figure_constructor(key, registry=registry)\n"
        "import plotly.graph_objs as go\n"
        "figure = go.Figure()\n"
        "print(json.dumps({\n"
        "    'before': before,\n"
        "    'after': pio.templates.default,\n"
        "    'figure_bgcolor': figure.layout.template.layout.plot_bgcolor,\n"
        "    'expected_bgcolor': "
        "pio.templates['plotly_white'].layout.plot_bgcolor,\n"
        "}))\n"
    )
    assert probe["before"] == "plotly"
    assert probe["after"] == "plotly_white"
    assert probe["figure_bgcolor"] == probe["expected_bgcolor"]
