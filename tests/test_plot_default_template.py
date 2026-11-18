"""The CLI leaves Plotly initialization to commands that need it."""

import json
from pathlib import Path
import subprocess
import sys
from tests.support.hang import HANG_SECONDS


def _run_probe(source: str) -> dict:
    """Run *source* in a fresh interpreter and return the JSON object it prints."""
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
        timeout=HANG_SECONDS,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_cli_entry_point_leaves_plotly_default_untouched() -> None:
    probe = _run_probe(
        "import json, plotly.io as pio\n"
        "import feedbax.cli  # noqa: F401\n"
        "import sys\n"
        "print(json.dumps({\n"
        "    'default': pio.templates.default,\n"
        "    'plot_imported': 'feedbax.plot' in sys.modules,\n"
        "}))\n"
    )
    assert probe["plot_imported"] is False
    assert probe["default"] == "plotly"
