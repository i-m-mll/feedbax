from pathlib import Path
import os

from feedbax.studio.paths import STUDIO_DATA_ROOT as BASE_DIR
GRAPHS_DIR = BASE_DIR / 'graphs'
TRAJECTORIES_DIR = Path(
    os.getenv('FEEDBAX_TRAJECTORIES_DIR', Path.home() / '.feedbax' / 'trajectories')
)


def ensure_dirs() -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
