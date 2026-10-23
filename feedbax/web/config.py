from pathlib import Path
import os

BASE_DIR = Path(os.getenv('FEEDBAX_WEB_DATA', Path.home() / '.feedbax' / 'web'))
GRAPHS_DIR = BASE_DIR / 'graphs'
TRAJECTORIES_DIR = Path(
    os.getenv('FEEDBAX_TRAJECTORIES_DIR', Path.home() / '.feedbax' / 'trajectories')
)


def ensure_dirs() -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
