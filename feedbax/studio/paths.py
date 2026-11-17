"""Paths shared by Studio and its execution entry points."""

import os
from pathlib import Path


STUDIO_DATA_ROOT = Path(os.getenv("FEEDBAX_WEB_DATA", Path.home() / ".feedbax" / "web"))
