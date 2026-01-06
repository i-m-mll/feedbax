"""Preset management for filter configurations."""

import json
import logging
from pathlib import Path
from typing import Any

from feedbax._experiments.config import PATHS

logger = logging.getLogger(__name__)


class PresetManager:
    """Manage filter preset saving and loading."""

    def __init__(self):
        """Initialize preset manager."""
        self.presets_dir = PATHS.db / "dashboard_presets"
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def save_preset(self, name: str, filters: list[dict[str, Any]]) -> bool:
        """Save a filter preset.

        Args:
            name: Preset name
            filters: List of filter dictionaries

        Returns:
            True if successful, False otherwise
        """
        try:
            preset_path = self.presets_dir / f"{name}.json"
            with open(preset_path, "w") as f:
                json.dump({"name": name, "filters": filters}, f, indent=2)
            logger.info(f"Saved preset '{name}' to {preset_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save preset '{name}': {e}")
            return False

    def load_preset(self, name: str) -> list[dict[str, Any]] | None:
        """Load a filter preset.

        Args:
            name: Preset name

        Returns:
            List of filter dictionaries, or None if not found
        """
        try:
            preset_path = self.presets_dir / f"{name}.json"
            if not preset_path.exists():
                logger.warning(f"Preset '{name}' not found at {preset_path}")
                return None

            with open(preset_path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded preset '{name}' from {preset_path}")
            return data.get("filters", [])
        except Exception as e:
            logger.error(f"Failed to load preset '{name}': {e}")
            return None

    def delete_preset(self, name: str) -> bool:
        """Delete a filter preset.

        Args:
            name: Preset name

        Returns:
            True if successful, False otherwise
        """
        try:
            preset_path = self.presets_dir / f"{name}.json"
            if preset_path.exists():
                preset_path.unlink()
                logger.info(f"Deleted preset '{name}'")
                return True
            else:
                logger.warning(f"Preset '{name}' not found")
                return False
        except Exception as e:
            logger.error(f"Failed to delete preset '{name}': {e}")
            return False

    def list_presets(self) -> list[str]:
        """Get list of available preset names.

        Returns:
            Sorted list of preset names
        """
        try:
            preset_files = self.presets_dir.glob("*.json")
            names = [p.stem for p in preset_files]
            return sorted(names)
        except Exception as e:
            logger.error(f"Failed to list presets: {e}")
            return []
