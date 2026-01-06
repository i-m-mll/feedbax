"""Figure loading utilities for the dashboard."""

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from feedbax._experiments.config import PATHS
from feedbax._experiments.database import FigureRecord

logger = logging.getLogger(__name__)


@dataclass
class FigureData:
    """Container for loaded figure data."""

    type: Literal["png", "plotly"]
    content: str | dict  # base64 string for images, dict for Plotly JSON
    path: Path


class FigureLoader:
    """Load figures from filesystem based on database records."""

    def load_figure(
        self,
        figure_record: FigureRecord,
        format: str = "png",
    ) -> Optional[FigureData]:
        """Load figure and return appropriate data for Dash component.

        Args:
            figure_record: FigureRecord from database
            format: File format to load ("png" or "json")

        Returns:
            FigureData object with loaded content, or None if file doesn't exist
        """
        path = self.get_figure_path(figure_record, format)

        if not path.exists():
            logger.warning(f"Figure file not found: {path}")
            return None

        try:
            if format == "png":
                return self._load_png(path)
            elif format == "json":
                return self._load_plotly_json(path)
            else:
                logger.error(f"Unsupported format: {format}")
                return None
        except Exception as e:
            logger.error(f"Error loading figure {path}: {e}")
            return None

    def _load_png(self, path: Path) -> FigureData:
        """Load PNG image as base64-encoded string.

        Args:
            path: Path to PNG file

        Returns:
            FigureData with base64-encoded image
        """
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        return FigureData(
            type="png",
            content=f"data:image/png;base64,{encoded}",
            path=path,
        )

    def _load_plotly_json(self, path: Path) -> FigureData:
        """Load Plotly figure from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            FigureData with Plotly figure dict
        """
        with open(path, "r") as f:
            fig_dict = json.load(f)

        return FigureData(
            type="plotly",
            content=fig_dict,
            path=path,
        )

    @staticmethod
    def get_figure_path(figure_record: FigureRecord, format: str = "png") -> Path:
        """Construct path to figure file.

        Args:
            figure_record: FigureRecord from database
            format: File format extension

        Returns:
            Path to figure file
        """
        # Use the get_path method we added to FigureRecord
        return figure_record.get_path(format)

    def check_figure_exists(self, figure_record: FigureRecord, format: str = "png") -> bool:
        """Check if figure file exists on disk.

        Args:
            figure_record: FigureRecord from database
            format: File format extension

        Returns:
            True if file exists, False otherwise
        """
        path = self.get_figure_path(figure_record, format)
        return path.exists()
