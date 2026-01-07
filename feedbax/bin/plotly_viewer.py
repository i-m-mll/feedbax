#!/usr/bin/env python3
"""Written with the help of Claude 3.5 Sonnet"""

import json
import os
import sys
from pathlib import Path

from PyQt5.QtCore import QSize, Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ruamel.yaml import YAML

from feedbax.misc import get_md5_hexdigest

# Minimal HTML template
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <script defer src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body{margin:0;padding:0;}</style>
</head>
<body>
    <div id="plot"></div>
    <script>
        window.addEventListener('load', function() {
            Plotly.newPlot('plot', %s, %s, {responsive: true});
        });
    </script>
</body>
</html>
"""


yaml = YAML(typ="safe")
yaml.default_flow_style = None


class PlotlyViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plotly Viewer")
        self.setGeometry(100, 100, 800, 600)  # Start with normal width

        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create splitter for plot and metadata
        self.splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(self.splitter)

        # Create web view
        self.web_view = QWebEngineView()
        self.splitter.addWidget(self.web_view)

        # Create metadata panel
        self.metadata_widget = QWidget()
        self.metadata_layout = QVBoxLayout(self.metadata_widget)

        # Add YAML content group
        self.yaml_group = QGroupBox("Hyperparameters")
        self.yaml_layout = QVBoxLayout()
        self.yaml_text = QTextEdit()
        self.yaml_text.setReadOnly(True)
        self.yaml_layout.addWidget(self.yaml_text)
        self.yaml_group.setLayout(self.yaml_layout)
        self.metadata_layout.addWidget(self.yaml_group)

        self.splitter.addWidget(self.metadata_widget)
        self.splitter.setSizes([700, 300])  # Default split ratio

        # Initially hide the metadata panel
        self.metadata_widget.setVisible(False)
        self.metadata_visible = False
        self.original_width = 800

        # Connect signal for window resizing
        self.web_view.loadFinished.connect(self.adjust_window_size)

        # Initialize cache directory
        self.cache_dir = Path.home() / ".plotly_viewer_cache"
        self.cache_dir.mkdir(exist_ok=True)

        self.fig_width = None
        self.fig_height = None

        self.zoom_factor = 1.0

    def get_cache_path(self, content):
        """Generate a cache file path based on content hash"""
        content_hash = get_md5_hexdigest(content)
        return self.cache_dir / f"{content_hash}.html"

    def create_html(self, fig_dict):
        """Create HTML content from figure dictionary"""
        data = fig_dict.get("data", [])
        layout = fig_dict.get("layout", {})

        if "margin" not in layout:
            layout["margin"] = dict(l=10, r=10, t=30, b=10)

        return TEMPLATE % (json.dumps(data), json.dumps(layout))

    def adjust_window_size(self, ok):
        if ok:
            if self.fig_width is not None and self.fig_height is not None:
                self.resize_window([self.fig_width, self.fig_height])
            else:
                js = """
                var rect = document.querySelector('.plotly').getBoundingClientRect();
                [rect.width, rect.height];
                """
                self.web_view.page().runJavaScript(js, self.resize_window)

            # Store the width for later adjustments when showing/hiding metadata
            if self.isVisible() and not self.metadata_visible:
                self.original_width = self.width()

    def toggle_metadata_panel(self, show):
        """Toggle the metadata panel and resize window accordingly"""
        self.metadata_visible = show
        self.metadata_widget.setVisible(show)

        # When showing metadata, extend window width
        if self.isVisible():  # Only adjust if window is already visible
            current_geom = self.geometry()
            metadata_width = 300 if show else 0

            if show:
                # Extend window width to show metadata
                new_width = self.original_width + metadata_width
                self.setGeometry(
                    current_geom.x(), current_geom.y(), new_width, current_geom.height()
                )
                # Set splitter sizes to give most space to the plot
                self.splitter.setSizes([self.original_width - 50, metadata_width - 50])
            else:
                # Reduce window width when hiding metadata
                self.setGeometry(
                    current_geom.x(), current_geom.y(), self.original_width, current_geom.height()
                )

    def resize_window(self, dimensions):
        if dimensions:
            width, height = dimensions
            width += 40
            height += 80

            cursor_pos = QApplication.desktop().cursor().pos()
            screen = QApplication.desktop().screenGeometry(cursor_pos)

            MIN_WIDTH, MIN_HEIGHT = 400, 300
            max_width = int(screen.width() * 0.9)
            max_height = int(screen.height() * 0.9)

            width = max(MIN_WIDTH, min(width, max_width))
            height = max(MIN_HEIGHT, min(height, max_height))

            # Save this as our base width without metadata
            self.original_width = width

            # If metadata is visible, add its width
            if self.metadata_visible:
                width += 300

            x = screen.x() + (screen.width() - width) // 2
            y = screen.y() + (screen.height() - height) // 2

            self.setGeometry(x, y, width, height)
            if not self.isVisible():
                self.show()

            # Adjust splitter if metadata visible
            if self.metadata_visible:
                self.splitter.setSizes([self.original_width - 50, 300 - 50])

    def load_yaml(self, yaml_path):
        """Load and display associated YAML metadata if it exists"""
        try:
            if yaml_path.exists():
                with open(yaml_path, "r") as f:
                    yaml_content = yaml.load(f)

                # Format YAML content nicely
                formatted_yaml = yaml.dump(yaml_content)
                self.yaml_text.setText(formatted_yaml)

                # Content loaded but don't show yet (will be shown by toggle_metadata_panel)
                return True
            else:
                self.toggle_metadata_panel(False)
                return False
        except Exception as e:
            print(f"Error loading YAML: {str(e)}")
            self.toggle_metadata_panel(False)
            return False

    def load_plot(self, filename):
        try:
            with open(filename, "r") as f:
                fig_dict = json.load(f)

            try:
                layout = fig_dict.get("layout", {})
                self.fig_width = layout.get("width")
                self.fig_height = layout.get("height")
            except:
                self.fig_width = None
                self.fig_height = None

            cache_path = self.get_cache_path(fig_dict)

            if not cache_path.exists():
                html_content = self.create_html(fig_dict)
                with open(cache_path, "w") as f:
                    f.write(html_content)

            url = QUrl.fromLocalFile(str(cache_path.absolute()))
            self.web_view.setUrl(url)

            # Check for associated YAML file
            yaml_path = Path(filename).with_suffix(".yml")
            if not yaml_path.exists():
                yaml_path = Path(filename).with_suffix(".yaml")

            # Load and display YAML if exists
            has_yaml = self.load_yaml(yaml_path)

            base_filename = os.path.basename(filename)
            if has_yaml:
                self.setWindowTitle(f"Plotly Viewer - {base_filename} (with metadata)")
                # Extend window width to show metadata panel
                self.toggle_metadata_panel(True)
            else:
                self.setWindowTitle(f"Plotly Viewer - {base_filename}")

        except json.JSONDecodeError:
            QMessageBox.critical(self, "Error", "Invalid JSON file")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading plot: {str(e)}")

    # Zoom in/out with ctrl++/ctrl+-
    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
                self.zoom_factor *= 1.25
                self.web_view.setZoomFactor(self.zoom_factor)
            elif event.key() == Qt.Key_Minus:
                self.zoom_factor *= 0.8
                self.web_view.setZoomFactor(self.zoom_factor)
            elif event.key() == Qt.Key_0:
                self.zoom_factor = 1.0
                self.web_view.setZoomFactor(self.zoom_factor)
            elif event.key() == Qt.Key_M:
                # Toggle metadata panel visibility with Ctrl+M
                self.toggle_metadata_panel(not self.metadata_visible)


def main():
    app = QApplication(sys.argv)
    viewer = PlotlyViewer()

    if len(sys.argv) > 1:
        viewer.load_plot(sys.argv[1])

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
