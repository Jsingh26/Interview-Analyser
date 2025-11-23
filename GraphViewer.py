from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QDialog, QSlider, QGroupBox, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import io
import pandas as pd


class GraphViewer(QDialog):
    """Dialog for viewing individual graphs with zoom and pan functionality"""

    def __init__(self, figure, parent=None):
        super().__init__(parent)
        self.figure = figure
        self.canvas = None
        self.init_ui()
        self.setWindowTitle("Graph Viewer")
        self.resize(800, 600)

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Emotion Analysis Visualization")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Graph display area
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # Controls section
        controls_widget = self.create_controls_widget()
        main_layout.addWidget(controls_widget)

        # Action buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(close_btn)

        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

    def create_controls_widget(self):
        """Create widget with graph controls"""
        widget = QGroupBox("Graph Controls")
        layout = QHBoxLayout()

        # Zoom controls
        zoom_label = QLabel("Zoom:")
        layout.addWidget(zoom_label)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(50)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(25)
        self.zoom_slider.valueChanged.connect(self.zoom_graph)
        layout.addWidget(self.zoom_slider)

        self.zoom_label = QLabel("100%")
        layout.addWidget(self.zoom_label)

        # Reset button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        layout.addWidget(reset_btn)

        widget.setLayout(layout)
        return widget

    def zoom_graph(self, value):
        """Zoom the graph based on slider value"""
        self.zoom_label.setText(f"{value}%")

        # Get current figure size
        width, height = self.figure.get_size_inches()

        # Calculate new size based on zoom percentage
        new_width = width * (value / 100.0)
        new_height = height * (value / 100.0)

        # Update figure size
        self.figure.set_size_inches(new_width, new_height)

        # Redraw canvas
        self.canvas.draw()

    def reset_view(self):
        """Reset graph to original size"""
        self.zoom_slider.setValue(100)
        self.figure.set_size_inches(8, 6)  # Default matplotlib size
        self.canvas.draw()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key.Key_Escape:
            self.accept()
        elif event.key() == Qt.Key.Key_R:
            self.reset_view()
        super().keyPressEvent(event)
