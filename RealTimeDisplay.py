from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QGroupBox, QFrame, QSplitter, QProgressBar, QGraphicsOpacityEffect)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer, QSize
from PyQt6.QtGui import QFont, QPixmap, QImage, QColor, QLinearGradient, QPalette
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
import numpy as np


class RealTimeDisplay(QWidget):
    """Real-time display widget for live emotion analysis"""

    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.current_stats = {}
        self.graph_data = {'timestamps': [], 'confidence': [], 'nervousness': []}
        self.init_ui()

    def init_ui(self):
        """Initialize the real-time display interface"""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Create splitter for 60% camera, 40% stats/graphs
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setSizes([600, 400])  # Initial sizes

        # Left side: Camera feed (60%)
        self.camera_widget = self.create_camera_widget()
        splitter.addWidget(self.camera_widget)

        # Right side: Stats and graphs (40%)
        self.stats_graphs_widget = self.create_stats_graphs_widget()
        splitter.addWidget(self.stats_graphs_widget)

        # Set stretch factors
        splitter.setStretchFactor(0, 6)  # 60% for camera
        splitter.setStretchFactor(1, 4)  # 40% for stats/graphs

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def create_camera_widget(self):
        """Create camera feed display widget"""
        widget = QGroupBox("Live Camera Feed")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # Camera display area
        self.camera_label = QLabel("Camera feed will appear here")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(400, 300)
        self.camera_label.setStyleSheet("""
            QLabel {
                background-color: #000;
                color: #666;
                border: 2px dashed #666;
                border-radius: 10px;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.camera_label)

        # Camera controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        self.start_camera_btn = QPushButton("🎥 Start Analysis")
        self.start_camera_btn.clicked.connect(self.start_analysis)
        self.start_camera_btn.setMinimumHeight(40)
        self.start_camera_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:pressed {
                background-color: #1B5E20;
            }
        """)
        controls_layout.addWidget(self.start_camera_btn)

        self.stop_camera_btn = QPushButton("⏹️ Stop Analysis")
        self.stop_camera_btn.clicked.connect(self.stop_analysis)
        self.stop_camera_btn.setEnabled(False)
        self.stop_camera_btn.setMinimumHeight(40)
        self.stop_camera_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)
        controls_layout.addWidget(self.stop_camera_btn)

        self.reset_btn = QPushButton("🔄 Reset Data")
        self.reset_btn.clicked.connect(self.reset_data)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setMinimumHeight(40)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)
        controls_layout.addWidget(self.reset_btn)

        self.generate_report_btn = QPushButton("📊 Generate Report")
        self.generate_report_btn.clicked.connect(self.generate_report)
        self.generate_report_btn.setEnabled(False)
        self.generate_report_btn.setMinimumHeight(40)
        self.generate_report_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)
        controls_layout.addWidget(self.generate_report_btn)

        layout.addLayout(controls_layout)

        # Status indicator
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #666; font-weight: bold;")
        layout.addWidget(self.status_label)

        widget.setLayout(layout)
        return widget

    def create_stats_graphs_widget(self):
        """Create stats and graphs widget"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Real-time stats cards
        self.stats_widget = self.create_stats_cards()
        layout.addWidget(self.stats_widget)

        # Real-time graph
        self.graph_widget = self.create_graph_widget()
        layout.addWidget(self.graph_widget)

        widget.setLayout(layout)
        return widget

    def create_stats_cards(self):
        """Create real-time statistics cards"""
        widget = QGroupBox("Real-Time Statistics")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Confidence card
        self.confidence_card = self.create_metric_card("💪 Confidence", "50.0%", "#4facfe")
        layout.addWidget(self.confidence_card)

        # Nervousness card
        self.nervousness_card = self.create_metric_card("😰 Nervousness", "50.0%", "#f093fb")
        layout.addWidget(self.nervousness_card)

        # Dominant emotion card
        self.emotion_card = self.create_metric_card("🎭 Dominant Emotion", "Neutral", "#43e97b")
        layout.addWidget(self.emotion_card)

        # Analysis status
        self.analysis_status = QLabel("Analysis: Stopped")
        self.analysis_status.setStyleSheet("""
            QLabel {
                color: #f44336;
                font-weight: bold;
                font-size: 12px;
                padding: 5px;
                background-color: rgba(244, 67, 54, 0.1);
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.analysis_status)

        widget.setLayout(layout)
        return widget

    def create_metric_card(self, title, value, color):
        """Create animated metric card"""
        card = QFrame()
        card.setMinimumHeight(80)
        card.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {color}, stop:1 rgba(255, 255, 255, 0.1));
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 10, 15, 10)

        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        title_label.setStyleSheet("color: white;")
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        value_label.setStyleSheet("color: white;")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_label)

        # Store label for updates
        card.value_label = value_label

        return card

    def create_graph_widget(self):
        """Create real-time graph widget"""
        widget = QGroupBox("Live Confidence vs Nervousness")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure.patch.set_facecolor('#2b2b2b')
        self.canvas = FigureCanvas(self.figure)

        # Create initial plot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')

        self.line_confidence, = self.ax.plot([], [], 'g-', label='Confidence', linewidth=2)
        self.line_nervousness, = self.ax.plot([], [], 'r-', label='Nervousness', linewidth=2)

        self.ax.set_xlim(0, 60)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel('Time (seconds)', color='white')
        self.ax.set_ylabel('Percentage (%)', color='white')
        self.ax.set_title('Real-Time Emotion Tracking', color='white')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        layout.addWidget(self.canvas)

        widget.setLayout(layout)
        return widget

    def set_analyzer(self, analyzer):
        """Set the real-time analyzer instance"""
        self.analyzer = analyzer

        # Connect signals
        if self.analyzer:
            self.analyzer.frame_updated.connect(self.update_camera_feed)
            self.analyzer.stats_updated.connect(self.update_stats)
            self.analyzer.graph_data_updated.connect(self.update_graph)
            self.analyzer.error_occurred.connect(self.handle_error)

    def start_analysis(self):
        """Start real-time analysis"""
        if not self.analyzer:
            from RealTimeAnalyzer import RealTimeAnalyzer
            self.analyzer = RealTimeAnalyzer()
            self.set_analyzer(self.analyzer)

        try:
            self.analyzer.start()
            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.status_label.setText("Status: Analyzing...")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.analysis_status.setText("Analysis: Running")
            self.analysis_status.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 5px;
                    background-color: rgba(76, 175, 80, 0.1);
                    border-radius: 5px;
                }
            """)
        except Exception as e:
            self.handle_error(f"Failed to start analysis: {str(e)}")

    def stop_analysis(self):
        """Stop real-time analysis"""
        if self.analyzer:
            self.analyzer.stop_analysis()

        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self.generate_report_btn.setEnabled(True)  # Enable report generation after stopping
        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
        self.analysis_status.setText("Analysis: Stopped")
        self.analysis_status.setStyleSheet("""
            QLabel {
                color: #f44336;
                font-weight: bold;
                font-size: 12px;
                padding: 5px;
                background-color: rgba(244, 67, 54, 0.1);
                border-radius: 5px;
            }
        """)

        # Show final results summary
        self.show_final_results()

    def reset_data(self):
        """Reset all data and graphs"""
        if self.analyzer:
            self.analyzer.reset_data()

        # Reset graph
        self.line_confidence.set_data([], [])
        self.line_nervousness.set_data([], [])
        self.canvas.draw()

        # Reset stats cards
        self.confidence_card.value_label.setText("50.0%")
        self.nervousness_card.value_label.setText("50.0%")
        self.emotion_card.value_label.setText("Neutral")

        # Reset camera feed
        self.camera_label.setPixmap(QPixmap())
        self.camera_label.setText("Camera feed will appear here")

    def update_camera_feed(self, pixmap):
        """Update camera feed display"""
        if pixmap:
            # Scale pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.camera_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.camera_label.setPixmap(scaled_pixmap)

    def update_stats(self, stats):
        """Update statistics cards"""
        self.current_stats = stats

        # Update confidence
        confidence_val = f"{stats.get('confidence', 50.0):.1f}%"
        self.confidence_card.value_label.setText(confidence_val)

        # Update nervousness
        nervousness_val = f"{stats.get('nervousness', 50.0):.1f}%"
        self.nervousness_card.value_label.setText(nervousness_val)

        # Update dominant emotion
        emotion_val = stats.get('dominant_emotion', 'neutral').capitalize()
        self.emotion_card.value_label.setText(emotion_val)

    def update_graph(self, timestamps, confidence_data, nervousness_data):
        """Update real-time graph"""
        self.graph_data = {
            'timestamps': timestamps,
            'confidence': confidence_data,
            'nervousness': nervousness_data
        }

        # Update plot data
        self.line_confidence.set_data(timestamps, confidence_data)
        self.line_nervousness.set_data(timestamps, nervousness_data)

        # Adjust x-axis limits to show last 60 seconds
        if timestamps:
            max_time = max(timestamps)
            self.ax.set_xlim(max(0, max_time - 60), max_time)

        self.canvas.draw()

    def handle_error(self, error_message):
        """Handle analysis errors"""
        self.status_label.setText(f"Status: Error - {error_message}")
        self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
        self.stop_analysis()

    def generate_report(self):
        """Generate and export a comprehensive report"""
        if not self.analyzer:
            self.status_label.setText("Status: No data available for report")
            return

        try:
            # Generate report
            report = self.analyzer.generate_report()
            if not report:
                self.status_label.setText("Status: No data available for report")
                return

            # Export to CSV
            csv_filename = self.analyzer.export_report_to_csv()
            if csv_filename:
                self.status_label.setText(f"Status: Report exported to {csv_filename}")
                # Show report summary in a message box or status
                self.show_report_summary(report)
            else:
                self.status_label.setText("Status: Failed to export report")

        except Exception as e:
            self.status_label.setText(f"Status: Report generation failed - {str(e)}")

    def show_final_results(self):
        """Show final results summary when analysis is stopped"""
        if not self.analyzer or not self.analyzer.timestamps:
            return

        # Calculate final statistics
        total_frames = len(self.analyzer.timestamps)
        duration = self.analyzer.timestamps[-1] if self.analyzer.timestamps else 0

        avg_confidence = np.mean(self.analyzer.confidence_history) if self.analyzer.confidence_history else 50.0
        avg_nervousness = np.mean(self.analyzer.nervousness_history) if self.analyzer.nervousness_history else 50.0

        # Update status with final summary
        summary_text = f"Final Results: {total_frames} frames, {duration:.1f}s duration, Avg Confidence: {avg_confidence:.1f}%, Avg Nervousness: {avg_nervousness:.1f}%"
        self.status_label.setText(summary_text)
        self.status_label.setStyleSheet("color: #2196F3; font-weight: bold;")

    def show_report_summary(self, report):
        """Show a summary of the generated report"""
        session_info = report.get('session_info', {})
        overall_stats = report.get('overall_statistics', {})

        summary = f"""
Report Generated Successfully!

Session Info:
- Duration: {session_info.get('duration_seconds', 0):.1f} seconds
- Total Frames: {session_info.get('total_frames', 0)}
- Analysis Rate: {session_info.get('analysis_rate_fps', 0):.1f} FPS

Overall Statistics:
- Average Confidence: {overall_stats.get('average_confidence', 0):.1f}%
- Average Nervousness: {overall_stats.get('average_nervousness', 0):.1f}%
- Max Confidence: {overall_stats.get('max_confidence', 0):.1f}%
- Min Confidence: {overall_stats.get('min_confidence', 0):.1f}%

Data exported to CSV file.
"""
        # For now, just update status. In a full implementation, you might show a dialog
        self.status_label.setText("Report generated and exported successfully!")

    def closeEvent(self, event):
        """Handle widget close event"""
        self.stop_analysis()
        super().closeEvent(event)
