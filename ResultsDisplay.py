from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QGroupBox, QTextEdit, QTableWidget, QTableWidgetItem,
                             QSplitter, QFrame, QScrollArea, QHeaderView, QTabWidget,
                             QGraphicsOpacityEffect)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer, QSize, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage, QColor, QLinearGradient, QPalette
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import pandas as pd
from datetime import datetime
import numpy as np


class ModernResultsDisplay(QWidget):
    """Modern results display with advanced visualizations and animations"""

    def __init__(self):
        super().__init__()
        self.df = None
        self.stats = None
        self.figure = None
        self.init_ui()

    def init_ui(self):
        """Initialize the modern user interface"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(20)

        # Header with animation
        header = self.create_animated_header()
        main_layout.addWidget(header)

        # Create tab widget for different views
        self.results_tabs = QTabWidget()
        self.results_tabs.setDocumentMode(True)
        main_layout.addWidget(self.results_tabs)

        # Tab 1: Dashboard Overview
        dashboard_tab = self.create_dashboard_tab()
        self.results_tabs.addTab(dashboard_tab, "📊 Dashboard")

        # Tab 2: Detailed Analytics
        analytics_tab = self.create_analytics_tab()
        self.results_tabs.addTab(analytics_tab, "📈 Analytics")

        # Tab 3: Data Table
        table_tab = self.create_table_tab()
        self.results_tabs.addTab(table_tab, "📋 Data Table")

        # Action buttons
        buttons_layout = self.create_action_buttons()
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    def create_animated_header(self):
        """Create animated header"""
        header_frame = QFrame()
        header_layout = QVBoxLayout(header_frame)

        title_label = QLabel("📊 Analysis Results Dashboard")
        title_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            color: white;
            padding: 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(102, 126, 234, 0.3), stop:1 rgba(118, 75, 162, 0.3));
            border-radius: 10px;
        """)
        header_layout.addWidget(title_label)

        return header_frame

    def create_dashboard_tab(self):
        """Create dashboard overview tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)

        # Metrics cards
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)

        self.confidence_card = self.create_metric_card("💪 Confidence", "N/A", "#4facfe")
        self.nervousness_card = self.create_metric_card("😰 Nervousness", "N/A", "#f093fb")
        self.emotion_card = self.create_metric_card("😊 Dominant", "N/A", "#43e97b")

        metrics_layout.addWidget(self.confidence_card)
        metrics_layout.addWidget(self.nervousness_card)
        metrics_layout.addWidget(self.emotion_card)

        layout.addLayout(metrics_layout)

        # Graph display area
        self.graph_widget = QGroupBox("📈 Visual Analysis")
        graph_layout = QVBoxLayout()

        self.canvas_widget = QWidget()
        self.canvas_layout = QVBoxLayout(self.canvas_widget)
        graph_layout.addWidget(self.canvas_widget)

        self.graph_widget.setLayout(graph_layout)
        layout.addWidget(self.graph_widget)

        # Quick stats
        self.quick_stats_widget = self.create_quick_stats_widget()
        layout.addWidget(self.quick_stats_widget)

        return tab

    def create_analytics_tab(self):
        """Create detailed analytics tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)

        # Detailed report
        self.detailed_report = QGroupBox("📄 Detailed Analysis Report")
        report_layout = QVBoxLayout()

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMinimumHeight(400)
        report_layout.addWidget(self.report_text)

        self.detailed_report.setLayout(report_layout)
        layout.addWidget(self.detailed_report)

        return tab

    def create_table_tab(self):
        """Create data table tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)

        # Search and filter controls
        controls_layout = QHBoxLayout()

        search_label = QLabel("🔍 Filter:")
        controls_layout.addWidget(search_label)

        from PyQt6.QtWidgets import QLineEdit, QComboBox
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search in table...")
        self.search_box.textChanged.connect(self.filter_table)
        controls_layout.addWidget(self.search_box)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Modern data table
        self.data_table_widget = QGroupBox("📊 Emotion Data Timeline")
        table_layout = QVBoxLayout()

        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setStyleSheet("""
            QTableWidget {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                color: white;
                gridline-color: rgba(255, 255, 255, 0.1);
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
            }
            QHeaderView::section {
                background: rgba(102, 126, 234, 0.5);
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
        """)
        table_layout.addWidget(self.data_table)

        self.data_table_widget.setLayout(table_layout)
        layout.addWidget(self.data_table_widget)

        return tab

    def create_metric_card(self, title, value, color):
        """Create animated metric card"""
        card = QFrame()
        card.setMinimumHeight(120)
        card.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {color}, stop:1 rgba(255, 255, 255, 0.1));
                border-radius: 15px;
                border: 2px solid rgba(255, 255, 255, 0.2);
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 15, 20, 15)

        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_label.setStyleSheet("color: white;")
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        value_label.setStyleSheet("color: white;")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_label)

        # Store label for updates
        card.value_label = value_label

        return card

    def create_quick_stats_widget(self):
        """Create quick statistics widget"""
        widget = QGroupBox("⚡ Quick Statistics")
        layout = QVBoxLayout()

        self.quick_stats_text = QTextEdit()
        self.quick_stats_text.setReadOnly(True)
        self.quick_stats_text.setMaximumHeight(200)
        layout.addWidget(self.quick_stats_text)

        widget.setLayout(layout)
        return widget

    def create_action_buttons(self):
        """Create action buttons layout"""
        layout = QHBoxLayout()
        layout.setSpacing(15)
        layout.addStretch()

        self.export_csv_btn = QPushButton("💾 Export CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.setMinimumHeight(45)
        self.export_csv_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.export_csv_btn)

        self.export_graph_btn = QPushButton("📊 Export Graph")
        self.export_graph_btn.clicked.connect(self.export_graph)
        self.export_graph_btn.setEnabled(False)
        self.export_graph_btn.setMinimumHeight(45)
        self.export_graph_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.export_graph_btn)

        self.export_report_btn = QPushButton("📄 Export Report")
        self.export_report_btn.clicked.connect(self.export_report)
        self.export_report_btn.setEnabled(False)
        self.export_report_btn.setMinimumHeight(45)
        self.export_report_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.export_report_btn)

        return layout

    def update_results(self, df, stats, figure):
        """Update display with new results"""
        self.df = df
        self.stats = stats
        self.figure = figure

        # Animate updates
        self.animate_metric_cards()

        # Update all displays
        self.update_metric_cards()
        self.update_graph_display()
        self.update_quick_stats()
        self.update_detailed_report()
        self.update_data_table()

        # Enable export buttons
        self.export_csv_btn.setEnabled(True)
        self.export_graph_btn.setEnabled(True)
        self.export_report_btn.setEnabled(True)

    def update_metric_cards(self):
        """Update metric cards with data"""
        if not self.stats:
            return

        # Update confidence card
        confidence_val = f"{self.stats['confidence_median']:.1f}%"
        self.confidence_card.value_label.setText(confidence_val)

        # Update nervousness card
        nervousness_val = f"{self.stats['nervousness_median']:.1f}%"
        self.nervousness_card.value_label.setText(nervousness_val)

        # Update emotion card
        emotion_val = self.stats['dominant_emotion_overall'].capitalize()
        self.emotion_card.value_label.setText(emotion_val)

    def update_graph_display(self):
        """Update graph display"""
        if not self.figure:
            return

        # Clear previous canvas
        for i in reversed(range(self.canvas_layout.count())):
            self.canvas_layout.itemAt(i).widget().setParent(None)

        # Create new canvas
        canvas = FigureCanvas(self.figure)
        canvas.setMinimumHeight(600)
        canvas.setMinimumWidth(1200)
        self.canvas_layout.addWidget(canvas)

    def update_quick_stats(self):
        """Update quick statistics"""
        if not self.stats or self.df is None:
            return

        stats_html = f"""
        <div style='color: white;'>
        <h3 style='color: #4facfe;'>📊 Statistical Summary</h3>

        <table style='width: 100%; border-collapse: collapse;'>
        <tr>
            <td style='padding: 8px;'><b>📈 Confidence Median:</b></td>
            <td style='padding: 8px; color: #4facfe;'>{self.stats['confidence_median']:.1f}%</td>
            <td style='padding: 8px;'><b>📉 Nervousness Median:</b></td>
            <td style='padding: 8px; color: #f093fb;'>{self.stats['nervousness_median']:.1f}%</td>
        </tr>
        <tr>
            <td style='padding: 8px;'><b>⬆️ Max Confidence:</b></td>
            <td style='padding: 8px; color: #43e97b;'>{self.df['confidence_percentage'].max():.1f}%</td>
            <td style='padding: 8px;'><b>⬆️ Max Nervousness:</b></td>
            <td style='padding: 8px; color: #fa709a;'>{self.df['nervousness_percentage'].max():.1f}%</td>
        </tr>
        <tr>
            <td style='padding: 8px;'><b>⏱️ Duration:</b></td>
            <td style='padding: 8px;'>{self.stats['total_duration']}s</td>
            <td style='padding: 8px;'><b>🎬 Frames Analyzed:</b></td>
            <td style='padding: 8px;'>{len(self.df)}</td>
        </tr>
        </table>

        <p style='margin-top: 15px;'><b>🎭 Overall Assessment:</b></p>
        <ul>
            <li>Confidence Level: <span style='color: #4facfe;'><b>{'High' if self.stats['confidence_median'] > 60 else 'Medium' if self.stats['confidence_median'] > 40 else 'Low'}</b></span></li>
            <li>Nervousness Level: <span style='color: #f093fb;'><b>{'High' if self.stats['nervousness_median'] > 60 else 'Medium' if self.stats['nervousness_median'] > 40 else 'Low'}</b></span></li>
            <li>Dominant Emotion: <span style='color: #43e97b;'><b>{self.stats['dominant_emotion_overall'].capitalize()}</b></span></li>
        </ul>
        </div>
        """

        self.quick_stats_text.setHtml(stats_html)

    def update_detailed_report(self):
        """Update detailed report"""
        if not self.stats or self.df is None:
            return

        emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        emotion_breakdown = ""
        for emotion in emotion_cols:
            avg_val = self.df[emotion].mean()
            max_val = self.df[emotion].max()
            color = self.get_emotion_color(emotion)

            emotion_breakdown += f"""
            <div style='margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 5px;'>
                <span style='color: {color}; font-size: 16px;'><b>{emotion.capitalize()}</b></span><br/>
                Average: {avg_val:.1f}% | Peak: {max_val:.1f}%
            </div>
            """

        report_html = f"""
        <div style='color: white; font-size: 13px;'>
        <h2 style='color: #4facfe; text-align: center;'>📄 Comprehensive Analysis Report</h2>

        <div style='background: rgba(102, 126, 234, 0.2); padding: 15px; border-radius: 10px; margin: 15px 0;'>
            <h3>📊 Executive Summary</h3>
            <p><b>Analysis Date:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><b>Total Duration:</b> {self.stats['total_duration']} seconds</p>
            <p><b>Frames Analyzed:</b> {len(self.df)} frames</p>
            <p><b>Sample Rate:</b> 1 frame per second</p>
        </div>

        <div style='background: rgba(102, 126, 234, 0.2); padding: 15px; border-radius: 10px; margin: 15px 0;'>
            <h3>💪 Confidence Analysis</h3>
            <p><b>Median:</b> {self.stats['confidence_median']:.1f}%</p>
            <p><b>Average:</b> {self.stats['confidence_mean']:.1f}%</p>
            <p><b>Std Deviation:</b> {self.stats['confidence_std']:.1f}%</p>
            <p><b>Range:</b> {self.df['confidence_percentage'].min():.1f}% - {self.df['confidence_percentage'].max():.1f}%</p>
        </div>

        <div style='background: rgba(240, 147, 251, 0.2); padding: 15px; border-radius: 10px; margin: 15px 0;'>
            <h3>😰 Nervousness Analysis</h3>
            <p><b>Median:</b> {self.stats['nervousness_median']:.1f}%</p>
            <p><b>Average:</b> {self.stats['nervousness_mean']:.1f}%</p>
            <p><b>Std Deviation:</b> {self.stats['nervousness_std']:.1f}%</p>
            <p><b>Range:</b> {self.df['nervousness_percentage'].min():.1f}% - {self.df['nervousness_percentage'].max():.1f}%</p>
        </div>

        <div style='background: rgba(67, 233, 123, 0.2); padding: 15px; border-radius: 10px; margin: 15px 0;'>
            <h3>🎭 Detailed Emotion Breakdown</h3>
            {emotion_breakdown}
        </div>

        <div style='background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin: 15px 0;'>
            <h3>📈 Key Insights</h3>
            <ul>
                <li>The subject showed <b>{self.get_confidence_description()}</b> throughout the video.</li>
                <li>Nervousness levels were <b>{self.get_nervousness_description()}</b>.</li>
                <li>The most prevalent emotion was <b style='color: #43e97b;'>{self.stats['dominant_emotion_overall']}</b>.</li>
                <li>Emotional variability (std dev) suggests <b>{self.get_stability_description()}</b>.</li>
            </ul>
        </div>
        </div>
        """

        self.report_text.setHtml(report_html)

    def update_data_table(self):
        """Update data table"""
        if self.df is None:
            return

        display_cols = ['timestamp', 'confidence_percentage', 'nervousness_percentage',
                       'dominant_emotion', 'happy', 'sad', 'angry', 'fear', 'neutral']

        self.data_table.setColumnCount(len(display_cols))
        self.data_table.setHorizontalHeaderLabels([col.replace('_', ' ').title() for col in display_cols])

        # Show all rows (limit if too many)
        max_rows = min(len(self.df), 1000)
        display_df = self.df[display_cols].head(max_rows)

        self.data_table.setRowCount(len(display_df))

        for row in range(len(display_df)):
            for col, column_name in enumerate(display_cols):
                value = display_df.iloc[row, col]

                if 'percentage' in column_name or column_name in ['happy', 'sad', 'angry', 'fear', 'neutral']:
                    item = QTableWidgetItem(f"{value:.1f}%")
                elif column_name == 'timestamp':
                    item = QTableWidgetItem(f"{value}s")
                else:
                    item = QTableWidgetItem(str(value))

                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.data_table.setItem(row, col, item)

        self.data_table.resizeColumnsToContents()
        header = self.data_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def filter_table(self, text):
        """Filter table based on search text"""
        for row in range(self.data_table.rowCount()):
            match = False
            for col in range(self.data_table.columnCount()):
                item = self.data_table.item(row, col)
                if item and text.lower() in item.text().lower():
                    match = True
                    break
            self.data_table.setRowHidden(row, not match)

    def get_emotion_color(self, emotion):
        """Get color for emotion"""
        colors = {
            'happy': '#43e97b',
            'sad': '#4facfe',
            'angry': '#fa709a',
            'fear': '#fee140',
            'neutral': '#c471f5',
            'surprise': '#f093fb',
            'disgust': '#667eea'
        }
        return colors.get(emotion, '#ffffff')

    def get_confidence_description(self):
        """Get confidence level description"""
        median = self.stats['confidence_median']
        if median > 70:
            return "consistently high confidence levels"
        elif median > 50:
            return "moderate to high confidence levels"
        elif median > 30:
            return "moderate confidence with some uncertainty"
        else:
            return "lower confidence levels"

    def get_nervousness_description(self):
        """Get nervousness level description"""
        median = self.stats['nervousness_median']
        if median > 60:
            return "notably high"
        elif median > 40:
            return "moderately elevated"
        else:
            return "relatively low"

    def get_stability_description(self):
        """Get emotional stability description"""
        std = self.stats['confidence_std']
        if std > 20:
            return "high emotional variability"
        elif std > 10:
            return "moderate emotional consistency"
        else:
            return "stable emotional state"

    def animate_metric_cards(self):
        """Animate metric cards entrance"""
        cards = [self.confidence_card, self.nervousness_card, self.emotion_card]

        for i, card in enumerate(cards):
            effect = QGraphicsOpacityEffect(card)
            card.setGraphicsEffect(effect)

            animation = QPropertyAnimation(effect, b"opacity")
            animation.setDuration(800)
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
            animation.setEasingCurve(QEasingCurve.Type.OutCubic)

            # Delay each card slightly
            QTimer.singleShot(i * 200, animation.start)

    def export_csv(self):
        """Export data to CSV"""
        if self.df is None:
            return

        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", f"emotion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                self.df.to_csv(file_path, index=False)
                self.show_success_notification(f"CSV exported successfully to:\n{file_path}")
            except Exception as e:
                self.show_error_notification(f"Error exporting CSV: {str(e)}")

    def export_graph(self):
        """Export graph to image"""
        if self.figure is None:
            return

        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Graph", f"emotion_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.show_success_notification(f"Graph exported successfully to:\n{file_path}")
            except Exception as e:
                self.show_error_notification(f"Error exporting graph: {str(e)}")

    def export_report(self):
        """Export detailed report to HTML"""
        if not self.stats or self.df is None:
            return

        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Files (*.html)"
        )

        if file_path:
            try:
                html_content = self.report_text.toHtml()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.show_success_notification(f"Report exported successfully to:\n{file_path}")
            except Exception as e:
                self.show_error_notification(f"Error exporting report: {str(e)}")

    def show_success_notification(self, message):
        """Show success notification"""
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Success")
        msg.setText(message)
        msg.exec()

    def show_error_notification(self, message):
        """Show error notification"""
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Error")
        msg.setText(message)
        msg.exec()
