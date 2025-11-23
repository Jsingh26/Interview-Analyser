import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit,
                             QSplitter, QGroupBox, QMessageBox, QStatusBar, QFrame, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPixmap, QDragEnterEvent, QDropEvent, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve, QTimer
import pandas as pd
import cv2

# Import our custom modules
from VideoProcessor import VideoProcessor
from ResultsDisplay import ModernResultsDisplay
from GraphViewer import GraphViewer
from RealTimeDisplay import RealTimeDisplay


class VideoEmotionGUI(QMainWindow):
    """Main GUI application for video emotion analysis"""

    def __init__(self):
        super().__init__()
        self.video_path = None
        self.processor = None
        self.results_display = None
        self.init_ui()
        self.setWindowTitle("Video Emotion Analysis Tool")
        self.resize(1400, 900)
        self.setStyleSheet("QMainWindow { background-color: black; }")
        self.animate_window()

    def generate_video_thumbnail(self, video_path):
        """Generate a thumbnail from the video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError("Could not open video file for thumbnail generation")

        # Get video dimensions
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read the first frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read frame from video")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Convert to QPixmap and resize based on video aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        max_width = 200
        aspect_ratio = video_width / video_height
        thumbnail_width = min(max_width, int(max_width * aspect_ratio))
        thumbnail_height = int(thumbnail_width / aspect_ratio)
        thumbnail_size = QSize(thumbnail_width, thumbnail_height)
        pixmap = pixmap.scaled(thumbnail_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        return pixmap

    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Title
        title_label = QLabel("Video Emotion Analysis Tool")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2196F3; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        # Create tab widget for separate pages
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: Upload & Analysis
        upload_tab = QWidget()
        upload_layout = QVBoxLayout(upload_tab)
        upload_layout.setContentsMargins(20, 20, 20, 20)
        upload_layout.setSpacing(20)

        # Upload section
        upload_widget = self.create_upload_widget()
        upload_layout.addWidget(upload_widget)

        # Progress section
        progress_widget = self.create_progress_widget()
        upload_layout.addWidget(progress_widget)

        self.tab_widget.addTab(upload_tab, "Upload & Analysis")

        # Tab 2: Results
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        results_layout.setContentsMargins(20, 20, 20, 20)
        results_layout.setSpacing(20)

        # Results section
        self.results_widget = self.create_results_widget()
        results_layout.addWidget(self.results_widget)

        self.tab_widget.addTab(results_tab, "Results")

        # Tab 3: Real-Time Analysis
        realtime_tab = QWidget()
        realtime_layout = QVBoxLayout(realtime_tab)
        realtime_layout.setContentsMargins(20, 20, 20, 20)
        realtime_layout.setSpacing(20)

        # Real-time analysis section
        self.realtime_display = RealTimeDisplay()
        realtime_layout.addWidget(self.realtime_display)

        self.tab_widget.addTab(realtime_tab, "Real-Time Analysis")

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Select a video file to begin analysis")

    def create_upload_widget(self):
        """Create widget for video file upload"""
        widget = QGroupBox("Video Upload")
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Upload instructions
        instructions = QLabel("Select a video file to analyze emotions:")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Drag and drop area
        self.drop_area = DropArea()
        self.drop_area.file_dropped.connect(self.handle_file_drop)
        layout.addWidget(self.drop_area)

        # File selection buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)

        self.select_file_btn = QPushButton("Select Video File")
        self.select_file_btn.clicked.connect(self.select_video_file)
        self.select_file_btn.setMinimumHeight(35)
        self.select_file_btn.setStyleSheet("""
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
        """)
        buttons_layout.addWidget(self.select_file_btn)

        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("""
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
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)
        self.analyze_btn.setMinimumHeight(35)
        buttons_layout.addWidget(self.analyze_btn)

        layout.addLayout(buttons_layout)

        # Supported formats info
        formats_label = QLabel("Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM")
        formats_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(formats_label)

        widget.setLayout(layout)
        return widget

    def create_progress_widget(self):
        """Create widget for progress tracking"""
        widget = QGroupBox("Analysis Progress")
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(25)
        layout.addWidget(self.progress_bar)

        # Status text
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(120)
        layout.addWidget(self.status_text)

        widget.setLayout(layout)
        return widget

    def create_results_widget(self):
        """Create widget for displaying results"""
        from PyQt6.QtWidgets import QScrollArea
        self.results_display = ModernResultsDisplay()

        # Create scroll area for results
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.results_display)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        return scroll_area

    def select_video_file(self):
        """Open file dialog to select video file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm)"
        )

        if file_path:
            self.handle_file_selection(file_path)

    def handle_file_drop(self, file_path):
        """Handle dropped file"""
        self.handle_file_selection(file_path)

    def handle_file_selection(self, file_path):
        """Handle video file selection"""
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Error", "Selected file does not exist!")
            return

        # Check if it's a video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        if not any(file_path.lower().endswith(ext) for ext in video_extensions):
            QMessageBox.warning(self, "Warning", "Please select a valid video file!")
            return

        self.video_path = file_path
        self.status_bar.showMessage(f"Video selected: {os.path.basename(file_path)}")
        self.analyze_btn.setEnabled(True)

        # Generate and display thumbnail
        try:
            thumbnail_pixmap = self.generate_video_thumbnail(file_path)
            self.drop_area.update_file_info(file_path, thumbnail_pixmap)
        except Exception as e:
            print(f"Error generating thumbnail: {e}")
            self.drop_area.update_file_info(file_path)

    def start_analysis(self):
        """Start the video emotion analysis"""
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please select a video file first!")
            return

        # Disable controls during analysis
        self.select_file_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing...")

        # Clear previous results
        self.progress_bar.setValue(0)
        self.status_text.clear()

        # Start background processing
        self.processor = VideoProcessor(self.video_path)
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.analysis_complete.connect(self.show_results)
        self.processor.error_occurred.connect(self.handle_error)
        self.processor.log_message.connect(self.log_message)
        self.processor.start()

    def update_progress(self, percentage, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(percentage)
        self.status_bar.showMessage(message)

    def log_message(self, message):
        """Add message to status text"""
        self.status_text.append(message)
        # Scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def show_results(self, df, stats, figure):
        """Display analysis results"""
        # Enable controls
        self.select_file_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Start Analysis")

        # Update results display
        self.results_display.update_results(df, stats, figure)

        # Switch to results tab
        self.tab_widget.setCurrentIndex(1)

        # Update status
        self.status_bar.showMessage("Analysis complete! View results in the Results tab.")

        # Show success message
        QMessageBox.information(self, "Success", "Video emotion analysis completed successfully!")

    def handle_error(self, error_message):
        """Handle analysis errors"""
        # Enable controls
        self.select_file_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Start Analysis")

        # Show error message
        QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis:\n\n{error_message}")

        # Update status
        self.status_bar.showMessage("Analysis failed - check error message above")

    def closeEvent(self, event):
        """Handle application close event"""
        if self.processor and self.processor.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Analysis is still running. Do you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            self.processor.terminate()

        # Stop real-time analysis if running
        if hasattr(self, 'realtime_display') and self.realtime_display.analyzer:
            self.realtime_display.stop_analysis()

        event.accept()

    def animate_window(self):
        """Animate the window appearance"""
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(1000)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.start()


class DropArea(QFrame):
    """Custom widget for drag and drop file selection"""

    file_dropped = pyqtSignal(str)  # Signal emitted when file is dropped

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Box)
        self.setMinimumHeight(100)
        self.setAcceptDrops(True)
        self.selected_file = None
        self.init_ui()

    def init_ui(self):
        """Initialize the drop area UI"""
        layout = QVBoxLayout()

        self.icon_label = QLabel("📹")
        self.icon_label.setFont(QFont("Arial", 48))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)

        self.text_label = QLabel("Drag and drop video file here\nor click 'Select Video File'")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setWordWrap(True)
        layout.addWidget(self.text_label)

        self.file_info_label = QLabel("")
        self.file_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_info_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        layout.addWidget(self.file_info_label)

        self.setLayout(layout)

        # Set styling
        self.setStyleSheet("""
            DropArea {
                border: 2px dashed #2196F3;
                background-color: #f0f8ff;
                border-radius: 10px;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                DropArea {
                    border: 2px dashed #4CAF50;
                    background-color: #e8f5e8;
                }
            """)

    def dragLeaveEvent(self, event):
        """Handle drag leave event"""
        self.setStyleSheet("""
            DropArea {
                border: 2px dashed #aaa;
                background-color: #f9f9f9;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        """Handle file drop event"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_dropped.emit(file_path)

        self.setStyleSheet("""
            DropArea {
                border: 2px dashed #aaa;
                background-color: #f9f9f9;
            }
        """)

    def update_file_info(self, file_path, thumbnail_pixmap=None):
        """Update file information display"""
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        # Format file size
        if file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"

        self.file_info_label.setText(f"Selected: {file_name} ({size_str})")
        self.text_label.setText("Video file selected successfully!")

        # Update thumbnail if provided
        if thumbnail_pixmap:
            self.icon_label.setPixmap(thumbnail_pixmap)
            self.icon_label.setScaledContents(True)
        else:
            # Reset to default icon if no thumbnail
            self.icon_label.setPixmap(QPixmap())
            self.icon_label.setText("📹")
            self.icon_label.setFont(QFont("Arial", 48))


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Video Emotion Analysis Tool")
    app.setApplicationVersion("1.0")

    # Create and show main window
    window = VideoEmotionGUI()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
