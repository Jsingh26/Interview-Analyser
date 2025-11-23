import os
import sys
from PyQt6.QtCore import QThread, pyqtSignal
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path to import bc.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bc import VideoEmotionAnalyzer


class VideoProcessor(QThread):
    """Worker thread for processing video emotion analysis in the background"""

    # Signals for communication with GUI
    progress_updated = pyqtSignal(int, str)  # progress percentage, status message
    analysis_complete = pyqtSignal(object, object, object)  # df, stats, fig
    error_occurred = pyqtSignal(str)  # error message
    log_message = pyqtSignal(str)  # general log messages

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.analyzer = None

    def run(self):
        """Run the video emotion analysis in background thread"""
        try:
            self.log_message.emit("Initializing video emotion analyzer...")
            self.analyzer = VideoEmotionAnalyzer(self.video_path)

            self.log_message.emit("Starting emotion extraction from video...")
            self.progress_updated.emit(10, "Extracting emotions from video...")

            # Extract emotions from video
            self.analyzer.extract_emotions_from_video()

            self.log_message.emit("Creating data analysis...")
            self.progress_updated.emit(40, "Processing emotion data...")

            # Create DataFrame
            df = self.analyzer.create_dataframe()

            self.log_message.emit("Calculating statistics...")
            self.progress_updated.emit(60, "Calculating statistics...")

            # Calculate statistics
            stats = self.analyzer.calculate_statistics(df)

            self.log_message.emit("Generating visualizations...")
            self.progress_updated.emit(80, "Creating visualizations...")

            # Create visualizations
            fig = self.analyzer.create_visualizations(df, stats)

            self.log_message.emit("Saving results...")
            self.progress_updated.emit(95, "Saving results...")

            # Save data to CSV
            csv_file = self.analyzer.save_data_to_csv(df)

            self.progress_updated.emit(100, "Analysis complete!")
            self.log_message.emit(f"Analysis completed successfully. Results saved to: {csv_file}")

            # Emit completion signal with results
            self.analysis_complete.emit(df, stats, fig)

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.log_message.emit(error_msg)
            self.error_occurred.emit(error_msg)
            self.progress_updated.emit(0, "Analysis failed")

    def get_supported_formats(self):
        """Return list of supported video formats"""
        return ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
