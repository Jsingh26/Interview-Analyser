import cv2
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
from deepface import DeepFace
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class RealTimeAnalyzer(QThread):
    """Real-time emotion analyzer for camera feed"""

    # Signals
    frame_updated = pyqtSignal(object)  # QPixmap of current frame
    stats_updated = pyqtSignal(dict)    # Current stats dictionary
    graph_data_updated = pyqtSignal(list, list, list)  # timestamps, confidence, nervousness
    error_occurred = pyqtSignal(str)    # Error message

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.is_running = False
        self.cap = None

        # Data storage for graphs (keep last 60 seconds of data)
        self.max_data_points = 60
        self.timestamps = deque(maxlen=self.max_data_points)
        self.confidence_history = deque(maxlen=self.max_data_points)
        self.nervousness_history = deque(maxlen=self.max_data_points)

        # Confidence mapping (same as bc.py)
        self.confidence_mapping = {
            'happy': 0.8,
            'neutral': 0.6,
            'surprise': 0.5,
            'angry': 0.3,
            'disgust': 0.2,
            'fear': 0.1,
            'sad': 0.2
        }

        # Current stats
        self.current_stats = {
            'confidence': 50.0,
            'nervousness': 50.0,
            'dominant_emotion': 'neutral',
            'emotions': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
                        'sad': 0, 'surprise': 0, 'neutral': 100}
        }

        self.start_time = 0

    def run(self):
        """Main analysis loop"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.error_occurred.emit("Could not access camera")
                return

            self.is_running = True
            self.start_time = time.time()

            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.error_occurred.emit("Failed to capture frame")
                    break

                # Process frame for display
                display_frame = self.process_frame_for_display(frame)
                self.frame_updated.emit(display_frame)

                # Analyze emotions
                try:
                    self.analyze_emotions(frame)
                except Exception as e:
                    print(f"Emotion analysis error: {e}")
                    # Continue with default values

                # Update graph data
                current_time = time.time() - self.start_time
                self.timestamps.append(current_time)
                self.confidence_history.append(self.current_stats['confidence'])
                self.nervousness_history.append(self.current_stats['nervousness'])

                # Emit updated data
                self.stats_updated.emit(self.current_stats.copy())
                self.graph_data_updated.emit(
                    list(self.timestamps),
                    list(self.confidence_history),
                    list(self.nervousness_history)
                )

                # Small delay to prevent overwhelming the system
                time.sleep(0.5)  # ~2 FPS analysis

        except Exception as e:
            self.error_occurred.emit(f"Real-time analysis error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()

    def process_frame_for_display(self, frame):
        """Process frame for display in GUI"""
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Convert to QPixmap
        pixmap = QPixmap.fromImage(q_image)

        return pixmap

    def analyze_emotions(self, frame):
        """Analyze emotions in the current frame"""
        try:
            # Use DeepFace for emotion analysis
            result = DeepFace.analyze(frame,
                                    actions=['emotion'],
                                    enforce_detection=False,
                                    silent=True)

            # Handle both single face and multiple faces scenarios
            if isinstance(result, list):
                emotions = result[0]['emotion']
            else:
                emotions = result['emotion']

            # Calculate confidence and nervousness
            confidence_score, nervousness_score = self.calculate_confidence_nervousness(emotions)

            # Update current stats
            self.current_stats.update({
                'confidence': confidence_score,
                'nervousness': nervousness_score,
                'dominant_emotion': max(emotions, key=emotions.get),
                'emotions': emotions
            })

        except Exception as e:
            # If analysis fails, keep previous values or use defaults
            print(f"Frame analysis failed: {e}")
            pass

    def calculate_confidence_nervousness(self, emotions):
        """Calculate confidence and nervousness percentages"""
        # Confidence calculation: weighted sum of positive emotions
        confidence_emotions = ['happy', 'neutral', 'surprise']
        nervousness_emotions = ['fear', 'sad', 'angry', 'disgust']

        confidence_raw = sum(emotions.get(emotion, 0) * self.confidence_mapping.get(emotion, 0.5)
                           for emotion in confidence_emotions)

        nervousness_raw = sum(emotions.get(emotion, 0) * (1 - self.confidence_mapping.get(emotion, 0.5))
                            for emotion in nervousness_emotions)

        # Normalize to percentages
        total_raw = confidence_raw + nervousness_raw
        if total_raw > 0:
            confidence_percentage = (confidence_raw / total_raw) * 100
            nervousness_percentage = (nervousness_raw / total_raw) * 100
        else:
            confidence_percentage = 50.0
            nervousness_percentage = 50.0

        # Ensure they sum to 100%
        total = confidence_percentage + nervousness_percentage
        if total > 0:
            confidence_percentage = (confidence_percentage / total) * 100
            nervousness_percentage = (nervousness_percentage / total) * 100

        return round(confidence_percentage, 1), round(nervousness_percentage, 1)

    def stop_analysis(self):
        """Stop the real-time analysis"""
        self.is_running = False
        if self.cap:
            self.cap.release()

    def reset_data(self):
        """Reset all stored data"""
        self.timestamps.clear()
        self.confidence_history.clear()
        self.nervousness_history.clear()
        self.start_time = time.time()

        # Reset stats to defaults
        self.current_stats = {
            'confidence': 50.0,
            'nervousness': 50.0,
            'dominant_emotion': 'neutral',
            'emotions': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
                        'sad': 0, 'surprise': 0, 'neutral': 100}
        }

    def generate_report(self):
        """Generate a comprehensive report of the real-time analysis"""
        if not self.timestamps:
            return None

        # Calculate overall statistics
        total_frames = len(self.timestamps)
        duration = self.timestamps[-1] if self.timestamps else 0

        # Calculate averages
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 50.0
        avg_nervousness = np.mean(self.nervousness_history) if self.nervousness_history else 50.0

        # Calculate peaks and minimums
        max_confidence = max(self.confidence_history) if self.confidence_history else 50.0
        min_confidence = min(self.confidence_history) if self.confidence_history else 50.0
        max_nervousness = max(self.nervousness_history) if self.nervousness_history else 50.0
        min_nervousness = min(self.nervousness_history) if self.nervousness_history else 50.0

        # Calculate stability (variance)
        confidence_variance = np.var(self.confidence_history) if self.confidence_history else 0
        nervousness_variance = np.var(self.nervousness_history) if self.nervousness_history else 0

        # Create report dictionary
        report = {
            'session_info': {
                'total_frames': total_frames,
                'duration_seconds': round(duration, 2),
                'analysis_rate_fps': round(total_frames / duration, 2) if duration > 0 else 0,
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)),
                'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time + duration))
            },
            'overall_statistics': {
                'average_confidence': round(avg_confidence, 2),
                'average_nervousness': round(avg_nervousness, 2),
                'max_confidence': round(max_confidence, 2),
                'min_confidence': round(min_confidence, 2),
                'max_nervousness': round(max_nervousness, 2),
                'min_nervousness': round(min_nervousness, 2),
                'confidence_stability': round(confidence_variance, 2),
                'nervousness_stability': round(nervousness_variance, 2)
            },
            'data_points': {
                'timestamps': list(self.timestamps),
                'confidence_values': list(self.confidence_history),
                'nervousness_values': list(self.nervousness_history)
            }
        }

        return report

    def export_report_to_csv(self, filename=None):
        """Export the analysis data to CSV file"""
        if not self.timestamps:
            return False

        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"realtime_emotion_analysis_{timestamp}.csv"

        try:
            # Create DataFrame with the analysis data
            data = {
                'timestamp_seconds': list(self.timestamps),
                'confidence_percentage': list(self.confidence_history),
                'nervousness_percentage': list(self.nervousness_history)
            }

            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            return filename
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
