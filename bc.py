import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VideoEmotionAnalyzer:
    def __init__(self, video_path):
        """
        Initialize the Video Emotion Analyzer
        
        Args:
            video_path (str): Path to the input video file
        """
        self.video_path = video_path
        self.emotion_data = []
        self.confidence_mapping = {
            'happy': 0.8,
            'neutral': 0.6,
            'surprise': 0.5,
            'angry': 0.3,
            'disgust': 0.2,
            'fear': 0.1,
            'sad': 0.2
        }
        
    def extract_emotions_from_video(self):
        """
        Extract emotions from video at 1-second intervals
        """
        print("Starting video emotion analysis...")
        
        # Open video file
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {self.video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {duration:.2f} seconds")
        
        frame_interval = fps  # Process every 1 second
        current_time = 0
        
        while True:
            # Set frame position (1-second intervals)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_time * fps)
            
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Analyze emotion using DeepFace
                result = DeepFace.analyze(frame, 
                                        actions=['emotion'], 
                                        enforce_detection=False,
                                        silent=True)
                
                # Handle both single face and multiple faces scenarios
                if isinstance(result, list):
                    emotions = result[0]['emotion']
                else:
                    emotions = result['emotion']
                
                # Calculate confidence and nervousness percentages
                confidence_score, nervousness_score = self.calculate_confidence_nervousness(emotions)
                
                # Store data
                self.emotion_data.append({
                    'timestamp': current_time,
                    'confidence_percentage': confidence_score,
                    'nervousness_percentage': nervousness_score,
                    'dominant_emotion': max(emotions, key=emotions.get),
                    **emotions
                })
                
                print(f"Time: {current_time}s - Confidence: {confidence_score:.1f}% - Nervousness: {nervousness_score:.1f}%")
                
            except Exception as e:
                print(f"Error analyzing frame at {current_time}s: {str(e)}")
                # Add default values for failed analysis
                self.emotion_data.append({
                    'timestamp': current_time,
                    'confidence_percentage': 50.0,
                    'nervousness_percentage': 50.0,
                    'dominant_emotion': 'neutral',
                    'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 
                    'sad': 0, 'surprise': 0, 'neutral': 100
                })
            
            current_time += 1
            
            # Break if we've reached the end of the video
            if current_time >= duration:
                break
        
        cap.release()
        print(f"\nAnalysis complete! Processed {len(self.emotion_data)} frames.")
        
    def calculate_confidence_nervousness(self, emotions):
        """
        Calculate confidence and nervousness percentages based on emotions
        
        Args:
            emotions (dict): Dictionary of emotion percentages from DeepFace
            
        Returns:
            tuple: (confidence_percentage, nervousness_percentage)
        """
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
    
    def create_dataframe(self):
        """
        Create and return a pandas DataFrame with the emotion data
        """
        if not self.emotion_data:
            raise ValueError("No emotion data available. Run extract_emotions_from_video() first.")
        
        df = pd.DataFrame(self.emotion_data)
        return df
    
    def calculate_statistics(self, df):
        """
        Calculate median and other statistics
        """
        stats = {
            'confidence_median': df['confidence_percentage'].median(),
            'nervousness_median': df['nervousness_percentage'].median(),
            'confidence_mean': df['confidence_percentage'].mean(),
            'nervousness_mean': df['nervousness_percentage'].mean(),
            'confidence_std': df['confidence_percentage'].std(),
            'nervousness_std': df['nervousness_percentage'].std(),
            'total_duration': df['timestamp'].max(),
            'dominant_emotion_overall': df['dominant_emotion'].mode().iloc[0] if not df['dominant_emotion'].mode().empty else 'neutral'
        }
        return stats
    
    def create_visualizations(self, df, stats):
        """
        Create comprehensive visualizations
        """
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Video Emotion Analysis: Confidence vs Nervousness', fontsize=16, fontweight='bold')
        
        # 1. Time series plot
        axes[0, 0].plot(df['timestamp'], df['confidence_percentage'], 
                       label='Confidence', color='green', linewidth=2, marker='o', markersize=4)
        axes[0, 0].plot(df['timestamp'], df['nervousness_percentage'], 
                       label='Nervousness', color='red', linewidth=2, marker='s', markersize=4)
        axes[0, 0].axhline(y=stats['confidence_median'], color='green', 
                          linestyle='--', alpha=0.7, label=f'Confidence Median: {stats["confidence_median"]:.1f}%')
        axes[0, 0].axhline(y=stats['nervousness_median'], color='red', 
                          linestyle='--', alpha=0.7, label=f'Nervousness Median: {stats["nervousness_median"]:.1f}%')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Percentage (%)')
        axes[0, 0].set_title('Confidence vs Nervousness Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution histogram
        axes[0, 1].hist(df['confidence_percentage'], bins=20, alpha=0.7, 
                       label='Confidence', color='green', edgecolor='black')
        axes[0, 1].hist(df['nervousness_percentage'], bins=20, alpha=0.7, 
                       label='Nervousness', color='red', edgecolor='black')
        axes[0, 1].axvline(stats['confidence_median'], color='green', linestyle='--', linewidth=2)
        axes[0, 1].axvline(stats['nervousness_median'], color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Percentage (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Confidence and Nervousness')
        axes[0, 1].legend()
        
        # 3. Box plot
        box_data = [df['confidence_percentage'], df['nervousness_percentage']]
        box_plot = axes[1, 0].boxplot(box_data, labels=['Confidence', 'Nervousness'], 
                                     patch_artist=True, notch=True)
        box_plot['boxes'][0].set_facecolor('green')
        box_plot['boxes'][1].set_facecolor('red')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].set_title('Box Plot: Statistical Summary')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Emotion breakdown pie chart
        emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_means = [df[col].mean() for col in emotion_cols]
        colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_cols)))
        wedges, texts, autotexts = axes[1, 1].pie(emotion_means, labels=emotion_cols, autopct='%1.1f%%',
                                                 colors=colors, startangle=90)
        axes[1, 1].set_title('Average Emotion Distribution')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'emotion_analysis_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved as: {plot_filename}")
        
        return fig
    
    def generate_report(self, df, stats):
        """
        Generate a comprehensive text report
        """
        report = f"""
========================================
VIDEO EMOTION ANALYSIS REPORT
========================================

Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Video File: {self.video_path}
Total Duration: {stats['total_duration']} seconds
Frames Analyzed: {len(df)}

CONFIDENCE METRICS:
------------------
Median Confidence: {stats['confidence_median']:.1f}%
Average Confidence: {stats['confidence_mean']:.1f}%
Standard Deviation: {stats['confidence_std']:.1f}%
Maximum Confidence: {df['confidence_percentage'].max():.1f}%
Minimum Confidence: {df['confidence_percentage'].min():.1f}%

NERVOUSNESS METRICS:
-------------------
Median Nervousness: {stats['nervousness_median']:.1f}%
Average Nervousness: {stats['nervousness_mean']:.1f}%
Standard Deviation: {stats['nervousness_std']:.1f}%
Maximum Nervousness: {df['nervousness_percentage'].max():.1f}%
Minimum Nervousness: {df['nervousness_percentage'].min():.1f}%

OVERALL ASSESSMENT:
------------------
Dominant Emotion: {stats['dominant_emotion_overall']}
Overall Confidence Level: {'High' if stats['confidence_median'] > 60 else 'Medium' if stats['confidence_median'] > 40 else 'Low'}
Overall Nervousness Level: {'High' if stats['nervousness_median'] > 60 else 'Medium' if stats['nervousness_median'] > 40 else 'Low'}

DETAILED EMOTION BREAKDOWN:
--------------------------
"""
        
        emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        for emotion in emotion_cols:
            avg_val = df[emotion].mean()
            report += f"{emotion.capitalize()}: {avg_val:.1f}% (avg)\n"
        
        return report
    
    def save_data_to_csv(self, df):
        """
        Save the emotion data to a CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'emotion_data_{timestamp}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Data saved to: {csv_filename}")
        return csv_filename
    
    def run_complete_analysis(self):
        """
        Run the complete emotion analysis pipeline
        """
        try:
            # Step 1: Extract emotions from video
            self.extract_emotions_from_video()
            
            # Step 2: Create DataFrame
            df = self.create_dataframe()
            
            # Step 3: Calculate statistics
            stats = self.calculate_statistics(df)
            
            # Step 4: Create visualizations
            fig = self.create_visualizations(df, stats)
            
            # Step 5: Generate report
            report = self.generate_report(df, stats)
            
            # Step 6: Save data
            csv_file = self.save_data_to_csv(df)
            
            # Display results
            print("\n" + "="*50)
            print("ANALYSIS COMPLETE!")
            print("="*50)
            print(report)
            
            # Display tabulated data (first 10 and last 10 rows)
            print("\nTABULATED DATA SAMPLE:")
            print("-" * 70)
            display_df = df[['timestamp', 'confidence_percentage', 'nervousness_percentage', 'dominant_emotion']]
            
            if len(display_df) > 20:
                print("First 10 entries:")
                print(display_df.head(10).to_string(index=False))
                print("\n..." + f" [{len(display_df)-20} rows omitted] " + "...\n")
                print("Last 10 entries:")
                print(display_df.tail(10).to_string(index=False))
            else:
                print(display_df.to_string(index=False))
            
            # Show plot
            plt.show()
            
            return df, stats, fig
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Use the video file in the current directory
    video_path = "test subject.mp4"

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' not found in current directory.")
        print("Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"  - {file}")
        print("\nPlease ensure the video file is in the current directory.")
        exit(1)

    # Create analyzer instance
    analyzer = VideoEmotionAnalyzer(video_path)

    # Run complete analysis
    try:
        df, stats, fig = analyzer.run_complete_analysis()
        print(f"\nAnalysis completed successfully!")
        print(f"Median Confidence Score: {stats['confidence_median']:.1f}%")
        print(f"Median Nervousness Score: {stats['nervousness_median']:.1f}%")

    except ImportError as e:
        print(f"Import error: {str(e)}")
        print("\nMissing dependencies. Please install:")
        print("pip install deepface opencv-python pandas matplotlib seaborn numpy")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Ensure video file is not corrupted")
        print("2. Check if all dependencies are installed")
        print("3. Make sure you have sufficient RAM for video processing")
