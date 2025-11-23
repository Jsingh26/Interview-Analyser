# Video Emotion Analysis Tool

## Description
This project is a comprehensive Video Emotion Analysis Tool leveraging the DeepFace library and a PyQt6 graphical user interface (GUI). It supports both batch processing of video files for emotion extraction and real-time emotion analysis from a webcam feed.

The application analyzes emotions such as happiness, neutrality, surprise, anger, disgust, fear, and sadness, and produces detailed reports, visualizations, and CSV outputs. The GUI offers an intuitive interface to upload videos, track analysis progress, view results, and perform real-time analysis.

---

## Features
- Batch analysis of video files (MP4, AVI, MOV, MKV, WMV, FLV, WebM)
- Real-time emotion analysis using the webcam
- Detailed emotion confidence and nervousness metrics
- Visualizations including time-series plots, distribution histograms, box plots, and pie charts
- Export emotion data to CSV files
- Comprehensive textual analysis reports
- User-friendly PyQt6 GUI with tabs for upload, results, and real-time analysis

---

## Installation

### Prerequisites
- Python 3.x (recommended Python 3.7+)

### Dependencies
Install the required Python packages using pip. The GUI-specific dependencies are listed in `requirements_gui.txt`. Core analysis dependencies include DeepFace, OpenCV, pandas, matplotlib, seaborn, and numpy.

Run the following commands to install all dependencies:

```bash
pip install -r requirements_gui.txt
pip install deepface opencv-python pandas matplotlib seaborn numpy
```

---

## Usage

### Running the GUI Application
Launch the main application by running:

```bash
python VideoEmotionGUI.py
```

Features include:

- Select or drag-and-drop supported video files for batch analysis
- Monitor analysis progress with a real-time progress bar and log messages
- View detailed results including emotion analysis tables and visualizations
- Access real-time emotion analysis from the connected webcam in the dedicated tab

### Supported Video Formats
The batch video analysis supports the following file formats:

- MP4
- AVI
- MOV
- MKV
- WMV
- FLV
- WebM

---

## Real-Time Emotion Analysis

The real-time analysis tab allows continuous emotion detection from a connected webcam. It provides live updates of confidence and nervousness metrics, dominant emotions, and graphical trends.

---

## Results

- After analysis, detailed emotion data is saved as CSV files with timestamps.
- Visualizations are generated and saved as high-resolution PNG images.
- Comprehensive text reports summarize the emotion statistics.
- Batch analysis results are viewable within the GUI under the "Results" tab.
- Real-time analysis outputs live visual feedback and can export session summary data.

---

## Command Line Example Usage (Standalone)

You can run the core video emotion analysis from the command line using the `bc.py` script:

```bash
python bc.py
```

This will analyze the default video file `test subject.mp4` located in the current directory and output results, reports, and visualizations.

---

## License and Contact

This project is provided as-is under no specific license. For questions or contributions, please contact the maintainer.

---

Thank you for using the Video Emotion Analysis Tool!
