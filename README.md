# SentinelAI

SentinelAI is a real-time intelligent surveillance prototype that analyzes video to detect security anomalies:
- **Weapon Detection**: Detects guns and knives.
- **Unattended Bags**: Detects bags left stationary for a configurable duration.
- **Crowd Detection**: Alerts when the number of people exceeds a threshold.

Powered by **YOLOv8** for detection and **DeepSORT** for tracking.

## Project Structure

- `src/`: Core source code.
    - `detection.py`: YOLOv8 wrapper.
    - `tracking.py`: DeepSORT wrapper.
    - `rules.py`: Event logic engine.
    - `visualize.py`: Visualization helpers.
    - `streamlit_app.py`: Main dashboard application.
- `videos/`: Place your demo videos here.
- `requirements.txt`: Python dependencies.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Model**:
    The system uses `yolov8n.pt` by default. It will be downloaded automatically by Ultralytics on first run, or you can place it in the root directory.

## Usage

Run the Streamlit dashboard:

```bash
streamlit run src/streamlit_app.py
```

## Features

- **Real-time Dashboard**: View annotated video, alerts, and system status.
- **Configurable Rules**: Adjust thresholds for crowd count, bag timer, and confidence.
- **Video Source**: Support for sample videos, file upload, or webcam.

## Configuration

You can adjust the following in the sidebar:
- **Confidence Threshold**: Minimum confidence for detections.
- **Crowd Threshold**: Number of people to trigger a crowd alert.
- **Bag Stationary Limit**: Seconds a bag must be stationary to trigger an alert.
- **Weapon Persistence**: Number of consecutive frames a weapon must be detected to trigger an alert.
