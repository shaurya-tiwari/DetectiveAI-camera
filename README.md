# DetectiveAI - Intelligent Surveillance System

Real-time weapon detection and surveillance system powered by YOLOv8 and DeepSORT.

## Features

- 🔫 **Weapon Detection** - Detects guns using specialized firearm detection model
- 🔪 **Knife Detection** - Identifies knives and sharp objects
- 👥 **Crowd Detection** - Alerts when crowd size exceeds threshold
- 🎒 **Unattended Bag Detection** - Detects stationary bags left unattended
- 📹 **Real-time Processing** - Live video analysis with tracking
- 🎯 **Smart Alerts** - Intelligent alert system with cooldowns

## Project Structure

```
DetectiveAI/
├── src/                    # Core application code
│   ├── detection.py        # YOLO detection wrapper
│   ├── tracking.py         # DeepSORT tracking
│   ├── rules.py            # Alert rules engine
│   ├── visualize.py        # Visualization utilities
│   └── streamlit_app.py    # Main dashboard application
├── videos/                 # Sample videos for testing
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run src/streamlit_app.py
```

The firearm detection model will automatically download from Hugging Face on first run (~6MB).

## Usage

1. **Select Model**: Choose between "Firearm Detection" (recommended) or "Local COCO Model"
2. **Choose Video Source**: Sample video, upload your own, or use webcam
3. **Start Surveillance**: Click "▶️ Start Surveillance"
4. **Monitor Alerts**: Watch the Alert Log panel for weapon/crowd/bag detections

## Models

### Firearm Detection Model (Default)
- **Source**: [Subh775/Firearm_Detection_Yolov8n](https://huggingface.co/Subh775/Firearm_Detection_Yolov8n)
- **Accuracy**: 89% mAP@0.5
- **Classes**: Gun
- **Performance**: ~4ms per image on GPU

### Local COCO Model (Alternative)
- **Classes**: 80 COCO classes (person, knife, scissors, bags, etc.)
- **Note**: Does NOT detect guns

## Configuration

Key settings in `src/streamlit_app.py`:

```python
CONF_THRESHOLD = 0.15          # Detection confidence threshold
WEAPON_PERSIST_FRAMES = 2      # Frames before weapon alert
CROWD_THRESHOLD = 20           # People count for crowd alert
BAG_STATIONARY_SECONDS = 5     # Time before bag alert
```

## How It Works

1. **Detection**: YOLOv8 detects objects in each frame
2. **Tracking**: DeepSORT tracks objects across frames
3. **Rules Engine**: Analyzes tracks for anomalies
4. **Alerts**: Triggers alerts based on configured rules
5. **Visualization**: Displays annotated video with alerts

## Requirements

- Python 3.8+
- Webcam (optional, for live detection)
- GPU recommended for better performance

## Technologies

- **YOLOv8** - Object detection
- **DeepSORT** - Multi-object tracking
- **Streamlit** - Web dashboard
- **OpenCV** - Video processing
- **Hugging Face** - Model hosting

## License

MIT License

## Credits

- Firearm detection model by [Subh775](https://huggingface.co/Subh775)
- Built with Ultralytics YOLOv8
