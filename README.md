# 🎯 SentinelAI — Intelligent Surveillance System

Real-time weapon, crowd, and unattended bag detection powered by **YOLOv8 + DeepSORT + Streamlit**.

---

## 📌 What This Project Does

SentinelAI is a real-time smart surveillance system designed to monitor CCTV / camera feeds and automatically alert security personnel when anomalies occur:
- 🔫 **Weapon Detection:** Identifies firearms (pistols, rifles) and sharp objects (knives).
- 🎒 **Unattended Bag Detection:** Alerts if a bag is left alone without its owner nearby.
- 👥 **Crowd Detection:** Monitors group density and flags overcrowded areas.
- 🥊 **Altercation Detection (Planned):** Detects physical fights or aggressive behavior.

---

## 🧠 How It Works — End-to-End Pipeline

```
┌─────────────────┐
│ Camera / Video  │ (Input: MP4, Upload, or Live Webcam)
└────────┬────────┘
         │  Frame-by-Frame (OpenCV)
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. DETECTION (src/detection.py)                             │
│    YOLOv8 ONNX model scans the frame                        │
│    ➜ Detects bounding boxes (x1, y1, x2, y2), conf, class   │
└────────┬────────────────────────────────────────────────────┘
         │  Detections list
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. TRACKING (src/tracking.py)                               │
│    DeepSORT assigns unique persistent IDs (e.g., ID #3)     │
│    ➜ Remembers objects across frames even when moving       │
└────────┬────────────────────────────────────────────────────┘
         │  Tracked Objects with IDs
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. RULES ENGINE (src/rules.py)                              │
│    Evaluates business logic & rules (3-frame weapon check,  │
│    unattended bag timer, crowd counting)                    │
└────────┬────────────────────────────────────────────────────┘
         │  Active Alerts
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. VISUALIZATION & DASHBOARD (src/visualize.py & app)       │
│    Draws boxes + text overlay on frame                      │
│    ➜ Streams live video feed & alerts to Streamlit UI       │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 Core Logic & Decision Rules (How the AI Thinks)

The system doesn't just guess; it follows precise, intelligent rules built into `src/rules.py` to avoid false alarms:

### 1. 🔫 Weapon Alert Logic (`src/rules.py`)
> **Goal:** Detect guns/knives quickly while ignoring 1-frame glitches or false flickers.

```
Frame 1: Pistol detected (Conf > 0.20) ──► Counter = 1 (No alert yet)
Frame 2: Pistol detected               ──► Counter = 2 (No alert yet)
Frame 3: Pistol detected               ──► Counter = 3 ──► 🚨 TRIGGER WEAPON ALERT!
```
- **Rule:** A weapon must be detected for **3 consecutive frames** (`WEAPON_PERSIST_FRAMES = 3`).
- **Why?** Prevents random visual noise or single-frame glitches from triggering false alarms.
- **Cooldown:** Once alerted, enforces a **5-second cooldown** per object ID so it doesn't spam alerts every millisecond.

---

### 2. 🎒 Unattended Bag Logic (`src/rules.py`)
> **Goal:** Flag suspicious luggage or backpacks left behind by owners.

```
Is a Bag detected?
   │
   ├──► Is the Bag stationary? (Moved < 10 pixels over time)
   │       │
   │       ├──► YES ──► Is there a Person within 150 pixels of the bag?
   │       │               │
   │       │               ├──► YES ──► Owner is present ──► RESET TIMER ⏱️
   │       │               │
   │       │               └──► NO  ──► Bag is ALONE! ──► Start / Increment Timer ⏱️
   │       │                                                  │
   │       │                                                  └──► Timer >= 5 Seconds? (Configurable to 5 min)
   │       │                                                          │
   │       │                                                          └──► 🚨 TRIGGER UNATTENDED BAG ALERT!
   │       │
   │       └──► NO (Bag is moving) ──► Someone is carrying it ──► RESET TIMER ⏱️
```
- **Step 1 — Stationary Check:** Tracks bag center point `(cx, cy)`. If it moves > 10 pixels, it's being carried (timer resets).
- **Step 2 — Proximity Check:** Measures distance between bag center and all detected people. If any person is within **150 pixels**, the owner is nearby.
- **Step 3 — Timer Threshold:** If the bag stays stationary AND alone for `BAG_STATIONARY_SECONDS` (e.g. 5 seconds or 5 minutes), the alarm triggers!

---

### 3. 👥 Crowd Alert Logic (`src/rules.py`)
> **Goal:** Monitor public safety and detect overcrowding in real time.

```
Count total active "Person" tracks in current frame
   │
   ├──► Count >= 20 people (CROWD_THRESHOLD = 20)
   │       │
   │       └──► 🚨 TRIGGER CROWD ALERT! ("Crowd detected: 24 people")
   │
   └──► Count < 20 people ──► Normal status (No alert)
```
- **Rule:** Whenever the number of tracked people in a single frame hits or exceeds `CROWD_THRESHOLD` (default: 20), an alert fires.
- **Cooldown:** 10-second cooldown between crowd alerts to avoid log clutter.

---

## 📁 Project Structure & Codebase Mapping

```
DetectiveAI-camera/
├── src/
│   ├── detection.py       # 🔍 Loads ONNX YOLOv8 model & extracts boxes + confidence
│   ├── tracking.py        # 🎯 DeepSORT object tracker (assigns persistent IDs)
│   ├── rules.py           # 🧠 Business Logic (Weapon persist, Bag timer, Crowd count)
│   ├── visualize.py       # 🎨 Draws bounding boxes, IDs, and red alert text on frames
│   └── streamlit_app.py   # 💻 Modern Streamlit Light Dashboard UI
├── models/
│   └── best.onnx          # 🤖 Pre-trained YOLOv8 weapon detection model
├── videos/
│   └── cam1.mp4           # 📹 Test surveillance video footage
├── requirements.txt       # 📦 Python dependency list
└── README.md              # 📖 Project documentation
```

---

## 🔬 Code Architecture — How Modules Work Together

### 1. Detection (`src/detection.py`)
Loads the lightweight ONNX model and auto-selects GPU if CUDA is available, otherwise CPU.

```python
# Auto-detect hardware
self.device = "0" if torch.cuda.is_available() else "cpu"
self.model = YOLO("models/best.onnx", task="detect")

# Inference returns box coordinates, confidence score, and class label
detections = detector.detect(frame, conf_threshold=0.20)
# Output format: [(x1, y1, x2, y2, confidence, "pistol"), ...]
```

### 2. Tracking (`src/tracking.py`)
Uses **DeepSORT** to assign a unique, persistent ID to every detected object across continuous frames.

```python
# DeepSORT updates track positions across frames
tracks = tracker.update(detections, frame)

# Each track contains:
track.track_id        # Unique persistent ID string e.g. "3"
track.to_ltrb()       # Bounding box [x1, y1, x2, y2]
track.get_det_class() # Class name e.g. "pistol"
```

### 3. Rules Engine (`src/rules.py`)
Processes active tracks and applies safety rules. Auto-adapts based on model classes.

```python
# Process tracks against rules every frame
alerts = rules.process(tracks, frame_idx, frame_timestamp)
# Returns list of alert dictionaries: [{"type": "WEAPON", "message": "..."}, ...]
```

---

## ⚡ Performance & Benchmarks

| Hardware | Provider | Inference Latency | FPS |
|---|---|---|---|
| **NVIDIA GPU (CUDA)** | `CUDAExecutionProvider` | **~4 ms** | **~250 FPS** |
| **Apple Silicon (CPU)** | `CPUExecutionProvider` | **~31 ms** | **~32 FPS** |
| **Standard Intel CPU** | `CPUExecutionProvider` | ~50–80 ms | ~15–20 FPS |

> ℹ️ **Note on Mac vs GPU:** Apple Silicon chips do not support NVIDIA CUDA. On Mac, the system automatically uses optimized CPU execution (~31ms / 32 FPS), which is smooth and real-time. On NVIDIA GPUs with `onnxruntime-gpu`, latency drops to **4ms**.

To enable GPU on NVIDIA machines:
```bash
uv pip uninstall onnxruntime
uv pip install onnxruntime-gpu
```

---

## 🤖 Model Specifications

- **Format:** ONNX (`best.onnx`) — highly optimized for production inference without heavy PyTorch overhead.
- **Classes:** `pistol`, `knife`
- **Source:** Trained on public weapon detection datasets ([Hadi959/weapon-detection-yolov8](https://huggingface.co/Hadi959/weapon-detection-yolov8)).
- **Size:** ~12 MB

---

## 🚀 Quickstart & Setup

### Step 1 — Create Virtual Environment
```bash
cd DetectiveAI-camera
uv venv --python 3.11
source .venv/bin/activate
```

### Step 2 — Install Dependencies
```bash
uv pip install ultralytics streamlit opencv-python-headless numpy \
               deep-sort-realtime torch pillow onnxruntime "setuptools<70" onnx
```

### Step 3 — Run the Dashboard
```bash
streamlit run src/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🎛️ Key Configuration Parameters

In `src/streamlit_app.py`:

```python
CONF_THRESHOLD = 0.20        # Confidence cutoff for detections (0.0 to 1.0)
WEAPON_PERSIST_FRAMES = 3    # Frames weapon must persist before alert triggers
BAG_STATIONARY_SECONDS = 5   # Time bag must be alone & still before alert
CROWD_THRESHOLD = 20         # Number of people required to trigger crowd alert
```

---

## 🔭 Current Scope vs 🚀 Future Roadmap

### Current Scope (Today)
The codebase architecture in `src/rules.py` is **100% complete** for all 4 alert types. The active model loaded (`best.onnx`) is trained specifically on **weapons** (`pistol`, `knife`).

| Feature | Code Status | Current Model Status |
|---|---|---|
| 🔫 **Weapon Detection** | ✅ Complete | ✅ Active & Working |
| 👥 **Crowd Detection** | ✅ Complete | ⏳ Ready (Requires model with `person` class) |
| 🎒 **Unattended Bag** | ✅ Complete | ⏳ Ready (Requires model with `bag` class) |
| 🥊 **Altercation Detection** | ✅ Complete | ⏳ Ready (Requires model with `fight` class) |

---

### Future Roadmap

#### Phase 1 — Combined Multi-Class Model
Train a unified YOLOv8 model combining weapons, people, bags, and altercation poses so all 4 rules run simultaneously under one model.

#### Phase 2 — Context-Aware Role Classification ("Smart Skip")
> *"Not every gun is a threat — ignore authorized personnel."*

Integrate a second classifier to check if the person holding the weapon is an authorized officer:
- 👮 **Police Officer:** Detects uniform/badge ──► **Suppress Alert**
- 🔒 **Security Guard:** Detects guard vest/ID ──► **Suppress Alert**
- 🦹 **Unidentified Person:** No uniform detected ──► **🚨 Fire Weapon Alert**

```
Weapon Detected!
       │
       ▼
Classify Person Holding Weapon
       │
       ├──► Uniform / Security Badge Recognized ──► Suppress Alert (Authorized)
       │
       └──► Civilian / Unidentified ──► 🚨 FIRE WEAPON ALERT
```

#### Phase 3 — Edge Deployment & Multi-Camera
- **Dockerization:** Containerize app for seamless server deployment.
- **RTSP IP Camera Streams:** Connect directly to commercial CCTV networks.
- **NVIDIA Jetson Deployment:** Run inference directly on edge hardware for 4ms low latency.
