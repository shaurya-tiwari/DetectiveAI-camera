# Intelligent Surveillance System

Real-time weapon, crowd, and unattended bag detection powered by **YOLOv8 + DeepSORT + Streamlit**.

---

## 📌 What This Project Does

SentinelAI is an AI-powered surveillance dashboard that:
- Reads live video from a **file, upload, or webcam**
- Detects **weapons (pistol, knife)** in every frame
- **Tracks** detected objects across frames with persistent IDs
- **Fires smart alerts** when a weapon appears for 3+ consecutive frames
- Displays everything in a **live browser dashboard**

---

## 🧠 How It Works — Full Pipeline

```
[ Video File / Webcam ]
        │
        ▼ (OpenCV reads frame-by-frame)
┌─────────────────────────────────────┐
│  STEP 1 — Detection (detection.py)  │
│  YOLOv8 ONNX model scans the frame  │
│  Output: [x1,y1,x2,y2, conf, class] │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  STEP 2 — Tracking (tracking.py)    │
│  DeepSORT links detections across   │
│  frames — assigns persistent IDs    │
│  Output: Track ID + bounding box    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  STEP 3 — Rules Engine (rules.py)   │
│  Checks: weapon seen 3+ frames?     │
│  Checks: bag stationary & alone?    │
│  Checks: 20+ people in frame?       │
│  Output: Alert list                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  STEP 4 — Visualize (visualize.py)  │
│  Draws bounding boxes + alert text  │
│  on frame using OpenCV              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  STEP 5 — Dashboard (streamlit_app) │
│  Streams annotated frames to browser│
│  Updates Alert Log & System Status  │
└─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
DetectiveAI-camera/
├── src/
│   ├── detection.py       # YOLO ONNX inference wrapper
│   ├── tracking.py        # DeepSORT multi-object tracker
│   ├── rules.py           # Anomaly detection rules engine
│   ├── visualize.py       # Bounding box + alert overlay
│   └── streamlit_app.py   # Web dashboard (main entry point)
├── models/
│   └── best.onnx          # Trained weapon detection model
├── videos/
│   └── cam1.mp4           # Sample test video
├── requirements.txt
└── README.md
```

---

## 🔬 Code — How Each Module Works

### 1. Detection (`src/detection.py`)
Loads the ONNX model and runs inference on every frame.

```python
# Auto-selects GPU if available, falls back to CPU
self.device = "0" if torch.cuda.is_available() else "cpu"
self.model = YOLO("models/best.onnx", task="detect")

# Run inference on a frame
detections = self.model.predict(frame, conf=0.20, device=self.device)
# Returns: [(x1, y1, x2, y2, confidence, "pistol"), ...]
```

### 2. Tracking (`src/tracking.py`)
DeepSORT assigns a persistent ID to each detected object across frames.

```python
# Convert [x1,y1,x2,y2] → [x1,y1,width,height] (DeepSORT format)
ds_input = [([x1, y1, x2-x1, y2-y1], conf, name) for ...]
tracks = DeepSort(n_init=1).update_tracks(ds_input, frame=frame)

# Each track has:
track.track_id        # persistent ID e.g. "3"
track.to_ltrb()       # [x1, y1, x2, y2]
track.get_det_class() # "pistol"
track.is_confirmed()  # True after n_init frames
```

### 3. Rules Engine (`src/rules.py`)
Auto-detects which rules to run based on what the model can detect.

```python
# On startup — reads model class names
self.has_weapons = any("pistol" or "gun" or "knife" in class_names)
self.has_persons = any("person" in class_names)
self.has_bags    = any("bag" in class_names)

# Weapon rule — only fires after 3 confirmed frames
if weapon_seen_frames >= 3:
    alerts.append({"type": "WEAPON", "message": "Weapon detected!"})
```

### 4. Visualization (`src/visualize.py`)
Draws bounding boxes and alert text directly on the video frame.

```python
cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
cv2.putText(frame, f"ID:{tid} pistol", (x1, y1-10), ...)
```

---

## ⚡ Performance

| Hardware | Detection Latency | FPS |
|---|---|---|
| **NVIDIA GPU (CUDA)** | **~4 ms** | ~250 FPS |
| **Apple Silicon (CPU)** | **~31 ms** | ~32 FPS |
| **Intel CPU** | ~50–80 ms | ~15 FPS |

> GPU performance requires `onnxruntime-gpu` + CUDA installed.
> Apple Silicon has no CUDA support — runs on CPU only.
> 32 FPS on CPU is sufficient for real-time surveillance.

**To enable GPU (NVIDIA machines only):**
```bash
uv pip uninstall onnxruntime
uv pip install onnxruntime-gpu
```

---

## 🤖 Model

- **Model:** YOLOv8 exported to ONNX format
- **Source:** [Hadi959/weapon-detection-yolov8](https://huggingface.co/Hadi959/weapon-detection-yolov8)
- **Classes:** `pistol`, `knife`
- **Format:** `.onnx` (runs via ONNXRuntime — no PyTorch needed at inference)
- **Size:** ~12 MB

### Swapping the model
Drop any `.onnx` file into `models/` and update one line:
```python
# src/streamlit_app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "your_model.onnx")
```
Rules auto-adapt to whatever classes the new model has.

---

## 🚀 Setup & Run

### Step 1 — Install uv (if not installed)
```bash
pip install uv
```

### Step 2 — Create virtual environment
```bash
cd DetectiveAI-camera
uv venv --python 3.11
source .venv/bin/activate
```

### Step 3 — Install packages
```bash
uv pip install ultralytics streamlit opencv-python-headless numpy \
               deep-sort-realtime torch pillow onnxruntime "setuptools<70" onnx
```

### Step 4 — Run
```bash
streamlit run src/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 🎛️ Configuration

All key settings in `src/streamlit_app.py`:

```python
CONF_THRESHOLD = 0.20        # Min confidence to accept a detection (0–1)
CROWD_THRESHOLD = 20         # Person count to trigger crowd alert
BAG_STATIONARY_SECONDS = 5   # Seconds before unattended bag alert
WEAPON_PERSIST_FRAMES = 3    # Consecutive frames before weapon alert fires
```

---

## 📹 Video Sources

| Source | How to use |
|---|---|
| **Sample Video** | Drop `.mp4/.avi/.mov` into `videos/` → select from dropdown |
| **Upload Video** | Click Browse in sidebar → pick any video file |
| **Webcam** | Select Webcam → allow camera permission → Start |

---

## 🧩 Tech Stack

| Library | Role |
|---|---|
| **YOLOv8 (Ultralytics)** | Object detection neural network |
| **ONNXRuntime** | Runs the `.onnx` model efficiently |
| **DeepSORT** | Multi-object tracking across frames |
| **OpenCV** | Video reading, frame processing, drawing |
| **Streamlit** | Browser-based live dashboard |
| **PyTorch** | GPU/CPU tensor backend |
| **NumPy** | Bounding box coordinate arrays |

---

## ❓ Common Issues

| Error | Fix |
|---|---|
| `No module named 'onnxruntime'` | `uv pip install onnxruntime` |
| `No module named 'pkg_resources'` | `uv pip install "setuptools<70"` |
| `streamlit: command not found` | Activate venv first: `source .venv/bin/activate` |
| Wrong detections | Raise `CONF_THRESHOLD` to `0.40` or swap to a better model |
| Black webcam screen | Allow camera in System Settings → Privacy → Camera |

---

## 📊 Alert Types

| Alert | Trigger Condition |
|---|---|
| `WEAPON` | Pistol/knife detected for 3+ consecutive frames |
| `CROWD` | 20+ people visible simultaneously |
| `UNATTENDED_BAG` | Bag stationary for 5s with no person within 150px |

> **Note:** CROWD and UNATTENDED_BAG alerts only activate if the loaded model has those classes. Current model (`pistol`, `knife`) only triggers WEAPON alerts.

---

## 🔭 Current Scope — What We Detect Today

The system is **fully architected** for all 4 alert types. The rules engine, tracking, and UI are ready for all of them. What limits us today is the **model** — our current model was only trained on 2 classes:

| Alert Type | Status | Reason |
|---|---|---|
| 🔫 **Weapon (pistol/knife)** | ✅ **Working** | Model trained on guns + knives |
| 👥 **Crowd Detection** | ⏳ Ready, needs model | No "person" class in current model |
| 🎒 **Unattended Bag** | ⏳ Ready, needs model | No "bag" class in current model |
| 🥊 **Altercation / Fight** | ⏳ Ready, needs model | No "fight" class in current model |

> The code for ALL four rules is already written. Swapping to a richer model (e.g. trained on person + bag + weapons) will automatically activate all alerts — zero code changes needed.

---

## 🚀 Future Plans

### Phase 1 — Multi-class Model Training
Train a custom YOLOv8 model on a larger combined dataset:
- **Weapons:** pistol, rifle, knife, taser
- **People:** person (for crowd counting)
- **Bags:** backpack, suitcase, handbag (for unattended bag detection)
- **Altercation:** fighting, physical contact scenes

This single model upgrade will unlock all 4 alert types instantly.

---

### Phase 2 — Context-Aware Detection (Smart Skip)
> *"Not every gun is a threat"*

Teach the system to **ignore authorized personnel** carrying weapons:

- 👮 **Skip police officers** — detect uniform + badge alongside gun → suppress alert
- 🔒 **Skip security guards** — detect security vest + ID badge → suppress alert
- 🏥 **Skip medical staff** — context-aware role classification

This requires a **role classification model** running alongside weapon detection — classifying the *person holding the weapon*, not just the weapon itself.

```
Weapon detected
      │
      ▼
Is the holder wearing a uniform? ──Yes──► Suppress alert (authorized)
      │
     No
      │
      ▼
🚨 Fire WEAPON alert
```

---

### Phase 3 — Advanced Features

| Feature | Description |
|---|---|
| **Fight / Altercation detection** | Detect aggressive physical contact using pose estimation |
| **Abandoned object tracking** | Track bags left behind even after the person leaves frame |
| **License plate recognition** | Link vehicle plates to alert events |
| **Multi-camera support** | Monitor multiple CCTV feeds simultaneously |
| **Alert history export** | Save alert logs to CSV / PDF reports |
| **SMS / Email notifications** | Push real-time alerts to security personnel |
| **GPU acceleration** | Switch to `onnxruntime-gpu` for ~4ms latency on NVIDIA hardware |

---

### Phase 4 — Deployment
- Package as a **Docker container** for easy deployment on any CCTV server
- Support **RTSP streams** from real IP cameras
- Deploy on **NVIDIA Jetson** edge devices for on-site inference
- Cloud dashboard with **multi-site monitoring**

